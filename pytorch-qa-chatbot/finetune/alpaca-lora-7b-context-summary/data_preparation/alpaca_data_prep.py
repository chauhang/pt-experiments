import argparse
import json
import re

import pandas as pd
from tqdm import tqdm
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pickle
import concurrent.futures
import os
import requests
import openai


def cleanhtml(raw_html):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def get_url(df):
    url = []
    for index, row in df.iterrows():
        url.append(f"https://stackoverflow.com/questions/{row['pt_post_id']}/")

    df["source"] = url
    return df


def load_and_clean_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df["pt_answer"] = df["pt_answer"].apply(lambda x: cleanhtml(x))
    df["question"] = df["pt_title"].str.lower()
    df["answer"] = df["pt_answer"].str.lower()
    df = get_url(df)
    df = df[["question", "answer", "source"]]
    return df


def create_faiss_index(df):
    """create faiss index to fetch nearest docs and create context column"""

    print("creating FAISS index")
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048)
    print("chunking pages into smaller sub-pages")

    pages = []
    for index, i in df.iterrows():
        texts = "QUESTION: " + i["question"] + "\nANSWER: " + i["answer"]
        meta = {"source": i["source"]}
        pages.extend(splitter.create_documents([texts], [meta]))
    # pickle.dump(pages, open('so_pages.pkl', 'wb'))

    embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.from_documents([pages.pop(0)], embeddings)
    i, step = 0, 50
    while i < len(pages):
        if i % 500 == 0:
            print(i, "pages done")
        texts = [d.page_content for d in pages[i : i + step]]
        meta = [d.metadata for d in pages[i : i + step]]
        docsearch.add_texts(texts, meta)
        i += step
    print(len(pages), "pages done")
    # docsearch.save_local('so_context_summary_faiss_index')
    return docsearch


def add_context_column(df, faiss_index):
    """query nearest 2 docs and create new column context"""

    # docsearch = FAISS.load_local("so_context_summary_faiss_index", embeddings)
    print("adding top two answer as context column")
    context = []

    for index, i in tqdm(df.iterrows(), total=len(df)):
        docs = faiss_index.similarity_search_with_score(i["question"], k=2)
        ans = []
        for doc in docs:
            text = doc[0].page_content.split("ANSWER:")[-1].lstrip()
            ans.append("Answer: " + doc[0].page_content.split("ANSWER:")[-1])
        for item in ans:
            if i["answer"][:20] in item:
                pass
            else:
                ans[0] = "Answer: " + i["answer"]
        context.append(", ".join(ans))
    df["context"] = context
    return df


def summarize_context(df):
    """using openai summarize this context"""

    if not os.environ["OPENAI_API_KEY"]:
        raise EnvironmentError(
            "OPENAI_API_KEY - key missing. set in the environment before running the script"
        )
    api_key = os.environ["OPENAI_API_KEY"]

    def get_qa_openai(context, index):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                api_key=api_key,
                messages=[{"role": "user", "content": context}],
            )

            qa = completion.choices[0].message.content

        except requests.exceptions.RequestException as e:
            print(f"Request failed with error: {str(e)}.")
            print(f"Waiting for 3 minutes before trying again...")
            time.sleep(180)

        return (qa, index)

    questions_ans = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = []
        for index, i in df.iterrows():

            context = f"Reduce the context size as much as you can in the following context but preserve the code and important details that answers the question: \
                        question: {i['question']}, context:{i['context']}"

            futures.append(executor.submit(get_qa_openai, context, index))

        for future, (_, row) in tqdm(
            zip(concurrent.futures.as_completed(futures), df.iterrows()), total=len(df)
        ):
            try:
                qa, ind = future.result()
                questions_ans.append((ind, qa))
            except Exception as exc:
                print(f"generated an exception: {exc}")

    for index, qa in questions_ans:
        df.at[index, "context_summary"] = qa

    ## drop na
    df = df.dropna()
    ## dropping rows having context len less than 20
    df["len"] = df["context_summary"].str.len()
    df = df.drop(df[df["len"] < 20].index)
    return df


def generate_data_in_alpaca_format(df, max_length=2048, output_file_path="final_data.json"):
    data = []

    for index, row in tqdm(df.iterrows()):
        d = {
            "instruction": row["question"][0:max_length],
            "input": row["context_summary"][0:max_length],
            "output": row["answer"][0:max_length],
        }

        data.append(d)

    print("Writing to: ", output_file_path)
    with open(output_file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack_overflow_dataset_path",
        type=str,
        default="../../../data_curation/data_sources/pt_question_answers_updated.csv",
    )
    args = parser.parse_args()
    so_df = load_and_clean_data(args.stack_overflow_dataset_path)
    print("SO Dataset: ", so_df.shape)

    embeddings = HuggingFaceEmbeddings()
    faiss_index = create_faiss_index(so_df)
    df = add_context_column(so_df, faiss_index)
    df = summarize_context(df)

    generate_data_in_alpaca_format(
        df=df, output_file_path="pytorch_so_context_summary_alpaca_format.json"
    )
