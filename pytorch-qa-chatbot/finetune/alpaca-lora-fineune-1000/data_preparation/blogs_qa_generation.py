import argparse
import concurrent.futures
import os

import openai
import pandas as pd
import requests
from langchain.text_splitter import MarkdownTextSplitter
from tqdm import tqdm


def load_and_split_data_blogs(dataset_path):
    df = pd.read_json(dataset_path)
    print("blogs dataset", df.shape)

    ## Using langchain - split the data into multiple pages
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("chunking pages into smaller sub-pages")
    pages = []
    for index, row in df.iterrows():
        markdown_text = row["text"]
        metadata = row["metadata"]
        docs = markdown_splitter.create_documents([markdown_text], [metadata])
        pages.extend(docs)

    print("total pages:", len(pages))
    return pages


def get_openai_api(context):
    if not os.environ["OPENAI_API_KEY"]:
        raise EnvironmentError(
            "OPENAI_API_KEY - key missing. set in the environment before running the script"
        )
    api_key = os.environ["OPENAI_API_KEY"]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", api_key=api_key, messages=[{"role": "user", "content": context}]
        )

        qa = completion.choices[0].message.content

    except requests.exceptions.RequestException as e:
        print(f"Request failed with error: {str(e)}.")
        print(f"Waiting for 3 minutes before trying again...")
        time.sleep(180)

    return qa


def get_qa(pages):
    print("Using openai to generate questions and answers")
    questions_ans = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        start = 0
        end = 216
        for i in pages[start:end]:

            context = f"Generate question and answer only in this format 'Question: Answer:' using this context \
            and decide the number of question and answer to be generated should be less than 4 and don't generate \
            too many same kind of questions: {i.page_content}. Important note: skip the context and do not generate question if it is not meaningful"

            futures.append(executor.submit(get_openai_api, context))

        for future, i in tqdm(
            zip(concurrent.futures.as_completed(futures), pages[start:end]),
            total=len(pages[start:end]),
        ):
            try:
                qa = future.result()
                questions_ans.append(
                    {"text": qa, "context": i.page_content, "metadata": i.metadata}
                )
            except Exception as exc:
                print(f"generated an exception: {exc}")
    df1 = pd.DataFrame(questions_ans)
    print("creating blogs_curated_data.json")
    df1.to_json("blogs_curated_data.json", orient="records")


def main(args):
    print("generating blogs dataset")
    blogs_pages = load_and_split_data_blogs(args.blogs_dataset_path)
    get_qa(blogs_pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blogs_dataset_path", type=str, default="blogs.json")

    args = parser.parse_args()
    main(args)
