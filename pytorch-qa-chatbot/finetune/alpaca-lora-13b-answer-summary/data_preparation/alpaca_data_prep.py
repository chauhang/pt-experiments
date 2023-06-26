import argparse
import json
import re

import pandas as pd
from tqdm import tqdm
import concurrent.futures
import requests
import os
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
    df["question"] = df["pt_title"].str.lower() + "\n" + df["pt_body"]
    df["question"] = df["question"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["pt_answer"].str.lower()
    df = get_url(df)
    df = df[["question", "answer", "source"]]
    return df


def summarize_answer(df):
    """using openai summarize answer"""

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

            context = f"Given a pytorch question and answer given below, How will an expert PyTorch engineer answer this question? Include code as appropriate and do not mentioned your role in the answer \
                    question: {i['question']}, answer:{i['answer']}"

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
        df.at[index, "short_answer"] = qa

    ## drop na
    df = df.dropna()
    return df


def generate_data_in_alpaca_format(df, max_length=2048, output_file_path="final_data.json"):
    data = []

    for index, row in tqdm(df.iterrows()):
        d = {
            "instruction": row["question"][0:max_length],
            "input": row["answer"][0:max_length],
            "output": row["short_answer"][0:max_length],
        }

        data.append(d)

    print("Writing to: ", output_file_path)
    with open(output_file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_overflow_dataset_path", type=str,                             default="../../../data_curation/data_sources/pt_question_answers_updated.csv")
    
    args = parser.parse_args()
    so_df = load_and_clean_data(args.stack_overflow_dataset_path)
    print("SO Dataset: ", so_df.shape)
#     so_df = so_df[:5000] ##using only 5k datapoints
    df = summarize_answer(so_df)

    generate_data_in_alpaca_format(
        df=df, output_file_path="pytorch_so_answer_summary_alpaca_format.json"
    )
