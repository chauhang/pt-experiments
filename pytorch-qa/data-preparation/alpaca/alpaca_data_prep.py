import argparse
import json
import re

import pandas as pd
from tqdm import tqdm


def cleanhtml(raw_html):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def load_and_clean_data(dataset_path):
    df = pd.read_json(dataset_path)

    df[["question", "answer"]] = df["text"].apply(
        lambda x: pd.Series(str(x).split("ANSWER: ", maxsplit=1))
    )

    df["question"] = df["question"].str.lstrip("QUESTION:")

    df = df[["question", "answer"]]

    df = df.dropna().reset_index(drop=True)
    df["question"] = df["question"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["answer"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["answer"].str.replace("\n\n\n", "\n")
    df["answer"] = df["answer"].str.replace("\n\n", "\n")
    df["answer"] = df["answer"].str.replace("\n \n", "\n")
    df["answer"] = df["answer"].str.lstrip("\n").str.rstrip("\n")
    df["question"] = df["question"].str.lower()
    df["answer"] = df["answer"].str.lower()

    return df


def generate_data_in_alpaca_format(df, max_length=2048, output_file_path="final_data.json"):
    data = []

    for index, row in tqdm(df.iterrows()):
        d = {
            "instruction": row["question"][0:max_length],
            "input": "",
            "output": row["answer"][0:max_length],
        }

        data.append(d)

    print("Writing to: ", output_file_path)
    with open(output_file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_overflow_dataset_path", type=str, default="stack_overflow.json")
    parser.add_argument("--pt_discuss_dataset_path", type=str, default="discussion_forum.json")
    parser.add_argument("--pt_docs_dataset_path", type=str, default="docs_qa_dataset_generated.json")
    parser.add_argument("--pt_blogs_discuss_path", type=str, default="blogs_qa_dataset_generated.json")
    args = parser.parse_args()
    so_df = load_and_clean_data(args.stack_overflow_dataset_path)
    print("SO Dataset: ", so_df.shape)
    discuss_df = load_and_clean_data(args.pt_discuss_dataset_path)
    print("Discuss Dataset: ", discuss_df.shape)
    docs_df = load_and_clean_data(args.pt_docs_dataset_path)
    print("Docs Dataset: ", docs_df.shape)
    blogs_df = load_and_clean_data(args.pt_blogs_discuss_path)
    print("Blogs Dataset: ", blogs_df.shape)

    final_df = pd.concat([so_df, discuss_df, docs_df, blogs_df])
    print("Merged Dataset: ", final_df.shape)

    generate_data_in_alpaca_format(df=final_df)
