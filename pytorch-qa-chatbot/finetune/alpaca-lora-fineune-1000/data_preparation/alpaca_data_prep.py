import argparse
import json
import re

import pandas as pd
from tqdm import tqdm


def cleanhtml(raw_html):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def load_and_clean_so_data(dataset_path):
    df = pd.read_json(dataset_path)

    df = df.sort_values("pt_score", ascending=False)
    df = df[0:400]

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
    df["source"] = "pt_stack_overflow"

    return df


def load_discuss_curated_data(dataset_path):
    df = pd.read_json(dataset_path)

    df = df.dropna().reset_index(drop=True)
    df["answer"] = df[["human answer"]]
    df["context"] = df["context"] + df["Accepted Answer"]
    df["source"] = "pt_discuss_forum"
    return df


def load_tutorial_data(dataset_path):
    df = pd.read_json(dataset_path)

    df = df[["question", "context", "answer"]]
    df["source"] = "pt_tutorial"
    return df


def load_faq_data(dataset_path):
    df = pd.read_json(dataset_path)

    df["question"] = df["question"].str.lower()
    df["context"] = df["context"].str.lower()
    df["answer"] = df["answer"].str.lower()
    df["source"] = "pt_faq"
    return df


def load_blogs_data(dataset_path):
    df = pd.read_json(dataset_path)

    df[["question", "answer"]] = df["text"].apply(
        lambda x: pd.Series(str(x).split("Answer: ", maxsplit=1))
    )

    df = blogs_df.dropna().reset_index(drop=True)
    df["question"] = df["question"].str.lstrip("Question:")
    df["question"] = df["question"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["answer"].apply(lambda x: cleanhtml(x))
    df["question"] = df["question"].str.lower()
    df["answer"] = df["answer"].str.lower()

    df = df[["question", "answer", "context"]]
    df["source"] = "pt_blogs"


def generate_data_in_alpaca_format(df, max_length=2048, output_file_path="final_data.json"):
    data = []

    for index, row in tqdm(df.iterrows()):
        d = {
            "instruction": row["question"],
            "input": row["context"][0:max_length],
            "output": row["answer"][0:max_length] + "<<end_of_text>>",
        }

        data.append(d)

    print("Writing to: ", output_file_path)
    with open(output_file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_overflow_dataset_path", type=str, default="stack_overflow.json")
    parser.add_argument(
        "--pt_discuss_dataset_path", type=str, default="discussion_forum_curated.json"
    )
    parser.add_argument("--pt_tutorial_dataset_path", type=str, default="pt_tutorial.json")
    parser.add_argument("--pt_faq_dataset_path", type=str, default="pt_faq.json")
    parser.add_argument("--blogs_curated_dataset_path", type=str, default="blogs_curated_data.json")
    args = parser.parse_args()
    so_df = load_and_clean_so_data(args.stack_overflow_dataset_path)
    print("SO Dataset: ", so_df.shape)
    discuss_df = load_discuss_curated_data(args.pt_discuss_dataset_path)
    print("Discuss Dataset: ", discuss_df.shape)
    tutorial_df = load_tutorial_data(args.pt_tutorial_dataset_path)
    print("Tutorial Dataset: ", discuss_df.shape)
    faq_df = load_faq_data(args.pt_faq_dataset_path)
    print("Faq Dataset: ", discuss_df.shape)
    blogs_df = load_blogs_data(args.blogs_curated_dataset_path)
    print("Blogs Dataset: ", discuss_df.shape)

    final_df = pd.concat(
        [so_df, discuss_df, so_df_final, tutorial_df, faq_df, blogs_df]
    ).reset_index()
    print("Merged Dataset: ", final_df.shape)

    generate_data_in_alpaca_format(df=final_df, output_file_path="pt_curated_1000.json")
