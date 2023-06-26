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
    df = pd.read_csv(dataset_path)

    df = df.sort_values("pt_score", ascending=False)
    df = df[0:400]

    df["pt_answer"] = df["pt_answer"].apply(lambda x: cleanhtml(x))
    df["pt_body"] = df["pt_body"].apply(lambda x: cleanhtml(x))

    df["question"] = df["pt_title"].str.lower()
    df["question"] = df["question"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["pt_answer"].str.lower()
    df["pt_body"] = df["pt_body"].str.lower()
    df['context'] = df['pt_body'] + df["answer"]
    
    df = df[['question', 'answer','context']]
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
    df["context"] = df["context "].str.lower()
    df["answer"] = df["answer"].str.lower()
    df["source"] = "pt_faq"
    return df


def load_blogs_data(dataset_path):
    df = pd.read_json(dataset_path)

    df[["question", "answer"]] = df["text"].apply(
        lambda x: pd.Series(str(x).split("Answer: ", maxsplit=1))
    )

    df = df.dropna().reset_index(drop=True)
    df["question"] = df["question"].str.lstrip("Question:")
    df["question"] = df["question"].apply(lambda x: cleanhtml(x))
    df["answer"] = df["answer"].apply(lambda x: cleanhtml(x))
    df["question"] = df["question"].str.lower()
    df["answer"] = df["answer"].str.lower()

    df = df[["question", "answer", "context"]]
    df["source"] = "pt_blogs"
    return df


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
    parser.add_argument("--stack_overflow_dataset_path", type=str, default="pt_question_answers.csv")
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
    print("Tutorial Dataset: ", tutorial_df.shape)
    faq_df = load_faq_data(args.pt_faq_dataset_path)
    print("Faq Dataset: ", faq_df.shape)
    blogs_df = load_blogs_data(args.blogs_curated_dataset_path)
    print("Blogs Dataset: ", blogs_df.shape)

    final_df = pd.concat(
        [so_df, discuss_df,tutorial_df, faq_df, blogs_df]
    ).reset_index()
    print("Merged Dataset: ", final_df.shape)

    generate_data_in_alpaca_format(df=final_df, output_file_path="pt_curated_1000_alpaca_format.json")
