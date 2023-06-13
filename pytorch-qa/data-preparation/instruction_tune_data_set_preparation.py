import pandas as pd
import re
import json


def read_dataset(path):
    df = pd.read_csv(path)
    return df


def get_url(df):
    url = []
    for index, row in df.iterrows():
        url.append(f"https://stackoverflow.com/questions/{row['pt_post_id']}/")

    df["source"] = url

    return df


def cleanhtml(raw_html):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def remove_html_tags(so_df):
    so_df["pt_title"] = so_df["pt_title"].apply(lambda x: cleanhtml(x))
    so_df["pt_body"] = so_df["pt_body"].apply(lambda x: cleanhtml(x))
    so_df["pt_answer"] = so_df["pt_answer"].apply(lambda x: cleanhtml(x))
    return so_df


def create_hf_dataset(so_df, output_filename):
    with open(output_filename, "w") as jsonl_file:
        for index, row in so_df.iterrows():
            instruction = row["instruction"]
            input_ = row["input"]
            output = row["output"]
            source = row["source"]
            text = {"instruction": instruction, "input": input_, "output": output, "source": source}
            jsonl_file.write(json.dumps(text) + "\n")


if __name__ == "__main__":
    so_df = read_dataset("pt_question_answers_updated.csv")
    so_df = get_url(so_df)
    so_df = so_df[["pt_title", "pt_body", "pt_answer", "source"]]
    so_df = remove_html_tags(so_df)
    so_df.rename(
        {"pt_title": "instruction", "pt_body": "input", "pt_answer": "output"}, axis=1, inplace=True
    )
    so_df = so_df[["instruction", "input", "output", "source"]]
    create_hf_dataset(so_df, output_filename="so_dataset.jsonl")
