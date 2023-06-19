import argparse
import json
import re

import pandas as pd
from tqdm import tqdm


def load_and_clean_data(dataset_path):
    df = pd.read_json(dataset_path)
    df["output"] = df["output"].str.replace("As an expert PyTorch engineer", "")
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would likely provide a similar answer to the one given above",
        "",
    )
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would suggest that", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would suggest to", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would suggest", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would recommend", "")
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer the question by explaining that", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer the question by suggesting", ""
    )
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would explain that", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would advise", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would provide", "")
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would give the following answer", ""
    )
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would likely", "")
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer by providing example code on how", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would first acknowledge that", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer this question by explaining", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer this question by suggesting", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer the question as follows", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer by suggesting", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer this question by providing", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would answer the question as follows", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer might answer this question by suggesting that", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer may answer the question by", ""
    )
    df["output"] = df["output"].str.replace(
        "An expert PyTorch engineer would approach this problem by", ""
    )
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would first", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would start by", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would agree that", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would answer", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would confirm", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer would", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer might suggest", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer might", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer will", "")
    df["output"] = df["output"].str.replace("An expert PyTorch engineer", "")
    df["output"] = df["output"].str.replace("An expert in PyTorch would answer the question", "")
    df["output"] = df["output"].str.replace("expert PyTorch engineer", "i")
    df["output"] = df["output"].str.replace("PyTorch engineer", "i")
    df["output"] = df["output"].str.replace("engineer", "")
    return df


def generate_data_in_alpaca_format(df, output_file_path="final_data.json"):
    print("Writing to: ", output_file_path)
    df.to_json(output_file_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack_overflow_dataset_path",
        type=str,
        default="pytorch_so_answer_summary_alpaca_format.json",
    )
    args = parser.parse_args()
    so_df = load_and_clean_data(args.stack_overflow_dataset_path)
    print("SO Dataset: ", so_df.shape)

    generate_data_in_alpaca_format(
        df=so_df, output_file_path="pytorch_so_answer_summary_alpaca_format_cleaned.jsonl"
    )
