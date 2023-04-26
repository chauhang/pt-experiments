import pandas as pd
import re
import os
import json
import pickle
from tqdm import tqdm
import argparse

tqdm.pandas()


def cleanhtml(raw_html):
    CLEANR = re.compile("<.*?>")
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def clean_data(df):
    df["pt_answer"] = df["pt_answer"].apply(lambda x: cleanhtml(x))

    df["question"] = df["pt_title"].str.lower()
    df["answer"] = df["pt_answer"].str.lower()

    df = df[["pt_post_id", "question", "answer"]]
    return df


## get qa and link to post
def get_so(df, source_name, folder_path):
    data = []

    for index, row in df.iterrows():
        text = "QUESTION: " + row["question"] + " ANSWER: " + row["answer"]
        data.append(
            {
                "text": text,
                "metadata": {"source": f"https://stackoverflow.com/questions/{row['pt_post_id']}/"},
            }
        )

    output_folder = folder_path + "/" + source_name

    if not os.path.exists(output_folder):
        print(f"creating folder {output_folder}")
        os.makedirs(output_folder)

    print(f"saving data into {output_folder} as {source_name}.json")
    with open(f"{output_folder}/{source_name}.json", "w") as f:
        json.dump(data, f)

    pickle.dump(data, open(f"{output_folder}/{source_name}.pkl", "wb"))


def main(folder_path, source_path, source_name):

    df = pd.read_csv(source_path)

    # removing html tags
    df = clean_data(df)

    # saving final json
    get_so(df, source_name, folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save data files.")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="knowledgebase",
        help="Path where the output files will be saved",
    )
    parser.add_argument(
        "--source_path", type=str, default="pt_question_answers_updated.csv", help="Path to df"
    )
    parser.add_argument("--source_name", type=str, default="file", help="Name of the output files")
    args = parser.parse_args()

    main(args.folder_path, args.source_path, args.source_name)
