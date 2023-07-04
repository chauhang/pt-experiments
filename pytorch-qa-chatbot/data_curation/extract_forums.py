import os
import pickle
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from bs4 import BeautifulSoup as BSHTML
from simplejson import JSONDecodeError

global host
host = "https://discuss.pytorch.org"


def _process_cooked(cooked):
    bs = BSHTML(cooked)
    p = " ".join([x.get_text() for x in bs])
    return p


def process_row(row):
    t = row["id"]
    title = row["title"]
    try:
        r = requests.get(host + f"/t/{t}/posts.json").json()
    except JSONDecodeError:
        return None
    try:
        q = title + "? " + _process_cooked(r["post_stream"]["posts"][0]["cooked"])
        a = _process_cooked(
            [x["cooked"] for x in r["post_stream"]["posts"] if x["accepted_answer"] is True][0]
        )
    except IndexError:
        print(f"Skipping https://discuss.pytorch.org/t/{t}/")
        return None
    text = "QUESTION: " + q + " ANSWER: " + a
    return {"text": text, "metadata": {"source": f"https://discuss.pytorch.org/t/{t}/"}}


def fetch_page(page):
    period = "all"

    resp = requests.get(host + f"/top.json?page={page}&period={period}&per_page=100").json()
    df = pd.DataFrame(resp["topic_list"]["topics"])
    df = df.loc[df.has_accepted_answer == True, ["id", "title"]]
    return df


def extract_top_post_links(folder_path, source_name, total_pages):
    # total_pages = 1236

    solved_post_list = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_page, page) for page in range(total_pages)]
        for future in tqdm(as_completed(futures), total=total_pages):
            solved_post_list.append(future.result())

    output_folder = folder_path + "/" + source_name
    if not os.path.exists(output_folder):
        print(f"creating folder {output_folder}")
        os.makedirs(output_folder)

    print("saving data as csv in {output_folder} as {source_name}_post.csv")
    df = pd.concat(solved_post_list, ignore_index=True)
    df.to_csv(f"{output_folder}/{source_name}_post.csv", index=False)


def get_forum(folder_path, source_name):

    df = pd.read_csv(f"{folder_path}/{source_name}/{source_name}_post.csv")
    data = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                data.append(result)

    print(f"saving data into {folder_path}/{source_name} as {source_name}.json")
    with open(f"{folder_path}/{source_name}/{source_name}.json", "w") as f:
        json.dump(data, f)

    pickle.dump(data, open(f"{folder_path}/{source_name}/{source_name}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and save data files.")
    parser.add_argument(
        "--total_pages", type=int, default=1236, help="Total number of top pages to crawl"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="knowledgebase",
        help="Path where the output files will be saved",
    )
    parser.add_argument("--source_name", type=str, default="file", help="Name of the output files")
    args = parser.parse_args()

    extract_top_post_links(args.folder_path, args.source_name, args.total_pages)
    get_forum(args.folder_path, args.source_name)
