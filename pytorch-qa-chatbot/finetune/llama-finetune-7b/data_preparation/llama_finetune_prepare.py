import argparse
import json
import re

import markdown
import pandas as pd
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter


def split_pages(df, category, chunk_size=1024, chunk_overlap=50, separator="\n"):
    character_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
    )
    pages = []
    for index, row in df.iterrows():
        markdown_text = row["text"]
        metadata = row["metadata"]
        metadata["category"] = category
        docs = character_splitter.create_documents([markdown_text], [metadata])
        pages.extend(docs)

    return pages


def read_and_split_blogs(blog_dataset_path):
    blogs_df = pd.read_json(blog_dataset_path)
    print("Total Blogs: ", blogs_df.shape)
    pages = split_pages(blogs_df, "pytorch blogs")
    print("Number of pages: ", len(pages))
    return pages


def read_and_split_docs(docs_dataset_path):
    docs_df = pd.read_json(docs_dataset_path)
    print("\nTotal Docs: ", docs_df.shape)
    docs_df["contains_table"] = docs_df.text.str.contains("---------------------------")
    docs_df = docs_df[docs_df.contains_table == False]
    docs_df.drop("contains_table", axis=1, inplace=True)
    print("After removing columns: ", docs_df.shape)
    pages = split_pages(docs_df, "pytorch docs")
    print("Number of pages: ", len(pages))
    return pages


def split_fields(doc):
    page_content_dict = json.loads(doc.json())
    content = page_content_dict["page_content"]

    # remove heading * symbols
    pattern = r"\n\*+\n"
    text = re.sub(pattern, "", content)

    html_text = markdown.markdown(text, extensions=["markdown.extensions.extra"])

    soup = BeautifulSoup(html_text, "html.parser")
    clean_text = soup.get_text()
    clean_text = clean_text.replace('{: style="text-align: center;"}', "")

    text = {
        "text": clean_text,
        "source": page_content_dict["metadata"]["source"],
        "category": page_content_dict["metadata"]["category"],
    }

    return text


def generate_jsonl_file(output_file_name, blog_pages, doc_pages):
    with open(output_file_name, "w") as jsonl_file:
        for doc in blog_pages:
            text = split_fields(doc)
            jsonl_file.write(json.dumps(text) + "\n")

        for doc in doc_pages:
            text = split_fields(doc)
            jsonl_file.write(json.dumps(text) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_docs_dataset_path", type=str, default="docs.json")
    parser.add_argument("--pt_blogs_discuss_path", type=str, default="blogs.json")
    args = parser.parse_args()
    blog_pages = read_and_split_blogs(blog_dataset_path=args.pt_blogs_discuss_path)
    doc_pages = read_and_split_docs(docs_dataset_path=args.pt_docs_dataset_path)

    generate_jsonl_file(
        output_file_name="blogs_docs_dataset.jsonl", blog_pages=blog_pages, doc_pages=doc_pages
    )
