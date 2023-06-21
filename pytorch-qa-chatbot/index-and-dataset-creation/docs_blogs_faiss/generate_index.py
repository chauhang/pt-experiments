import argparse

from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS


def load_dataset_as_pages(dataset_name):
    loader = HuggingFaceDatasetLoader(dataset_name)
    pages = loader.load()
    return pages


def create_index(pages):
    embeddings = HuggingFaceEmbeddings()

    index = FAISS.from_documents([pages.pop(0)], embeddings)
    i, step = 0, 50
    while i < len(pages):
        if i % 500 == 0:
            print(i, "pages done")
        texts = [d.page_content for d in pages[i : i + step]]
        meta = [d.metadata for d in pages[i : i + step]]
        index.add_texts(texts, meta)
        i += step
    print(len(pages), "pages done")

    return index


def save_index(index, output_path):
    index.save_local(output_path)


def main(args):
    pages = load_dataset_as_pages(dataset_name=args.dataset_name_or_path)
    pt_blogs_docs_index = create_index(pages)
    save_index(index=pt_blogs_docs_index, output_path=args.index_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Index")
    parser.add_argument(
        "--dataset_name_or_path", type=str, default="shrinath-suresh/blogs-docs-splitted"
    )
    parser.add_argument("--index_save_path", type=str, default="docs_blogs_faiss_index")
    args = parser.parse_args()
    main(args)
