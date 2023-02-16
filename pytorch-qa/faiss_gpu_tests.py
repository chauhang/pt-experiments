import time

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


def read_embeddings(file_name="embeddings_14k.csv"):
    df = pd.read_csv(file_name)
    df["embeddings"] = df["embeddings"].apply(
        lambda x: np.fromstring(
            x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=" "
        )
    )

    return df


def index_using_faiss_GpuIndexFlatL2(df, iterations=6):
    x = df.embeddings.tolist()
    x = np.array(x)

    x = x.reshape(x.shape[0], -1).astype("float32")
    d = x.shape[1]

    start_time = None
    for i in tqdm(range(iterations)):
        if i == 1:
            start_time = time.time()

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        index.add(x)

    print(
        "time taken to add - using  faiss.GpuIndexFlatL2: ",
        (time.time() - start_time) / iterations * 1000,
    )


def index_using_hf_faiss(df, iterations=6, device=None):
    embeddings_dataset = Dataset.from_pandas(df)

    start_time = time.time()

    for i in range(iterations):
        if i == 1:
            start_time = time.time()

        if device == "cpu":
            embeddings_dataset.add_faiss_index(column="embeddings")
        else:
            embeddings_dataset.add_faiss_index(column="embeddings", device=device)

    if device == "cpu":
        print(
            "time taken to add - using  dataset.add_faiss_index - cpu:",
            (time.time() - start_time) / iterations * 1000,
        )

    else:
        print(
            "time taken to add - using  dataset.add_faiss_index - gpu:",
            (time.time() - start_time) / iterations * 1000,
        )


if __name__ == "__main__":
    print("Loading dataset")
    df = read_embeddings()

    print("Input Shape: ", df.shape)

    print("\nIndexing using faissGpuIndexFlatL2")
    index_using_faiss_GpuIndexFlatL2(df)

    print("\nIndexing using hf_faiss gpu")
    index_using_hf_faiss(df, device=0)

    print("\nIndexing using hf_faiss cpu")
    index_using_hf_faiss(df, device="cpu")
