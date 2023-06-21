### Introduction

This experiment creates an index using pytorch blogs and docs and saves it in local directory.


### Setup

Before generating the index, the dataset needs to be prepared

1. Follow the Blogs and Docs preparation script from [data-curation](../../data_curation) directory.

2. Once `blogs.json` and `docs.json` are available, use the [hf-dataset](hf_dataset_preparation_blogs_docs.py)
script to create a huggingface dataset. 

### Generate HF Dataset

```bash
python hf_dataset_preparation_blogs_docs.py 
```

1. Upload the `blogs_splitted_dataset.jsonl` to huggingface and create a new dataset. 
To upload the dataset to huggingface, check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html)

### Generate and save index

To generate the index, run the following command

```
python generate_index.py --dataset_name_or_path <HF-DATASET-NAME> --index_save_path docs_blogs_faiss_index
```

The pytorch blogs and docs will be chunked and indexed into the directory - `docs_blogs_faiss_index`.
