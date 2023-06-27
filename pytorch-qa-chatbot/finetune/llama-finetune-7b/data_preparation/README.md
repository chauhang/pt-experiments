## Steps for Data Preparation

Data Sources

1. Docs dataset - docs.json
2. Blogs dataset - blogs.json

Download the datasets into the current directory

Run the following command

```
python llama_finetune_prepare.py --pt_docs_dataset_path docs.json --pt_blogs_dataset_path blogs.json
```

The command loads docs and blogs data and generates the dataset as paragraphs.

The dataset will be stored as - `blogs_docs_dataset.jsonl`