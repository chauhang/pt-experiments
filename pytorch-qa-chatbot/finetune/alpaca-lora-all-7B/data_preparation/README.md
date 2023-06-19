## Steps for Data Preparation

Data Sources

1. Stack overflow dataset - stack_overflow.json
2. Discuss Forum dataset - discussion_forum.json
3. Docs dataset - docs_qa_dataset.json
4. Blogs Forum dataset - blogs_qa_dataset.json

Download the datasets into the current directory

Run the following command

```
python alpaca_data_prep.py
```

The command loads SO, PT Discuss forum, PT Docs and Blogs data and generates the dataset in alpaca format.

The dataset will be stored as - `pytorch_all_alpaca_format.json`