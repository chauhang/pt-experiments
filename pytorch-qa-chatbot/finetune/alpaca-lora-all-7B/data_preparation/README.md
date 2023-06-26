## Steps for Data Preparation

Data Sources for blogs and docs
1. Docs dataset - docs.json
2. Blogs Forum dataset - blogs.json

Once the dataset is downloaded into the directory, run the following command to prepare the datasets

Set the OPEN AI Key

```
export OPENAI_API_KEY = "" #Insert your openai key here
```

run the following command to perform question generation on docs and blogs

```
python docs_blogs_qa_generation --docs_dataset_path docs.json --blogs_dataset_path blogs.json
```

Following output files will be generated

```
docs_qa_dataset.json
blogs_qa_dataset.json
```

If you already have these files, skip the previous steps and proceed for dataset preparation

Download the remaining datasets

1. Stack overflow dataset - stack_overflow.json
2. Discuss Forum dataset - discussion_forum.json


Run the following command

```
python alpaca_data_prep.py
```

The command loads SO, PT Discuss forum, PT Docs and Blogs data and generates the dataset in alpaca format.

The dataset will be stored as - `pytorch_all_alpaca_format.json`