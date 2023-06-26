## Steps for Data Preparation

Data Sources

Hand curated dataset generated from PyTorch discussion forums, PyTorch tutorials, PyTorch FAQ and SO and Blogs dataset. 

Download the datasets into the current directory.

To generate blogs curated data using OPENAI, download blogs.json from data curation step into the current directory.

Files to be present - blogs.json, blogs_curated_data.json, discussion_forum_curated.json, pt_tutorial.json, pytorch_faq.json, pt_question_answers.csv

As we are using OPENAI to generate qa, please provide OPENAI API KEY in environment variable.
```
export OPENAI_API_KEY=''
```

Run the following command
```
python blogs_qa_generation.py
```

To generate data in alpaca format.

Run the following command

```
python alpaca_data_prep.py
```

The command loads curated dataset generates the dataset in alpaca format.

The dataset will be stored as - `pt_curated_1000_alpaca_fomrat.json`