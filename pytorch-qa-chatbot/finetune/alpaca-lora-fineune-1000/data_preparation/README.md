## Steps for Data Preparation

Data Sources

Hand curated dataset generated from PyTorch discussion forums, PyTorch tutorials, PyTorch FAQ and SO and Blogs dataset. 

Download the datasets into the current directory.

To generate blogs curated data using OPENAI, download blogs.json from data curation step into the current directory.

Files to be present - 

1. blogs.json
2. blogs_curated_data.json
3. discussion_forum_curated.json
4. pt_tutorial.json
5. pytorch_faq.json
6. pt_question_answers.csv

As we are using OPENAI to generate qa, please provide OPENAI API KEY in environment variable.
```
export OPENAI_API_KEY=''
```

Run the following command
```
python blogs_qa_generation.py
```

To generate data in alpaca format, run the following command

```
python alpaca_data_prep.py
```

The command loads curated dataset generates the dataset in alpaca format.

The dataset will be stored as - `pt_curated_1000_alpaca_fomrat.json`