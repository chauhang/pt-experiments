## Steps for Data Preparation

Data Sources

1. Stack overflow dataset - "../../../data_curation/data_sources/pt_question_answers.csv"

Download the datasets into the current directory from the above path

As we are using OPENAI to summarize answer, please provide OPENAI API KEY in environment variable.
```
export OPENAI_API_KEY=''
```

Run the following command

```
python alpaca_data_prep.py --stack_overflow_dataset_path 'pt_question_answers.csv'
```

The command loads SO data and generates the dataset in alpaca format.

The dataset will be stored as - `pytorch_so_answer_summary_alpaca_format.json`