## Steps for Data Preparation

Data Sources

1. Stack overflow dataset answer summarized - pytorch_so_answer_summary_alpaca_format.json

Download the datasets into the current directory

Run the following command

```
python alpaca_data_prep.py
```

The command loads SO data and generates the dataset in alpaca format.

The dataset will be stored as - `pytorch_so_answer_summary_alpaca_format_cleaned.json`