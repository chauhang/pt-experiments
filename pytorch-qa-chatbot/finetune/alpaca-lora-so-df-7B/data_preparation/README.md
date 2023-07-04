## Steps for Data Preparation

Data Sources

1. Stack overflow dataset - so_discuss_data.json
2. Discuss Forum dataset - discussion_forum.json

Download the datasets into the current directory

Run the following command

```
python alpaca_data_prep.py
```

The command loads both SO and PT Discuss forum data and generates the dataset in alpaca format.

The dataset will be stored as - `so_discuss_data.json`