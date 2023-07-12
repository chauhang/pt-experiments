### Objective

To evaluate finetuned LLMs with different set of prompts and inputs

### Setup

Install the required packages

```
pip install requirements.txt
```

Update the models to be tested in `models.json` file

For ex:

```
[
  "shrinath-suresh/alpaca-lora-7b-answer-summary",
  "shrinath-suresh/alpaca-lora-7b-context-summary"
]

```

To include new questions, update `questions.json`
To update or add a new prompt, edit `prompts.json`

### Start the tests

To start the tests, run the following command

```
python run_test.py --results_path llm_results
```

Script will test the models against the prompts (by default 6) and questions (by default 20)
and generate the output in `llm_results` directory

For each model, there will be folder with two csv files (results of with and without context)

For ex:

```.
├── shrinath-suresh_alpaca-lora-7b-answer-summary
│   ├── test_shrinath-suresh_alpaca-lora-7b-answer-summary_only_question.csv
│   └── test_shrinath-suresh_alpaca-lora-7b-answer-summary_question_with_context.csv
├── shrinath-suresh_instruction-tune-so
│   ├── test_shrinath-suresh_instruction-tune-so_only_question.csv
│   └── test_shrinath-suresh_instruction-tune-so_question_with_context.csv


```

Note: At the end of the tests, huggingface cache directory is removed `/home/ubuntu/.cache/huggingface` to maintain disk space.
To avoid, comment the `remove_cache` call from main method.

### Generating summary

When testing more number of models, it is difficult to open each and every file and validate the output.

Use the following script to summarize all the results in a single excel file.

```
python generate_summary.py
```

At the end of the script two summary files will be created.

1. only_question_summary.xlsx - contains the summary of llm tests with only questions
2. question_with_context_summary.xlsx - contains the summary of llm tests with question and context.
