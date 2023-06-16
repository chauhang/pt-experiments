### Introduction

This folder contains script to run the tests with and without context

### Setup

Install the required packages

```
pip install -r requirements.txt
```

### Scripts

1. `langchain_demo.py` - use this script to test finetuned model with question (no context)
2. `langchain_demo_with_context.py` - use this script to test finetuned model with question and context

To test finetuned model with only question, run the following command


### Testing model with question

```
python langchain_demo.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Prompts for the tests are available in [only questions prompts](only_question_prompts.json).

To test with different prompt, pass it as an argument

```
python langchain_demo.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary --prompt_name ONLY_QUESTION_PROMPT_BASIC_PROMPT
```

Copy the public url from terminal, paste it in browser and start testing


### Testing model with question and context


```
python langchain_demo_with_context.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Prompts for the tests are available in [question with context prompts](question_with_context.json).

To test with different prompt, pass it as an argument

```
python langchain_demo_with_context.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary --prompt_name QUESTION_WITH_CONTEXT_PROMPT_BASIC_PROMPT
```

Copy the public url from terminal, paste it in browser and start testing


