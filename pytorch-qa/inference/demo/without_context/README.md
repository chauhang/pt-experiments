### Prerequisite

Install the required packages

```
pip install -r requirements.txt
```


### Testing model without context

`langchain_demo.py` - use this script to test fine-tuned model with question (no context)

```
python langchain_demo.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Prompts for the tests are available in [only questions prompts](only_question_prompts.json).

To test with different prompt, pass it as an argument

```
python langchain_demo.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary --prompt_name ONLY_QUESTION_PROMPT_BASIC_PROMPT
```

Copy the public url from terminal, paste it in browser and start testing
