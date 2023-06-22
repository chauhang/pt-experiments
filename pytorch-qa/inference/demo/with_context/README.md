### Prerequisite
Install the required packages

```
pip install -r requirements.txt
```

### Testing model with question and context


`langchain_demo_with_context.py` - use this script to test finetuned model with question and context

```
python langchain_demo_with_context.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Prompts for the tests are available in [question with context prompts](question_with_context.json).

To test with different prompt, pass it as an argument

```
python langchain_demo_with_context.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary --prompt_name QUESTION_WITH_CONTEXT_PROMPT_BASIC_PROMPT
```

Copy the public url from terminal, paste it in browser and start testing


