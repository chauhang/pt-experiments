### Introduction

Following three scenarios are convered in this tests

1. Single QA with finetuned model
2. Multi-turn QA with finetuned model
3. Single QA with finetuned model and index 
4. Single QA with vicuna 13b model and index

### Prerequisite
Install the required packages

```
pip install -r requirements.txt
```

Export your huggingface key as environment variable

```
export HUGGINGFACE_KEY = "" #Insert your huggingface key here
```

### Single QA with finetuned model

To test single qa with finetuned model, run the following command

```bash
python run_chatbot.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Copy the public url from terminal, paste it in browser and start testing

### Multi-turn QA with finetuned model

To test multi turn qa with finetuned model, run the following command

```bash
python run_chatbot.py --multiturn True --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Copy the public url from terminal, paste it in browser and start testing


### Single QA with finetuned model and index

To test multi turn qa with finetuned model with index, run the following command

```bash
python run_chatbot_with_index.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

Copy the public url from terminal, paste it in browser and start testing

### Single QA with vicuna 13b model and index

To test multi turn qa with vicuna 13b model with index, run the following command

```bash
python run_chatbot_with_index.py --model_name jagadeesh/vicuna-13b
```

Copy the public url from terminal, paste it in browser and start testing
