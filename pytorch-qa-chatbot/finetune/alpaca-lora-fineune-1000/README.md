### Introduction

In this experiment, alpaca-7b model is instruction tuned with hand curated dataset generated from PyTorch discussion forums, PyTorch tutorials, PyTorch FAQ and SO and Blogs dataset.  

### Data curation

Before running the model data preparation step, ensure the data curation files are present in the system.

Check the [data curation readme](../../data_curation/README.md) for more details

We have hand curated dataset in following size from sources.
1. 178 questions and answers from [discussion forums](https://discuss.pytorch.org/)
2. 168 questions and answers from [tutorials](https://pytorch.org/tutorials/)
3. 37 questions and answers from [faq](https://pytorch.org/docs/stable/notes/faq.html)
4. 400 top questions and answers from SO
5. 216 questions and answers from blogs generated using OPENAI

### Model data preparation

To prepare the dataset in the alpaca format run the script from [data_preparation](data_preparation/README.md) folder

Run the following command

```
python alpaca_data_prep.py --stack_overflow_dataset_path pt_question_answers.csv --pt_discuss_dataset_path discussion_forum_curated.json --pt_tutorial_dataset_path pt_tutorial.json --pt_faq_dataset_path pt_faq.json --blogs_curated_dataset_path blogs_curated_data.json
```

Dataset is created in alpaca format with file with name `pt_curated_1000_alpaca_format.json`

Upload the dataset to huggingface. check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html) for more info.

### Dataset Format

For this test, the question, context and answers are used.

1. The title of the post is used as question(instruction) 
2. The post body/context is used as context(input).
3. The hand curated answer is used as output(output)


Example:

Instruction

```
How to get intermediate CNN layer features from my trained model?
``` 

Input
```
I have my own CNN model, I want to get intermediate CNN layer features from my trained model, so that I can supply them as input to autoencoders. Create two different neural network modules. Train it and use the module that is of your interest
```

Output
```
Create two different neural network modules. Train it and use the module that is of your interest
```

### Prompt Template

Alpaca lora - default prompt template is used.

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n
```

check the [repo](https://github.com/tloen/alpaca-lora/tree/main/templates) for more details


### Model finetuning

We will be using our own alpaca lora model as [base](../llama-finetune-7b) trained on llama-7b using pytorch dataset.

Follow the environment setup instructions from the [alpaca-lora repo](https://github.com/tloen/alpaca-lora.git).

Once the required packages are installed using 

```
pip install -r requirements.txt
```

Run the following command to finetune the model

Move to alpaca-lora

```
cd alpaca-lora
```

and run the following command to start the training

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv_endpoint 127.0.0.1 \
  --rdzv_id 12345 \
  --rdzv_backend c10d \
  finetune.py \
  --base_model 'shrinath-suresh/llama-finetune-7b' \
  --data_path 'pt_curated_1000_alpaca_format.json' \
  --output_dir './llama-finetune-pytorch-1000-delta' \
  --batch_size 1 \
  --micro_batch_size 1 \
  --num_epochs 15 \
  --cutoff_len 2048 \
  --val_set_size 0
```

Once the training is finished, the delta is saved in the outputdir (llama-finetune-pytorch-1000-delta)


### Generate full model

Once the training process is completed only the adapter files are saved under (./llama-finetune-pytorch-1000-delta) directory . 

Use the [export_hf_checkpoint.py](../../utils/export_hf_checkpoint.py) to generate the hf checkpoint

```
python export_hf_checkpoint.py --base_model shrinath-suresh/llama-finetune-7b --lora_weights ./llama-finetune-pytorch-1000-delta/ --output_model_name llama-finetune-pytorch-1000
```

The entire model and the tokenizer is saved to the `llama-finetune-pytorch-1000` directory

### Upload model to huggingface

Use the [push_to_hub.py](../../utils/push_to_hub.py) to push the model into huggingface

```
export HUGGINGFACE_KEY="" #Insert your HF api key here
python push_to_hub.py --local_model_path ./llama-finetune-pytorch-1000 --hf_model_name <user-name>/llama-finetune-pytorch-1000
```

### Inference

To run the basic inference, use the [generate](https://github.com/tloen/alpaca-lora/blob/main/generate.py) script from alpaca lora

```
python generate.py --base_model shrinath-suresh/llama-finetune-7b --lora_weights <user-name>/llama-finetune-pytorch-1000-delta --share_gradio True
```

Copy the public URL from the terminal and open it in browser and test the inference.





