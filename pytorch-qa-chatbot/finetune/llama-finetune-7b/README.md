### Introduction

In this experiment, we finetune llama-7b using alpaca-lora using PyTorch Blogs and Docs dataset.
### Data curation

Before running the model data preparation step, ensure the data curation files are present in the system.

Check the [data curation readme](../../data_curation/README.md) for more details

### Model data preparation

To prepare the dataset run the script from [data_preparation](data_preparation/README.md) folder

Run the following command

```
python llama_finetune_prepare.py --pt_docs_dataset_path docs.json --pt_blogs_dataset_path blogs.json
```

Dataset is created paragraphs with file with name `blogs_docs_dataset.jsonl`

Upload the dataset to huggingface. check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html) for more info.

### Dataset Format

For this test, the dataset is prepared as paragraphs from the blogs and docs

Example:

```
Text: \nlayout: blog_detail\ntitle: What\u2019s New in PyTorch Profiler 1.9?'\nauthor: Sabrina Smai, Program Manager on the AI Framework team at Microsoft\n\nPyTorch Profiler v1.9 has been released! The goal of this new release (previous PyTorch Profiler release) is to provide you with new state-of-the-art tools to help diagnose and fix machine learning performance issues regardless of whether you are working on one or numerous machines. The objective is to target the execution steps that are the most costly in time and/or memory, and visualize the work load distribution between GPUs and CPUs. \nHere is a summary of the five major features being released: \n### source: https://pytorch.org/blog/pytorch-profiler-1.9-released/ \n### category: pytorch blogs
``` 

### Prompt Template

We have used custom formatting for input. 

```
f"### Text: {text} \n### source: {source} \n### category: {category}\n"
```

check the [repo](https://github.com/tloen/alpaca-lora/tree/main/templates) for more details


### Model finetuning

Follow the environment setup instructions from the [alpaca-lora repo](https://github.com/tloen/alpaca-lora.git).

Once the required packages are installed using 

```
pip install -r requirements.txt
```

We will be using our own llama_finetune.py for training. Copy the file into alpaca-lora folder

```
cp llama_finetune.py alpaca-lora
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
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path 'blogs_docs_dataset.jsonl' \
  --output_dir './llama-finetune-7b-delta' \
  --batch_size 4 \
  --micro_batch_size 1 \
  --num_epochs 5 \
  --cutoff_len 1024 \
  --val_set_size 0
```

Once the training is finished, the delta is saved in the outputdir (llama-finetune-7b-delta)


### Generate full model

Once the training process is completed only the adapter files are saved under (./llama-finetune-7b-delta) directory . 

Use the [export_hf_checkpoint.py](../../utils/export_hf_checkpoint.py) to generate the hf checkpoint

```
python export_hf_checkpoint.py --base_model decapoda-research/llama-7b-hf --lora_weights ./llama-finetune-7b-delta/ --output_model_name llama-finetune-7b
```

The entire model and the tokenizer is saved to the `llama-finetune-7b` directory

### Upload model to huggingface

Use the [push_to_hub.py](../../utils/push_to_hub.py) to push the model into huggingface

```
export HUGGINGFACE_KEY="" #Insert your HF api key here
python push_to_hub.py --local_model_path ./llama-finetune-7b --hf_model_name <user-name>/llama-finetune-7b
```

### Usage

We will be using this model as base model to instruction tune on our dataset.





