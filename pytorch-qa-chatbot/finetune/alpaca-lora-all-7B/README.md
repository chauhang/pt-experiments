### Introduction

In this experiment, alpaca-7b model is instruction tuned with Stack Overflow, Discussion Forum, PyTorch Docs and Blogs dataset

### Data curation

Before running the model data preparation step, ensure the data curation files are present in the system.

Check the [data curation readme](../../data_curation/README.md) for more details

### Model data preparation

To prepare the dataset in the alpaca format run the script from [data_preparation](data_preparation/README.md) folder

Run the following command

```
python alpaca_data_prep.py --stack_overflow_dataset_path stack_overflow.json --pt_discuss_dataset_path discussion_forum.json --pt_docs_dataset_path docs_qa_dataset.json --pt_blogs_discuss_path blogs_qa_dataset.json
```

Dataset is created in alpaca format with file with name `pytorch_all_alpaca_format.json`

Upload the dataset to huggingface. check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html) for more info.

### Dataset Format

For this test only the question and context are considered.

From the stack overflow and discussion forum posts, the title of the post is used as question(instruction) and accepted answer is used as answer(output).

Example:

Instruction

```
how do i save a trained model in pytorch?
``` 

Input - Empty
```

```

Output
```
found this page on their github repo:\n\nrecommended approach for saving a model\nthere are two main approaches for serializing and restoring a model.\nthe first (recommended) saves and loads only the model parameters:\ntorch.save(the_model.state_dict(), path)\n\nthen later:\nthe_model = themodelclass(*args, **kwargs)\nthe_model.load_state_dict(torch.load(path))\n\nthe second saves and loads the entire model:\ntorch.save(the_model, path)\n\nthen later:\nthe_model = torch.load(path)\n\nhowever in this case, the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors.\n\n\nsee also: save and load the model section from the official pytorch tutorials.\n
```

### Prompt Template

Alpaca lora - default prompt template is used.

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n
```

check the [repo](https://github.com/tloen/alpaca-lora/tree/main/templates) for more details


### Model finetuning

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

```
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_endpoint 127.0.0.1 --rdzv_id 12345 --rdzv_backend c10d finetune.py --base_model 'decapoda-research/llama-7b-hf' --data_path 'so_discuss_alpaca_format.json' --output_dir './alpaca-lora-all-7B' --batch_size 4 --micro_batch_size 1 --num_epochs 5 --cutoff_len 512 --val_set_size 0
```

Once the training is finished, the delta is saved in the outputdir (alpaca-lora-all-7B)


### Generate full model

Once the training process is completed only the adapter files are saved. To generate the full model use the alpaca lora [export_hf_checkpoint](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py) script

```
export BASE_MODEL=decapoda-research/llama-7b-hf 
```

Open the file and replace the delta path from
```
lora_model = PeftModel.from_pretrained(
    base_model,
    "tloen/alpaca-lora-7b",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)
```

with 
```
lora_model = PeftModel.from_pretrained(
    base_model,
    "alpaca-lora-all-7B",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)
```


And run the command to export the model

```
python export_hf_checkpoint.py
```

The full model is generated in the current directory. 

### Upload model to huggingface

Use the following code snippet

```
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("hf_ckpt")
api_key = "" ### Insert your HF key here

model.push_to_hub(repo_id="<user-name>/alpaca-lora-all-7B", private=True, use_auth_token=api_key)
```

Check this [tutorial](https://huggingface.co/docs/transformers/model_sharing) for more details


### Inference

To run the basic inference, use the [generate](https://github.com/tloen/alpaca-lora/blob/main/generate.py) script from alpaca lora

```
python generate.py --base_model decapoda-research/llama-7b-hf --lora_weights <user-name>/alpaca-lora-all-7B --share_gradio True
```

Copy the public URL from the terminal and open it in browser and test the inference.





