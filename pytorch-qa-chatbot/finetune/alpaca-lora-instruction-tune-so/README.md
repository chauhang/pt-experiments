### Introduction

In this experiment, our llama-7b finetuned model is instruction tuned with SO dataset.

### Data curation

Before running the model data preparation step, ensure the data curation files are present in the system.

Check the [data curation readme](../../data_curation/README.md) for more details


### Model data preparation

To prepare the dataset in the alpaca format run the script from [data_preparation](data_preparation/README.md) folder

Run the following command

```
python alpaca_data_prep.py --stack_overflow_dataset_path pt_question_answers.csv
```

Dataset is created in alpaca format with file with name `so_dataset_alpaca_format.jsonl`

Upload the dataset to huggingface. check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html) for more info.

### Dataset Format

For this test, the question, context and answers are used.

1. The title of the post is used as question(instruction) 
2. The question context is used as context(input).
3. The accepted answer is used as output(output)


Example:

Instruction

```
Extracting the top-k value-indices from a 1-D Tensor
``` 

Input
```
Given a 1-D tensor in Torch (torch.Tensor), containing values which can be compared (say floating point), how can we extract the indices of the top-k values in that tensor? Apart from the brute-force method, I am looking for some API call, that Torch/lua provides, which can perform this task efficiently.
```

Output
```
As of pull request #496 Torch now includes a built-in API named torch.topk. Example: &gt; t = torch.Tensor{9, 1, 8, 2, 7, 3, 6, 4, 5} -- obtain the 3 smallest elements &gt; res = t:topk(3) &gt; print(res) 1 2 3 [torch.DoubleTensor of size 3] -- you can also get the indices in addition &gt; res, ind = t:topk(3) &gt; print(ind) 2 4 6 [torch.LongTensor of size 3] -- alternatively you can obtain the k largest elements as follow -- (see the API documentation for more details) &gt; res = t:topk(3, true) &gt; print(res) 9 8 7 [torch.DoubleTensor of size 3] At the time of writing the CPU implementation follows a sort and narrow approach (there are plans to improve it in the future). That being said an optimized GPU implementation for cutorch is currently being reviewed.
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
  --base_model '<user-name>/llama-finetune-7b' \
  --data_path 'so_dataset_alpaca_format.jsonl' \
  --output_dir './instruction-tune-so-delta' \
  --batch_size 1 \
  --micro_batch_size 1 \
  --num_epochs 5 \
  --cutoff_len 2048 \
  --val_set_size 0
```

Once the training is finished, the delta is saved in the outputdir (llama-finetune-pytorch-1000-delta)


### Generate full model

Once the training process is completed only the adapter files are saved under (./llama-finetune-pytorch-1000-delta) directory . 

Use the [export_hf_checkpoint.py](../../utils/export_hf_checkpoint.py) to generate the hf checkpoint

```
python export_hf_checkpoint.py --base_model <user-name>/llama-finetune-7b --lora_weights ./instruction-tune-so-delta/ --output_model_name instruction-tune-so
```

The entire model and the tokenizer is saved to the `instruction-tune-so` directory

### Upload model to huggingface

Use the [push_to_hub.py](../../utils/push_to_hub.py) to push the model into huggingface

```
export HUGGINGFACE_KEY="" #Insert your HF api key here
python push_to_hub.py --local_model_path ./instruction-tune-so --hf_model_name <user-name>/instruction-tune-so
```

### Inference

To run the basic inference, use the [generate](https://github.com/tloen/alpaca-lora/blob/main/generate.py) script from alpaca lora

```
python generate.py --base_model <user-name>/llama-finetune-7b --lora_weights <user-name>/instruction-tune-so-delta --share_gradio True
```

Copy the public URL from the terminal and open it in browser and test the inference.





