### Introduction

In this experiment, alpaca-13b model is instruction tuned with the summarized version of the Stack Overflow dataset

### Data curation

Before running the model data preparation step, ensure the data curation files are present in the system.

Check the [data curation readme](../../data_curation/README.md) for more details

### Model data preparation

To prepare the dataset in the alpaca format run the script from [data_preparation](data_preparation/README.md) folder

Run the following command

```
python alpaca_data_prep.py --stack_overflow_dataset_path pt_question_answers.csv
```

Dataset is created in alpaca format with file with name `pytorch_so_answer_summary_alpaca_format.json`

Upload the dataset to huggingface. check this [tutorial](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html) for more info.

### Dataset Format

For this test, the question, context and answers are used.

From the stack overflow and discussion forum posts, 

1. The title and the question body of the post is used as question(instruction) 
2. The accepted answer is used as context(input).
3. The summarized version of the answer is used as output(output)

Example:

Instruction

```
extracting the top-k value-indices from a 1-d tensor Given a 1-D tensor in Torch (torch.Tensor), containing values which can be compared (say floating point), how can we extract the indices of the top-k values in that tensor? Apart from the brute-force method, I am looking for some API call, that Torch/lua provides, which can perform this task efficiently.
``` 

Input
```
as of pull request #496 torch now includes a built-in api named torch.topk. example: &gt; t = torch.tensor{9, 1, 8, 2, 7, 3, 6, 4, 5} -- obtain the 3 smallest elements &gt; res = t:topk(3) &gt; print(res) 1 2 3 [torch.doubletensor of size 3] -- you can also get the indices in addition &gt; res, ind = t:topk(3) &gt; print(ind) 2 4 6 [torch.longtensor of size 3] -- alternatively you can obtain the k largest elements as follow -- (see the api documentation for more details) &gt; res = t:topk(3, true) &gt; print(res) 9 8 7 [torch.doubletensor of size 3] at the time of writing the cpu implementation follows a sort and narrow approach (there are plans to improve it in the future). that being said an optimized gpu implementation for cutorch is currently being reviewed.
```

Output
```
An expert PyTorch engineer would suggest using the built-in API call torch.topk to efficiently extract the top-k value indices from a 1-D tensor. The engineer would provide an example code snippet showcasing how to use the API call to obtain the k largest or smallest elements along with their indices. The engineer would also mention that while the current CPU implementation follows a sort and narrow approach, an optimized GPU implementation for cutorch is under review.
```

### Prompt Template

Alpaca lora - default prompt template is used.

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n
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

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv_endpoint 127.0.0.1 \
  --rdzv_id 12345 \
  --rdzv_backend c10d \
  finetune.py \
  --base_model 'decapoda-research/llama-13b-hf' \
  --data_path 'pytorch_so_answer_summary_alpaca_format.json' \
  --output_dir './alpaca-lora-13B-answer-summary-delta' \
  --batch_size 1 \
  --micro_batch_size 1 \
  --num_epochs 5 \
  --cutoff_len 2048 \
  --val_set_size 0
```

Once the training is finished, the delta is saved in the outputdir (alpaca-lora-13B-answer-summary-delta)


### Generate full model

Once the training process is completed only the adapter files are saved under (./alpaca-lora-13B-answer-summary-delta) directory . 

Use the [export_hf_checkpoint.py](../../utils/export_hf_checkpoint.py) to generate the hf checkpoint

```
python export_hf_checkpoint.py --base_model decapoda-research/llama-13b-hf --lora_weights ./alpaca-lora-13b-answer-summary-delta/ --output_model_name alpaca-lora-13b-answer-summary
```

The entire model and the tokenizer is saved to the `alpaca-lora-13b-answer-summary` directory

### Upload model to huggingface

Use the [push_to_hub.py](../../utils/push_to_hub.py) to push the model into huggingface

```
export HUGGINGFACE_KEY="" #Insert your HF api key here
python push_to_hub.py --local_model_path ./alpaca-lora-13b-answer-summary --hf_model_name <user-name>/alpaca-lora-13b-answer-summary
```

### Inference

To run the basic inference, use the [generate](https://github.com/tloen/alpaca-lora/blob/main/generate.py) script from alpaca lora

```
python generate.py --base_model decapoda-research/llama-13b-hf --lora_weights <user-name>/alpaca-lora-13B-answer-summary --share_gradio True
```

Copy the public URL from the terminal and open it in browser and test the inference.





