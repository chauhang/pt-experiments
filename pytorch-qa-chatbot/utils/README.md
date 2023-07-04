### Exporting HF Checkpoint

`export_hf_checkpoint.py` can be used for saving the entire model in huggingface format.

The script is adapted from - https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py

To save the model and tokenizer, following inputs are needed

1. Base model (--base_model) - HF model name of the base model used for training
2. Lora Weights (--lora_weights) - Local path to the lora weights (adapter files)
3. Output model name (--output_model_name) - Name of the huggingface model name to be generated

```
python export_hf_checkpoint.py --base_model decapoda-research/llama-7b-hf --lora_weights /home/ubuntu/lora_weights --output_model_name alpaca-7b-test
```

The huggingface model and the tokenizer files are saved under `alpaca-7b-test`


### Pushing model to huggingface hub

`push_to_hub.py` can be used to push the local hf model into HugingfaceHub.

Set the api key as environment variable

```
export HUGGINGFACE_KEY="" #Insert your HF api key here
```

Run the following command

```
python push_to_hub.py --local_model_path <local-model-path> --hf_model_name <hf-model-name>
```

For ex:

```
python push_to_hub.py --local_model_path /home/ubunu/alpaca-7b-test --hf_model_name shrinath-suresh/alpaca-7b-test
```
