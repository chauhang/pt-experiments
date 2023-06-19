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
