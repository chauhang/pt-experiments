## Tests for Better Trasnformers

See HF docs for latest list of supported models: https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models

## Download the models using

```bash
python download_model.py -model_path ./  --model_name facebook/opt-13b
```

Script prints the model path something like: `models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5`

```bash
export MODEL_PATH=<path-to-model-snapshot-above>
```


## Run using

```bash
python hf_generate.py \
    --name_or_path $MODEL_PATH \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 256 \
    --better_transformer True \
    --prompts \
      "Welcome to this bold new era" \
      "It happened one day" 
```
