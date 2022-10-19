## Bloom

Bloom is an Open Multilingual Language Model. To know more about bloom - https://huggingface.co/bigscience/bloom


## Objective

The objective is to load and infer the bloom model in standalone torchserve instance.

## Mode

This example demonstrates, bloom text generation using torchserve.

For more information - https://huggingface.co/docs/transformers/model_doc/bloom

## Setting up the task


Select the bloom model name from - https://huggingface.co/docs/transformers/model_doc/bloom

Update the setup_config.json with the model name.

For example,

```
{
 "model_name":"bigscience/bloom-560m",
 "max_length":"150"
}
```

## Package the model

Run the following command to package the model

```
torch-model-archiver --model-name bloom --version 1.0  --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json"

```

`bloom.mar` file will be generated in the current directory

Create a new directory model_store and move bloom model inside it

```
mkdir model_store
mv bloom.mar model_store
```

## Start torchserve

```
torchserve --start --model-store model_store --models bloom=bloom.mar --ncs
```

## Run inference

```
curl -X POST http://127.0.0.1:8080/predictions/bloom -T Text_gen_artifacts/sample_text.txt
```


## Sample output

```
Today the weather is really nice and I am planning on  holidays with my friends. What do you think of the holidays and the weather?
```