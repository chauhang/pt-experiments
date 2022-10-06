## Bloom

Bloom is an Open Multilingual Language Model. To know more about bloom - https://huggingface.co/bigscience/bloom


## Objective

The objective is to serve the bloom model in standalone torcserve instance

## Downloading the model

Clone the serve repository

```
git clone https://github.com/pytorch/serve.git
```

Move to the Hugging face example folder

```
examples/Huggingface_Transformers
```

Replace the `setup_config.json`, `Download_Transformer_models`, `Transformer_handler_generalized` files with the files from current directory.


## Setting up the task

Update the task name in `mode` parameter of `setup_config.json`. 

Select the bloom model name from - https://huggingface.co/docs/transformers/model_doc/bloom

For example, to infer token classification task

```
{
 "model_name":"bigscience/bloom-560m",
 "mode":"token_classification",
 "do_lower_case":true,
 "num_labels":"2",
 "save_mode":"pretrained",
 "max_length":"150",
 "captum_explanation":true,
 "embedding_name": "bert",
 "FasterTransformer":false,
 "model_parallel":false
}
```

And for sequence classifiation task

```
{
 "model_name":"bigscience/bloom-560m",
 "mode":"sequence_classification",
 "do_lower_case":true,
 "num_labels":"2",
 "save_mode":"pretrained",
 "max_length":"150",
 "captum_explanation":true,
 "embedding_name": "bert",
 "FasterTransformer":false,
 "model_parallel":false
}
```

## Download the model

To download the bloom model, run the following script

```
python Download_Transformer_models
```

The script will download the bloom model, tokenizer and supporting artifacts inside `Transformer_model` directory


## Package the model

Run the following command to package the model

```
torch-model-archiver --model-name bloom --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./Token_classification_artifacts/index_to_name.json"

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
curl -X POST http://127.0.0.1:8080/predictions/bloom -T Token_classification_artifacts/sample_text_captum_input.txt
```
