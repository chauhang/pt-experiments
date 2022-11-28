## News Classification using HF with torchdynamo optimization

The example is adapted from https://github.com/mlflow/mlflow-torchserve/tree/master/examples/BertNewsClassification 

This example demonstrates model optimization using torchdynamo.


## Prerequisites

Install torch from nightly - https://pytorch.org/ - check Install PyTorch section

Install other dependent packages using pip

```

pip install mlflow, sklearn, transformers, torchtext, pandas, datasets
```

## Running the example

Update the torch dynamo backend in the script

For training use any one of the backend

1. eager
2. ts_nvf_user
3. aot_cudagraphs

by default the script is set to eager backend

Check `my_compiler` method for more details

Run the following command 

```
torchrun news_classifier.py --max_epochs 1
```

By default script utilizes all the gpus available in the machine


