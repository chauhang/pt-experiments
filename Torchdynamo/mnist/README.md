## News Classification using HF with torchdynamo optimization

The example is adapted from https://github.com/pytorch/examples/tree/main/mnist

This example demonstrates model optimization using torchdynamo.


## Prerequisites

Install torch from nightly - https://pytorch.org/ - check Install PyTorch section


## Running the example

The entire training module is wrapped with dynamo optimization decorator

```
@dynamo.optimize("aot_cudagraphs")
```

Select any one of the training backend

1. eager
2. ts_nvf_user
3. aot_cudagraphs

For more information - https://github.com/pytorch/torchdynamo#existing-backends


Run the following command 

```
python mnist.py
```
