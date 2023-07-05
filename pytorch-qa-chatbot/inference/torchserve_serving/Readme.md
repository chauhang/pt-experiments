# Serving LLM using Torchserve and Huggingface accelerate


## Clone Repo

```bash
git clone https://github.com/pytorch/serve
cd serve
```

## Download the model

Navigate to examples/large_models/

- Login with HF cli
- set `use_auth_token=True` in Download_model.py file

```bash
python ../utils/Download_model.py --model_name jagadeesh/vicuna-13b
```

The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints.

`model/models--jagadeesh--vicuna-13b/snapshots/ca289bf4a98a874f27d3fa166b164827b82e6753`

## Packaging the model

1. Update the model-config.yaml file with `handler` and `frontend` settings
   
```bash
# Generate archive file
torch-model-archiver --model-name vicuna-13b --version 1.0 --handler custom_handler.py -r requirements.txt --config-file model-config.yaml --archive-format tgz

# Move model to model_store
mkdir model_store
mv vicuna-13b.tar.gz model_store/
```

## Start Torchserve

```bash
torchserve --ncs --start --model-store model_store --models vicuna-13b.tar.gz
```

## HTTP Inference

```bash
curl -v "http://localhost:8080/predictions/vicuna-13b" -F "data=How to save a model in pytorch?"
```

## gRPC

### Install gRPC python dependencies

```bash
pip install -U grpcio protobuf grpcio-tools
```

### Generate python gRPC client stub using the proto files

Navigate to serve folder

```bash
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=./ --grpc_python_out=./ frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

### Inference

```bash
python gprc_client.py infer_stream vicuna-13b "How to save a model in pytorch?"
```
