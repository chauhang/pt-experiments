# Loading vicuna-13b with PiPPy (PyTorch Native Large inference solution)


## Clone Repo


git clone https://github.com/pytorch/serve
cd serve

## Install gRPC python dependencies

pip install -U grpcio protobuf grpcio-tools

## Generate python gRPC client stub using the proto files

python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto

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

```bash
torch-model-archiver --model-name vicuna-13b --version 1.0 --handler custom_handler.py -r requirements.txt --config-file model-config.yaml --archive-format tgz
```

## Start Torchserve

```bash
torchserve --ncs --start --model-store model_store --models vicuna-13b.tar.gz
```

## HTTP Inference

```bash
curl -v "http://localhost:8080/predictions/vicuna-13b" -F "data=How to save a model in pytorch?"
```

## gRPC Inference

```bash
python gprc_client.py infer_stream vicuna-13b "How to save a model in pytorch?"
```

## Vicuna-13b with UI

```bash
python langchain_demo.py 
```
## Finetuned with UI 

