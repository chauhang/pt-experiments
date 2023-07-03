### Introduction

Following three scenarios are convered in this tests

1. Single QA with finetuned model
2. Multi-turn QA with finetuned model
3. Single QA with finetuned model and index 
4. Single QA with vicuna 13b model and index

### Prerequisite
Install the required packages

```bash
pip install -r requirements.txt
```

### Serve Model with Torchserve 

##### Clone Repo

```bash
git clone https://github.com/pytorch/serve
cd serve
```

#### Download the model

Navigate to examples/large_models/

- Login with HF cli
- set `use_auth_token=True` in Download_model.py file

```bash
python ../utils/Download_model.py --model_name shrinath-suresh/alpaca-lora-7b-answer-summary
```

The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints.

`model/models--shrinath-suresh--alpaca-lora-7b-answer-summary/snapshots/ca289bf4a98a874f27d3fa166b164827b82e6753`

#### Packaging the model

1. Update the model-config.yaml file with `handler` and `frontend` settings
   
```bash
# Generate archive file
torch-model-archiver --model-name alpaca-7b --version 1.0 --handler custom_handler.py -r requirements.txt --config-file model-config.yaml --archive-format tgz

# Move model to model_store
mkdir model_store
mv alpaca-7b.tar.gz model_store/
```

#### Start Torchserve

```bash
torchserve --ncs --start --model-store model_store --models alpaca-7b.tar.gz
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

### Single QA with finetuned model

To test single qa with finetuned model, run the following command

```bash
python run_chatbot.py --model_name alpaca-7b
```

Copy the public url from terminal, paste it in browser and start testing

### Multi-turn QA with finetuned model

To test multi turn qa with finetuned model, run the following command

```bash
python run_chatbot.py --multiturn True --model_name alpaca-7b
```

Copy the public url from terminal, paste it in browser and start testing


### Single QA with finetuned model and index

To test multi turn qa with finetuned model with index, run the following command

Download and extract index from s3

```bash
python run_chatbot_with_index.py --model_name alpaca-7b
```

Copy the public url from terminal, paste it in browser and start testing

### Single QA with vicuna 13b model and index

To test multi turn qa with vicuna 13b model with index, run the following command

```bash
python run_chatbot_with_index.py --model_name jagadeesh/vicuna-13b
```

Copy the public url from terminal, paste it in browser and start testing
