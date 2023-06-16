# Loading vicuna-13b with PiPPy (PyTorch Native Large inference solution)


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
## Prepare context

```bash
aws s3 cp s3://kubeflow-dataset/pytorch-qa/docs_blogs_faiss_index.zip .

unzip docs_blogs_faiss_index.zip
```

## Standalone langchain demo on finetune model with and without index

```bash
# Model     : shrinath-suresh/alpaca-lora-7b-answer-summary
# Context   : yes
python langchain_demo_finetuned.py --prompt_name QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT --prompt_type question_with_context --index_path docs_blogs_faiss_index

# Model     : shrinath-suresh/alpaca-lora-7b-answer-summary
# Context   : no
python langchain_demo_finetuned.py --prompt_name ONLY_QUESTION_ADVANCED_PROMPT --prompt_type only_question
```

Note: Check `prompts.json` for other prompts

## Vicuna-13b with Torchserve without index

### Start Torchserve

1. Start Torchserve as described in the previous steps
2. Bring up demo UI using below command

```bash
python langchain_demo_vicuna_torchserve_grpc.py
```
## Finetuned with Torchserve with and without index

### Start Torchserve

1. Start Torchserve as described in the previous steps
2. Bring up demo UI using below command

```bash
# Model     : shrinath-suresh/alpaca-lora-7b-answer-summary
# Context   : yes
# Torchserve: yes
python langchain_demo_finetuned_tochserve_grpc.py --model_name <Model name given at archive generation> --prompt_name QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT --prompt_type question_with_context --index_path docs_blogs_faiss_index

# Model     : shrinath-suresh/alpaca-lora-7b-answer-summary
# Context   : no
# Torchserve: yes
python langchain_demo_finetuned_tochserve_grpc.py --model_name <Model name given at archive generation> --prompt_name ONLY_QUESTION_ADVANCED_PROMPT --prompt_type only_question
```