### Benchmark

The goal is to conduct a benchmark by loading the embedding model on one GPU and the LLM model on the other GPU.

## Register Embedding Model

Create a `model_store` directory if not present

```bash
mkdir model_store
```

Ensure the following files are present in the directory

1. embedding_handler.py - torchserve custom handler to load and generate embeddings
2. model-config.yaml - embedding model parameters
3. config.properties - torchserve config properties

Run the following command to generate the mar file

```bash
torch-model-archiver --model-name embedding --version 1.0 --handler embedding_handler.py --config-file model-config.yaml --export-path model_store/ -f
```

Start the torchserve and wait for the embedding model to register

```bash
torchserve --ncs --start --model-store model_store --models embedding.mar
```

One the embedding model is registered, run the following command to check if the embedding model works

```
curl -v "http://localhost:8080/predictions/embedding" -F "data=This is a test sentence"
```

The response will consist of a list of outputs with a dimension of 1x768.
