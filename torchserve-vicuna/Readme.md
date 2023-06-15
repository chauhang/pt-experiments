# Loading vicuna-13b with PiPPy (PyTorch Native Large inference solution)


## Download the model

```bash
```
## Packaging the model

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler custom_handler.py --extra-files  -r requirements.txt --config-file model-config.yaml --archive-format tgz
```

## Inference

```bash
curl -v "http://localhost:8080/predictions/vicuna-13b" -F "data=what is pytorch ?"
```