# This example covers steps on using Torchdynamo for model optimization. This offer reduction in inference latency compared with base model. Here we use BERT model in Sequence classification mode with Torchserve.

### Step 1: Download model

Run [Download_Transformer_models.py](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Download_Transformer_models.py) script to download model.


```bash
python Download_Transformer_models.py
```

The script downloads model to `Transformer_model` directory.

### Step2: Modify setup config

In the setup_config.json :

backend : ofi , onnxrt or other available inference backends.

**__Refer__**: https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers#setting-the-setup_configjson

### Step 3: Generate MAR file

```bash
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./custom_handler.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json" -r requirements.txt
```

### Step 4: Start torchserve

```bash
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ncs
```

### Step 5: Run inference

```bash
curl -v "curl -X POST http://127.0.0.1:8080/predictions/my_tc -T " -T sample_text.txt
```

**__Note__** Refer example model optimization with torch dynamo here: `torchdynamo_inference_example.py`