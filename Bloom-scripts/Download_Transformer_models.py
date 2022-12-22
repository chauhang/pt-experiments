import transformers
import os
import json
import torch
from transformers import BloomTokenizerFast, BloomForSequenceClassification, BloomConfig, BloomForTokenClassification
from transformers import set_seed
""" This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
"""
print('Transformers version',transformers.__version__)
set_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transformers_model_dowloader(mode,pretrained_model_name,num_labels,do_lower_case):
    print("Download model and tokenizer", pretrained_model_name)
    if mode == "sequence_classification":
        config = BloomConfig.from_pretrained(pretrained_model_name,num_labels=num_labels,torchscript=False)
        model = BloomForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
        tokenizer = BloomTokenizerFast.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
    elif mode == "token_classification":
        config = BloomConfig.from_pretrained(pretrained_model_name,num_labels=num_labels,torchscript=False)
        model = BloomForTokenClassification.from_pretrained(pretrained_model_name, config=config)
        tokenizer = BloomTokenizerFast.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
    else:
        raise Exception("Unknown mode: {}. Supported modes are sequence_classification and token_classification")


    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

    print("Save model and tokenizer/ Torchscript model based on the setting from setup_config", pretrained_model_name, 'in directory', NEW_DIR)
    model.save_pretrained(NEW_DIR)
    tokenizer.save_pretrained(NEW_DIR)
    return

if __name__== "__main__":
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'setup_config.json')
    f = open(filename)
    settings = json.load(f)
    mode = settings["mode"]
    model_name = settings["model_name"]
    num_labels = int(settings["num_labels"])
    do_lower_case = settings["do_lower_case"]
    save_mode = settings["save_mode"]
    transformers_model_dowloader(mode,model_name, num_labels,do_lower_case)
