import logging
import time
from abc import ABC

import packaging.version
import requests
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )

class ModelHandler(BaseHandler, ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        if torch.cuda.is_available():
            self.map_location = "cuda"
            self.device = torch.device(self.map_location)
        else:
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        dtype_str = ctx.model_yaml_config["handler"]["dtype"]
        torch.manual_seed(seed)

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

        dtype = dtypes.get(dtype_str, torch.float32)

        self.model = LlamaForCausalLM.from_pretrained(
            model_path, 
            use_cache=False, 
            max_memory={i: ctx.model_yaml_config["handler"]["max_gpu_memory"] for i in range(ctx.model_yaml_config["handler"]["num_gpus"])},
            low_cpu_mem_usage=ctx.model_yaml_config["handler"]["low_cpu_mem_usage"],
            device_map=ctx.model_yaml_config["handler"]["device_map"],
            torch_dtype=dtype
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, return_tensors="pt")

        self.max_length = ctx.model_yaml_config["handler"]["max_length"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            logger.info("Received text: %s", input_text)

            inputs = self.tokenizer(
                [input_text],
                max_length=self.max_length,
                return_tensors="pt",
            )
        return inputs

    def inference(self, inputs):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        inferences = []
        inputs = inputs.to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
        )
        inferences.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        logger.info("Generated text: %s", inferences)

        print("Generated text", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
