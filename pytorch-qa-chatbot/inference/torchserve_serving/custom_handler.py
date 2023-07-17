import logging
from abc import ABC
import json

import packaging.version
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, TextIteratorStreamer
from threading import Thread

from ts.torch_handler.base_handler import BaseHandler
from ts.protocol.otf_message_handler import send_intermediate_predict_response

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
        self.context = ctx
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
            max_memory={
                i: ctx.model_yaml_config["handler"]["max_gpu_memory"]
                for i in range(ctx.model_yaml_config["handler"]["num_gpus"])
            },
            low_cpu_mem_usage=ctx.model_yaml_config["handler"]["low_cpu_mem_usage"],
            device_map=ctx.model_yaml_config["handler"]["device_map"],
            torch_dtype=dtype,
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path, use_fast=False, return_tensors="pt"
        )
        self.streamer = TextIteratorStreamer(self.tokenizer)

        self.top_p = ctx.model_yaml_config["handler"]["top_p"]
        self.top_k = ctx.model_yaml_config["handler"]["top_k"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]
        self.temperature = ctx.model_yaml_config["handler"]["temperature"]

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
            inputs = input_text.split("||")
            prompt = inputs[0]
            self.model_kwargs = (
                json.loads(inputs[1])
                if len(inputs) > 1
                else {
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature
                }
            )

            inputs = self.tokenizer(
                [prompt],
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

        inputs = inputs.to(self.device)
        logger.info("Model kwargs for genrate %s", self.model_kwargs)
        generation_kwargs = dict(
            inputs, streamer=self.streamer, do_sample=True, **self.model_kwargs
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.streamer:
            send_intermediate_predict_response(
                [new_text],
                self.context.request_ids,
                "Intermediate Prediction success",
                200,
                self.context,
            )
        return ["</s><s>"]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
