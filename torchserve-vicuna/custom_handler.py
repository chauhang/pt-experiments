import logging
import time
from abc import ABC

import packaging.version
import requests
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler
from ts.handler_utils.distributed.pt_pippy import initialize_rpc_workers, get_pipeline_driver

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )
class ModelHandler(BasePippyHandler, ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        super().initialize(ctx)
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = self.local_rank %  torch.cuda.device_count()

        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        dtype_str = ctx.model_yaml_config["handler"]["dtype"]
        torch.manual_seed(seed)

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

        dtype = dtypes.get(dtype_str, torch.float32)

        skip_init_start = time.perf_counter()
        with torch.device("meta"):
            self.model = LlamaForCausalLM.from_pretrained(
                model_path, use_cache=False, torch_dtype=dtype
            )
        skip_init_end = time.perf_counter()
        
        logger.info(
            f" init model time on meta device took {skip_init_end - skip_init_start} seconds"
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, return_tensors="pt")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = ctx.model_yaml_config["handler"]["max_length"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]

        logger.info("Instantiating model Pipeline")
        pippy_compile_time_start = time.perf_counter()
        self.model = get_pipeline_driver(self.model, self.world_size, ctx)
        pippy_compile_time_end = time.perf_counter()

        logger.info(
            f" pippy compile time took {pippy_compile_time_end- pippy_compile_time_start} seconds on rank {self.local_rank}"
        )

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
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        (input_ids_batch, _) = input_batch
        inferences = []
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.model.generate(
            input_ids_batch,
            do_sample=True,
            max_new_tokens=int(self.setup_config["max_length"]),
            top_p=0.95,
            top_k=60,
        )
        for i, _ in enumerate(outputs):
            inferences.append(
                self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            )

        logger.info("Generated text: '%s'", inferences)

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
