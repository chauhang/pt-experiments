import json
import logging
import os
from typing import Any

import huggingface_hub as hf_hub
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.base import LLM
import torch
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from lib.torchserve_endpoint import TorchServeEndpoint
from transformers import TextIteratorStreamer
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)


def load_model(model_name):
    logger.info(f"Loading model: {model_name}")
    if not os.environ["HUGGINGFACE_KEY"]:
        raise EnvironmentError(
            "HUGGINGFACE_KEY - key missing. set in the environment before running the script"
        )
    hf_hub.login(token=os.environ["HUGGINGFACE_KEY"])
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=True
    )
    return model


def read_prompt_from_path(prompt_path):
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"{prompt_path} not found")
    logger.info(f"Reading prompts from path: {prompt_path}")
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict


def create_prompt_template(prompt_str, inputs):
    prompt = PromptTemplate(template=prompt_str, input_variables=inputs)
    return prompt


def create_chat_bot(
    model_name,
    prompt_template,
    model=None,
    ts_host=None,
    ts_port=None,
    ts_protocol=None,
    torchserve=False,
    max_tokens=None,
    enable_memory=True,
):
    llm = memory = None
    if torchserve:
        llm = TorchServeEndpoint(
            host=ts_host, port=ts_port, protocol=ts_protocol, model_name=model_name, verbose=True
        )
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)

        class CustomLLM(LLM):

            """Streamer Object"""

            streamer: Optional[TextIteratorStreamer] = None

            def _call(self, prompt, stop=None) -> str:
                self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
                splitted_prompt, params = prompt.split("||")
                inputs = tokenizer([splitted_prompt], return_tensors="pt")
                params_dict = json.loads(params)
                inputs = inputs.to("cuda")
                generation_kwargs = dict(inputs, streamer=self.streamer, **params_dict)
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                generated_text = ""
                return generated_text

            @property
            def _identifying_params(self):
                return {"name_of_model": model_name}

            @property
            def _llm_type(self) -> str:
                return "custom"

        llm = CustomLLM()

    if enable_memory:
        memory = ConversationBufferWindowMemory(
            k=3, memory_key="chat_history", input_key="question"
        )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm, memory=memory, output_key="result")
    return llm_chain, memory, llm
