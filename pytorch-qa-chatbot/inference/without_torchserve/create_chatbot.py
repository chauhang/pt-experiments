import json
import logging
import os

import huggingface_hub as hf_hub
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer

logger = logging.getLogger(__name__)


def load_model(model_name):
    logger.info(f"Loading model: {model_name}")
    if not os.environ["HUGGINGFACE_KEY"]:
        raise EnvironmentError(
            "HUGGINGFACE_KEY - key missing. set in the environment before running the script"
        )
    hf_hub.login(token=os.environ["HUGGINGFACE_KEY"])
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
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


def create_chat_bot(model_name, model, prompt, max_tokens, index=None, enable_memory=True):

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)

    class CustomLLM(LLM):
        def _call(self, prompt, stop=None) -> str:
            inputs = tokenizer([prompt], return_tensors="pt")

            response = model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens)
            response = tokenizer.decode(response[0])
            return response

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
        llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")
        return llm_chain, memory
    else:
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain, llm
