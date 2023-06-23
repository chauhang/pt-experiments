import json
import logging
import os

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory

from torchserve_endpoint import TorchServeEndpoint
logger = logging.getLogger(__name__)



def read_prompt_from_path(prompt_path):
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"{prompt_path} not found")
    logger.info(f"Reading prompts from path: {prompt_path}")
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict


def create_chat_bot(ts_host, ts_port, model_name, prompt_template, index=None):

    llm = TorchServeEndpoint(host=ts_host, port=ts_port, model_name=model_name, verbose=True)

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", input_key="question")

    if index:
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question", "context", "top_p", "top_k", "max_new_tokens"]
        )
    else:
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question", "top_p", "top_k", "max_new_tokens"]
        )
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")
    return llm_chain, memory

