import argparse
import logging
import os
import sys

from lib.create_chatbot import (
    load_model,
    read_prompt_from_path,
    create_chat_bot,
    create_prompt_template,
)
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

logging.basicConfig(
    filename="pytorch-chatbot-with-index.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# sys.path.append("..")
from lib.chat_ui import launch_gradio_interface


def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path - {index_path} does not exists")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain with context demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument('--torchserve', action='store_true', help='Enable torchserve')
    parser.add_argument('--callback', action='store_true', help='Enable callback')
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--prompt_path", type=str, default="question_with_context_prompts.json")
    parser.add_argument(
        "--prompt_name", type=str, default="QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT"
    )
    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")
    parser.add_argument("--torchserve_host", type=str, default="localhost")
    parser.add_argument("--torchserve_port", type=str, default="7070")
    parser.add_argument("--torchserve_protocol", type=str, default="gRPC")

    args = parser.parse_args()

    if not args.torchserve and args.callback:
        raise ValueError(
            f"Invalid Value - Cannot run callback when torchserve is False"
        )
    
    model = None
    if not args.torchserve:
        model = load_model(model_name=args.model_name)
    index = load_index(index_path=args.index_path)

    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    prompt_str = prompt_dict[args.prompt_name]

    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    
    prompt_template = create_prompt_template(
        prompt_str=prompt_str,
        inputs=["chat_history", "question", "context", "top_p", "top_k", "max_new_tokens"],
    )

    chain, memory, llm = create_chat_bot(
        model_name=args.model_name,
        model=model,
        prompt_template=prompt_template,
        ts_host=args.torchserve_host,
        ts_port=args.torchserve_port,
        ts_protocol=args.torchserve_protocol,
        max_tokens=args.max_tokens,
        torchserve=args.torchserve,
        index=index,
    )
    # result = run_query(llm_chain=llm_chain, index_path=args.index_path, question="How to save the model", memory=memory)

    launch_gradio_interface(llm=llm, chain=chain, index=index, memory=memory, torchserve=args.torchserve, protocol=args.torchserve_ptotocol, callback_flag=args.callback)
(base) ubuntu@ip-172-31-5-82:~/finetune_testing/common_ui/pt-experiments/pt-experiments/pytorch-qa-chatbot/inference$ cat run_chatbot_with_tools.py 
import sys
from create_chatbot import load_model, create_chat_bot
from langchain import PromptTemplate
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.utilities import WikipediaAPIWrapper

from lib.chat_ui import launch_gradio_interface


def init_agent(tools, llm):
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1,
        return_intermediate_steps=True,
    )
    return agent_chain


def create_tools(chain, wikipedia):
    tool_list = [
        Tool(
            name="pytorch search",
            func=chain.run,
            description="Use this to answer questions only related to pytorch",
            return_direct=True,
        ),
        Tool(
            name="wikipedia search",
            func=wikipedia.run,
            description="Use this to search wikipedia for general questions which is not related to pytorch",
            return_direct=True,
        ),
    ]
    return tool_list


def create_llm_chain():

    model_name = "shrinath-suresh/vicuna-13b"
    max_tokens = 2048

    model = load_model(model_name)

    prompt_template = (
        "Below is an instruction that describes a task. "
        "If question is related to pytorch Write a response "
        "that appropriately completes the request."
        "\n\n### Instruction:\n{question}\n\n### Response:\n"
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["question", "top_p", "top_k", "max_new_tokens"]
    )

    llm_chain, llm = create_chat_bot(
        model_name=model_name,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        enable_memory=False,
    )
    return llm_chain


def create_agent_chain(chain):
    ## tool 1
    wikipedia = WikipediaAPIWrapper()

    tools = create_tools(chain=chain, wikipedia=wikipedia)

    a_chain = init_agent(tools, chain.llm)

    template = """Answer the following questions as best you can. You have access to the following tools:

     pytorch search: Use this to answer questions only related to pytorch
     wikipedia search: Use this to search wikipedia for general questions which is not related to pytorch

     Use the following format:

     Question: the input question you must answer
     Thought: you should always think about what to do
     Action: the action to take, should be one of [pytorch search, wikipedia search]
     Action Input: the input to the action
     Observation: the result of the action
     ... (this Thought/Action/Action Input/Observation cannot repeat)
     Thought: I now know the final answer
     Final Answer: the final answer to the original input question, stop after this.

     Begin!

     Question: {input}
     Thought:{agent_scratchpad}"""

    a_chain.agent.llm_chain.prompt.template = template
    return a_chain


# def run_query(chain):
#     answer = chain('how do i check if pytorch is using gpu?')
#
#     print(answer['intermediate_steps'][0][0].log.split('Final Answer: ')[-1])
#


if __name__ == "__main__":
    llm_chain = create_llm_chain()
    agent_chain = create_agent_chain(chain=llm_chain)
    launch_gradio_interface(chain=agent_chain, memory=None)