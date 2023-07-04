import argparse
import logging
import os
import sys

#from chat_ui import launch_gradio_interface
from create_chatbot import read_prompt_from_path, create_chat_bot
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS

logging.basicConfig(
    filename="pytorch-chatbot-with-index.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

sys.path.append('../..')

from ui.chat_ui import launch_gradio_interface

def load_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path - {index_path} does not exists")
    # embeddings = HuggingFaceEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain with context demo")
    parser.add_argument(
        "--model_name", type=str, default="alpaca-7b"
    )
    parser.add_argument('--torchserve', action='store_true', help='Enable torchserve')
    parser.add_argument('--callback', action='store_true', help='Enable callback')

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

    index = load_index(index_path=args.index_path)

    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    llm_chain, memory, llm = create_chat_bot(
        ts_host=args.torchserve_host,
        ts_port=args.torchserve_port,
        ts_protocol=args.torchserve_protocol,
        model_name=args.model_name,
        prompt_template=prompt_template,
        index=index,
    )

    launch_gradio_interface(llm=llm, llm_chain=llm_chain, memory=memory, torchserve=args.torchserve, callback_flag=args.callback, index=index)
