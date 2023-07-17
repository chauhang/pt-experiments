import argparse
import logging
import os
import uuid

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
    parser.add_argument("--torchserve", action="store_true", help="Enable torchserve")
    parser.add_argument("--callback", action="store_true", help="Enable callback")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--prompt_path", type=str, default="question_with_context_prompts.json")
    parser.add_argument(
        "--prompt_name", type=str, default="QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT"
    )
    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")
    parser.add_argument("--torchserve_host", type=str, default="localhost")
    parser.add_argument("--torchserve_port", type=str, default="80")
    parser.add_argument("--torchserve_protocol", type=str, default="gRPC")

    args = parser.parse_args()

    if not args.torchserve and args.callback:
        raise ValueError(f"Invalid Value - Cannot run callback when torchserve is False")

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
        inputs=[
            "chat_history",
            "question",
            "context",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
        ],
    )

    chain, memory, llm = create_chat_bot(
        model_name=args.model_name,
        model=model,
        prompt_template=prompt_template,
        ts_host=args.torchserve_host,
        ts_port=args.torchserve_port,
        ts_protocol=args.torchserve_protocol,
        session_id=uuid.uuid1(),
        torchserve=args.torchserve,
    )
    # result = run_query(llm_chain=llm_chain, index_path=args.index_path, question="How to save the model", memory=memory)

    launch_gradio_interface(
        llm=llm,
        chain=chain,
        index=index,
        memory=memory,
        torchserve=args.torchserve,
        protocol=args.torchserve_protocol,
        callback_flag=args.callback,
    )
