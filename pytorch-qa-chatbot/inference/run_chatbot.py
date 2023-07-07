import argparse
import logging

from lib.chat_ui import launch_gradio_interface
from lib.create_chatbot import (
    load_model,
    read_prompt_from_path,
    create_chat_bot,
    create_prompt_template,
)

logging.basicConfig(
    filename="pytorch-chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )

    parser.add_argument('--torchserve', action='store_true', help='Enable torchserve')
    parser.add_argument('--callback', action='store_true', help='Enable callback')
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--prompt_path", type=str, default="only_question_prompts.json")
    parser.add_argument("--prompt_name", type=str, default="ONLY_QUESTION_ADVANCED_PROMPT")
    parser.add_argument("--multiturn", type=bool, default=False)
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

    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)

    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )

    prompt_str = prompt_dict[args.prompt_name]
    logging.info(f"Using Prompt: {prompt_str}")

    prompt_template = create_prompt_template(
        prompt_str=prompt_str,
        inputs=["chat_history", "question", "top_p", "top_k", "max_new_tokens"],
    )


    llm_chain, memory, llm = create_chat_bot(
        model_name=args.model_name,
        model=model,
        prompt_template=prompt_template,
        ts_host=args.torchserve_host,
        ts_port=args.torchserve_port,
        ts_protocol=args.torchserve_protocol,
        max_tokens=args.max_tokens,
        torchserve=args.torchserve,
    )
    launch_gradio_interface(llm=llm, chain=llm_chain, memory=memory, torchserve=args.torchserve, protocol=args.torchserve_protocol, callback_flag=args.callback, multiturn=args.multiturn)