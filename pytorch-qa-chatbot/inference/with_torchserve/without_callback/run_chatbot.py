import argparse
import logging

from chat_ui import launch_gradio_interface
from create_chatbot import read_prompt_from_path, create_chat_bot

logging.basicConfig(
    filename="pytorch-chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


sys.path.append('../..')

from ui.chat_ui import launch_gradio_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument(
        "--model_name", type=str, default="alpaca-7b"
    )

    parser.add_argument("--prompt_path", type=str, default="only_question_prompts.json")
    parser.add_argument("--prompt_name", type=str, default="ONLY_QUESTION_ADVANCED_PROMPT")
    parser.add_argument("--multiturn", type=bool, default=False)
    parser.add_argument("--torchserve_host", type=str, default="localhost")
    parser.add_argument("--torchserve_port", type=str, default="7070")

    args = parser.parse_args()

    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    logging.info(f"Using Prompt: {prompt_template}")
    llm_chain, memory, llm = create_chat_bot(
        ts_host=args.torchserve_host,
        ts_port=args.torchserve_port,
        model_name=args.model_name,
        prompt_template=prompt_template,
    )

    launch_gradio_interface(llm=llm, llm_chain=llm_chain, memory=memory, multiturn=args.multiturn, index=None)
