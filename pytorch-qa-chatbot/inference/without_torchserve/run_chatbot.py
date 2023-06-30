import argparse
import logging

from chat_ui import launch_gradio_interface
from create_chatbot import load_model, read_prompt_from_path, create_chat_bot

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

    parser.add_argument("--prompt_path", type=str, default="only_question_prompts.json")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--prompt_name", type=str, default="ONLY_QUESTION_ADVANCED_PROMPT")
    parser.add_argument("--multiturn", type=bool, default=False)

    args = parser.parse_args()

    model = load_model(model_name=args.model_name)
    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    logging.info(f"Using Prompt: {prompt_template}")
    llm_chain, memory = create_chat_bot(
        model_name=args.model_name,
        model=model,
        prompt_template=prompt_template,
        max_tokens=args.max_tokens,
    )
    # result = run_query(llm_chain=llm_chain, question="How to save the model", memory=memory)

    launch_gradio_interface(chain=llm_chain, memory=memory, multiturn=args.multiturn)
