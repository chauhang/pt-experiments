import argparse
import logging

from chat_ui import launch_gradio_interface
from create_chatbot import load_model, read_prompt_from_path, create_chat_bot
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper


logging.basicConfig(
    filename="pytorch-chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


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


def init_agent(llm):
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1,
        return_intermediate_steps=True,
    )

    agent_chain.agent.llm_chain.prompt.template = """Answer the following questions as best you can. You have access to the following tools:

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
    Final Answer: the final answer to the original input question, stop after this

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
    return agent_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument("--model_name", type=str, default="shrinath-suresh/vicuna-13b")

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

    wikipedia = WikipediaAPIWrapper()

    tools = create_tools(chain=llm_chain, wikipedia=wikipedia)

    agent_chain = init_agent(llm=llm_chain.llm)

    # from infer_chatbot import run_query
    # result = run_query(chain=agent_chain, question="How to save the model", memory=memory, multiturn=False)
    # print(result)
    launch_gradio_interface(chain=agent_chain, memory=memory, multiturn=args.multiturn)
