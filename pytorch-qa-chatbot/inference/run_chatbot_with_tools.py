import logging

from create_chatbot import load_model, create_chat_bot
from langchain import PromptTemplate
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.utilities import WikipediaAPIWrapper
from lib.chat_ui import launch_gradio_interface

logging.basicConfig(
    filename="pytorch-chatbot-with-index.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


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
        template=prompt_template,
        input_variables=["question", "temperature", "top_p", "top_k", "max_new_tokens"],
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
