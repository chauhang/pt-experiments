import logging
import re
import sys
import time
import uuid
from typing import Callable
from typing import Union

import gradio as gr
import torch
from dotenv import load_dotenv
from huggingface_hub import list_models, ModelFilter
from langchain import LLMChain
from langchain.agents import AgentOutputParser
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.schema import Document
from langchain.tools import StructuredTool, tool
from langchain.vectorstores import FAISS
from llama_index import (
    LLMPredictor,
    PromptHelper,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    LangchainEmbedding,
)
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from pydantic import BaseModel, Field
from transformers import pipeline, TextStreamer, LlamaTokenizer, LlamaForCausalLM

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

retriever = None
tools = None

MAX_INPUT_SIZE = 4096
NUM_OUTPUT = 2048
MAX_CHUNK_OVERLAP = 20
prompt_helper = PromptHelper(MAX_INPUT_SIZE, NUM_OUTPUT, MAX_CHUNK_OVERLAP)

CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

#############################################################################
MODEL_ID = "jagadeesh/vicuna-13b"
## Insert you huggingface key here
HF_KEY = ""
tokenizer = LlamaTokenizer.from_pretrained(
    MODEL_ID, use_fast=False, use_auth_token=HF_KEY
)
streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)
model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory={0: "18GB", 1: "18GB", 2: "18GB", 3: "18GB", "cpu": "10GB"},
    torch_dtype=torch.float16,
    use_auth_token=HF_KEY,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    max_new_tokens=NUM_OUTPUT,
    device_map="auto",
)

hf_pipeline = HuggingFacePipeline(pipeline=pipe)
llm_predictor = LLMPredictor(llm=hf_pipeline)
#############################################################################
# api = ""

# tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# hf_hub.login(token=api)

# ## loading llama base model and configuring it with adapter

# base_model_name = 'decapoda-research/llama-7b-hf'

# base_model = LlamaForCausalLM.from_pretrained(
#             base_model_name,
#             torch_dtype=torch.float16,
#             device_map="auto",
#         )

# model = PeftModel.from_pretrained(
#             base_model,
#             'shrinath-suresh/alpaca-lora-all-7b-delta',
#             torch_dtype=torch.float16,
#             load_in_8bit=True
#         )

# streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)

# class CustomLLM(LLM):
#     model_name = "shrinath-suresh/alpaca-lora-all-7b-delta"
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         inputs = tokenizer([prompt], return_tensors="pt")

#         # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
#         generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=NUM_OUTPUT)
#         thread = Thread(target=model.generate, kwargs=generation_kwargs)
#         thread.start()
#         # response = model.generate(**inputs, streamer=streamer, max_new_tokens=NUM_OUTPUT)
#         # return response
#         return next(streamer)

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"name_of_model": self.model_name}

#     @property
#     def _llm_type(self) -> str:
#         return "custom"

# llm_predictor = LLMPredictor(llm=CustomLLM())
#############################################################################

hf_embed_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
embed_model = LangchainEmbedding(hf_embed_model)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model
)


def load_index():

    # load data
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
    )
    index = load_index_from_storage(storage_context, service_context=service_context)

    return index


def load_tools(index):

    # Insert your huggingface token here
    API_TOKEN = ""

    class get_query(BaseModel):
        query: str = Field(
            default=None,
            description="The name of the model and task for which you want the datasets.",
        )

    @tool
    def get_datasets(query: str) -> str:
        """Fetches dataset for a given model and task"""
        print("Log::::::::::::::::::::::::::::", query, type(query))
        pattern = r"model=(.*?)\s*(?:and\s*)?task=(.*?)$"
        match = re.search(pattern, query)
        model_name = match.group(1)
        task = match.group(2)
        print("Log::::::::::::::::::::::::::::", model_name, task)

        filter = ModelFilter(library="pytorch", model_name=model_name, task=task)

        list_of_models = list_models(filter=filter, token=API_TOKEN)
        if not list_of_models:
            return f"{model_name} Not Found!"
        filtered_model = [model for model in list_of_models if model.id == model_name]
        datasets = [
            string.split(":")[1] for string in filtered_model[0].tags if "dataset:" in string
        ]
        if not datasets:
            return f"No Datasets available for {model_name}"
        print("Log::::::::::::::::::::::::::::", ",".join(datasets))
        return ",".join(datasets)

    search_tools = [
        StructuredTool.from_function(
            func=get_datasets,
            name="Get Model Datasets",
            args_schema=get_query,
            return_direct=True,
            description="A search engine. Useful for when you need to answer questions about fetching datasets. For example the input should be like model=model and task=task. Do not use this tool with the same input/query",
        ),
    ]

    index_tools = [
        LlamaIndexTool.from_tool_config(
            IndexToolConfig(
                name="PyTorch Index",
                query_engine=index.as_query_engine(
                    similarity_top_k=3, response_mode="compact", service_context=service_context
                ),
                description="useful for answering questions related to pytorch. Do not use this tool for fetching dataset. Do not use this tool with the same input/query",
                return_direct=True,
            )
        ),
    ]

    return search_tools + index_tools


def get_tools(query):
    global tools, retriever

    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


class CustomPromptTemplate(StringPromptTemplate):

    # The template to use
    template: str
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        toolss = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in toolss])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in toolss])
        return self.template.format(**kwargs)


def chain_and_agent(index, prompt_with_history, output_parser):
    global retriever, tools

    tools = load_tools(index)
    print("TOOLS: ", tools)

    docs = [
        Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)
    ]

    vector_store = FAISS.from_documents(docs, hf_embed_model)

    retriever = vector_store.as_retriever()

    tools_emb = get_tools("")
    print("TOOLS Embedding: ", tools_emb)
    tool_names = [tool.name for tool in tools_emb]

    # llm = CustomLLM()
    llm = hf_pipeline

    # Insert your openai key here
    # OPENAI_KEY = ""
    # llm = OpenAI(streaming=True, temperature=0, callbacks=[StreamingStdOutCallbackHandler()], openai_api_key=OPENAI_KEY)

    llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation: "],
        allowed_tools=tool_names,
    )

    session_id = uuid.uuid4()

    message_history = RedisChatMessageHistory(str(session_id), "redis://localhost:6379/0", ttl=600)
    memory = ConversationBufferWindowMemory(
        k=2, memory_key="history", return_messages=True, chat_memory=message_history
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, streaming=True, tools=tools, verbose=True, memory=memory
    )
    return agent_chain


def run_query(agent_chain, query):
    response = agent_chain.run(input=query)
    return response


def test(query):
    print("::::::::::::::::::::::::::::::", query)
    return query


def main():

    index = load_index()

    # Set up the base template
    template_with_history = """Act like you are an expert PyTorch Engineer and provide answers to these questions from the developer community.
    If you don't know the answer say 'I am not sure about this, can you post the question on pytorch-discuss channel', don't make up an answer if you don't know.

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 1 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Previous conversation history:
    {history}

    New question: {input}
    {agent_scratchpad}"""

    prompt_with_history = CustomPromptTemplate(
        template=template_with_history,
        tools_getter=get_tools,
        input_variables=["input", "intermediate_steps", "history"],
    )

    output_parser = CustomOutputParser()

    agent_chain = chain_and_agent(index, prompt_with_history, output_parser)
    print("Agent chain complete!")
    #     resp = run_query(agent_chain, "what is pytorch")
    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        print("Sending Query!")
        bot_message = run_query(agent_chain, history[-1][0])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>", history[-1][0])
        bot_message = bot_message
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    with gr.Blocks(css=CSS, theme=gr.themes.Glass()) as demo:
        chatbot = gr.Chatbot(label="PyTorch Bot", show_label=True, elem_id="chatbot")
        msg = gr.Textbox(label="Enter Text", show_label=True)
        with gr.Row():
            generate = gr.Button("Generate")
            clear = gr.Button("Clear")

        # res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        #     bot, chatbot, chatbot
        # )

        res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(
        server_name="0.0.0.0", ssl_verify=False, debug=True, show_error=True, share=True
    )


if __name__ == "__main__":
    main()
