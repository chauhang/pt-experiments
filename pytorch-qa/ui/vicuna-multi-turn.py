import os
import torch
import uuid
from typing import Any, List, Mapping, Optional
from dotenv import load_dotenv
import langchain 
from threading import Thread
from tqdm import tqdm
from IPython.display import Markdown, display
from peft import PeftModel
from typing import List, Union, Callable
import re
import time
import gradio as gr
from argparse import ArgumentParser


from transformers import pipeline, TextStreamer, TextIteratorStreamer, LlamaTokenizer, LlamaForCausalLM

import llama_index
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.indices.composability import ComposableGraph
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, LlamaIndexTool, create_llama_agent
from llama_index import load_index_from_storage, download_loader, SummaryPrompt, LLMPredictor, GPTListIndex, GPTVectorStoreIndex, PromptHelper, StorageContext, ServiceContext, LangchainEmbedding, SimpleDirectoryReader
from llama_index.langchain_helpers.text_splitter import SentenceSplitter, TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser, NodeParser

from langchain.agents import Tool
from langchain.agents import Tool, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain import LLMChain
from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.callbacks import tracing_enabled
from langchain.agents import Agent, BaseSingleActionAgent, LLMSingleActionAgent, initialize_agent, StructuredChatAgent, ConversationalChatAgent, ConversationalAgent, AgentExecutor, ZeroShotAgent, AgentType
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.cache import RedisSemanticCache
from langchain.prompts import StringPromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import Document

import huggingface_hub as hf_hub

load_dotenv()

CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

retriever = None
tools = []

def loadLLM():
    # input Key here
    token = ""
    hf_hub.login(token=token)

    model_id = "jagadeesh/vicuna-13b"
    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
    streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, low_cpu_mem_usage=True, device_map="auto", max_memory={0:"18GB",1:"18GB",2:"18GB",3:"18GB","cpu":"10GB"}, torch_dtype=torch.float16)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, streamer=streamer, max_new_tokens=512, device_map="auto"
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe, verbose=True)
    llm_predictor = LLMPredictor(llm=hf_pipeline)
    return (hf_pipeline, llm_predictor)

def loadEmbeddedingModel():
    HF_Embed_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embed_model = LangchainEmbedding(HF_Embed_model)
    return (HF_Embed_model, embed_model)

def createServiceContext(llm_predictor, embed_model):
    return ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)


def loadIndex(index_dir, service_context):
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_dir),
    )
    return load_index_from_storage(storage_context, service_context=service_context)

def createTools(index, service_context):
    global tools
    tools = [
        LlamaIndexTool.from_tool_config(
            IndexToolConfig(
                name = "PyTorch Index",
                query_engine=index.as_query_engine(similarity_top_k=3, response_mode="simple_summarize", service_context=service_context),
                description=f"useful for answering questions related to pytorch.",
                tool_kwargs={"return_direct": True, "return_sources": True},
                return_sources=True
            )
        ),
    ]
    return tools

# test_engine=index.as_query_engine(similarity_top_k=3, response_mode="simple_summarize", service_context=service_context)
# test_engine.query("What are the different pytorch distributed techniques?")

def createToolEmbeddings(tools, hfEmbedModel):
    global retriever
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)]
    vector_store = FAISS.from_documents(docs, hfEmbedModel)
    retriever = vector_store.as_retriever()
    return retriever

def get_tools(query):
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
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


TEMPLATE_WITH_HISTORY = """You are an expert PyTorch Assistant and provide answers to questions from the developer community.

Given the context and the conversation history, try to answer the question. Use only the information provided in the context. Do not use any external knowledge beyond the given conversation.
If you think you don't know the answer say "I am not sure about this, can you post the question on pytorch-discuss channel", don't make up an answer if you don't know.

Use the following format:

Question: the input question you must answer
Thought: use {tool_names} tool
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 1 times)
Final Answer: the final answer to the original input question

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}
"""

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
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

def createAgent(toolsEmbeddings, prompt, llm, outputParser):
    tool_names = [tool.name for tool in toolsEmbeddings]
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return LLMSingleActionAgent(llm_chain=llm_chain, output_parser=outputParser, stop=["</s><s>"], allowed_tools=tool_names)

def createAgentChain(agent, toolsEmbeddings):
    session_id = uuid.uuid4()

    message_history = RedisChatMessageHistory(str(session_id), 'redis://localhost:6379/0', ttl=600)
    memory = ConversationBufferWindowMemory(k=2, memory_key="history", return_messages=True, chat_memory=message_history)

    return AgentExecutor.from_agent_and_tools(agent=agent, streaming=True, tools=toolsEmbeddings, verbose=True, memory=memory)

def runQuery(agentChain, query):
    return agentChain.run(input=query)

def startUI(agentChain):
    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        print("Sending Query!")
        bot_message = runQuery(agentChain, history[-1][0])
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

    demo.queue().launch(server_name="0.0.0.0", ssl_verify=False, debug=True, share=True, show_error=True)

def main(args):
    # langchain.llm_cache = RedisSemanticCache(
    #     redis_url="redis://localhost:6379",
    #     embedding=HuggingFaceEmbeddings()
    # )
    langchain.llm_cache = None

    print(":::::::::::::::::::::::::::::::::::::Loading LLM")
    hfPipeline, llmPredictor = loadLLM()
    print(":::::::::::::::::::::::::::::::::::::Loading Embed Model")
    hfEmbedModel, embedModel = loadEmbeddedingModel()
    print(":::::::::::::::::::::::::::::::::::::Create Service Context")
    serviceContext = createServiceContext(llmPredictor, embedModel)
    print(":::::::::::::::::::::::::::::::::::::Load Index")
    index = loadIndex(args.index_dir, serviceContext)
    print(":::::::::::::::::::::::::::::::::::::Create Tools")
    tools = createTools(index, serviceContext)
    print("||||||||||||||||||||||||||||||||||||| TOOLS:", tools)
    print(":::::::::::::::::::::::::::::::::::::Create Tool Embeddings")
    retriever = createToolEmbeddings(tools, hfEmbedModel)
    print(":::::::::::::::::::::::::::::::::::::Get Tools")
    toolsEmbeddings = get_tools("")
    print("||||||||||||||||||||||||||||||||||||| TOOL Embeddings", toolsEmbeddings)
    print(":::::::::::::::::::::::::::::::::::::Create Output Parser")
    outputParser = CustomOutputParser()
    print(":::::::::::::::::::::::::::::::::::::Create Prompt Template")
    promptTemplate = CustomPromptTemplate(
        template=TEMPLATE_WITH_HISTORY,
        tools_getter=get_tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "history"]
    )
    print(":::::::::::::::::::::::::::::::::::::Create Agent")
    agent = createAgent(toolsEmbeddings, promptTemplate, hfPipeline, outputParser)
    print(":::::::::::::::::::::::::::::::::::::Create Agent Chain")
    agentChain = createAgentChain(agent, toolsEmbeddings)
    print(":::::::::::::::::::::::::::::::::::::Start UI")
    startUI(agentChain)

if __name__ == "__main__":

    parser = ArgumentParser("PyTorch Chatbot")
    parser.add_argument(
        "--index_dir", default="~/pytorch_docs_512", help="Path to document index"
    )

    args = parser.parse_args()
    main(args)