import time
import argparse
import json
import os
import re
import time
import asyncio

from typing import Any
import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferWindowMemory
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
from typing import Iterable
from torchserve_endpoint import TorchServeEndpoint
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.callbacks import AsyncIteratorCallbackHandler
# import logging

# logging.basicConfig(filename="logfile.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

callback_handler = AsyncIteratorCallbackHandler()

def read_prompt_from_path(prompt_path="prompts.json"):
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict

def setup(model_name, prompt_type, prompt_template):
    global llm
    llm = TorchServeEndpoint(host="localhost", port="7070",model_name=model_name, verbose=True)

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", input_key="question")
    if prompt_type == "question_with_context":
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question", "context"]
        )
    else:
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question"]
        )
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")
    return llm_chain, memory

def load_index(index_path):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    if not os.path.exists(index_path):
        raise ValueError(f"Index path - {index_path} does not exists")
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index

async def run_query(prompt_type, llm_chain, question, index, memory):
    if prompt_type == "question_with_context":
        context = index.similarity_search(question, k=2)
        await llm_chain.arun({"question": question, "context": context}, callbacks=[callback_handler])
    else:
        await llm_chain.arun(question)
    memory.clear()

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ), font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

def launch_gradio_interface(prompt_type, llm_chain, index, memory):
    CSS ="""
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    #text_box.gradio-container { background-color: transparent; }
    #send_button { background-color: #6ee7b7; margin-top: 2.5%}
    """

    seafoam = Seafoam()
    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    async def bot(history):
        print("Sending Query!", history[-1][0])
        run = asyncio.create_task(run_query(prompt_type, llm_chain, history[-1][0], index, memory))
        history[-1][1] = ""
        foo = ""
        isResponse = False
        async for token in callback_handler.aiter():
            foo += token
            if "### Response:" in foo:
                history[-1][1] += foo.split("### Response:\n ")[1]
                isResponse = True
                foo = ""
                continue
            if isResponse:
                history[-1][1] += token
                yield history
        await run        

    with gr.Blocks(css=CSS, theme=seafoam) as demo:
        chatbot = gr.Chatbot(label="PyTorch Bot", show_label=True, elem_id="chatbot")
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=8):
                msg = gr.Textbox(show_label=False, elem_id="text_box")
            with gr.Column(scale=1):
                generate = gr.Button(value="Send", elem_id="send_button")
        clear = gr.Button("Clear")

        res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(server_name="0.0.0.0", ssl_verify=False, debug=True, share=True, show_error=True)


def main(args):
    prompt_dict = read_prompt_from_path()
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    llm_chain, memory = setup(
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        prompt_template=prompt_template,
    )
    index = load_index(args.index_path)
    #result = run_query(prompt_type=args.prompt_type, llm_chain=llm_chain, index_path=args.index_path, question="How to save the model", memory=memory)
    
    launch_gradio_interface(
        prompt_type=args.prompt_type, llm_chain=llm_chain, index=index, memory=memory
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument("--prompt_type", type=str, default="only_question")
    parser.add_argument("--prompt_name", type=str, default="ONLY_QUESTION_ADVANCED_PROMPT")

    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")

    args = parser.parse_args()
    print("Contexxt : ", args.prompt_type)
    main(args)
