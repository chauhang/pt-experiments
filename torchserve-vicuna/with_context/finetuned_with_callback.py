import argparse
import json
import os
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
from typing import Iterable
from torchserve_endpoint import TorchServeEndpoint
from langchain.callbacks import AsyncIteratorCallbackHandler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_btn = None
callback_handler = AsyncIteratorCallbackHandler()

def read_prompt_from_path(prompt_path="prompts.json"):
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict

def setup(model_name, prompt_template):
    global llm
    llm = TorchServeEndpoint(host="localhost", port="7070",model_name=model_name, verbose=True)

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", input_key="question")
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["chat_history", "question", "context"]
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")
    return llm_chain, memory

def load_index(index_path):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    if not os.path.exists(index_path):
        raise ValueError(f"Index path - {index_path} does not exists")
    faiss_index = FAISS.load_local(index_path, embeddings)
    return faiss_index

async def run_query(llm_chain, question, index):
    context = index.similarity_search(question, k=2)
    await llm_chain.arun({"question": question, "context": context}, callbacks=[callback_handler])

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

def launch_gradio_interface(llm_chain, index, memory):
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

    def stop_gen():
        global stop_btn
        stop_btn = True

    async def bot(history):
        global stop_btn
        print("Sending Query!", history[-1][0])
        run = asyncio.create_task(run_query(llm_chain, history[-1][0], index))
        history[-1][1] = ""
        flag = stop_btn = False
        foo = ""
        async for token in callback_handler.aiter():
            if stop_btn:
                break 
            foo += token
            if "### Response:" in foo:
                flag = True
                foo = ""
                continue 
            if flag:  
                history[-1][1] += token
                yield history
        await run        

    def mem_clear():
        logging.info(f"Clearing Memory")
        memory.clear()

    with gr.Blocks(css=CSS, theme=seafoam) as demo:
        chatbot = gr.Chatbot(label="PyTorch Bot", show_label=True, elem_id="chatbot")
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=8):
                msg = gr.Textbox(show_label=False, elem_id="text_box")
            with gr.Column(scale=1):
                generate = gr.Button(value="Send", elem_id="send_button")
        stop = gr.Button("Stop Generation")
        clear = gr.Button("Clear")
        stop.click(stop_gen, None, None, queue=False)

        res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        clear.click(mem_clear, None, chatbot, queue=False)

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
        prompt_template=prompt_template,
    )
    index = load_index(args.index_path)
    
    launch_gradio_interface(
        llm_chain=llm_chain, index=index, memory=memory
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument("--prompt_name", type=str, default="QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT")

    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")

    args = parser.parse_args()
    main(args)
