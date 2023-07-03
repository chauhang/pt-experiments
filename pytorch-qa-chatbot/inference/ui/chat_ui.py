import logging
from typing import Iterable
import time
from langchain.callbacks import AsyncIteratorCallbackHandler

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from infer_chatbot import run_query
import asyncio

logger = logging.getLogger(__name__)


stop_btn = None


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


def launch_gradio_interface(llm, llm_chain, memory, multiturn=False, index=False):
    CSS ="""
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    #text_box.gradio-container { background-color: transparent; }
    #send_button { background-color: #6ee7b7;}
    """

    seafoam = Seafoam()

    if "torchserve and with callback":
        callback = AsyncIteratorCallbackHandler()

    def user(user_message, history):
        return gr.update(value="", interactive=True), history + [[user_message, None]]

    def stop_gen():
        global stop_btn
        stop_btn = True
        return
    
    def mem_clear():
        logger.info("Clearing memory")
        memory.clear()

    async def bot(history, top_p, top_k, max_new_tokens):
        logger.info(f"Sending Query! {history[-1][0]} with top_p {top_p} top_k {top_k} and max_new_tokens {max_new_tokens}")
        bot_message = run_query(
            llm_chain=llm_chain,
            question=history[-1][0],
            memory=memory,
            multiturn=multiturn,
            index=index,
            top_p=top_p, 
            top_k=top_k, 
            max_new_tokens=max_new_tokens,
        )
        logger.info(f"Response: {bot_message}")
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    def torchserve_bot(history, top_p, top_k, max_new_tokens):
        global stop_btn
        logger.info(f"Sending Query! {history[-1][0]} with top_p {top_p} top_k {top_k} and max_new_tokens {max_new_tokens}")
        run_query(llm_chain, history[-1][0], memory, multiturn, index, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
        history[-1][1] = ""
        flag = stop_btn = False
        foo = ""
        for resp in llm.response:
            if stop_btn:
                break
            prediction = resp.prediction.decode("utf-8")
            foo += prediction
            print(prediction, flush=True, end="")
            if "### Response:" in foo:
                flag = True
                foo = ""
                continue
            if flag:
                history[-1][1] += prediction
                yield history

    async def torchserve_callback_bot(history, top_p, top_k, max_new_tokens):
        global stop_btn
        logger.info(f"Sending Query! {history[-1][0]} with top_p {top_p} top_k {top_k} and max_new_tokens {max_new_tokens}")
        run = asyncio.create_task(run_query(
            llm_chain=llm_chain,
            question=history[-1][0],
            memory=memory,
            callback=callback,
            multiturn=multiturn,
            index=index,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens))
        history[-1][1] = ""
        flag = stop_btn = False
        foo = ""
        async for token in callback.aiter():
            if stop_btn:
                return
            foo += token
            print(token, flush=True, end="")
            if "### Response:" in foo:
                flag = True
                foo = ""
                continue
            if flag:
                history[-1][1] += token
                yield history
        await run

    with gr.Blocks(css=CSS, theme=seafoam) as demo:
        chatbot = gr.Chatbot(label="PyTorch Bot", show_label=True, elem_id="chatbot")
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=8):
                msg = gr.Textbox(show_label=False, elem_id="text_box", max_lines=200)
            with gr.Column(scale=1):
                generate = gr.Button(value="Send", elem_id="send_button")
        with gr.Row().style(equal_height=True):
            top_p = gr.Slider(minimum=0, maximum=1, step=0.01,label="top_p", value=0.95, interactive=True)
            top_k = gr.Slider(minimum=1, maximum=100, step=1,label="top_k", value=40, interactive=True)
            max_new_tokens = gr.Textbox(show_label=True, label="Max New Tokens", value=512, interactive=True)
            stop = gr.Button("Stop Generation")
            clear = gr.Button("Clear")

        
        if "torchserve and without callback":
            res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                torchserve_bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )

            res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                torchserve_bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )

        elif "torchserve and with callback":
            res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                torchserve_callback_bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )

            res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                torchserve_callback_bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )

        elif "without torchserve"
            res = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )

            res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot, top_p, top_k, max_new_tokens], chatbot
            )




        res.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        stop.click(stop_gen, None, None, queue=False).then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        clear.click(mem_clear, None, chatbot, queue=False).then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    demo.queue().launch(server_name="0.0.0.0", ssl_verify=False, debug=True, share=True, show_error=True)
