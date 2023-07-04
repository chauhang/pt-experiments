import logging
import time
from typing import Iterable

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from infer_chatbot import run_query

logger = logging.getLogger(__name__)


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color or str = colors.emerald,
        secondary_hue: colors.Color or str = colors.blue,
        neutral_hue: colors.Color or str = colors.gray,
        spacing_size: sizes.Size or str = sizes.spacing_md,
        radius_size: sizes.Size or str = sizes.radius_md,
        text_size: sizes.Size or str = sizes.text_lg,
        font: fonts.Font
        or str
        or Iterable[fonts.Font or str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        or str
        or Iterable[fonts.Font or str] = (
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


def launch_gradio_interface(chain, memory, multiturn=False, index=False):
    CSS = """
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    #text_box.gradio-container { background-color: transparent; }
    #send_button { background-color: #6ee7b7; margin-top: 2.5%}
    """

    seafoam = Seafoam()

    def user(user_message, history):
        return gr.update(value="", interactive=True), history + [[user_message, None]]

    async def bot(history):
        logger.info(f"Sending Query: {history[-1][0]}")
        bot_message = run_query(
            chain=chain,
            question=history[-1][0],
            memory=memory,
            multiturn=multiturn,
            index=index,
        )
        logger.info(f"Response: {bot_message}")
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    def mem_clear():
        logger.info("Clearing memory")
        memory.clear()

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
        clear.click(mem_clear, None, chatbot, queue=False)

    demo.queue().launch(
        server_name="0.0.0.0", ssl_verify=False, debug=True, share=True, show_error=True
    )
