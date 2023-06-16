import re
import time

import gradio as gr
import torch
import requests
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time

import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc

GRPC_URL="localhost:7070"
URL = "http://localhost:8080/predictions/vicuna-13b"
MODEL_NAME = "vicuna-13b"
responses = None

def get_inference_stub():
    channel = grpc.insecure_channel("")
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

def infer_stream(stub, model_name, model_input):
    global responses
    input_data = {"data": bytes(model_input, 'utf-8')}

    responses = stub.StreamPredictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )
    return responses

class CustomLLM(LLM):
    model_name = "custom_model"

    def _call(self, prompt, stop=None) -> str:        
        responses = infer_stream(get_inference_stub(), MODEL_NAME, prompt)
        return ""

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


llm = CustomLLM()

memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

template = "{chat_history} Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")


def run_query(question):
    global responses
    llm_chain.run(question)

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

def launch_gradio_interface():
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

    def bot(history):
        global responses
        print("Sending Query!", history[-1][0])
        run_query(history[-1][0])
        history[-1][1] = ""
        try:
            flag = False
            for resp in responses:
                prediction = resp.prediction.decode("utf-8")
                print(prediction, flush=True, end="")
                if "### Response:" in prediction:
                    flag = True
                    continue 
                if flag:  
                    history[-1][1] += prediction
                    time.sleep(0.05)
                    yield history
            flag = False
        except grpc.RpcError as e:
            exit(1)

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

if __name__ == "__main__":
    launch_gradio_interface()
