import time
import argparse
import json
import os
import time

import gradio as gr
import huggingface_hub as hf_hub
import torch
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.vectorstores.faiss import FAISS
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer
from langchain.memory import ConversationBufferWindowMemory
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

def load_model(model_name):
    print("Loading model: ", model_name)
    if not os.environ["HUGGINGFACE_KEY"]:
        raise EnvironmentError(
            "HUGGINGFACE_KEY - key missing. set in the environment before running the script"
        )
    hf_hub.login(token=os.environ["HUGGINGFACE_KEY"])
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def read_prompt_from_path(prompt_path="prompts.json"):
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict


def setup(model_name, prompt_type, model, prompt_template, max_tokens):

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)

    class CustomLLM(LLM):
        def _call(self, prompt, stop=None) -> str:
            inputs = tokenizer([prompt], return_tensors="pt")

            response = model.generate(**inputs, streamer=streamer, max_new_tokens=max_tokens)
            response = tokenizer.decode(response[0])
            return response

        @property
        def _identifying_params(self):
            return {"name_of_model": model_name}

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()
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


def parse_response(text):
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
        text = text.split("###")[0]
        return text
    else:
        return text


def run_query(prompt_type, llm_chain, question, index_path, memory):

    if prompt_type == "question_with_context":
        embeddings = HuggingFaceEmbeddings()
        if not os.path.exists(index_path):
            raise ValueError(f"Index path - {index_path} does not exists")
        faiss_index = FAISS.load_local(index_path, embeddings)
        context = faiss_index.similarity_search(question, k=2)
        result = llm_chain.run({"question": question, "context": context})
    else:
        result = llm_chain.run(question)
    memory.clear()
    parsed_response = parse_response(result)
    return parsed_response

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



def launch_gradio_interface(prompt_type, llm_chain, index_path, memory):
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
        print("Sending Query!", history[-1][0])
        bot_message = run_query(prompt_type, llm_chain, history[-1][0], index_path, memory)
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

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
    model = load_model(model_name=args.model_name)
    prompt_dict = read_prompt_from_path()
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    llm_chain, memory = setup(
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        model=model,
        prompt_template=prompt_template,
        max_tokens=args.max_tokens
    )

    launch_gradio_interface(
        prompt_type=args.prompt_type, llm_chain=llm_chain, index_path=args.index_path, memory=memory
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128
    )
    parser.add_argument("--prompt_type", type=str, default="only_question")
    parser.add_argument("--prompt_name", type=str, default="ONLY_QUESTION_ADVANCED_PROMPT")

    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")

    args = parser.parse_args()
    print("Contexxt : ", args.prompt_type)
    main(args)