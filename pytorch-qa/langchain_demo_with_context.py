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
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.faiss import FAISS
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer


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


def read_prompt_from_path(prompt_path):
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict


def setup(model_name, model, prompt_template, max_tokens):

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
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["chat_history", "question", "context"]
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


def run_query(llm_chain, question, index_path, memory):
    embeddings = HuggingFaceEmbeddings()
    if not os.path.exists(index_path):
        raise ValueError(f"Index path - {index_path} does not exists")
    faiss_index = FAISS.load_local(index_path, embeddings)
    context = faiss_index.similarity_search(question, k=2)
    result = llm_chain.run({"question": question, "context": context})
    print(memory.chat_memory.messages[1].content)
    memory.clear()
    parsed_response = parse_response(result)
    return parsed_response


def launch_gradio_interface(llm_chain, index_path, memory):
    CSS = """
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    """

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        print("Sending Query!")
        bot_message = run_query(
            question=history[-1][0],
            llm_chain=llm_chain,
            index_path=index_path,
            memory=memory
        )
        print(">>>>>>>>>>>>>>>>>>>>>>>>> Query: ", history[-1][0])
        print(">>>>>>>>>>>>>>>>>>>>>>>> Answer: ", bot_message)
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

        res = generate.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        res.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(
        server_name="0.0.0.0", ssl_verify=False, debug=True, show_error=True, share=True
    )


def main(args):
    model = load_model(model_name=args.model_name)
    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    if args.prompt_name not in prompt_dict:
        raise KeyError(
            f"Invalid key - {args.prompt_name}. Accepted values are {prompt_dict.keys()}"
        )
    prompt_template = prompt_dict[args.prompt_name]
    llm_chain, memory = setup(
        model_name=args.model_name,
        model=model,
        prompt_template=prompt_template,
        max_tokens=args.max_tokens
    )
    # result = run_query(llm_chain=llm_chain, index_path=args.index_path, question="How to save the model", memory=memory)

    launch_gradio_interface(
        llm_chain=llm_chain, index_path=args.index_path, memory=memory
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Langchain with context demo")
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128
    )
    parser.add_argument("--prompt_path", type=str, default="question_with_context.json")
    parser.add_argument("--prompt_name", type=str, default="QUESTION_WITH_CONTEXT_PROMPT_ADVANCED_PROMPT")
    parser.add_argument("--index_path", type=str, default="docs_blogs_faiss_index")

    args = parser.parse_args()
    main(args)
