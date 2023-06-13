import re
import time

import gradio as gr
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer

model = LlamaForCausalLM.from_pretrained(
    "shrinath-suresh/alpaca-lora-7b-answer-summary",
    torch_dtype=torch.float16,
    device_map="auto",
)


tokenizer = LlamaTokenizer.from_pretrained("shrinath-suresh/alpaca-lora-7b-answer-summary")
streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)


class CustomLLM(LLM):
    model_name = "shrinath-suresh/alpaca-lora-7b-answer-summary"

    def _call(self, prompt, stop=None) -> str:
        inputs = tokenizer([prompt], return_tensors="pt")

        response = model.generate(**inputs, streamer=streamer, max_new_tokens=128)
        response = tokenizer.decode(response[0])
        return response

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
    text = llm_chain.run(question)
    pattern = r"### Response:(.*?)###"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        latest_response = matches[-1].strip()
        print("\n\n")
        print("*" * 150)
        print(latest_response)
        print("*" * 150)
        print("\n\n")
        return latest_response
    else:
        return "I don't know"


def launch_gradio_interface():
    CSS = """
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    """

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        print("Sending Query!")
        bot_message = run_query(question=history[-1][0])
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


if __name__ == "__main__":
    launch_gradio_interface()
