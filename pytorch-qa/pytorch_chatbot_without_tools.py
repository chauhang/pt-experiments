import logging
import os
import sys
import time
from argparse import ArgumentParser
from typing import Any, List, Mapping, Optional

import gradio as gr
import huggingface_hub as hf_hub
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms.base import LLM
from llama_index import (
    LLMPredictor,
    PromptHelper,
    StorageContext,
    ServiceContext,
    LangchainEmbedding,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    load_index_from_storage,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.readers.file.markdown_reader import MarkdownReader
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from peft import PeftModel
from transformers import TextStreamer, LlamaTokenizer, LlamaForCausalLM

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def create_prompt_helper(args):
    logging.info("\ncreating prompt helper")
    MAX_CHUNK_OVERLAP = 20
    prompt_helper = PromptHelper(
        args.prompt_max_input_size, args.prompt_max_output, MAX_CHUNK_OVERLAP
    )
    return prompt_helper


def get_embed_model(args):
    logging.info("Selecting embed model")
    hf_embed_model = HuggingFaceInstructEmbeddings(model_name=args.embed_model_name)
    embed_model = LangchainEmbedding(hf_embed_model)
    return embed_model


def load_index():

    # load data
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./pytorch_docs_512"),
    )
    index = load_index_from_storage(storage_context, service_context=service_context)

    return index


def get_tokenizer(model_id, hf_key):
    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False, use_auth_token=hf_key)
    return tokenizer


def get_streamer(tokenizer, hf_key):
    return TextStreamer(tokenizer, skip_prompt=True, Timeout=5)


def read_hf_key_from_env():
    if not os.getenv("HUGGINGFACE_KEY"):
        raise EnvironmentError("Huggingface key HUGGINGFACE_KEY - missing")
    else:
        hf_key = os.getenv("HUGGINGFACE_KEY")

    return hf_key


def load_custom_llm_predictor(args):
    logging.info("Loading custom llm predictor")

    HF_KEY = read_hf_key_from_env()

    tokenizer = LlamaTokenizer.from_pretrained(args.custom_base_model)

    hf_hub.login(token=HF_KEY)

    base_model_name = args.custom_base_model

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model, args.custom_delta_model, torch_dtype=torch.float16, load_in_8bit=True
    )
    streamer = get_streamer(tokenizer=tokenizer, hf_key=HF_KEY)

    class CustomLLM(LLM):
        model_name = args.custom_delta_model

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            inputs = tokenizer([prompt], return_tensors="pt")

            # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
            generation_kwargs = dict(
                inputs, streamer=streamer, max_new_tokens=args.prompt_max_output
            )
            # thread = Thread(target=model.generate, kwargs=generation_kwargs)
            # thread.start()
            response = model.generate(
                **inputs, streamer=streamer, max_new_tokens=args.prompt_max_output
            )
            response = tokenizer.decode(response[0])
            return response
            # return next(streamer)

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"name_of_model": self.model_name}

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm_predictor = LLMPredictor(llm=CustomLLM())
    return llm_predictor


def create_llm_predictor(args):
    llm_predictor = load_custom_llm_predictor(args=args)
    logging.info("created LLM Predictor: ")

    return llm_predictor


def create_service_context(prompt_helper, llm_predictor, embed_model):
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model
    )
    return service_context


def create_text_splitter(args):
    text_splitter = TokenTextSplitter(chunk_size=args.text_splitter_chunk_size, chunk_overlap=200)
    simple_node_parser = SimpleNodeParser(text_splitter=text_splitter)
    return text_splitter, simple_node_parser


def set_metadata(filename):
    return {"source": filename}


def read_pytorch_docs(args):
    input_files = []
    doc_path = args.pytorch_docs_path
    for path, subdirs, files in os.walk(doc_path):
        for name in files:
            file = os.path.join(path, name)
            input_files.append(file)
    docs = SimpleDirectoryReader(
        input_dir=doc_path,
        recursive=True,
        file_extractor={".txt": MarkdownReader()},
        file_metadata=set_metadata,
    ).load_data()
    logging.info("Docs generated")
    return docs


def create_query_engine(service_context, docs, args):
    logging.info("Inside Create query engine")
    # nodes = simple_node_parser.get_nodes_from_documents(docs)
    logging.info("creating index")

    if args.load_index_from_local == True:
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./pytorch_docs_1024"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./pytorch_docs_1024"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="./pytorch_docs_1024"),
        )
        index = load_index_from_storage(storage_context, service_context=service_context)
        # index.storage_context.persist("./pytorch_docs_1024")
    else:
        index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    logging.info("Initializing query engine")
    query_engine = index.as_query_engine(service_context=service_context)
    return query_engine


def run_query(query, query_engine):
    response = query_engine.query(query)
    # print("<<<<<<<<<<<<<<<Response type: ", dir(response))
    print("Response source nodes: ", response.source_nodes)
    print("Response from backend: ", response.response)
    return response.response


def launch_gradio_interface(query_engine):
    CSS = """
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    """

    def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

    def bot(history):
        print("Sending Query!")
        bot_message = run_query(query=history[-1][0], query_engine=query_engine)
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


def main(args):
    logging.info("Entering main function")
    prompt_helper = create_prompt_helper(args)
    embed_model = get_embed_model(args)
    llm_predictor = create_llm_predictor(args)
    service_context = create_service_context(
        prompt_helper=prompt_helper, llm_predictor=llm_predictor, embed_model=embed_model
    )
    logging.info("Created service context: ")
    logging.info("Reading pytorch docs")
    docs = read_pytorch_docs(args)
    logging.info("Pytorch docs read successfully")
    query_engine = create_query_engine(service_context=service_context, docs=docs, args=args)
    logging.info("Query engine created successfully")
    # run_query(query="What is pytorch", query_engine=query_engine)
    launch_gradio_interface(query_engine=query_engine)


if __name__ == "__main__":
    parser = ArgumentParser("PyTorch Chatbot")
    parser.add_argument(
        "--document_index_path", default="./pytorch_docs_512", help="Path to document index"
    )
    parser.add_argument("--prompt_max_input_size", default=4096, help="Max input size for prompt")
    parser.add_argument(
        "--prompt_max_output", default=64, help="Max output length for prompt helper"
    )
    parser.add_argument(
        "--embed_model_name", default="hkunlp/instructor-xl", help="Default embed model name"
    )

    parser.add_argument(
        "--custom_base_model",
        default="decapoda-research/llama-7b-hf",
        help="Base llama model - HF model name",
    )

    parser.add_argument(
        "--custom_delta_model",
        default="shrinath-suresh/alpaca-lora-all-7b-delta",
        help="delta model - HF model name",
    )

    parser.add_argument("--text_splitter_chunk_size", default=1024, help="Text splitter chunk size")

    parser.add_argument(
        "--pytorch_docs_path", default="/home/ubuntu/text", help="Path to pytorch docs"
    )

    parser.add_argument(
        "--load_index_from_local", default=True, type=bool, help="Load index from local"
    )

    parser.add_argument("--query", default="What is pytorch?", help="User query")

    args = parser.parse_args()

    main(args)
