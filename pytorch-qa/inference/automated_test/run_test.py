import argparse
import json
import os
import re
import shutil

import pandas as pd
import torch
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.faiss import FAISS
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer
import huggingface_hub as hf_hub


def read_prompt_from_path(prompt_path):
    with open(prompt_path) as fp:
        prompt_dict = json.load(fp)
    return prompt_dict


def read_questions_from_path(question_path):
    with open(question_path) as fp:
        question_list = json.load(fp)
    return question_list


def load_model(model_name):
    print("Loading model: ", model_name)
    api = "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"
    hf_hub.login(token=api)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def setup(model_name, prompt_type, model, prompt_template):

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)

    class CustomLLM(LLM):
        def _call(self, prompt, stop=None) -> str:
            inputs = tokenizer([prompt], return_tensors="pt")

            response = model.generate(**inputs, streamer=streamer, max_new_tokens=128)
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


def run_query(prompt_type, llm_chain, question_list, memory):
    result_list = []
    chat_history_list = []
    actual_result_list = []
    for question in tqdm(question_list):
        if prompt_type == "question_with_context":
            embeddings = HuggingFaceEmbeddings()
            faiss_index = FAISS.load_local("docs_blogs_faiss_index", embeddings)
            context = faiss_index.similarity_search(question, k=2)
            result = llm_chain.run({"question": question, "context": context})
        else:
            result = llm_chain.run(question)

        chat_history = memory.chat_memory.messages[1].content
        chat_history_list.append(chat_history)

        actual_result_list.append(result)
        parsed_response = parse_response(result)
        result_list.append(parsed_response)
        memory.clear()

    return pd.DataFrame(
        {
            "question": question_list,
            "answer": result_list,
            "full_answer": actual_result_list,
            "chat_history": chat_history_list,
        }
    )


def write_model_output(model_name, prompt_type, df, result_dir):
    output_folder_name = model_name
    output_folder_name = output_folder_name.replace("/", "_")

    if os.path.exists(output_folder_name):
        print(f"Directory {output_folder_name} already present")
    else:
        os.mkdir(output_folder_name)

    if not os.path.exists(args.results_path):
        os.mkdir(result_dir)

    full_output_folder = os.path.join(result_dir, output_folder_name)
    if not os.path.exists(full_output_folder):
        os.mkdir(full_output_folder)

    output_file_path = os.path.join(
        full_output_folder, f"test_{output_folder_name}_{prompt_type}.csv"
    )
    print("Writing output to :", output_file_path)
    df.to_csv(output_file_path, index=False)


def infer(model_name, model, args):

    prompt_dict = read_prompt_from_path(prompt_path=args.prompt_path)
    question_list = read_questions_from_path(question_path=args.question_path)

    for prompt_type, prompt_list in prompt_dict.items():
        print("Processing prompt type: ", prompt_type)

        df_list = []
        for prompt_template in prompt_list:
            print("Running test for prpmpt template: ", prompt_template)
            llm_chain, memory = setup(
                model_name=model_name,
                prompt_type=prompt_type,
                model=model,
                prompt_template=prompt_template,
            )
            result_df = run_query(
                prompt_type=prompt_type,
                llm_chain=llm_chain,
                question_list=question_list,
                memory=memory,
            )
            result_df["prompt"] = prompt_template
            df_list.append(result_df)

        final_df = pd.concat(df_list, ignore_index=True)
        write_model_output(
            model_name=model_name,
            prompt_type=prompt_type,
            df=final_df,
            result_dir=args.results_path,
        )


def read_model_list(model_path):
    with open(model_path) as fp:
        model_list = json.load(fp)

    return model_list


def remove_cache(cache_path):
    shutil.rmtree(cache_path)


def main(args):
    model_list = read_model_list(model_path=args.model_path)
    for model_name in model_list:
        print("Processing model: ", model_name)
        print("Loading model: ", model_name)
        model = load_model(model_name)
        infer(model_name, model, args)
        remove_cache(cache_path=args.cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models.json")
    parser.add_argument("--prompt_path", type=str, default="prompts.json")
    parser.add_argument("--question_path", type=str, default="questions.json")
    parser.add_argument("--cache_path", type=str, default="/home/ubuntu/.cache/huggingface")
    parser.add_argument("--results_path", type=str, default="./latest_results")

    args = parser.parse_args()
    main(args)
