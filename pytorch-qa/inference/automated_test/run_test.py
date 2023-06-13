import argparse
import json

import pandas as pd
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferWindowMemory
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer


def read_prompt_and_questions(args):
    with open(args.prompt_path) as fp:
        prompt_dict = json.load(fp)
        prompt_list = prompt_dict[args.prompt_type]

    with open(args.question_path) as fp:
        question_list = json.load(fp)
    return prompt_list, question_list


def load_model(args):
    print("Loading model: ", args.model_name)
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model


def setup(args, model, prompt_template):

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, Timeout=5)

    class CustomLLM(LLM):
        model_name = args.model_name

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
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, output_key="result")
    return llm_chain, memory


def run_query(llm_chain, question_list, memory):
    result_list = []
    for question in tqdm(question_list):
        result = llm_chain.run(question)
        result_list.append(result)
        memory.clear()

    return pd.DataFrame({"question": question_list, "answer": result_list})


def main(args):
    prompt_list, question_list = read_prompt_and_questions(args)
    df_list = []
    model = load_model(args)
    for prompt_template in prompt_list:
        llm_chain, memory = setup(args, model=model, prompt_template=prompt_template)
        result_df = run_query(llm_chain=llm_chain, question_list=question_list, memory=memory)
        result_df["prompt"] = prompt_template
        df_list.append(result_df)

    final_df = pd.concat(df_list, ignore_index=True)
    output_model_name = args.model_name
    output_model_name = output_model_name.replace("/", "_")
    output_model_name = f"test_{output_model_name}.csv"
    final_df.to_csv(output_model_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="shrinath-suresh/alpaca-lora-7b-answer-summary"
    )
    parser.add_argument("--prompt_path", type=str, default="prompts.json")
    parser.add_argument("--prompt_type", type=str, default="only_question")
    parser.add_argument("--question_path", type=str, default="questions.json")

    args = parser.parse_args()
    main(args)
