import json
import time

import requests
from fastchat import client
from fastchat.conversation import (
    get_default_conv_template,
    compute_skip_echo_len,
    SeparatorStyle,
)
from flask import Flask, request
from flask_cors import CORS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# create a Flask instance
app = Flask(__name__)
app.app_context()
CORS(app)


model_name = "vicuna-13b"
controller_addr = "http://localhost:21001"
worker_addr = "http://localhost:21002"
client.set_baseurl("http://localhost:8000")
max_new_tokens = 512
temperature = 0.0
FAISS_INDEX_FILE = "faiss_index"
counter = 0


# initialises and loads the vector db
def initialize():
    print("Initializing...")
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(FAISS_INDEX_FILE, embeddings)
    return db


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def rephrase_query(query):
    # USER: Rephrase the question into a standalone question based on the previous answers
    prompt = f""" this is the chat history -
    {conv.get_prompt()}
    USER: Generate a standalone question which is based on the new question plus the chat history. Just create the standalone question without commentary. 
    New question: {query}
    Assistant:
    """
    completion = client.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )

    print("Rephrased question: ", completion.choices[0].message)
    conv.append_message(conv.roles[1], str(completion.choices[0].message))
    return completion.choices[0].message.content


# takes in the query, rephrases the query, finds the best matching context from the db,
# hits fast chat to find the correct answer from the context.
def gen_answer(query):
    global counter
    if counter != 0:
        # new conversation
        query = rephrase_query(query)

    counter += 1

    context = db.similarity_search(query, k=2)
    prompt = f"""
    context:
    {context}
    conversation:
    {conv.get_prompt()}
    Based on the above context, answer the question given below
    USER: {query}
    Assistant:
    """

    headers = {"User-Agent": "fastchat Client"}
    pload = {
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }
    qa_time = time.time()
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=pload,
        stream=True,
    )

    final_output = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            skip_echo_len = compute_skip_echo_len(model_name, conv, prompt)
            output = data["text"][skip_echo_len:].strip()
            output = post_process_code(output)
            final_output = output
            print(f"{conv.roles[1]}: {output}", end="\r")
            # yield output
    conv.append_message(conv.roles[1], final_output)
    print("Conv:", conv)

    return final_output


@app.route("/answer", methods=["POST"])
def answer_api():
    data = request.get_json()
    if not data["query"]:
        error_message = "Required paremeters : 'query'"
        return error_message
    else:
        user_query = data["query"]
    answer = gen_answer(user_query)

    return answer


@app.route("/clear_chat", methods=["GET"])
def clear_history():
    conv = get_default_conv_template(model_name).copy()
    global counter
    counter = 0

    return "success"


if __name__ == "__main__":
    db = initialize()
    conv = get_default_conv_template(model_name).copy()
    app.run(host="0.0.0.0", port=5000)
