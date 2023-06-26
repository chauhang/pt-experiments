import os
import glob
import json
import pandas as pd
import re
import pickle
import concurrent.futures
import requests
import openai
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
import argparse


def load_and_split_data_blogs(dataset_path):
    df = pd.read_json(dataset_path)
    print('blogs dataset',df.shape)

    ## Using langchain - split the data into multiple pages
    markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    print('chunking pages into smaller sub-pages')
    pages = []
    for index, row in df.iterrows():
        markdown_text = row["text"]
        metadata = row["metadata"]
        docs = markdown_splitter.create_documents([markdown_text], [metadata])
        pages.extend(docs)

    print('total pages:', len(pages))
    return pages

def load_and_split_data_docs(dataset_path):
    df = pd.read_json(dataset_path)
    print('docs dataset',df.shape)

    df = df[(df['text'].str.len() > 100)]
    df = df.reset_index(drop=True)

    pattern = r'\*{3,}'
    df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', x))
    df['text'] = df['text'].str.replace('\n\n', '\n')
    #df.to_csv('docs_cleaned.csv')

    ## Using langchain - split the data into multiple pages
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048)
    print('chunking pages into smaller sub-pages')
    pages = []
    for index, i in df.iterrows():
        pages.extend(splitter.create_documents([i['text']], [i['metadata']]))
    
    print('total pages:', len(pages))
    return pages

def get_openai_api(context):
    if not os.environ["OPENAI_API_KEY"]:
        raise EnvironmentError(
            "OPENAI_API_KEY - key missing. set in the environment before running the script"
        )
    api_key = os.environ["OPENAI_API_KEY"]
   
    try:
        completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", api_key = api_key,
                messages=[
                    {"role": "user", "content": context}
                  ]
                )

        qa = completion.choices[0].message.content

    except requests.exceptions.RequestException as e:
        print(f'Request failed with error: {str(e)}.')
        print(f'Waiting for 3 minutes before trying again...')
        time.sleep(180)
    
    return qa

def get_qa(pages):
    print('Using openai to generate questions and answers')
    questions_ans = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = []

        for i in pages:

            context = f"Generate question and answer only in this format 'Question: Answer:' using this context \
            and you can decide the number of question and answer to generate based \
            on context size but don't generate too many same kind of questions: {i.page_content}"

            futures.append(executor.submit(get_openai_api, context))

        for future, i in tqdm(zip(concurrent.futures.as_completed(futures), pages), total=len(pages)):
            try:
                qa = future.result()
                questions_ans.append({'text':qa, 'metadata':i.metadata})
            except Exception as exc:
                print(f'generated an exception: {exc}')

    df_qa = pd.DataFrame(questions_ans)
    #df_qa.to_csv(f'qa_openai_generated.csv')
    return df_qa

## Each row contains multiple question and answers. Split it into multiple rows to have one question and answer per row
def split_sentence_by_word(sentence, split_word):
    sentences = sentence.split(split_word)
    result = [''.join([split_word, s.strip()]) for s in sentences if s.strip()]
    return result

def final_qa(df_qa):
    qa_list = []

    for index,i in df_qa.iterrows():
        result = split_sentence_by_word(i['text'], 'Question: ')
        metadata = i['metadata']
        for i in result:
            qa_list.append({'text':i, 'metadata':metadata})
    return qa_list

## Remove the rows where LLM couldnt find the answer
def clean_rows_without_answer(qa_list):
    print('cleaning data')
    to_be_excluded = []
    for i in range(len(qa_list)):
        row = qa_list[i]
        text = row["text"]
        answer = text.split("Answer: ")[-1]

        if answer.strip() == "None":
            to_be_excluded.append(i)
            continue

        if "not" in answer.lower() and "supported" in answer.lower():
            to_be_excluded.append(i)
            continue

    for index in sorted(to_be_excluded, reverse=True):
        del qa_list[index]

    return qa_list
    
def main(args):
    print('generating docs dataset')
    docs_pages = load_and_split_data_docs(args.docs_dataset_path)
    docs_qa_df = get_qa(docs_pages)
    docs_qa_list = final_qa(docs_qa_df)
    docs_qa_list = clean_rows_without_answer(docs_qa_list)
    ##Write the final dataset
    with open(f"docs_qa_dataset.json", "w") as fp:
        json.dump(docs_qa_list, fp)
    print('creating docs_qa_dataset.json')
    
    print('generating blogs dataset')
    blogs_pages = load_and_split_data_blogs(args.blogs_dataset_path)
    blogs_qa_df = get_qa(blogs_pages)
    blogs_qa_list = final_qa(blogs_qa_df)
    blogs_qa_list = clean_rows_without_answer(blogs_qa_list)
    ##Write the final dataset
    with open(f"blogs_qa_dataset.json", "w") as fp:
        json.dump(blogs_qa_list, fp)
    print('creating blogs_qa_dataset.json')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dataset_path", type=str, default="docs.json")
    parser.add_argument("--blogs_dataset_path", type=str, default="blogs.json")

    args = parser.parse_args()
    main(args)
