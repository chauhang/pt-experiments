import os
import json
import tiktoken
from multiprocessing import Pool
from transformers import AutoTokenizer
import argparse

# enc = tiktoken.get_encoding("r50k_base")
enc = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
  # "gpt2"
)

def get_token_count(qa_pair):
    return len(enc.encode(str(qa_pair)))
   
def count_num_tokens(file_path):
    
    token_count_path = "/".join(file_path.split('/')[:-1])
    
    token_counts = {}

    print(f"Processing {file_path}...")
    with open(file_path, "r") as f:
        qa_pairs = [json.loads(x) for x in f.readlines()]
        print(f"Got {len(qa_pairs[0])} QA pairs for {file_path}")
    
    token_count = 0
    with Pool(24) as p:
        token_count = sum(p.map(get_token_count, qa_pairs))
    token_counts['token_count'] = token_count

    print(f"Got {token_count} tokens for {file_path}")
    # write to file

    print(f"saving token_count in {token_count_path}/token_counts.json")
    with open(os.path.join(token_count_path + "/token_counts.json"), "w") as f:
        json.dump(token_counts, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='count number of tokens in file')
    parser.add_argument('--file_path', type=str, default='knowledgebase', help='Path to file and output files to be saved')
    args = parser.parse_args()

    count_num_tokens(args.file_path)
