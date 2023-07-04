import argparse
import os
from transformers import LlamaForCausalLM


def push_model_to_hub(local_model_path, hf_model_name):
    model = LlamaForCausalLM.from_pretrained(local_model_path)
    if not os.environ["HUGGINGFACE_KEY"]:
        raise ValueError(
            "Huggingface key missing. Set api key in environment variable - HUGGINGFACE_KEY"
        )

    hf_api_key = os.environ["HUGGINGFACE_KEY"]

    model.push_to_hub(repo_id=hf_model_name, private=True, use_auth_token=hf_api_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Push To Hugingface Hub")
    parser.add_argument("--local_model_path", type=str, required=True)
    parser.add_argument("--hf_model_name", type=str, required=True)
    args = parser.parse_args()
    push_model_to_hub(local_model_path=args.local_model_path, hf_model_name=args.hf_model_name)
