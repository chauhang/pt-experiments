import argparse

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402


def generate_model(base_model, lora_weights_path, output_model_name):

    print("Initializing Tokenizer from: ", base_model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    print("Loading base model: ", base_model)
    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    print("Applying lora weights from: ", lora_weights_path)
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v for k, v in lora_model_sd.items() if "lora" not in k
    }

    print("Saving model to:", output_model_name)
    LlamaForCausalLM.save_pretrained(
        base_model, output_model_name, state_dict=deloreanized_sd, max_shard_size="400MB"
    )

    print("Saving tokenizer to:", output_model_name)
    tokenizer.save_pretrained(output_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generating Full model")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--output_model_name", type=str, required=True)
    args = parser.parse_args()
    generate_model(
        base_model=args.base_model,
        lora_weights_path=args.lora_weights,
        output_model_name=args.output_model_name,
    )
