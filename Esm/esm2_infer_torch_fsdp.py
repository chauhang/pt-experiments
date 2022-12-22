import torch
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
import esm
from argparse import ArgumentParser

def initialize(model_name):
    # init the distributed world with world_size 1
    print("Initializing")
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    print("Downloading data")
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

    model, vocab = esm.pretrained.load_model_and_alphabet_core( model_name, model_data, regression_data)


    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    model.eval()

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model, vocab


def infer(vocab, model, data):
    batch_converter = vocab.get_batch_converter()

    print("Running predictions")

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    print(results)


if __name__ == '__main__':
    parser = ArgumentParser("ESM fsdp Inference")

    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
    )

    args = parser.parse_args()

    model, vocab = initialize(model_name=args.model_name)

    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        (
            "protein2 with mask",
            "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        ),
        ("protein3", "K A <mask> I S Q"),
    ]

    infer(model=model, vocab=vocab, data=data)

