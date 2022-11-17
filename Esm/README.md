## ESM

This example loads the esm2 pretrained model using torch fsdp.

To know more about esm - https://github.com/facebookresearch/esm 

This example is refined from the fairscale fsdp implementation - https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py


## Setting up the environment

Create a new conda environment

```
conda env create --name esm --file python37.yml
```

Install torch>=1.13.0

```
pip install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Install esm using pip

```
pip install fair-esm
```

Or to install from source

```
pip install git+https://github.com/facebookresearch/esm.git
```

Install dependent packages (dllogger, openfold)

```
pip install fair-esm[esmfold]
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

## Running inference

Run the following command to run the inference 

```
python esm2_infer_torch_fsdp.py --model_name esm2_t36_3B_UR50D
```

Note: the tests are run in Tesla V100 - 16GB GPU machine, which can load `esm2_t36_3B_UR50D` and `esm2_t33_650M_UR50D`