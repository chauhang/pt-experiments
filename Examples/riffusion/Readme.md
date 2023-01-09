# This document covers steps for running riffusion library for music and audio generation with stable diffusion.

## Riffusion is a library for real-time music and audio generation with stable diffusion.


### Step 1: Clone riffusion repository

```bash
git clone https://github.com/riffusion/riffusion
cd riffusion
```

### Step 2: Create a conda env and install dependencies

```bash
conda create --name riffusion python=3.9
conda activate riffusion

pip install -r requirements.txt
```

### Step 3: Clone and navigate to example/riffusion folder in pt-experiments

```bash
git clone https://github.com/chauhang/pt-experiments
cd pt-experiments/Examples/riffusion
```

### Step 4: Add riffusion module to PYTHONPATH

```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/riffusion
```

### Step 5: Run example

```bash
python sample.py --input input.json
```

The output spectrogram image and audio files are written to `out` directory.
