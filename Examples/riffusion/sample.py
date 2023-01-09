import argparse
import os
import typing as T
from pathlib import Path
import json

import PIL
import torch
import dacite

from riffusion.datatypes import InferenceInput, InferenceOutput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

MODEL_ID = "riffusion/riffusion-model-v1"

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path("./seed_images")


def compute_request(
    inputs: InferenceInput,
    pipeline: RiffusionPipeline,
    seed_images_dir: str,
):
    """
    Does all the heavy lifting of the request.
    Args:
        inputs: The input dataclass
        pipeline: The riffusion model pipeline
        seed_images_dir: The directory where seed images are stored
    """
    # Load the seed image by ID
    init_image_path = Path(seed_images_dir, f"{inputs.seed_image_id}.png")

    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_id}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    mask_image: T.Optional[PIL.Image.Image] = None
    if inputs.mask_image_id:
        mask_image_path = Path(seed_images_dir, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")

    # Execute the model to get the spectrogram image
    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=mask_image,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct audio from the image
    # TODO(hayk): It may help performance a bit to cache this object
    converter = SpectrogramImageConverter(
        params=params, device=str(pipeline.device)
    )

    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    if not os.path.exists("out/"):
        os.mkdir("out")

    out_img_path = "out/spectrogram.jpg"
    image.save("out/spectrogram.jpg", exif=image.getexif())

    out_wav_path = "out/gen_sound.wav"
    segment.export(out_wav_path, format="wav")

    return InferenceOutput(
        image=out_img_path,
        audio=out_wav_path,
        duration_s=segment.duration_seconds,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = RiffusionPipeline.load_checkpoint(
        checkpoint=MODEL_ID,
        use_traced_unet=True,
        device=DEVICE,
    )

    with open(args.input) as fp:
        json_data = json.load(fp)
        print(json_data)

    inputs = dacite.from_dict(InferenceInput, json_data)
    response = compute_request(
        inputs=inputs,
        seed_images_dir=SEED_IMAGES_DIR,
        pipeline=pipeline,
    )

    print(response)
