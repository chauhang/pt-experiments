import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "darkstorm2150/Protogen_x3.4_Official_Release", torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

PROMPT = (
    "a beautiful perfect face girl in dgs illustration style, Anime fine"
    " details portrait of school girl in front of modern tokyo city landscape"
    " on the background deep bokeh, anime masterpiece, 8k, sharp high quality"
    " anime"
)
N_PROMPT = (
    "canvas frame, cartoon, 3d, ((disfigured)), ((bad art)),"
    " ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry,"
    " (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra"
    " fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)),"
    " (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad"
    " proportions))), ((extra limbs)), cloned face, (((disfigured))), out of"
    " frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed"
    " limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra"
    " legs))), mutated hands, (fused fingers), (too many fingers), (((long"
    " neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly"
    " drawn feet, poorly drawn face, out of frame, mutation, mutated, extra"
    " limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out"
    " of frame, blurry, bad art, bad anatomy, 3d render Steps: 30, Sampler:"
    " DPM++ SDE Karras, CFG scale: 10, Seed: 1495009790, Face restoration:"
    " CodeFormer, Size: 760x1024, Model hash: 60fe2f34, Denoising strength:"
    " 0.5, First pass size: 0x0"
)

image = pipe(
    PROMPT, negative_prompt=N_PROMPT, num_inference_steps=50, guidance_scale=7.5
).images[0]

image.save("output.png")
