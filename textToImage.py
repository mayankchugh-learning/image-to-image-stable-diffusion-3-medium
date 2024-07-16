import torch
import gradio as gr
from diffusers import StableDiffusion3Pipeline

def image_generator(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", 
                                                         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                         text_encoder_3=None,
                                                         tokenizer_3 = None)
    #pipeline.enable_model_cpu_offload()
    pipeline.to(device)

    image = pipeline(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low, resolution, blurry",
       # max_inference_steps=40,
        num_inference_steps=40,
        height=1024,
        width=1024,
        guidance_scale=9.0
    ).images[0]

    #image.show()
    return image

#image_generator("A magical cat doing spell")

interface = gr.Interface(
    fn=image_generator, 
    inputs=gr.Textbox(lines=2, placeholder = "Enter your prompt..."), 
    outputs=gr.Image(type = "pil"),
    title = "Image Generator App",
    description = "This is a simple image generator app using HuggingFace's Stable Diffusion 3 model.")

interface.launch()


print(interface)
