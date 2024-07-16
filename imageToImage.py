import torch
import gradio as gr
from PIL  import Image
import scipy.io.wavfile as wavfile

# Use a pipeline as a high-level helper
from transformers import pipeline

from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'  #use the Path to the library.
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# narrator = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

tts_model_path = "../Models/models--kakao-enterprise--vits-ljs/snapshots/3bcb8321394f671bd948ebf0d086d694dda95464"

narrator = pipeline("text-to-speech", model=tts_model_path, device=device)

# Load the pretrained weights
# caption_image = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)

model_path = "../Models/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90"

# Load the pretrained weights
caption_image = pipeline("image-to-text", model=model_path, device=device)

# Define the function to generate audio from text 
def generate_audio(text):
    # Generate the narrated text
    narrated_text = narrator(text)

    # Save the audio to WAV file
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"],
                  data=narrated_text["audio"][0])

    # Return the path to the saved output WAV file
    return "output.wav"

from diffusers import StableDiffusion3Pipeline

def image_generator(prompt):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", 
                                                         torch_dtype=torch.float32 if device == "cuda" else torch.float32,
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

def image_to_image(pil_image):

    semantics = caption_image(images=pil_image)[0]['generated_text']
    audio = generate_audio(semantics)
    image = image_generator(semantics)
    return audio, image

gr.close_all()

demo = gr.Interface(fn=image_to_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[ gr.Audio(label="Image Caption"), gr.Image(type = "pil")],
                    title = "Image To Image Generator App",
                    description = "This is a simple image image generator app using HuggingFace's Stable Diffusion 3 model.")
demo.launch()