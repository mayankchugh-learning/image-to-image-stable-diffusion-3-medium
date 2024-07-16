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

narrator = pipeline("text-to-speech", model=tts_model_path) 

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

def caption_my_image(pil_image):

    semantics = caption_image(images=pil_image)[0]['generated_text']
    audio = generate_audio(semantics)
    return audio


gr.close_all()

demo = gr.Interface(fn=caption_my_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[ gr.Audio(label="Image Caption")],
                    title="@IT AI Enthusiast (https://www.youtube.com/@itaienthusiast/) - Project 8: Image Captioning with AI",
                    description="THIS APPLICATION WILL BE USED TO CAPTION IMAGES WITH THE HELP OF AI")
demo.launch()