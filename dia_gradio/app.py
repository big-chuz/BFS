import gradio as gr
import torch
from transformers import AutoProcessor, DiaForConditionalGeneration
import scipy.io.wavfile
import random
import numpy as np

# Load the processor and model from Hugging Face
# This will be cached in the hugging_face volume you define in docker-compose
model_checkpoint = "nari-labs/Dia-1.6B-0626"
processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir="/root/.cache/huggingface")
model = DiaForConditionalGeneration.from_pretrained(model_checkpoint, cache_dir="/root/.cache/huggingface")


# Check for GPU availability and move the model to the GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_audio(text, max_new_tokens, cfg_scale, temperature, top_p, seed):
    """
    Generates audio from text using the Dia 1.6B model.
    """
    if not text:
        return None, None
    
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
        
    inputs = processor(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
        )

    # Decode the output
    audio_values = processor.batch_decode(outputs)[0].cpu().numpy()
    
    # Save to a temporary wav file
    sampling_rate = 44100
    output_path = "output.wav"
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_values)
    
    return output_path, seed

# Create the Gradio interface
title = "Dia 1.6B Text-to-Speech"
description = "Enter some text to generate audio using the Dia 1.6B model."
examples = [["Hello, my name is Dia. I am a text to speech model."]]

iface = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(lines=3, label="Text Input"),
        gr.Slider(
            label="Max New Tokens (Audio Length)",
            minimum=860,
            maximum=4096,
            value=3072,
            step=50,
            info="Controls the maximum length of the generated audio (more tokens = longer audio).",
        ),
        gr.Slider(
            label="CFG Scale (Guidance Strength)",
            minimum=1.0,
            maximum=5.0,
            value=3.0,
            step=0.1,
            info="Higher values increase adherence to the text prompt.",
        ),
        gr.Slider(
            label="Temperature (Randomness)",
            minimum=0.1,
            maximum=2.5,
            value=1.8,
            step=0.05,
            info="Lower values make the output more deterministic, higher values increase randomness.",
        ),
        gr.Slider(
            label="Top P (Nucleus Sampling)",
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.01,
            info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
        ),
        gr.Number(
            label="Generation Seed (Optional)",
            value=-1,
            precision=0,
            step=1,
            interactive=True,
            info="Set a generation seed for reproducible outputs. Leave empty or -1 for random seed.",
        ),
    ],
    outputs=[gr.Audio(label="Generated Audio"), gr.Textbox(label="Generation Seed")],
    title=title,
    description=description,
    examples=examples
)

# Launch the app so it's accessible within the Docker network
iface.launch(server_name="0.0.0.0", server_port=7860)
