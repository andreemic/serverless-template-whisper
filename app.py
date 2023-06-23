import torch
import whisper
import base64
from io import BytesIO

import os
import numpy as np
import requests

def download_audio_from_url(url):
    # Extract the filename from the URL
    filename = os.path.basename(url.split("?")[0])

    # Download the audio file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Save the downloaded file
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename

import time

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("base")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    audio_url = model_inputs.get("audio_url", None)
    if audio_url is None:
        raise ValueError("audio_url is required")
    args_overwrites = model_inputs.get("args_overwrites", None)

    audio_path = download_audio_from_url(audio_url)
    temperature = 0
    temperature_increment_on_fallback = 0.2
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]
    
    # Run the model
    args = {
        "language": None,
        "patience": None,
        "suppress_tokens": "-1",
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、"
    } if args_overwrites is None else args_overwrites

    start = time.time()
    outputs = model.transcribe(str(audio_path), temperature=temperature, **args)
    end = time.time()

    output = {"outputs": outputs}
    os.remove(audio_path)

    # Return the results as a dictionary
    return output
