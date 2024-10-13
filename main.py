from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

import os

for filename in uploaded.keys():
    print(f'User uploaded file "{filename}" with length {len(uploaded[filename])} bytes.')

from moviepy.editor import VideoFileClip

video_filename = list(uploaded.keys())[0]
audio_filename = "extracted_audio.wav"

# Extract audio
video = VideoFileClip(video_filename)
video.audio.write_audiofile(audio_filename)

print(f'Audio extracted and saved as {audio_filename}')

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to transcribe audio
def transcribe(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription.lower()

transcription = transcribe(audio_filename)
print("Transcription:")
print(transcription)

import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # Options: tiny, base, small, medium, large

# Transcribe the audio
result = model.transcribe(audio_filename, language='hi')  # Change language as needed
transcription = result['text']
print("Transcription:")
print(transcription)

def generate_srt(result, srt_filename):
    srt = ""
    for i, segment in enumerate(result['segments'], 1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip().replace('-->', '->')  # Avoid SRT conflict
        srt += f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n"
    
    with open(srt_filename, 'w') as f:
        f.write(srt)
    print(f'SRT subtitles saved as {srt_filename}')

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

# If using Whisper
if 'result' in locals():
    srt_filename = "subtitles.srt"
    generate_srt(result, srt_filename)


