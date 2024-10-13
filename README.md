# HindiSRT
Free Hindi Subtitle Generator
Sure! Below is a sample `README.md` file tailored to your project. You can customize it further based on your specific needs.

---

# Video Audio Transcription Tool

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The **Video Audio Transcription Tool** is a Python-based application that allows users to upload video files, extract audio, transcribe the audio into text using advanced speech recognition models, and generate SRT subtitle files. This tool leverages powerful libraries such as `moviepy`, `transformers`, `librosa`, `torch`, and `whisper` to provide accurate and efficient transcription services.

## Features

- **Video Upload**: Easily upload video files through Google Colab.
- **Audio Extraction**: Extract audio from video files using `moviepy`.
- **Transcription**:
  - **Wav2Vec2**: Utilize Facebook's Wav2Vec2 model for speech-to-text transcription.
  - **Whisper**: Use OpenAI's Whisper model for advanced transcription capabilities.
- **Subtitle Generation**: Create SRT files for video subtitles.
- **Easy Integration**: Designed to work seamlessly in Google Colab environments.

## Installation

To get started with this project, follow the steps below:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/video-audio-transcription-tool.git
   cd video-audio-transcription-tool
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Ensure you have `pip` installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The primary script is designed to run in a Google Colab environment. Follow the steps below to use the tool:

1. **Mount Google Drive**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Upload Video File**

   ```python
   from google.colab import files
   uploaded = files.upload()
   
   for filename in uploaded.keys():
       print(f'User uploaded file "{filename}" with length {len(uploaded[filename])} bytes.')
   ```

3. **Extract Audio from Video**

   ```python
   from moviepy.editor import VideoFileClip
   
   video_filename = list(uploaded.keys())[0]
   audio_filename = "extracted_audio.wav"
   
   video = VideoFileClip(video_filename)
   video.audio.write_audiofile(audio_filename)
   
   print(f'Audio extracted and saved as {audio_filename}')
   ```

4. **Transcribe Audio Using Wav2Vec2**

   ```python
   import torch
   from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
   import librosa
   import numpy as np
   
   tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
   model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
   
   def transcribe(audio_path):
       speech, rate = librosa.load(audio_path, sr=16000)
       input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values
       with torch.no_grad():
           logits = model(input_values).logits
       predicted_ids = torch.argmax(logits, dim=-1)
       transcription = tokenizer.decode(predicted_ids[0])
       return transcription.lower()
   
   transcription = transcribe(audio_filename)
   print("Transcription:")
   print(transcription)
   ```

5. **Transcribe Audio Using Whisper**

   ```python
   import whisper
   
   model = whisper.load_model("medium")  # Options: tiny, base, small, medium, large
   
   result = model.transcribe(audio_filename, language='hi')  # Change language as needed
   transcription = result['text']
   print("Transcription:")
   print(transcription)
   ```

6. **Generate SRT Subtitles**

   ```python
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
   
   if 'result' in locals():
       srt_filename = "subtitles.srt"
       generate_srt(result, srt_filename)
   ```

## Models Used

1. **Wav2Vec2**
   - **Description**: A powerful model for automatic speech recognition developed by Facebook AI.
   - **Repository**: [Wav2Vec2 on Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)
   
2. **Whisper**
   - **Description**: OpenAI’s Whisper is a general-purpose speech recognition model.
   - **Repository**: [Whisper on GitHub](https://github.com/openai/whisper)

## Project Structure

```
video-audio-transcription-tool/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── transcribe.py
└── srt_output/
    └── subtitles.srt
```

- **transcribe.py**: The main script containing the code you provided.
- **srt_output/**: Directory to store generated SRT subtitle files.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [MoviePy](https://zulko.github.io/moviepy/) for video and audio processing.
- [Transformers](https://huggingface.co/transformers/) by Hugging Face for the Wav2Vec2 model.
- [Whisper](https://github.com/openai/whisper) by OpenAI for advanced transcription capabilities.
- [Librosa](https://librosa.org/) for audio processing.

---

## Getting Started

1. **Fork the Repository**

   Click the "Fork" button at the top right of this page to create your own copy of the repository.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/your-username/video-audio-transcription-tool.git
   cd video-audio-transcription-tool
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Script**

   Open the `transcribe.py` script in Google Colab and follow the usage instructions.

---

Feel free to contribute to this project by opening issues or submitting pull requests. Happy transcribing!
