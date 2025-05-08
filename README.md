# NVIDIA Parakeet TDT ASR Demo

This repository contains scripts for automatic speech recognition (ASR) using NVIDIA's Parakeet TDT (Time Domain Transformer) model. The scripts demonstrate how to use the `nvidia/parakeet-tdt-0.6b-v2` model from the NeMo framework for transcribing audio files with various options.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Model Description](#model-description)
- [Scripts](#scripts)
  - [Basic Demo (`asr_demo.py`)](#basic-demo-asr_demopy)
  - [Long Audio Processing (`asr_with_split.py`)](#long-audio-processing-asr_with_splitpy)
  - [Command Line Interface (`asr_with_split_cmd.py`)](#command-line-interface-asr_with_split_cmdpy)
- [Usage Examples](#usage-examples)
- [Output Formats](#output-formats)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides tools for transcribing audio files using NVIDIA's state-of-the-art Parakeet TDT ASR model. The scripts handle various scenarios, from simple transcription to processing long audio files and generating subtitles in SRT format.

Key features:
- Basic transcription with word, character, and segment-level timestamps
- Long audio file processing with automatic splitting and merging
- Output in plain text or SRT subtitle format
- Command-line interface for easy usage

## Requirements

To use these scripts, you need:

```
pip install nvidia-nemo[asr]
pip install librosa soundfile numpy
```

Additional requirements:
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Audio files in WAV format (16kHz sample rate recommended)

## Model Description

### NVIDIA Parakeet TDT (Time Domain Transformer)

The `nvidia/parakeet-tdt-0.6b-v2` model is a state-of-the-art ASR model from NVIDIA's Parakeet family. It is based on the Time Domain Transformer architecture and offers:

- **High Accuracy**: Trained on large-scale datasets for robust speech recognition
- **Timestamp Support**: Provides precise timing information at character, word, and segment levels
- **Multilingual Capabilities**: Supports English and other languages
- **Efficient Processing**: Optimized for GPU acceleration

The model is accessed through NVIDIA's NeMo framework, which automatically downloads and caches the model weights when first used.

### Model Specifications

- **Architecture**: Time Domain Transformer (TDT)
- **Size**: 0.6 billion parameters
- **Version**: v2
- **Input**: Audio waveform (16kHz recommended)
- **Output**: Transcribed text with optional timestamps
- **Supported Languages**: Primarily English, with some multilingual capabilities

## Scripts

### Basic Demo (`asr_demo.py`)

This script demonstrates the basic usage of the Parakeet TDT model for transcribing audio files with timestamps.

Key features:
- Simple model loading and inference
- Extraction of timestamps at character, word, and segment levels
- Display of segment-level timestamps with corresponding text

Example output:
```
0.0s - 2.5s : Hello, this is a test.
2.5s - 5.0s : Welcome to the demo.
```

### Long Audio Processing (`asr_with_split.py`)

This script extends the basic functionality to handle long audio files by:
1. Splitting audio into overlapping segments
2. Transcribing each segment individually
3. Merging the results with proper timestamp alignment

Key functions:
- `load_audio()`: Loads and resamples audio files
- `split_audio_with_overlap()`: Divides long audio into manageable segments
- `merge_transcriptions_with_timestamps()`: Combines segment transcriptions
- `transcribe_long_audio()`: Main function for processing long audio
- `transcribe_text()`: Returns plain text transcription
- `transcribe_srt()`: Returns transcription in SRT subtitle format

### Command Line Interface (`asr_with_split_cmd.py`)

This script provides a user-friendly command-line interface to the functionality in `asr_with_split.py`.

Features:
- Simple command-line arguments
- Support for both text and SRT output formats
- Customizable segment duration and overlap
- Automatic output file naming with timestamps

## Usage Examples

### Basic Transcription

```python
import nemo.collections.asr as nemo_asr

# Load the model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# Simple transcription
output = asr_model.transcribe(['audio.wav'])
print(output[0].text)

# Transcription with timestamps
output = asr_model.transcribe(['audio.wav'], timestamps=True)
word_timestamps = output[0].timestamp['word']
segment_timestamps = output[0].timestamp['segment']
char_timestamps = output[0].timestamp['char']

# Print segments with timestamps
for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")
```

### Long Audio Transcription

```python
from asr_with_split import transcribe_text, transcribe_srt
import nemo.collections.asr as nemo_asr

# Load the model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# Get plain text transcription
text_result = transcribe_text(
    'long_audio.wav',
    asr_model,
    max_duration=60,  # Maximum segment duration in seconds
    overlap_duration=10  # Overlap between segments in seconds
)
print(text_result)

# Get SRT format transcription
srt_result = transcribe_srt(
    'long_audio.wav',
    asr_model,
    max_duration=60,
    overlap_duration=10
)
print(srt_result)
```

### Command Line Usage

```bash
# Transcribe to plain text
python asr_with_split_cmd.py --audio_file input.wav --type txt

# Transcribe to SRT format
python asr_with_split_cmd.py --audio_file input.wav --type srt

# Customize segment duration and overlap
python asr_with_split_cmd.py --audio_file input.wav --type srt --max_duration 120 --overlap_duration 15
```

## Output Formats

### Plain Text

The plain text output is a simple string containing the transcribed content.

Example:
```
Hello, this is a test. Welcome to the demo. This is an example of the transcription output in plain text format.
```

### SRT Format

The SRT format includes sequence numbers, timestamps, and text segments.

Example:
```
1
00:00:00,000 --> 00:00:02,500
Hello, this is a test.

2
00:00:02,500 --> 00:00:05,000
Welcome to the demo.

3
00:00:05,000 --> 00:00:10,200
This is an example of the transcription output in SRT format.
```

## Advanced Configuration

### Segment Duration and Overlap

For long audio files, you can adjust:

- `max_duration`: Maximum duration of each segment in seconds (default: 60)
  - Increase for better context but higher memory usage
  - Decrease if experiencing out-of-memory errors
  
- `overlap_duration`: Overlap between segments in seconds (default: 10)
  - Increase for better continuity between segments
  - Decrease for faster processing (but may affect quality at segment boundaries)

### Output Directory

By default, transcription results are saved in a `result` directory with filenames that include:
- Original audio filename
- Timestamp of transcription
- Format extension (.txt or .srt)

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Decrease `max_duration` to process smaller audio segments
   - Use a GPU with more memory

2. **Slow Processing**
   - Ensure you're using GPU acceleration
   - Decrease `overlap_duration` slightly (but keep at least 5 seconds)
   - Process shorter audio files or split manually

3. **Model Download Issues**
   - Ensure you have a stable internet connection
   - Check disk space for model cache

4. **Audio Format Problems**
   - Convert audio to 16kHz WAV format for best results
   - Use tools like ffmpeg: `ffmpeg -i input.mp3 -ar 16000 output.wav`
