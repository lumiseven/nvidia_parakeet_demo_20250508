#!/usr/bin/env python3
"""
Command-line script for audio transcription using NeMo Parakeet ASR model.
This script uses the functions from asr_with_split.py to transcribe audio files
and save the results in either plain text or SRT format.
"""

import argparse
import os
import sys
import datetime
import nemo.collections.asr as nemo_asr
from asr_with_split import transcribe_text, transcribe_srt

def main():
    """
    Main function to parse command-line arguments and run transcription.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using NeMo Parakeet ASR model"
    )
    
    # Add arguments
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["txt", "srt"], 
        default="txt",
        help="Output format type (txt or srt), default: txt"
    )
    
    parser.add_argument(
        "--audio_file", 
        type=str, 
        required=True,
        help="Path to the WAV audio file to transcribe"
    )
    
    parser.add_argument(
        "--max_duration", 
        type=int, 
        default=180,
        help="Maximum duration of each segment in seconds, default: 180"
    )
    
    parser.add_argument(
        "--overlap_duration", 
        type=int, 
        default=10,
        help="Overlap duration between segments in seconds, default: 10"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    if not args.audio_file.lower().endswith('.wav'):
        print(f"Warning: File '{args.audio_file}' may not be a WAV file. Continuing anyway...")
    
    # Load the ASR model (fixed value)
    print("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    print("Model loaded successfully.")
    
    # Create result directory if it doesn't exist
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Perform transcription based on type and save to file
    if args.type == "txt":
        print(f"Transcribing '{args.audio_file}' to plain text...")
        # Call the function to get the text result
        text_result = transcribe_text(
            args.audio_file,
            asr_model,
            max_duration=args.max_duration,
            overlap_duration=args.overlap_duration
        )
        
        # Generate the output file path
        output_file = f"{result_dir}/transcription_{os.path.basename(args.audio_file)}_{timestamp}.txt"
        
        # Write result to file
        with open(output_file, "w") as f:
            f.write(text_result)
            
    else:  # args.type == "srt"
        print(f"Transcribing '{args.audio_file}' to SRT format...")
        # Call the function to get the SRT result
        srt_result = transcribe_srt(
            args.audio_file,
            asr_model,
            max_duration=args.max_duration,
            overlap_duration=args.overlap_duration
        )
        
        # Generate the output file path
        output_file = f"{result_dir}/transcription_{os.path.basename(args.audio_file)}_{timestamp}.srt"
        
        # Write result to file
        with open(output_file, "w") as f:
            f.write(srt_result)
    
    # Print the output file path to the terminal
    print(f"\nOutput file path: {output_file}")

if __name__ == "__main__":
    main()
