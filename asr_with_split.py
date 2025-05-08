import nemo.collections.asr as nemo_asr
import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Dict, Tuple, Optional

def load_audio(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio file and resample if necessary.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Audio data as numpy array
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def save_audio_segment(audio: np.ndarray, file_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio segment to file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sample_rate: Sample rate
    """
    sf.write(file_path, audio, sample_rate)

def split_audio_with_overlap(
    audio_path: str, 
    max_duration: int = 60, 
    overlap_duration: int = 10,
    sample_rate: int = 16000
) -> List[str]:
    """
    Split audio into segments with overlap to avoid OOM errors.
    
    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration of each segment in seconds
        overlap_duration: Overlap duration between segments in seconds
        sample_rate: Sample rate of audio
        
    Returns:
        List of paths to temporary audio segments
    """
    # Load audio
    audio = load_audio(audio_path, sample_rate)
    
    # Calculate segment sizes in samples
    max_samples = max_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Calculate number of segments
    audio_length = len(audio)
    if audio_length <= max_samples:
        # Audio is shorter than max_duration, no need to split
        return [audio_path]
    
    # Calculate step size and number of segments
    step_samples = max_samples - overlap_samples
    num_segments = int(np.ceil((audio_length - overlap_samples) / step_samples))
    
    # Create temporary directory if it doesn't exist
    temp_dir = "temp_audio_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split audio and save segments
    segment_paths = []
    for i in range(num_segments):
        start_sample = i * step_samples
        end_sample = min(start_sample + max_samples, audio_length)
        
        segment = audio[start_sample:end_sample]
        segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
        save_audio_segment(segment, segment_path, sample_rate)
        segment_paths.append(segment_path)
    
    return segment_paths

def merge_transcriptions_with_timestamps(
    segment_results: List[Dict], 
    overlap_duration: int = 10
) -> Dict:
    """
    Merge transcription results from multiple segments, removing duplicated text
    in overlapping regions based on timestamps.
    
    Args:
        segment_results: List of transcription results with timestamps
        overlap_duration: Overlap duration between segments in seconds
        
    Returns:
        Merged transcription result
    """
    if len(segment_results) == 1:
        return segment_results[0]
    
    # Initialize merged result
    merged_result = {
        "text": "",
        "timestamp": {
            "word": [],
            "segment": [],
            "char": []
        }
    }
    
    # Time offset for each segment
    time_offset = 0
    
    for i, result in enumerate(segment_results):
        # Skip empty results
        if not result.timestamp['segment']:
            continue
            
        # For the first segment, just add all segments
        if i == 0:
            for stamp in result.timestamp['segment']:
                merged_result["timestamp"]["segment"].append({
                    "start": stamp["start"],
                    "end": stamp["end"],
                    "segment": stamp["segment"]
                })
            
            # Add word and char timestamps
            merged_result["timestamp"]["word"].extend(result.timestamp['word'])
            merged_result["timestamp"]["char"].extend(result.timestamp['char'])
            
            # Update time offset for next segment
            last_segment = result.timestamp['segment'][-1]
            time_offset = last_segment["end"] - overlap_duration
            
        else:
            # For subsequent segments, we need to handle overlaps
            # Find segments that start after the overlap point
            for stamp in result.timestamp['segment']:
                # Adjust timestamps with offset
                adjusted_start = stamp["start"] + time_offset
                adjusted_end = stamp["end"] + time_offset
                
                # Only add segments that end after the overlap point
                if stamp["start"] >= overlap_duration:
                    merged_result["timestamp"]["segment"].append({
                        "start": adjusted_start,
                        "end": adjusted_end,
                        "segment": stamp["segment"]
                    })
            
            # Add word timestamps that are after the overlap
            for word in result.timestamp['word']:
                if word["start"] >= overlap_duration:
                    merged_result["timestamp"]["word"].append({
                        "start": word["start"] + time_offset,
                        "end": word["end"] + time_offset,
                        "word": word["word"]
                    })
            
            # Add char timestamps that are after the overlap
            for char in result.timestamp['char']:
                if char["start"] >= overlap_duration:
                    merged_result["timestamp"]["char"].append({
                        "start": char["start"] + time_offset,
                        "end": char["end"] + time_offset,
                        "char": char["char"]
                    })
            
            # Update time offset for next segment
            if result.timestamp['segment']:
                last_segment = result.timestamp['segment'][-1]
                time_offset += last_segment["end"] - overlap_duration
    
    # Reconstruct the full text from segments
    merged_text = " ".join([stamp["segment"] for stamp in merged_result["timestamp"]["segment"]])
    merged_result["text"] = merged_text
    
    return merged_result

def transcribe_long_audio(
    audio_path: str, 
    model: nemo_asr.models.ASRModel,
    max_duration: int = 60, 
    overlap_duration: int = 10,
    with_timestamps: bool = True
) -> Dict:
    """
    Transcribe long audio by splitting it into overlapping segments and
    merging the results.
    
    Args:
        audio_path: Path to audio file
        model: ASR model
        max_duration: Maximum duration of each segment in seconds
        overlap_duration: Overlap duration between segments in seconds
        with_timestamps: Whether to include timestamps in the output
        
    Returns:
        Transcription result
    """
    # Split audio into segments
    segment_paths = split_audio_with_overlap(
        audio_path, 
        max_duration=max_duration, 
        overlap_duration=overlap_duration
    )
    
    # If there's only one segment, just transcribe it directly
    if len(segment_paths) == 1 and segment_paths[0] == audio_path:
        return model.transcribe([audio_path], timestamps=with_timestamps)[0]
    
    # Transcribe each segment
    segment_results = []
    for segment_path in segment_paths:
        result = model.transcribe([segment_path], timestamps=with_timestamps)[0]
        segment_results.append(result)
    
    # Merge results
    if with_timestamps:
        merged_result = merge_transcriptions_with_timestamps(
            segment_results, 
            overlap_duration=overlap_duration
        )
    else:
        # For non-timestamp mode, just concatenate the text
        merged_text = " ".join([result.text for result in segment_results])
        merged_result = {"text": merged_text}
    
    # Clean up temporary files
    temp_dir = "temp_audio_segments"
    if os.path.exists(temp_dir):
        for segment_path in segment_paths:
            if os.path.exists(segment_path):
                os.remove(segment_path)
        os.rmdir(temp_dir)
    
    return merged_result

def format_time_srt(seconds: float) -> str:
    """
    Format time in seconds to SRT format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in SRT format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def transcribe_text(
    audio_file: str,
    model: nemo_asr.models.ASRModel = None,
    max_duration: int = 60,
    overlap_duration: int = 10
) -> str:
    """
    Transcribe audio file and return plain text result.
    
    Args:
        audio_file: Path to audio file
        model: ASR model (if None, will load the default model)
        max_duration: Maximum duration of each segment in seconds
        overlap_duration: Overlap duration between segments in seconds
        
    Returns:
        Transcribed text
    """
    # Load model if not provided
    if model is None:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    
    # Transcribe audio
    result = transcribe_long_audio(
        audio_file,
        model,
        max_duration=max_duration,
        overlap_duration=overlap_duration,
        with_timestamps=False
    )
    
    return result["text"]

def transcribe_srt(
    audio_file: str,
    model: nemo_asr.models.ASRModel = None,
    max_duration: int = 60,
    overlap_duration: int = 10
) -> str:
    """
    Transcribe audio file and return result in SRT format.
    
    Args:
        audio_file: Path to audio file
        model: ASR model (if None, will load the default model)
        max_duration: Maximum duration of each segment in seconds
        overlap_duration: Overlap duration between segments in seconds
        
    Returns:
        SRT formatted string
    """
    # Load model if not provided
    if model is None:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    
    # Transcribe audio with timestamps
    result = transcribe_long_audio(
        audio_file,
        model,
        max_duration=max_duration,
        overlap_duration=overlap_duration,
        with_timestamps=True
    )
    
    # Convert to SRT format
    srt_content = ""
    for i, stamp in enumerate(result["timestamp"]["segment"]):
        # Sequence number
        srt_content += f"{i+1}\n"
        
        # Timestamps
        start_time = format_time_srt(stamp["start"])
        end_time = format_time_srt(stamp["end"])
        srt_content += f"{start_time} --> {end_time}\n"
        
        # Text
        srt_content += f"{stamp['segment']}\n\n"
    
    return srt_content

# Example usage:
if __name__ == "__main__":
    # Load the ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    
    # Example audio file
    audio_file = 'output.wav'
    
    # Parameters for long audio transcription
    max_duration = 60  # Maximum duration of each segment in seconds
    overlap_duration = 10  # Overlap duration between segments in seconds
    
    # Get plain text transcription
    text_result = transcribe_text(
        audio_file,
        asr_model,
        max_duration=max_duration,
        overlap_duration=overlap_duration
    )
    print(f"Plain text transcription: {text_result[:100]}...")
    
    # Get SRT transcription
    srt_result = transcribe_srt(
        audio_file,
        asr_model,
        max_duration=max_duration,
        overlap_duration=overlap_duration
    )
    print(f"SRT transcription (first few lines):\n{srt_result.split('\\n\\n', 1)[0]}\n...")
