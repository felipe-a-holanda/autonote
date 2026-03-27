"""
Merge diarization and transcription results
Combines speaker segments with transcribed text by matching timestamps
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from autonote.logger import log_info, log_error

def load_json(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return json.loads(path.read_text())

def find_speaker_at_time(timestamp: float, speaker_segments: list) -> str:
    for segment in speaker_segments:
        if segment["start"] <= timestamp <= segment["end"]:
            return segment["speaker_id"]
    
    min_distance = float('inf')
    closest_speaker = "UNKNOWN"
    for segment in speaker_segments:
        if timestamp < segment["start"]: distance = segment["start"] - timestamp
        elif timestamp > segment["end"]: distance = timestamp - segment["end"]
        else: distance = 0
        if distance < min_distance:
            min_distance = distance
            closest_speaker = segment["speaker_id"]
    return closest_speaker

def merge_diarization_transcription(diarization: dict, transcription: dict) -> dict:
    log_info("Merging diarization with transcription...")
    speaker_segments = diarization["segments"]
    transcription_segments = transcription["segments"]
    merged_segments = []

    for trans_seg in transcription_segments:
        timestamp = trans_seg["start"]
        speaker_id = find_speaker_at_time(timestamp, speaker_segments)
        merged_seg = {
            "speaker_id": speaker_id,
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "text": trans_seg["text"]
        }
        merged_segments.append(merged_seg)

    speaker_stats = {}
    for seg in merged_segments:
        speaker = seg["speaker_id"]
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"total_time": 0.0, "segment_count": 0, "word_count": 0}
        speaker_stats[speaker]["total_time"] += (seg["end"] - seg["start"])
        speaker_stats[speaker]["segment_count"] += 1
        speaker_stats[speaker]["word_count"] += len(seg["text"].split())

    result = {
        "version": "1.0",
        "audio_file": diarization.get("audio_file"),
        "audio_path": diarization.get("audio_path"),
        "duration": transcription.get("duration", diarization.get("duration")),
        "language": transcription.get("language"),
        "diarization_model": diarization.get("diarization_model"),
        "transcription_model": "faster-whisper",
        "num_speakers": diarization.get("num_speakers"),
        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "segments": merged_segments,
        "speaker_stats": speaker_stats,
        "labels": {},
        "source": "merged"
    }

    log_info(f"Merged {len(merged_segments)} segments")
    log_info(f"Speakers detected: {len(speaker_stats)}")
    return result

def save_merged(result: dict, output_file: str):
    output_path = Path(output_file)
    output_path.write_text(json.dumps(result, indent=2))
    log_info(f"Merged data saved to: {output_path}")

def run_merge(diarization_file: str, transcription_file: str, output_file: str = None):
    diarization = load_json(diarization_file)
    transcription = load_json(transcription_file)
    
    if not output_file:
        trans_path = Path(transcription_file)
        output_file = str(trans_path.parent / f"{trans_path.stem}_diarized.json")
        
    result = merge_diarization_transcription(diarization, transcription)
    save_merged(result, output_file)
    return output_file
