"""
Apply speaker labels to create final transcript
Takes labeled diarization data and outputs formatted text with speaker names
"""
import json
from pathlib import Path
from autonote.logger import log_info, log_error

def load_labeled_json(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    data = json.loads(path.read_text())
    if "segments" not in data or not data["segments"]:
        raise ValueError("No segments found in file")
    return data

def get_speaker_name(speaker_id: str, labels: dict) -> str:
    if speaker_id in labels:
        return labels[speaker_id].get("name", speaker_id)
    return speaker_id

def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def apply_labels_to_transcript(data: dict, format: str = "txt") -> str:
    segments = data["segments"]
    labels = data.get("labels", {})

    if format in ["txt", "md"]:
        lines = []
        current_speaker = None
        for seg in segments:
            speaker_id = seg["speaker_id"]
            speaker_name = get_speaker_name(speaker_id, labels)
            text = seg["text"].strip()
            if not text: continue
            if speaker_name != current_speaker:
                lines.append(f"\n**{speaker_name}**: {text}" if format == "md" else f"\n[{speaker_name}] {text}")
                current_speaker = speaker_name
            else:
                lines.append(text)
        return " ".join(lines).strip()
    elif format == "json":
        output_segments = []
        for seg in segments:
            speaker_id = seg["speaker_id"]
            speaker_name = get_speaker_name(speaker_id, labels)
            output_segments.append({
                "speaker": speaker_name, "speaker_id": speaker_id,
                "start": seg["start"], "end": seg["end"], "text": seg["text"]
            })
        return json.dumps({
            "audio_file": data.get("audio_file"), "duration": data.get("duration"),
            "language": data.get("language"), "num_speakers": data.get("num_speakers"),
            "segments": output_segments, "labels": labels
        }, indent=2)
    elif format == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            speaker_name = get_speaker_name(seg["speaker_id"], labels)
            text = seg["text"].strip()
            if not text: continue
            lines.append(str(i))
            lines.append(f"{format_srt_timestamp(seg['start'])} --> {format_srt_timestamp(seg['end'])}")
            lines.append(f"[{speaker_name}] {text}")
            lines.append("")
        return "\n".join(lines)
    elif format == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            speaker_name = get_speaker_name(seg["speaker_id"], labels)
            text = seg["text"].strip()
            if not text: continue
            lines.append(f"{format_vtt_timestamp(seg['start'])} --> {format_vtt_timestamp(seg['end'])}")
            lines.append(f"[{speaker_name}] {text}")
            lines.append("")
        return "\n".join(lines)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_transcript(content: str, output_file: str):
    output_path = Path(output_file)
    output_path.write_text(content)
    log_info(f"Transcript saved to: {output_path}")

def run_apply_labels(labeled_file: str, format: str = "txt", output_file: str = None):
    data = load_labeled_json(labeled_file)
    if not output_file:
        audio_file = data.get("audio_file", "transcript")
        base_name = Path(audio_file).stem
        labeled_path = Path(labeled_file)
        import re as _re
        base_stem = _re.sub(r"_speakers_labeled$", "", labeled_path.stem)
        base_stem = _re.sub(r"_labeled$", "", base_stem)
        output_file = str(labeled_path.parent / f"{base_name}.{format}")
        
    log_info("Applying labels to transcript...")
    transcript = apply_labels_to_transcript(data, format=format)
    save_transcript(transcript, output_file)
    log_info("Transcript complete!")
    return output_file
