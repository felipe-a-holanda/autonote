"""
Speaker diarization script using pyannote.audio
Identifies who spoke when in an audio file
"""
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from autonote.logger import log_info, log_error

try:
    from pyannote.audio import Pipeline
    import torch
    import torchaudio
except ImportError:
    pass

TARGET_SAMPLE_RATE = 16000

def ensure_16khz_wav(audio_file: str) -> tuple[str, bool]:
    needs_conversion = True
    if audio_file.endswith(".wav"):
        try:
            import wave
            with wave.open(audio_file, "rb") as wf:
                if wf.getframerate() == TARGET_SAMPLE_RATE and wf.getnchannels() == 1:
                    needs_conversion = False
        except Exception:
            pass

    if not needs_conversion:
        return audio_file, False

    log_info("Converting audio to 16kHz mono WAV for diarization...")
    waveform, sr = torchaudio.load(audio_file)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, TARGET_SAMPLE_RATE)
    tmp.close()
    return tmp.name, True

def diarize_audio(audio_file: str, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None, hf_token: str = None) -> dict:
    log_info("Loading diarization model...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        log_error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load pyannote model. Make sure HF_TOKEN is set.")

    log_info(f"Analyzing speakers in: {audio_file}")
    processed_file, is_temp = ensure_16khz_wav(audio_file)

    params = {}
    if num_speakers is not None: params["num_speakers"] = num_speakers
    if min_speakers is not None: params["min_speakers"] = min_speakers
    if max_speakers is not None: params["max_speakers"] = max_speakers

    try:
        diarization_output = pipeline(processed_file, **params)
    finally:
        if is_temp:
            Path(processed_file).unlink(missing_ok=True)

    segments = []
    speaker_stats = {}
    diarization = diarization_output.speaker_diarization

    for turn, track, speaker in diarization.itertracks(yield_label=True):
        seg = {
            "speaker_id": speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.end - turn.start
        }
        segments.append(seg)

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"total_time": 0.0, "segment_count": 0}
        speaker_stats[speaker]["total_time"] += seg["duration"]
        speaker_stats[speaker]["segment_count"] += 1

    duration = max(seg["end"] for seg in segments) if segments else 0
    result = {
        "version": "1.0",
        "audio_file": str(Path(audio_file).name),
        "audio_path": str(Path(audio_file).absolute()),
        "duration": duration,
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "num_speakers": len(speaker_stats),
        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "segments": segments,
        "speaker_stats": speaker_stats,
        "labels": {},
        "source": "local_diarization"
    }

    log_info(f"Diarization complete! Detected {len(speaker_stats)} speaker(s) across {len(segments)} segments.")
    for speaker, stats in sorted(speaker_stats.items()):
        log_info(f"  {speaker}: {stats['total_time']:.1f}s ({stats['segment_count']} segments)")

    return result

def save_diarization(result: dict, output_file: str):
    output_path = Path(output_file)
    output_path.write_text(json.dumps(result, indent=2))
    log_info(f"Diarization saved to: {output_path}")

def run_diarize(audio_file: str, speakers: int = None, min_speakers: int = None, max_speakers: int = None, hf_token: str = None, output_file: str = None):
    hf_token = hf_token or os.getenv("HF_TOKEN")
    if not hf_token:
        log_error("Warning: No HuggingFace token provided natively. Required for first download.")
        
    if not output_file:
        audio_path = Path(audio_file)
        output_file = str(audio_path.parent / f"{audio_path.stem}_speakers.json")
        
    result = diarize_audio(audio_file, num_speakers=speakers, min_speakers=min_speakers, max_speakers=max_speakers, hf_token=hf_token)
    save_diarization(result, output_file)
    return output_file
