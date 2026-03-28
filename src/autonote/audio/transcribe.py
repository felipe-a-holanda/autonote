#!/usr/bin/env python3

"""
Transcription script using openai-whisper
Transcribes audio files to text with timestamps
"""

import argparse
import json
import os
import sys
from pathlib import Path
from autonote.logger import log_info, log_error

# CTranslate2 (used by faster_whisper) needs CUDA libs that may be installed
# as pip packages (e.g. nvidia-cublas-cu12) rather than system-wide.
# Add their lib dirs to LD_LIBRARY_PATH before importing.
def _setup_cuda_lib_paths():
    site_packages = Path(__file__).resolve().parent.parent.parent.parent
    nvidia_dir = site_packages / "nvidia"
    if nvidia_dir.is_dir():
        lib_dirs = [str(p) for p in nvidia_dir.glob("*/lib") if p.is_dir()]
        if lib_dirs:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))

_setup_cuda_lib_paths()

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    raise ImportError(
        "faster-whisper is required for transcription. "
        "Install it with: uv add faster-whisper"
    ) from e



def transcribe_audio(
    audio_file: str,
    model_size: str = "base",
    device: str = "auto",
    language: str = None,
    output_format: str = "txt"
) -> dict:
    """
    Transcribe an audio file using openai-whisper

    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v3, turbo)
        device: Device to use (cpu, cuda, auto)
        language: Language code (None for auto-detect)
        output_format: Output format (txt, json, srt, vtt)

    Returns:
        dict with transcription results
    """
    # Determine device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    log_info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    log_info(f"Transcribing: {audio_file}")
    segments_gen, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
    )

    log_info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Materialise the generator and reformat to our expected structure
    all_segments = []
    full_text_parts = []
    for segment in segments_gen:
        text = segment.text.strip()
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })
        full_text_parts.append(text)

    return {
        "language": info.language,
        "segments": all_segments,
        "text": " ".join(full_text_parts),
    }


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_transcription(result: dict, output_file: str, format: str):
    """Save transcription in specified format"""
    output_path = Path(output_file)

    if format == "txt":
        output_path.write_text(result["text"])

    elif format == "json":
        output_path.write_text(json.dumps(result, indent=2))

    elif format == "srt":
        lines = []
        for i, seg in enumerate(result["segments"], 1):
            lines.append(str(i))
            lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        output_path.write_text("\n".join(lines))

    elif format == "vtt":
        lines = ["WEBVTT", ""]
        for seg in result["segments"]:
            lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        output_path.write_text("\n".join(lines))

    log_info(f"Transcription saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using openai-whisper")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-m", "--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                       help="Whisper model size (default: base)")
    parser.add_argument("-d", "--device", default="auto",
                       choices=["cpu", "cuda", "auto"],
                       help="Device to use (default: auto)")
    parser.add_argument("-l", "--language", default=None,
                       help="Language code (default: auto-detect)")
    parser.add_argument("-f", "--format", default="txt",
                       choices=["txt", "json", "srt", "vtt"],
                       help="Output format (default: txt)")
    parser.add_argument("-o", "--output",
                       help="Output file (default: audio_file.txt)")

    args = parser.parse_args()

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        audio_path = Path(args.audio_file)
        output_file = audio_path.with_suffix(f".{args.format}")

    # Transcribe
    try:
        result = transcribe_audio(
            args.audio_file,
            model_size=args.model,
            device=args.device,
            language=args.language,
            output_format=args.format
        )

        # Save results
        save_transcription(result, output_file, args.format)

        # Print summary
        print(f"\nTranscription complete!", file=sys.stderr)
        print(f"Segments: {len(result['segments'])}", file=sys.stderr)

    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
