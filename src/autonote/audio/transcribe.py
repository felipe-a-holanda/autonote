#!/usr/bin/env python3

"""
Transcription script supporting both local (faster-whisper) and external APIs
Transcribes audio files to text with timestamps
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from autonote.logger import log_info, log_error


def transcribe_audio(
    audio_file: str,
    model_size: str = "base",
    device: str = "auto",
    language: Optional[str] = None,
    output_format: str = "txt",
    provider: Optional[str] = None,
    api_key: Optional[str] = None
) -> dict:
    """
    Transcribe an audio file using configured provider (local or external API)

    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (for local provider)
        device: Device to use (for local provider: cpu, cuda, auto)
        language: Language code (None for auto-detect)
        output_format: Output format (txt, json, srt, vtt) - for compatibility
        provider: Transcription provider (local, assemblyai, or None for config default)
        api_key: API key for external providers (or None to use config)

    Returns:
        dict with transcription results containing:
            - language: detected or specified language
            - segments: list of dicts with start, end, text
            - text: full transcription text
    """
    from autonote.audio.transcription_providers import create_transcription_provider
    from autonote.config import config
    
    if provider is None:
        provider = config.get("TRANSCRIPTION_PROVIDER", "local")
    
    if api_key is None and provider == "assemblyai":
        api_key = config.get("ASSEMBLYAI_API_KEY", "")
    
    transcription_provider = create_transcription_provider(
        provider=provider,
        model_size=model_size,
        device=device,
        api_key=api_key
    )
    
    log_info(f"Using transcription provider: {transcription_provider.get_provider_name()}")
    result = transcription_provider.transcribe(audio_file, language=language)
    
    return result


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
    parser = argparse.ArgumentParser(description="Transcribe audio using local or external transcription providers")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-m", "--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                       help="Whisper model size for local provider (default: base)")
    parser.add_argument("-d", "--device", default="auto",
                       choices=["cpu", "cuda", "auto"],
                       help="Device to use for local provider (default: auto)")
    parser.add_argument("-l", "--language", default=None,
                       help="Language code (default: auto-detect)")
    parser.add_argument("-f", "--format", default="txt",
                       choices=["txt", "json", "srt", "vtt"],
                       help="Output format (default: txt)")
    parser.add_argument("-o", "--output",
                       help="Output file (default: audio_file.txt)")
    parser.add_argument("-p", "--provider", default=None,
                       choices=["local", "assemblyai"],
                       help="Transcription provider (default: from config or 'local')")
    parser.add_argument("-k", "--api-key", default=None,
                       help="API key for external providers (default: from config)")

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
            output_format=args.format,
            provider=args.provider,
            api_key=args.api_key
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
