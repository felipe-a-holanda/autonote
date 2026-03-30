import sys
import argparse
from autonote.config import config
# Will import modules as we refactor them

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonote - Complete meeting recording, transcription, and summarization workflow",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # clean
    clean_parser = subparsers.add_parser("clean", help="Delete ALL audio files (WAV and MP3)")
    clean_parser.add_argument("file", help="Audio file to clean around")
    
    # list
    list_parser = subparsers.add_parser("list", help="List all recordings")

    # record
    record_parser = subparsers.add_parser("record", help="Start recording a meeting")
    record_parser.add_argument("-d", "--duration", type=int, help="Recording duration in seconds")
    record_parser.add_argument("-t", "--title", help="Meeting title")
    record_parser.add_argument("-o", "--output", help="Output file path")
    
    # compress
    compress_parser = subparsers.add_parser("compress", help="Compress WAV to MP3")
    compress_parser.add_argument("file", help="Audio file to compress")
    compress_parser.add_argument("--bitrate", default="128k", help="MP3 bitrate")
    compress_parser.add_argument("--delete-wav", action="store_true", help="Delete original WAV")

    # transcribe
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    transcribe_parser.add_argument("file", help="Audio file to transcribe")
    transcribe_parser.add_argument("-m", "--model", help="Whisper model size")
    transcribe_parser.add_argument("-l", "--language", help="Language code")
    transcribe_parser.add_argument("-f", "--format", default="txt", choices=["txt", "json", "srt", "vtt"])
    transcribe_parser.add_argument("-o", "--output", help="Output file")
    transcribe_parser.add_argument("-p", "--provider", choices=["local", "assemblyai"], help="Transcription provider")
    transcribe_parser.add_argument("-k", "--api-key", help="API key for external providers")
    
    # reformat
    reformat_parser = subparsers.add_parser("reformat", help="Reformat transcript using LLM")
    reformat_parser.add_argument("file", help="Transcript file")
    reformat_parser.add_argument("-m", "--model", help="LLM model or preset (fast, smart, cheap, local)")
    reformat_parser.add_argument("-u", "--ollama-url", help="Ollama API URL")
    reformat_parser.add_argument("-o", "--output", help="Output file")

    # summarize
    summarize_parser = subparsers.add_parser("summarize", help="Summarize transcript using LLM")
    summarize_parser.add_argument("file", help="Transcript file")
    summarize_parser.add_argument("-m", "--model", help="LLM model or preset (fast, smart, cheap, local)")
    summarize_parser.add_argument("-u", "--ollama-url", help="Ollama API URL")
    summarize_parser.add_argument("-f", "--format", default="md", choices=["txt", "md", "json"])
    summarize_parser.add_argument("-o", "--output", help="Output file")
    summarize_parser.add_argument("--no-action-items", action="store_true", help="Skip action items")

    # diarize
    diarize_parser = subparsers.add_parser("diarize", help="Perform speaker diarization")
    diarize_parser.add_argument("file", help="Audio file")
    diarize_parser.add_argument("-s", "--speakers", type=int, help="Expected number of speakers")
    diarize_parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    diarize_parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    diarize_parser.add_argument("--hf-token", help="HuggingFace API token")
    diarize_parser.add_argument("-o", "--output", help="Output file")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge diarization and transcription")
    merge_parser.add_argument("diarization_file", help="Diarization JSON file")
    merge_parser.add_argument("transcription_file", help="Transcription JSON file")
    merge_parser.add_argument("-o", "--output", help="Output file")

    # label
    label_parser = subparsers.add_parser("label", help="Interactively label speakers")
    label_parser.add_argument("file", help="Diarized JSON file")
    label_parser.add_argument("-o", "--output", help="Output file")
    label_parser.add_argument("--non-interactive", action="store_true", help="Skip prompts")

    # apply-labels
    apply_parser = subparsers.add_parser("apply-labels", help="Apply speaker labels to transcript")
    apply_parser.add_argument("file", help="Labeled JSON file")
    apply_parser.add_argument("-f", "--format", default="txt", choices=["txt", "md", "json", "srt", "vtt"])
    apply_parser.add_argument("-o", "--output", help="Output file")

    # extract-metadata
    extract_parser = subparsers.add_parser("extract-metadata", help="Extract structural metadata via LLM")
    extract_parser.add_argument("file", help="Transcript text or markdown file")
    extract_parser.add_argument("-m", "--model", help="LLM model or preset (fast, smart, cheap, local)")
    extract_parser.add_argument("-u", "--ollama-url", help="Ollama API URL")
    extract_parser.add_argument("-o", "--output", help="Output JSON file")

    # frontmatter
    fm_parser = subparsers.add_parser("frontmatter", help="Add or update YAML frontmatter")
    fm_parser.add_argument("file", help="Markdown file to update")
    fm_parser.add_argument("--kind", choices=["formatted", "summary"], default="formatted", help="Type of file")
    fm_parser.add_argument("--metadata", help="Path to _metadata.json")
    fm_parser.add_argument("--extracted", help="Path to _extracted_metadata.json")

    # wikilink
    wiki_parser = subparsers.add_parser("wikilink", help="Inject [[wikilinks]] into a markdown file")
    wiki_parser.add_argument("file", help="Markdown file to process")
    wiki_parser.add_argument("--entities", required=True, help="Path to entities.yml")

    # update-index
    index_parser = subparsers.add_parser("update-index", help="Append meeting to Meetings.md index")
    index_parser.add_argument("summary_file", help="Path to _summary.md file")
    index_parser.add_argument("--index", required=True, help="Path to Meetings.md index file")

    # obsidian
    obsidian_parser = subparsers.add_parser("obsidian", help="Run Obsidian post-processing on an existing meeting (vault copy, frontmatter, index)")
    obsidian_parser.add_argument("file", help="Audio file (WAV or MP3) of the meeting")

    # reprocess
    reprocess_parser = subparsers.add_parser("reprocess", help="Selectively re-run pipeline steps on an existing meeting or batch of meetings")
    reprocess_parser.add_argument("file", nargs="?", help="Audio file (WAV or MP3) of a single meeting")
    reprocess_parser.add_argument("--since", metavar="DATE", help="Batch: reprocess all meetings on or after this date (YYYY-MM-DD or YYYYMMDD)")
    reprocess_parser.add_argument("--until", metavar="DATE", help="Batch: reprocess all meetings on or before this date (YYYY-MM-DD or YYYYMMDD)")
    reprocess_parser.add_argument("--all", dest="all_meetings", action="store_true", help="Batch: reprocess all meetings in the recordings directory")
    reprocess_parser.add_argument("--reformat", action="store_true", help="Re-run LLM reformatting")
    reprocess_parser.add_argument("--summarize", action="store_true", help="Re-run LLM summarization")
    reprocess_parser.add_argument("--obsidian", action="store_true", help="Re-run Obsidian post-processing (vault copy, frontmatter, index)")
    reprocess_parser.add_argument("-m", "--model", help="LLM model or preset")
    reprocess_parser.add_argument("--quiet", action="store_true", help="Hide logs during batch reprocessing (default: verbose)")

    # process
    process_parser = subparsers.add_parser("process", help="Process existing recording")
    process_parser.add_argument("file", help="Audio file")
    process_parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    process_parser.add_argument("-s", "--speakers", type=int, help="Number of speakers")
    process_parser.add_argument("--no-reformat", action="store_true", help="Skip LLM reformatting")
    process_parser.add_argument("--no-compress", action="store_true", help="Skip MP3 compression")
    process_parser.add_argument("--keep-wav", action="store_true", help="Keep WAV after compression")
    process_parser.add_argument("--clean", action="store_true", help="Delete all audio files")
    process_parser.add_argument("-p", "--provider", choices=["local", "assemblyai"], help="Transcription provider")
    process_parser.add_argument("-k", "--api-key", help="API key for external transcription providers")

    # process-last
    process_last_parser = subparsers.add_parser("process-last", help="Process most recent recording")
    process_last_parser.add_argument("--diarize", action="store_true")
    process_last_parser.add_argument("-s", "--speakers", type=int)
    process_last_parser.add_argument("--no-reformat", action="store_true")
    process_last_parser.add_argument("--no-compress", action="store_true")
    process_last_parser.add_argument("--keep-wav", action="store_true")
    process_last_parser.add_argument("--clean", action="store_true")
    process_last_parser.add_argument("-p", "--provider", choices=["local", "assemblyai"], help="Transcription provider")
    process_last_parser.add_argument("-k", "--api-key", help="API key for external transcription providers")

    # full
    full_parser = subparsers.add_parser("full", help="Record and process")
    full_parser.add_argument("-d", "--duration", type=int, help="Recording duration")
    full_parser.add_argument("-t", "--title", help="Meeting title")
    full_parser.add_argument("--diarize", action="store_true")
    full_parser.add_argument("-s", "--speakers", type=int)
    full_parser.add_argument("--no-reformat", action="store_true")
    full_parser.add_argument("--no-compress", action="store_true")
    full_parser.add_argument("--keep-wav", action="store_true")
    full_parser.add_argument("--clean", action="store_true")
    full_parser.add_argument("-p", "--provider", choices=["local", "assemblyai"], help="Transcription provider")
    full_parser.add_argument("-k", "--api-key", help="API key for external transcription providers")

    return parser

def cmd_list(args):
    import os
    recdir = config["RECORDINGS_DIR"]
    if not os.path.exists(recdir):
        print(f"Recordings directory not found: {recdir}")
        return
    for root, dirs, files in os.walk(recdir):
        for d in dirs:
            print(f"Directory: {os.path.join(root, d)}")
        for f in files:
            print(f"  {f}")
        break  # Just top layer or whatever

def cmd_clean(args):
    import os
    audio_file = args.file
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Deleted: {audio_file}")
    
    mp3_file = audio_file.rsplit(".", 1)[0] + ".mp3"
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"Deleted: {mp3_file}")

def cmd_record(args):
    from autonote.audio.record import record_audio
    record_audio(args.duration, args.output, args.title)

def cmd_compress(args):
    from autonote.audio.compress import compress_audio
    compress_audio(args.file, args.bitrate, args.delete_wav)

def cmd_transcribe(args):
    from autonote.audio.transcribe import transcribe_audio, save_transcription
    from pathlib import Path
    from autonote.config import config
    
    model = args.model or config.get("WHISPER_MODEL")
    lang = args.language or config.get("WHISPER_LANGUAGE")
    provider = getattr(args, 'provider', None)
    api_key = getattr(args, 'api_key', None)
    
    if args.output:
        out_file = args.output
    else:
        out_file = str(Path(args.file).with_suffix(f".{args.format}"))
        
    result = transcribe_audio(args.file, model_size=model, language=lang, output_format=args.format, 
                             provider=provider, api_key=api_key)
    save_transcription(result, out_file, args.format)

def cmd_reformat(args):
    from autonote.audio.reformat import run_reformat
    run_reformat(args.file, model=args.model, ollama_url=args.ollama_url, output_file=args.output)

def cmd_summarize(args):
    from autonote.audio.summarize import run_summarize
    run_summarize(args.file, model=args.model, ollama_url=args.ollama_url, format=args.format, 
                  output_file=args.output, skip_action_items=args.no_action_items)

def cmd_diarize(args):
    from autonote.audio.diarize import run_diarize
    run_diarize(args.file, speakers=args.speakers, min_speakers=args.min_speakers, 
                max_speakers=args.max_speakers, hf_token=args.hf_token, output_file=args.output)

def cmd_merge(args):
    from autonote.audio.merge_diarization import run_merge
    run_merge(args.diarization_file, args.transcription_file, output_file=args.output)

def cmd_label(args):
    from autonote.audio.label import run_label
    run_label(args.file, output_file=args.output, non_interactive=args.non_interactive)

def cmd_apply_labels(args):
    from autonote.audio.apply_labels import run_apply_labels
    run_apply_labels(args.file, format=args.format, output_file=args.output)

def cmd_extract_metadata(args):
    from autonote.obsidian.extract_metadata import run_extract_metadata
    run_extract_metadata(args.file, model=args.model, ollama_url=args.ollama_url, output=args.output)

def cmd_frontmatter(args):
    from autonote.obsidian.frontmatter import run_frontmatter
    run_frontmatter(args.file, kind=args.kind, metadata=args.metadata, extracted=args.extracted)

def cmd_wikilink(args):
    from autonote.obsidian.wikilink import run_wikilinks
    run_wikilinks(args.file, entities=args.entities)

def cmd_update_index(args):
    from autonote.obsidian.update_index import run_update_index
    run_update_index(args.summary_file, index=args.index)

def cmd_obsidian(args):
    from pathlib import Path
    from autonote.orchestrator import run_obsidian_postprocess
    from autonote.logger import log_error

    audio = Path(args.file)
    if not audio.exists():
        log_error(f"File not found: {audio}")
        return

    base = audio.stem
    meeting_dir = audio.parent
    formatted_file = str(meeting_dir / f"{base}_formatted.md")
    summary_file = str(meeting_dir / f"{base}_summary.md")

    if not Path(formatted_file).exists():
        formatted_file = ""
    if not Path(summary_file).exists():
        log_error(f"Summary file not found: {base}_summary.md — cannot run obsidian postprocess")
        return

    run_obsidian_postprocess(str(audio), formatted_file, summary_file)


def _discover_audio_files(since_date: str | None = None, until_date: str | None = None) -> list:
    """Return sorted list of audio file paths from the recordings directory, optionally filtered by date."""
    import glob
    from pathlib import Path

    recordings_dir = config.get("RECORDINGS_DIR")
    since = since_date.replace("-", "") if since_date else None  # normalize to YYYYMMDD
    until = until_date.replace("-", "") if until_date else None  # normalize to YYYYMMDD

    audio_files = []
    for date_dir in sorted(Path(recordings_dir).iterdir()):
        if not date_dir.is_dir() or not date_dir.name.isdigit() or len(date_dir.name) != 8:
            continue
        if since and date_dir.name < since:
            continue
        if until and date_dir.name > until:
            continue
        for meeting_dir in sorted(date_dir.iterdir()):
            if not meeting_dir.is_dir():
                continue
            for ext in ("wav", "mp3"):
                candidates = list(meeting_dir.glob(f"meeting_*.{ext}"))
                if candidates:
                    audio_files.append(str(candidates[0]))
                    break

    return audio_files


def _reprocess_single(audio_path: str, reformat: bool, summarize: bool, obsidian: bool, model: str | None) -> None:
    from pathlib import Path
    from autonote.logger import log_error, log_info

    audio = Path(audio_path)
    base = audio.stem
    meeting_dir = audio.parent
    formatted_file = str(meeting_dir / f"{base}_formatted.md")
    summary_file = str(meeting_dir / f"{base}_summary.md")

    if reformat:
        transcript_file = str(meeting_dir / f"{base}.txt")
        if not Path(transcript_file).exists():
            log_error(f"Skipping reformat — transcript not found: {transcript_file}")
        else:
            from autonote.audio.reformat import run_reformat
            log_info(f"Reformatting: {audio.name}")
            formatted_file = run_reformat(transcript_file, model=model)

    if summarize:
        source = formatted_file if Path(formatted_file).exists() else str(meeting_dir / f"{base}.txt")
        if not Path(source).exists():
            log_error(f"Skipping summarize — no source found for: {audio.name}")
        else:
            from autonote.audio.summarize import run_summarize
            log_info(f"Summarizing: {audio.name}")
            summary_file = run_summarize(source, model=model)

    if obsidian:
        from autonote.orchestrator import run_obsidian_postprocess
        if not Path(formatted_file).exists():
            formatted_file = ""
        if not Path(summary_file).exists():
            log_error(f"Skipping obsidian — summary not found: {summary_file}")
            return
        run_obsidian_postprocess(audio_path, formatted_file, summary_file)


def cmd_reprocess(args):
    from pathlib import Path
    from autonote.logger import log_error, log_info

    if not any([args.reformat, args.summarize, args.obsidian]):
        log_error("Specify at least one step: --reformat, --summarize, --obsidian")
        return

    if args.file:
        audio = Path(args.file)
        if not audio.exists():
            log_error(f"File not found: {audio}")
            return
        _reprocess_single(str(audio), args.reformat, args.summarize, args.obsidian, args.model)

    elif args.since or args.until or args.all_meetings:
        from tqdm import tqdm
        from autonote.logger import set_quiet
        audio_files = _discover_audio_files(since_date=args.since, until_date=args.until)
        if not audio_files:
            log_error("No meetings found matching the criteria.")
            return
        log_info(f"Found {len(audio_files)} meeting(s) to reprocess.")
        if args.quiet:
            set_quiet(True)
        try:
            bar = tqdm(audio_files, desc="Reprocessing", unit="meeting", dynamic_ncols=True)
            for audio_path in bar:
                name = Path(audio_path).parent.name
                bar.set_postfix_str(name)
                _reprocess_single(audio_path, args.reformat, args.summarize, args.obsidian, args.model)
        finally:
            set_quiet(False)

    else:
        log_error("Provide a file, --since DATE, or --all")

def cmd_process(args):
    from autonote.orchestrator import run_process
    run_process(args.file, diarize=args.diarize, speakers=args.speakers,
                no_reformat=args.no_reformat, no_compress=args.no_compress,
                keep_wav=args.keep_wav, clean=args.clean,
                provider=args.provider, api_key=args.api_key)

def cmd_process_last(args):
    from autonote.orchestrator import run_process_last
    run_process_last(diarize=args.diarize, speakers=args.speakers,
                     no_reformat=args.no_reformat, no_compress=args.no_compress,
                     keep_wav=args.keep_wav, clean=args.clean,
                     provider=args.provider, api_key=args.api_key)

def cmd_full(args):
    from autonote.orchestrator import run_full
    run_full(duration=args.duration, title=args.title, diarize=args.diarize,
             speakers=args.speakers, no_reformat=args.no_reformat,
             no_compress=args.no_compress, keep_wav=args.keep_wav, clean=args.clean,
             provider=args.provider, api_key=args.api_key)

def main():
    try:
        _dispatch()
    except KeyboardInterrupt:
        sys.exit(0)

def _dispatch():
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "transcribe":
        cmd_transcribe(args)
    elif args.command == "reformat":
        cmd_reformat(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "diarize":
        cmd_diarize(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "label":
        cmd_label(args)
    elif args.command == "apply-labels":
        cmd_apply_labels(args)
    elif args.command == "extract-metadata":
        cmd_extract_metadata(args)
    elif args.command == "frontmatter":
        cmd_frontmatter(args)
    elif args.command == "wikilink":
        cmd_wikilink(args)
    elif args.command == "update-index":
        cmd_update_index(args)
    elif args.command == "obsidian":
        cmd_obsidian(args)
    elif args.command == "reprocess":
        cmd_reprocess(args)
    elif args.command == "process":
        cmd_process(args)
    elif args.command == "process-last":
        cmd_process_last(args)
    elif args.command == "full":
        cmd_full(args)
    else:
        print(f"Command {args.command} is not implemented yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
