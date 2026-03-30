import os
import re
import shutil
import glob
from pathlib import Path
from autonote.logger import log_info, log_error, log_success
from autonote.config import config

from autonote.audio.record import run_record
from autonote.audio.transcribe import transcribe_audio, save_transcription
from autonote.audio.reformat import run_reformat
from autonote.audio.summarize import run_summarize
from autonote.audio.compress import compress_audio
from autonote.obsidian.extract_metadata import run_extract_metadata
from autonote.obsidian.frontmatter import run_frontmatter
from autonote.obsidian.wikilink import run_wikilinks
from autonote.obsidian.update_index import run_update_index

def _slugify(title: str, max_len: int = 60) -> str:
    s = title.strip().replace("/", "-").replace("\\", "-").replace("\x00", "")
    s = re.sub(r'[:\*\?"<>\|]', "", s)
    s = re.sub(r"[\s\-]+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0].strip()
    return s or "meeting"


def _resolve_vault_title(summary_file: str | None, time: str) -> str:
    """
    Title priority:
    1. User-provided recording name (frontmatter title)
    2. First non-boilerplate heading in summary body (LLM-inferred) as fallback
    3. Time as final fallback
    """
    BOILERPLATE = {
        "meeting summary", "action items", "summary",
        "overview", "meeting overview", "participants", 
        "key discussion points", "discussion points",
        "decisions made", "decisions", "key warnings", 
        "things to remember", "warnings", "next steps", 
        "follow-up", "notes", "main tasks", "tasks",
        "content analysis", "formatting guidelines",
        "additional notes", "key sections", "conclusion",
        "key takeaways", "takeaways", "per-person updates",
        "architecture clarification", "blockers", "agenda",
        "background", "context", "objectives", "goals"
    }
    if summary_file:
        path = Path(summary_file)
        if path.exists():
            from autonote.obsidian.frontmatter import parse_existing_frontmatter
            content = path.read_text(encoding="utf-8")
            fm, body = parse_existing_frontmatter(content)
            user_tag = (fm.get("title") or "").strip()
            inferred = ""
            for line in body.splitlines():
                m = re.match(r"^#{1,3} (.+)", line)
                if m:
                    heading = m.group(1).strip()
                    # Normalize heading: remove emojis, numbered prefixes, and extra whitespace
                    normalized = re.sub(r'^[\d\.\s]*', '', heading)  # Remove leading numbers and dots
                    # Remove all emoji characters including compound emojis and variation selectors
                    normalized = re.sub(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F]+', '', normalized)
                    normalized = normalized.strip()
                    if normalized.lower() not in BOILERPLATE:
                        inferred = heading
                        break
            if user_tag:
                return user_tag
            if inferred:
                return inferred
    return time


def _find_unique_vault_dest(vault_dir: Path, folder_name: str) -> Path:
    candidate = vault_dir / folder_name
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = vault_dir / f"{folder_name} ({n})"
        if not candidate.exists():
            return candidate
        n += 1


def _patch_transcript_wikilink(summary_vault_path: Path, new_stem: str) -> None:
    try:
        content = summary_vault_path.read_text(encoding="utf-8")
        patched = re.sub(
            r"^(transcript:\s*)'?\[\[.*?\]\]'?",
            f"transcript: '[[{new_stem}]]'",
            content,
            flags=re.MULTILINE,
        )
        if patched != content:
            summary_vault_path.write_text(patched, encoding="utf-8")
    except Exception as e:
        log_error(f"Warning: could not patch transcript wikilink: {e}")


def run_obsidian_postprocess(source_file: str, formatted_file: str, summary_file: str, extracted_meta: str | None = None):
    source_path = Path(source_file)
    base = source_path.stem.replace("_formatted", "")
    metadata_file = source_path.parent / f"{base}_metadata.json"

    entities_file = config.get("ENTITIES_FILE")
    vault_dir = config.get("VAULT_DIR")
    meeting_index = config.get("MEETING_INDEX")

    if extracted_meta is None and formatted_file and Path(formatted_file).exists():
        log_info("Obsidian: extracting meeting metadata...")
        try:
            extracted_meta = run_extract_metadata(formatted_file)
        except Exception as e:
            log_error(f"Failed to extract metadata: {e}")

    if formatted_file and Path(formatted_file).exists():
        log_info("Obsidian: adding frontmatter to formatted transcript...")
        try:
            run_frontmatter(formatted_file, kind="formatted", 
                            metadata=str(metadata_file) if metadata_file.exists() else None, 
                            extracted=extracted_meta)
        except Exception as e:
            log_error(f"Failed to apply frontmatter to formatted file: {e}")

    if summary_file and Path(summary_file).exists():
        log_info("Obsidian: adding frontmatter to summary...")
        try:
            run_frontmatter(summary_file, kind="summary", 
                            metadata=str(metadata_file) if metadata_file.exists() else None, 
                            extracted=extracted_meta)
        except Exception as e:
            log_error(f"Failed to apply frontmatter to summary file: {e}")

    if Path(entities_file).exists():
        log_info("Obsidian: injecting wikilinks...")
        try:
            if formatted_file and Path(formatted_file).exists(): run_wikilinks(formatted_file, entities=entities_file)
            if summary_file and Path(summary_file).exists(): run_wikilinks(summary_file, entities=entities_file)
        except Exception as e:
            log_error(f"Failed to inject wikilinks: {e}")

    if meeting_index and summary_file and Path(summary_file).exists():
        log_info("Obsidian: updating meeting index...")
        try:
            run_update_index(summary_file, index=meeting_index)
        except Exception as e:
            log_error(f"Failed to update index: {e}")

    if vault_dir:
        vault_subdir = config.get("VAULT_SUBDIR")
        vault_base = Path(vault_dir) / vault_subdir if vault_subdir else Path(vault_dir)
        file_for_path = formatted_file if formatted_file else summary_file
        if file_for_path:
            fp = Path(file_for_path)
            meeting_dir_name = fp.parent.name
            date_dir_name = fp.parent.parent.name

            if (len(date_dir_name) == 8 and date_dir_name.isdigit()
                    and meeting_dir_name.startswith("meeting_")):
                from autonote.obsidian.frontmatter import parse_timestamp_from_filename
                formatted_date, time_part = parse_timestamp_from_filename(meeting_dir_name)
                raw_title = _resolve_vault_title(summary_file, time_part)
                slug = _slugify(raw_title)
                folder_name = f"{formatted_date} {slug}"
                vault_dest = _find_unique_vault_dest(vault_base, folder_name)
            else:
                folder_name = ""
                vault_dest = vault_base

            log_info(f"Obsidian: copying files to vault: {vault_dest}")
            os.makedirs(vault_dest, exist_ok=True)

            transcript_stem = f"{folder_name} - transcript" if folder_name else (Path(formatted_file).stem if formatted_file else "transcript")
            summary_stem = folder_name if folder_name else (Path(summary_file).stem if summary_file else "summary")

            if formatted_file and Path(formatted_file).exists():
                dest = vault_dest / f"{transcript_stem}.md"
                shutil.copy(formatted_file, dest)
                log_success(f"Vault: {dest.relative_to(Path(vault_dir))}")

            if summary_file and Path(summary_file).exists():
                dest = vault_dest / f"{summary_stem}.md"
                shutil.copy(summary_file, dest)
                if folder_name:
                    _patch_transcript_wikilink(dest, transcript_stem)
                log_success(f"Vault: {dest.relative_to(Path(vault_dir))}")


def run_process(audio_file: str, diarize=False, no_reformat=False, no_compress=False, keep_wav=False, clean=False, provider: str | None = None, api_key: str | None = None, **kwargs):
    if not Path(audio_file).exists():
        log_error(f"Audio file not found: {audio_file}")
        return

    if diarize:
        try:
            from autonote.audio.diarize import run_diarize
            from autonote.audio.merge_diarization import run_merge
            from autonote.audio.label import run_label
            from autonote.audio.apply_labels import run_apply_labels
        except ImportError:
            log_error("Diarization requires pyannote.audio. Install with: uv sync --extra diarize")
            return
        log_info("Step 2: Identifying speakers...")
        speakers_file = run_diarize(audio_file, speakers=kwargs.get("speakers"))

        log_info("Step 3: Transcribing audio...")
        trans_result = transcribe_audio(audio_file, output_format="json", provider=provider, api_key=api_key)
        transcription_json = str(Path(audio_file).with_suffix(".json"))
        save_transcription(trans_result, transcription_json, "json")

        log_info("Step 4: Merging speaker info with transcription...")
        diarized_file = run_merge(speakers_file, transcription_json)

        log_info("Step 5: Label speakers with their names...")
        labeled_file = run_label(diarized_file)

        log_info("Step 6: Creating final transcript with speaker names...")
        transcription_file = run_apply_labels(labeled_file, format="txt")
    else:
        log_info("Step 2: Transcribing audio...")
        trans_result = transcribe_audio(audio_file, output_format="txt", provider=provider, api_key=api_key)
        transcription_file = str(Path(audio_file).with_suffix(".txt"))
        save_transcription(trans_result, transcription_file, "txt")

    formatted_file = ""
    summarize_input = transcription_file
    if not no_reformat:
        log_info("Step: Reformatting transcription...")
        formatted_file = run_reformat(transcription_file)
        summarize_input = formatted_file

    log_info("Step: Generating summary and extracting metadata in parallel...")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_summary = executor.submit(run_summarize, summarize_input)
        future_meta = executor.submit(run_extract_metadata, summarize_input) if (summarize_input and Path(summarize_input).exists()) else None
        summary_file = future_summary.result()
        extracted_meta = None
        if future_meta:
            try:
                extracted_meta = future_meta.result()
            except Exception as e:
                log_error(f"Failed to extract metadata: {e}")

    log_info("Step: Obsidian post-processing...")
    run_obsidian_postprocess(audio_file, formatted_file, summary_file, extracted_meta=extracted_meta)

    if clean:
        log_info("Step: Removing all audio files...")
        if Path(audio_file).exists(): os.remove(audio_file)
        mp3_file = str(Path(audio_file).with_suffix(".mp3"))
        if Path(mp3_file).exists(): os.remove(mp3_file)
        log_success("Removed audio files.")
    elif not no_compress:
        log_info("Step: Compressing audio to MP3...")
        compress_audio(audio_file, delete_wav=not keep_wav)

    log_success("Processing complete!")


def run_process_last(**kwargs):
    recordings_dir = config.get("RECORDINGS_DIR")
    wav_files = glob.glob(os.path.join(recordings_dir, "**/*.wav"), recursive=True)
    mp3_files = glob.glob(os.path.join(recordings_dir, "**/*.mp3"), recursive=True)
    all_audio = wav_files + mp3_files
    
    if not all_audio:
        log_error("No audio files found in recordings directory.")
        return

    latest_audio = max(all_audio, key=os.path.getmtime)
    log_info(f"Most recent recording found: {latest_audio}")
    run_process(latest_audio, **kwargs)


def run_full(duration=None, title=None, **kwargs):
    log_info("Starting full meeting workflow...")
    audio_file = run_record(duration=duration, title=title)
    if not audio_file or not Path(audio_file).exists():
        log_error("Recording failed.")
        return
    run_process(audio_file, **kwargs)
