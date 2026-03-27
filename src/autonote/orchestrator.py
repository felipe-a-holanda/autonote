import os
import shutil
import glob
from pathlib import Path
from autonote.logger import log_info, log_error, log_success
from autonote.config import config

from autonote.audio.record import run_record
from autonote.audio.transcribe import transcribe_audio, save_transcription
from autonote.audio.diarize import run_diarize
from autonote.audio.merge_diarization import run_merge
from autonote.audio.label import run_label
from autonote.audio.apply_labels import run_apply_labels
from autonote.audio.reformat import run_reformat
from autonote.audio.summarize import run_summarize
from autonote.audio.compress import compress_audio
from autonote.obsidian.extract_metadata import run_extract_metadata
from autonote.obsidian.frontmatter import run_frontmatter
from autonote.obsidian.wikilink import run_wikilinks
from autonote.obsidian.update_index import run_update_index

def run_obsidian_postprocess(source_file: str, formatted_file: str, summary_file: str):
    source_path = Path(source_file)
    base = source_path.stem.replace("_formatted", "")
    metadata_file = source_path.parent / f"{base}_metadata.json"
    
    entities_file = config.get("ENTITIES_FILE", str(Path(__file__).parent.parent.parent.parent / "entities.yml"))
    vault_dir = config.get("VAULT_DIR", "")
    meeting_index = config.get("MEETING_INDEX", "")

    extracted_meta = None
    if formatted_file and Path(formatted_file).exists():
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
        file_for_path = formatted_file if formatted_file else summary_file
        if file_for_path:
            fp = Path(file_for_path)
            meeting_dir_name = fp.parent.name
            date_dir_name = fp.parent.parent.name
            
            meeting_subdir = ""
            if len(date_dir_name) == 8 and date_dir_name.isdigit() and meeting_dir_name.startswith("meeting_"):
                formatted_date = f"{date_dir_name[:4]}-{date_dir_name[4:6]}-{date_dir_name[6:]}"
                meeting_subdir = f"{formatted_date}/{meeting_dir_name}"
            
            vault_dest = Path(vault_dir) / meeting_subdir if meeting_subdir else Path(vault_dir)
            log_info(f"Obsidian: copying files to vault: {vault_dest}")
            os.makedirs(vault_dest, exist_ok=True)
            if formatted_file and Path(formatted_file).exists():
                shutil.copy(formatted_file, vault_dest)
                log_success(f"Vault: {meeting_subdir + '/' if meeting_subdir else ''}{Path(formatted_file).name}")
            if summary_file and Path(summary_file).exists():
                shutil.copy(summary_file, vault_dest)
                log_success(f"Vault: {meeting_subdir + '/' if meeting_subdir else ''}{Path(summary_file).name}")


def run_process(audio_file: str, diarize=False, no_reformat=False, no_compress=False, keep_wav=False, clean=False, **kwargs):
    if not Path(audio_file).exists():
        log_error(f"Audio file not found: {audio_file}")
        return

    if diarize:
        log_info("Step 2: Identifying speakers...")
        speakers_file = run_diarize(audio_file, speakers=kwargs.get("speakers"))
        
        log_info("Step 3: Transcribing audio...")
        trans_result = transcribe_audio(audio_file, output_format="json")
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
        trans_result = transcribe_audio(audio_file, output_format="txt")
        transcription_file = str(Path(audio_file).with_suffix(".txt"))
        save_transcription(trans_result, transcription_file, "txt")

    formatted_file = ""
    summarize_input = transcription_file
    if not no_reformat:
        log_info("Step: Reformatting transcription...")
        formatted_file = run_reformat(transcription_file)
        summarize_input = formatted_file

    log_info("Step: Generating summary...")
    summary_file = run_summarize(summarize_input)

    log_info("Step: Obsidian post-processing...")
    run_obsidian_postprocess(audio_file, formatted_file, summary_file)

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
    recordings_dir = config.get("RECORDINGS_DIR", str(Path(os.getcwd()) / "recordings"))
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
