"""
Interactive speaker labeling
Shows actual quotes from each speaker and prompts user to identify them
"""
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from autonote.logger import log_info, log_error

def load_diarized_json(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    data = json.loads(path.read_text())
    if "segments" not in data or not data["segments"]:
        raise ValueError("No segments found in file")
    if "text" not in data["segments"][0]:
        raise ValueError("Segments don't contain transcribed text. Merge first.")
    return data

def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def get_speaker_samples(data: dict, speaker_id: str, num_samples: int = 3, random_samples: bool = False) -> list:
    speaker_segments = [seg for seg in data["segments"] if seg["speaker_id"] == speaker_id and seg["text"].strip()]
    if not speaker_segments: return []
    if random_samples:
        if len(speaker_segments) <= num_samples: return speaker_segments
        return random.sample(speaker_segments, num_samples)
    else:
        samples = []
        if len(speaker_segments) >= num_samples:
            samples.append(speaker_segments[0])
            samples.append(speaker_segments[len(speaker_segments) // 2])
            samples.append(speaker_segments[-1])
        else:
            samples = speaker_segments[:num_samples]
        return samples

def display_quotes(samples: list, max_length: int = 100):
    if not samples:
        print("  (No quotes available)")
        return
    print("Sample quotes:")
    for sample in samples:
        timestamp = format_time(sample["start"])
        text = sample["text"].strip()
        if len(text) > max_length: text = text[:max_length] + "..."
        print(f"  [{timestamp}] \"{text}\"")

def interactive_label_speakers(data: dict) -> dict:
    speakers = sorted(set(seg["speaker_id"] for seg in data["segments"]))
    print("\nAutonote Speaker Labeling")
    print("═" * 70)
    print(f"\nFound {len(speakers)} speaker(s)")
    print()

    labels = data.get("labels", {})
    for speaker_id in speakers:
        print("─" * 70)
        stats = data.get("speaker_stats", {}).get(speaker_id, {})
        total_time = stats.get("total_time", 0)
        segment_count = stats.get("segment_count", 0)
        print(f"{speaker_id} ({format_time(total_time)}, {segment_count} segments)\n")

        while True:
            display_quotes(get_speaker_samples(data, speaker_id, num_samples=3))
            print()
            try:
                response = input(f"Who is {speaker_id}? (or 'm' for more quotes) ").strip()
                if response.lower() == 'm':
                    print()
                    display_quotes(get_speaker_samples(data, speaker_id, num_samples=5, random_samples=True))
                    print()
                    continue
                if response:
                    labels[speaker_id] = {
                        "name": response, "email": None, "role": None,
                        "source": "manual", "labeled_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    }
                    print(f"✓ Labeled as \"{response}\"")
                    break
                else:
                    skip = input("Skip this speaker? (y/n) ").strip().lower()
                    if skip == 'y':
                        labels[speaker_id] = {
                            "name": speaker_id, "email": None, "role": None,
                            "source": "skipped", "labeled_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                        }
                        print(f"⊘ Skipped, will use {speaker_id}")
                        break
            except (KeyboardInterrupt, EOFError):
                log_error("Labeling interrupted")
                raise SystemExit(1)
        print()
    print("─" * 70)
    print("✓ All speakers labeled!\n")
    data["labels"] = labels
    return data

def run_label(diarized_file: str, output_file: str = None, non_interactive: bool = False):
    data = load_diarized_json(diarized_file)
    input_path = Path(diarized_file)
    
    if not output_file:
        if input_path.stem.endswith("_diarized"):
            base_name = input_path.stem.replace("_diarized", "")
            output_file = str(input_path.parent / f"{base_name}_speakers_labeled.json")
        else:
            output_file = str(input_path.parent / f"{input_path.stem}_labeled.json")
            
    if non_interactive:
        log_info("Non-interactive mode: skipping speaker labeling")
    else:
        data = interactive_label_speakers(data)
        
    Path(output_file).write_text(json.dumps(data, indent=2))
    log_info(f"Saved labeled data to: {output_file}")
    return output_file
