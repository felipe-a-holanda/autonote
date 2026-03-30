"""
Manages meetings_calendar.yml file for recurring meeting patterns.
Provides CLI command to auto-generate calendar from past meeting metadata.
"""
import json
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from autonote.logger import log_info, log_error
from autonote.config import config


def parse_metadata_files(recordings_dir: Path) -> list[dict]:
    """Scan all _metadata.json files and extract meeting info."""
    meetings = []
    
    for meta_file in recordings_dir.rglob("*_metadata.json"):
        # Skip extracted_metadata files
        if "extracted_metadata" in meta_file.name:
            continue
            
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            title = data.get("title", "").strip()
            timestamp = data.get("timestamp", "")
            
            if not title or not timestamp:
                continue
            
            # Parse timestamp to get day of week and time
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                meetings.append({
                    "title": title,
                    "day": dt.strftime("%A"),
                    "time": dt.strftime("%H:%M"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "weekday": dt.weekday(),  # 0=Monday, 6=Sunday
                })
            except ValueError:
                continue
                
        except (json.JSONDecodeError, IOError) as e:
            log_error(f"Failed to read {meta_file}: {e}")
            continue
    
    return meetings


def find_recurring_patterns(meetings: list[dict], min_occurrences: int = 2) -> list[dict]:
    """Group meetings by title and find recurring time patterns."""
    # Group by normalized title
    by_title = defaultdict(list)
    for m in meetings:
        # Normalize title (lowercase, strip common suffixes like numbers)
        normalized = m["title"].lower().strip()
        by_title[normalized].append(m)
    
    recurring = []
    for title, occurrences in by_title.items():
        if len(occurrences) < min_occurrences:
            continue
        
        # Find most common day/time combination
        time_patterns = defaultdict(int)
        for m in occurrences:
            pattern = f"{m['day']} {m['time']}"
            time_patterns[pattern] += 1
        
        # Get the most common pattern
        most_common = max(time_patterns.items(), key=lambda x: x[1])
        pattern_str, count = most_common
        
        # Parse pattern back
        day, time = pattern_str.split(" ", 1)
        
        # Determine schedule description
        weekday_num = occurrences[0]["weekday"]
        if count >= len(occurrences) * 0.8:  # 80% of occurrences match
            if weekday_num < 5:  # Monday-Friday
                schedule = f"weekdays {time}"
            else:
                schedule = f"{day.lower()} {time}"
        else:
            schedule = f"{day.lower()} {time}"
        
        recurring.append({
            "name": title.title(),  # Capitalize for display
            "schedule": schedule,
            "keywords": [title.lower()],
            "occurrences": len(occurrences),
            "pattern_match": f"{count}/{len(occurrences)}"
        })
    
    # Sort by occurrence count (most frequent first)
    recurring.sort(key=lambda x: x["occurrences"], reverse=True)
    return recurring


def generate_calendar_yaml(recurring_meetings: list[dict]) -> str:
    """Generate YAML content for meetings_calendar.yml."""
    # Build the structure
    calendar = {
        "recurring": [
            {
                "name": m["name"],
                "schedule": m["schedule"],
                "keywords": m["keywords"]
            }
            for m in recurring_meetings
        ]
    }
    
    # Add header comment
    header = """# Meetings Calendar
# Auto-generated from past meeting metadata
# Edit this file to refine recurring meeting patterns
#
# Format:
#   name: Display name for the meeting
#   schedule: When it occurs (e.g., "weekdays 11:30", "thursday 14:00")
#   keywords: List of keywords to help identify this meeting type

"""
    
    yaml_content = yaml.dump(calendar, default_flow_style=False, sort_keys=False)
    return header + yaml_content


def load_calendar(calendar_path: Path) -> dict:
    """Load meetings_calendar.yml file."""
    if not calendar_path.exists():
        return {"recurring": []}
    
    try:
        content = calendar_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
        return data
    except yaml.YAMLError as e:
        log_error(f"Failed to parse calendar file: {e}")
        return {"recurring": []}


def run_calendar_sync(recordings_dir: str = None, output_file: str = None, min_occurrences: int = 2):
    """
    Scan past meeting metadata and generate meetings_calendar.yml.
    
    Args:
        recordings_dir: Path to recordings directory (defaults to config)
        output_file: Path to output calendar file (defaults to config)
        min_occurrences: Minimum number of occurrences to consider a meeting recurring
    """
    recordings_path = Path(recordings_dir or config.get("RECORDINGS_DIR"))
    if not recordings_path.exists():
        log_error(f"Recordings directory not found: {recordings_path}")
        return
    
    log_info(f"Scanning meetings in: {recordings_path}")
    meetings = parse_metadata_files(recordings_path)
    log_info(f"Found {len(meetings)} meetings with titles")
    
    if not meetings:
        log_error("No meetings found with titles. Cannot generate calendar.")
        return
    
    log_info(f"Finding recurring patterns (min {min_occurrences} occurrences)...")
    recurring = find_recurring_patterns(meetings, min_occurrences)
    
    if not recurring:
        log_info("No recurring patterns found. Try lowering --min-occurrences.")
        return
    
    log_info(f"Found {len(recurring)} recurring meeting patterns:")
    for m in recurring:
        log_info(f"  - {m['name']}: {m['schedule']} ({m['pattern_match']} occurrences)")
    
    # Generate YAML
    yaml_content = generate_calendar_yaml(recurring)
    
    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        calendar_file = config.get("MEETINGS_CALENDAR")
        if calendar_file:
            output_path = Path(calendar_file)
        else:
            # Default to same directory as entities.yml
            entities_file = config.get("ENTITIES_FILE")
            if entities_file:
                output_path = Path(entities_file).parent / "meetings_calendar.yml"
            else:
                output_path = Path("./meetings_calendar.yml")
    
    output_path.write_text(yaml_content, encoding="utf-8")
    log_info(f"Calendar saved to: {output_path}")
    log_info("Review and edit the file to refine recurring meeting patterns.")
    return str(output_path)
