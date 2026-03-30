"""
Appends a new entry to the Obsidian Meetings.md index file after each recording.
Reads frontmatter from the summary file to build the index row.
"""
from pathlib import Path
from datetime import datetime
from autonote.logger import log_info, log_error

def parse_frontmatter(content: str) -> dict:
    if not content.startswith("---"): return {}
    end = content.find("\n---", 3)
    if end == -1: return {}
    fm_text = content[4:end]
    fm: dict = {}
    for line in fm_text.splitlines():
        if ":" not in line: continue
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            fm[key] = [v.strip().strip("\"'") for v in inner.split(",")] if inner else []
        else:
            fm[key] = raw.strip("\"'")
    return fm

def ensure_index_header(index_path: Path) -> None:
    if not index_path.exists():
        index_path.write_text(
            "# Meetings\n\n| Meeting | Date | Time | Day | Tags | Participants |\n|---------|------|------|-----|------|--------------|\n",
            encoding="utf-8",
        )
    else:
        # Migrate old format to new format if needed
        content = index_path.read_text(encoding="utf-8")
        if "| Meeting | Date | Tags | Participants |" in content:
            log_info("Migrating Meetings.md to new format with Time and Day columns...")
            migrate_index_format(index_path)

def build_row(summary_path: Path, fm: dict) -> str:
    date = fm.get("date", "")
    time = fm.get("time", "")
    title = fm.get("title", "") or summary_path.stem.replace("_", " ")
    stem = summary_path.stem
    link = f"[[{stem}\\|{date} {title}]]" if title else f"[[{stem}]]"
    
    # Calculate day of week from date
    day = ""
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            day = dt.strftime("%A")  # Full day name (Monday, Tuesday, etc.)
        except ValueError:
            pass
    
    tags = fm.get("tags", [])
    tags_str = " ".join(f"#{t}" for t in tags) if tags else ""
    participants = fm.get("participants", [])
    participants_str = ", ".join(participants) if participants else ""
    return f"| {link} | {date} | {time} | {day} | {tags_str} | {participants_str} |\n"

def entry_already_exists(index_path: Path, stem: str) -> bool:
    if not index_path.exists(): return False
    return stem in index_path.read_text(encoding="utf-8")

def migrate_index_format(index_path: Path) -> None:
    """Migrate old index format to new format with Time and Day columns."""
    import re
    content = index_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            new_lines.append(line)  # Keep "# Meetings"
        elif i == 1:
            new_lines.append(line)  # Keep empty line
        elif i == 2:
            # Update header
            new_lines.append("| Meeting | Date | Time | Day | Tags | Participants |")
        elif i == 3:
            # Update separator
            new_lines.append("|---------|------|------|-----|------|--------------|")
        elif line.startswith("|") and "[[" in line:
            # Parse existing row and rebuild with time/day
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:  # Old format: | Meeting | Date | Tags | Participants |
                meeting_link = parts[1]
                date = parts[2]
                tags = parts[3]
                participants = parts[4]
                
                # Extract time and day from the meeting filename in the link
                match = re.search(r"meeting_(\d{8})_(\d{6})", meeting_link)
                time = ""
                day = ""
                if match:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    time = f"{time_str[:2]}:{time_str[2:4]}"
                    try:
                        dt = datetime.strptime(date_str, "%Y%m%d")
                        day = dt.strftime("%A")
                    except ValueError:
                        pass
                
                new_lines.append(f"| {meeting_link} | {date} | {time} | {day} | {tags} | {participants} |")
        else:
            new_lines.append(line)
    
    index_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    log_info("Migration complete!")

def run_update_index(summary_file: str, index: str):
    summary_path = Path(summary_file)
    if not summary_path.exists(): raise FileNotFoundError(f"Summary file not found: {summary_file}")
    index_path = Path(index)
    content = summary_path.read_text(encoding="utf-8")
    fm = parse_frontmatter(content)
    if entry_already_exists(index_path, summary_path.stem):
        log_info(f"Index entry already exists for: {summary_path.stem}")
        return
    ensure_index_header(index_path)
    row = build_row(summary_path, fm)
    with index_path.open("a", encoding="utf-8") as f: f.write(row)
    log_info(f"Index updated: {index_path}")
    return str(index_path)
