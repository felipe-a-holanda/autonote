"""
Appends a new entry to the Obsidian Meetings.md index file after each recording.
Reads frontmatter from the summary file to build the index row.
"""
from pathlib import Path
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
            "# Meetings\n\n| Meeting | Date | Tags | Participants |\n|---------|------|------|--------------|\n",
            encoding="utf-8",
        )

def build_row(summary_path: Path, fm: dict) -> str:
    date = fm.get("date", "")
    title = fm.get("title", "") or summary_path.stem.replace("_", " ")
    stem = summary_path.stem
    link = f"[[{stem}\\|{date} {title}]]" if title else f"[[{stem}]]"
    tags = fm.get("tags", [])
    tags_str = " ".join(f"#{t}" for t in tags) if tags else ""
    participants = fm.get("participants", [])
    participants_str = ", ".join(participants) if participants else ""
    return f"| {link} | {date} | {tags_str} | {participants_str} |\n"

def entry_already_exists(index_path: Path, stem: str) -> bool:
    if not index_path.exists(): return False
    return stem in index_path.read_text(encoding="utf-8")

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
