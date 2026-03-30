"""
Manages YAML frontmatter for Obsidian-compatible markdown files.
Prepends or updates frontmatter based on filename, metadata.json, and extracted metadata.
"""
import json
import re
import yaml
from pathlib import Path
from autonote.logger import log_info, log_error

def parse_timestamp_from_filename(filename: str) -> tuple[str, str]:
    match = re.search(r'meeting_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', filename)
    if match:
        y, mo, d, hh, mm, _ = match.groups()
        return f"{y}-{mo}-{d}", f"{hh}:{mm}"
    return "", ""

def read_metadata_json(path: Path) -> dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def parse_existing_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"): 
        return {}, content
    end = content.find("\n---", 3)
    if end == -1: 
        return {}, content
    
    fm_text = content[4:end]
    body = content[end + 4:].lstrip("\n")
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body

def render_frontmatter(fm: dict) -> str:
    if not fm:
        return ""
    fm_text = yaml.dump(fm, default_flow_style=False, sort_keys=False)
    return f"---\n{fm_text}---"

def build_frontmatter_dict(file_path: Path, metadata_path: Path | None, extracted: dict | None, kind: str) -> dict:
    date, time = parse_timestamp_from_filename(file_path.name)
    meta = read_metadata_json(metadata_path)
    user_tag = meta.get("title", "")
    ext = extracted or {}

    fm: dict = {}
    fm["date"] = date
    fm["time"] = time
    
    # LLM-inferred title takes priority, with user_tag as separate field
    llm_title = ext.get("meeting_title", "").strip()
    if llm_title:
        fm["title"] = llm_title
    else:
        fm["title"] = user_tag  # Fallback to user tag if LLM didn't infer a title
    
    # Keep user_tag as a separate field for reference
    if user_tag:
        fm["user_tag"] = user_tag
    
    fm["tags"] = ext.get("tags", [])
    fm["participants"] = ext.get("participants", [])
    fm["jira_tickets"] = ext.get("jira_tickets", [])
    fm["topics"] = ext.get("topics", [])

    if kind == "summary":
        stem = re.sub(r"_summary$", "", file_path.stem)
        stem = re.sub(r"_formatted$", "", stem)
        fm["transcript"] = f"[[{stem}_formatted]]"

    return fm

def apply_frontmatter(file_path: Path, new_fm: dict) -> None:
    content = file_path.read_text(encoding="utf-8")
    existing_fm, body = parse_existing_frontmatter(content)

    if existing_fm:
        merged = dict(existing_fm)
        for k, v in new_fm.items():
            if k not in merged: merged[k] = v
            elif isinstance(v, list) and not merged.get(k): merged[k] = v
            elif not merged.get(k) and v: merged[k] = v
        final_fm = merged
        final_body = body
    else:
        final_fm = new_fm
        final_body = content

    rendered = render_frontmatter(final_fm)
    text_to_write = f"{rendered}\n\n{final_body}" if rendered else final_body
    file_path.write_text(text_to_write, encoding="utf-8")

def run_frontmatter(file: str, kind: str = "formatted", metadata: str = None, extracted: str = None):
    file_path = Path(file)
    if not file_path.exists(): 
        raise FileNotFoundError(f"File not found: {file}")
    
    metadata_path = Path(metadata) if metadata else None
    ext_data = {}
    if extracted:
        extracted_path = Path(extracted)
        if extracted_path.exists():
            try: ext_data = json.loads(extracted_path.read_text(encoding="utf-8"))
            except: log_error(f"Warning: could not read extracted metadata: {extracted}")

    new_fm = build_frontmatter_dict(file_path, metadata_path, ext_data, kind)
    apply_frontmatter(file_path, new_fm)
    log_info(f"Frontmatter updated in: {file_path}")
