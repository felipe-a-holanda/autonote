"""
Extracts structured metadata from a meeting transcript via LLM.
Outputs a JSON file with participants, topics, jira_tickets, and tags.
"""
import json
import re
from pathlib import Path
from autonote.logger import log_info, log_error
from autonote.config import config, DEFAULT_MODEL
from autonote.llm import query_llm

DEFAULT_OLLAMA_URL = "http://localhost:11434"

EXTRACTION_PROMPT = """You are a metadata extractor. Read the meeting transcript below and return ONLY a JSON object (no markdown, no explanation) with these fields:

- "participants": list of full names of people mentioned (strings)
- "topics": list of key technical topics, products, or concepts discussed (strings, 2-4 words each)
- "jira_tickets": list of Jira ticket IDs found (pattern like EW-1234, ABC-567)
- "tags": list of 2-5 short lowercase tags summarizing the meeting type/content

Example output:
{"participants": ["Alice", "Bob"], "topics": ["Authentication", "Database Migration"], "jira_tickets": ["EW-101"], "tags": ["backend", "sprint"]}

Transcript:
{transcript}

Return ONLY the JSON object. No other text."""

def extract_jira_tickets(text: str) -> list[str]:
    return sorted(set(re.findall(r'\b[A-Z]{2,10}-\d+\b', text)))

def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text

def query_metadata(prompt: str, model: str, api_base: str, source_file: str = None) -> str:
    try:
        return query_llm(prompt=prompt, model=model, api_base=api_base, timeout=120, source_file=source_file, stage="extract_metadata")
    except Exception as e:
        log_error(f"Warning: LLM request failed: {e}")
        return ""

def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

def run_extract_metadata(transcript_file: str, model: str = None, ollama_url: str = None, output: str = None):
    _model = model or config.get("MODEL_METADATA", DEFAULT_MODEL)
    _url = ollama_url or config.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    
    file_path = Path(transcript_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
        
    text = file_path.read_text(encoding="utf-8")
    text = strip_frontmatter(text)
    if not text.strip(): raise ValueError("Transcript is empty")
    
    words = text.split()
    if len(words) > 3000: text = " ".join(words[:3000])
    
    full_text = file_path.read_text(encoding="utf-8")
    regex_tickets = extract_jira_tickets(full_text)
    
    log_info(f"Extracting metadata using {_model}...")
    raw = query_metadata(EXTRACTION_PROMPT.replace("{transcript}", text), _model, _url, source_file=transcript_file)
    metadata = parse_llm_json(raw) if raw else {}
    
    for field in ("participants", "topics", "jira_tickets", "tags"):
        if field not in metadata or not isinstance(metadata[field], list): metadata[field] = []
        
    all_tickets = sorted(set(metadata["jira_tickets"] + regex_tickets))
    metadata["jira_tickets"] = all_tickets
    
    if output:
        output_path = Path(output)
    else:
        stem = re.sub(r"_formatted$", "", file_path.stem)
        stem = re.sub(r"_summary$", "", stem)
        output_path = file_path.with_name(f"{stem}_extracted_metadata.json")
        
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log_info(f"Extracted metadata saved to: {output_path}")
    return str(output_path)
