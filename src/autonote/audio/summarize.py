"""
Meeting summarization script using Ollama
Takes transcription text and generates meeting notes, summaries, and action items
"""
import json
from pathlib import Path
from autonote.logger import log_info, log_error
from autonote.config import config

try:
    import requests
except ImportError:
    pass

SUMMARY_PROMPT = """You are an AI assistant that creates concise meeting summaries. Analyze the following meeting transcription and provide:

1. **Meeting Summary**: A brief 2-3 sentence overview of the meeting
2. **Key Discussion Points**: Bullet points of main topics discussed
3. **Action Items**: Specific tasks or follow-ups identified (if any)
4. **Decisions Made**: Key decisions or conclusions reached (if any)
5. **Participants**: List any identifiable participants or speakers (if mentioned)

Transcription:
{transcription}

Please format your response in markdown."""

ACTION_ITEMS_PROMPT = """Extract all action items and tasks from this meeting transcription. For each action item, identify:
- The task/action
- Who is responsible (if mentioned)
- Any deadlines or timeframes (if mentioned)

Transcription:
{transcription}

Format as a markdown checklist with details."""

def query_ollama(prompt: str, model: str, ollama_url: str) -> str:
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        log_error(f"Error querying Ollama: {e}")
        raise RuntimeError(f"Ollama query failed: {e}")

def load_transcription(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcription file not found: {file_path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("text", "")
    else:
        return path.read_text()

def summarize_meeting(transcription: str, model: str, ollama_url: str, include_action_items: bool = True) -> dict:
    log_info(f"Generating meeting summary using {model}...")
    summary = query_ollama(SUMMARY_PROMPT.format(transcription=transcription), model=model, ollama_url=ollama_url)
    result = {"summary": summary}
    if include_action_items:
        log_info("Extracting action items...")
        action_items = query_ollama(ACTION_ITEMS_PROMPT.format(transcription=transcription), model=model, ollama_url=ollama_url)
        result["action_items"] = action_items
    return result

def save_summary(result: dict, output_file: str, format: str):
    output_path = Path(output_file)
    if format in ["txt", "md"]:
        lines = ["# Meeting Summary\n", result["summary"]]
        if "action_items" in result:
            lines.extend(["\n\n# Action Items\n", result["action_items"]])
        output_path.write_text("\n".join(lines))
    elif format == "json":
        output_path.write_text(json.dumps(result, indent=2))
    log_info(f"Summary saved to: {output_path}")

def run_summarize(transcription_file: str, model: str = None, ollama_url: str = None, format: str = "md", output_file: str = None, skip_action_items: bool = False):
    model = model or config.get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_url = ollama_url or config.get("OLLAMA_URL", "http://localhost:11434")
    
    transcription = load_transcription(transcription_file)
    if not transcription.strip():
        raise ValueError("Transcription is empty")
        
    if not output_file:
        trans_path = Path(transcription_file)
        suffix = ".md" if format == "md" else f".{format}"
        import re as _re
        base_stem = _re.sub(r"_formatted$", "", trans_path.stem)
        output_file = str(trans_path.with_name(f"{base_stem}_summary{suffix}"))
        
    result = summarize_meeting(transcription, model=model, ollama_url=ollama_url, include_action_items=not skip_action_items)
    save_summary(result, output_file, format)
    log_info("Summarization complete!")
    return output_file
