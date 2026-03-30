"""
Meeting summarization using LLM.
Takes transcription text and generates meeting notes, summaries, and action items.
"""
import json
from pathlib import Path
from autonote.logger import log_info, log_error
from autonote.config import config, DEFAULT_MODEL

try:
    import requests
except ImportError:
    pass

from autonote.llm import query_llm

SUMMARY_PROMPT = """You are an AI assistant that creates highly detailed and precise meeting summaries. Analyze the following meeting transcription and meticulously extract the following:

1. **Meeting Summary**: A brief, specific 2-3 sentence overview that captures the true essence of the meeting (avoid generic filler).
2. **Key Discussion Points**: Detailed bullet points of the main topics. Do not oversimplify. Explain *why* things were discussed.
3. **Per-Person Updates (for standups/syncs)**: If this is a standup or multi-person update meeting, clearly separate what each participant is working on, their blockers, and their next steps. If not, omit this section.
4. **Key Warnings & Things to Remember**: Explicitly list any risks, edge cases, technical caveats, or gotchas mentioned (e.g., "deploying to prod has no safeguards", "docs are outdated").
5. **Decisions Made**: Key decisions or conclusions reached (if any).
6. **Participants**: List all identifiable participants.

Do NOT lose critical technical details, ticket numbers, or cross-person coordination context. Be specific.

Transcription:
{transcription}

Please format your response in markdown."""

ACTION_ITEMS_PROMPT = """Extract all actionable tasks from this meeting transcription. Do not be vague.

For each action item, identify:
- The specific task/action (include ticket numbers if mentioned)
- Who is responsible
- Any deadlines, dependencies, or timeframes
- Any important context (e.g., "Check with X before starting Y")

Transcription:
{transcription}

Format as a markdown checklist with details."""

# query_ollama has been replaced by query_llm from autonote.llm

def load_transcription(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcription file not found: {file_path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("text", "")
    else:
        return path.read_text()

def summarize_meeting(transcription: str, model: str, ollama_url: str, include_action_items: bool = True, source_file: str = None) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    log_info(f"Generating summary and action items in parallel using {model}...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_summary = executor.submit(
            query_llm,
            prompt=SUMMARY_PROMPT.format(transcription=transcription),
            model=model, api_base=ollama_url, source_file=source_file, stage="summarize",
        )
        future_actions = executor.submit(
            query_llm,
            prompt=ACTION_ITEMS_PROMPT.format(transcription=transcription),
            model=model, api_base=ollama_url, source_file=source_file, stage="summarize_action_items",
        ) if include_action_items else None
        result = {"summary": future_summary.result()}
        if future_actions:
            result["action_items"] = future_actions.result()
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
    model = model or config.get("MODEL_SUMMARIZE")
    ollama_url = ollama_url or config.get("OLLAMA_URL")
    
    transcription = load_transcription(transcription_file)
    if not transcription.strip():
        raise ValueError("Transcription is empty")
        
    if not output_file:
        trans_path = Path(transcription_file)
        suffix = ".md" if format == "md" else f".{format}"
        import re as _re
        base_stem = _re.sub(r"_formatted$", "", trans_path.stem)
        output_file = str(trans_path.with_name(f"{base_stem}_summary{suffix}"))
        
    result = summarize_meeting(transcription, model=model, ollama_url=ollama_url, include_action_items=not skip_action_items, source_file=transcription_file)
    save_summary(result, output_file, format)
    log_info("Summarization complete!")
    return output_file
