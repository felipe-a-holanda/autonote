"""
Transcript reformatting using LLM.
Takes raw transcription text and lightly cleans it up.
"""
import json
import re
from pathlib import Path
from autonote.logger import log_info, log_error
from autonote.config import config

try:
    import requests
    from tqdm import tqdm
except ImportError:
    pass

from autonote.llm import query_llm, resolve_model
from autonote.config import DEFAULT_MODEL

SYSTEM_PROMPT = """You are a professional transcript editor. Your ONLY job is to lightly clean up raw speech-to-text transcription:

- Fix obvious transcription errors (misheard words, broken sentences)
- Add paragraph breaks at natural topic changes
- Fix punctuation and capitalization
- Remove filler words (um, uh, you know, like, etc.)

Do NOT rewrite sentences. Do NOT add commentary. Do NOT summarize or shorten the content.
Return ONLY the cleaned, full-length transcript. No preamble."""

def query_reformat(transcription: str, model: str, api_base: str, source_file: str = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Raw transcription:\n{transcription}\n\nCRITICAL: Output the ENTIRE transcript reformatted. DO NOT SUMMARIZE. DO NOT SHORTEN."}
    ]
    return query_llm(messages=messages, model=model, api_base=api_base, timeout=1800, source_file=source_file, stage="reformat")

def chunk_transcription(text: str, max_words: int = 400) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words and word[-1] in ".?!":
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        elif len(current_chunk) >= max_words + 150:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def load_transcription(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcription file not found: {file_path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("text", "")
    else:
        return path.read_text()

def run_reformat(transcription_file: str, model: str = None, ollama_url: str = None, output_file: str = None):
    model = model or config.get("MODEL_REFORMAT", DEFAULT_MODEL)
    ollama_url = ollama_url or config.get("OLLAMA_URL", "http://localhost:11434")
    
    transcription = load_transcription(transcription_file)
    if not transcription.strip():
        raise ValueError("Transcription is empty")
        
    if not output_file:
        trans_path = Path(transcription_file)
        output_file = str(trans_path.with_name(f"{trans_path.stem}_formatted.md"))
        
    resolved = resolve_model(model)
    if resolved.startswith("ollama/"):
        chunks = chunk_transcription(transcription, max_words=500)
        log_info(f"Split transcription into {len(chunks)} chunks.")
    else:
        chunks = [transcription]
        log_info("API model detected — sending full transcript in one request.")
    
    output_path = Path(output_file)
    output_path.write_text("")
    
    for chunk in tqdm(chunks, desc=f"Reformatting ({model})", unit="chunk", dynamic_ncols=True):
        result = query_reformat(chunk, model=model, api_base=ollama_url, source_file=transcription_file)
        
        result = result.strip()
        result = re.sub(r"^(?:Here is|Here's).*?(?:transcript|text)?.*?:?\s*\n+", "", result, flags=re.IGNORECASE).strip()
        result = re.sub(r"^\**(?:Cleaned|Formatted|Reformatted)?\s*Transcript\**:\s*\n+", "", result, flags=re.IGNORECASE).strip()
        
        with output_path.open("a", encoding="utf-8") as f:
            f.write(result + "\n\n")
            
    log_info(f"Reformatted transcript saved to: {output_file}")
    return output_file
