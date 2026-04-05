import logging
import os
import json
import re
import time
import litellm
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from autonote.logger import log_error, log_info
from autonote.config import config, DEFAULT_MODEL

_logger = logging.getLogger(__name__)

# Turn off litellm telemetry and set reasonable defaults
litellm.telemetry = False

# Default presets for LLM models
LLM_PRESETS = {
    "local": config.get("PRESET_LOCAL"),
    "cheap": config.get("PRESET_CHEAP"),
    "fast": config.get("PRESET_FAST"),
    "smart": config.get("PRESET_SMART"),
}

def _recording_base(source_file: str) -> tuple[Path, str]:
    """Return (recording_dir, base_stem) stripping known pipeline suffixes."""
    p = Path(source_file)
    stem = re.sub(r"(_formatted|_summary|_extracted_metadata)$", "", p.stem)
    return p.parent, stem


def _append_cost_log(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: float,
    cost_brl: Optional[float],
    source_file: Optional[str] = None,
    stage: Optional[str] = None,
    duration_s: Optional[float] = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "ts": ts,
        "stage": stage,
        "model": model,
        "source_file": source_file,
        "duration_s": round(duration_s, 2) if duration_s is not None else None,
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "tokens_total": total_tokens,
        "cost_usd": round(cost_usd, 8),
        "cost_brl": round(cost_brl, 6) if cost_brl is not None else None,
    }

    # Global JSONL log
    log_path = config.get("LLM_COST_LOG")
    if log_path:
        recording_dir = str(_recording_base(source_file)[0]) if source_file else None
        global_entry = {**entry, "recording_dir": recording_dir}
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(global_entry) + "\n")
        except Exception as e:
            log_error(f"Failed to write cost log: {e}")

    # Per-recording cost file
    if source_file:
        _write_recording_cost(entry, source_file)


def _write_recording_cost(entry: Dict[str, Any], source_file: str) -> None:
    """Append a cost entry to the per-recording _llm_costs.json file."""
    recording_dir, base = _recording_base(source_file)
    cost_file = recording_dir / f"{base}_llm_costs.json"
    existing: list = []
    if cost_file.exists():
        try:
            existing = json.loads(cost_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    # Strip source_file from per-recording entry (redundant there)
    rec_entry = {k: v for k, v in entry.items() if k != "source_file"}
    existing.append(rec_entry)
    try:
        cost_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log_error(f"Failed to write recording cost file: {e}")


def resolve_model(model: str) -> str:
    """Resolve a preset name or bare model name to a fully-qualified model string.
    
    All models should be fully-qualified with a provider prefix (e.g. 'ollama/llama3.1:8b',
    'openai/gpt-4o'). Any bare name without a '/' is assumed to be a local Ollama model.
    """
    model = model or config.get("MODEL")
    if model in LLM_PRESETS:
        model = LLM_PRESETS[model]
    if "/" not in model:
        model = f"ollama/{model}"
    return model


def query_llm(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    source_file: Optional[str] = None,
    stage: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Unified LLM query interface using litellm.
    Supports local Ollama models and cloud providers automatically.
    
    Args:
        prompt: A single string prompt (user message).
        messages: A list of message dicts [{"role": "user", "content": "..."}].
        model: The model string (e.g., 'openai/gpt-4o', 'ollama/llama3.1:8b'). 
               Can also be a preset name like 'fast', 'smart', 'cheap', 'local'.
               Defaults to OLLAMA_MODEL or LLM_MODEL config.
        api_base: API base URL. Defaults to OLLAMA_URL if an ollama model is used.
    """
    if messages is None:
        if prompt is None:
            raise ValueError("Must provide either prompt or messages")
        messages = [{"role": "user", "content": prompt}]
        
    resolved = resolve_model(model or config.get("MODEL"))
    if resolved != (model or config.get("MODEL")):
        log_info(f"Using LLM preset: {model} -> {resolved}")
    model = resolved
        
    ollama_url = config.get("OLLAMA_URL")
    if model.startswith("ollama/"):
        if not api_base:
            api_base = ollama_url
    elif api_base and api_base.rstrip("/") == ollama_url.rstrip("/"):
        # Don't forward the Ollama URL to cloud providers
        api_base = None

    try:
        log_info(f"Querying LLM provider for model: {model}")
        _t0 = time.monotonic()
        response = litellm.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            # Using higher timeout for local models and large context
            timeout=kwargs.pop("timeout", 300),
            **kwargs
        )
        duration_s = time.monotonic() - _t0

        # Log usage and cost
        usage = getattr(response, "usage", None)
        if usage:
            # usage can be a dict or an object depending on litellm version/provider
            if hasattr(usage, "get"):
                prompt_t = usage.get("prompt_tokens", 0)
                completion_t = usage.get("completion_tokens", 0)
                total_t = usage.get("total_tokens", 0)
            else:
                prompt_t = getattr(usage, "prompt_tokens", 0)
                completion_t = getattr(usage, "completion_tokens", 0)
                total_t = getattr(usage, "total_tokens", 0)

            try:
                cost = litellm.completion_cost(completion_response=response)
                cost_str = ""
                brl_cost: Optional[float] = None
                if cost > 0:
                    try:
                        usd_to_brl = float(config.get("USD_TO_BRL"))
                        brl_cost = cost * usd_to_brl
                        cost_str = f" (${cost:.6f} / R${brl_cost:.4f})"
                    except (ValueError, TypeError):
                        cost_str = f" (${cost:.6f})"
                elif not model.startswith("ollama/"):
                    cost_str = " (cost unknown — model not in litellm pricing DB)"

                log_info(f"LLM Usage: {total_t} tokens (In: {prompt_t}, Out: {completion_t}){cost_str} [{duration_s:.1f}s]")
                _append_cost_log(model, prompt_t, completion_t, total_t, cost, brl_cost, source_file, stage, duration_s)
                _logger.debug(
                    "LLM usage: %s %d tokens [%.1fs]", model, total_t, duration_s,
                    extra={"structured": {
                        "event": "llm_usage",
                        "model": model,
                        "stage": stage,
                        "tokens_in": prompt_t,
                        "tokens_out": completion_t,
                        "tokens_total": total_t,
                        "cost_usd": round(cost, 8),
                        "duration_s": round(duration_s, 2),
                    }},
                )
            except Exception:
                # Fallback if cost calculation fails (e.g. unknown local model)
                log_info(f"LLM Usage: {total_t} tokens (In: {prompt_t}, Out: {completion_t}) [{duration_s:.1f}s]")
                _append_cost_log(model, prompt_t, completion_t, total_t, 0.0, None, source_file, stage, duration_s)
                _logger.debug(
                    "LLM usage: %s %d tokens [%.1fs]", model, total_t, duration_s,
                    extra={"structured": {
                        "event": "llm_usage",
                        "model": model,
                        "stage": stage,
                        "tokens_in": prompt_t,
                        "tokens_out": completion_t,
                        "tokens_total": total_t,
                        "cost_usd": 0.0,
                        "duration_s": round(duration_s, 2),
                    }},
                )

        return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error querying LLM ({model}): {e}")
        raise RuntimeError(f"LLM query failed: {e}")
