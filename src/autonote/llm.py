import os
import litellm
from typing import Optional, List, Dict, Any
from autonote.logger import log_error, log_info
from autonote.config import config

# Turn off litellm telemetry and set reasonable defaults
litellm.telemetry = False

# Default presets for LLM models
LLM_PRESETS = {
    "local": config.get("PRESET_LOCAL", "ollama/llama3.1:8b"),
    "cheap": config.get("PRESET_CHEAP", "deepseek/deepseek-chat"),
    "fast": config.get("PRESET_FAST", "openai/gpt-5.4"),
    "smart": config.get("PRESET_SMART", "anthropic/claude-sonnet-4-6"),
}

def query_llm(
    prompt: Optional[str] = None, 
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None, 
    api_base: Optional[str] = None,
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
        
    model = model or config.get("LLM_MODEL", config.get("OLLAMA_MODEL", "llama3.1:8b"))
    
    # Check for presets
    if model in LLM_PRESETS:
        log_info(f"Using LLM preset: {model} -> {LLM_PRESETS[model]}")
        model = LLM_PRESETS[model]
    
    # If the user just specified a model name like 'llama3.1:8b', we assume Ollama for backward compatibility
    # unless it explicitly has a provider prefix like 'openai/' or 'anthropic/'.
    if "/" not in model and not model.startswith("gpt-") and not model.startswith("claude-"):
        model = f"ollama/{model}"
        
    ollama_url = config.get("OLLAMA_URL", "http://localhost:11434")
    if model.startswith("ollama/"):
        if not api_base:
            api_base = ollama_url
    elif api_base and api_base.rstrip("/") == ollama_url.rstrip("/"):
        # Don't forward the Ollama URL to cloud providers
        api_base = None

    try:
        log_info(f"Querying LLM provider for model: {model}")
        response = litellm.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            # Using higher timeout for local models and large context
            timeout=kwargs.pop("timeout", 300),
            **kwargs
        )

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
                if cost > 0:
                    try:
                        usd_to_brl = float(config.get("USD_TO_BRL", "5.50"))
                        brl_cost = cost * usd_to_brl
                        cost_str = f" (${cost:.6f} / R${brl_cost:.4f})"
                    except (ValueError, TypeError):
                        cost_str = f" (${cost:.6f})"
                elif not model.startswith("ollama/"):
                    cost_str = " (cost unknown — model not in litellm pricing DB)"

                log_info(f"LLM Usage: {total_t} tokens (In: {prompt_t}, Out: {completion_t}){cost_str}")
            except Exception:
                # Fallback if cost calculation fails (e.g. unknown local model)
                log_info(f"LLM Usage: {total_t} tokens (In: {prompt_t}, Out: {completion_t})")

        return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error querying LLM ({model}): {e}")
        raise RuntimeError(f"LLM query failed: {e}")
