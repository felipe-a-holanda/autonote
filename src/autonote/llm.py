import os
import litellm
from typing import Optional, List, Dict, Any
from autonote.logger import log_error, log_info
from autonote.config import config

# Turn off litellm telemetry and set reasonable defaults
litellm.telemetry = False

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
               Defaults to OLLAMA_MODEL or LLM_MODEL config.
        api_base: API base URL. Defaults to OLLAMA_URL if an ollama model is used.
    """
    if messages is None:
        if prompt is None:
            raise ValueError("Must provide either prompt or messages")
        messages = [{"role": "user", "content": prompt}]
        
    model = model or config.get("LLM_MODEL", config.get("OLLAMA_MODEL", "llama3.1:8b"))
    
    # If the user just specified a model name like 'llama3.1:8b', we assume Ollama for backward compatibility
    # unless it explicitly has a provider prefix like 'openai/' or 'anthropic/'.
    if "/" not in model and not model.startswith("gpt-") and not model.startswith("claude-"):
        model = f"ollama/{model}"
        
    if model.startswith("ollama/") and not api_base:
        api_base = config.get("OLLAMA_URL", "http://localhost:11434")

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
        return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error querying LLM ({model}): {e}")
        raise RuntimeError(f"LLM query failed: {e}")
