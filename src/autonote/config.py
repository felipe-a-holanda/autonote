import os
from dotenv import load_dotenv

DEFAULT_MODEL = "llama3.1:8b"

def get_config() -> dict:
    """Builds the comprehensive configuration dict using python-dotenv."""
    
    # Try multiple .autonoterc locations
    potential_files = [
        os.path.abspath(".autonoterc"),
        os.path.expanduser("~/.autonoterc")
    ]
    
    for filepath in potential_files:
        if os.path.exists(filepath):
            load_dotenv(filepath)
            break
            
    config = {}
    
    config["RECORDINGS_DIR"] = os.environ.get("RECORDINGS_DIR", os.path.abspath("./recordings"))
    config["WHISPER_MODEL"] = os.environ.get("WHISPER_MODEL", "turbo")
    config["WHISPER_LANGUAGE"] = os.environ.get("WHISPER_LANGUAGE", "")
    _default_model = os.environ.get("MODEL", os.environ.get("LLM_MODEL", os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)))
    config["MODEL"] = _default_model
    config["MODEL_REFORMAT"] = os.environ.get("MODEL_REFORMAT", _default_model)
    config["MODEL_SUMMARIZE"] = os.environ.get("MODEL_SUMMARIZE", _default_model)
    config["MODEL_METADATA"] = os.environ.get("MODEL_METADATA", _default_model)
    config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    config["DEBUG"] = os.environ.get("DEBUG", "false")
    config["VAULT_DIR"] = os.environ.get("VAULT_DIR", "")
    config["MEETING_INDEX"] = os.environ.get("MEETING_INDEX", "")
    config["ENTITIES_FILE"] = os.environ.get("ENTITIES_FILE", os.path.abspath("./entities.yml"))
    config["MIC_VOLUME"] = os.environ.get("MIC_VOLUME", "2.0")
    config["MIC_SOURCE"] = os.environ.get("MIC_SOURCE", "")
    config["SYSTEM_SOURCE"] = os.environ.get("SYSTEM_SOURCE", "")
    config["LLM_COST_LOG"] = os.environ.get("LLM_COST_LOG", os.path.expanduser("~/.autonote_costs.jsonl"))
    
    # Simple FX rate logic
    fx_rate = os.environ.get("USD_TO_BRL")
    if not fx_rate:
        potential_fx_files = [
            os.path.abspath(".autonote_fx"),
            os.path.expanduser("~/.autonote_fx")
        ]
        for fx_path in potential_fx_files:
            if os.path.exists(fx_path):
                try:
                    with open(fx_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            fx_rate = content
                            break
                except Exception:
                    pass
    
    config["USD_TO_BRL"] = fx_rate or "5.50"
            
    return config

config = get_config()
