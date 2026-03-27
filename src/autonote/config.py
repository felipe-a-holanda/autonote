import os
from dotenv import load_dotenv

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
    config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    config["OLLAMA_REFORMAT_MODEL"] = os.environ.get("OLLAMA_REFORMAT_MODEL", "llama3.1:8b")
    config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    config["DEBUG"] = os.environ.get("DEBUG", "false")
    config["VAULT_DIR"] = os.environ.get("VAULT_DIR", "")
    config["MEETING_INDEX"] = os.environ.get("MEETING_INDEX", "")
    config["ENTITIES_FILE"] = os.environ.get("ENTITIES_FILE", os.path.abspath("./entities.yml"))
    config["MIC_VOLUME"] = os.environ.get("MIC_VOLUME", "2.0")
            
    return config

config = get_config()
