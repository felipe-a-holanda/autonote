import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from autonote.config import get_config, DEFAULT_MODEL


class TestGetConfig:
    
    @patch("os.path.exists", return_value=False)
    @patch.dict(os.environ, {}, clear=True)
    def test_default_config_values(self, mock_exists):
        config = get_config()
        
        assert config["TRANSCRIPTION_PROVIDER"] == "local"
        assert config["WHISPER_MODEL"] == "turbo"
        assert config["WHISPER_LANGUAGE"] == ""
        assert config["MODEL"] == DEFAULT_MODEL
        assert config["DEBUG"] == "false"
        assert config["VAULT_SUBDIR"] == "meetings"
        assert config["MIC_VOLUME"] == "2.0"
        assert config["USD_TO_BRL"] == "5.50"
    
    @patch.dict(os.environ, {
        "RECORDINGS_DIR": "/custom/recordings",
        "TRANSCRIPTION_PROVIDER": "assemblyai",
        "WHISPER_MODEL": "large-v3",
        "MODEL": "openai/gpt-4o",
        "DEBUG": "true",
    }, clear=True)
    def test_environment_overrides(self):
        config = get_config()
        
        assert config["RECORDINGS_DIR"] == "/custom/recordings"
        assert config["TRANSCRIPTION_PROVIDER"] == "assemblyai"
        assert config["WHISPER_MODEL"] == "large-v3"
        assert config["MODEL"] == "openai/gpt-4o"
        assert config["DEBUG"] == "true"
    
    @patch.dict(os.environ, {"USD_TO_BRL": "6.25"}, clear=True)
    def test_fx_rate_from_env(self):
        config = get_config()
        assert config["USD_TO_BRL"] == "6.25"
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="5.75")
    @patch.dict(os.environ, {}, clear=True)
    def test_fx_rate_from_file(self, mock_file, mock_exists):
        mock_exists.return_value = True
        config = get_config()
        assert config["USD_TO_BRL"] == "5.75"
    
    @patch.dict(os.environ, {
        "PRESET_LOCAL": "ollama/llama3.2:3b",
        "PRESET_CHEAP": "groq/llama-3.1-70b",
        "PRESET_FAST": "openai/gpt-4o-mini",
        "PRESET_SMART": "anthropic/claude-3-5-sonnet-20241022",
    }, clear=True)
    def test_model_presets(self):
        config = get_config()
        
        assert config["PRESET_LOCAL"] == "ollama/llama3.2:3b"
        assert config["PRESET_CHEAP"] == "groq/llama-3.1-70b"
        assert config["PRESET_FAST"] == "openai/gpt-4o-mini"
        assert config["PRESET_SMART"] == "anthropic/claude-3-5-sonnet-20241022"
    
    @patch.dict(os.environ, {
        "MODEL": "openai/gpt-4o",
        "MODEL_REFORMAT": "openai/gpt-4o-mini",
        "MODEL_SUMMARIZE": "anthropic/claude-3-5-sonnet-20241022",
        "MODEL_METADATA": "ollama/llama3.1:8b",
    }, clear=True)
    def test_stage_specific_models(self):
        config = get_config()
        
        assert config["MODEL"] == "openai/gpt-4o"
        assert config["MODEL_REFORMAT"] == "openai/gpt-4o-mini"
        assert config["MODEL_SUMMARIZE"] == "anthropic/claude-3-5-sonnet-20241022"
        assert config["MODEL_METADATA"] == "ollama/llama3.1:8b"
    
    @patch.dict(os.environ, {
        "VAULT_DIR": "/home/user/vault",
        "VAULT_SUBDIR": "work-meetings",
        "MEETING_INDEX": "index.md",
    }, clear=True)
    def test_vault_configuration(self):
        config = get_config()
        
        assert config["VAULT_DIR"] == "/home/user/vault"
        assert config["VAULT_SUBDIR"] == "work-meetings"
        assert config["MEETING_INDEX"] == "index.md"
