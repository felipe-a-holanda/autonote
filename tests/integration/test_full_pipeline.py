"""
Integration tests for the full pipeline.
These tests verify that multiple components work together correctly.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


@pytest.mark.integration
class TestConfigIntegration:
    
    @patch.dict("os.environ", {
        "RECORDINGS_DIR": "/custom/recordings",
        "MODEL": "openai/gpt-4o",
        "VAULT_DIR": "/vault"
    }, clear=True)
    @patch("os.path.exists", return_value=False)
    def test_config_with_multiple_env_vars(self, mock_exists):
        from autonote.config import get_config
        
        config = get_config()
        
        assert config["RECORDINGS_DIR"] == "/custom/recordings"
        assert config["MODEL"] == "openai/gpt-4o"
        assert config["VAULT_DIR"] == "/vault"


@pytest.mark.integration
class TestLLMIntegration:
    
    @patch("autonote.llm.litellm.completion")
    @patch("autonote.llm.config", {
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_COST_LOG": "/tmp/test_costs.jsonl",
        "USD_TO_BRL": "5.50"
    })
    @patch("builtins.open", new_callable=mock_open)
    def test_query_llm_with_cost_logging(self, mock_file, mock_completion):
        from autonote.llm import query_llm
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        mock_completion.return_value = mock_response
        
        with patch("autonote.llm.litellm.completion_cost", return_value=0.001):
            result = query_llm(
                prompt="Test",
                model="openai/gpt-4o",
                source_file="/recordings/test/meeting.txt",
                stage="test"
            )
        
        assert result == "Response"
        assert mock_file.called


@pytest.mark.integration  
class TestOrchestratorIntegration:
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_full_workflow(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
title: Sprint Planning
---
# 🚀 Q2 Goals
# Meeting Summary
# Action Items
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "14:30")
        
        assert result == "Sprint Planning"
    
    def test_slugify_and_vault_dest_integration(self):
        from autonote.orchestrator import _slugify, _find_unique_vault_dest
        
        title = "Project: Alpha / Beta Review"
        slug = _slugify(title)
        
        assert "/" not in slug
        assert ":" not in slug
        
        with patch("pathlib.Path.exists", return_value=False):
            dest = _find_unique_vault_dest(Path("/vault"), slug)
            assert dest == Path(f"/vault/{slug}")
