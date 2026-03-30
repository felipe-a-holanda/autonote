import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from autonote.llm import resolve_model, query_llm, _recording_base, _append_cost_log


class TestResolveModel:
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_preset_local(self):
        with patch("autonote.llm.LLM_PRESETS", {"local": "ollama/llama3.2:3b"}):
            assert resolve_model("local") == "ollama/llama3.2:3b"
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_preset_fast(self):
        with patch("autonote.llm.LLM_PRESETS", {"fast": "openai/gpt-4o"}):
            assert resolve_model("fast") == "openai/gpt-4o"
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_preset_smart(self):
        with patch("autonote.llm.LLM_PRESETS", {"smart": "anthropic/claude-3-5-sonnet-20241022"}):
            assert resolve_model("smart") == "anthropic/claude-3-5-sonnet-20241022"
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_fully_qualified_model(self):
        assert resolve_model("openai/gpt-4o") == "openai/gpt-4o"
        assert resolve_model("anthropic/claude-3-5-sonnet-20241022") == "anthropic/claude-3-5-sonnet-20241022"
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_bare_model_name(self):
        assert resolve_model("llama3.2:3b") == "ollama/llama3.2:3b"
        assert resolve_model("mistral") == "ollama/mistral"
    
    @patch("autonote.llm.config", {"MODEL": "ollama/llama3.1:8b"})
    def test_resolve_none_uses_default(self):
        assert resolve_model(None) == "ollama/llama3.1:8b"


class TestRecordingBase:
    
    def test_recording_base_with_formatted_suffix(self):
        source = "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000_formatted.md"
        recording_dir, base_stem = _recording_base(source)
        
        assert recording_dir == Path("/recordings/20260330/meeting_20260330_120000")
        assert base_stem == "meeting_20260330_120000"
    
    def test_recording_base_with_summary_suffix(self):
        source = "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000_summary.md"
        recording_dir, base_stem = _recording_base(source)
        
        assert recording_dir == Path("/recordings/20260330/meeting_20260330_120000")
        assert base_stem == "meeting_20260330_120000"
    
    def test_recording_base_with_metadata_suffix(self):
        source = "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000_extracted_metadata.json"
        recording_dir, base_stem = _recording_base(source)
        
        assert recording_dir == Path("/recordings/20260330/meeting_20260330_120000")
        assert base_stem == "meeting_20260330_120000"
    
    def test_recording_base_without_suffix(self):
        source = "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000.txt"
        recording_dir, base_stem = _recording_base(source)
        
        assert recording_dir == Path("/recordings/20260330/meeting_20260330_120000")
        assert base_stem == "meeting_20260330_120000"


class TestQueryLLM:
    
    @patch("autonote.llm.litellm.completion")
    @patch("autonote.llm.config", {
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_COST_LOG": None,
        "USD_TO_BRL": "5.50"
    })
    def test_query_llm_with_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_completion.return_value = mock_response
        
        result = query_llm(prompt="Test prompt", model="ollama/llama3.1:8b")
        
        assert result == "Test response"
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "ollama/llama3.1:8b"
        assert call_args[1]["messages"] == [{"role": "user", "content": "Test prompt"}]
    
    @patch("autonote.llm.litellm.completion")
    @patch("autonote.llm.config", {
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_COST_LOG": None,
        "USD_TO_BRL": "5.50"
    })
    def test_query_llm_with_messages(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_completion.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Test prompt"}
        ]
        result = query_llm(messages=messages, model="ollama/llama3.1:8b")
        
        assert result == "Test response"
        call_args = mock_completion.call_args
        assert call_args[1]["messages"] == messages
    
    @patch("autonote.llm.litellm.completion")
    @patch("autonote.llm.config", {
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_COST_LOG": None,
        "USD_TO_BRL": "5.50"
    })
    def test_query_llm_with_preset(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_completion.return_value = mock_response
        
        with patch("autonote.llm.LLM_PRESETS", {"fast": "openai/gpt-4o"}):
            result = query_llm(prompt="Test", model="fast")
            
            call_args = mock_completion.call_args
            assert call_args[1]["model"] == "openai/gpt-4o"
    
    def test_query_llm_no_prompt_or_messages_raises(self):
        with pytest.raises(ValueError, match="Must provide either prompt or messages"):
            query_llm()
    
    @patch("autonote.llm.litellm.completion")
    @patch("autonote.llm.config", {
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "LLM_COST_LOG": None,
        "USD_TO_BRL": "5.50"
    })
    def test_query_llm_error_handling(self, mock_completion):
        mock_completion.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="LLM query failed"):
            query_llm(prompt="Test")


class TestAppendCostLog:
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("autonote.llm.config", {"LLM_COST_LOG": "/tmp/costs.jsonl"})
    @patch("autonote.llm._write_recording_cost")
    def test_append_cost_log_writes_global_log(self, mock_write_rec, mock_file):
        _append_cost_log(
            model="openai/gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
            cost_brl=0.0275,
            source_file="/recordings/test/meeting.txt",
            stage="summarize",
            duration_s=2.5
        )
        
        mock_file.assert_called_once_with("/tmp/costs.jsonl", "a")
        mock_write_rec.assert_called_once()
    
    @patch("autonote.llm.config", {"LLM_COST_LOG": None})
    @patch("autonote.llm._write_recording_cost")
    def test_append_cost_log_no_global_log(self, mock_write_rec):
        _append_cost_log(
            model="openai/gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
            cost_brl=0.0275,
            source_file="/recordings/test/meeting.txt",
            stage="summarize",
            duration_s=2.5
        )
        
        mock_write_rec.assert_called_once()
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("autonote.llm.config", {"LLM_COST_LOG": "/tmp/costs.jsonl"})
    @patch("autonote.llm._write_recording_cost")
    def test_append_cost_log_handles_write_error(self, mock_write_rec, mock_file):
        mock_file.side_effect = Exception("Write error")
        
        _append_cost_log(
            model="openai/gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.005,
            cost_brl=0.0275,
            source_file="/recordings/test/meeting.txt",
            stage="summarize",
            duration_s=2.5
        )
        
        mock_write_rec.assert_called_once()


class TestWriteRecordingCost:
    
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.write_text")
    def test_write_recording_cost_new_file(self, mock_write, mock_exists):
        from autonote.llm import _write_recording_cost
        
        entry = {
            "ts": "2026-03-30T12:00:00Z",
            "stage": "summarize",
            "model": "openai/gpt-4o",
            "duration_s": 2.5,
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_total": 150,
            "cost_usd": 0.005,
            "cost_brl": 0.0275,
        }
        
        _write_recording_cost(entry, "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000.txt")
        
        mock_write.assert_called_once()
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text", return_value='[{"stage": "reformat"}]')
    @patch("pathlib.Path.write_text")
    def test_write_recording_cost_append_to_existing(self, mock_write, mock_read, mock_exists):
        from autonote.llm import _write_recording_cost
        
        entry = {
            "ts": "2026-03-30T12:00:00Z",
            "stage": "summarize",
            "model": "openai/gpt-4o",
            "duration_s": 2.5,
            "tokens_in": 100,
            "tokens_out": 50,
            "tokens_total": 150,
            "cost_usd": 0.005,
            "cost_brl": 0.0275,
        }
        
        _write_recording_cost(entry, "/recordings/20260330/meeting_20260330_120000/meeting_20260330_120000.txt")
        
        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][0]
        assert "reformat" in written_data
        assert "summarize" in written_data
