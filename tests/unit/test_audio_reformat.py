"""Tests for audio/reformat.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autonote.audio.reformat import chunk_transcription, load_transcription, run_reformat


class TestChunkTranscription:

    def test_short_text_single_chunk(self):
        text = "Hello world. This is a short transcript."
        chunks = chunk_transcription(text, max_words=100)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_splits_at_sentence_boundary(self):
        # "end." is the 401st word; len(current_chunk)==401 >= 400 and ends in "." → split
        words = ["word"] * 400 + ["end."] + ["extra"] * 10
        text = " ".join(words)
        chunks = chunk_transcription(text, max_words=400)
        assert len(chunks) == 2

    def test_hard_splits_at_max_plus_150(self):
        # 551 words with no sentence boundary
        words = ["word"] * 551
        text = " ".join(words)
        chunks = chunk_transcription(text, max_words=400)
        assert len(chunks) == 2

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_transcription("", max_words=100)
        assert chunks == []

    def test_remainder_chunk_included(self):
        words = ["w"] * 10
        text = " ".join(words)
        chunks = chunk_transcription(text, max_words=100)
        assert len(chunks) == 1
        assert len(chunks[0].split()) == 10


class TestLoadTranscription:

    def test_loads_txt_file(self, tmp_path):
        txt = tmp_path / "transcript.txt"
        txt.write_text("Hello from text file.")
        result = load_transcription(str(txt))
        assert result == "Hello from text file."

    def test_loads_json_file(self, tmp_path):
        jf = tmp_path / "transcript.json"
        jf.write_text(json.dumps({"text": "Hello from JSON."}))
        result = load_transcription(str(jf))
        assert result == "Hello from JSON."

    def test_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transcription(str(tmp_path / "missing.txt"))

    def test_json_missing_text_key_returns_empty(self, tmp_path):
        jf = tmp_path / "transcript.json"
        jf.write_text(json.dumps({"other": "data"}))
        result = load_transcription(str(jf))
        assert result == ""


class TestRunReformat:

    @patch("autonote.audio.reformat.query_llm", return_value="Cleaned transcript text.")
    @patch("autonote.audio.reformat.config", {"MODEL_REFORMAT": "ollama/llama3", "OLLAMA_URL": "http://localhost:11434"})
    def test_creates_output_file(self, mock_llm, tmp_path):
        input_file = tmp_path / "transcript.txt"
        input_file.write_text("Raw transcription here. Some words to fill.")

        result = run_reformat(str(input_file))

        assert Path(result).exists()
        assert "_formatted" in result

    @patch("autonote.audio.reformat.query_llm", return_value="Cleaned.")
    @patch("autonote.audio.reformat.config", {"MODEL_REFORMAT": "ollama/llama3", "OLLAMA_URL": "http://localhost:11434"})
    def test_respects_custom_output_path(self, mock_llm, tmp_path):
        input_file = tmp_path / "transcript.txt"
        input_file.write_text("Some content.")
        output_file = str(tmp_path / "custom_output.md")

        result = run_reformat(str(input_file), output_file=output_file)

        assert result == output_file
        assert Path(output_file).read_text().strip() == "Cleaned."

    @patch("autonote.audio.reformat.config", {"MODEL_REFORMAT": "ollama/llama3", "OLLAMA_URL": "http://localhost:11434"})
    def test_raises_on_empty_transcription(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   ")
        with pytest.raises(ValueError, match="empty"):
            run_reformat(str(empty))

    @patch("autonote.audio.reformat.query_llm", return_value="Cleaned.")
    @patch("autonote.audio.reformat.config", {"MODEL_REFORMAT": "ollama/llama3", "OLLAMA_URL": "http://localhost:11434"})
    def test_strips_llm_preamble(self, mock_llm, tmp_path):
        mock_llm.return_value = "Here is the cleaned transcript:\nActual content."
        input_file = tmp_path / "transcript.txt"
        input_file.write_text("Some content for the transcript.")
        output_file = str(tmp_path / "out.md")

        run_reformat(str(input_file), output_file=output_file)

        content = Path(output_file).read_text()
        assert "Here is the cleaned transcript:" not in content
        assert "Actual content." in content
