"""Tests for audio/summarize.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from autonote.audio.summarize import load_transcription, save_summary, run_summarize


class TestLoadTranscription:

    def test_loads_txt_file(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_text("Hello there.")
        assert load_transcription(str(f)) == "Hello there."

    def test_loads_json_file(self, tmp_path):
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps({"text": "JSON text."}))
        assert load_transcription(str(f)) == "JSON text."

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transcription(str(tmp_path / "nope.txt"))


class TestSaveSummary:

    def test_saves_md_format(self, tmp_path):
        output = str(tmp_path / "out.md")
        save_summary({"summary": "Summary text.", "action_items": "- Do X"}, output, "md")
        content = Path(output).read_text()
        assert "# Meeting Summary" in content
        assert "Summary text." in content
        assert "# Action Items" in content
        assert "- Do X" in content

    def test_saves_txt_format(self, tmp_path):
        output = str(tmp_path / "out.txt")
        save_summary({"summary": "Summary text."}, output, "txt")
        content = Path(output).read_text()
        assert "Summary text." in content

    def test_saves_json_format(self, tmp_path):
        output = str(tmp_path / "out.json")
        save_summary({"summary": "Summary text.", "action_items": "- Do Y"}, output, "json")
        data = json.loads(Path(output).read_text())
        assert data["summary"] == "Summary text."

    def test_no_action_items_section_when_absent(self, tmp_path):
        output = str(tmp_path / "out.md")
        save_summary({"summary": "Just a summary."}, output, "md")
        content = Path(output).read_text()
        assert "# Action Items" not in content


class TestRunSummarize:

    @patch("autonote.audio.summarize.query_llm", return_value="Generated summary.")
    @patch("autonote.audio.summarize.config", {
        "MODEL_SUMMARIZE": "ollama/llama3",
        "OLLAMA_URL": "http://localhost:11434",
    })
    def test_creates_output_file(self, mock_llm, tmp_path):
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("Alice: Let's sync up. Bob: Agreed.")

        result = run_summarize(str(transcript))

        assert Path(result).exists()
        assert "_summary" in result

    @patch("autonote.audio.summarize.query_llm", return_value="Summary.")
    @patch("autonote.audio.summarize.config", {
        "MODEL_SUMMARIZE": "ollama/llama3",
        "OLLAMA_URL": "http://localhost:11434",
    })
    def test_respects_custom_output_path(self, mock_llm, tmp_path):
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("Some content.")
        output = str(tmp_path / "custom.md")

        result = run_summarize(str(transcript), output_file=output)
        assert result == output

    @patch("autonote.audio.summarize.config", {
        "MODEL_SUMMARIZE": "ollama/llama3",
        "OLLAMA_URL": "http://localhost:11434",
    })
    def test_raises_on_empty_transcription(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   ")
        with pytest.raises(ValueError, match="empty"):
            run_summarize(str(empty))

    @patch("autonote.audio.summarize.query_llm", return_value="Summary.")
    @patch("autonote.audio.summarize.config", {
        "MODEL_SUMMARIZE": "ollama/llama3",
        "OLLAMA_URL": "http://localhost:11434",
    })
    def test_skip_action_items_flag(self, mock_llm, tmp_path):
        transcript = tmp_path / "meeting.txt"
        transcript.write_text("Some content.")

        run_summarize(str(transcript), skip_action_items=True)

        # query_llm should have been called only once (summary only, no action items)
        assert mock_llm.call_count == 1
