import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from autonote.audio.transcribe import format_timestamp, save_transcription


class TestFormatTimestamp:

    def test_zero(self):
        assert format_timestamp(0.0) == "00:00:00,000"

    def test_one_hour(self):
        assert format_timestamp(3600.0) == "01:00:00,000"

    def test_mixed(self):
        # 1h 2m 3.456s
        seconds = 3600 + 120 + 3.456
        assert format_timestamp(seconds) == "01:02:03,456"

    def test_milliseconds(self):
        assert format_timestamp(0.5) == "00:00:00,500"

    def test_no_fractional_millis_truncated(self):
        result = format_timestamp(61.0)
        assert result == "00:01:01,000"


class TestSaveTranscription:

    def test_saves_txt(self, tmp_path):
        result = {"text": "Hello world.", "segments": [], "language": "en"}
        out = str(tmp_path / "out.txt")
        save_transcription(result, out, "txt")
        assert Path(out).read_text() == "Hello world."

    def test_saves_json(self, tmp_path):
        import json
        result = {"text": "Hello.", "segments": [], "language": "en"}
        out = str(tmp_path / "out.json")
        save_transcription(result, out, "json")
        data = json.loads(Path(out).read_text())
        assert data["text"] == "Hello."

    def test_saves_srt(self, tmp_path):
        result = {
            "text": "Hello.",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello."},
                {"start": 1.5, "end": 3.0, "text": "World."},
            ],
        }
        out = str(tmp_path / "out.srt")
        save_transcription(result, out, "srt")
        content = Path(out).read_text()
        assert "00:00:00,000 --> 00:00:01,500" in content
        assert "Hello." in content
        assert "World." in content

    def test_saves_vtt(self, tmp_path):
        result = {
            "text": "Hi.",
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hi."}],
        }
        out = str(tmp_path / "out.vtt")
        save_transcription(result, out, "vtt")
        content = Path(out).read_text()
        assert "WEBVTT" in content
        assert "Hi." in content
