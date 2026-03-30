import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestParseTimestampFromFilename:
    
    def test_parse_timestamp_valid_filename(self):
        from autonote.obsidian.frontmatter import parse_timestamp_from_filename
        
        date, time = parse_timestamp_from_filename("meeting_20260330_143025_summary.md")
        
        assert date == "2026-03-30"
        assert time == "14:30"
    
    def test_parse_timestamp_no_match(self):
        from autonote.obsidian.frontmatter import parse_timestamp_from_filename
        
        date, time = parse_timestamp_from_filename("random_file.md")
        
        assert date == ""
        assert time == ""


class TestReadMetadataJson:
    
    @patch("pathlib.Path.exists", return_value=False)
    def test_read_metadata_json_file_not_exists(self, mock_exists):
        from autonote.obsidian.frontmatter import read_metadata_json
        
        result = read_metadata_json(Path("/tmp/metadata.json"))
        
        assert result == {}
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_read_metadata_json_valid(self, mock_read, mock_exists):
        from autonote.obsidian.frontmatter import read_metadata_json
        
        mock_read.return_value = '{"title": "Test Meeting", "participants": ["Alice", "Bob"]}'
        
        result = read_metadata_json(Path("/tmp/metadata.json"))
        
        assert result["title"] == "Test Meeting"
        assert "Alice" in result["participants"]
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_read_metadata_json_invalid_json(self, mock_read, mock_exists):
        from autonote.obsidian.frontmatter import read_metadata_json
        
        mock_read.return_value = "not valid json"
        
        result = read_metadata_json(Path("/tmp/metadata.json"))
        
        assert result == {}
