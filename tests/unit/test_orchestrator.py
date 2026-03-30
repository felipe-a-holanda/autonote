import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


class TestSlugify:
    
    def test_slugify_basic(self):
        from autonote.orchestrator import _slugify
        
        assert _slugify("My Meeting Title") == "My Meeting Title"
        assert _slugify("Test/Meeting") == "Test Meeting"
        assert _slugify("Test\\Meeting") == "Test Meeting"
    
    def test_slugify_special_chars(self):
        from autonote.orchestrator import _slugify
        
        assert _slugify("Meeting: Q1 Review") == "Meeting Q1 Review"
        assert _slugify('File*Name?"Test"') == "FileNameTest"
        assert _slugify("Test<>|Meeting") == "TestMeeting"
    
    def test_slugify_whitespace(self):
        from autonote.orchestrator import _slugify
        
        assert _slugify("  Multiple   Spaces  ") == "Multiple Spaces"
        assert _slugify("Test---Meeting") == "Test Meeting"
    
    def test_slugify_max_length(self):
        from autonote.orchestrator import _slugify
        
        long_title = "This is a very long meeting title that exceeds the maximum length"
        result = _slugify(long_title, max_len=30)
        assert len(result) <= 30
        assert result == "This is a very long meeting"
    
    def test_slugify_empty_returns_default(self):
        from autonote.orchestrator import _slugify
        
        assert _slugify("") == "meeting"
        assert _slugify("   ") == "meeting"
        assert _slugify(":::") == "meeting"


class TestResolveVaultTitle:
    
    @patch("pathlib.Path.exists", return_value=False)
    def test_resolve_vault_title_no_file(self, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "12:30"
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_with_frontmatter_title(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
title: My Custom Title
---
# Meeting Summary
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "My Custom Title"
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_inferred_from_heading(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
---
# Project Alpha Discussion
# Meeting Summary
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "Project Alpha Discussion"
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_skips_boilerplate(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
---
# Meeting Summary
# Action Items
# Project Kickoff
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "Project Kickoff"
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_removes_emojis(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
---
# 🚀 Sprint Planning
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "🚀 Sprint Planning"
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_resolve_vault_title_fallback_to_time(self, mock_read, mock_exists):
        from autonote.orchestrator import _resolve_vault_title
        
        mock_read.return_value = """---
---
# Meeting Summary
# Overview
Content here"""
        
        result = _resolve_vault_title("/tmp/summary.md", "12:30")
        assert result == "12:30"
    
    def test_resolve_vault_title_none_file(self):
        from autonote.orchestrator import _resolve_vault_title
        
        result = _resolve_vault_title(None, "14:45")
        assert result == "14:45"


class TestFindUniqueVaultDest:
    
    @patch("pathlib.Path.exists", return_value=False)
    def test_find_unique_vault_dest_no_conflict(self, mock_exists):
        from autonote.orchestrator import _find_unique_vault_dest
        
        result = _find_unique_vault_dest(Path("/vault"), "meeting-folder")
        assert result == Path("/vault/meeting-folder")
    
    @patch("pathlib.Path.exists")
    def test_find_unique_vault_dest_with_conflict(self, mock_exists):
        from autonote.orchestrator import _find_unique_vault_dest
        
        mock_exists.side_effect = [True, True, False]
        
        result = _find_unique_vault_dest(Path("/vault"), "meeting-folder")
        assert result == Path("/vault/meeting-folder (3)")
