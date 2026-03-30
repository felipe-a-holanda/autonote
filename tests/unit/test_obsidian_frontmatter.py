import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestParseExistingFrontmatter:
    
    def test_parse_frontmatter_with_valid_yaml(self):
        from autonote.obsidian.frontmatter import parse_existing_frontmatter
        from datetime import date
        
        content = """---
title: My Meeting
date: 2026-03-30
tags: [meeting, project]
---
# Meeting Content
Some text here"""
        
        fm, body = parse_existing_frontmatter(content)
        
        assert fm["title"] == "My Meeting"
        assert fm["date"] == date(2026, 3, 30)
        assert "Meeting Content" in body
    
    def test_parse_frontmatter_no_frontmatter(self):
        from autonote.obsidian.frontmatter import parse_existing_frontmatter
        
        content = "# Just a heading\nSome content"
        
        fm, body = parse_existing_frontmatter(content)
        
        assert fm == {}
        assert body == content
    
    def test_parse_frontmatter_empty_frontmatter(self):
        from autonote.obsidian.frontmatter import parse_existing_frontmatter
        
        content = """---
---
# Content"""
        
        fm, body = parse_existing_frontmatter(content)
        
        assert fm == {}
        assert "# Content" in body
    
    def test_parse_frontmatter_incomplete(self):
        from autonote.obsidian.frontmatter import parse_existing_frontmatter
        
        content = """---
title: Test
# Missing closing ---"""
        
        fm, body = parse_existing_frontmatter(content)
        
        assert fm == {}
        assert "title: Test" in body


class TestRenderFrontmatter:
    
    def test_render_frontmatter_basic(self):
        from autonote.obsidian.frontmatter import render_frontmatter
        
        fm = {
            "title": "Test Meeting",
            "date": "2026-03-30",
            "tags": ["meeting"]
        }
        
        result = render_frontmatter(fm)
        
        assert result.startswith("---\n")
        assert result.endswith("---")
        assert "title: Test Meeting" in result
        assert "date: '2026-03-30'" in result or "date: 2026-03-30" in result
    
    def test_render_frontmatter_empty(self):
        from autonote.obsidian.frontmatter import render_frontmatter
        
        result = render_frontmatter({})
        
        assert result == ""
    
    def test_render_frontmatter_with_lists(self):
        from autonote.obsidian.frontmatter import render_frontmatter
        
        fm = {
            "tags": ["tag1", "tag2", "tag3"],
            "participants": ["Alice", "Bob"]
        }
        
        result = render_frontmatter(fm)
        
        assert "tags:" in result
        assert "participants:" in result
