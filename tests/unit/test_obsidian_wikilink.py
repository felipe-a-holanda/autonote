import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestLoadEntities:
    
    @patch("pathlib.Path.exists", return_value=False)
    def test_load_entities_file_not_exists(self, mock_exists):
        from autonote.obsidian.wikilink import load_entities
        
        result = load_entities(Path("/tmp/entities.yml"))
        
        assert result == {"people": [], "products": []}
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_load_entities_with_people_and_products(self, mock_read, mock_exists):
        from autonote.obsidian.wikilink import load_entities
        
        mock_read.return_value = """# Entities file
people:
  - Alice Smith
  - Bob Jones
  - "Charlie Brown"

products:
  - Product A
  - Product B
"""
        
        result = load_entities(Path("/tmp/entities.yml"))
        
        assert "Alice Smith" in result["people"]
        assert "Bob Jones" in result["people"]
        assert "Charlie Brown" in result["people"]
        assert "Product A" in result["products"]
        assert "Product B" in result["products"]
    
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_load_entities_ignores_comments(self, mock_read, mock_exists):
        from autonote.obsidian.wikilink import load_entities
        
        mock_read.return_value = """# Comment line
people:
  # Another comment
  - Alice
  - Bob
"""
        
        result = load_entities(Path("/tmp/entities.yml"))
        
        assert len(result["people"]) == 2
        assert "Alice" in result["people"]


class TestSplitSections:
    
    def test_split_sections_no_frontmatter(self):
        from autonote.obsidian.wikilink import split_sections
        
        content = "# Meeting\nSome text here"
        
        frontmatter, segments = split_sections(content)
        
        assert frontmatter == ""
        assert len(segments) > 0
    
    def test_split_sections_with_frontmatter(self):
        from autonote.obsidian.wikilink import split_sections
        
        content = """---
title: Test
---
# Meeting"""
        
        frontmatter, segments = split_sections(content)
        
        assert "title: Test" in frontmatter
        assert any("Meeting" in seg[0] for seg in segments)
    
    def test_split_sections_preserves_code_blocks(self):
        from autonote.obsidian.wikilink import split_sections
        
        content = """# Meeting
Some text
```python
code here
```
More text"""
        
        frontmatter, segments = split_sections(content)
        
        code_segments = [seg for seg, protected in segments if protected and "```" in seg]
        assert len(code_segments) > 0
    
    def test_split_sections_preserves_wikilinks(self):
        from autonote.obsidian.wikilink import split_sections
        
        content = "Text with [[existing link]] here"
        
        frontmatter, segments = split_sections(content)
        
        protected_segments = [seg for seg, protected in segments if protected]
        assert any("[[existing link]]" in seg for seg in protected_segments)
