"""Tests for obsidian/update_index.py."""

import pytest
from pathlib import Path
from unittest.mock import patch

from autonote.obsidian.update_index import (
    parse_frontmatter,
    build_row,
    entry_already_exists,
    ensure_index_header,
    migrate_index_format,
    run_update_index,
)


class TestParseFrontmatter:

    def test_parses_basic_fields(self):
        content = "---\ntitle: My Meeting\ndate: 2026-04-01\n---\nBody text."
        fm = parse_frontmatter(content)
        assert fm["title"] == "My Meeting"
        assert fm["date"] == "2026-04-01"

    def test_returns_empty_dict_when_no_frontmatter(self):
        assert parse_frontmatter("No frontmatter here.") == {}

    def test_returns_empty_dict_when_no_closing_dashes(self):
        content = "---\ntitle: broken\n"
        assert parse_frontmatter(content) == {}

    def test_parses_list_field(self):
        content = '---\ntags: [meeting, work]\nparticipants: ["Alice", "Bob"]\n---\n'
        fm = parse_frontmatter(content)
        assert fm["tags"] == ["meeting", "work"]
        assert fm["participants"] == ["Alice", "Bob"]

    def test_parses_empty_list(self):
        content = "---\ntags: []\n---\n"
        fm = parse_frontmatter(content)
        assert fm["tags"] == []

    def test_strips_quotes_from_values(self):
        content = '---\ntitle: "Quoted Title"\n---\n'
        fm = parse_frontmatter(content)
        assert fm["title"] == "Quoted Title"

    def test_ignores_lines_without_colon(self):
        content = "---\ntitle: Good\nnot a key value pair\n---\n"
        fm = parse_frontmatter(content)
        assert "not a key value pair" not in fm
        assert fm["title"] == "Good"


class TestBuildRow:

    def test_basic_row(self):
        summary_path = Path("/vault/meetings/meeting_20260401_100000_summary.md")
        fm = {"date": "2026-04-01", "time": "10:00", "title": "Standup", "tags": ["standup"], "participants": ["Alice"]}
        row = build_row(summary_path, fm)
        assert "2026-04-01" in row
        assert "10:00" in row
        assert "Standup" in row
        assert "#standup" in row
        assert "Alice" in row

    def test_computes_day_of_week(self):
        summary_path = Path("/vault/meeting_summary.md")
        fm = {"date": "2026-04-01", "time": "09:00", "title": "Test", "tags": [], "participants": []}
        row = build_row(summary_path, fm)
        assert "Wednesday" in row  # 2026-04-01 is a Wednesday

    def test_falls_back_to_stem_when_no_title(self):
        summary_path = Path("/vault/meeting_20260401_100000_summary.md")
        fm = {"date": "2026-04-01", "time": "", "title": "", "tags": [], "participants": []}
        row = build_row(summary_path, fm)
        assert "meeting_20260401_100000_summary" in row

    def test_empty_participants_and_tags(self):
        summary_path = Path("/vault/meeting_summary.md")
        fm = {"date": "2026-04-01", "time": "", "title": "T", "tags": [], "participants": []}
        row = build_row(summary_path, fm)
        # Tags and participants columns should be empty
        parts = row.split("|")
        assert len(parts) >= 7  # | link | date | time | day | tags | participants |

    def test_invalid_date_skips_day(self):
        summary_path = Path("/vault/meeting_summary.md")
        fm = {"date": "not-a-date", "time": "", "title": "T", "tags": [], "participants": []}
        row = build_row(summary_path, fm)
        # Should not crash; day column will be empty
        assert "T" in row


class TestEntryAlreadyExists:

    def test_returns_false_when_index_does_not_exist(self, tmp_path):
        index = tmp_path / "index.md"
        assert entry_already_exists(index, "meeting_20260401") is False

    def test_returns_true_when_stem_in_index(self, tmp_path):
        index = tmp_path / "index.md"
        index.write_text("| [[meeting_20260401_summary]] | ... |\n")
        assert entry_already_exists(index, "meeting_20260401_summary") is True

    def test_returns_false_when_stem_not_in_index(self, tmp_path):
        index = tmp_path / "index.md"
        index.write_text("| [[other_meeting]] | ... |\n")
        assert entry_already_exists(index, "meeting_20260401_summary") is False


class TestEnsureIndexHeader:

    def test_creates_index_when_missing(self, tmp_path):
        index = tmp_path / "Meetings.md"
        ensure_index_header(index)
        content = index.read_text()
        assert "# Meetings" in content
        assert "| Meeting |" in content

    def test_does_not_overwrite_existing_index(self, tmp_path):
        index = tmp_path / "Meetings.md"
        index.write_text("# Meetings\n\n| Meeting | Date | Time | Day | Tags | Participants |\n|---------|------|------|-----|------|--------------|\n")
        ensure_index_header(index)
        # Should still have the header
        content = index.read_text()
        assert "# Meetings" in content

    def test_migrates_old_format(self, tmp_path):
        index = tmp_path / "Meetings.md"
        old_content = (
            "# Meetings\n\n"
            "| Meeting | Date | Tags | Participants |\n"
            "|---------|------|------|--------------|\n"
            "| [[meeting_20260330_120000_summary]] | 2026-03-30 | #standup | Alice |\n"
        )
        index.write_text(old_content)
        ensure_index_header(index)
        new_content = index.read_text()
        assert "| Time |" in new_content or "Time" in new_content


class TestMigrateIndexFormat:

    def test_migrates_header_and_rows(self, tmp_path):
        index = tmp_path / "Meetings.md"
        old_content = (
            "# Meetings\n\n"
            "| Meeting | Date | Tags | Participants |\n"
            "|---------|------|------|--------------|\n"
            "| [[meeting_20260330_120000_summary\\|2026-03-30 Standup]] | 2026-03-30 | #standup | Alice |\n"
        )
        index.write_text(old_content)
        migrate_index_format(index)
        new_content = index.read_text()
        assert "Time" in new_content
        assert "Day" in new_content
        assert "12:00" in new_content  # extracted from filename


class TestRunUpdateIndex:

    def test_raises_when_summary_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_update_index(str(tmp_path / "missing.md"), str(tmp_path / "index.md"))

    def test_skips_when_entry_already_exists(self, tmp_path):
        summary = tmp_path / "meeting_20260401_summary.md"
        summary.write_text("---\ntitle: T\ndate: 2026-04-01\n---\n")
        index = tmp_path / "Meetings.md"
        index.write_text("meeting_20260401_summary already here\n")
        result = run_update_index(str(summary), str(index))
        assert result is None

    def test_creates_index_and_appends_row(self, tmp_path):
        summary = tmp_path / "meeting_20260401_120000_summary.md"
        summary.write_text('---\ntitle: My Meeting\ndate: 2026-04-01\ntime: 12:00\ntags: [standup]\nparticipants: [Alice]\n---\n')
        index = tmp_path / "Meetings.md"
        result = run_update_index(str(summary), str(index))
        assert result is not None
        content = index.read_text()
        assert "My Meeting" in content

    def test_does_not_duplicate_entries(self, tmp_path):
        summary = tmp_path / "meeting_20260401_summary.md"
        summary.write_text('---\ntitle: T\ndate: 2026-04-01\n---\n')
        index = tmp_path / "Meetings.md"

        run_update_index(str(summary), str(index))
        run_update_index(str(summary), str(index))

        content = index.read_text()
        assert content.count("meeting_20260401_summary") == 1
