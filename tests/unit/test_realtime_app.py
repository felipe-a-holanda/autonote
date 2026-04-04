"""Tests for realtime app helpers (non-TUI logic)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autonote.realtime.models import (
    ActionItem,
    ActionItemsUpdate,
    ContradictionAlert,
    CustomPromptResult,
    ReplySuggestion,
    SummaryUpdate,
    TranscriptSegment,
)


# ---------------------------------------------------------------------------
# _append_transcript  (pure file I/O — no TUI needed)
# ---------------------------------------------------------------------------

class TestAppendTranscript:
    def _make_app(self):
        from autonote.realtime.app import RealtimeApp
        return RealtimeApp(api_key="test", model=None, title="Test")

    def test_writes_jsonl_line(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        segment = TranscriptSegment(
            speaker="Me",
            text="Hello world",
            timestamp_start=0.0,
            timestamp_end=1.5,
        )
        app._append_transcript(segment)

        lines = transcript_path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["speaker"] == "Me"
        assert record["text"] == "Hello world"
        assert record["start"] == 0.0
        assert record["end"] == 1.5
        assert "wall_time" in record

    def test_appends_multiple_lines(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        for i in range(3):
            seg = TranscriptSegment(
                speaker="Me",
                text=f"Message {i}",
                timestamp_start=float(i),
                timestamp_end=float(i + 1),
            )
            app._append_transcript(seg)

        lines = transcript_path.read_text().splitlines()
        assert len(lines) == 3

    def test_no_op_when_path_is_none(self, tmp_path):
        app = self._make_app()
        app._transcript_path = None  # Not set

        segment = TranscriptSegment(
            speaker="Me",
            text="Should not be written",
            timestamp_start=0.0,
            timestamp_end=1.0,
        )
        app._append_transcript(segment)  # Should not raise

    def test_timestamps_rounded_to_3_decimals(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        seg = TranscriptSegment(
            speaker="Them",
            text="Precise timing",
            timestamp_start=1.23456789,
            timestamp_end=2.987654321,
        )
        app._append_transcript(seg)

        record = json.loads(transcript_path.read_text().strip())
        # Should be rounded to 3 decimal places
        assert record["start"] == round(1.23456789, 3)
        assert record["end"] == round(2.987654321, 3)

    def test_wall_time_is_iso_format(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        seg = TranscriptSegment(
            speaker="Me",
            text="Timing test",
            timestamp_start=0.0,
            timestamp_end=1.0,
        )
        app._append_transcript(seg)

        record = json.loads(transcript_path.read_text().strip())
        wall_time = record["wall_time"]
        # Should match ISO format like "2026-04-04T12:00:00Z"
        assert "T" in wall_time
        assert wall_time.endswith("Z")


# ---------------------------------------------------------------------------
# DebugLog.export (no running app needed, just test the data structure)
# ---------------------------------------------------------------------------

class TestDebugLogExport:
    def test_export_returns_joined_lines(self):
        from autonote.realtime.app import DebugLog

        log = DebugLog.__new__(DebugLog)
        log._lines = ["[10:00:00] First message", "[10:00:01] Second message"]

        result = log.export()
        assert result == "[10:00:00] First message\n[10:00:01] Second message"

    def test_export_empty(self):
        from autonote.realtime.app import DebugLog

        log = DebugLog.__new__(DebugLog)
        log._lines = []

        result = log.export()
        assert result == ""


# ---------------------------------------------------------------------------
# Widget CSS defaults (structural, no DOM needed)
# ---------------------------------------------------------------------------

class TestWidgetDefaults:
    def test_transcript_log_has_css(self):
        from autonote.realtime.app import TranscriptLog
        assert "border" in TranscriptLog.DEFAULT_CSS

    def test_summary_panel_has_css(self):
        from autonote.realtime.app import SummaryPanel
        assert "border" in SummaryPanel.DEFAULT_CSS

    def test_action_items_panel_has_css(self):
        from autonote.realtime.app import ActionItemsPanel
        assert "border" in ActionItemsPanel.DEFAULT_CSS

    def test_alerts_panel_has_css(self):
        from autonote.realtime.app import AlertsPanel
        assert "border" in AlertsPanel.DEFAULT_CSS

    def test_debug_log_has_css(self):
        from autonote.realtime.app import DebugLog
        assert "border" in DebugLog.DEFAULT_CSS


# ---------------------------------------------------------------------------
# RealtimeApp constructor
# ---------------------------------------------------------------------------

class TestRealtimeAppInit:
    def test_init_stores_params(self):
        from autonote.realtime.app import RealtimeApp
        app = RealtimeApp(api_key="key123", model="gpt4", title="My Meeting")
        assert app._api_key == "key123"
        assert app._model == "gpt4"
        assert app._meeting_title == "My Meeting"

    def test_init_defaults(self):
        from autonote.realtime.app import RealtimeApp
        app = RealtimeApp()
        assert app._api_key is None
        assert app._model is None
        assert app._meeting_title == ""

    def test_init_pipeline_objects_none(self):
        from autonote.realtime.app import RealtimeApp
        app = RealtimeApp(api_key="key")
        assert app._recorder is None
        assert app._transcriber is None
        assert app._context_manager is None
        assert app._pipeline_task is None
        assert app._consumer_task is None
        assert app._status_task is None
        assert app._transcript_path is None


# ---------------------------------------------------------------------------
# run_realtime_app (just verify it creates and runs the app)
# ---------------------------------------------------------------------------

class TestRunRealtimeApp:
    def test_creates_and_runs_app(self):
        from autonote.realtime.app import run_realtime_app, RealtimeApp

        with patch.object(RealtimeApp, "run") as mock_run:
            run_realtime_app(api_key="test-key", model="default", title="Meeting")
            mock_run.assert_called_once()

    def test_passes_params_to_app(self):
        from autonote.realtime.app import run_realtime_app, RealtimeApp

        created_apps = []

        original_init = RealtimeApp.__init__

        def tracking_init(self, **kwargs):
            created_apps.append(kwargs)
            original_init(self, **kwargs)

        with patch.object(RealtimeApp, "__init__", tracking_init):
            with patch.object(RealtimeApp, "run"):
                run_realtime_app(api_key="key", model="m", title="T")

        assert len(created_apps) == 1
        assert created_apps[0]["api_key"] == "key"
        assert created_apps[0]["model"] == "m"
        assert created_apps[0]["title"] == "T"


# ---------------------------------------------------------------------------
# _update_action_items formatting logic (test independently)
# ---------------------------------------------------------------------------

class TestActionItemsFormatting:
    """Test the text formatting for action items without running the TUI."""

    def _format_items(self, items: list[ActionItem]) -> str:
        """Replicate the formatting logic from _update_action_items."""
        if not items:
            return "No action items yet."
        lines = []
        for item in items:
            status_icon = {"new": "+", "updated": "~", "completed": "v"}.get(item.status, "?")
            assignee = f" @{item.assignee}" if item.assignee else ""
            lines.append(f"[{status_icon}] {item.description}{assignee}")
        return "\n".join(lines)

    def test_new_item(self):
        item = ActionItem(id="1", description="Deploy service", source_timestamp=10.0, status="new")
        result = self._format_items([item])
        assert "[+] Deploy service" in result

    def test_updated_item(self):
        item = ActionItem(id="1", description="Fix bug", source_timestamp=10.0, status="updated")
        result = self._format_items([item])
        assert "[~] Fix bug" in result

    def test_completed_item(self):
        item = ActionItem(id="1", description="Review PR", source_timestamp=10.0, status="completed")
        result = self._format_items([item])
        assert "[v] Review PR" in result

    def test_with_assignee(self):
        item = ActionItem(id="1", description="Implement auth", assignee="Alice", source_timestamp=10.0)
        result = self._format_items([item])
        assert "@Alice" in result

    def test_without_assignee(self):
        item = ActionItem(id="1", description="Implement auth", source_timestamp=10.0)
        result = self._format_items([item])
        assert "@" not in result

    def test_empty_items_returns_placeholder(self):
        result = self._format_items([])
        assert result == "No action items yet."

    def test_multiple_items(self):
        items = [
            ActionItem(id="1", description="Task A", source_timestamp=1.0, status="new"),
            ActionItem(id="2", description="Task B", source_timestamp=2.0, status="completed"),
        ]
        result = self._format_items(items)
        lines = result.splitlines()
        assert len(lines) == 2
        assert "[+] Task A" in lines[0]
        assert "[v] Task B" in lines[1]
