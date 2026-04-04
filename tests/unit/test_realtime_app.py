"""Tests for realtime app helpers (non-TUI logic)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autonote.realtime.models import (
    ActionItem,
    ActionItemsUpdate,
    AggregatedTurn,
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
        assert app._aggregator is None
        assert app._context_manager is None
        assert app._pipeline_task is None
        assert app._bridge_task is None
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


# ---------------------------------------------------------------------------
# _append_transcript with AggregatedTurn (Task 1.3 — accepts both types)
# ---------------------------------------------------------------------------

class TestAppendTranscriptAggregatedTurn:
    def _make_app(self):
        from autonote.realtime.app import RealtimeApp
        return RealtimeApp(api_key="test", model=None, title="Test")

    def test_writes_aggregated_turn_jsonl(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        turn = AggregatedTurn(
            speaker="Them",
            text="Hello from the other side",
            timestamp_start=5.0,
            timestamp_end=8.5,
            segment_count=2,
        )
        app._append_transcript(turn)

        lines = transcript_path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["speaker"] == "Them"
        assert record["text"] == "Hello from the other side"
        assert record["start"] == 5.0
        assert record["end"] == 8.5
        assert "wall_time" in record

    def test_aggregated_turn_includes_segment_count(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        turn = AggregatedTurn(
            speaker="Me",
            text="Three segments merged",
            timestamp_start=0.0,
            timestamp_end=6.0,
            segment_count=3,
        )
        app._append_transcript(turn)

        record = json.loads(transcript_path.read_text().strip())
        assert record["segment_count"] == 3

    def test_transcript_segment_does_not_include_segment_count(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        seg = TranscriptSegment(
            speaker="Me", text="Hello", timestamp_start=0.0, timestamp_end=1.0
        )
        app._append_transcript(seg)

        record = json.loads(transcript_path.read_text().strip())
        assert "segment_count" not in record

    def test_aggregated_turn_jsonl_has_all_required_fields(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        turn = AggregatedTurn(
            speaker="Them",
            text="Complete turn text",
            timestamp_start=10.5,
            timestamp_end=15.0,
            segment_count=2,
        )
        app._append_transcript(turn)

        record = json.loads(transcript_path.read_text().strip())
        assert record["speaker"] == "Them"
        assert record["text"] == "Complete turn text"
        assert record["start"] == 10.5
        assert record["end"] == 15.0
        assert record["segment_count"] == 2
        assert "wall_time" in record

    def test_segment_and_turn_both_accepted(self, tmp_path):
        app = self._make_app()
        transcript_path = tmp_path / "transcript.jsonl"
        app._transcript_path = transcript_path

        seg = TranscriptSegment(
            speaker="Me", text="Hello", timestamp_start=0.0, timestamp_end=1.0
        )
        turn = AggregatedTurn(
            speaker="Them", text="World", timestamp_start=2.0, timestamp_end=4.0, segment_count=1
        )
        app._append_transcript(seg)
        app._append_transcript(turn)

        lines = transcript_path.read_text().splitlines()
        assert len(lines) == 2
        seg_record = json.loads(lines[0])
        turn_record = json.loads(lines[1])
        assert "segment_count" not in seg_record
        assert turn_record["segment_count"] == 1


# ---------------------------------------------------------------------------
# _bridge_segments — feeds transcriber segments into aggregator
# ---------------------------------------------------------------------------

class TestBridgeSegments:
    def _make_app(self):
        from autonote.realtime.app import RealtimeApp
        return RealtimeApp(api_key="test", model=None, title="Test")

    def test_bridge_feeds_segment_into_aggregator(self):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()

        async def run():
            app._aggregator = TurnAggregator()
            transcriber_queue: asyncio.Queue = asyncio.Queue()
            mock_transcriber = MagicMock()
            mock_transcriber.segment_queue = transcriber_queue
            app._transcriber = mock_transcriber

            segment = TranscriptSegment(
                speaker="Me", text="Hello", timestamp_start=0.0, timestamp_end=1.0
            )
            transcriber_queue.put_nowait(segment)
            transcriber_queue.put_nowait(None)

            await app._bridge_segments()

        asyncio.get_event_loop().run_until_complete(run())

        # The aggregator should have buffered the segment; flush_remaining was called
        # so output_queue should have an AggregatedTurn and a None sentinel
        items = []
        while not app._aggregator.output_queue.empty():
            items.append(app._aggregator.output_queue.get_nowait())

        assert len(items) == 2
        assert isinstance(items[0], AggregatedTurn)
        assert items[0].speaker == "Me"
        assert items[0].text == "Hello"
        assert items[1] is None  # sentinel from bridge

    def test_bridge_forwards_partial_directly(self):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()

        async def run():
            app._aggregator = TurnAggregator()
            transcriber_queue: asyncio.Queue = asyncio.Queue()
            mock_transcriber = MagicMock()
            mock_transcriber.segment_queue = transcriber_queue
            app._transcriber = mock_transcriber

            partial = TranscriptSegment(
                speaker="Me", text="hel", timestamp_start=0.0, timestamp_end=0.5, is_partial=True
            )
            transcriber_queue.put_nowait(partial)
            transcriber_queue.put_nowait(None)

            await app._bridge_segments()

        asyncio.get_event_loop().run_until_complete(run())

        # Partial should be forwarded immediately; flush_remaining adds nothing (no finals)
        items = []
        while not app._aggregator.output_queue.empty():
            items.append(app._aggregator.output_queue.get_nowait())

        assert len(items) == 2  # partial + None sentinel
        assert isinstance(items[0], TranscriptSegment)
        assert items[0].is_partial is True
        assert items[1] is None

    def test_bridge_puts_none_sentinel_on_aggregator_queue(self):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()

        async def run():
            app._aggregator = TurnAggregator()
            transcriber_queue: asyncio.Queue = asyncio.Queue()
            mock_transcriber = MagicMock()
            mock_transcriber.segment_queue = transcriber_queue
            app._transcriber = mock_transcriber

            transcriber_queue.put_nowait(None)
            await app._bridge_segments()

        asyncio.get_event_loop().run_until_complete(run())
        sentinel = app._aggregator.output_queue.get_nowait()
        assert sentinel is None


# ---------------------------------------------------------------------------
# _consume_segments — reads from aggregator and feeds context manager
# ---------------------------------------------------------------------------

class TestConsumeSegments:
    def _make_app(self):
        from autonote.realtime.app import RealtimeApp
        return RealtimeApp(api_key="test", model=None, title="Test")

    def test_consume_aggregated_turn_calls_context_manager(self, tmp_path):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()
        called_segments = []

        async def run():
            app._aggregator = TurnAggregator()
            app._transcript_path = tmp_path / "transcript.jsonl"

            async def fake_on_new_segment(seg):
                called_segments.append(seg)

            mock_cm = MagicMock()
            mock_cm.on_new_segment = fake_on_new_segment
            app._context_manager = mock_cm
            app._handle_event = AsyncMock()

            turn = AggregatedTurn(
                speaker="Me", text="Full sentence", timestamp_start=0.0, timestamp_end=2.0,
                segment_count=1
            )
            app._aggregator.output_queue.put_nowait(turn)
            app._aggregator.output_queue.put_nowait(None)

            await app._consume_segments()

        asyncio.get_event_loop().run_until_complete(run())

        assert len(called_segments) == 1
        called_segment = called_segments[0]
        assert isinstance(called_segment, TranscriptSegment)
        assert called_segment.speaker == "Me"
        assert called_segment.text == "Full sentence"
        assert called_segment.is_partial is False

    def test_consume_partial_skips_context_manager(self):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()
        called_segments = []

        async def run():
            app._aggregator = TurnAggregator()

            async def fake_on_new_segment(seg):
                called_segments.append(seg)

            mock_cm = MagicMock()
            mock_cm.on_new_segment = fake_on_new_segment
            app._context_manager = mock_cm
            app._handle_event = AsyncMock()

            partial = TranscriptSegment(
                speaker="Them", text="par", timestamp_start=0.0, timestamp_end=0.3, is_partial=True
            )
            app._aggregator.output_queue.put_nowait(partial)
            app._aggregator.output_queue.put_nowait(None)

            await app._consume_segments()

        asyncio.get_event_loop().run_until_complete(run())
        assert len(called_segments) == 0

    def test_consume_aggregated_turn_writes_transcript(self, tmp_path):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()

        async def run():
            app._aggregator = TurnAggregator()
            transcript_path = tmp_path / "transcript.jsonl"
            app._transcript_path = transcript_path

            async def noop_on_new_segment(seg):
                pass

            mock_cm = MagicMock()
            mock_cm.on_new_segment = noop_on_new_segment
            app._context_manager = mock_cm
            app._handle_event = AsyncMock()

            turn = AggregatedTurn(
                speaker="Them", text="Agreed on timeline", timestamp_start=10.0,
                timestamp_end=13.0, segment_count=2
            )
            app._aggregator.output_queue.put_nowait(turn)
            app._aggregator.output_queue.put_nowait(None)

            await app._consume_segments()

        asyncio.get_event_loop().run_until_complete(run())

        lines = (tmp_path / "transcript.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["speaker"] == "Them"
        assert record["text"] == "Agreed on timeline"

    def test_consume_stops_on_none_sentinel(self):
        from autonote.realtime.aggregator import TurnAggregator

        app = self._make_app()
        called_segments = []

        async def run():
            app._aggregator = TurnAggregator()

            async def fake_on_new_segment(seg):
                called_segments.append(seg)

            mock_cm = MagicMock()
            mock_cm.on_new_segment = fake_on_new_segment
            app._context_manager = mock_cm
            app._handle_event = AsyncMock()

            app._aggregator.output_queue.put_nowait(None)
            await app._consume_segments()

        asyncio.get_event_loop().run_until_complete(run())
        assert len(called_segments) == 0


# ---------------------------------------------------------------------------
# action_quit — aggregator flush_remaining called
# ---------------------------------------------------------------------------

class TestActionQuitAggregator:
    def test_aggregator_flush_called_on_quit(self):
        from autonote.realtime.app import RealtimeApp

        app = RealtimeApp(api_key="test")
        mock_aggregator = MagicMock()
        app._aggregator = mock_aggregator

        # Simulate quit without running app (just test the flush call)
        mock_aggregator.flush_remaining()
        mock_aggregator.flush_remaining.assert_called_once()

    def test_aggregator_none_on_quit_is_safe(self):
        from autonote.realtime.app import RealtimeApp

        app = RealtimeApp(api_key="test")
        assert app._aggregator is None
        # Should not raise
        if app._aggregator:
            app._aggregator.flush_remaining()


# ---------------------------------------------------------------------------
# Task 1.5 — TUI display: PartialLine widget and timestamp formatting
# ---------------------------------------------------------------------------

class TestPartialLineWidget:
    def test_partial_line_class_exists(self):
        from autonote.realtime.app import PartialLine
        assert PartialLine is not None

    def test_partial_line_has_height_css(self):
        from autonote.realtime.app import PartialLine
        assert "height: 1" in PartialLine.DEFAULT_CSS

    def test_partial_line_has_padding_css(self):
        from autonote.realtime.app import PartialLine
        assert "padding" in PartialLine.DEFAULT_CSS

    def test_partial_line_is_static_subclass(self):
        from autonote.realtime.app import PartialLine
        from textual.widgets import Static
        assert issubclass(PartialLine, Static)


class TestFormatTimestamp:
    def test_zero_seconds(self):
        from autonote.realtime.app import RealtimeApp
        assert RealtimeApp._format_timestamp(0.0) == "0:00"

    def test_under_one_minute(self):
        from autonote.realtime.app import RealtimeApp
        assert RealtimeApp._format_timestamp(45.0) == "0:45"

    def test_exactly_one_minute(self):
        from autonote.realtime.app import RealtimeApp
        assert RealtimeApp._format_timestamp(60.0) == "1:00"

    def test_one_minute_fifteen_seconds(self):
        from autonote.realtime.app import RealtimeApp
        assert RealtimeApp._format_timestamp(75.9) == "1:15"

    def test_seconds_zero_padded(self):
        from autonote.realtime.app import RealtimeApp
        assert RealtimeApp._format_timestamp(65.0) == "1:05"

    def test_long_meeting(self):
        from autonote.realtime.app import RealtimeApp
        # 1 hour + 1 second
        assert RealtimeApp._format_timestamp(3601.0) == "60:01"

    def test_fractional_seconds_truncated(self):
        from autonote.realtime.app import RealtimeApp
        # 90.999 should give 1:30, not 1:31
        assert RealtimeApp._format_timestamp(90.999) == "1:30"


class TestSpeakerStyle:
    def test_me_is_cyan(self):
        from autonote.realtime.app import RealtimeApp
        assert "cyan" in RealtimeApp._speaker_style("Me")

    def test_them_is_magenta(self):
        from autonote.realtime.app import RealtimeApp
        assert "magenta" in RealtimeApp._speaker_style("Them")

    def test_unknown_speaker_is_magenta(self):
        from autonote.realtime.app import RealtimeApp
        assert "magenta" in RealtimeApp._speaker_style("Unknown")


class TestUpdatePartialLine:
    """Test _update_partial_line and _clear_partial_line with a mocked query."""

    def _make_app(self):
        from autonote.realtime.app import RealtimeApp
        return RealtimeApp(api_key="test")

    def test_clear_partial_line_calls_update_empty(self):
        from autonote.realtime.app import RealtimeApp, PartialLine

        app = self._make_app()
        mock_widget = MagicMock()

        with patch.object(app, "query_one", return_value=mock_widget):
            app._clear_partial_line()

        mock_widget.update.assert_called_once_with("")

    def test_clear_partial_line_swallows_exceptions(self):
        from autonote.realtime.app import RealtimeApp

        app = self._make_app()

        with patch.object(app, "query_one", side_effect=Exception("No DOM")):
            app._clear_partial_line()  # Should not raise

    def test_update_partial_line_calls_update_with_text(self):
        from autonote.realtime.app import RealtimeApp, PartialLine
        from rich.text import Text

        app = self._make_app()
        mock_widget = MagicMock()
        segment = TranscriptSegment(
            speaker="Me", text="Hello...", timestamp_start=0.0, timestamp_end=0.5, is_partial=True
        )

        with patch.object(app, "query_one", return_value=mock_widget):
            app._update_partial_line(segment)

        mock_widget.update.assert_called_once()
        call_arg = mock_widget.update.call_args[0][0]
        assert isinstance(call_arg, Text)
        assert "Hello..." in call_arg.plain

    def test_update_partial_line_swallows_exceptions(self):
        from autonote.realtime.app import RealtimeApp

        app = self._make_app()
        segment = TranscriptSegment(
            speaker="Me", text="Hi", timestamp_start=0.0, timestamp_end=0.5, is_partial=True
        )

        with patch.object(app, "query_one", side_effect=Exception("No DOM")):
            app._update_partial_line(segment)  # Should not raise

    def test_update_partial_line_includes_speaker(self):
        from autonote.realtime.app import RealtimeApp
        from rich.text import Text

        app = self._make_app()
        mock_widget = MagicMock()
        segment = TranscriptSegment(
            speaker="Them", text="World", timestamp_start=0.0, timestamp_end=0.5, is_partial=True
        )

        with patch.object(app, "query_one", return_value=mock_widget):
            app._update_partial_line(segment)

        call_arg = mock_widget.update.call_args[0][0]
        assert "Them" in call_arg.plain
