"""Tests for reasoning ContextManager and MeetingState."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

from autonote.reasoning.context_manager import MeetingState, ContextManager
from autonote.realtime.models import (
    TranscriptSegment,
    AggregatedTurn,
    ActionItem,
    SummaryUpdate,
    ActionItemsUpdate,
    ContradictionAlert,
    ReplySuggestion,
    CustomPromptResult,
)


def make_segment(text: str = "Hello", speaker: str = "Me", start: float = 0.0, end: float = 1.0, partial: bool = False) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker,
        text=text,
        timestamp_start=start,
        timestamp_end=end,
        is_partial=partial,
    )


def make_turn(text: str = "Hello world", speaker: str = "Me", start: float = 0.0, end: float = 2.0, segment_count: int = 2) -> AggregatedTurn:
    return AggregatedTurn(
        speaker=speaker,
        text=text,
        timestamp_start=start,
        timestamp_end=end,
        segment_count=segment_count,
    )


def make_dispatcher() -> MagicMock:
    dispatcher = MagicMock()
    dispatcher.run = AsyncMock(return_value='{"new_items": [], "updated_items": []}')
    return dispatcher


# ---------------------------------------------------------------------------
# MeetingState
# ---------------------------------------------------------------------------

class TestMeetingState:

    def test_add_segment_increments_counters(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        seg = make_segment()
        state.add_segment(seg)
        assert state.segments_since_last_summary == 1
        assert state.segments_since_last_action_scan == 1
        assert "Me" in state.speakers

    def test_get_transcript_text_all(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_segment(make_segment("Hello", "Me", 0.0, 1.0))
        state.add_segment(make_segment("World", "Them", 1.0, 2.0))
        text = state.get_transcript_text()
        assert "[Me @ 0s]: Hello" in text
        assert "[Them @ 1s]: World" in text

    def test_get_transcript_text_last_n(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        for i in range(5):
            state.add_segment(make_segment(f"msg{i}", start=float(i), end=float(i+1)))
        text = state.get_transcript_text(last_n=2)
        assert "msg3" in text
        assert "msg4" in text
        assert "msg0" not in text

    def test_get_full_context_includes_summary(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.current_summary = "This is the summary."
        state.add_segment(make_segment())
        ctx = state.get_full_context()
        assert "## Summary So Far" in ctx
        assert "This is the summary." in ctx

    def test_get_full_context_includes_action_items(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.action_items = [ActionItem(id="1", description="Do X", source_timestamp=0.0)]
        state.add_segment(make_segment())
        ctx = state.get_full_context()
        assert "## Action Items" in ctx
        assert "Do X" in ctx

    def test_get_full_context_no_summary_or_items(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_segment(make_segment("Hello"))
        ctx = state.get_full_context()
        assert "## Recent Transcript" in ctx
        assert "## Summary So Far" not in ctx

    def test_recent_window_is_bounded(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        for i in range(60):
            state.add_segment(make_segment(f"msg{i}"))
        assert len(state.recent_window) == 50  # maxlen=50

    def test_add_turn_appends_to_turns_list(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        turn = make_turn()
        state.add_turn(turn)
        assert len(state.turns) == 1
        assert state.turns[0] is turn

    def test_add_turn_increments_counters(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn())
        assert state.turns_since_last_summary == 1
        assert state.turns_since_last_action_scan == 1

    def test_add_turn_multiple_increments_counters(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn())
        state.add_turn(make_turn())
        state.add_turn(make_turn())
        assert state.turns_since_last_summary == 3
        assert state.turns_since_last_action_scan == 3

    def test_add_turn_updates_speakers_set(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn(speaker="Me"))
        state.add_turn(make_turn(speaker="Them"))
        assert "Me" in state.speakers
        assert "Them" in state.speakers

    def test_add_turn_does_not_affect_segment_counters(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn())
        assert state.segments_since_last_summary == 0
        assert state.segments_since_last_action_scan == 0

    def test_add_segment_does_not_affect_turn_counters(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_segment(make_segment())
        assert state.turns_since_last_summary == 0
        assert state.turns_since_last_action_scan == 0

    def test_turns_list_starts_empty(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        assert state.turns == []
        assert state.turns_since_last_summary == 0
        assert state.turns_since_last_action_scan == 0


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManager:

    def _make_cm(self, events=None, summary_every=5, action_every=3) -> tuple[ContextManager, list]:
        received = events if events is not None else []

        async def on_event(evt):
            received.append(evt)

        dispatcher = make_dispatcher()
        cm = ContextManager(
            dispatcher=dispatcher,
            on_event=on_event,
            summary_every_n=summary_every,
            action_scan_every_n=action_every,
            session_id="test-session",
        )
        return cm, received

    async def test_on_new_segment_emits_event(self):
        cm, received = self._make_cm()
        seg = make_segment()
        await cm.on_new_segment(seg)
        assert seg in received

    async def test_partial_segments_do_not_trigger_reasoning(self):
        cm, received = self._make_cm(summary_every=1, action_every=1)
        seg = make_segment(partial=True)
        with patch.object(cm, "_fire_task") as mock_fire:
            await cm.on_new_segment(seg)
            mock_fire.assert_not_called()

    async def test_summary_triggered_after_n_segments(self):
        cm, _ = self._make_cm(summary_every=3, action_every=100)
        with patch.object(cm, "_fire_task") as mock_fire, \
             patch.object(cm, "_run_summary") as mock_run_summary:
            mock_run_summary.return_value = MagicMock()
            for i in range(3):
                await cm.on_new_segment(make_segment(f"s{i}", start=float(i), end=float(i+1)))
            # Summary should have fired once
            summary_calls = [c for c in mock_fire.call_args_list]
            assert len(summary_calls) >= 1

    async def test_action_scan_triggered_after_n_segments(self):
        cm, _ = self._make_cm(summary_every=100, action_every=3)
        with patch.object(cm, "_fire_task") as mock_fire:
            for i in range(3):
                await cm.on_new_segment(make_segment(f"s{i}", start=float(i), end=float(i+1)))
            assert mock_fire.called

    async def test_handle_custom_prompt(self):
        cm, received = self._make_cm()
        with patch.object(cm._custom_prompt_worker, "execute", new=AsyncMock(
            return_value=CustomPromptResult(prompt="Q?", result="A.", timestamp=0.0)
        )):
            await cm.handle_custom_prompt("Q?")
        custom_results = [e for e in received if isinstance(e, CustomPromptResult)]
        assert len(custom_results) == 1
        assert custom_results[0].result == "A."

    async def test_handle_reply_request(self):
        cm, received = self._make_cm()
        with patch.object(cm._reply_worker, "execute", new=AsyncMock(
            return_value=ReplySuggestion(suggestions=["Yes!"], context="ctx", triggered_by="manual")
        )):
            await cm.handle_reply_request("hint")
        replies = [e for e in received if isinstance(e, ReplySuggestion)]
        assert len(replies) == 1

    async def test_handle_custom_prompt_exception_does_not_propagate(self):
        cm, _ = self._make_cm()
        with patch.object(cm._custom_prompt_worker, "execute", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
            # Should not raise
            await cm.handle_custom_prompt("Q?")

    async def test_shutdown_cancels_running_tasks(self):
        cm, _ = self._make_cm()
        await cm.shutdown()
        assert len(cm._running_tasks) == 0

    async def test_run_summary_updates_state_and_emits_event(self):
        cm, received = self._make_cm()
        cm.state.add_segment(make_segment("Hello", end=5.0))
        with patch.object(cm._summary_worker, "execute", new=AsyncMock(return_value="New summary")):
            await cm._run_summary()
        assert cm.state.current_summary == "New summary"
        summary_events = [e for e in received if isinstance(e, SummaryUpdate)]
        assert len(summary_events) == 1
        assert summary_events[0].summary == "New summary"

    async def test_run_action_items_updates_state_and_emits_event(self):
        cm, received = self._make_cm()
        new_items = [ActionItem(id="1", description="Fix it", source_timestamp=0.0)]
        with patch.object(cm._action_item_worker, "execute", new=AsyncMock(return_value=new_items)):
            await cm._run_action_items()
        assert cm.state.action_items == new_items
        action_events = [e for e in received if isinstance(e, ActionItemsUpdate)]
        assert len(action_events) == 1

    async def test_run_contradictions_emits_alerts(self):
        cm, received = self._make_cm()
        alert = ContradictionAlert(
            description="Conflict",
            statement_a="A",
            statement_a_timestamp=0.0,
            statement_b="B",
            statement_b_timestamp=1.0,
            severity="medium",
        )
        with patch.object(cm._contradiction_worker, "execute", new=AsyncMock(return_value=[alert])):
            await cm._run_contradictions()
        alerts = [e for e in received if isinstance(e, ContradictionAlert)]
        assert len(alerts) == 1

    async def test_segment_counter_resets_after_trigger(self):
        cm, _ = self._make_cm(summary_every=2, action_every=100)
        with patch.object(cm, "_fire_task"):
            for i in range(2):
                await cm.on_new_segment(make_segment(f"s{i}", start=float(i), end=float(i+1)))
        assert cm.state.segments_since_last_summary == 0
