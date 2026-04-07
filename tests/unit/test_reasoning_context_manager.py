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
from autonote.reasoning.mission import MissionBrief


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

    def test_get_turn_transcript_empty(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        assert state.get_turn_transcript() == ""

    def test_get_turn_transcript_single_turn(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn("Hello world", speaker="Me"))
        text = state.get_turn_transcript()
        assert text == "Me: Hello world"

    def test_get_turn_transcript_multiple_turns(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn("First", speaker="Me"))
        state.add_turn(make_turn("Second", speaker="Them"))
        text = state.get_turn_transcript()
        assert "Me: First" in text
        assert "Them: Second" in text

    def test_get_turn_transcript_last_n(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        for i in range(5):
            state.add_turn(make_turn(f"msg{i}", speaker="Me"))
        text = state.get_turn_transcript(last_n=2)
        assert "msg3" in text
        assert "msg4" in text
        assert "msg0" not in text

    def test_get_turn_transcript_last_n_none_returns_all(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        for i in range(3):
            state.add_turn(make_turn(f"msg{i}", speaker="Me"))
        text = state.get_turn_transcript(last_n=None)
        assert "msg0" in text
        assert "msg1" in text
        assert "msg2" in text

    def test_get_full_context_uses_turn_transcript_when_turns_available(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_turn(make_turn("Turn text", speaker="Me"))
        ctx = state.get_full_context()
        assert "Me: Turn text" in ctx

    def test_get_full_context_falls_back_to_segment_transcript_when_no_turns(self):
        state = MeetingState(session_id="s1", start_time=0.0)
        state.add_segment(make_segment("Segment text", speaker="Me"))
        ctx = state.get_full_context()
        assert "[Me @ 0s]: Segment text" in ctx
        assert "Me: Segment text" not in ctx


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManager:

    def _make_cm(self, events=None, summary_every=5, action_every=3) -> tuple[ContextManager, list]:
        received = events if events is not None else []

        async def on_event(evt):
            received.append(evt)

        dispatcher = make_dispatcher()
        brief = MissionBrief(
            name="test", goal="test", role="tester",
            summary_every_n_turns=summary_every,
            action_items_every_n_turns=action_every,
        )
        cm = ContextManager(
            dispatcher=dispatcher,
            on_event=on_event,
            mission_brief=brief,
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

    async def test_run_summary_uses_turn_transcript_when_turns_available(self):
        cm, received = self._make_cm()
        cm.state.add_turn(make_turn("Hello from Me", speaker="Me", start=0.0, end=2.0))
        captured: dict = {}
        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return "Summary"
        with patch.object(cm._summary_worker, "execute", new=AsyncMock(side_effect=fake_execute)):
            await cm._run_summary()
        assert "Me: Hello from Me" in captured.get("new_segments", "")

    async def test_run_summary_falls_back_to_segment_transcript_when_no_turns(self):
        cm, received = self._make_cm()
        cm.state.add_segment(make_segment("Seg text", speaker="Me", start=0.0, end=1.0))
        captured: dict = {}
        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return "Summary"
        with patch.object(cm._summary_worker, "execute", new=AsyncMock(side_effect=fake_execute)):
            await cm._run_summary()
        assert "[Me @ 0s]: Seg text" in captured.get("new_segments", "")

    async def test_run_action_items_uses_turn_transcript_when_turns_available(self):
        cm, _ = self._make_cm()
        cm.state.add_turn(make_turn("Action item turn", speaker="Them", start=0.0, end=2.0))
        captured: dict = {}
        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return []
        with patch.object(cm._action_item_worker, "execute", new=AsyncMock(side_effect=fake_execute)):
            await cm._run_action_items()
        assert "Them: Action item turn" in captured.get("recent_transcript", "")

    async def test_run_contradictions_uses_turn_transcript_when_turns_available(self):
        cm, _ = self._make_cm()
        cm.state.add_turn(make_turn("Contradiction turn", speaker="Me", start=0.0, end=2.0))
        captured: dict = {}
        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return []
        with patch.object(cm._contradiction_worker, "execute", new=AsyncMock(side_effect=fake_execute)):
            await cm._run_contradictions()
        assert "Me: Contradiction turn" in captured.get("recent_transcript", "")


# ---------------------------------------------------------------------------
# ContextManager — turn-based triggers
# ---------------------------------------------------------------------------

class TestContextManagerTurnTriggers:

    def _make_cm(self, summary_turns=5, action_turns=3) -> tuple[ContextManager, list]:
        received: list = []

        async def on_event(evt):
            received.append(evt)

        dispatcher = make_dispatcher()
        brief = MissionBrief(
            name="test", goal="test", role="tester",
            summary_every_n_turns=summary_turns,
            action_items_every_n_turns=action_turns,
        )
        cm = ContextManager(
            dispatcher=dispatcher,
            on_event=on_event,
            mission_brief=brief,
            session_id="test-turns",
        )
        return cm, received

    async def test_on_new_turn_emits_event(self):
        cm, received = self._make_cm()
        turn = make_turn()
        await cm.on_new_turn(turn)
        assert turn in received

    async def test_on_new_turn_adds_to_state(self):
        cm, _ = self._make_cm()
        turn = make_turn(speaker="Them")
        await cm.on_new_turn(turn)
        assert len(cm.state.turns) == 1
        assert "Them" in cm.state.speakers

    async def test_summary_triggered_after_n_turns(self):
        cm, _ = self._make_cm(summary_turns=5, action_turns=100)
        with patch.object(cm, "_fire_task") as mock_fire, \
             patch.object(cm, "_run_summary") as mock_run_summary:
            mock_run_summary.return_value = MagicMock()
            for i in range(5):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
            assert mock_fire.called

    async def test_summary_not_triggered_before_threshold(self):
        cm, _ = self._make_cm(summary_turns=5, action_turns=100)
        with patch.object(cm, "_fire_task") as mock_fire:
            for i in range(4):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
            mock_fire.assert_not_called()

    async def test_action_items_triggered_after_n_turns(self):
        cm, _ = self._make_cm(summary_turns=100, action_turns=3)
        with patch.object(cm, "_fire_task") as mock_fire, \
             patch.object(cm, "_run_action_items") as mock_run_ai:
            mock_run_ai.return_value = MagicMock()
            for i in range(3):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
            assert mock_fire.called

    async def test_action_items_not_triggered_before_threshold(self):
        cm, _ = self._make_cm(summary_turns=100, action_turns=3)
        with patch.object(cm, "_fire_task") as mock_fire:
            for i in range(2):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
            mock_fire.assert_not_called()

    async def test_turn_counter_resets_after_summary_trigger(self):
        cm, _ = self._make_cm(summary_turns=3, action_turns=100)
        with patch.object(cm, "_fire_task"):
            for i in range(3):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
        assert cm.state.turns_since_last_summary == 0

    async def test_turn_counter_resets_after_action_trigger(self):
        cm, _ = self._make_cm(summary_turns=100, action_turns=3)
        with patch.object(cm, "_fire_task"):
            for i in range(3):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
        assert cm.state.turns_since_last_action_scan == 0

    async def test_contradiction_check_is_time_based_not_turn_based(self):
        cm, _ = self._make_cm()
        with patch.object(cm, "_fire_task") as mock_fire, \
             patch.object(cm, "_run_contradictions") as mock_contra:
            mock_contra.return_value = MagicMock()
            # Feed many turns — contradictions should NOT trigger (time hasn't passed)
            for i in range(20):
                await cm.on_new_turn(make_turn(f"text{i}", start=float(i), end=float(i+2)))
            contradiction_calls = [
                c for c in mock_fire.call_args_list
                if mock_contra.called
            ]
            # _run_contradictions should not have been called without time passing
            assert not mock_contra.called

    async def test_contradiction_triggers_when_time_passes(self):
        cm, _ = self._make_cm()
        cm._last_contradiction_check -= 200  # simulate 200s passing
        with patch.object(cm, "_fire_task") as mock_fire, \
             patch.object(cm, "_run_contradictions") as mock_contra:
            mock_contra.return_value = MagicMock()
            await cm.on_new_turn(make_turn())
            assert mock_fire.called

    async def test_default_summary_every_n_turns(self):
        dispatcher = make_dispatcher()
        cm = ContextManager(dispatcher=dispatcher, on_event=AsyncMock(), session_id="t")
        assert cm.SUMMARY_EVERY_N_TURNS == 5

    async def test_default_action_scan_every_n_turns(self):
        dispatcher = make_dispatcher()
        cm = ContextManager(dispatcher=dispatcher, on_event=AsyncMock(), session_id="t")
        assert cm.ACTION_SCAN_EVERY_N_TURNS == 3


# ---------------------------------------------------------------------------
# ContextManager — coach in-flight guard & speaker filter
# ---------------------------------------------------------------------------

class TestCoachGuard:
    """Tests for the coach in-flight guard (Option A) and speaker filter (Option C)."""

    def _make_cm_with_coach(self, coach_every: int = 1) -> tuple[ContextManager, list]:
        received: list = []

        async def on_event(evt):
            received.append(evt)

        dispatcher = make_dispatcher()
        brief = MissionBrief(
            name="test",
            goal="test",
            role="tester",
            summary_enabled=False,
            action_items_enabled=False,
            contradictions_enabled=False,
            coach_every_n_turns=coach_every,
        )
        cm = ContextManager(
            dispatcher=dispatcher,
            on_event=on_event,
            mission_brief=brief,
            session_id="test-coach",
        )
        return cm, received

    async def test_coach_skips_my_turns(self):
        """Coach should NOT fire on speaker='Me' turns."""
        cm, _ = self._make_cm_with_coach(coach_every=1)
        with patch.object(cm, "_fire_task") as mock_fire:
            await cm.on_new_turn(make_turn("I said something", speaker="Me"))
            mock_fire.assert_not_called()

    async def test_coach_fires_on_them_turns(self):
        """Coach SHOULD fire on speaker='Them' turns."""
        cm, _ = self._make_cm_with_coach(coach_every=1)
        with patch.object(cm, "_fire_task") as mock_fire:
            await cm.on_new_turn(make_turn("They said something", speaker="Them"))
            assert mock_fire.called

    async def test_coach_in_flight_guard_prevents_stacking(self):
        """Second 'Them' turn should NOT fire coach when one is already in-flight."""
        cm, _ = self._make_cm_with_coach(coach_every=1)
        with patch.object(cm, "_fire_task"):
            await cm.on_new_turn(make_turn("first", speaker="Them", start=0.0, end=1.0))
            assert cm._coach_in_flight is True

            # Simulate: coach is still running, second turn arrives
            await cm.on_new_turn(make_turn("second", speaker="Them", start=2.0, end=3.0))
            assert cm._coach_pending is True

    async def test_coach_in_flight_guard_fires_only_once(self):
        """Multiple 'Them' turns while in-flight should result in exactly 1 _fire_task call."""
        cm, _ = self._make_cm_with_coach(coach_every=1)
        with patch.object(cm, "_fire_task") as mock_fire:
            # First turn: fires coach
            await cm.on_new_turn(make_turn("first", speaker="Them", start=0.0, end=1.0))
            # Next 3 turns: all skipped (in-flight), only set pending
            for i in range(3):
                await cm.on_new_turn(make_turn(f"turn{i}", speaker="Them", start=float(i+2), end=float(i+3)))
            assert mock_fire.call_count == 1

    async def test_coach_follow_up_fires_after_completion(self):
        """When coach completes with pending=True, a follow-up should fire."""
        cm, received = self._make_cm_with_coach(coach_every=1)
        # Add some turns to give coach context
        cm.state.add_turn(make_turn("context", speaker="Them", start=0.0, end=1.0))

        from autonote.realtime.models import CoachSuggestion
        mock_suggestion = CoachSuggestion(
            should_speak=True, confidence="high",
            suggestion="do this", argument_used=None,
            reasoning="test", timestamp=1.0,
        )
        with patch.object(cm._coach_worker, "execute", new=AsyncMock(return_value=mock_suggestion)):
            # Simulate: set pending before running coach
            cm._coach_pending = True
            cm._coach_in_flight = True
            await cm._run_coach()

        # After completion: the follow-up should have fired (in_flight set again)
        # The first _run_coach sets in_flight=False in finally, sees pending=True, fires another
        assert cm._coach_in_flight is True  # follow-up was spawned

    async def test_coach_resets_in_flight_on_failure(self):
        """If coach fails, in-flight flag should still reset so future calls work."""
        cm, _ = self._make_cm_with_coach(coach_every=1)
        cm.state.add_turn(make_turn("context", speaker="Them"))

        with patch.object(cm._coach_worker, "execute", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
            cm._coach_in_flight = True
            await cm._run_coach()

        assert cm._coach_in_flight is False

    async def test_my_turns_still_increment_coach_counter(self):
        """'Me' turns should still count towards turns_since_last_coach via add_turn."""
        cm, _ = self._make_cm_with_coach(coach_every=3)
        cm.state.add_turn(make_turn("a", speaker="Me"))
        cm.state.add_turn(make_turn("b", speaker="Me"))
        assert cm.state.turns_since_last_coach == 2
