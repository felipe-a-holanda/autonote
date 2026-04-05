"""Context Manager — accumulates meeting state and triggers LLM reasoning tasks.

Ported from meeting-copilot/backend/reasoning/context_manager.py.
The WebSocket broadcast_fn is replaced with a generic async event callback
that the TUI (or any consumer) binds to.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any, Optional

from autonote.realtime.models import (
    TranscriptSegment,
    AggregatedTurn,
    ActionItem,
    SummaryUpdate,
    ActionItemsUpdate,
    ContradictionAlert,
    RealtimeEvent,
)
from autonote.reasoning.dispatcher import LLMDispatcher
from autonote.reasoning.mission import MissionBrief
from autonote.reasoning.workers.summary import SummaryWorker
from autonote.reasoning.workers.action_items import ActionItemWorker
from autonote.reasoning.workers.contradictions import ContradictionWorker
from autonote.reasoning.workers.coach import CoachWorker
from autonote.reasoning.workers.custom import CustomPromptWorker
from autonote.reasoning.workers.reply import ReplyWorker

logger = logging.getLogger(__name__)


@dataclass
class MeetingState:
    """Accumulated state of the meeting, fed to LLM workers."""

    session_id: str
    start_time: float

    segments: list[TranscriptSegment] = field(default_factory=list)
    turns: list[AggregatedTurn] = field(default_factory=list)
    speakers: set[str] = field(default_factory=set)
    current_summary: str = ""
    action_items: list[ActionItem] = field(default_factory=list)
    recent_window: deque = field(default_factory=lambda: deque(maxlen=50))
    segments_since_last_summary: int = 0
    segments_since_last_action_scan: int = 0
    turns_since_last_summary: int = 0
    turns_since_last_action_scan: int = 0
    turns_since_last_coach: int = 0
    turns_since_last_reply: int = 0

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)
        self.recent_window.append(segment)
        self.speakers.add(segment.speaker)
        self.segments_since_last_summary += 1
        self.segments_since_last_action_scan += 1

    def add_turn(self, turn: AggregatedTurn) -> None:
        self.turns.append(turn)
        self.speakers.add(turn.speaker)
        self.turns_since_last_summary += 1
        self.turns_since_last_action_scan += 1
        self.turns_since_last_coach += 1
        self.turns_since_last_reply += 1

    def get_transcript_text(self, last_n: Optional[int] = None) -> str:
        if last_n is not None:
            source = list(self.recent_window)[-last_n:]
        else:
            source = self.segments
        return "\n".join(
            f"[{s.speaker} @ {s.timestamp_start:.0f}s]: {s.text}"
            for s in source
        )

    def get_turn_transcript(self, last_n: Optional[int] = None) -> str:
        """Format turns as 'Speaker: text' blocks, optionally limited to last N turns."""
        source = self.turns[-last_n:] if last_n is not None else self.turns
        return "\n".join(f"{t.speaker}: {t.text}" for t in source)

    def get_full_context(self) -> str:
        parts: list[str] = []
        if self.current_summary:
            parts.append(f"## Summary So Far\n{self.current_summary}")
        if self.action_items:
            items_text = "\n".join(
                f"- {a.description} (assigned: {a.assignee or 'TBD'})"
                for a in self.action_items
            )
            parts.append(f"## Action Items\n{items_text}")
        if self.turns:
            transcript = self.get_turn_transcript(last_n=30)
        else:
            transcript = self.get_transcript_text(last_n=30)
        parts.append(f"## Recent Transcript\n{transcript}")
        return "\n\n".join(parts)


# Type for the event callback — receives any RealtimeEvent
EventCallback = Callable[[RealtimeEvent], Awaitable[None]]
# Type for the debug callback — mirrors TUI's _debug(msg, level)
DebugCallback = Callable[[str, str], None]


class ContextManager:
    """Manages meeting state and triggers reasoning tasks on configurable thresholds.

    All thresholds and mode toggles can be overridden via a MissionBrief profile.
    When no profile is provided, defaults match the class-level constants below.

    Args:
        dispatcher: The LLM dispatcher for routing reasoning tasks.
        on_event: Async callback invoked for every event.
        mission_brief: Optional profile that overrides all thresholds and toggles.
        session_id: Identifier for this meeting session.
    """

    # Defaults — used when no profile overrides them
    SUMMARY_EVERY_N_TURNS: int = 5
    ACTION_SCAN_EVERY_N_TURNS: int = 3
    CONTRADICTION_CHECK_SECONDS: int = 120
    COACH_EVERY_N_TURNS: int = 3
    REPLY_EVERY_N_TURNS: int = 0  # 0 = manual only

    def __init__(
        self,
        dispatcher: LLMDispatcher,
        on_event: EventCallback,
        *,
        on_debug: Optional[DebugCallback] = None,
        mission_brief: Optional[MissionBrief] = None,
        session_id: str = "",
    ) -> None:
        self.state = MeetingState(
            session_id=session_id,
            start_time=time.time(),
        )
        self.dispatcher = dispatcher
        self.on_event = on_event
        self._on_debug = on_debug
        self._running_tasks: set[asyncio.Task] = set()
        self._last_contradiction_check: float = time.time()
        self._mission_brief = mission_brief

        self._summary_worker = SummaryWorker(dispatcher)
        self._action_item_worker = ActionItemWorker(dispatcher)
        self._contradiction_worker = ContradictionWorker(dispatcher)
        self._custom_prompt_worker = CustomPromptWorker(dispatcher)
        self._reply_worker = ReplyWorker(dispatcher)
        self._coach_worker = CoachWorker(dispatcher)

        # Apply profile overrides if a brief is provided
        if mission_brief is not None:
            self.SUMMARY_EVERY_N_TURNS = mission_brief.summary_every_n_turns
            self.ACTION_SCAN_EVERY_N_TURNS = mission_brief.action_items_every_n_turns
            self.CONTRADICTION_CHECK_SECONDS = mission_brief.contradictions_every_seconds
            self.COACH_EVERY_N_TURNS = mission_brief.coach_every_n_turns
            self.REPLY_EVERY_N_TURNS = mission_brief.reply_every_n_turns

    async def on_new_segment(self, segment: TranscriptSegment) -> None:
        """Process a new transcript segment.

        Adds the segment to state, emits it via the event callback, and
        triggers reasoning tasks based on thresholds. Only final (non-partial)
        segments count towards reasoning triggers.
        """
        self.state.add_segment(segment)
        await self.on_event(segment)

        # Only trigger reasoning on final segments
        if segment.is_partial:
            return

        if self.state.segments_since_last_summary >= self.SUMMARY_EVERY_N_SEGMENTS:
            self._fire_task(self._run_summary())
            self.state.segments_since_last_summary = 0

        if self.state.segments_since_last_action_scan >= self.ACTION_SCAN_EVERY_N_SEGMENTS:
            self._fire_task(self._run_action_items())
            self.state.segments_since_last_action_scan = 0

        now = time.time()
        if now - self._last_contradiction_check >= self.CONTRADICTION_CHECK_SECONDS:
            self._fire_task(self._run_contradictions())
            self._last_contradiction_check = now

    async def on_new_turn(self, turn: AggregatedTurn) -> None:
        """Process a completed aggregated turn.

        Adds the turn to state, emits it via the event callback, and
        triggers reasoning tasks based on turn-count thresholds.
        Mode toggles and thresholds come from the MissionBrief when provided.
        """
        self.state.add_turn(turn)
        await self.on_event(turn)

        brief = self._mission_brief
        summary_on = brief is None or brief.summary_enabled
        actions_on = brief is None or brief.action_items_enabled
        contradictions_on = brief is None or brief.contradictions_enabled

        n_turns = len(self.state.turns)
        if summary_on or actions_on:
            self._debug(
                f"Turn #{n_turns} [{turn.speaker}] — "
                f"summary in {self.SUMMARY_EVERY_N_TURNS - self.state.turns_since_last_summary} turns, "
                f"actions in {self.ACTION_SCAN_EVERY_N_TURNS - self.state.turns_since_last_action_scan} turns"
            )

        if summary_on and self.state.turns_since_last_summary >= self.SUMMARY_EVERY_N_TURNS:
            self._fire_task(self._run_summary())
            self.state.turns_since_last_summary = 0

        if actions_on and self.state.turns_since_last_action_scan >= self.ACTION_SCAN_EVERY_N_TURNS:
            self._fire_task(self._run_action_items())
            self.state.turns_since_last_action_scan = 0

        now = time.time()
        if contradictions_on and now - self._last_contradiction_check >= self.CONTRADICTION_CHECK_SECONDS:
            self._fire_task(self._run_contradictions())
            self._last_contradiction_check = now

        if self.REPLY_EVERY_N_TURNS > 0 and self.state.turns_since_last_reply >= self.REPLY_EVERY_N_TURNS:
            self._fire_task(self._run_reply_auto())
            self.state.turns_since_last_reply = 0

        if self._mission_brief and self.COACH_EVERY_N_TURNS > 0 and self.state.turns_since_last_coach >= self.COACH_EVERY_N_TURNS:
            self._fire_task(self._run_coach())
            self.state.turns_since_last_coach = 0

    async def handle_custom_prompt(self, prompt: str) -> None:
        """Run a user's freeform prompt against the meeting context."""
        self._debug(f"LLM: custom prompt → {prompt}", "info")
        try:
            timestamp = (
                self.state.segments[-1].timestamp_end if self.state.segments else 0.0
            )
            result = await self._custom_prompt_worker.execute(
                full_context=self.state.get_full_context(),
                user_prompt=prompt,
                timestamp=timestamp,
            )
            self._debug("LLM: custom prompt answered", "ok")
            await self.on_event(result)
        except Exception as exc:
            self._debug(f"LLM: custom prompt failed — {exc}", "error")
            logger.warning("Custom prompt failed: %s", exc)

    async def handle_summary_request(self) -> None:
        """Manually trigger a summary update."""
        self.state.turns_since_last_summary = self.SUMMARY_EVERY_N_TURNS
        self._fire_task(self._run_summary())
        self.state.turns_since_last_summary = 0

    async def handle_action_items_request(self) -> None:
        """Manually trigger an action items scan."""
        self.state.turns_since_last_action_scan = self.ACTION_SCAN_EVERY_N_TURNS
        self._fire_task(self._run_action_items())
        self.state.turns_since_last_action_scan = 0

    async def handle_contradiction_request(self) -> None:
        """Manually trigger a contradiction check."""
        self._fire_task(self._run_contradictions())
        self._last_contradiction_check = time.time()

    async def handle_reply_request(self, context_hint: str = "") -> None:
        """Generate reply suggestions based on the current meeting context."""
        try:
            suggestion = await self._reply_worker.execute(
                full_context=self.state.get_full_context(),
                context_hint=context_hint or "No specific context provided.",
            )
            await self.on_event(suggestion)
        except Exception as exc:
            logger.warning("Reply request failed: %s", exc)

    async def handle_coach_request(self) -> None:
        """Manually trigger a coach suggestion (requires mission_brief to be set)."""
        if self._mission_brief:
            await self._run_coach()

    def _debug(self, msg: str, level: str = "info") -> None:
        if self._on_debug is not None:
            try:
                self._on_debug(msg, level)
            except Exception:
                pass
        logger.debug(msg)

    def _fire_task(self, coro: Any) -> None:
        """Spawn a background task and track it for cleanup."""
        task = asyncio.create_task(coro)
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    async def _run_summary(self) -> None:
        self._debug("LLM: running summary...", "info")
        try:
            if self.state.turns:
                new_segments = self.state.get_turn_transcript(
                    last_n=self.SUMMARY_EVERY_N_TURNS
                )
                covered_until = self.state.turns[-1].timestamp_end
            else:
                new_segments = self.state.get_transcript_text(
                    last_n=self.SUMMARY_EVERY_N_SEGMENTS
                )
                covered_until = self.state.segments[-1].timestamp_end if self.state.segments else 0.0
            result = await self._summary_worker.execute(
                current_summary=self.state.current_summary,
                new_segments=new_segments,
            )
            self.state.current_summary = result
            self._debug("LLM: summary updated", "ok")
            await self.on_event(
                SummaryUpdate(
                    summary=result,
                    covered_until=covered_until,
                )
            )
        except Exception as exc:
            self._debug(f"LLM: summary failed — {exc}", "error")
            logger.warning("Summary task failed, skipping: %s", exc)

    async def _run_contradictions(self) -> None:
        self._debug("LLM: checking for contradictions...", "info")
        try:
            if self.state.turns:
                recent_transcript = self.state.get_turn_transcript(last_n=20)
            else:
                recent_transcript = self.state.get_transcript_text(last_n=20)
            alerts = await self._contradiction_worker.execute(
                current_summary=self.state.current_summary,
                recent_transcript=recent_transcript,
            )
            self._debug(f"LLM: contradiction check done ({len(alerts)} alerts)", "ok")
            for alert in alerts:
                await self.on_event(alert)
        except Exception as exc:
            self._debug(f"LLM: contradiction check failed — {exc}", "error")
            logger.warning("Contradiction check failed, skipping: %s", exc)

    async def _run_action_items(self) -> None:
        self._debug("LLM: scanning action items...", "info")
        try:
            if self.state.turns:
                recent_transcript = self.state.get_turn_transcript(last_n=10)
            else:
                recent_transcript = self.state.get_transcript_text(last_n=10)
            updated_items = await self._action_item_worker.execute(
                full_context=self.state.get_full_context(),
                recent_transcript=recent_transcript,
                existing_items=self.state.action_items,
            )
            self.state.action_items = updated_items
            self._debug(f"LLM: action items done ({len(updated_items)} items)", "ok")
            await self.on_event(
                ActionItemsUpdate(items=self.state.action_items)
            )
        except Exception as exc:
            self._debug(f"LLM: action items failed — {exc}", "error")
            logger.warning("Action items task failed, skipping: %s", exc)

    async def _run_reply_auto(self) -> None:
        last_turn = self.state.turns[-1].text if self.state.turns else ""
        self._debug(f"LLM: auto reply — last turn: \"{last_turn}\"", "info")
        try:
            suggestion = await self._reply_worker.execute(
                full_context=self.state.get_full_context(),
                context_hint="Auto-triggered after turn.",
            )
            await self.on_event(suggestion)
            self._debug("LLM: auto reply done", "ok")
        except Exception as exc:
            self._debug(f"LLM: auto reply failed — {exc}", "error")
            logger.warning("Auto reply failed: %s", exc)

    async def _run_coach(self) -> None:
        self._debug("LLM: running coach...", "info")
        try:
            if self.state.turns:
                recent = self.state.get_turn_transcript(last_n=5)
                timestamp = self.state.turns[-1].timestamp_end
            else:
                recent = self.state.get_transcript_text(last_n=5)
                timestamp = self.state.segments[-1].timestamp_end if self.state.segments else 0.0
            self._debug(f"LLM: coach context (last 5 turns):\n{recent}", "info")
            suggestion = await self._coach_worker.execute(
                full_context=self.state.get_full_context(),
                recent_transcript=recent,
                mission_brief_text=self._mission_brief.format_for_prompt(),  # type: ignore[union-attr]
                timestamp=timestamp,
            )
            self._debug(f"LLM: coach done (should_speak={suggestion.should_speak}, confidence={suggestion.confidence})", "ok")
            await self.on_event(suggestion)
        except Exception as exc:
            self._debug(f"LLM: coach failed — {exc}", "error")
            logger.warning("Coach task failed, skipping: %s", exc)

    async def shutdown(self) -> None:
        """Cancel all running reasoning tasks."""
        for task in list(self._running_tasks):
            if not task.done():
                task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
