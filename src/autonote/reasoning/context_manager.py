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
    ActionItem,
    SummaryUpdate,
    ActionItemsUpdate,
    ContradictionAlert,
    RealtimeEvent,
)
from autonote.reasoning.dispatcher import LLMDispatcher
from autonote.reasoning.workers.summary import SummaryWorker
from autonote.reasoning.workers.action_items import ActionItemWorker
from autonote.reasoning.workers.contradictions import ContradictionWorker
from autonote.reasoning.workers.custom import CustomPromptWorker
from autonote.reasoning.workers.reply import ReplyWorker

logger = logging.getLogger(__name__)


@dataclass
class MeetingState:
    """Accumulated state of the meeting, fed to LLM workers."""

    session_id: str
    start_time: float

    segments: list[TranscriptSegment] = field(default_factory=list)
    speakers: set[str] = field(default_factory=set)
    current_summary: str = ""
    action_items: list[ActionItem] = field(default_factory=list)
    recent_window: deque = field(default_factory=lambda: deque(maxlen=50))
    segments_since_last_summary: int = 0
    segments_since_last_action_scan: int = 0

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)
        self.recent_window.append(segment)
        self.speakers.add(segment.speaker)
        self.segments_since_last_summary += 1
        self.segments_since_last_action_scan += 1

    def get_transcript_text(self, last_n: Optional[int] = None) -> str:
        if last_n is not None:
            source = list(self.recent_window)[-last_n:]
        else:
            source = self.segments
        return "\n".join(
            f"[{s.speaker} @ {s.timestamp_start:.0f}s]: {s.text}"
            for s in source
        )

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
        parts.append(f"## Recent Transcript\n{self.get_transcript_text(last_n=30)}")
        return "\n\n".join(parts)


# Type for the event callback — receives any RealtimeEvent
EventCallback = Callable[[RealtimeEvent], Awaitable[None]]


class ContextManager:
    """Manages meeting state and triggers reasoning tasks on configurable thresholds.

    Args:
        dispatcher: The LLM dispatcher for routing reasoning tasks.
        on_event: Async callback invoked for every event (transcript segment,
            summary update, action items, etc.). The TUI binds to this.
        summary_every_n: Trigger summary update every N final segments.
        action_scan_every_n: Trigger action item scan every N final segments.
        contradiction_check_seconds: Interval for contradiction checks.
        session_id: Identifier for this meeting session.
    """

    SUMMARY_EVERY_N_SEGMENTS: int = 10
    ACTION_SCAN_EVERY_N_SEGMENTS: int = 5
    CONTRADICTION_CHECK_SECONDS: int = 120

    def __init__(
        self,
        dispatcher: LLMDispatcher,
        on_event: EventCallback,
        *,
        summary_every_n: Optional[int] = None,
        action_scan_every_n: Optional[int] = None,
        contradiction_check_seconds: Optional[int] = None,
        session_id: str = "",
    ) -> None:
        self.state = MeetingState(
            session_id=session_id,
            start_time=time.time(),
        )
        self.dispatcher = dispatcher
        self.on_event = on_event
        self._running_tasks: set[asyncio.Task] = set()
        self._last_contradiction_check: float = time.time()

        self._summary_worker = SummaryWorker(dispatcher)
        self._action_item_worker = ActionItemWorker(dispatcher)
        self._contradiction_worker = ContradictionWorker(dispatcher)
        self._custom_prompt_worker = CustomPromptWorker(dispatcher)
        self._reply_worker = ReplyWorker(dispatcher)

        if summary_every_n is not None:
            self.SUMMARY_EVERY_N_SEGMENTS = summary_every_n
        if action_scan_every_n is not None:
            self.ACTION_SCAN_EVERY_N_SEGMENTS = action_scan_every_n
        if contradiction_check_seconds is not None:
            self.CONTRADICTION_CHECK_SECONDS = contradiction_check_seconds

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

    async def handle_custom_prompt(self, prompt: str) -> None:
        """Run a user's freeform prompt against the meeting context."""
        try:
            timestamp = (
                self.state.segments[-1].timestamp_end if self.state.segments else 0.0
            )
            result = await self._custom_prompt_worker.execute(
                full_context=self.state.get_full_context(),
                user_prompt=prompt,
                timestamp=timestamp,
            )
            await self.on_event(result)
        except Exception as exc:
            logger.warning("Custom prompt failed: %s", exc)

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

    def _fire_task(self, coro: Any) -> None:
        """Spawn a background task and track it for cleanup."""
        task = asyncio.create_task(coro)
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    async def _run_summary(self) -> None:
        try:
            new_segments = self.state.get_transcript_text(
                last_n=self.SUMMARY_EVERY_N_SEGMENTS
            )
            result = await self._summary_worker.execute(
                current_summary=self.state.current_summary,
                new_segments=new_segments,
            )
            self.state.current_summary = result
            await self.on_event(
                SummaryUpdate(
                    summary=result,
                    covered_until=self.state.segments[-1].timestamp_end,
                )
            )
        except Exception as exc:
            logger.warning("Summary task failed, skipping: %s", exc)

    async def _run_contradictions(self) -> None:
        try:
            alerts = await self._contradiction_worker.execute(
                current_summary=self.state.current_summary,
                recent_transcript=self.state.get_transcript_text(last_n=20),
            )
            for alert in alerts:
                await self.on_event(alert)
        except Exception as exc:
            logger.warning("Contradiction check failed, skipping: %s", exc)

    async def _run_action_items(self) -> None:
        try:
            updated_items = await self._action_item_worker.execute(
                full_context=self.state.get_full_context(),
                recent_transcript=self.state.get_transcript_text(last_n=10),
                existing_items=self.state.action_items,
            )
            self.state.action_items = updated_items
            await self.on_event(
                ActionItemsUpdate(items=self.state.action_items)
            )
        except Exception as exc:
            logger.warning("Action items task failed, skipping: %s", exc)

    async def shutdown(self) -> None:
        """Cancel all running reasoning tasks."""
        for task in list(self._running_tasks):
            if not task.done():
                task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
