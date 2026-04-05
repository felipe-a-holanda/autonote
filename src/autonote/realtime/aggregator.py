"""Turn aggregation for the real-time meeting copilot pipeline.

Groups consecutive :class:`TranscriptSegment` finals from the same speaker
into :class:`AggregatedTurn` events.  Partials are forwarded immediately so
the TUI can show a live "typing" line.

Flush triggers (for final segments):
  1. Speaker change — incoming speaker differs from buffered speaker.
  2. Silence gap  — gap between last segment end and incoming segment start
                     exceeds ``silence_threshold`` (default 2.0 s).
  3. Max duration — total buffered turn duration exceeds ``max_turn_duration``
                     (default 30.0 s).
  4. Wall-clock silence — real elapsed time since last final exceeds
                     ``silence_threshold`` (run_silence_timer task).

Usage::

    aggregator = TurnAggregator()
    aggregator.feed(segment)          # called for each TranscriptSegment
    aggregator.flush_remaining()      # called at session end

    # Consumer reads from aggregator.output_queue
    item = await aggregator.output_queue.get()
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from autonote.realtime.models import AggregatedTurn, TranscriptSegment


class TurnAggregator:
    """Aggregates final transcript segments into speaker turns.

    Partials are forwarded immediately to ``output_queue``.
    Finals are buffered and flushed as :class:`AggregatedTurn` on trigger.
    """

    def __init__(
        self,
        *,
        silence_threshold: float = 2.0,
        max_turn_duration: float = 30.0,
        on_debug: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.silence_threshold = silence_threshold
        self.max_turn_duration = max_turn_duration
        self._on_debug = on_debug

        self.output_queue: asyncio.Queue[AggregatedTurn | TranscriptSegment | None] = (
            asyncio.Queue()
        )

        self._buffer: list[TranscriptSegment] = []
        self._current_speaker: Optional[str] = None
        self._turn_start_time: Optional[float] = None
        self._last_segment_end: Optional[float] = None
        self._wall_time_start: Optional[datetime] = None
        self._last_final_wall_time: float = 0.0  # time.time() of last buffered final

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, segment: TranscriptSegment) -> None:
        """Process one incoming segment.

        Partials are put on ``output_queue`` immediately.
        Finals are buffered; a flush may be triggered first.
        """
        if segment.is_partial:
            self.output_queue.put_nowait(segment)
            return

        # Final segment — decide whether to flush before buffering.
        flush_reason = self._flush_reason(segment)
        if flush_reason:
            self._dbg(
                f"[{segment.speaker}] agg: flush triggered ({flush_reason}, buf={len(self._buffer)})",
                "warn",
            )
            self._flush()

        # Log gap from last segment end (AAI word timestamps).
        gap_info = ""
        if self._last_segment_end is not None:
            gap = segment.timestamp_start - self._last_segment_end
            gap_info = f" gap={gap:.2f}s"

        buf_after = len(self._buffer) + 1
        self._dbg(
            f"[{segment.speaker}] agg: buffered \"{segment.text}\" "
            f"aai={segment.timestamp_start:.2f}–{segment.timestamp_end:.2f}s"
            f"{gap_info} buf→{buf_after}"
        )

        # Buffer the final segment.
        if not self._buffer:
            # First segment in a new turn — record start metadata.
            self._current_speaker = segment.speaker
            self._turn_start_time = segment.timestamp_start
            self._wall_time_start = datetime.now(timezone.utc)

        self._buffer.append(segment)
        self._last_segment_end = segment.timestamp_end
        self._last_final_wall_time = time.time()

    def flush_remaining(self) -> None:
        """Flush any buffered segments at session end.

        No-op when the buffer is empty.
        """
        if self._buffer:
            self._dbg(f"[{self._current_speaker}] agg: flush remaining (buf={len(self._buffer)})", "warn")
        self._flush()

    async def run_silence_timer(self) -> None:
        """Background task: flush buffered turn when wall-clock silence exceeds threshold.

        Polls every 100 ms. Fires when real time since the last buffered final
        exceeds ``silence_threshold``, bypassing the AAI-timestamp-based check.
        This prevents long holds caused by the other speaker being slow to respond.
        """
        while True:
            await asyncio.sleep(0.1)
            if not self._buffer or self._last_final_wall_time == 0.0:
                continue
            elapsed = time.time() - self._last_final_wall_time
            if elapsed >= self.silence_threshold:
                self._dbg(
                    f"[{self._current_speaker}] agg: wall-clock flush "
                    f"(silence={elapsed:.2f}s >= {self.silence_threshold}s, buf={len(self._buffer)})",
                    "warn",
                )
                self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dbg(self, msg: str, level: str = "info") -> None:
        if self._on_debug is not None:
            self._on_debug(msg, level)

    def _flush_reason(self, incoming: TranscriptSegment) -> Optional[str]:
        """Return the flush reason string, or None if no flush needed."""
        if not self._buffer:
            return None

        if incoming.speaker != self._current_speaker:
            return f"speaker-change ({self._current_speaker}→{incoming.speaker})"

        if self._last_segment_end is not None:
            gap = incoming.timestamp_start - self._last_segment_end
            if gap > self.silence_threshold:
                return f"silence-gap={gap:.2f}s (threshold={self.silence_threshold}s)"

        if self._turn_start_time is not None:
            duration = incoming.timestamp_end - self._turn_start_time
            if duration > self.max_turn_duration:
                return f"max-duration={duration:.1f}s"

        return None

    def _flush(self) -> None:
        """Merge buffered segments into one :class:`AggregatedTurn` and enqueue it."""
        if not self._buffer:
            return

        flush_wall_time = time.time()
        first_received = self._buffer[0].received_wall_time
        text = " ".join(seg.text for seg in self._buffer)
        turn = AggregatedTurn(
            speaker=self._current_speaker or "",
            text=text,
            timestamp_start=self._buffer[0].timestamp_start,
            timestamp_end=self._buffer[-1].timestamp_end,
            segment_count=len(self._buffer),
            wall_time_start=self._wall_time_start,
            wall_time_end=datetime.now(timezone.utc),
            first_received_wall_time=first_received,
            flushed_wall_time=flush_wall_time,
        )

        hold_ms = (flush_wall_time - first_received) * 1000 if first_received > 0 else -1
        self._dbg(
            f"[{turn.speaker}] agg: emitting turn n={turn.segment_count} "
            f"aai={turn.timestamp_start:.2f}–{turn.timestamp_end:.2f}s "
            f"held={hold_ms:.0f}ms \"{text}\"",
            "ok",
        )

        self.output_queue.put_nowait(turn)

        # Reset buffer state.
        self._buffer = []
        self._current_speaker = None
        self._turn_start_time = None
        self._last_segment_end = None
        self._wall_time_start = None
