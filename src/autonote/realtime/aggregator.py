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

Usage::

    aggregator = TurnAggregator()
    aggregator.feed(segment)          # called for each TranscriptSegment
    aggregator.flush_remaining()      # called at session end

    # Consumer reads from aggregator.output_queue
    item = await aggregator.output_queue.get()
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

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
    ) -> None:
        self.silence_threshold = silence_threshold
        self.max_turn_duration = max_turn_duration

        self.output_queue: asyncio.Queue[AggregatedTurn | TranscriptSegment | None] = (
            asyncio.Queue()
        )

        self._buffer: list[TranscriptSegment] = []
        self._current_speaker: Optional[str] = None
        self._turn_start_time: Optional[float] = None
        self._last_segment_end: Optional[float] = None
        self._wall_time_start: Optional[datetime] = None

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
        if self._should_flush(segment):
            self._flush()

        # Buffer the final segment.
        if not self._buffer:
            # First segment in a new turn — record start metadata.
            self._current_speaker = segment.speaker
            self._turn_start_time = segment.timestamp_start
            self._wall_time_start = datetime.now(timezone.utc)

        self._buffer.append(segment)
        self._last_segment_end = segment.timestamp_end

    def flush_remaining(self) -> None:
        """Flush any buffered segments at session end.

        No-op when the buffer is empty.
        """
        self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_flush(self, incoming: TranscriptSegment) -> bool:
        """Return True if the buffer should be flushed before adding *incoming*."""
        if not self._buffer:
            return False

        # 1. Speaker change.
        if incoming.speaker != self._current_speaker:
            return True

        # 2. Silence gap exceeds threshold.
        if self._last_segment_end is not None:
            gap = incoming.timestamp_start - self._last_segment_end
            if gap > self.silence_threshold:
                return True

        # 3. Turn would exceed max duration.
        if self._turn_start_time is not None:
            duration = incoming.timestamp_end - self._turn_start_time
            if duration > self.max_turn_duration:
                return True

        return False

    def _flush(self) -> None:
        """Merge buffered segments into one :class:`AggregatedTurn` and enqueue it."""
        if not self._buffer:
            return

        text = " ".join(seg.text for seg in self._buffer)
        turn = AggregatedTurn(
            speaker=self._current_speaker or "",
            text=text,
            timestamp_start=self._buffer[0].timestamp_start,
            timestamp_end=self._buffer[-1].timestamp_end,
            segment_count=len(self._buffer),
            wall_time_start=self._wall_time_start,
            wall_time_end=datetime.now(timezone.utc),
        )
        self.output_queue.put_nowait(turn)

        # Reset buffer state.
        self._buffer = []
        self._current_speaker = None
        self._turn_start_time = None
        self._last_segment_end = None
        self._wall_time_start = None
