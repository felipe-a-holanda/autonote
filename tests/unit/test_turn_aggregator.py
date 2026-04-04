"""Unit tests for TurnAggregator.

All tests are synchronous — asyncio.Queue.put_nowait / get_nowait work
without a running event loop in Python 3.10+.
"""

from __future__ import annotations

import pytest

from autonote.realtime.aggregator import TurnAggregator
from autonote.realtime.models import AggregatedTurn, TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(
    speaker: str = "Me",
    text: str = "hello",
    start: float = 0.0,
    end: float = 1.0,
    is_partial: bool = False,
) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker,
        text=text,
        timestamp_start=start,
        timestamp_end=end,
        is_partial=is_partial,
    )


def _drain(aggregator: TurnAggregator) -> list[AggregatedTurn | TranscriptSegment | None]:
    """Drain all items currently in output_queue without blocking."""
    items: list = []
    q = aggregator.output_queue
    while not q.empty():
        items.append(q.get_nowait())
    return items


# ---------------------------------------------------------------------------
# Basic turn creation
# ---------------------------------------------------------------------------

class TestSingleSegmentTurn:
    """A single final segment flushed via flush_remaining → one AggregatedTurn."""

    def test_produces_one_turn(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "hi", 0.0, 1.0))
        agg.flush_remaining()
        items = _drain(agg)
        assert len(items) == 1
        assert isinstance(items[0], AggregatedTurn)

    def test_turn_fields(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "hi there", 1.5, 2.5))
        agg.flush_remaining()
        turn: AggregatedTurn = _drain(agg)[0]
        assert turn.speaker == "Me"
        assert turn.text == "hi there"
        assert turn.timestamp_start == 1.5
        assert turn.timestamp_end == 2.5
        assert turn.segment_count == 1

    def test_type_discriminator(self):
        agg = TurnAggregator()
        agg.feed(_seg())
        agg.flush_remaining()
        turn = _drain(agg)[0]
        assert turn.type == "aggregated_turn"


# ---------------------------------------------------------------------------
# Multi-segment merging (same speaker, within thresholds)
# ---------------------------------------------------------------------------

class TestSameSpeakerMerge:
    def test_two_segments_merge_into_one_turn(self):
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "hello", 0.0, 1.0))
        agg.feed(_seg("Me", "world", 1.1, 2.0))  # gap = 0.1 s < threshold
        agg.flush_remaining()
        items = _drain(agg)
        assert len(items) == 1
        assert items[0].segment_count == 2

    def test_text_joined_with_space(self):
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "hello", 0.0, 1.0))
        agg.feed(_seg("Me", "world", 1.1, 2.0))
        agg.flush_remaining()
        turn = _drain(agg)[0]
        assert turn.text == "hello world"

    def test_timestamp_start_and_end(self):
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "a", 1.0, 2.0))
        agg.feed(_seg("Me", "b", 2.1, 3.5))
        agg.flush_remaining()
        turn = _drain(agg)[0]
        assert turn.timestamp_start == 1.0
        assert turn.timestamp_end == 3.5

    def test_segment_count_matches_number_of_finals(self):
        agg = TurnAggregator(silence_threshold=10.0)
        for i in range(5):
            agg.feed(_seg("Me", f"word{i}", float(i), float(i) + 0.8))
        agg.flush_remaining()
        turn = _drain(agg)[0]
        assert turn.segment_count == 5


# ---------------------------------------------------------------------------
# Flush triggers
# ---------------------------------------------------------------------------

class TestSpeakerChangeFlush:
    def test_speaker_change_flushes_previous_turn(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "I think", 0.0, 1.0))
        agg.feed(_seg("Them", "I agree", 1.5, 2.5))  # triggers flush of "Me"
        items = _drain(agg)
        # "Me" turn should be on the queue already
        assert len(items) == 1
        assert items[0].speaker == "Me"

    def test_both_turns_present_after_flush_remaining(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "I think", 0.0, 1.0))
        agg.feed(_seg("Them", "I agree", 1.5, 2.5))
        agg.flush_remaining()
        items = _drain(agg)
        assert len(items) == 2
        assert items[0].speaker == "Me"
        assert items[1].speaker == "Them"


class TestSilenceGapFlush:
    def test_gap_exceeds_threshold_flushes(self):
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "first", 0.0, 1.0))
        # gap = 5.0 - 1.0 = 4.0 s > 2.0 s → flush
        agg.feed(_seg("Me", "second", 5.0, 6.0))
        items = _drain(agg)
        assert len(items) == 1
        assert items[0].text == "first"

    def test_gap_within_threshold_does_not_flush(self):
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "first", 0.0, 1.0))
        # gap = 2.9 - 1.0 = 1.9 s < 2.0 s → no flush
        agg.feed(_seg("Me", "second", 2.9, 3.5))
        items = _drain(agg)
        assert items == []  # still buffered

    def test_gap_exactly_at_threshold_does_not_flush(self):
        """Boundary: gap == threshold → no flush (strictly greater triggers)."""
        agg = TurnAggregator(silence_threshold=2.0)
        agg.feed(_seg("Me", "first", 0.0, 1.0))
        agg.feed(_seg("Me", "second", 3.0, 4.0))  # gap = 2.0 s exactly
        items = _drain(agg)
        assert items == []


class TestMaxDurationFlush:
    def test_turn_exceeds_max_duration_flushes(self):
        agg = TurnAggregator(silence_threshold=100.0, max_turn_duration=5.0)
        agg.feed(_seg("Me", "start", 0.0, 1.0))
        # incoming end = 6.0 → duration = 6.0 - 0.0 = 6.0 > 5.0 → flush first
        agg.feed(_seg("Me", "end", 5.5, 6.0))
        items = _drain(agg)
        assert len(items) == 1
        assert items[0].text == "start"

    def test_turn_within_max_duration_not_flushed(self):
        agg = TurnAggregator(silence_threshold=100.0, max_turn_duration=30.0)
        agg.feed(_seg("Me", "a", 0.0, 1.0))
        agg.feed(_seg("Me", "b", 1.5, 2.0))
        items = _drain(agg)
        assert items == []


# ---------------------------------------------------------------------------
# Partials
# ---------------------------------------------------------------------------

class TestPartialsForwardedImmediately:
    def test_partial_appears_in_queue_without_flush(self):
        agg = TurnAggregator()
        partial = _seg("Me", "typing...", 0.0, 0.5, is_partial=True)
        agg.feed(partial)
        items = _drain(agg)
        assert len(items) == 1
        assert items[0] is partial
        assert items[0].is_partial is True

    def test_partial_does_not_affect_buffer(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "final", 0.0, 1.0))
        agg.feed(_seg("Me", "typing...", 1.0, 1.2, is_partial=True))
        # Only the partial should be in the queue — final is still buffered
        items = _drain(agg)
        assert len(items) == 1
        assert items[0].is_partial is True

    def test_multiple_partials_all_forwarded(self):
        agg = TurnAggregator()
        for i in range(3):
            agg.feed(_seg("Them", f"part{i}", float(i), float(i) + 0.5, is_partial=True))
        items = _drain(agg)
        assert len(items) == 3
        assert all(it.is_partial for it in items)


# ---------------------------------------------------------------------------
# flush_remaining edge cases
# ---------------------------------------------------------------------------

class TestFlushRemaining:
    def test_flushes_buffered_segment(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "done", 0.0, 1.0))
        assert agg.output_queue.empty()
        agg.flush_remaining()
        assert not agg.output_queue.empty()

    def test_empty_buffer_is_noop(self):
        agg = TurnAggregator()
        agg.flush_remaining()  # nothing buffered
        assert agg.output_queue.empty()

    def test_double_flush_does_not_duplicate(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "hello", 0.0, 1.0))
        agg.flush_remaining()
        agg.flush_remaining()  # second call — buffer is empty now
        items = _drain(agg)
        assert len(items) == 1

    def test_flush_remaining_clears_buffer(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "a", 0.0, 1.0))
        agg.flush_remaining()
        _drain(agg)
        # Feed a new segment after flush — should create a fresh turn
        agg.feed(_seg("Me", "b", 5.0, 6.0))
        agg.flush_remaining()
        items = _drain(agg)
        assert len(items) == 1
        assert items[0].text == "b"


# ---------------------------------------------------------------------------
# Wall time
# ---------------------------------------------------------------------------

class TestWallTime:
    def test_wall_time_start_and_end_set_on_flush(self):
        agg = TurnAggregator()
        agg.feed(_seg("Me", "hi", 0.0, 1.0))
        agg.flush_remaining()
        turn: AggregatedTurn = _drain(agg)[0]
        assert turn.wall_time_start is not None
        assert turn.wall_time_end is not None
        assert turn.wall_time_end >= turn.wall_time_start
