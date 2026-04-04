"""Tests for VADMonitor async worker.

Uses synchronous helpers where possible (asyncio.Queue.put_nowait / get_nowait
work without a running event loop in Python 3.10+). Async tests use pytest-asyncio
auto mode — no decorator needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autonote.realtime.models import SpeechStateEvent
from autonote.realtime.vad_monitor import VADMonitor, _WINDOW_SAMPLES, _SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vad(responses: list[bool]) -> MagicMock:
    """Return a mock SileroVAD whose is_speech() cycles through `responses`."""
    vad = MagicMock()
    vad.is_speech.side_effect = responses
    return vad


def _silence_chunk(n_windows: int = 1) -> bytes:
    """Return raw s16le bytes representing n_windows × 512 silent samples."""
    samples = np.zeros(_WINDOW_SAMPLES * n_windows, dtype=np.int16)
    return samples.tobytes()


def _speech_chunk(n_windows: int = 1) -> bytes:
    """Return raw s16le bytes representing n_windows × 512 non-zero samples."""
    samples = np.ones(_WINDOW_SAMPLES * n_windows, dtype=np.int16) * 1000
    return samples.tobytes()


def _make_monitor(
    responses: list[bool],
    silence_threshold: float = 1.5,
    speaker: str = "Me",
) -> tuple[VADMonitor, asyncio.Queue, asyncio.Queue]:
    """Create a VADMonitor with mocked VAD and return (monitor, in_q, out_q)."""
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    vad = _make_vad(responses)
    monitor = VADMonitor(
        speaker=speaker,
        vad=vad,
        input_queue=in_q,
        output_queue=out_q,
        silence_threshold=silence_threshold,
    )
    return monitor, in_q, out_q


def _drain(q: asyncio.Queue) -> list:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


# ---------------------------------------------------------------------------
# Synchronous _process_window tests (no event loop needed)
# ---------------------------------------------------------------------------

class TestProcessWindowStateMachine:
    """Direct tests of _process_window() without an async event loop."""

    def test_silence_to_speech_emits_speech_start(self):
        monitor, _, out_q = _make_monitor(responses=[True])
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        assert not monitor._in_speech
        monitor._process_window(audio)

        assert monitor._in_speech
        events = _drain(out_q)
        assert len(events) == 1
        assert events[0].event_type == "speech_start"
        assert events[0].speaker == "Me"

    def test_speech_start_timestamp_is_window_start_time(self):
        monitor, _, out_q = _make_monitor(responses=[True])
        # Advance sample count by 2 windows manually
        monitor._sample_count = _WINDOW_SAMPLES * 2
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        monitor._process_window(audio)

        event = out_q.get_nowait()
        expected_time = (_WINDOW_SAMPLES * 2) / _SAMPLE_RATE
        assert event.timestamp == pytest.approx(expected_time)

    def test_continuous_speech_no_spurious_events(self):
        """Many consecutive speech windows → only one speech_start, no speech_end."""
        monitor, _, out_q = _make_monitor(responses=[True] * 10)
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        for _ in range(10):
            monitor._process_window(audio)

        events = _drain(out_q)
        assert len(events) == 1
        assert events[0].event_type == "speech_start"

    def test_brief_silence_does_not_emit_speech_end(self):
        """Silence shorter than threshold does not trigger speech_end."""
        # silence_threshold = 1.5s → need 1.5 * 16000 / 512 = 46.875 → 47 windows
        # We feed 10 silence windows (< threshold)
        monitor, _, out_q = _make_monitor(
            responses=[True] + [False] * 10,
            silence_threshold=1.5,
        )
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        for _ in range(11):
            monitor._process_window(audio)

        events = _drain(out_q)
        # Only speech_start, no speech_end
        assert all(e.event_type == "speech_start" for e in events)
        assert monitor._in_speech  # still in speech state

    def test_silence_exceeding_threshold_emits_speech_end(self):
        """Silence >= threshold triggers speech_end with silence_duration set."""
        # threshold = 0.032s → 1 window (512 / 16000 = 0.032s)
        threshold = _WINDOW_SAMPLES / _SAMPLE_RATE  # exactly 1 window
        monitor, _, out_q = _make_monitor(
            responses=[True, False],
            silence_threshold=threshold,
        )
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        monitor._process_window(audio)  # speech_start
        monitor._process_window(audio)  # silence — exactly threshold → speech_end

        events = _drain(out_q)
        assert len(events) == 2
        assert events[0].event_type == "speech_start"
        assert events[1].event_type == "speech_end"
        assert events[1].silence_duration == pytest.approx(threshold)
        assert not monitor._in_speech

    def test_speech_end_silence_duration_accumulates_over_multiple_windows(self):
        """silence_duration reflects all consecutive silent windows at the moment it fires."""
        # 3 silent windows after speech; threshold between 2 and 3 windows → fires on 3rd
        n_silence = 3
        threshold = (2.5 * _WINDOW_SAMPLES) / _SAMPLE_RATE  # fires when 3rd window counted
        monitor, _, out_q = _make_monitor(
            responses=[True] + [False] * n_silence,
            silence_threshold=threshold,
        )
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        for _ in range(1 + n_silence):
            monitor._process_window(audio)

        events = _drain(out_q)
        assert events[-1].event_type == "speech_end"
        # Silence duration when speech_end fires = 3 windows * (512/16000)s
        expected_silence = (n_silence * _WINDOW_SAMPLES) / _SAMPLE_RATE
        assert events[-1].silence_duration == pytest.approx(expected_silence)

    def test_silence_timer_resets_on_speech_resumption(self):
        """Speech resuming before threshold resets the silence timer."""
        # threshold = 3 windows
        threshold = (3 * _WINDOW_SAMPLES) / _SAMPLE_RATE
        # Pattern: speech, silence×2, speech (resets timer), silence×2, silence×1 → total 3 silence at end
        monitor, _, out_q = _make_monitor(
            responses=[True, False, False, True, False, False, False],
            silence_threshold=threshold,
        )
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        # Initial speech
        monitor._process_window(audio)  # True → speech_start
        assert monitor._in_speech

        # 2 silence windows (< threshold, no speech_end yet)
        monitor._process_window(audio)  # False
        monitor._process_window(audio)  # False
        assert monitor._in_speech  # still in speech

        # Speech resumes — should reset silence timer
        monitor._process_window(audio)  # True → reset timer, no new event
        assert monitor._silence_start_sample is None

        # Now 3 more silence windows → crosses threshold
        monitor._process_window(audio)  # False — starts new silence timer
        monitor._process_window(audio)  # False
        monitor._process_window(audio)  # False — exactly threshold → speech_end

        events = _drain(out_q)
        assert events[0].event_type == "speech_start"
        assert events[-1].event_type == "speech_end"
        assert not monitor._in_speech

    def test_starting_in_silence_no_events(self):
        """Windows with no speech never emit events."""
        monitor, _, out_q = _make_monitor(responses=[False] * 5)
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)

        for _ in range(5):
            monitor._process_window(audio)

        assert _drain(out_q) == []
        assert not monitor._in_speech

    def test_speaker_label_propagated(self):
        monitor, _, out_q = _make_monitor(responses=[True], speaker="Them")
        audio = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
        monitor._process_window(audio)

        event = out_q.get_nowait()
        assert event.speaker == "Them"


# ---------------------------------------------------------------------------
# _process_chunk tests (bytes → windows)
# ---------------------------------------------------------------------------

class TestProcessChunk:
    def test_exact_window_size_chunk(self):
        """A chunk of exactly 512 samples (1024 bytes) produces one window."""
        monitor, _, out_q = _make_monitor(responses=[True])
        monitor._process_chunk(_speech_chunk(n_windows=1))

        assert out_q.get_nowait().event_type == "speech_start"

    def test_multi_window_chunk_processed_in_order(self):
        """A chunk spanning 3 windows processes all 3 windows in sequence."""
        monitor, _, out_q = _make_monitor(responses=[False, False, True])
        monitor._process_chunk(_silence_chunk(n_windows=2) + _speech_chunk(n_windows=1))

        events = _drain(out_q)
        assert len(events) == 1
        assert events[0].event_type == "speech_start"

    def test_partial_chunk_buffered_until_next(self):
        """A chunk with fewer than 512 samples is buffered, not processed yet."""
        monitor, _, out_q = _make_monitor(responses=[True])
        # Send half a window (256 samples = 512 bytes)
        half = np.zeros(256, dtype=np.int16).tobytes()
        monitor._process_chunk(half)
        assert _drain(out_q) == []  # not processed yet
        assert len(monitor._buffer) == 256

        # Send the second half — now a full window is available
        monitor._process_chunk(half)
        assert len(_drain(out_q)) == 1

    def test_remaining_samples_stay_in_buffer(self):
        """After full windows are consumed, leftover samples stay in buffer."""
        monitor, _, out_q = _make_monitor(responses=[True])
        # 1.5 windows of samples
        samples = np.ones(int(_WINDOW_SAMPLES * 1.5), dtype=np.int16) * 500
        monitor._process_chunk(samples.tobytes())

        # One window processed
        assert len(_drain(out_q)) == 1
        # Half a window remaining in buffer
        assert len(monitor._buffer) == _WINDOW_SAMPLES // 2


# ---------------------------------------------------------------------------
# Async run() tests
# ---------------------------------------------------------------------------

class TestVADMonitorRun:
    async def test_run_stops_on_none_sentinel(self):
        """run() terminates when it reads None from the input queue."""
        monitor, in_q, out_q = _make_monitor(responses=[])
        in_q.put_nowait(None)

        await monitor.run()  # should return without hanging

    async def test_speech_to_silence_emits_both_events(self):
        """Full scenario: speech then silence exceeding threshold."""
        # 1 window of speech, then enough silence windows to exceed 1.5s threshold
        # 1.5s / (512/16000) = 46.875 → need 47 silence windows
        n_silence = 47
        responses = [True] + [False] * n_silence
        monitor, in_q, out_q = _make_monitor(responses=responses, silence_threshold=1.5)

        # One speech chunk + n_silence silence chunks + sentinel
        in_q.put_nowait(_speech_chunk(n_windows=1))
        for _ in range(n_silence):
            in_q.put_nowait(_silence_chunk(n_windows=1))
        in_q.put_nowait(None)

        await monitor.run()

        events = _drain(out_q)
        speech_starts = [e for e in events if e.event_type == "speech_start"]
        speech_ends = [e for e in events if e.event_type == "speech_end"]

        assert len(speech_starts) == 1
        assert len(speech_ends) == 1
        assert speech_ends[0].silence_duration >= 1.5

    async def test_silence_to_speech_emits_speech_start(self):
        """Starting silent then encountering speech emits speech_start."""
        responses = [False] * 5 + [True]
        monitor, in_q, out_q = _make_monitor(responses=responses)

        for _ in range(5):
            in_q.put_nowait(_silence_chunk())
        in_q.put_nowait(_speech_chunk())
        in_q.put_nowait(None)

        await monitor.run()

        events = _drain(out_q)
        assert len(events) == 1
        assert events[0].event_type == "speech_start"

    async def test_continuous_speech_no_spurious_events(self):
        """Only one speech_start emitted for continuous speech."""
        n = 20
        monitor, in_q, out_q = _make_monitor(responses=[True] * n)

        for _ in range(n):
            in_q.put_nowait(_speech_chunk())
        in_q.put_nowait(None)

        await monitor.run()

        events = _drain(out_q)
        assert len(events) == 1
        assert events[0].event_type == "speech_start"

    async def test_brief_pause_no_speech_end(self):
        """Brief silence (< threshold) between speech segments does not emit speech_end."""
        # threshold = 1.5s, brief pause = 5 windows (5*512/16000 = 0.16s)
        n_brief_silence = 5
        responses = [True] * 3 + [False] * n_brief_silence + [True] * 3
        monitor, in_q, out_q = _make_monitor(responses=responses, silence_threshold=1.5)

        for _ in range(3):
            in_q.put_nowait(_speech_chunk())
        for _ in range(n_brief_silence):
            in_q.put_nowait(_silence_chunk())
        for _ in range(3):
            in_q.put_nowait(_speech_chunk())
        in_q.put_nowait(None)

        await monitor.run()

        events = _drain(out_q)
        speech_ends = [e for e in events if e.event_type == "speech_end"]
        assert len(speech_ends) == 0

    async def test_empty_queue_just_sentinel_no_events(self):
        """Only None in queue → no events emitted."""
        monitor, in_q, out_q = _make_monitor(responses=[])
        in_q.put_nowait(None)

        await monitor.run()

        assert _drain(out_q) == []
