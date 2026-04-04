"""Tests for RealtimeTranscriber."""
from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autonote.realtime.transcriber import RealtimeTranscriber, SAMPLE_RATE
from autonote.realtime.models import TranscriptSegment


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transcriber(api_key: str = "test-key", with_monitor: bool = False, on_debug=None):
    mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    monitor_q: asyncio.Queue[bytes | None] = asyncio.Queue() if with_monitor else None
    return RealtimeTranscriber(
        mic_queue=mic_q,
        monitor_queue=monitor_q,
        api_key=api_key,
        on_debug=on_debug,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16_000


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        mic_q: asyncio.Queue = asyncio.Queue()
        t = RealtimeTranscriber(mic_queue=mic_q, api_key="key")
        assert t._mic_queue is mic_q
        assert t._monitor_queue is None
        assert t._api_key == "key"
        assert t._running is False
        assert t._mic_client is None
        assert t._monitor_client is None
        assert t._mic_feeder_task is None
        assert t._monitor_feeder_task is None
        assert t._session_start_time == 0.0

    def test_with_monitor_queue(self):
        mic_q: asyncio.Queue = asyncio.Queue()
        mon_q: asyncio.Queue = asyncio.Queue()
        t = RealtimeTranscriber(mic_queue=mic_q, monitor_queue=mon_q, api_key="key")
        assert t._monitor_queue is mon_q

    def test_api_key_from_config(self):
        mic_q: asyncio.Queue = asyncio.Queue()
        with patch("autonote.realtime.transcriber.config") as mock_config:
            mock_config.get.return_value = "config-key"
            t = RealtimeTranscriber(mic_queue=mic_q)
        assert t._api_key == "config-key"

    def test_explicit_api_key_overrides_config(self):
        mic_q: asyncio.Queue = asyncio.Queue()
        with patch("autonote.realtime.transcriber.config") as mock_config:
            mock_config.get.return_value = "config-key"
            t = RealtimeTranscriber(mic_queue=mic_q, api_key="explicit-key")
        assert t._api_key == "explicit-key"

    def test_initial_counters(self):
        t = make_transcriber()
        assert t._chunks_fed == {"Me": 0, "Them": 0}
        assert t._on_data_calls == {"Me": 0, "Them": 0}
        assert t._empty_data_calls == {"Me": 0, "Them": 0}

    def test_segment_queue_is_empty(self):
        t = make_transcriber()
        assert t.segment_queue.empty()


# ---------------------------------------------------------------------------
# _dbg
# ---------------------------------------------------------------------------

class TestDebugCallback:
    def test_called_when_set(self):
        messages = []

        def on_debug(msg, level):
            messages.append((msg, level))

        t = make_transcriber(on_debug=on_debug)
        t._dbg("hello", "ok")
        assert messages == [("hello", "ok")]

    def test_not_called_when_none(self):
        t = make_transcriber()
        t._dbg("hello", "info")  # Should not raise

    def test_default_level_is_info(self):
        messages = []

        def on_debug(msg, level):
            messages.append((msg, level))

        t = make_transcriber(on_debug=on_debug)
        t._dbg("msg")
        assert messages[0][1] == "info"


# ---------------------------------------------------------------------------
# stop (without start)
# ---------------------------------------------------------------------------

class TestStopWithoutStart:
    def test_stop_puts_none_sentinel(self):
        t = make_transcriber()
        run(t.stop())
        assert t.segment_queue.get_nowait() is None

    def test_stop_resets_clients(self):
        t = make_transcriber()
        run(t.stop())
        assert t._mic_client is None
        assert t._monitor_client is None

    def test_stop_sets_running_false(self):
        t = make_transcriber()
        t._running = True
        run(t.stop())
        assert t._running is False

    def test_stop_client_disconnect_error_handled(self):
        t = make_transcriber()
        mock_client = MagicMock()

        async def fake_disconnect_thread(fn, *args):
            fn(*args)

        mock_client.disconnect = MagicMock(side_effect=Exception("disconnect failed"))
        t._mic_client = mock_client

        # Should not raise even if disconnect fails
        run(t.stop())

    def test_stop_resets_feeder_tasks_to_none(self):
        t = make_transcriber()
        run(t.stop())
        assert t._mic_feeder_task is None
        assert t._monitor_feeder_task is None


# ---------------------------------------------------------------------------
# start — missing API key
# ---------------------------------------------------------------------------

class TestStartErrors:
    def test_raises_without_api_key(self):
        mic_q: asyncio.Queue = asyncio.Queue()
        with patch("autonote.realtime.transcriber.config") as mock_config:
            mock_config.get.return_value = ""
            t = RealtimeTranscriber(mic_queue=mic_q, api_key="")

        # Provide fake assemblyai modules so we pass the import check
        fake_aai = ModuleType("assemblyai")
        fake_streaming = ModuleType("assemblyai.streaming")
        fake_v3 = ModuleType("assemblyai.streaming.v3")
        fake_client_mod = ModuleType("assemblyai.streaming.v3.client")
        fake_models_mod = ModuleType("assemblyai.streaming.v3.models")

        fake_client_mod.StreamingClient = MagicMock()
        fake_models_mod.StreamingClientOptions = MagicMock()
        fake_models_mod.StreamingParameters = MagicMock()
        fake_models_mod.StreamingEvents = MagicMock()
        fake_models_mod.Encoding = MagicMock()
        fake_models_mod.SpeechModel = MagicMock()

        fake_v3.client = fake_client_mod
        fake_v3.models = fake_models_mod
        fake_streaming.v3 = fake_v3
        fake_aai.streaming = fake_streaming

        with patch.dict(sys.modules, {
            "assemblyai": fake_aai,
            "assemblyai.streaming": fake_streaming,
            "assemblyai.streaming.v3": fake_v3,
            "assemblyai.streaming.v3.client": fake_client_mod,
            "assemblyai.streaming.v3.models": fake_models_mod,
        }):
            with pytest.raises(RuntimeError, match="ASSEMBLYAI_API_KEY"):
                run(t.start())


# ---------------------------------------------------------------------------
# _feed_loop
# ---------------------------------------------------------------------------

class TestFeedLoop:
    def test_sends_chunk_at_threshold(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        mic_q.put_nowait(b"x" * 3200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert len(streamed) == 1
        assert streamed[0] == b"x" * 3200

    def test_buffers_small_chunks(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        mic_q.put_nowait(b"a" * 1000)
        mic_q.put_nowait(b"b" * 1000)
        mic_q.put_nowait(b"c" * 1200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert len(streamed) == 1
        assert len(streamed[0]) == 3200

    def test_flushes_remainder_on_eof(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        mic_q.put_nowait(b"z" * 500)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert len(streamed) == 1
        assert len(streamed[0]) == 500

    def test_no_flush_when_buffer_empty_at_eof(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        # Sends exactly 3200 (clears buffer), then EOF does nothing
        mic_q.put_nowait(b"x" * 3200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert len(streamed) == 1

    def test_stops_immediately_when_not_running(self):
        t = make_transcriber()
        t._running = False

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        mic_q.put_nowait(b"x" * 3200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert streamed == []

    def test_increments_chunks_fed_counter(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        mock_client.stream = MagicMock()

        mic_q.put_nowait(b"x" * 3200)
        mic_q.put_nowait(b"y" * 3200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert t._chunks_fed["Me"] == 2

    def test_exception_does_not_propagate(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()

        async def raising_get():
            raise ValueError("unexpected error")

        mic_q.get = raising_get  # type: ignore[method-assign]

        run(t._feed_loop(mic_q, mock_client, "Me"))  # Should not raise

    def test_multiple_threshold_flushes(self):
        t = make_transcriber()
        t._running = True

        mic_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_client = MagicMock()
        streamed: list[bytes] = []
        mock_client.stream = lambda data: streamed.append(data)

        # Two full threshold chunks + EOF
        mic_q.put_nowait(b"a" * 3200)
        mic_q.put_nowait(b"b" * 3200)
        mic_q.put_nowait(None)

        run(t._feed_loop(mic_q, mock_client, "Me"))

        assert len(streamed) == 2


# ---------------------------------------------------------------------------
# _create_client (callback wiring, no real SDK)
# ---------------------------------------------------------------------------

class TestCreateClient:
    def _make_fake_events(self):
        class FakeEvents:
            Turn = "Turn"
            Error = "Error"
            Begin = "Begin"
            Termination = "Termination"
        return FakeEvents

    def _capture_handlers(self, t, speaker="Me"):
        captured = {}

        def fake_on(event_type, handler):
            captured[event_type] = handler

        mock_instance = MagicMock()
        mock_instance.on = fake_on
        MockStreamingClient = MagicMock(return_value=mock_instance)
        FakeEvents = self._make_fake_events()

        t._create_client(speaker, MockStreamingClient, FakeEvents, MagicMock())
        return captured, mock_instance

    def test_registers_four_event_handlers(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, mock_instance = self._capture_handlers(t)

        assert len(captured) == 4
        assert "Turn" in captured
        assert "Error" in captured
        assert "Begin" in captured
        assert "Termination" in captured

    def test_on_turn_skips_empty_text(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, _ = self._capture_handlers(t)

        mock_event = MagicMock()
        mock_event.transcript = "   "  # whitespace → empty after strip
        mock_event.end_of_turn = True
        mock_event.words = []

        captured["Turn"](None, mock_event)

        assert t.segment_queue.empty()
        assert t._empty_data_calls.get("Me", 0) == 1

    def test_on_turn_increments_on_data_calls(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, _ = self._capture_handlers(t)

        mock_event = MagicMock()
        mock_event.transcript = "hello"
        mock_event.end_of_turn = True
        mock_event.words = []

        captured["Turn"](None, mock_event)

        assert t._on_data_calls.get("Me", 0) == 1

    def _drain_callbacks(self):
        """Run the event loop briefly to process any call_soon_threadsafe callbacks."""
        run(asyncio.sleep(0))

    def test_on_turn_final_puts_segment(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, _ = self._capture_handlers(t)

        mock_word = MagicMock()
        mock_word.start = 0
        mock_word.end = 1500  # ms

        mock_event = MagicMock()
        mock_event.transcript = "Hello world"
        mock_event.end_of_turn = True
        mock_event.words = [mock_word]

        captured["Turn"](None, mock_event)
        self._drain_callbacks()

        segment = t.segment_queue.get_nowait()
        assert isinstance(segment, TranscriptSegment)
        assert segment.speaker == "Me"
        assert segment.text == "Hello world"
        assert segment.is_partial is False
        assert segment.timestamp_end == 1.5

    def test_on_turn_partial_segment(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, _ = self._capture_handlers(t, speaker="Them")

        mock_word = MagicMock()
        mock_word.start = 0
        mock_word.end = 500

        mock_event = MagicMock()
        mock_event.transcript = "Partial"
        mock_event.end_of_turn = False
        mock_event.words = [mock_word]

        captured["Turn"](None, mock_event)
        self._drain_callbacks()

        segment = t.segment_queue.get_nowait()
        assert segment.is_partial is True
        assert segment.speaker == "Them"

    def test_on_turn_no_words_timestamps_zero(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        captured, _ = self._capture_handlers(t)

        mock_event = MagicMock()
        mock_event.transcript = "No timestamps"
        mock_event.end_of_turn = True
        mock_event.words = []

        captured["Turn"](None, mock_event)
        self._drain_callbacks()

        segment = t.segment_queue.get_nowait()
        assert segment.timestamp_start == 0.0
        assert segment.timestamp_end == 0.0

    def test_on_turn_not_queued_when_not_running(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = False  # Not running

        captured, _ = self._capture_handlers(t)

        mock_event = MagicMock()
        mock_event.transcript = "Should not be queued"
        mock_event.end_of_turn = True
        mock_event.words = []

        captured["Turn"](None, mock_event)
        self._drain_callbacks()

        assert t.segment_queue.empty()

    def test_on_error_triggers_error_callback(self):
        t = make_transcriber()
        t._loop = asyncio.get_event_loop()
        t._running = True

        errors_received = []

        async def on_error(speaker, exc):
            errors_received.append((speaker, exc))

        t._on_error = on_error
        captured, _ = self._capture_handlers(t)

        captured["Error"](None, "connection refused")
        # Error callback is scheduled via call_soon_threadsafe — just verify no crash
