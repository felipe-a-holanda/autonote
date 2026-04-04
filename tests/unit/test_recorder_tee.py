"""Tests for recorder VAD queue tee functionality (Task 3.3)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from autonote.realtime.recorder import CHUNK_SIZE, RealtimeRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blocking_proc():
    """Mock ffmpeg process whose stdout.read blocks until cancelled."""
    mock_proc = AsyncMock()
    mock_proc.stdout = MagicMock()

    async def blocking_read(n):
        await asyncio.sleep(3600)
        return b""

    mock_proc.stdout.read = blocking_read
    mock_proc.send_signal = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    return mock_proc


def _make_chunked_proc(chunks: list[bytes]):
    """Mock ffmpeg process that yields the given chunks then EOF."""
    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()
    remaining = list(chunks)

    async def mock_read(n):
        return remaining.pop(0) if remaining else b""

    mock_proc.stdout.read = mock_read
    return mock_proc


# ---------------------------------------------------------------------------
# VAD queue existence
# ---------------------------------------------------------------------------

class TestVADQueueAttributes:
    def test_mic_vad_queue_exists(self):
        r = RealtimeRecorder()
        assert hasattr(r, "mic_vad_queue")
        assert isinstance(r.mic_vad_queue, asyncio.Queue)

    def test_monitor_vad_queue_exists(self):
        r = RealtimeRecorder()
        assert hasattr(r, "monitor_vad_queue")
        assert isinstance(r.monitor_vad_queue, asyncio.Queue)

    def test_vad_queues_start_empty(self):
        r = RealtimeRecorder()
        assert r.mic_vad_queue.empty()
        assert r.monitor_vad_queue.empty()


# ---------------------------------------------------------------------------
# _reader_loop tee behaviour
# ---------------------------------------------------------------------------

class TestReaderLoopTee:
    async def test_chunks_appear_in_both_queues(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._stopping = False
        r._active_stream_count = 1

        chunk_a = b"a" * CHUNK_SIZE
        chunk_b = b"b" * CHUNK_SIZE
        proc = _make_chunked_proc([chunk_a, chunk_b, b""])

        main_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad_q: asyncio.Queue[bytes | None] = asyncio.Queue()

        await r._reader_loop(proc, "Me", main_q, vad_q)

        main_items = []
        while not main_q.empty():
            main_items.append(main_q.get_nowait())

        vad_items = []
        while not vad_q.empty():
            vad_items.append(vad_q.get_nowait())

        assert main_items[0] == chunk_a
        assert main_items[1] == chunk_b
        assert main_items[2] is None  # sentinel

        assert vad_items[0] == chunk_a
        assert vad_items[1] == chunk_b
        assert vad_items[2] is None  # sentinel

    async def test_sentinel_sent_to_both_on_eof(self):
        r = RealtimeRecorder()
        r._is_recording = False
        r._stopping = True
        r._active_stream_count = 0

        proc = _make_chunked_proc([b""])  # immediate EOF

        main_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad_q: asyncio.Queue[bytes | None] = asyncio.Queue()

        await r._reader_loop(proc, "Me", main_q, vad_q)

        assert main_q.get_nowait() is None
        assert vad_q.get_nowait() is None

    async def test_no_vad_queue_still_works(self):
        """Existing callers without vad_queue continue to work."""
        r = RealtimeRecorder()
        r._is_recording = False
        r._stopping = True
        r._active_stream_count = 0

        chunk = b"x" * CHUNK_SIZE
        proc = _make_chunked_proc([chunk, b""])

        main_q: asyncio.Queue[bytes | None] = asyncio.Queue()

        await r._reader_loop(proc, "Me", main_q)  # no vad_queue

        items = []
        while not main_q.empty():
            items.append(main_q.get_nowait())

        assert items[0] == chunk
        assert items[1] is None

    async def test_chunks_are_identical_references(self):
        """Same bytes object goes into both queues (no copy)."""
        r = RealtimeRecorder()
        r._is_recording = False
        r._stopping = True
        r._active_stream_count = 0

        chunk = b"data" * 1024
        proc = _make_chunked_proc([chunk, b""])

        main_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad_q: asyncio.Queue[bytes | None] = asyncio.Queue()

        await r._reader_loop(proc, "Me", main_q, vad_q)

        main_chunk = main_q.get_nowait()
        vad_chunk = vad_q.get_nowait()

        assert main_chunk == vad_chunk


# ---------------------------------------------------------------------------
# start() drains and wires VAD queues
# ---------------------------------------------------------------------------

class TestStartDrainsVADQueues:
    async def test_start_drains_stale_vad_items(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)

        # Pre-populate VAD queues with stale data
        r.mic_vad_queue.put_nowait(b"stale")
        r.monitor_vad_queue.put_nowait(b"stale")

        mock_proc = _make_blocking_proc()
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await r.start(mic_source="test_mic", monitor_source="")

        assert r.mic_vad_queue.empty()

        # Cleanup
        if r._mic_reader_task:
            r._mic_reader_task.cancel()

    async def test_start_wires_mic_vad_queue(self, tmp_path):
        """After start(), mic_vad_queue receives the same chunks as mic_queue."""
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)

        chunk = b"pcm" * (CHUNK_SIZE // 3)
        reads = [chunk, b""]
        idx = 0

        async def mock_read(n):
            nonlocal idx
            val = reads[idx] if idx < len(reads) else b""
            idx += 1
            return val

        mock_proc = AsyncMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read = mock_read
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await r.start(mic_source="test_mic", monitor_source="")

        # Let reader task run
        await asyncio.sleep(0.05)

        vad_items = []
        while not r.mic_vad_queue.empty():
            vad_items.append(r.mic_vad_queue.get_nowait())

        assert chunk in vad_items


# ---------------------------------------------------------------------------
# stop() sends sentinels to VAD queues
# ---------------------------------------------------------------------------

class TestStopSentinels:
    async def test_stop_puts_none_in_mic_vad_queue(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await r.start(mic_source="test_mic", monitor_source="")

        await r.stop()

        # Drain until None
        sentinel_found = False
        while not r.mic_vad_queue.empty():
            item = r.mic_vad_queue.get_nowait()
            if item is None:
                sentinel_found = True
                break

        assert sentinel_found, "mic_vad_queue should contain a None sentinel after stop()"

    async def test_stop_puts_none_in_monitor_vad_queue(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await r.start(mic_source="test_mic", monitor_source="test_monitor")

        await r.stop()

        sentinel_found = False
        while not r.monitor_vad_queue.empty():
            item = r.monitor_vad_queue.get_nowait()
            if item is None:
                sentinel_found = True
                break

        assert sentinel_found, "monitor_vad_queue should contain a None sentinel after stop()"
