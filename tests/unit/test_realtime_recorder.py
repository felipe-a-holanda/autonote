"""Tests for RealtimeRecorder."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autonote.realtime.recorder import (
    AudioDevice,
    CHUNK_SIZE,
    DeviceDefaults,
    RealtimeRecorder,
    RecordingStats,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TestRecordingStats:
    def test_default_values(self):
        stats = RecordingStats()
        assert stats.duration_seconds == 0.0
        assert stats.chunks_processed == 0
        assert stats.bytes_read == 0
        assert stats.is_recording is False
        assert stats.file_path is None
        assert stats.audio_files == []

    def test_custom_values(self):
        stats = RecordingStats(
            duration_seconds=120.5,
            chunks_processed=100,
            bytes_read=409600,
            is_recording=True,
            file_path="/tmp/meeting",
            audio_files=["mic.wav", "monitor.wav"],
        )
        assert stats.duration_seconds == 120.5
        assert stats.chunks_processed == 100
        assert stats.bytes_read == 409600
        assert stats.is_recording is True
        assert stats.file_path == "/tmp/meeting"
        assert len(stats.audio_files) == 2

    def test_audio_files_default_factory(self):
        s1 = RecordingStats()
        s2 = RecordingStats()
        s1.audio_files.append("a.wav")
        assert s2.audio_files == []  # Independent lists


class TestAudioDevice:
    def test_audio_device(self):
        device = AudioDevice(name="alsa_input.usb", description="USB Microphone")
        assert device.name == "alsa_input.usb"
        assert device.description == "USB Microphone"


class TestDeviceDefaults:
    def test_default_values(self):
        d = DeviceDefaults()
        assert d.source == ""
        assert d.sink == ""
        assert d.monitor == ""

    def test_custom_values(self):
        d = DeviceDefaults(source="mic", sink="speakers", monitor="speakers.monitor")
        assert d.source == "mic"
        assert d.monitor == "speakers.monitor"


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------

class TestCheckDependencies:
    def test_both_present(self):
        with patch("shutil.which", return_value="/usr/bin/tool"):
            ok, errors = run(RealtimeRecorder.check_dependencies())
        assert ok is True
        assert errors == []

    def test_missing_pactl(self):
        def which_side(name):
            return None if name == "pactl" else "/usr/bin/ffmpeg"
        with patch("shutil.which", side_effect=which_side):
            ok, errors = run(RealtimeRecorder.check_dependencies())
        assert ok is False
        assert any("pactl" in e for e in errors)

    def test_missing_ffmpeg(self):
        def which_side(name):
            return None if name == "ffmpeg" else "/usr/bin/pactl"
        with patch("shutil.which", side_effect=which_side):
            ok, errors = run(RealtimeRecorder.check_dependencies())
        assert ok is False
        assert any("ffmpeg" in e for e in errors)

    def test_both_missing(self):
        with patch("shutil.which", return_value=None):
            ok, errors = run(RealtimeRecorder.check_dependencies())
        assert ok is False
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# _build_stream_cmd
# ---------------------------------------------------------------------------

class TestBuildStreamCmd:
    def test_without_file(self):
        cmd = RealtimeRecorder._build_stream_cmd("my_source")
        assert cmd[0] == "ffmpeg"
        assert "my_source" in cmd
        assert "s16le" in cmd
        assert "pipe:1" in cmd
        assert "-map" not in cmd

    def test_with_file(self):
        cmd = RealtimeRecorder._build_stream_cmd("my_source", "/tmp/out.wav")
        assert cmd[0] == "ffmpeg"
        assert "my_source" in cmd
        assert "/tmp/out.wav" in cmd
        assert "-map" in cmd
        assert "pipe:1" in cmd

    def test_sample_rate_16k(self):
        cmd = RealtimeRecorder._build_stream_cmd("src")
        assert "16000" in cmd

    def test_mono_channel(self):
        cmd = RealtimeRecorder._build_stream_cmd("src")
        idx = cmd.index("-ac")
        assert cmd[idx + 1] == "1"

    def test_pulse_input_format(self):
        cmd = RealtimeRecorder._build_stream_cmd("src")
        idx = cmd.index("-f")
        assert cmd[idx + 1] == "pulse"


# ---------------------------------------------------------------------------
# _make_recording_path
# ---------------------------------------------------------------------------

class TestMakeRecordingPath:
    def test_creates_directory(self, tmp_path):
        ts = "20260404_120000"
        meeting_dir, mic_file, monitor_file = RealtimeRecorder._make_recording_path(
            str(tmp_path), ts
        )
        assert meeting_dir.exists()
        assert meeting_dir.name == f"meeting_{ts}"
        assert meeting_dir.parent.name == "20260404"

    def test_filenames(self, tmp_path):
        ts = "20260404_120000"
        meeting_dir, mic_file, monitor_file = RealtimeRecorder._make_recording_path(
            str(tmp_path), ts
        )
        assert mic_file == f"meeting_{ts}_mic.wav"
        assert monitor_file == f"meeting_{ts}_monitor.wav"

    def test_idempotent_mkdir(self, tmp_path):
        ts = "20260404_120000"
        RealtimeRecorder._make_recording_path(str(tmp_path), ts)
        # Calling again should not raise
        RealtimeRecorder._make_recording_path(str(tmp_path), ts)


# ---------------------------------------------------------------------------
# _write_metadata
# ---------------------------------------------------------------------------

class TestWriteMetadata:
    def test_writes_json_sidecar(self, tmp_path):
        ts = "20260404_120000"
        audio_files = ["meeting_20260404_120000_mic.wav"]
        RealtimeRecorder._write_metadata(tmp_path, "Test Meeting", ts, audio_files)

        meta_file = tmp_path / f"meeting_{ts}_metadata.json"
        assert meta_file.exists()
        data = json.loads(meta_file.read_text())
        assert data["title"] == "Test Meeting"
        assert data["timestamp"] == ts
        assert data["date"] == "20260404"
        assert data["audio_files"] == audio_files
        assert data["source"] == "realtime_recording"
        assert "created_at" in data

    def test_overwrites_existing(self, tmp_path):
        ts = "20260404_120000"
        RealtimeRecorder._write_metadata(tmp_path, "First", ts, [])
        RealtimeRecorder._write_metadata(tmp_path, "Second", ts, [])
        meta_file = tmp_path / f"meeting_{ts}_metadata.json"
        data = json.loads(meta_file.read_text())
        assert data["title"] == "Second"

    def test_empty_audio_files(self, tmp_path):
        ts = "20260404_120000"
        RealtimeRecorder._write_metadata(tmp_path, "Empty", ts, [])
        meta_file = tmp_path / f"meeting_{ts}_metadata.json"
        data = json.loads(meta_file.read_text())
        assert data["audio_files"] == []


# ---------------------------------------------------------------------------
# _drain_queue
# ---------------------------------------------------------------------------

class TestDrainQueue:
    def test_empties_queue(self):
        q: asyncio.Queue = asyncio.Queue()
        for item in [b"a", b"b", b"c"]:
            q.put_nowait(item)
        assert not q.empty()
        RealtimeRecorder._drain_queue(q)
        assert q.empty()

    def test_empty_queue_no_error(self):
        q: asyncio.Queue = asyncio.Queue()
        RealtimeRecorder._drain_queue(q)
        assert q.empty()

    def test_drains_all_items(self):
        q: asyncio.Queue = asyncio.Queue()
        for i in range(50):
            q.put_nowait(i)
        RealtimeRecorder._drain_queue(q)
        assert q.empty()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_recording_initial_false(self):
        r = RealtimeRecorder()
        assert r.is_recording is False

    def test_has_monitor_initial_false(self):
        r = RealtimeRecorder()
        assert r.has_monitor is False

    def test_has_monitor_with_process(self):
        r = RealtimeRecorder()
        r._monitor_process = MagicMock()
        assert r.has_monitor is True

    def test_meeting_dir_initial_none(self):
        r = RealtimeRecorder()
        assert r.meeting_dir is None

    def test_meeting_dir_when_set(self):
        r = RealtimeRecorder()
        r._meeting_dir = "/tmp/meeting"
        assert r.meeting_dir == "/tmp/meeting"

    def test_recording_stats_when_idle(self):
        r = RealtimeRecorder()
        stats = r.recording_stats
        assert stats.is_recording is False
        assert stats.duration_seconds == 0.0
        assert stats.chunks_processed == 0
        assert stats.bytes_read == 0

    def test_recording_stats_while_recording(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._start_time = time.monotonic() - 5.0
        r._chunks_processed = 42
        r._bytes_read = 163840

        stats = r.recording_stats
        assert stats.is_recording is True
        assert stats.duration_seconds >= 5.0
        assert stats.chunks_processed == 42
        assert stats.bytes_read == 163840

    def test_recording_stats_audio_files_copy(self):
        r = RealtimeRecorder()
        r._audio_files = ["mic.wav", "monitor.wav"]
        stats = r.recording_stats
        stats.audio_files.append("extra.wav")
        assert r._audio_files == ["mic.wav", "monitor.wav"]

    def test_set_crash_callback(self):
        r = RealtimeRecorder()
        cb = AsyncMock()
        r.set_crash_callback(cb)
        assert r._crash_callback is cb


# ---------------------------------------------------------------------------
# get_defaults
# ---------------------------------------------------------------------------

class TestGetDefaults:
    def test_uses_config_mic_override(self):
        r = RealtimeRecorder()
        with patch("autonote.realtime.recorder.config") as mock_config:
            mock_config.get.side_effect = lambda k, d="": {
                "MIC_SOURCE": "custom_mic",
                "SYSTEM_SOURCE": "",
            }.get(k, d)
            with patch.object(RealtimeRecorder, "_run_pactl", new_callable=AsyncMock, return_value="default_sink\n"):
                defaults = run(r.get_defaults())
        assert defaults.source == "custom_mic"

    def test_uses_config_system_override(self):
        r = RealtimeRecorder()
        with patch("autonote.realtime.recorder.config") as mock_config:
            mock_config.get.side_effect = lambda k, d="": {
                "MIC_SOURCE": "",
                "SYSTEM_SOURCE": "custom_monitor",
            }.get(k, d)
            with patch.object(RealtimeRecorder, "_run_pactl", new_callable=AsyncMock, return_value="default_source\n"):
                defaults = run(r.get_defaults())
        assert defaults.monitor == "custom_monitor"

    def test_auto_detect_from_pactl(self):
        r = RealtimeRecorder()

        async def mock_pactl(*args):
            return {"get-default-source": "mic_source\n", "get-default-sink": "out_sink\n"}.get(args[0], "")

        with patch("autonote.realtime.recorder.config") as mock_config:
            mock_config.get.return_value = ""
            with patch.object(RealtimeRecorder, "_run_pactl", side_effect=mock_pactl):
                defaults = run(r.get_defaults())

        assert defaults.source == "mic_source"
        assert defaults.sink == "out_sink"
        assert defaults.monitor == "out_sink.monitor"

    def test_pactl_failure_graceful(self):
        r = RealtimeRecorder()

        async def failing_pactl(*args):
            raise RuntimeError("pactl not found")

        with patch("autonote.realtime.recorder.config") as mock_config:
            mock_config.get.return_value = ""
            with patch.object(RealtimeRecorder, "_run_pactl", side_effect=failing_pactl):
                defaults = run(r.get_defaults())

        assert defaults.source == ""
        assert defaults.monitor == ""


# ---------------------------------------------------------------------------
# _handle_crash
# ---------------------------------------------------------------------------

class TestHandleCrash:
    def test_resets_recording_state(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._mic_process = MagicMock()
        r._monitor_process = MagicMock()
        r._start_time = time.monotonic()
        r._meeting_dir = "/tmp/meeting"
        r._timestamp = "20260404_120000"
        r._audio_files = ["mic.wav"]
        r._title = "Test"

        run(r._handle_crash())

        assert r._is_recording is False
        assert r._mic_process is None
        assert r._monitor_process is None
        assert r._start_time is None
        assert r._meeting_dir is None
        assert r._timestamp == ""
        assert r._audio_files == []
        assert r._title == ""

    def test_invokes_crash_callback(self):
        r = RealtimeRecorder()
        cb = AsyncMock()
        r._crash_callback = cb

        run(r._handle_crash())

        cb.assert_awaited_once()

    def test_no_callback_no_error(self):
        r = RealtimeRecorder()
        run(r._handle_crash())  # Should not raise

    def test_callback_exception_swallowed(self):
        r = RealtimeRecorder()
        cb = AsyncMock(side_effect=Exception("boom"))
        r._crash_callback = cb

        run(r._handle_crash())  # Should not propagate exception
        cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# start / stop guard checks
# ---------------------------------------------------------------------------

class TestStartStopGuards:
    def test_stop_when_not_recording_raises(self):
        r = RealtimeRecorder()
        with pytest.raises(RuntimeError, match="Not currently recording"):
            run(r.stop())

    def test_start_when_already_recording_raises(self):
        r = RealtimeRecorder()
        r._is_recording = True
        with pytest.raises(RuntimeError, match="already in progress"):
            run(r.start(mic_source="test", monitor_source=""))

    def test_start_with_no_mic_source_raises(self):
        r = RealtimeRecorder()

        async def mock_get_defaults():
            return DeviceDefaults(source="", sink="", monitor="")

        with patch.object(r, "get_defaults", side_effect=mock_get_defaults):
            with pytest.raises(RuntimeError, match="No microphone source"):
                run(r.start())


# ---------------------------------------------------------------------------
# _reader_loop
# ---------------------------------------------------------------------------

class TestReaderLoop:
    def test_reads_chunks_to_queue(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._stopping = False
        r._active_stream_count = 1

        chunks = [b"x" * CHUNK_SIZE, b"y" * CHUNK_SIZE, b""]

        async def mock_read(n):
            return chunks.pop(0)

        mock_stdout = MagicMock()
        mock_stdout.read = mock_read
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        run(r._reader_loop(mock_process, "Me", queue))

        items = []
        while not queue.empty():
            items.append(queue.get_nowait())

        assert items[0] == b"x" * CHUNK_SIZE
        assert items[1] == b"y" * CHUNK_SIZE
        assert items[2] is None
        assert r._chunks_processed == 2
        assert r._bytes_read == CHUNK_SIZE * 2

    def test_eof_puts_none_sentinel(self):
        r = RealtimeRecorder()
        r._is_recording = False
        r._stopping = True
        r._active_stream_count = 0

        async def mock_read(n):
            return b""

        mock_stdout = MagicMock()
        mock_stdout.read = mock_read
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        run(r._reader_loop(mock_process, "Me", queue))

        assert queue.get_nowait() is None

    def test_crash_detection_when_all_streams_exit(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._stopping = False
        r._active_stream_count = 1
        cb = AsyncMock()
        r._crash_callback = cb

        async def mock_read(n):
            return b""

        mock_stdout = MagicMock()
        mock_stdout.read = mock_read
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        run(r._reader_loop(mock_process, "Me", queue))

        cb.assert_awaited_once()

    def test_no_crash_when_stopping(self):
        r = RealtimeRecorder()
        r._is_recording = True
        r._stopping = True  # Graceful stop
        r._active_stream_count = 1
        cb = AsyncMock()
        r._crash_callback = cb

        async def mock_read(n):
            return b""

        mock_stdout = MagicMock()
        mock_stdout.read = mock_read
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        run(r._reader_loop(mock_process, "Me", queue))

        cb.assert_not_awaited()

    def test_partial_stream_exit_no_crash(self):
        """When one of two streams exits, crash handler not triggered yet."""
        r = RealtimeRecorder()
        r._is_recording = True
        r._stopping = False
        r._active_stream_count = 2  # Two streams running

        async def mock_read(n):
            return b""

        mock_stdout = MagicMock()
        mock_stdout.read = mock_read
        mock_process = MagicMock()
        mock_process.stdout = mock_stdout

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        run(r._reader_loop(mock_process, "Me", queue))

        assert r._active_stream_count == 1


# ---------------------------------------------------------------------------
# Start/stop integration (mocked subprocess)
# ---------------------------------------------------------------------------

def _make_blocking_proc():
    """Create a mock ffmpeg process whose stdout.read blocks indefinitely."""
    mock_proc = AsyncMock()
    mock_proc.stdout = MagicMock()

    async def blocking_read(n):
        await asyncio.sleep(3600)  # Block until cancelled
        return b""

    mock_proc.stdout.read = blocking_read
    mock_proc.send_signal = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    return mock_proc


class TestStartStopIntegration:
    def test_start_sets_recording_state(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source=""))

        assert r.is_recording is True
        assert r._mic_source == "test_mic"
        assert r._monitor_source == ""

        # Clean up: cancel reader tasks
        if r._mic_reader_task:
            r._mic_reader_task.cancel()

    def test_start_mic_only_no_monitor_process(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source=""))

        assert r.has_monitor is False
        if r._mic_reader_task:
            r._mic_reader_task.cancel()

    def test_start_with_monitor_source(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source="test_monitor"))

        assert r._monitor_process is not None
        for task in (r._mic_reader_task, r._monitor_reader_task):
            if task:
                task.cancel()

    def test_start_creates_meeting_dir_when_save_to_file(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=True)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source=""))

        assert r.meeting_dir is not None
        assert Path(r.meeting_dir).exists()
        if r._mic_reader_task:
            r._mic_reader_task.cancel()

    def test_stop_returns_stats(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source=""))

        assert r.is_recording is True
        stats = run(r.stop())

        assert stats.is_recording is False
        assert isinstance(stats.duration_seconds, float)
        assert r.is_recording is False

    def test_stop_resets_process_refs(self, tmp_path):
        r = RealtimeRecorder(recordings_dir=str(tmp_path), save_to_file=False)
        mock_proc = _make_blocking_proc()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            run(r.start(mic_source="test_mic", monitor_source=""))

        run(r.stop())

        assert r._mic_process is None
        assert r._monitor_process is None
