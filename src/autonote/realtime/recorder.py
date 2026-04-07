"""Dual-stream audio recorder for real-time meeting capture.

Captures mic ("Me") and system audio ("Them") via two separate ffmpeg/PulseAudio
processes. Each stream writes raw 16-kHz mono s16le PCM to an asyncio.Queue that
downstream consumers (e.g. the AssemblyAI transcriber) read from.

Ported from meeting-copilot/backend/audio/recorder.py — stripped of
FastAPI/WebSocket/pipeline bindings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable, Optional

from autonote.config import config

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4096  # bytes per read from ffmpeg stdout


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioDevice:
    """A PulseAudio source or sink."""

    name: str
    description: str


@dataclass
class DeviceDefaults:
    """Default PulseAudio source, sink, and derived monitor source."""

    source: str = ""
    sink: str = ""
    monitor: str = ""


@dataclass
class RecordingStats:
    """Stats collected during / after a recording session."""

    duration_seconds: float = 0.0
    chunks_processed: int = 0
    bytes_read: int = 0
    is_recording: bool = False
    file_path: Optional[str] = None
    audio_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class RealtimeRecorder:
    """Captures mic + system audio via PulseAudio/ffmpeg as separate PCM streams.

    Two separate ffmpeg processes are used — one for the mic source
    (speaker_label="Me") and one for the system monitor source
    (speaker_label="Them"). This gives deterministic speaker attribution
    at zero ML cost.

    Consumers read PCM bytes from :attr:`mic_queue` and :attr:`monitor_queue`.
    """

    def __init__(
        self,
        *,
        recordings_dir: Optional[str] = None,
        save_to_file: bool = False,
    ) -> None:
        self._recordings_dir = recordings_dir or config.get("RECORDINGS_DIR", "./recordings")
        self._save_to_file = save_to_file

        # Public queues — consumers read PCM bytes from these
        self.mic_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.monitor_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        # VAD queues — tee of mic/monitor for VADMonitor workers
        self.mic_vad_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.monitor_vad_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        # Internal state
        self._mic_process: Optional[asyncio.subprocess.Process] = None
        self._monitor_process: Optional[asyncio.subprocess.Process] = None
        self._mic_reader_task: Optional[asyncio.Task] = None
        self._monitor_reader_task: Optional[asyncio.Task] = None
        self._is_recording: bool = False
        self._stopping: bool = False
        self._active_stream_count: int = 0
        self._start_time: Optional[float] = None
        self._chunks_processed: int = 0
        self._bytes_read: int = 0
        self._mic_source: str = ""
        self._monitor_source: str = ""
        self._meeting_dir: Optional[str] = None
        self._timestamp: str = ""
        self._audio_files: list[str] = []
        self._title: str = ""
        self._crash_callback: Optional[Callable[[], Awaitable[None]]] = None

    def set_crash_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register an async callback invoked when all ffmpeg streams exit unexpectedly."""
        self._crash_callback = callback

    # ------------------------------------------------------------------
    # Dependency & device discovery
    # ------------------------------------------------------------------

    @staticmethod
    async def check_dependencies() -> tuple[bool, list[str]]:
        """Verify that pactl and ffmpeg are available on PATH.

        Returns:
            (all_ok, list_of_error_messages)
        """
        errors: list[str] = []
        if not shutil.which("pactl"):
            errors.append("pactl not found. Install pulseaudio-utils: sudo apt install pulseaudio-utils")
        if not shutil.which("ffmpeg"):
            errors.append("ffmpeg not found. Install ffmpeg: sudo apt install ffmpeg")
        return (len(errors) == 0, errors)

    @staticmethod
    async def _run_pactl(*args: str) -> str:
        """Run a pactl command and return stdout, or raise RuntimeError."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            raise RuntimeError("pactl is not installed or not on PATH")

        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else "unknown error"
            raise RuntimeError(f"pactl {' '.join(args)} failed: {err}")

        return stdout.decode()

    async def get_defaults(self) -> DeviceDefaults:
        """Get default source, sink, and derived monitor source name.

        Respects MIC_SOURCE and SYSTEM_SOURCE overrides from autonote config.
        """
        defaults = DeviceDefaults()

        # Check config overrides first
        cfg_mic = config.get("MIC_SOURCE", "").strip()
        cfg_system = config.get("SYSTEM_SOURCE", "").strip()

        if cfg_mic:
            defaults.source = cfg_mic
            logger.info("Using MIC_SOURCE override: %s", cfg_mic)
        else:
            try:
                output = await self._run_pactl("get-default-source")
                defaults.source = output.strip()
            except RuntimeError as exc:
                logger.warning("Could not get default source: %s", exc)

        if cfg_system:
            defaults.monitor = cfg_system
            logger.info("Using SYSTEM_SOURCE override: %s", cfg_system)
        else:
            try:
                output = await self._run_pactl("get-default-sink")
                defaults.sink = output.strip()
            except RuntimeError as exc:
                logger.warning("Could not get default sink: %s", exc)
            if defaults.sink:
                defaults.monitor = f"{defaults.sink}.monitor"

        return defaults

    # ------------------------------------------------------------------
    # ffmpeg command builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream_cmd(
        source: str,
        file_path: Optional[str] = None,
    ) -> list[str]:
        """Return the ffmpeg argument list for a single PulseAudio source stream.

        Outputs raw 16-kHz mono s16le PCM to pipe:1. When *file_path* is
        provided, also writes a PCM WAV file as a second output.
        """
        if file_path:
            return [
                "ffmpeg",
                "-f", "pulse", "-i", source,
                "-map", "0:a", "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1",
                "-map", "0:a", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", file_path,
            ]
        return [
            "ffmpeg",
            "-f", "pulse", "-i", source,
            "-ar", "16000", "-ac", "1",
            "-f", "s16le", "pipe:1",
        ]

    @staticmethod
    def _make_recording_path(recordings_dir: str, timestamp: str) -> tuple[Path, str, str]:
        """Create the dated meeting directory; return (meeting_dir, mic_filename, monitor_filename)."""
        date_str = timestamp[:8]
        meeting_dir = Path(recordings_dir) / date_str / f"meeting_{timestamp}"
        meeting_dir.mkdir(parents=True, exist_ok=True)
        mic_filename = f"meeting_{timestamp}_mic.wav"
        monitor_filename = f"meeting_{timestamp}_monitor.wav"
        return meeting_dir, mic_filename, monitor_filename

    @staticmethod
    def _write_metadata(
        meeting_dir: Path,
        title: str,
        timestamp: str,
        audio_files: list[str],
    ) -> None:
        """Write a JSON metadata sidecar file."""
        metadata_file = meeting_dir / f"meeting_{timestamp}_metadata.json"
        date_str = timestamp[:8]
        metadata = {
            "title": title,
            "timestamp": timestamp,
            "date": date_str,
            "audio_files": audio_files,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "realtime_recording",
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        mic_source: Optional[str] = None,
        monitor_source: Optional[str] = None,
        title: str = "",
    ) -> None:
        """Launch separate ffmpeg processes for mic and monitor streams.

        Args:
            mic_source: PulseAudio source for mic. None = auto-detect.
            monitor_source: PulseAudio monitor source. None = auto-detect.
                Pass empty string "" to force mic-only mode.
            title: Meeting title for metadata.
        """
        if self._is_recording:
            raise RuntimeError("Recording is already in progress")

        # Auto-detect devices if not explicitly provided
        if mic_source is None or monitor_source is None:
            defaults = await self.get_defaults()
            if mic_source is None:
                mic_source = defaults.source
            if monitor_source is None:
                monitor_source = defaults.monitor

        if not mic_source:
            raise RuntimeError("No microphone source available")

        # Prepare optional WAV file paths
        mic_file_path: Optional[str] = None
        monitor_file_path: Optional[str] = None
        meeting_dir_path: Optional[str] = None
        recording_timestamp: str = ""

        if self._save_to_file:
            recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meeting_dir, mic_filename, monitor_filename = self._make_recording_path(
                self._recordings_dir, recording_timestamp
            )
            mic_file_path = str(meeting_dir / mic_filename)
            monitor_file_path = str(meeting_dir / monitor_filename) if monitor_source else None
            meeting_dir_path = str(meeting_dir)
            logger.info("WAV output: mic=%s monitor=%s", mic_file_path, monitor_file_path)

        # Launch mic process (speaker_label="Me")
        mic_cmd = self._build_stream_cmd(mic_source, mic_file_path)
        logger.info("Starting mic ffmpeg: source=%s", mic_source)
        self._mic_process = await asyncio.create_subprocess_exec(
            *mic_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Launch monitor process (speaker_label="Them") if available
        if monitor_source:
            monitor_cmd = self._build_stream_cmd(monitor_source, monitor_file_path)
            logger.info("Starting monitor ffmpeg: source=%s", monitor_source)
            self._monitor_process = await asyncio.create_subprocess_exec(
                *monitor_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

        self._mic_source = mic_source
        self._monitor_source = monitor_source or ""
        self._is_recording = True
        self._stopping = False
        self._start_time = time.monotonic()
        self._chunks_processed = 0
        self._bytes_read = 0
        self._meeting_dir = meeting_dir_path
        self._timestamp = recording_timestamp
        self._audio_files = [p for p in [mic_file_path, monitor_file_path] if p is not None]
        self._title = title

        # Track active streams for crash detection
        self._active_stream_count = 1  # mic always started
        if self._monitor_process is not None:
            self._active_stream_count = 2

        # Drain old queues
        self._drain_queue(self.mic_queue)
        self._drain_queue(self.monitor_queue)
        self._drain_queue(self.mic_vad_queue)
        self._drain_queue(self.monitor_vad_queue)

        self._mic_reader_task = asyncio.create_task(
            self._reader_loop(self._mic_process, "Me", self.mic_queue, self.mic_vad_queue)
        )
        if self._monitor_process is not None:
            self._monitor_reader_task = asyncio.create_task(
                self._reader_loop(self._monitor_process, "Them", self.monitor_queue, self.monitor_vad_queue)
            )

    async def stop(self) -> RecordingStats:
        """Stop both ffmpeg processes and return recording stats."""
        if not self._is_recording:
            raise RuntimeError("Not currently recording")

        self._stopping = True

        # Graceful shutdown for each process
        if self._mic_process is not None:
            await self._stop_process(self._mic_process)
        if self._monitor_process is not None:
            await self._stop_process(self._monitor_process)

        # Cancel reader tasks after ffmpeg has exited
        for task in (self._mic_reader_task, self._monitor_reader_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._mic_reader_task = None
        self._monitor_reader_task = None

        # Signal consumers that streams are done
        await self.mic_queue.put(None)
        await self.monitor_queue.put(None)
        await self.mic_vad_queue.put(None)
        await self.monitor_vad_queue.put(None)

        duration = time.monotonic() - self._start_time if self._start_time is not None else 0.0

        # Write metadata sidecar if we were saving to file
        joined_file: Optional[str] = None
        if self._meeting_dir and self._audio_files:
            try:
                meeting_dir = Path(self._meeting_dir)
                self._write_metadata(
                    meeting_dir, self._title, self._timestamp,
                    [Path(f).name for f in self._audio_files],
                )
                logger.info("Metadata written for %s", self._meeting_dir)
            except Exception as exc:
                logger.error("Failed to write recording metadata: %s", exc)

            # Mix mic + monitor into a single joined WAV when both streams exist
            if len(self._audio_files) == 2:
                joined_path = Path(self._meeting_dir) / f"meeting_{self._timestamp}_joined.wav"
                try:
                    joined_file = await self._join_audio(
                        self._audio_files[0], self._audio_files[1], str(joined_path)
                    )
                except Exception as exc:
                    logger.error("Failed to create joined audio: %s", exc)

        all_audio = list(self._audio_files)
        if joined_file:
            all_audio.append(joined_file)
        stats = RecordingStats(
            duration_seconds=duration,
            chunks_processed=self._chunks_processed,
            bytes_read=self._bytes_read,
            is_recording=False,
            file_path=self._meeting_dir,
            audio_files=all_audio,
        )

        self._is_recording = False
        self._mic_process = None
        self._monitor_process = None
        self._start_time = None
        self._meeting_dir = None
        self._timestamp = ""
        self._audio_files = []
        self._title = ""

        logger.info(
            "Recording stopped — duration=%.1f s chunks=%d bytes=%d",
            duration, stats.chunks_processed, stats.bytes_read,
        )
        return stats

    @staticmethod
    async def _join_audio(mic_path: str, monitor_path: str, out_path: str) -> str:
        """Mix mic and monitor WAV files into a single mono WAV using ffmpeg amix.

        Returns the output path on success, raises RuntimeError on failure.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", mic_path,
            "-i", monitor_path,
            "-filter_complex", "amix=inputs=2:duration=longest:normalize=0",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            out_path,
        ]
        logger.info("Joining audio: %s + %s -> %s", mic_path, monitor_path, out_path)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg amix failed (rc={proc.returncode}): {stderr.decode()[-500:]}"
            )
        logger.info("Joined audio written to %s", out_path)
        return out_path

    async def _stop_process(self, proc: asyncio.subprocess.Process) -> None:
        """Send SIGINT to *proc*, wait up to 5 s, then SIGKILL if needed."""
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("ffmpeg did not exit within 5 s — sending SIGKILL")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

    # ------------------------------------------------------------------
    # Reader loop
    # ------------------------------------------------------------------

    async def _reader_loop(
        self,
        process: asyncio.subprocess.Process,
        speaker_label: str,
        queue: asyncio.Queue[bytes | None],
        vad_queue: asyncio.Queue[bytes | None] | None = None,
    ) -> None:
        """Read PCM chunks from ffmpeg stdout and push them to the consumer queue.

        Pushes ``None`` on EOF to signal stream end.  Implements crash
        detection: if all streams exit while recording is active and we're
        not in a graceful stop, the crash callback is invoked.

        When *vad_queue* is provided, each chunk is also tee'd into it so a
        downstream :class:`VADMonitor` can process the same audio in parallel.
        """
        assert process.stdout is not None

        try:
            while True:
                chunk = await process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    logger.info("Reader loop (%s): EOF from ffmpeg stdout", speaker_label)
                    break
                self._chunks_processed += 1
                self._bytes_read += len(chunk)
                await queue.put(chunk)
                if vad_queue is not None:
                    await vad_queue.put(chunk)
        except asyncio.CancelledError:
            logger.debug("Reader loop (%s) cancelled", speaker_label)
            raise
        except Exception as exc:
            logger.error("Reader loop (%s) unexpected error: %s", speaker_label, exc)
        finally:
            # Signal consumers that this stream is done
            await queue.put(None)
            if vad_queue is not None:
                await vad_queue.put(None)

            # Crash detection
            if self._is_recording and not self._stopping:
                self._active_stream_count -= 1
                if self._active_stream_count > 0:
                    logger.warning(
                        "Stream '%s' exited unexpectedly — %d stream(s) still running",
                        speaker_label, self._active_stream_count,
                    )
                else:
                    logger.error("All audio streams have exited unexpectedly — triggering crash handler")
                    await self._handle_crash()

    async def _handle_crash(self) -> None:
        """Reset recording state after an unrecoverable stream crash."""
        self._is_recording = False
        self._mic_process = None
        self._monitor_process = None
        self._mic_reader_task = None
        self._monitor_reader_task = None
        self._start_time = None
        self._meeting_dir = None
        self._timestamp = ""
        self._audio_files = []
        self._title = ""
        if self._crash_callback is not None:
            try:
                await self._crash_callback()
            except Exception as exc:
                logger.error("Crash callback raised an exception: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        """Discard all pending items in a queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def meeting_dir(self) -> Optional[str]:
        """Path to the active meeting directory (set only when save_to_file=True)."""
        return self._meeting_dir

    @property
    def has_monitor(self) -> bool:
        """Whether a monitor (system audio) stream is active."""
        return self._monitor_process is not None

    @property
    def recording_stats(self) -> RecordingStats:
        """Current stats snapshot (safe to call while recording)."""
        duration = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        return RecordingStats(
            duration_seconds=duration,
            chunks_processed=self._chunks_processed,
            bytes_read=self._bytes_read,
            is_recording=self._is_recording,
            file_path=self._meeting_dir,
            audio_files=list(self._audio_files),
        )
