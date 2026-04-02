"""Live transcription via AssemblyAI real-time streaming.

Runs two parallel AssemblyAI RealtimeTranscriber sessions — one for the mic
stream ("Me") and one for the system audio stream ("Them"). PCM bytes are read
from the recorder's asyncio.Queues and fed to the SDK's sync ``stream()``
method via dedicated feeder tasks.

Final and partial transcripts are converted to :class:`TranscriptSegment`
models and pushed to a shared output queue for downstream consumption
(ContextManager / TUI).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable

from autonote.config import config
from autonote.realtime.models import TranscriptSegment

logger = logging.getLogger(__name__)

# Sample rate must match the recorder's ffmpeg output (16 kHz mono s16le)
SAMPLE_RATE = 16_000


class RealtimeTranscriber:
    """Bridges recorder PCM queues to AssemblyAI real-time transcription.

    Two AssemblyAI sessions run in parallel. Each produces
    :class:`TranscriptSegment` events on :attr:`segment_queue`.

    Usage::

        transcriber = RealtimeTranscriber(
            mic_queue=recorder.mic_queue,
            monitor_queue=recorder.monitor_queue,
        )
        await transcriber.start()
        # ... read from transcriber.segment_queue ...
        await transcriber.stop()
    """

    def __init__(
        self,
        *,
        mic_queue: asyncio.Queue[bytes | None],
        monitor_queue: Optional[asyncio.Queue[bytes | None]] = None,
        api_key: Optional[str] = None,
        on_error: Optional[Callable[[str, Exception], Awaitable[None]]] = None,
    ) -> None:
        self._mic_queue = mic_queue
        self._monitor_queue = monitor_queue
        self._api_key = api_key or config.get("ASSEMBLYAI_API_KEY", "")
        self._on_error = on_error

        # Public output queue — consumers read TranscriptSegments from here
        self.segment_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()

        self._mic_transcriber = None
        self._monitor_transcriber = None
        self._mic_feeder_task: Optional[asyncio.Task] = None
        self._monitor_feeder_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_start_time: float = 0.0
        self._running: bool = False

    async def start(self) -> None:
        """Connect to AssemblyAI and begin streaming from recorder queues."""
        try:
            import assemblyai as aai
        except ImportError:
            raise RuntimeError(
                "assemblyai package not installed. "
                "Install with: pip install -e '.[realtime]'"
            )

        if not self._api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY not set. Add it to your .autonoterc or "
                "pass --api-key to the CLI."
            )

        aai.settings.api_key = self._api_key
        self._loop = asyncio.get_running_loop()
        self._session_start_time = time.monotonic()
        self._running = True

        # Create mic transcriber (always)
        self._mic_transcriber = self._create_transcriber("Me", aai)
        self._mic_transcriber.connect()
        logger.info("AssemblyAI mic transcriber connected")

        # Create monitor transcriber (if monitor queue provided)
        if self._monitor_queue is not None:
            self._monitor_transcriber = self._create_transcriber("Them", aai)
            self._monitor_transcriber.connect()
            logger.info("AssemblyAI monitor transcriber connected")

        # Spawn feeder tasks that bridge asyncio queues → sync SDK
        self._mic_feeder_task = asyncio.create_task(
            self._feed_loop(self._mic_queue, self._mic_transcriber, "Me")
        )
        if self._monitor_queue is not None and self._monitor_transcriber is not None:
            self._monitor_feeder_task = asyncio.create_task(
                self._feed_loop(self._monitor_queue, self._monitor_transcriber, "Them")
            )

    async def stop(self) -> None:
        """Gracefully close all AssemblyAI sessions."""
        self._running = False

        # Cancel feeder tasks
        for task in (self._mic_feeder_task, self._monitor_feeder_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Close transcriber sessions (sync calls, run in thread to avoid blocking)
        for transcriber in (self._mic_transcriber, self._monitor_transcriber):
            if transcriber is not None:
                try:
                    await asyncio.to_thread(transcriber.close)
                except Exception as exc:
                    logger.warning("Error closing transcriber: %s", exc)

        self._mic_transcriber = None
        self._monitor_transcriber = None
        self._mic_feeder_task = None
        self._monitor_feeder_task = None

        # Signal downstream that no more segments are coming
        await self.segment_queue.put(None)
        logger.info("Transcriber stopped")

    def _create_transcriber(self, speaker_label: str, aai_module):
        """Create an AssemblyAI RealtimeTranscriber for one audio stream."""

        def on_data(transcript) -> None:
            """Called from SDK thread — bridge to asyncio."""
            # Import here to access types without top-level aai import
            from assemblyai import types as aai_types

            is_final = isinstance(transcript, aai_types.RealtimeFinalTranscript)
            is_partial = isinstance(transcript, aai_types.RealtimePartialTranscript)

            if not (is_final or is_partial):
                return

            text = transcript.text.strip()
            if not text:
                return

            segment = TranscriptSegment(
                speaker=speaker_label,
                text=text,
                timestamp_start=transcript.audio_start / 1000.0,
                timestamp_end=transcript.audio_end / 1000.0,
                is_partial=is_partial,
            )

            # Thread-safe push to asyncio queue
            if self._loop is not None and self._running:
                self._loop.call_soon_threadsafe(self.segment_queue.put_nowait, segment)

        def on_error(error) -> None:
            """Called from SDK thread on transcription errors."""
            logger.error("AssemblyAI error (%s): %s", speaker_label, error)
            if self._on_error and self._loop:
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self._on_error(speaker_label, Exception(str(error))),
                )

        def on_open(session) -> None:
            logger.info(
                "AssemblyAI session opened (%s): id=%s",
                speaker_label,
                getattr(session, "session_id", "unknown"),
            )

        def on_close() -> None:
            logger.info("AssemblyAI session closed (%s)", speaker_label)

        return aai_module.RealtimeTranscriber(
            sample_rate=SAMPLE_RATE,
            encoding=aai_module.AudioEncoding.pcm_s16le,
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )

    async def _feed_loop(
        self,
        pcm_queue: asyncio.Queue[bytes | None],
        transcriber,
        speaker_label: str,
    ) -> None:
        """Read PCM chunks from an asyncio queue and feed them to the SDK.

        The SDK's ``stream(bytes)`` method is non-blocking (puts into an
        internal queue), so calling it from the async context is safe.
        """
        try:
            while self._running:
                chunk = await pcm_queue.get()
                if chunk is None:
                    logger.info("Feed loop (%s): received EOF sentinel", speaker_label)
                    break
                # stream() is thread-safe and non-blocking (internal queue put)
                transcriber.stream(chunk)
        except asyncio.CancelledError:
            logger.debug("Feed loop (%s) cancelled", speaker_label)
        except Exception as exc:
            logger.error("Feed loop (%s) error: %s", speaker_label, exc)
