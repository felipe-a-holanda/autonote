"""Live transcription via AssemblyAI Streaming v3.

Runs two parallel StreamingClient sessions — one for the mic stream ("Me")
and one for the system audio stream ("Them"). PCM bytes are read from the
recorder's asyncio.Queues and fed to the SDK's sync ``stream()`` method via
dedicated feeder tasks.

Final and partial transcripts are converted to :class:`TranscriptSegment`
models and pushed to a shared output queue for downstream consumption.
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

DebugCallback = Callable[[str, str], None]  # (message, level)


class RealtimeTranscriber:
    """Bridges recorder PCM queues to AssemblyAI Streaming v3.

    Two StreamingClient sessions run in parallel (mic = "Me", monitor = "Them").
    Each produces :class:`TranscriptSegment` events on :attr:`segment_queue`.

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
        on_debug: Optional[DebugCallback] = None,
    ) -> None:
        self._mic_queue = mic_queue
        self._monitor_queue = monitor_queue
        self._api_key = api_key or config.get("ASSEMBLYAI_API_KEY", "")
        self._on_error = on_error
        self._on_debug = on_debug

        # Public output queue — consumers read TranscriptSegments from here
        self.segment_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()

        self._mic_client = None
        self._monitor_client = None
        self._mic_feeder_task: Optional[asyncio.Task] = None
        self._monitor_feeder_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_start_time: float = 0.0
        self._running: bool = False

        # Debug counters
        self._chunks_fed: dict[str, int] = {"Me": 0, "Them": 0}
        self._on_data_calls: dict[str, int] = {"Me": 0, "Them": 0}
        self._empty_data_calls: dict[str, int] = {"Me": 0, "Them": 0}

    def _dbg(self, msg: str, level: str = "info") -> None:
        if self._on_debug is not None:
            self._on_debug(msg, level)

    async def start(self) -> None:
        """Connect to AssemblyAI and begin streaming from recorder queues."""
        try:
            from assemblyai.streaming.v3.client import StreamingClient
            from assemblyai.streaming.v3.models import (
                StreamingClientOptions,
                StreamingParameters,
                StreamingEvents,
                Encoding,
                SpeechModel,
            )
        except ImportError:
            raise RuntimeError(
                "assemblyai package not installed or outdated. "
                "Install with: pip install -e '.[realtime]'"
            )

        if not self._api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY not set. Add it to your .autonoterc or "
                "pass --api-key to the CLI."
            )

        self._loop = asyncio.get_running_loop()
        self._session_start_time = time.monotonic()
        self._running = True

        options = StreamingClientOptions(api_key=self._api_key)
        params = StreamingParameters(
            sample_rate=SAMPLE_RATE,
            encoding=Encoding.pcm_s16le,
            speech_model=SpeechModel.universal_streaming_english,
        )

        # Create mic client (always)
        self._mic_client = self._create_client("Me", StreamingClient, StreamingEvents, options)
        await asyncio.to_thread(self._mic_client.connect, params)
        logger.info("AssemblyAI mic client connected")

        # Create monitor client (if monitor queue provided)
        if self._monitor_queue is not None:
            self._monitor_client = self._create_client("Them", StreamingClient, StreamingEvents, options)
            await asyncio.to_thread(self._monitor_client.connect, params)
            logger.info("AssemblyAI monitor client connected")

        # Spawn feeder tasks
        self._mic_feeder_task = asyncio.create_task(
            self._feed_loop(self._mic_queue, self._mic_client, "Me")
        )
        if self._monitor_queue is not None and self._monitor_client is not None:
            self._monitor_feeder_task = asyncio.create_task(
                self._feed_loop(self._monitor_queue, self._monitor_client, "Them")
            )

    async def stop(self) -> None:
        """Gracefully close all AssemblyAI sessions."""
        self._running = False

        for task in (self._mic_feeder_task, self._monitor_feeder_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        for client in (self._mic_client, self._monitor_client):
            if client is not None:
                try:
                    await asyncio.to_thread(client.disconnect, True)
                except Exception as exc:
                    logger.warning("Error disconnecting client: %s", exc)

        self._mic_client = None
        self._monitor_client = None
        self._mic_feeder_task = None
        self._monitor_feeder_task = None

        await self.segment_queue.put(None)
        logger.info("Transcriber stopped")

    def _create_client(self, speaker_label: str, StreamingClient, StreamingEvents, options):
        """Create a StreamingClient for one audio stream."""
        client = StreamingClient(options=options)

        def on_turn(c, event) -> None:
            self._on_data_calls[speaker_label] = self._on_data_calls.get(speaker_label, 0) + 1
            text = event.transcript.strip()
            if not text:
                self._empty_data_calls[speaker_label] = self._empty_data_calls.get(speaker_label, 0) + 1
                return

            is_final = event.end_of_turn
            kind = "final" if is_final else "partial"
            self._dbg(f"[{speaker_label}] {kind}: \"{text[:60]}\"", "ok" if is_final else "info")

            # Timestamps from word boundaries (ms → s)
            ts_start = event.words[0].start / 1000.0 if event.words else 0.0
            ts_end = event.words[-1].end / 1000.0 if event.words else 0.0

            segment = TranscriptSegment(
                speaker=speaker_label,
                text=text,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                is_partial=not is_final,
            )

            if self._loop is not None and self._running:
                self._loop.call_soon_threadsafe(self.segment_queue.put_nowait, segment)

        def on_error(c, error) -> None:
            logger.error("AssemblyAI error (%s): %s", speaker_label, error)
            self._dbg(f"[{speaker_label}] AAI error: {error}", "error")
            if self._on_error and self._loop:
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self._on_error(speaker_label, Exception(str(error))),
                )

        def on_begin(c, event) -> None:
            logger.info("AssemblyAI session opened (%s): id=%s", speaker_label, event.id)
            self._dbg(f"[{speaker_label}] session opened (id={event.id})", "ok")

        def on_terminate(c, event) -> None:
            logger.info("AssemblyAI session terminated (%s)", speaker_label)
            self._dbg(f"[{speaker_label}] session terminated", "warn")

        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Termination, on_terminate)

        return client

    async def _feed_loop(
        self,
        pcm_queue: asyncio.Queue[bytes | None],
        client,
        speaker_label: str,
    ) -> None:
        """Read PCM chunks from an asyncio queue and feed them to the SDK.

        asyncio.StreamReader.read(n) may return fewer than n bytes, which can
        produce chunks shorter than AssemblyAI's 50 ms minimum. We accumulate
        bytes in a buffer and only send once we have at least MIN_SEND_BYTES
        (100 ms at 16 kHz s16le = 3200 bytes).
        """
        # 100 ms @ 16 kHz mono s16le = 16000 * 2 * 0.1 = 3200 bytes
        _MIN_SEND_BYTES = 3200
        _LOG_EVERY = 50
        _buf = bytearray()
        try:
            while self._running:
                chunk = await pcm_queue.get()
                if chunk is None:
                    logger.info("Feed loop (%s): received EOF sentinel", speaker_label)
                    self._dbg(f"[{speaker_label}] feed loop: EOF — stream ended", "warn")
                    # Flush any remaining buffered audio
                    if _buf:
                        client.stream(bytes(_buf))
                    break
                _buf.extend(chunk)
                if len(_buf) < _MIN_SEND_BYTES:
                    continue
                to_send = bytes(_buf)
                _buf.clear()
                client.stream(to_send)
                self._chunks_fed[speaker_label] = self._chunks_fed.get(speaker_label, 0) + 1
                n = self._chunks_fed[speaker_label]
                if n == 1:
                    self._dbg(f"[{speaker_label}] first PCM chunk sent to AAI ({len(to_send)} B)", "ok")
                elif n % _LOG_EVERY == 0:
                    empty = self._empty_data_calls.get(speaker_label, 0)
                    hits = self._on_data_calls.get(speaker_label, 0)
                    self._dbg(
                        f"[{speaker_label}] {n} chunks fed | on_data: {hits} ({empty} empty)"
                    )
        except asyncio.CancelledError:
            logger.debug("Feed loop (%s) cancelled", speaker_label)
        except Exception as exc:
            logger.error("Feed loop (%s) error: %s", speaker_label, exc)
            self._dbg(f"[{speaker_label}] feed loop error: {exc}", "error")
