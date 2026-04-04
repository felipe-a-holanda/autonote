"""VADMonitor: async worker that processes raw PCM audio and emits speech state events.

Reads raw s16le 16 kHz mono PCM bytes from an asyncio.Queue, feeds them through
SileroVAD in 512-sample windows, tracks speech/silence transitions, and emits
`SpeechStateEvent` objects to an output queue on state changes.
"""

from __future__ import annotations

import asyncio

import numpy as np

from autonote.logger import log_info, log_error
from autonote.realtime.models import SpeechStateEvent
from autonote.realtime.vad import SileroVAD

# Silero VAD requires multiples of 512 samples at 16 kHz
_WINDOW_SAMPLES = 512
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # s16le


class VADMonitor:
    """Async worker: reads PCM chunks → emits SpeechStateEvents.

    State machine:
      silence → (speech detected) → speech_start event, enter speech state
      speech  → (silence detected) → start silence timer
      speech  → (silence ≥ threshold) → speech_end event, enter silence state
      speech  → (speech resumes during silence timer) → reset silence timer
    """

    def __init__(
        self,
        speaker: str,
        vad: SileroVAD,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        silence_threshold: float = 1.5,
        sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        self.speaker = speaker
        self._vad = vad
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._silence_threshold = silence_threshold
        self._sample_rate = sample_rate

        # Internal state
        self._in_speech: bool = False
        self._sample_count: int = 0
        self._silence_start_sample: int | None = None
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

    async def run(self) -> None:
        """Process audio chunks until None sentinel is received."""
        log_info(f"VADMonitor[{self.speaker}] started.")
        while True:
            chunk = await self._input_queue.get()
            if chunk is None:
                log_info(f"VADMonitor[{self.speaker}] received sentinel, stopping.")
                break
            self._process_chunk(chunk)

    def _process_chunk(self, raw: bytes) -> None:
        """Convert raw s16le bytes to float32 and process in 512-sample windows."""
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, audio])

        while len(self._buffer) >= _WINDOW_SAMPLES:
            window = self._buffer[:_WINDOW_SAMPLES]
            self._buffer = self._buffer[_WINDOW_SAMPLES:]
            self._process_window(window)

    def _process_window(self, audio: np.ndarray) -> None:
        """Run VAD on a 512-sample window and update state machine."""
        window_start_sample = self._sample_count
        window_start_time = window_start_sample / self._sample_rate
        self._sample_count += _WINDOW_SAMPLES

        is_speech = self._vad.is_speech(audio, self._sample_rate)

        if is_speech:
            if not self._in_speech:
                # silence → speech transition
                self._in_speech = True
                self._silence_start_sample = None
                event = SpeechStateEvent(
                    speaker=self.speaker,
                    event_type="speech_start",
                    timestamp=window_start_time,
                )
                self._output_queue.put_nowait(event)
                log_info(
                    f"VADMonitor[{self.speaker}] speech_start @ {window_start_time:.2f}s"
                )
            else:
                # continuing speech — reset any pending silence timer
                self._silence_start_sample = None
        else:
            # not speech
            if self._in_speech:
                if self._silence_start_sample is None:
                    # first silent window after speech
                    self._silence_start_sample = window_start_sample

                silence_samples = self._sample_count - self._silence_start_sample
                silence_duration = silence_samples / self._sample_rate

                if silence_duration >= self._silence_threshold:
                    # silence threshold exceeded → speech end
                    self._in_speech = False
                    end_time = self._sample_count / self._sample_rate
                    event = SpeechStateEvent(
                        speaker=self.speaker,
                        event_type="speech_end",
                        timestamp=end_time,
                        silence_duration=silence_duration,
                    )
                    self._output_queue.put_nowait(event)
                    log_info(
                        f"VADMonitor[{self.speaker}] speech_end @ {end_time:.2f}s "
                        f"(silence {silence_duration:.2f}s)"
                    )
                    self._silence_start_sample = None
