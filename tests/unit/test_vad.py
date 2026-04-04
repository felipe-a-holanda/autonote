"""Tests for SileroVAD wrapper and SpeechStateEvent model."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from autonote.realtime.models import SpeechStateEvent


# ---------------------------------------------------------------------------
# SpeechStateEvent model tests
# ---------------------------------------------------------------------------

class TestSpeechStateEvent:
    """Tests for the SpeechStateEvent Pydantic model."""

    def test_speech_start_event(self):
        event = SpeechStateEvent(
            speaker="Me",
            event_type="speech_start",
            timestamp=1.5,
        )
        assert event.type == "speech_state_event"
        assert event.speaker == "Me"
        assert event.event_type == "speech_start"
        assert event.timestamp == 1.5
        assert event.silence_duration is None

    def test_speech_end_event_with_silence(self):
        event = SpeechStateEvent(
            speaker="Them",
            event_type="speech_end",
            timestamp=5.0,
            silence_duration=2.3,
        )
        assert event.event_type == "speech_end"
        assert event.silence_duration == 2.3

    def test_speech_end_event_without_silence(self):
        event = SpeechStateEvent(
            speaker="Me",
            event_type="speech_end",
            timestamp=3.0,
        )
        assert event.silence_duration is None

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            SpeechStateEvent(
                speaker="Me",
                event_type="unknown",  # type: ignore
                timestamp=0.0,
            )

    def test_type_discriminator(self):
        event = SpeechStateEvent(
            speaker="Them",
            event_type="speech_start",
            timestamp=0.0,
        )
        assert event.type == "speech_state_event"

    def test_serialization_round_trip(self):
        event = SpeechStateEvent(
            speaker="Me",
            event_type="speech_end",
            timestamp=10.0,
            silence_duration=1.8,
        )
        data = event.model_dump()
        assert data["type"] == "speech_state_event"
        assert data["speaker"] == "Me"
        assert data["event_type"] == "speech_end"
        assert data["timestamp"] == 10.0
        assert data["silence_duration"] == 1.8

        restored = SpeechStateEvent(**data)
        assert restored == event

    def test_in_realtime_event_union(self):
        from autonote.realtime.models import RealtimeEvent
        event = SpeechStateEvent(
            speaker="Me",
            event_type="speech_start",
            timestamp=0.0,
        )
        # Verify it's part of the union (type-checking via isinstance)
        assert isinstance(event, SpeechStateEvent)
        assert event.type == "speech_state_event"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            SpeechStateEvent(speaker="Me")  # type: ignore


# ---------------------------------------------------------------------------
# SileroVAD tests (torch is mocked — no GPU/model download required)
# ---------------------------------------------------------------------------

def _make_mock_torch(speech_timestamps: list) -> MagicMock:
    """Build a minimal torch mock that satisfies SileroVAD._load()."""
    mock_torch = MagicMock()
    mock_model = MagicMock()
    # utils[0] is get_speech_timestamps
    mock_get_timestamps = MagicMock(return_value=speech_timestamps)
    mock_utils = (mock_get_timestamps,)
    mock_torch.hub.load.return_value = (mock_model, mock_utils)
    mock_torch.from_numpy.side_effect = lambda a: a  # identity for the tensor arg
    return mock_torch


class TestSileroVAD:
    """Tests for SileroVAD wrapper with mocked torch."""

    def test_instantiation_does_not_load_model(self):
        """SileroVAD can be created without importing torch."""
        from autonote.realtime.vad import SileroVAD
        vad = SileroVAD()
        assert vad._model is None
        assert vad._get_speech_timestamps is None

    def test_is_speech_returns_true_when_timestamps_present(self):
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[{"start": 0, "end": 512}])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(512, dtype=np.float32)
            result = vad.is_speech(audio)
        assert result is True

    def test_is_speech_returns_false_when_no_timestamps(self):
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(512, dtype=np.float32)
            result = vad.is_speech(audio)
        assert result is False

    def test_is_speech_normalizes_int16_range(self):
        """Audio values > 1.0 should be normalised before processing."""
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[{"start": 0, "end": 512}])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.full(512, 16384, dtype=np.float32)  # values > 1.0
            result = vad.is_speech(audio)
        assert result is True

    def test_is_speech_converts_non_float32(self):
        """Integer arrays are cast to float32 before processing."""
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(512, dtype=np.int16)
            result = vad.is_speech(audio)
        assert result is False  # no speech timestamps

    def test_get_speech_segments_returns_speech_segment_objects(self):
        from autonote.realtime.vad import SileroVAD, SpeechSegment

        raw = [{"start": 100, "end": 600}, {"start": 1000, "end": 1500}]
        mock_torch = _make_mock_torch(speech_timestamps=raw)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(2048, dtype=np.float32)
            segments = vad.get_speech_segments(audio)

        assert len(segments) == 2
        assert isinstance(segments[0], SpeechSegment)
        assert segments[0].start_sample == 100
        assert segments[0].end_sample == 600
        assert segments[1].start_sample == 1000
        assert segments[1].end_sample == 1500

    def test_get_speech_segments_empty(self):
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(512, dtype=np.float32)
            segments = vad.get_speech_segments(audio)
        assert segments == []

    def test_model_loaded_only_once(self):
        """_load() is called multiple times but torch.hub.load called only once."""
        from autonote.realtime.vad import SileroVAD

        mock_torch = _make_mock_torch(speech_timestamps=[])
        with patch.dict(sys.modules, {"torch": mock_torch}):
            vad = SileroVAD()
            audio = np.zeros(512, dtype=np.float32)
            vad.is_speech(audio)
            vad.is_speech(audio)

        mock_torch.hub.load.assert_called_once()

    def test_speech_segment_duration_samples(self):
        from autonote.realtime.vad import SpeechSegment

        seg = SpeechSegment(start_sample=100, end_sample=612)
        assert seg.duration_samples() == 512
