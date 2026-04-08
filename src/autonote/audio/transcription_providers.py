#!/usr/bin/env python3

"""
Generic transcription provider interface and implementations
Supports both local (faster-whisper) and external API providers (AssemblyAI, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from autonote.logger import log_info, log_error


class TranscriptionProvider(ABC):
    """Base class for all transcription providers"""
    
    @abstractmethod
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> dict:
        """
        Transcribe an audio file
        
        Args:
            audio_file: Path to audio file
            language: Optional language code
            
        Returns:
            dict with keys:
                - language: detected or specified language
                - segments: list of dicts with start, end, text
                - text: full transcription text
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider"""
        pass


class LocalWhisperProvider(TranscriptionProvider):
    """Local faster-whisper transcription provider"""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None
        
    def _setup_cuda_lib_paths(self):
        """Setup CUDA library paths for CTranslate2"""
        import os
        site_packages = Path(__file__).resolve().parent.parent.parent.parent
        nvidia_dir = site_packages / "nvidia"
        if nvidia_dir.is_dir():
            lib_dirs = [str(p) for p in nvidia_dir.glob("*/lib") if p.is_dir()]
            if lib_dirs:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    
    def _get_model(self):
        """Lazy load the Whisper model"""
        if self._model is not None:
            return self._model
            
        self._setup_cuda_lib_paths()
        
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required for local transcription. "
                "Install it with: uv add faster-whisper"
            ) from e
        
        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        log_info(f"Loading Whisper model: {self.model_size} on {device} ({compute_type})")
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        return self._model
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> dict:
        """Transcribe using local faster-whisper model"""
        model = self._get_model()
        
        log_info(f"Transcribing with local Whisper: {audio_file}")
        segments_gen, info = model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
        )
        
        log_info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        all_segments = []
        full_text_parts = []
        for segment in segments_gen:
            text = segment.text.strip()
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })
            full_text_parts.append(text)
        
        return {
            "language": info.language,
            "segments": all_segments,
            "text": " ".join(full_text_parts),
        }
    
    def get_provider_name(self) -> str:
        return f"local-whisper-{self.model_size}"


class AssemblyAIProvider(TranscriptionProvider):
    """AssemblyAI external API transcription provider"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AssemblyAI API key is required")
        self.api_key = api_key
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> dict:
        """Transcribe using AssemblyAI API"""
        try:
            import assemblyai as aai
        except ImportError as e:
            raise ImportError(
                "assemblyai package is required for AssemblyAI transcription. "
                "Install it with: uv add assemblyai"
            ) from e
        
        aai.settings.api_key = self.api_key
        
        log_info(f"Uploading to AssemblyAI: {audio_file}")
        transcriber = aai.Transcriber()
        
        config = aai.TranscriptionConfig(
            language_code=language if language else None,
            speech_models=["universal-3-pro"],
            speaker_labels=True,
        )
        
        log_info("Starting AssemblyAI transcription...")
        transcript = transcriber.transcribe(audio_file, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        detected_language = transcript.language_code or "unknown"
        audio_duration = transcript.audio_duration or 0.0
        log_info(f"AssemblyAI transcription complete. Language: {detected_language}, Duration: {audio_duration:.1f}s")

        try:
            from autonote.llm import _append_cost_log
            from autonote.config import config
            cost_per_min = float(config.get("ASSEMBLYAI_COST_PER_MINUTE"))
            cost_usd = (audio_duration / 60.0) * cost_per_min
            usd_to_brl = float(config.get("USD_TO_BRL"))
            _append_cost_log(
                model="assemblyai/universal-3-pro",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=cost_usd,
                cost_brl=cost_usd * usd_to_brl,
                source_file=audio_file,
                stage="transcription",
                duration_s=audio_duration,
            )
        except Exception as e:
            log_error(f"Failed to log AssemblyAI cost: {e}")
        
        segments = []

        # Use utterances (speaker-diarized) when available
        if transcript.utterances:
            for utt in transcript.utterances:
                segments.append({
                    "start": utt.start / 1000.0,
                    "end": utt.end / 1000.0,
                    "text": utt.text,
                    "speaker": utt.speaker,
                })
        elif transcript.words:
            for word in transcript.words:
                segments.append({
                    "start": word.start / 1000.0,
                    "end": word.end / 1000.0,
                    "text": word.text,
                })

        if not segments and transcript.text:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": transcript.text,
            })

        return {
            "language": detected_language,
            "segments": segments,
            "text": transcript.text or "",
        }
    
    def get_provider_name(self) -> str:
        return "assemblyai"


def create_transcription_provider(
    provider: str = "local",
    model_size: str = "base",
    device: str = "auto",
    api_key: Optional[str] = None
) -> TranscriptionProvider:
    """
    Factory function to create transcription providers
    
    Args:
        provider: Provider name ("local", "assemblyai")
        model_size: For local provider, the Whisper model size
        device: For local provider, the device to use
        api_key: For external providers, the API key
        
    Returns:
        TranscriptionProvider instance
    """
    provider = provider.lower()
    
    if provider == "local":
        return LocalWhisperProvider(model_size=model_size, device=device)
    elif provider == "assemblyai":
        if not api_key:
            raise ValueError("AssemblyAI requires an API key. Set ASSEMBLYAI_API_KEY environment variable.")
        return AssemblyAIProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown transcription provider: {provider}. Supported: local, assemblyai")
