import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_recording_dir(temp_dir):
    """Create a sample recording directory structure."""
    recording_dir = temp_dir / "recordings" / "20260330" / "meeting_20260330_120000"
    recording_dir.mkdir(parents=True)
    
    (recording_dir / "meeting_20260330_120000.txt").write_text("Sample transcript")
    (recording_dir / "meeting_20260330_120000_formatted.md").write_text("# Formatted transcript")
    
    return recording_dir


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "RECORDINGS_DIR": "/tmp/recordings",
        "TRANSCRIPTION_PROVIDER": "local",
        "WHISPER_MODEL": "turbo",
        "MODEL": "ollama/llama3.1:8b",
        "OLLAMA_URL": "http://localhost:11434",
        "DEBUG": "false",
        "USD_TO_BRL": "5.50",
        "LLM_COST_LOG": None,
        "VAULT_DIR": "",
        "VAULT_SUBDIR": "meetings",
    }
