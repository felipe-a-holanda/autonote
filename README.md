# Autonote

Autonote is a modular, extensible, native Python meeting copilot. It replaces a legacy Bash orchestration script with a pure Python approach.

## Installation

```bash
cd autonote
pip install -e .
```

You'll need a `.hushnoterc` file in either your home directory (`~/.hushnoterc`) or locally in the `autonote` code folder. 

### Configuration (`.hushnoterc`)
```env
RECORDINGS_DIR="./recordings"
WHISPER_MODEL="turbo"
WHISPER_LANGUAGE="en"
OLLAMA_MODEL="llama3.1:8b"
OLLAMA_REFORMAT_MODEL="llama3.1:8b"
OLLAMA_URL="http://localhost:11434"
VAULT_DIR="/path/to/my/obisdian/vault"
MEETING_INDEX="/path/to/my/obisdian/vault/Meetings.md"
ENTITIES_FILE="./entities.yml"
```

### Dependencies
Ensure you have the following system dependencies installed:
- `ffmpeg`
- `pactl` (pulseaudio-utils or pipewire-pulse)
- Python 3.10+


## Usage

Autonote offers granular components and macro-orchestration pipelines. You can utilize the underlying python modules atomically from the terminal, or simply run the `autonote process` and `autonote full` commands.

### Atomic Actions
```bash
# Record for generic use
autonote record

# Manually transcribe
autonote transcribe recordings/my_audio.mp3

# Apply LLM cleanups
autonote reformat recordings/my_audio.txt

# Create an LLM meeting summary
autonote summarize recordings/my_audio_formatted.md
```

### Pipelines / Full Execution
If you just want the meeting pipeline executed without fuss, Autonote orchestrator wraps all atomic pieces.

```bash
# Records a meeting, then triggers transcription, AI formatting, summarization, and Obsidian integration
autonote full

# Reruns process pipeline on a specific audio file
autonote process recordings/meeting_20251005_143022.wav

# Auto-locates the latest .wav/.mp3 in recordings folder and processes it
autonote process-last
```

## Why Autonote?
Older iterations of the "hushnote" architecture depended heavily on legacy bash scripts which limited testability, maintainability, and cross-platform flexibility. Autonote abstracts Bash sub-processing entirely in favor of an easily extensible Python configuration approach.
