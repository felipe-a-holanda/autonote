# Autonote

Autonote is a modular, extensible, native Python meeting copilot. It replaces a legacy Bash orchestration script with a pure Python approach.

## Installation

```bash
cd autonote
pip install -e .
```

You'll need a `.autonoterc` file in either your home directory (`~/.autonoterc`) or locally in the `autonote` code folder. We have heavily documented an example in `.autonoterc.example`.

### Configuration (`.autonoterc`)
```env
RECORDINGS_DIR="./recordings"

# Transcription settings
TRANSCRIPTION_PROVIDER="local"  # Options: "local" (default), "assemblyai"
WHISPER_MODEL="turbo"  # For local provider only
WHISPER_LANGUAGE="en"
ASSEMBLYAI_API_KEY=""  # Required for AssemblyAI provider

# LLM settings
OLLAMA_MODEL="llama3.1:8b"
OLLAMA_REFORMAT_MODEL="llama3.1:8b"
OLLAMA_URL="http://localhost:11434"

# Obsidian integration
VAULT_DIR="/path/to/my/obisdian/vault"
MEETING_INDEX="/path/to/my/obisdian/vault/Meetings.md"
ENTITIES_FILE="./entities.yml"
```

### Dependencies
Ensure you have the following system dependencies installed:
- `ffmpeg`
- `pactl` (pulseaudio-utils or pipewire-pulse)
- Python 3.10+

### External Transcription APIs

By default, Autonote uses the local `faster-whisper` model for transcription. However, you can configure it to use external transcription APIs for potentially better accuracy, speed, or language support.

#### Supported Providers
- **`local`** (default): Uses faster-whisper running on your machine
- **`assemblyai`**: Uses AssemblyAI's cloud transcription API

#### Configuration
Set the provider globally in your `.autonoterc`:
```env
TRANSCRIPTION_PROVIDER="assemblyai"
ASSEMBLYAI_API_KEY="your-api-key-here"
```

Or override per-command using CLI flags:
```bash
autonote transcribe audio.mp3 --provider assemblyai --api-key YOUR_KEY
```

#### Getting an AssemblyAI API Key
1. Sign up at https://www.assemblyai.com/
2. Copy your API key from the dashboard
3. Add it to your `.autonoterc` file

#### Installing Dependencies
For AssemblyAI support, install with the external transcription extras:
```bash
pip install -e ".[transcribe-external]"
# or with uv
uv sync --extra transcribe-external
```

## Usage

Autonote offers granular components and macro-orchestration pipelines. You can utilize the underlying python modules atomically from the terminal, or simply run the `autonote process` and `autonote full` commands.

### Atomic Actions
```bash
# Record for generic use
autonote record

# Manually transcribe (uses default provider from config)
autonote transcribe recordings/my_audio.mp3

# Use specific transcription provider
autonote transcribe recordings/my_audio.mp3 --provider assemblyai --api-key YOUR_API_KEY

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

### Obsidian Integration

After processing, autonote copies files to your Obsidian vault with human-readable names derived from the meeting content:

```
VAULT_DIR/
  2026-03-28 Sprint Planning - work/
    2026-03-28 Sprint Planning - work.md        ← summary
    2026-03-28 Sprint Planning - work - transcript.md
```

The title is inferred from the first meaningful heading in the LLM-generated summary. If you provided a name at recording time (`autonote record -t "work"`), it is appended as a tag. The date is always prepended.

To run only the Obsidian post-processing step on an existing meeting:

```bash
autonote obsidian recordings/20260328/meeting_20260328_081457/meeting_20260328_081457.wav
```

### Reprocessing Existing Meetings

Use `reprocess` to selectively re-run pipeline steps on meetings that have already been recorded and transcribed. Each flag is independent:

```bash
# Re-run summarization only, then update vault
autonote reprocess recordings/.../meeting_TIMESTAMP.wav --summarize --obsidian

# Re-run both reformat and summarization, then update vault
autonote reprocess recordings/.../meeting_TIMESTAMP.wav --reformat --summarize --obsidian

# Use a specific model
autonote reprocess recordings/.../meeting_TIMESTAMP.wav --reformat --summarize --obsidian -m smart
```

**Batch reprocessing** — target all meetings on or after a date, or your entire recordings folder:

```bash
# All meetings since March 1st
autonote reprocess --since 2026-03-01 --summarize --obsidian

# All meetings ever
autonote reprocess --all --reformat --summarize --obsidian
```

Errors on individual meetings (e.g. missing transcript) are logged and skipped — the batch continues.

## Why Autonote?
Older iterations of the "hushnote" architecture depended heavily on legacy bash scripts which limited testability, maintainability, and cross-platform flexibility. Autonote abstracts Bash sub-processing entirely in favor of an easily extensible Python configuration approach.
