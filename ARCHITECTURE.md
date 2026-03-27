# Autonote Architecture

Autonote is a modular, extensible, native Python meeting copilot. It replaces a legacy Bash orchestration script with a pure Python approach.

## Directory Structure

```text
autonote/
├── pyproject.toml              # Build system, dependencies, and CLI entry point
├── src/
│   └── autonote/
│       ├── __init__.py
│       ├── cli.py              # Central entry point mapping CLI commands to Python functions
│       ├── config.py           # Universal configuration loader (from .hushnoterc and env vars)
│       ├── logger.py           # Centralized terminal output formatting
│       ├── orchestrator.py     # Chains python calls for full workflows (process, process-last, full)
│       ├── audio/              # Meeting processing modules
│       │   ├── compress.py           # Uses ffmpeg to compress WAV to MP3
│       │   ├── record.py             # Uses pactl and ffmpeg to capture system audio
│       │   ├── transcribe.py         # Interfaces with local Faster-Whisper to transcribe
│       │   ├── diarize.py            # Interfaces with pyannote.audio to identify speakers
│       │   ├── merge_diarization.py  # Merges json files from transcription and diarization
│       │   ├── label.py              # CLI loop to identify speakers manually
│       │   ├── apply_labels.py       # Reconstructs speaker transcripts from tags
│       │   ├── reformat.py           # Reformats spoken-word to clean transcripts via LLM
│       │   └── summarize.py          # Summarizes meeting using Ollama
│       └── obsidian/           # Obsidian vault integration modules
│           ├── extract_metadata.py   # Uses LLM to extract JSON traits (topics, attendees, tags)
│           ├── frontmatter.py        # Composes YAML frontmatter
│           ├── wikilink.py           # Auto-links Obsidian tags via local regex matching
│           └── update_index.py       # Pushes meetings to Vault meeting index
```

## Modular Design

Unlike older iterations which chained terminal commands using sub-processes, Autonote defines explicit atomic functions (like `run_transcribe(audio_file) -> result`). The python modules remain highly portable, easily testable in isolation, and are chained programmatically through the `orchestrator.py` module for major commands like `autonote process`.

For example, when `autonote full` is executed, the following flow is typically executed purely in Python:
1. `record.run_record`
2. `transcribe.transcribe_audio`
3. `reformat.run_reformat`
4. `summarize.run_summarize`
5. `orchestrator.run_obsidian_postprocess`
6. `compress.compress_audio`

Any of these modules can be individually executed directly via the CLI, such as `autonote transcribe my_file.mp3`.
