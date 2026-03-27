# Autonote Developer Notes for Claude

You are interacting with the `autonote` repository, a modular meeting-recording and transcription pipeline written in Python. 

Here are some instructions and system guidelines to consider while working inside this application.

## High Level Philosophy
- **Modular Native Python**: This project represents a migration away from bash wrappers. Do *not* write functionality as Shell calls (`subprocess.run("hushnote script.sh")`). If an orchestration is needed, it should be orchestrated internally via `autonote/src/autonote/orchestrator.py` or `autonote/src/autonote/cli.py`.
- **System Boundaries**: System tool dependencies (like `ffmpeg` or `pactl`) must be called via `subprocess` from within a modular Python file (see `compress.py` and `record.py`), but Python scripts must *not* execute other Python scripts via `subprocess`.
- **Separation of Concerns**: When adding a new feature (e.g., an LLM embedding tool), isolate the tool as a single pure Python module with clear inputs and outputs, and a `run_XYZ` function exposed to the CLI router.
- **Robust Decorator Logging**: Rely on `autonote.logger` methods (`log_info`, `log_error`, `log_success`) to output console statuses. Avoid raw `print` statements in newly created feature modules.
- **Fail Gracefully**: Tools like `diarization` or `Ollama` requests might timeout or lack API keys. Throw a python generic error or return cleanly, allowing downstream routines to skip unavailable outputs.

## Code Constraints
- Use Python 3.10+ type hints.
- Parse `autonote.config` values to set module defaults.

Always strive to keep `autonote` minimal, importable, extensible, and free of system-level glue logic!
