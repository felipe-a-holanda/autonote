import json
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone
from rich.console import Console
from rich.theme import Theme
from autonote.config import config

custom_theme = Theme({
    "info": "blue",
    "success": "bold green",
    "error": "bold red",
    "warn": "bold yellow",
    "debug": "dim"
})

console = Console(stderr=True, theme=custom_theme)

_quiet = False

def set_quiet(value: bool) -> None:
    global _quiet
    _quiet = value

def log_info(msg: str):
    if not _quiet:
        console.print(f"[info][INFO][/info] {msg}")

def log_success(msg: str):
    if not _quiet:
        console.print(f"[success][SUCCESS][/success] {msg}")

def log_error(msg: str):
    console.print(f"[error][ERROR][/error] {msg}")

def log_warn(msg: str):
    console.print(f"[warn][WARN][/warn] {msg}")

def log_debug(msg: str):
    if config.get("DEBUG").lower() == "true":
        console.print(f"[debug][DEBUG] {msg}[/debug]")


class _StructuredJsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    If the record carries a ``structured`` extra dict, its keys are merged
    into the envelope.  Otherwise a minimal ``{ts, level, logger, message}``
    object is written.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            entry.update(structured)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_json_logging(prefix: str = "autonote") -> pathlib.Path:
    """Attach a JSONL FileHandler to the ``autonote`` logger namespace.

    Safe to call multiple times — adds the handler only once per path.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(f"{prefix}_{ts}.jsonl")
    autonote_logger = logging.getLogger("autonote")
    for h in autonote_logger.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve()):
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_StructuredJsonFormatter())
    handler.addFilter(logging.Filter("autonote"))
    autonote_logger.addHandler(handler)
    return path


def configure_file_logging(prefix: str = "autonote") -> pathlib.Path:
    """Attach a FileHandler to the root logger and return the log file path.

    Safe to call multiple times — adds the handler only once per path.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(f"{prefix}_{ts}.log")
    # Keep root at WARNING so noisy third-party libs (websockets, httpcore, LiteLLM)
    # don't flood the log with binary frame dumps.  Only autonote-namespaced loggers
    # need DEBUG+ so pipeline INFO messages reach the file handler.
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("autonote").setLevel(logging.DEBUG)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve()):
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)  # logger-level filtering handles third-party noise
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    return path
