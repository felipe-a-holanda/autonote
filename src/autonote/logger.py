import logging
import pathlib
import sys
import time
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


def configure_file_logging(prefix: str = "autonote") -> pathlib.Path:
    """Attach a FileHandler to the root logger and return the log file path.

    Safe to call multiple times — adds the handler only once per path.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = pathlib.Path(f"{prefix}_{ts}.log")
    # Ensure autonote-namespaced loggers emit DEBUG+ so INFO pipeline messages
    # reach the file handler (root logger defaults to WARNING which would drop them).
    logging.getLogger("autonote").setLevel(logging.DEBUG)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve()):
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    return path
