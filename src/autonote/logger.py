import sys
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

def log_info(msg: str):
    console.print(f"[info][INFO][/info] {msg}")

def log_success(msg: str):
    console.print(f"[success][SUCCESS][/success] {msg}")

def log_error(msg: str):
    console.print(f"[error][ERROR][/error] {msg}")

def log_warn(msg: str):
    console.print(f"[warn][WARN][/warn] {msg}")

def log_debug(msg: str):
    if config.get("DEBUG", "false").lower() == "true":
        console.print(f"[debug][DEBUG] {msg}[/debug]")
