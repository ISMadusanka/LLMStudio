"""Logging setup. Uses Rich for pretty console output when available."""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False
_ROOT = "llmstudio"


def setup_logging(level: Optional[str] = None, *, rich_console: bool = True) -> None:
    """Configure the ``llmstudio`` logger hierarchy. Safe to call repeatedly."""
    global _CONFIGURED
    lvl = (level or os.environ.get("LLMSTUDIO_LOG_LEVEL") or "INFO").upper()

    handlers: list[logging.Handler] = []
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    if rich_console:
        try:
            from rich.logging import RichHandler

            handlers = [RichHandler(rich_tracebacks=True, show_path=False, markup=True)]
            fmt = "%(message)s"
            datefmt = "[%X]"
        except Exception:  # pragma: no cover - rich always installed, but be safe
            handlers = []

    logging.basicConfig(
        level=lvl,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers or None,
        force=True,
    )
    logging.getLogger(_ROOT).setLevel(lvl)
    # Quiet down noisy third parties unless explicitly debugging.
    if lvl != "DEBUG":
        for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "datasets"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    full = name if name.startswith(_ROOT) else f"{_ROOT}.{name}"
    return logging.getLogger(full)
