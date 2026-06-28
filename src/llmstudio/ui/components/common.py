"""Small UI helper functions shared across pages."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, Optional

from llmstudio.core.training.job import JobStatus

_STATUS_COLORS = {
    JobStatus.PENDING: "#94a3b8",
    JobStatus.PREPARING: "#0ea5e9",
    JobStatus.DOWNLOADING: "#0ea5e9",
    JobStatus.RUNNING: "#6366f1",
    JobStatus.PAUSING: "#f59e0b",
    JobStatus.PAUSED: "#f59e0b",
    JobStatus.COMPLETED: "#22c55e",
    JobStatus.FAILED: "#ef4444",
    JobStatus.CANCELLED: "#94a3b8",
    JobStatus.RESUMABLE: "#a855f7",
}


def status_badge(status: JobStatus) -> str:
    color = _STATUS_COLORS.get(status, "#94a3b8")
    return (
        f'<span style="display:inline-block;padding:3px 11px;border-radius:999px;'
        f'font-size:.74rem;font-weight:700;letter-spacing:.01em;background:{color};'
        f'color:white;">{status.value}</span>'
    )


def stream_task(fn: Callable[[Callable[[str], None]], Any]) -> Iterator[str]:
    """Run ``fn(progress)`` in a background thread, yielding accumulated log text.

    ``fn`` receives a ``progress(msg)`` callback; every message is appended and the
    full text re-yielded so a Gradio Textbox shows a live, growing log.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def progress(msg: str) -> None:
        q.put(("log", msg))

    def runner() -> None:
        try:
            result = fn(progress)
            q.put(("result", result))
        except Exception as exc:  # surface to the UI
            q.put(("error", str(exc)))
        finally:
            q.put(sentinel)

    threading.Thread(target=runner, daemon=True).start()

    lines: list[str] = []
    while True:
        item = q.get()
        if item is sentinel:
            break
        kind, payload = item
        if kind == "log":
            lines.append(str(payload))
        elif kind == "error":
            lines.append(f"❌ {payload}")
        elif kind == "result" and payload:
            lines.append(f"✅ Done.")
        yield "\n".join(lines)
    if lines:
        yield "\n".join(lines)


def metrics_dataframe(metrics: list[dict[str, Any]], y: str = "loss"):
    """Build a pandas DataFrame of (step, <y>) for plotting; safe if empty."""
    import pandas as pd

    rows = [{"step": m.get("step", 0), y: m[y]} for m in metrics if y in m]
    if not rows:
        return pd.DataFrame({"step": [], y: []})
    return pd.DataFrame(rows)
