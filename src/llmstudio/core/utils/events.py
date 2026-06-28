"""A tiny thread-safe pub/sub event bus.

Training runs in a background thread and emits log lines + metric points; the
Gradio UI consumes them to drive live charts and a streaming log view. Each job
gets a bounded history buffer (so a page refresh can replay recent events) plus
live subscriber queues.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Iterator, Optional

# Event kinds
LOG = "log"
METRIC = "metric"
STATUS = "status"
ERROR = "error"


@dataclass
class Event:
    job_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    """In-process event fan-out keyed by ``job_id``."""

    def __init__(self, history_size: int = 5000) -> None:
        self._history_size = history_size
        self._lock = threading.RLock()
        self._history: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=history_size))
        self._subs: dict[str, list[Queue]] = defaultdict(list)

    # -- producer side ------------------------------------------------------
    def publish(self, event: Event) -> None:
        with self._lock:
            self._history[event.job_id].append(event)
            subscribers = list(self._subs.get(event.job_id, ()))
        for q in subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def log(self, job_id: str, message: str, level: str = "info") -> None:
        self.publish(Event(job_id, LOG, {"message": message, "level": level}))

    def metric(self, job_id: str, step: int, values: dict[str, Any]) -> None:
        self.publish(Event(job_id, METRIC, {"step": step, **values}))

    def status(self, job_id: str, status: str, detail: Optional[str] = None) -> None:
        self.publish(Event(job_id, STATUS, {"status": status, "detail": detail}))

    def error(self, job_id: str, message: str) -> None:
        self.publish(Event(job_id, ERROR, {"message": message}))

    # -- consumer side ------------------------------------------------------
    def history(self, job_id: str, kinds: Optional[set[str]] = None) -> list[Event]:
        with self._lock:
            events = list(self._history.get(job_id, ()))
        if kinds:
            events = [e for e in events if e.kind in kinds]
        return events

    def metrics_frame(self, job_id: str) -> list[dict[str, Any]]:
        """Return metric events as plain dicts (handy for plotting)."""
        return [e.data for e in self.history(job_id, kinds={METRIC})]

    def subscribe(self, job_id: str) -> "Subscription":
        q: Queue = Queue()
        with self._lock:
            self._subs[job_id].append(q)
        return Subscription(self, job_id, q)

    def _unsubscribe(self, job_id: str, q: Queue) -> None:
        with self._lock:
            if q in self._subs.get(job_id, []):
                self._subs[job_id].remove(q)

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._history.pop(job_id, None)


class Subscription:
    """Context-managed live stream of events for one job."""

    def __init__(self, bus: EventBus, job_id: str, q: Queue) -> None:
        self._bus = bus
        self._job_id = job_id
        self._q = q

    def stream(self, timeout: float = 1.0) -> Iterator[Event]:
        """Yield events as they arrive. Returns (stops) on a poison ``None``."""
        while True:
            try:
                event = self._q.get(timeout=timeout)
            except Empty:
                continue
            if event is None:  # poison pill
                return
            yield event

    def close(self) -> None:
        self._bus._unsubscribe(self._job_id, self._q)
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Process-wide default bus shared by the training manager and the UI.
_DEFAULT_BUS: Optional[EventBus] = None
_BUS_LOCK = threading.Lock()


def default_bus() -> EventBus:
    global _DEFAULT_BUS
    if _DEFAULT_BUS is None:
        with _BUS_LOCK:
            if _DEFAULT_BUS is None:
                _DEFAULT_BUS = EventBus()
    return _DEFAULT_BUS
