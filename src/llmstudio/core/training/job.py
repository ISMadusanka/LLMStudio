"""Training jobs: persisted state machine + runtime pause/stop control.

A :class:`JobRow` is the durable record (survives restarts, enabling crash
recovery). :class:`JobControl` is the in-memory, thread-safe signal the UI uses
to ask a running job to pause or stop; the training callback polls it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Boolean, Float, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from llmstudio.core.utils.db import Base, Database, get_database
from llmstudio.core.utils.logging import get_logger

log = get_logger("training.job")


class JobStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    DOWNLOADING = "downloading"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMABLE = "resumable"  # was running when the process died


ACTIVE_STATUSES = {
    JobStatus.PREPARING,
    JobStatus.DOWNLOADING,
    JobStatus.RUNNING,
    JobStatus.PAUSING,
}
TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING.value)
    base_model_key: Mapped[str] = mapped_column(String(120), default="")
    dataset_id: Mapped[str] = mapped_column(String(80), default="")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    run_dir: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[float] = mapped_column(Float, default=0.0)
    finished_at: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    current_epoch: Mapped[float] = mapped_column(Float, default=0.0)
    last_checkpoint: Mapped[str] = mapped_column(Text, default="")
    best_metric: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    resumable: Mapped[bool] = mapped_column(Boolean, default=False)
    registered_model_id: Mapped[str] = mapped_column(String(80), default="")


@dataclass
class Job:
    """Detached, UI-friendly snapshot of a job."""

    id: str
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    base_model_key: str = ""
    dataset_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0.0
    last_checkpoint: str = ""
    best_metric: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    resumable: bool = False
    registered_model_id: str = ""

    @classmethod
    def from_row(cls, row: JobRow) -> "Job":
        return cls(
            id=row.id,
            name=row.name,
            status=JobStatus(row.status),
            base_model_key=row.base_model_key,
            dataset_id=row.dataset_id,
            config=_loads(row.config_json),
            run_dir=row.run_dir,
            created_at=row.created_at,
            updated_at=row.updated_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            current_step=row.current_step,
            total_steps=row.total_steps,
            current_epoch=row.current_epoch,
            last_checkpoint=row.last_checkpoint,
            best_metric=row.best_metric,
            metrics=_loads(row.metrics_json),
            error=row.error,
            resumable=row.resumable,
            registered_model_id=row.registered_model_id,
        )

    @property
    def progress(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.current_step / self.total_steps)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def can_pause(self) -> bool:
        return self.status == JobStatus.RUNNING

    @property
    def can_resume(self) -> bool:
        return self.status in {JobStatus.PAUSED, JobStatus.RESUMABLE} and bool(self.last_checkpoint)

    def status_label(self) -> str:
        return self.status.value.capitalize()


def _loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}


def make_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


class JobStore:
    """Durable CRUD for jobs."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_database()

    def create(self, *, config: dict[str, Any], name: str, run_dir: str,
               base_model_key: str, dataset_id: str, job_id: Optional[str] = None) -> Job:
        now = time.time()
        job_id = job_id or make_job_id()
        with self.db.session() as sess:
            row = JobRow(
                id=job_id,
                name=name,
                status=JobStatus.PENDING.value,
                base_model_key=base_model_key,
                dataset_id=dataset_id,
                config_json=json.dumps(config),
                run_dir=run_dir,
                created_at=now,
                updated_at=now,
            )
            sess.add(row)
            sess.flush()
            return Job.from_row(row)

    def get(self, job_id: str) -> Optional[Job]:
        with self.db.session() as sess:
            row = sess.get(JobRow, job_id)
            return Job.from_row(row) if row else None

    def list(self) -> list[Job]:
        with self.db.session() as sess:
            rows = sess.execute(select(JobRow).order_by(JobRow.created_at.desc())).scalars().all()
            return [Job.from_row(r) for r in rows]

    def update(self, job_id: str, **fields: Any) -> Optional[Job]:
        with self.db.session() as sess:
            row = sess.get(JobRow, job_id)
            if not row:
                return None
            if "status" in fields and isinstance(fields["status"], JobStatus):
                fields["status"] = fields["status"].value
            if "metrics" in fields:
                fields["metrics_json"] = json.dumps(fields.pop("metrics"))
            for key, value in fields.items():
                if hasattr(row, key):
                    setattr(row, key, value)
            row.updated_at = time.time()
            sess.flush()
            return Job.from_row(row)

    def set_status(self, job_id: str, status: JobStatus, *, error: str = "") -> None:
        extra: dict[str, Any] = {"status": status}
        if status == JobStatus.RUNNING:
            extra["started_at"] = time.time()
        if status in TERMINAL_STATUSES:
            extra["finished_at"] = time.time()
        if error:
            extra["error"] = error
        self.update(job_id, **extra)

    def find_resumable(self) -> list[Job]:
        return [j for j in self.list() if j.can_resume]

    def delete(self, job_id: str) -> bool:
        with self.db.session() as sess:
            row = sess.get(JobRow, job_id)
            if not row:
                return False
            sess.delete(row)
            return True


# ---------------------------------------------------------------------------
# Runtime control (in-memory, thread-safe)
# ---------------------------------------------------------------------------
@dataclass
class JobControl:
    """Cooperative pause/stop signalling between the UI and a running trainer."""

    _pause: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)

    def request_pause(self) -> None:
        self._pause.set()

    def request_stop(self) -> None:
        self._stop.set()

    def pause_requested(self) -> bool:
        return self._pause.is_set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def should_halt(self) -> bool:
        """True if either pause or stop has been requested."""
        return self._pause.is_set() or self._stop.is_set()

    def clear(self) -> None:
        self._pause.clear()
        self._stop.clear()
