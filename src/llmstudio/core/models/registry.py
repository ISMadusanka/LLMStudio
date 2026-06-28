"""Registry of fine-tuned models.

Every finished model is recorded here with its base model, training config,
final metrics, dataset lineage, and on-disk location, so users can browse, load,
chat with, and export their models later. Backed by the shared SQLite database.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Float, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from llmstudio.core.utils.db import Base, Database, get_database
from llmstudio.core.utils.logging import get_logger

log = get_logger("models.registry")

# Export / artifact kinds
KIND_LORA = "lora"
KIND_MERGED_16BIT = "merged_16bit"
KIND_MERGED_4BIT = "merged_4bit"
KIND_GGUF = "gguf"

STATUS_READY = "ready"
STATUS_EXPORTING = "exporting"
STATUS_FAILED = "failed"


class RegisteredModelRow(Base):
    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    base_model_key: Mapped[str] = mapped_column(String(120), default="")
    base_repo: Mapped[str] = mapped_column(String(200), default="")
    artifact_kind: Mapped[str] = mapped_column(String(40), default=KIND_LORA)
    quantization: Mapped[str] = mapped_column(String(20), default="qlora")
    path: Mapped[str] = mapped_column(Text, default="")
    dataset_id: Mapped[str] = mapped_column(String(80), default="")
    job_id: Mapped[str] = mapped_column(String(80), default="")
    status: Mapped[str] = mapped_column(String(20), default=STATUS_READY)
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    notes: Mapped[str] = mapped_column(Text, default="")


@dataclass
class ModelRecord:
    """Detached, UI-friendly view of a registry row."""

    id: str
    name: str
    base_model_key: str = ""
    base_repo: str = ""
    artifact_kind: str = KIND_LORA
    quantization: str = "qlora"
    path: str = ""
    dataset_id: str = ""
    job_id: str = ""
    status: str = STATUS_READY
    created_at: float = 0.0
    updated_at: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_row(cls, row: RegisteredModelRow) -> "ModelRecord":
        return cls(
            id=row.id,
            name=row.name,
            base_model_key=row.base_model_key,
            base_repo=row.base_repo,
            artifact_kind=row.artifact_kind,
            quantization=row.quantization,
            path=row.path,
            dataset_id=row.dataset_id,
            job_id=row.job_id,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metrics=_loads(row.metrics_json),
            config=_loads(row.config_json),
            notes=row.notes,
        )

    @property
    def created_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at)) if self.created_at else ""

    def exists_on_disk(self) -> bool:
        return bool(self.path) and Path(self.path).exists()


def _loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text) if text else {}
    except Exception:
        return {}


def make_model_id(name: str) -> str:
    from llmstudio.core.data.preparer import slugify

    return f"{slugify(name)[:40]}-{uuid.uuid4().hex[:8]}"


class ModelRegistry:
    """CRUD over fine-tuned model records."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or get_database()

    def register(
        self,
        *,
        name: str,
        path: str,
        base_model_key: str = "",
        base_repo: str = "",
        artifact_kind: str = KIND_LORA,
        quantization: str = "qlora",
        dataset_id: str = "",
        job_id: str = "",
        metrics: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        notes: str = "",
        model_id: Optional[str] = None,
        status: str = STATUS_READY,
    ) -> ModelRecord:
        now = time.time()
        model_id = model_id or make_model_id(name)
        with self.db.session() as sess:
            row = sess.get(RegisteredModelRow, model_id)
            if row is None:
                row = RegisteredModelRow(id=model_id, created_at=now)
                sess.add(row)
            row.name = name
            row.path = str(path)
            row.base_model_key = base_model_key
            row.base_repo = base_repo
            row.artifact_kind = artifact_kind
            row.quantization = quantization
            row.dataset_id = dataset_id
            row.job_id = job_id
            row.status = status
            row.updated_at = now
            row.metrics_json = json.dumps(metrics or {})
            row.config_json = json.dumps(config or {})
            row.notes = notes
            sess.flush()
            record = ModelRecord.from_row(row)
        log.info("Registered model '%s' (%s) at %s", name, model_id, path)
        return record

    def list(self) -> list[ModelRecord]:
        with self.db.session() as sess:
            rows = sess.execute(
                select(RegisteredModelRow).order_by(RegisteredModelRow.created_at.desc())
            ).scalars().all()
            return [ModelRecord.from_row(r) for r in rows]

    def get(self, model_id: str) -> Optional[ModelRecord]:
        with self.db.session() as sess:
            row = sess.get(RegisteredModelRow, model_id)
            return ModelRecord.from_row(row) if row else None

    def update_metrics(self, model_id: str, metrics: dict[str, Any]) -> None:
        with self.db.session() as sess:
            row = sess.get(RegisteredModelRow, model_id)
            if row:
                row.metrics_json = json.dumps(metrics)
                row.updated_at = time.time()

    def set_status(self, model_id: str, status: str) -> None:
        with self.db.session() as sess:
            row = sess.get(RegisteredModelRow, model_id)
            if row:
                row.status = status
                row.updated_at = time.time()

    def delete(self, model_id: str, *, remove_files: bool = False) -> bool:
        with self.db.session() as sess:
            row = sess.get(RegisteredModelRow, model_id)
            if not row:
                return False
            path = row.path
            sess.delete(row)
        if remove_files and path and Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)
            log.info("Removed model files at %s", path)
        return True
