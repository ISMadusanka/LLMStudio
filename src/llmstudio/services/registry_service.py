"""Registry service: list / inspect / delete fine-tuned models for the UI."""

from __future__ import annotations

from typing import Any, Optional

from llmstudio.config import Settings
from llmstudio.core.models.registry import ModelRecord, ModelRegistry
from llmstudio.core.utils.logging import get_logger

log = get_logger("services.registry")


class RegistryService:
    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings

    def list_models(self) -> list[ModelRecord]:
        return self.registry.list()

    def get(self, model_id: str) -> Optional[ModelRecord]:
        return self.registry.get(model_id)

    def table_rows(self) -> list[list[str]]:
        """Rows for a Gradio Dataframe: [name, base, kind, quant, created, id]."""
        rows: list[list[str]] = []
        for m in self.registry.list():
            rows.append([
                m.name,
                m.base_model_key,
                m.artifact_kind,
                m.quantization,
                m.created_str,
                m.id,
            ])
        return rows

    def choices(self) -> list[tuple[str, str]]:
        """(label, model_id) for dropdowns."""
        return [(f"{m.name} ({m.base_model_key})", m.id) for m in self.registry.list()]

    def details(self, model_id: str) -> dict[str, Any]:
        m = self.registry.get(model_id)
        if not m:
            return {}
        return {
            "id": m.id,
            "name": m.name,
            "base_model": m.base_model_key,
            "base_repo": m.base_repo,
            "artifact_kind": m.artifact_kind,
            "quantization": m.quantization,
            "path": m.path,
            "on_disk": m.exists_on_disk(),
            "dataset_id": m.dataset_id,
            "job_id": m.job_id,
            "created": m.created_str,
            "metrics": m.metrics,
            "config": m.config,
            "notes": m.notes,
        }

    def delete(self, model_id: str, *, remove_files: bool = False) -> bool:
        return self.registry.delete(model_id, remove_files=remove_files)
