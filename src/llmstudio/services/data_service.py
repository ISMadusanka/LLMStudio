"""Data service: staging uploads, previewing, mapping, and preparing datasets."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from llmstudio.config import Settings
from llmstudio.core.assistant.data_assistant import DataAssistant, MappingSuggestion
from llmstudio.core.data import (
    DataPreparer,
    FieldMapping,
    PreparedDataset,
    RawDataset,
    list_prepared,
    load_paths,
    load_prepared,
)
from llmstudio.core.utils.logging import get_logger

log = get_logger("services.data")


class DataService:
    def __init__(self, settings: Settings, data_assistant: Optional[DataAssistant] = None) -> None:
        self.settings = settings
        self.paths = settings.resolved_paths
        self.data_assistant = data_assistant
        self.preparer = DataPreparer(self.paths.datasets)

    # ----------------------------------------------------------- uploading
    def stage_uploads(self, file_paths: list[str], *, copy: bool = True) -> tuple[Path, RawDataset]:
        """Copy uploaded files into the workspace and load them into a RawDataset."""
        upload_dir = self.paths.uploads / uuid.uuid4().hex[:10]
        upload_dir.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        for fp in file_paths:
            src = Path(fp)
            dest = upload_dir / src.name
            if copy:
                shutil.copy2(src, dest)
            else:
                dest = src
            staged.append(dest)
        raw = load_paths(staged)
        log.info("Staged %d file(s) → %d records", len(staged), raw.n_records)
        return upload_dir, raw

    def preview(self, raw: RawDataset, n: int = 5) -> dict[str, Any]:
        return {
            "columns": raw.columns,
            "n_records": raw.n_records,
            "sample": raw.sample(n),
            "notes": raw.notes,
        }

    # ----------------------------------------------------------- mapping
    def suggest_mapping(self, raw: RawDataset) -> MappingSuggestion:
        if self.data_assistant is not None:
            return self.data_assistant.suggest_mapping(raw.columns, raw.sample(5))
        from llmstudio.core.data import guess_mapping

        return MappingSuggestion(mapping=guess_mapping(raw.columns), source="heuristic")

    # ----------------------------------------------------------- preparing
    def prepare(
        self,
        raw: RawDataset,
        mapping: FieldMapping,
        *,
        name: str,
        eval_ratio: Optional[float] = None,
        max_seq_length: Optional[int] = None,
    ) -> PreparedDataset:
        return self.preparer.prepare(
            raw,
            mapping,
            name=name,
            eval_ratio=self.settings.training.eval_ratio if eval_ratio is None else eval_ratio,
            seed=self.settings.training.seed,
            max_seq_length=max_seq_length or self.settings.training.max_seq_length,
        )

    # ----------------------------------------------------------- listing
    def list_datasets(self) -> list[PreparedDataset]:
        return list_prepared(self.paths.datasets)

    def get_dataset(self, dataset_id: str) -> Optional[PreparedDataset]:
        directory = self.paths.dataset_dir(dataset_id)
        if not (directory / "dataset_info.json").exists():
            return None
        return load_prepared(directory)

    def dataset_choices(self) -> list[tuple[str, str]]:
        """(label, dataset_id) for dropdowns."""
        out = []
        for ds in self.list_datasets():
            out.append((f"{ds.name} · {ds.n_train} train ({ds.task_format.value})", ds.dataset_id))
        return out
