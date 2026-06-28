"""Dataset preparation: the glue that turns a RawDataset into a train-ready,
on-disk dataset with a validation report and reproducible train/eval split.

On-disk layout (under ``<datasets>/<dataset_id>/``):
    train.jsonl          # one Example.to_record() per line
    eval.jsonl           # held-out split (may be empty)
    dataset_info.json    # DatasetSchema (format, counts, mapping, stats)
    validation.json      # ValidationReport
    preview.txt          # a few rendered examples for eyeballing
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llmstudio.core.data.formatter import normalize_records, render_preview
from llmstudio.core.data.loaders import RawDataset
from llmstudio.core.data.schema import DatasetSchema, Example, FieldMapping, TaskFormat
from llmstudio.core.data.validator import ValidationReport, validate_examples
from llmstudio.core.utils.logging import get_logger

log = get_logger("data.preparer")

TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
INFO_FILE = "dataset_info.json"
VALIDATION_FILE = "validation.json"
PREVIEW_FILE = "preview.txt"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "dataset"


def make_dataset_id(name: str) -> str:
    return f"{slugify(name)[:40]}-{uuid.uuid4().hex[:8]}"


@dataclass
class PreparedDataset:
    dataset_id: str
    name: str
    directory: Path
    schema: DatasetSchema
    report: Optional[ValidationReport]
    train_path: Path
    eval_path: Path
    info_path: Path

    @property
    def task_format(self) -> TaskFormat:
        return self.schema.task_format

    @property
    def n_train(self) -> int:
        return self.schema.n_train

    @property
    def n_eval(self) -> int:
        return self.schema.n_eval


def _write_jsonl(path: Path, examples: list[Example]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")


def read_jsonl_examples(path: Path, limit: Optional[int] = None) -> list[Example]:
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                examples.append(Example.from_record(json.loads(line)))
    return examples


class DataPreparer:
    """Prepares datasets and persists them under ``datasets_root``."""

    def __init__(self, datasets_root: Path) -> None:
        self.datasets_root = Path(datasets_root)
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        raw: RawDataset,
        mapping: FieldMapping,
        *,
        name: str,
        eval_ratio: float = 0.05,
        seed: int = 3407,
        max_seq_length: int = 2048,
        shuffle: bool = True,
        dataset_id: Optional[str] = None,
    ) -> PreparedDataset:
        """Normalize, validate, split, and write a dataset to disk."""
        dataset_id = dataset_id or make_dataset_id(name)
        out_dir = self.datasets_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Preparing dataset '%s' (%s) from %d raw records", name, dataset_id, raw.n_records)

        examples, dropped = normalize_records(raw.records, mapping)
        report = validate_examples(examples, max_seq_length=max_seq_length, dropped=dropped)

        # Drop hard-empty-output rows from the training set, but keep the report.
        bad_rows = {
            r for issue in report.issues if issue.code == "empty_output" for r in issue.rows
        }
        # report.rows are truncated to a preview, so recompute precisely:
        usable = [ex for i, ex in enumerate(examples) if i not in bad_rows] if bad_rows else examples

        # Reproducible split.
        indices = list(range(len(usable)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        n_eval = int(len(usable) * eval_ratio) if eval_ratio > 0 else 0
        n_eval = min(n_eval, max(0, len(usable) - 1))  # never leave train empty
        eval_idx = set(indices[:n_eval])
        train_examples = [usable[i] for i in indices[n_eval:]]
        eval_examples = [usable[i] for i in indices[:n_eval]]

        train_path = out_dir / TRAIN_FILE
        eval_path = out_dir / EVAL_FILE
        _write_jsonl(train_path, train_examples)
        _write_jsonl(eval_path, eval_examples)

        has_system = any(ex.system() for ex in usable[:200])
        schema = DatasetSchema(
            dataset_id=dataset_id,
            name=name,
            task_format=mapping.task_format,
            n_examples=len(usable),
            n_train=len(train_examples),
            n_eval=len(eval_examples),
            has_system=has_system,
            source_columns=raw.columns,
            mapping=mapping.to_dict(),
            source_files=raw.source_files,
            stats=report.stats,
        )

        (out_dir / INFO_FILE).write_text(json.dumps(schema.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / VALIDATION_FILE).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_preview(out_dir / PREVIEW_FILE, train_examples)

        log.info("Prepared '%s': %d train / %d eval", dataset_id, len(train_examples), len(eval_examples))
        return PreparedDataset(
            dataset_id=dataset_id,
            name=name,
            directory=out_dir,
            schema=schema,
            report=report,
            train_path=train_path,
            eval_path=eval_path,
            info_path=out_dir / INFO_FILE,
        )

    @staticmethod
    def _write_preview(path: Path, examples: list[Example], n: int = 5) -> None:
        blocks = []
        for i, ex in enumerate(examples[:n], 1):
            blocks.append(f"===== Example {i} =====\n{render_preview(ex)}")
        path.write_text("\n\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------------------
# Loading prepared datasets back from disk
# ---------------------------------------------------------------------------
def load_prepared(directory: Path) -> PreparedDataset:
    directory = Path(directory)
    info = json.loads((directory / INFO_FILE).read_text(encoding="utf-8"))
    schema = DatasetSchema.from_dict(info)
    report = None
    vfile = directory / VALIDATION_FILE
    if vfile.exists():
        vdata = json.loads(vfile.read_text(encoding="utf-8"))
        report = ValidationReport(
            n_examples=vdata.get("n_examples", schema.n_examples),
            n_valid=vdata.get("n_valid", schema.n_examples),
            stats=vdata.get("stats", {}),
        )
    return PreparedDataset(
        dataset_id=schema.dataset_id,
        name=schema.name,
        directory=directory,
        schema=schema,
        report=report,
        train_path=directory / TRAIN_FILE,
        eval_path=directory / EVAL_FILE,
        info_path=directory / INFO_FILE,
    )


def list_prepared(datasets_root: Path) -> list[PreparedDataset]:
    datasets_root = Path(datasets_root)
    if not datasets_root.exists():
        return []
    out: list[PreparedDataset] = []
    for child in sorted(datasets_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (child / INFO_FILE).exists():
            try:
                out.append(load_prepared(child))
            except Exception as exc:  # pragma: no cover
                log.warning("Skipping unreadable dataset %s: %s", child.name, exc)
    return out
