"""Filesystem path discovery and resolution.

Keeps all path logic in one place so the rest of the app never hard-codes
locations. ``ResolvedPaths`` turns the (relative, user-editable) ``PathsConfig``
into absolute paths rooted at the runtime home directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime
    from llmstudio.config.settings import PathsConfig


def find_repo_root(start: Path | None = None) -> Path:
    """Best-effort discovery of the project root (where ``config/`` lives).

    Searches upward from the current working directory, then from this file's
    location (covers ``pip install -e .`` editable installs). Falls back to the
    cwd so callers always get *some* directory.
    """
    candidates: list[Path] = []
    start = (start or Path.cwd()).resolve()
    candidates.extend([start, *start.parents])
    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    for p in candidates:
        if (p / "config" / "default.yaml").exists():
            return p
    for p in candidates:
        if (p / "pyproject.toml").exists():
            return p
    return start


def find_config_dir() -> Path:
    """Directory holding ``default.yaml`` / ``models.yaml`` (override via env)."""
    env = os.environ.get("LLMSTUDIO_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return find_repo_root() / "config"


def default_config_path() -> Path:
    return find_config_dir() / "default.yaml"


def models_catalog_path() -> Path:
    env = os.environ.get("LLMSTUDIO_MODELS_CATALOG")
    if env:
        return Path(env).expanduser().resolve()
    return find_config_dir() / "models.yaml"


@dataclass(frozen=True)
class ResolvedPaths:
    """Absolute, ready-to-use runtime paths."""

    home: Path
    uploads: Path
    datasets: Path
    runs: Path
    models: Path
    hf_cache: Path
    logs: Path
    registry_db: Path

    @classmethod
    def build(cls, home_root: Path, cfg: "PathsConfig") -> "ResolvedPaths":
        def sub(value: str) -> Path:
            p = Path(value).expanduser()
            return p if p.is_absolute() else home_root / p

        return cls(
            home=home_root,
            uploads=sub(cfg.uploads),
            datasets=sub(cfg.datasets),
            runs=sub(cfg.runs),
            models=sub(cfg.models),
            hf_cache=sub(cfg.hf_cache),
            logs=sub(cfg.logs),
            registry_db=sub(cfg.registry_db),
        )

    def ensure_dirs(self) -> "ResolvedPaths":
        """Create all runtime directories if missing. Idempotent."""
        for p in (self.home, self.uploads, self.datasets, self.runs, self.models, self.hf_cache, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        self.registry_db.parent.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, job_id: str) -> Path:
        """Directory for a single training run (checkpoints, logs, config)."""
        return self.runs / job_id

    def model_dir(self, model_id: str) -> Path:
        """Directory for an exported / registered fine-tuned model."""
        return self.models / model_id

    def upload_dir(self, dataset_id: str) -> Path:
        return self.uploads / dataset_id

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets / dataset_id
