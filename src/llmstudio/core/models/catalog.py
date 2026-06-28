"""The base-model catalog — the dropdown of fine-tunable open-source LLMs.

Loaded from ``config/models.yaml`` (see :func:`llmstudio.config.models_catalog_path`).
A small built-in fallback list keeps the app usable if the YAML is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from llmstudio.config import models_catalog_path
from llmstudio.core.utils.logging import get_logger

log = get_logger("models.catalog")


@dataclass
class ModelCatalogEntry:
    key: str
    name: str
    family: str
    hf_id: str
    params_b: float
    context_length: int = 4096
    chat_template: str = "chatml"
    unsloth_id: Optional[str] = None
    unsloth_4bit_id: Optional[str] = None
    gated: bool = False
    license: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def candidate_repos(
        self,
        load_in_4bit: bool,
        prefer_unsloth_4bit: bool = True,
        allow_official_fallback: bool = True,
    ) -> list[str]:
        """Ordered list of repo IDs to try when downloading/loading this model."""
        cands: list[str] = []
        if load_in_4bit:
            if prefer_unsloth_4bit and self.unsloth_4bit_id:
                cands.append(self.unsloth_4bit_id)
            if self.unsloth_id:
                cands.append(self.unsloth_id)
            if allow_official_fallback:
                cands.append(self.hf_id)
            if self.unsloth_4bit_id and self.unsloth_4bit_id not in cands:
                cands.append(self.unsloth_4bit_id)
        else:
            if self.unsloth_id:
                cands.append(self.unsloth_id)
            cands.append(self.hf_id)
        return list(dict.fromkeys(c for c in cands if c))

    @property
    def is_recommended(self) -> bool:
        return "recommended" in self.tags

    def label(self) -> str:
        """Display label for dropdowns (no decorative marks)."""
        return f"{self.name} ({self.params_b:g}B)"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelCatalogEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# Built-in fallback so the catalog is never empty.
_FALLBACK_ENTRIES = [
    ModelCatalogEntry(
        key="qwen2.5-3b-instruct",
        name="Qwen2.5 3B Instruct",
        family="qwen",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        unsloth_id="unsloth/Qwen2.5-3B-Instruct",
        unsloth_4bit_id="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        params_b=3.09,
        context_length=32768,
        chat_template="qwen-2.5",
        tags=["recommended", "small"],
        description="Built-in fallback entry.",
    ),
]


class ModelCatalog:
    """A queryable collection of :class:`ModelCatalogEntry`."""

    def __init__(self, entries: list[ModelCatalogEntry]) -> None:
        self._entries = entries
        self._by_key = {e.key: e for e in entries}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ModelCatalog":
        path = Path(path) if path else models_catalog_path()
        if not path.exists():
            log.warning("Model catalog not found at %s; using built-in fallback.", path)
            return cls(list(_FALLBACK_ENTRIES))
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            raw = data.get("models", data if isinstance(data, list) else [])
            entries = [ModelCatalogEntry.from_dict(d) for d in raw]
            if not entries:
                raise ValueError("empty catalog")
            log.info("Loaded %d base models from %s", len(entries), path.name)
            return cls(entries)
        except Exception as exc:  # pragma: no cover
            log.error("Failed to read model catalog (%s); using fallback.", exc)
            return cls(list(_FALLBACK_ENTRIES))

    # -- queries ------------------------------------------------------------
    def all(self) -> list[ModelCatalogEntry]:
        return list(self._entries)

    def get(self, key: str) -> ModelCatalogEntry:
        if key not in self._by_key:
            raise KeyError(f"Unknown model key: {key!r}")
        return self._by_key[key]

    def find(self, key: str) -> Optional[ModelCatalogEntry]:
        return self._by_key.get(key)

    def keys(self) -> list[str]:
        return list(self._by_key)

    def families(self) -> list[str]:
        return sorted({e.family for e in self._entries})

    def all_tags(self) -> list[str]:
        tags: list[str] = []
        for e in self._entries:
            for t in e.tags:
                if t not in tags:
                    tags.append(t)
        return tags

    def filter(self, tag: Optional[str] = None, family: Optional[str] = None) -> list[ModelCatalogEntry]:
        out = self._entries
        if tag:
            out = [e for e in out if tag in e.tags]
        if family:
            out = [e for e in out if e.family == family]
        return list(out)

    def recommended(self) -> list[ModelCatalogEntry]:
        return [e for e in self._entries if e.is_recommended]

    def choices(self) -> list[tuple[str, str]]:
        """(label, key) tuples for a Gradio dropdown, recommended first."""
        ordered = sorted(self._entries, key=lambda e: (not e.is_recommended, e.params_b))
        return [(e.label(), e.key) for e in ordered]
