"""Load raw user data from many file formats into a uniform record list.

Structured formats (csv/tsv/json/jsonl/xlsx/parquet) become one record per row.
Unstructured documents (txt/pdf/docx/md) become ``{"text": ...}`` records (one
per page/paragraph chunk) suitable for completion-style tuning or for the data
assistant to synthesize instruction/output pairs from.

Optional parsers (pypdf, python-docx, openpyxl) are imported lazily; a clear
error is raised only if the user actually loads that format without the parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from llmstudio.core.utils.logging import get_logger

log = get_logger("data.loaders")

STRUCTURED_EXT = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"}
DOCUMENT_EXT = {".txt", ".md", ".pdf", ".docx"}
SUPPORTED_EXT = STRUCTURED_EXT | DOCUMENT_EXT


@dataclass
class RawDataset:
    """A loaded-but-not-yet-structured dataset: a list of dict records."""

    records: list[dict[str, Any]]
    source_files: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.records)

    def sample(self, n: int = 5) -> list[dict[str, Any]]:
        return self.records[:n]

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.records)

    def infer_columns(self) -> list[str]:
        cols: list[str] = []
        for rec in self.records[:200]:  # scan a window
            for k in rec:
                if k not in cols:
                    cols.append(k)
        self.columns = cols
        return cols


class DataLoadError(RuntimeError):
    """Raised when a file cannot be parsed or a parser dependency is missing."""


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataLoadError(f"{path.name}: invalid JSON on line {ln}: {exc}") from exc
            records.append(obj if isinstance(obj, dict) else {"value": obj})
    return records


def _read_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [d if isinstance(d, dict) else {"value": d} for d in data]
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "records", "items"):
            if isinstance(data.get(key), list):
                return [d if isinstance(d, dict) else {"value": d} for d in data[key]]
        return [data]  # a single object
    raise DataLoadError(f"{path.name}: unsupported top-level JSON type {type(data).__name__}")


def _read_tabular(path: Path, sep: Optional[str] = None) -> list[dict[str, Any]]:
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path, sep=sep)
    except ImportError as exc:
        raise DataLoadError(
            f"Reading {suffix} needs an extra parser. Install with: pip install 'llmstudio[data]'"
        ) from exc
    except Exception as exc:
        raise DataLoadError(f"{path.name}: failed to parse ({exc})") from exc
    df = df.where(df.notna(), None)  # NaN -> None for clean JSON
    return df.to_dict(orient="records")


def _read_text(path: Path, min_chars: int = 1) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Split into paragraphs on blank lines; keep reasonably sized chunks.
    chunks = [c.strip() for c in text.split("\n\n")]
    records = [{"text": c, "source": path.name} for c in chunks if len(c) >= min_chars]
    if not records and text.strip():
        records = [{"text": text.strip(), "source": path.name}]
    return records


def _read_pdf(path: Path) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise DataLoadError(
            "Reading PDF needs pypdf. Install with: pip install 'llmstudio[data]'"
        ) from exc
    reader = PdfReader(str(path))
    records: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages, 1):
        content = (page.extract_text() or "").strip()
        if content:
            records.append({"text": content, "page": i, "source": path.name})
    return records


def _read_docx(path: Path) -> list[dict[str, Any]]:
    try:
        import docx  # python-docx
    except Exception as exc:
        raise DataLoadError(
            "Reading DOCX needs python-docx. Install with: pip install 'llmstudio[data]'"
        ) from exc
    document = docx.Document(str(path))
    paras = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    # Group consecutive paragraphs into ~paragraph-sized records.
    return [{"text": p, "source": path.name} for p in paras]


_READERS = {
    ".jsonl": _read_jsonl,
    ".ndjson": _read_jsonl,
    ".json": _read_json,
    ".csv": _read_tabular,
    ".tsv": _read_tabular,
    ".xlsx": _read_tabular,
    ".xls": _read_tabular,
    ".parquet": _read_tabular,
    ".txt": _read_text,
    ".md": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_file(path: Union[str, Path]) -> RawDataset:
    """Load a single file into a :class:`RawDataset`."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise DataLoadError(
            f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXT)}"
        )
    log.info("Loading %s", path.name)
    records = reader(path)
    ds = RawDataset(records=records, source_files=[str(path)])
    ds.infer_columns()
    log.info("Loaded %d records from %s", ds.n_records, path.name)
    return ds


def load_paths(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> RawDataset:
    """Load one or many files / a directory into a single combined RawDataset."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXT)
            if not files:
                raise DataLoadError(f"No supported files found in {p}")
            paths = files
        else:
            paths = [p]

    combined: list[dict[str, Any]] = []
    sources: list[str] = []
    notes: list[str] = []
    for item in paths:
        try:
            ds = load_file(item)
            combined.extend(ds.records)
            sources.extend(ds.source_files)
        except DataLoadError as exc:
            notes.append(str(exc))
            log.warning("Skipping %s: %s", item, exc)

    if not combined:
        raise DataLoadError("No records could be loaded. " + (" ".join(notes) if notes else ""))

    result = RawDataset(records=combined, source_files=sources, notes=notes)
    result.infer_columns()
    return result
