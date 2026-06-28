"""Data pipeline: load → normalize → validate → split → persist.

Typical flow::

    from llmstudio.core.data import load_paths, guess_mapping, DataPreparer

    raw = load_paths("my_data.csv")
    mapping = guess_mapping(raw.columns)        # or build a FieldMapping yourself
    preparer = DataPreparer(datasets_root)
    prepared = preparer.prepare(raw, mapping, name="support-bot")
    print(prepared.report.to_markdown())
"""

from llmstudio.core.data.formatter import (
    NormalizationError,
    normalize_record,
    normalize_records,
    render_preview,
)
from llmstudio.core.data.loaders import (
    DOCUMENT_EXT,
    STRUCTURED_EXT,
    SUPPORTED_EXT,
    DataLoadError,
    RawDataset,
    load_file,
    load_paths,
)
from llmstudio.core.data.preparer import (
    DataPreparer,
    PreparedDataset,
    list_prepared,
    load_prepared,
    make_dataset_id,
    read_jsonl_examples,
    slugify,
)
from llmstudio.core.data.schema import (
    DatasetSchema,
    Example,
    FieldMapping,
    TaskFormat,
    guess_mapping,
)
from llmstudio.core.data.validator import Issue, ValidationReport, validate_examples

__all__ = [
    "DOCUMENT_EXT",
    "STRUCTURED_EXT",
    "SUPPORTED_EXT",
    "DataLoadError",
    "DataPreparer",
    "DatasetSchema",
    "Example",
    "FieldMapping",
    "Issue",
    "NormalizationError",
    "PreparedDataset",
    "RawDataset",
    "TaskFormat",
    "ValidationReport",
    "guess_mapping",
    "list_prepared",
    "load_file",
    "load_paths",
    "load_prepared",
    "make_dataset_id",
    "normalize_record",
    "normalize_records",
    "read_jsonl_examples",
    "render_preview",
    "slugify",
    "validate_examples",
]
