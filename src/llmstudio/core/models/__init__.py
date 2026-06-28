"""Base-model catalog, on-demand downloader, and fine-tuned-model registry."""

from llmstudio.core.models.catalog import ModelCatalog, ModelCatalogEntry
from llmstudio.core.models.downloader import (
    DownloadResult,
    GatedModelError,
    ModelDownloadError,
    download_model,
    download_repo,
    is_cached,
    repo_size_bytes,
)
from llmstudio.core.models.registry import (
    KIND_GGUF,
    KIND_LORA,
    KIND_MERGED_4BIT,
    KIND_MERGED_16BIT,
    STATUS_READY,
    ModelRecord,
    ModelRegistry,
    make_model_id,
)

__all__ = [
    "KIND_GGUF",
    "KIND_LORA",
    "KIND_MERGED_4BIT",
    "KIND_MERGED_16BIT",
    "STATUS_READY",
    "DownloadResult",
    "GatedModelError",
    "ModelCatalog",
    "ModelCatalogEntry",
    "ModelDownloadError",
    "ModelRecord",
    "ModelRegistry",
    "download_model",
    "download_repo",
    "is_cached",
    "make_model_id",
    "repo_size_bytes",
]
