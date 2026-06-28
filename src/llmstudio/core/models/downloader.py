"""On-demand model downloading from the Hugging Face Hub.

Base models are downloaded **only when the user selects one and starts a run**
(the assistant model is the only thing fetched at setup time). The downloader
tries the catalog's candidate repos in order (Unsloth 4-bit → Unsloth mirror →
official), so a missing pre-quantized repo gracefully falls back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from llmstudio.core.models.catalog import ModelCatalogEntry
from llmstudio.core.utils.logging import get_logger

log = get_logger("models.downloader")

ProgressFn = Callable[[str], None]


class ModelDownloadError(RuntimeError):
    pass


class GatedModelError(ModelDownloadError):
    """Repo requires accepting a license and/or an HF token."""


@dataclass
class DownloadResult:
    repo_id: str
    local_path: str
    candidates_tried: list[str]
    size_bytes: int = 0

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1e9, 2)


def _emit(progress: Optional[ProgressFn], message: str) -> None:
    log.info(message)
    if progress:
        try:
            progress(message)
        except Exception:
            pass


def is_cached(repo_id: str, cache_dir: Optional[Union[str, Path]] = None) -> bool:
    """Return True if the repo already appears in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, cache_dir=str(cache_dir) if cache_dir else None, local_files_only=True)
        return True
    except Exception:
        return False


def repo_size_bytes(repo_id: str, token: Optional[str] = None) -> int:
    """Best-effort total size of a model repo (0 if it can't be determined)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, token=token, files_metadata=True)
        return sum((getattr(s, "size", None) or 0) for s in (info.siblings or []))
    except Exception as exc:  # pragma: no cover - network/optional
        log.debug("repo_size_bytes(%s) failed: %s", repo_id, exc)
        return 0


def _snapshot(repo_id: str, cache_dir: Optional[str], token: Optional[str]) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        token=token,
        # Skip artifacts we never need for fine-tuning.
        ignore_patterns=["*.gguf", "*.pth", "original/*", "*.onnx"],
    )


def _is_gated_error(exc: Exception) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return (
        "gated" in name.lower()
        or "401" in text
        or "403" in text
        or "awaiting a review" in text
        or "access to model" in text
        or "you must be authenticated" in text
    )


def download_repo(
    repo_id: str,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> DownloadResult:
    """Download a single repo by ID into the HF cache."""
    cache = str(cache_dir) if cache_dir else None
    _emit(progress, f"Downloading {repo_id} …")
    try:
        path = _snapshot(repo_id, cache, token)
    except Exception as exc:
        if _is_gated_error(exc):
            raise GatedModelError(
                f"'{repo_id}' is gated. Accept its license on the Hugging Face model "
                f"page and set HF_TOKEN, then retry."
            ) from exc
        raise ModelDownloadError(f"Failed to download '{repo_id}': {exc}") from exc
    _emit(progress, f"Downloaded {repo_id}.")
    return DownloadResult(repo_id=repo_id, local_path=path, candidates_tried=[repo_id])


def download_model(
    entry: ModelCatalogEntry,
    *,
    load_in_4bit: bool,
    prefer_unsloth_4bit: bool = True,
    allow_official_fallback: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> DownloadResult:
    """Download a catalog model, trying candidate repos in priority order.

    Returns the first repo that downloads successfully. Re-raises
    :class:`GatedModelError` immediately (retrying other repos won't help if the
    user lacks access to the official one and there's no open mirror).
    """
    candidates = entry.candidate_repos(load_in_4bit, prefer_unsloth_4bit, allow_official_fallback)
    if not candidates:
        raise ModelDownloadError(f"No candidate repositories for model '{entry.key}'.")

    tried: list[str] = []
    last_error: Optional[Exception] = None
    for repo_id in candidates:
        tried.append(repo_id)
        try:
            result = download_repo(repo_id, cache_dir=cache_dir, token=token, progress=progress)
            result.candidates_tried = tried
            return result
        except GatedModelError:
            raise
        except ModelDownloadError as exc:
            last_error = exc
            _emit(progress, f"{repo_id} unavailable, trying next candidate…")
            continue

    raise ModelDownloadError(
        f"Could not download model '{entry.key}'. Tried: {tried}. Last error: {last_error}"
    )
