"""LLM Studio — a no-code studio for fine-tuning open-source LLMs.

Public surface is intentionally small. Heavy submodules (training, inference,
assistant) lazily import torch/transformers/unsloth only when used, so importing
``llmstudio`` stays cheap and works on machines without a GPU.
"""

import os as _os

from llmstudio.version import __version__

__all__ = ["__version__", "get_settings"]


def get_settings():
    """Convenience accessor for the resolved application settings.

    Imported lazily to avoid pulling the config stack at package import time.
    """
    from llmstudio.config import get_settings as _get_settings

    return _get_settings()


def _bootstrap_hf_home() -> None:
    """Point the Hugging Face cache at the configured workspace, early.

    Runs before transformers/huggingface_hub are imported so that BOTH the
    downloader and ``from_pretrained`` share one cache directory. This lets us
    load models by repo ID (reusing the downloaded weights, no re-download) —
    which also avoids a transformers local-path bug triggered by ``_is_local``.
    Respects an existing HF_HOME / HF_HUB_CACHE if the user set one.
    """
    if any(_os.environ.get(v) for v in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")):
        return
    try:
        hf = get_settings().resolved_paths.hf_cache
        hf.mkdir(parents=True, exist_ok=True)
        _os.environ["HF_HOME"] = str(hf)
        # HF_HUB_CACHE is the dir snapshot_download/from_pretrained actually use
        # for model weights; pinning it here matches what the downloader wrote so
        # already-downloaded models are reused (no re-download).
        _os.environ["HF_HUB_CACHE"] = str(hf)
    except Exception:
        pass


_bootstrap_hf_home()
