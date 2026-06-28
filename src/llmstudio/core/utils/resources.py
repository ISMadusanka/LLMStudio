"""GPU/CPU resource inspection and release.

The functions here are import-safe: they degrade to no-ops when torch/CUDA are
absent, so they can be called unconditionally from the services layer.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

from llmstudio.core.utils.logging import get_logger

log = get_logger("utils.resources")


@dataclass
class MemorySnapshot:
    """A point-in-time view of system + GPU memory (GB)."""

    cpu_total_gb: float = 0.0
    cpu_used_gb: float = 0.0
    gpu_total_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_free_gb: float = 0.0
    gpu_count: int = 0


def free_gpu_memory() -> None:
    """Run GC and release cached CUDA memory. No-op when torch/CUDA absent."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # not fatal
                pass
    except Exception as exc:  # pragma: no cover - torch optional
        log.debug("free_gpu_memory skipped: %s", exc)


def unload(*owners: object, attrs: tuple[str, ...] = ("model", "tokenizer", "_model", "_tokenizer")) -> None:
    """Best-effort unload: clear known heavy attributes on the given owners, then
    free GPU memory.

    NOTE: the caller must also drop its *own* references (e.g. ``self._x = None``)
    for memory to actually be reclaimed; Python can't release a name held by the
    caller from in here.
    """
    for owner in owners:
        for attr in attrs:
            if hasattr(owner, attr):
                try:
                    setattr(owner, attr, None)
                except Exception:
                    pass
    free_gpu_memory()


def memory_snapshot() -> MemorySnapshot:
    """Capture CPU + (primary) GPU memory usage. Safe without torch/psutil."""
    snap = MemorySnapshot()
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap.cpu_total_gb = vm.total / 1e9
        snap.cpu_used_gb = (vm.total - vm.available) / 1e9
    except Exception as exc:  # pragma: no cover
        log.debug("psutil unavailable: %s", exc)

    try:
        import torch

        if torch.cuda.is_available():
            snap.gpu_count = torch.cuda.device_count()
            free_b, total_b = torch.cuda.mem_get_info()  # current device
            snap.gpu_total_gb = total_b / 1e9
            snap.gpu_free_gb = free_b / 1e9
            snap.gpu_used_gb = (total_b - free_b) / 1e9
    except Exception as exc:  # pragma: no cover
        log.debug("cuda mem info unavailable: %s", exc)
    return snap


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def current_device_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return None
    return None
