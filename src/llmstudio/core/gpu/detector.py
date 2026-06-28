"""Detect NVIDIA GPUs and their available VRAM.

Two backends, tried in order:
    1. ``pynvml`` (nvidia-ml-py) — works without torch, used by setup/doctor.
    2. ``torch.cuda`` — used when the training stack is installed.
Both degrade gracefully: on a machine with no NVIDIA GPU, ``detect_gpus``
returns a report with ``available=False`` instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llmstudio.core.utils.logging import get_logger

log = get_logger("gpu.detector")

_BYTES_PER_GB = 1024**3  # GiB — matches how nvidia-smi reports memory


@dataclass
class GpuDevice:
    index: int
    name: str
    total_gb: float
    free_gb: float
    used_gb: float

    @property
    def utilization_pct(self) -> float:
        return round(100.0 * self.used_gb / self.total_gb, 1) if self.total_gb else 0.0

    def describe(self) -> str:
        return (
            f"GPU {self.index}: {self.name} — "
            f"{self.free_gb:.1f} GB free / {self.total_gb:.1f} GB total"
        )


@dataclass
class GpuReport:
    available: bool
    devices: list[GpuDevice] = field(default_factory=list)
    source: str = "none"  # pynvml | torch | none
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    torch_version: Optional[str] = None
    error: Optional[str] = None

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def primary(self) -> Optional[GpuDevice]:
        return self.devices[0] if self.devices else None

    @property
    def total_vram_gb(self) -> float:
        return round(sum(d.total_gb for d in self.devices), 1)

    @property
    def max_free_gb(self) -> float:
        return round(max((d.free_gb for d in self.devices), default=0.0), 1)

    @property
    def min_free_gb(self) -> float:
        return round(min((d.free_gb for d in self.devices), default=0.0), 1)

    def summary(self) -> str:
        if not self.available:
            return f"No CUDA GPU detected ({self.error or 'CPU-only'})."
        lines = [d.describe() for d in self.devices]
        meta = []
        if self.driver_version:
            meta.append(f"driver {self.driver_version}")
        if self.cuda_version:
            meta.append(f"CUDA {self.cuda_version}")
        if meta:
            lines.append(" · ".join(meta))
        return "\n".join(lines)


def _detect_via_pynvml() -> Optional[GpuReport]:
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        log.debug("nvmlInit failed: %s", exc)
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        devices: list[GpuDevice] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            devices.append(
                GpuDevice(
                    index=i,
                    name=name,
                    total_gb=mem.total / _BYTES_PER_GB,
                    free_gb=mem.free / _BYTES_PER_GB,
                    used_gb=mem.used / _BYTES_PER_GB,
                )
            )
        driver = None
        cuda = None
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver = driver.decode() if isinstance(driver, bytes) else driver
        except Exception:
            pass
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda = f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:
            pass
        return GpuReport(
            available=len(devices) > 0,
            devices=devices,
            source="pynvml",
            driver_version=driver,
            cuda_version=cuda,
            torch_version=_torch_version(),
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _detect_via_torch() -> Optional[GpuReport]:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return GpuReport(available=False, source="torch", torch_version=torch.__version__, error="torch.cuda.is_available() is False")
    devices: list[GpuDevice] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
        except Exception:
            total_b = props.total_memory
            free_b = props.total_memory - torch.cuda.memory_reserved(i)
        devices.append(
            GpuDevice(
                index=i,
                name=props.name,
                total_gb=total_b / _BYTES_PER_GB,
                free_gb=free_b / _BYTES_PER_GB,
                used_gb=(total_b - free_b) / _BYTES_PER_GB,
            )
        )
    return GpuReport(
        available=True,
        devices=devices,
        source="torch",
        cuda_version=getattr(torch.version, "cuda", None),
        torch_version=torch.__version__,
    )


def _torch_version() -> Optional[str]:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def detect_gpus(prefer: str = "pynvml") -> GpuReport:
    """Return a :class:`GpuReport` describing available GPUs.

    Args:
        prefer: ``"pynvml"`` (default) reports the most accurate *system-wide*
            free memory; ``"torch"`` reflects what the current torch process
            sees. Falls back to the other backend automatically.
    """
    order = ("pynvml", "torch") if prefer == "pynvml" else ("torch", "pynvml")
    last: Optional[GpuReport] = None
    for backend in order:
        report = _detect_via_pynvml() if backend == "pynvml" else _detect_via_torch()
        if report is not None:
            last = report
            if report.available:
                return report
    return last or GpuReport(available=False, source="none", error="no GPU backend available")
