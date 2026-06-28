"""System service: GPU/memory status, first-run setup, and an environment doctor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from llmstudio.config import Settings
from llmstudio.core.assistant.llm import AssistantLLM
from llmstudio.core.gpu.detector import GpuReport, detect_gpus
from llmstudio.core.models.catalog import ModelCatalog
from llmstudio.core.utils.logging import get_logger
from llmstudio.core.utils.resources import memory_snapshot

log = get_logger("services.system")

ProgressFn = Callable[[str], None]


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str


class SystemService:
    def __init__(self, settings: Settings, catalog: ModelCatalog, assistant: AssistantLLM) -> None:
        self.settings = settings
        self.catalog = catalog
        self.assistant = assistant

    # ---------------------------------------------------------------- info
    def gpu_report(self) -> GpuReport:
        return detect_gpus()

    def memory(self) -> dict[str, Any]:
        snap = memory_snapshot()
        return snap.__dict__.copy()

    def status_markdown(self) -> str:
        gpu = self.gpu_report()
        lines = ["### System", gpu.summary()]
        if gpu.torch_version:
            lines.append(f"torch {gpu.torch_version}")
        lines.append(f"\n**Assistant:** {self._assistant_status()}")
        lines.append(f"**Workspace:** `{self.settings.home_root}`")
        lines.append(f"**Base models in catalog:** {len(self.catalog.all())}")
        return "\n\n".join(lines)

    def _assistant_status(self) -> str:
        if not self.settings.assistant.enabled:
            return "disabled"
        return self.settings.assistant.model_id if self.assistant.available() else "unavailable (install [train])"

    # --------------------------------------------------------------- setup
    def setup(self, *, download_assistant: bool = True, progress: Optional[ProgressFn] = None) -> dict[str, Any]:
        """First-run setup: ensure dirs, report GPU, pre-download the assistant model."""
        def emit(msg: str) -> None:
            log.info(msg)
            if progress:
                progress(msg)

        emit("Creating workspace directories…")
        self.settings.ensure_dirs()

        gpu = self.gpu_report()
        emit(f"GPU: {gpu.summary().splitlines()[0] if gpu.available else 'none detected'}")

        result: dict[str, Any] = {"gpu": gpu.summary(), "workspace": str(self.settings.home_root)}

        if download_assistant and self.settings.assistant.enabled:
            if not self.assistant.available():
                emit("Training stack not installed — skipping assistant download. "
                     "Run `pip install 'llmstudio[train]'` then re-run setup.")
                result["assistant"] = "skipped (no training stack)"
            else:
                from llmstudio.core.models.downloader import download_repo

                model_id = self.assistant.resolve_model_id()
                emit(f"Downloading assistant model: {model_id} (one-time)…")
                try:
                    download_repo(
                        model_id,
                        token=self.settings.hf_token(),  # cache_dir=None → shared HF_HOME cache
                        progress=emit,
                    )
                    result["assistant"] = model_id
                    emit("Assistant model ready.")
                except Exception as exc:
                    result["assistant"] = f"failed: {exc}"
                    emit(f"Assistant download failed: {exc}")
        emit("Setup complete.")
        return result

    # -------------------------------------------------------------- doctor
    def doctor(self) -> list[DoctorCheck]:
        checks: list[DoctorCheck] = []

        # torch + CUDA
        try:
            import torch

            checks.append(DoctorCheck("PyTorch", True, torch.__version__))
            checks.append(DoctorCheck(
                "CUDA", torch.cuda.is_available(),
                f"cuda {getattr(torch.version, 'cuda', '?')}" if torch.cuda.is_available() else "no CUDA device",
            ))
        except Exception:
            checks.append(DoctorCheck("PyTorch", False, "not installed — pip install 'llmstudio[train]'"))

        # unsloth
        try:
            import unsloth  # noqa: F401

            checks.append(DoctorCheck("Unsloth", True, "installed"))
        except Exception:
            checks.append(DoctorCheck("Unsloth", False, "not installed (needed for training)"))

        # GPU
        gpu = self.gpu_report()
        checks.append(DoctorCheck("GPU", gpu.available, gpu.summary().splitlines()[0] if gpu.available else "none"))

        # HF token
        has_token = bool(self.settings.hf_token())
        checks.append(DoctorCheck("HF token", has_token, "set" if has_token else "unset (needed for gated models)"))

        # workspace writable
        try:
            self.settings.ensure_dirs()
            checks.append(DoctorCheck("Workspace", True, str(self.settings.home_root)))
        except Exception as exc:
            checks.append(DoctorCheck("Workspace", False, str(exc)))

        return checks
