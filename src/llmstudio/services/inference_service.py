"""Inference service: load a registry model or a paused run's checkpoint and chat.

Keeps a single resident model (single-GPU friendly) and refuses to load while a
training job is actively using the GPU — the user must pause first, which is the
intended "probe the latest checkpoint" workflow.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional, Union

from llmstudio.config import Settings
from llmstudio.core.inference.engine import GenerationParams, InferenceEngine, Messages
from llmstudio.core.models.registry import ModelRegistry
from llmstudio.core.training.job import JobStatus
from llmstudio.core.training.manager import JobManager
from llmstudio.core.utils.logging import get_logger

log = get_logger("services.inference")

_BUSY_STATUSES = {JobStatus.RUNNING, JobStatus.DOWNLOADING, JobStatus.PREPARING, JobStatus.PAUSING}


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry, jobs: JobManager) -> None:
        self.settings = settings
        self.registry = registry
        self.jobs = jobs
        self._engine: Optional[InferenceEngine] = None
        self._label: str = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------- guards
    def _assert_gpu_free(self) -> None:
        for job in self.jobs.list():
            if job.status in _BUSY_STATUSES:
                raise RuntimeError(
                    f"Job '{job.name or job.id}' is {job.status.value}. Pause it before running inference."
                )

    @property
    def current_label(self) -> str:
        return self._label if (self._engine and self._engine.is_loaded) else ""

    # -------------------------------------------------------------- loading
    def load_registered(self, model_id: str, *, progress: Optional[Callable[[str], None]] = None) -> str:
        record = self.registry.get(model_id)
        if record is None:
            raise ValueError(f"Model '{model_id}' not found in registry.")
        if not record.exists_on_disk():
            raise FileNotFoundError(f"Model files are missing at {record.path}.")
        load_in_4bit = record.artifact_kind == "merged_4bit" or (
            record.artifact_kind == "lora" and record.quantization == "qlora"
        )
        chat_template = record.config.get("chat_template")
        max_seq = int(record.config.get("max_seq_length", 4096))
        return self._load(record.path, load_in_4bit, chat_template, max_seq, label=f"{record.name}", progress=progress)

    def load_checkpoint(self, job_id: str, *, progress: Optional[Callable[[str], None]] = None) -> str:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        if job.status == JobStatus.RUNNING:
            raise RuntimeError("Pause the run before probing its checkpoint.")
        if not job.last_checkpoint:
            raise RuntimeError("This run has no checkpoint yet.")
        from llmstudio.core.training.config import TrainingConfig

        cfg = TrainingConfig.from_dict(job.config)
        return self._load(
            job.last_checkpoint,
            cfg.load_in_4bit,
            cfg.chat_template,
            cfg.max_seq_length,
            label=f"{job.name or job.id} @ {job.current_step} steps",
            progress=progress,
        )

    def _load(
        self,
        path: str,
        load_in_4bit: bool,
        chat_template: Optional[str],
        max_seq: int,
        *,
        label: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        self._assert_gpu_free()
        with self._lock:
            if self._engine is not None:
                self._engine.unload()
            self._engine = InferenceEngine(
                path,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq,
                chat_template=chat_template,
                hf_token=self.settings.hf_token(),
            )
            self._engine.load(progress=progress)
            self._label = label
        log.info("Loaded for inference: %s", label)
        return label

    # ---------------------------------------------------------- generation
    def chat(self, messages: Messages, params: Optional[GenerationParams] = None) -> str:
        if self._engine is None or not self._engine.is_loaded:
            raise RuntimeError("No model loaded. Load a model or checkpoint first.")
        return self._engine.generate(messages, params)

    def generate_stream(self, messages: Messages, params: Optional[GenerationParams] = None) -> Iterator[str]:
        if self._engine is None or not self._engine.is_loaded:
            raise RuntimeError("No model loaded. Load a model or checkpoint first.")
        return self._engine.generate_stream(messages, params)

    def unload(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.unload()
                self._engine = None
                self._label = ""
