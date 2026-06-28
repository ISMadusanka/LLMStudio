"""Job manager: schedules training runs and exposes pause / resume / cancel.

Runs each job in a background thread, serialized by a global lock (single-GPU
assumption). Persists enough state that an interrupted run becomes *resumable*
after a restart. Calls an optional ``on_before_train`` hook right before the
model loads — the services layer wires this to unload the LLM assistant so it
never competes with the fine-tune for VRAM.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from llmstudio.config import Settings, get_settings
from llmstudio.core.models.catalog import ModelCatalog
from llmstudio.core.models.downloader import download_model, download_repo
from llmstudio.core.models.registry import ModelRegistry, make_model_id
from llmstudio.core.training.config import ExportFormat, FinetuneMethod, TrainingConfig
from llmstudio.core.training.engine import UnslothEngine
from llmstudio.core.training.job import (
    ACTIVE_STATUSES,
    Job,
    JobControl,
    JobStatus,
    JobStore,
    make_job_id,
)
from llmstudio.core.utils.events import EventBus, default_bus
from llmstudio.core.utils.logging import get_logger
from llmstudio.core.utils.resources import free_gpu_memory

log = get_logger("training.manager")

# Map artifact/export format to the registry's artifact kind.
_EXPORT_TO_KIND = {
    ExportFormat.LORA.value: "lora",
    ExportFormat.MERGED_16BIT.value: "merged_16bit",
    ExportFormat.MERGED_4BIT.value: "merged_4bit",
    ExportFormat.GGUF.value: "gguf",
}


class JobManager:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        bus: Optional[EventBus] = None,
        catalog: Optional[ModelCatalog] = None,
        store: Optional[JobStore] = None,
        registry: Optional[ModelRegistry] = None,
        on_before_train: Optional[Callable[[], None]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or default_bus()
        self.catalog = catalog or ModelCatalog.load()
        self.store = store or JobStore()
        self.registry = registry or ModelRegistry()
        self.on_before_train = on_before_train

        self._controls: dict[str, JobControl] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._train_lock = threading.Lock()  # serialize GPU usage
        self._lock = threading.Lock()

    # --------------------------------------------------------------- public
    def submit(self, config: TrainingConfig, name: str) -> Job:
        job_id = make_job_id()
        run_dir = self.settings.resolved_paths.run_dir(job_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        job = self.store.create(
            job_id=job_id,  # MUST match the id the worker thread looks up
            config=config.to_dict(),
            name=name,
            run_dir=str(run_dir),
            base_model_key=config.base_model_key,
            dataset_id=config.dataset_id,
        )
        log.info("Submitted job %s ('%s')", job_id, name)
        self._start_thread(job_id, resume_from=None)
        return job

    def resume(self, job_id: str) -> Optional[Job]:
        job = self.store.get(job_id)
        if not job or not job.can_resume:
            return job
        self.store.set_status(job_id, JobStatus.PENDING)
        self._start_thread(job_id, resume_from=job.last_checkpoint)
        return self.store.get(job_id)

    def pause(self, job_id: str) -> Optional[Job]:
        control = self._controls.get(job_id)
        job = self.store.get(job_id)
        if job and job.status == JobStatus.RUNNING and control:
            control.request_pause()
            self.store.set_status(job_id, JobStatus.PAUSING)
            self.bus.log(job_id, "Pause requested.")
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        control = self._controls.get(job_id)
        job = self.store.get(job_id)
        if not job:
            return None
        if job.is_active and control:
            control.request_stop()
            self.bus.log(job_id, "Cancellation requested.")
        elif job.status in {JobStatus.PENDING, JobStatus.PAUSED, JobStatus.RESUMABLE}:
            self.store.set_status(job_id, JobStatus.CANCELLED)
        return self.store.get(job_id)

    def get(self, job_id: str) -> Optional[Job]:
        return self.store.get(job_id)

    def list(self) -> list[Job]:
        return self.store.list()

    def recover(self) -> list[Job]:
        """On startup, reconcile jobs that were active when the process died."""
        recovered: list[Job] = []
        for job in self.store.list():
            if job.status in ACTIVE_STATUSES:
                if job.last_checkpoint and Path(job.last_checkpoint).exists():
                    self.store.update(job.id, status=JobStatus.RESUMABLE, resumable=True)
                    self.bus.log(job.id, "Recovered as resumable after restart.")
                else:
                    self.store.set_status(job.id, JobStatus.FAILED, error="Interrupted before first checkpoint.")
                recovered.append(self.store.get(job.id))  # type: ignore[arg-type]
        if recovered:
            log.info("Recovered %d interrupted job(s).", len(recovered))
        return recovered

    # -------------------------------------------------------------- internal
    def _start_thread(self, job_id: str, resume_from: Optional[str]) -> None:
        with self._lock:
            self._controls[job_id] = JobControl()
            t = threading.Thread(target=self._run, args=(job_id, resume_from), name=f"train-{job_id}", daemon=True)
            self._threads[job_id] = t
            t.start()

    def _run(self, job_id: str, resume_from: Optional[str]) -> None:
        log.info("[%s] Worker thread started (resume=%s).", job_id, bool(resume_from))
        # Serialize on the GPU; queued jobs wait here as PENDING.
        acquired = self._train_lock.acquire(blocking=False)
        if not acquired:
            log.info("[%s] Another job holds the GPU — queued.", job_id)
            self.bus.log(job_id, "Another job is using the GPU — queued.")
            self._train_lock.acquire()
        try:
            self._run_locked(job_id, resume_from)
        finally:
            self._train_lock.release()
            free_gpu_memory()
            log.info("[%s] Worker thread finished.", job_id)

    def _run_locked(self, job_id: str, resume_from: Optional[str]) -> None:
        job = self.store.get(job_id)
        if job is None:
            log.error("[%s] Job not found in store — aborting (this is a bug).", job_id)
            return
        if job.status == JobStatus.CANCELLED:
            log.info("[%s] Job was cancelled before it started.", job_id)
            return
        control = self._controls.get(job_id) or JobControl()
        self._controls[job_id] = control

        try:
            cfg = TrainingConfig.from_dict(job.config)
            entry = self.catalog.find(cfg.base_model_key)
            log.info("[%s] Run starting: %s on %s.", job_id, cfg.method.value, cfg.base_model_key)

            # 1) Download base model (network only; no GPU needed yet).
            self.store.set_status(job_id, JobStatus.DOWNLOADING)
            self.bus.status(job_id, JobStatus.DOWNLOADING.value, "Fetching base model…")
            log.info("[%s] Downloading base model…", job_id)
            token = self.settings.hf_token()
            # cache_dir=None → use the shared HF cache (HF_HOME, set at startup),
            # so the engine can load by repo ID from the same cache.

            def _progress(msg: str) -> None:
                self.bus.log(job_id, msg)

            if cfg.base_repo:
                dl = download_repo(cfg.base_repo, token=token, progress=_progress)
            elif entry is not None:
                dl = download_model(
                    entry,
                    load_in_4bit=cfg.load_in_4bit,
                    prefer_unsloth_4bit=self.settings.download.prefer_unsloth_4bit,
                    allow_official_fallback=self.settings.download.allow_official_fallback,
                    token=token,
                    progress=_progress,
                )
            else:
                raise RuntimeError(f"Unknown base model '{cfg.base_model_key}' and no base_repo set.")

            if control.stop_requested():
                self.store.set_status(job_id, JobStatus.CANCELLED)
                return

            # 2) Free VRAM before loading the trainable model (unload assistant).
            if self.on_before_train:
                try:
                    self.on_before_train()
                    self.bus.log(job_id, "Released assistant model to free VRAM for training.")
                except Exception as exc:  # never block training on this
                    log.warning("on_before_train hook failed: %s", exc)
            free_gpu_memory()

            # 3) Train.
            self.store.set_status(job_id, JobStatus.RUNNING)
            log.info("[%s] Base model ready; launching trainer.", job_id)
            model_id = make_model_id(job.name or cfg.base_model_key)
            export_dir = self.settings.resolved_paths.model_dir(model_id)

            engine = UnslothEngine(
                config=cfg,
                # Load by repo ID (not the local snapshot path) so transformers'
                # _is_local code path — buggy in 4.57.2 — is not triggered. The
                # weights are already in the shared HF cache from the download.
                model_repo=dl.repo_id,
                run_dir=Path(job.run_dir),
                export_dir=export_dir,
                train_path=Path(cfg.dataset_dir or self.settings.resolved_paths.dataset_dir(cfg.dataset_id)) / "train.jsonl",
                eval_path=Path(cfg.dataset_dir or self.settings.resolved_paths.dataset_dir(cfg.dataset_id)) / "eval.jsonl",
                bus=self.bus,
                store=self.store,
                control=control,
                job_id=job_id,
                hf_token=token,
            )
            result = engine.run(resume_from=resume_from)

            # 4) Finalize.
            if result.completed:
                self._register_completed(job_id, model_id, cfg, result, export_dir)
            elif result.stop_reason == "pause":
                self.store.update(job_id, status=JobStatus.PAUSED, last_checkpoint=result.last_checkpoint or "", resumable=True)
                self.bus.status(job_id, JobStatus.PAUSED.value, "Paused — resume or run inference on the checkpoint.")
            else:  # stop / cancel
                self.store.update(job_id, status=JobStatus.CANCELLED, last_checkpoint=result.last_checkpoint or "")
                self.bus.status(job_id, JobStatus.CANCELLED.value, "Cancelled.")

        except ImportError as exc:
            msg = f"Training stack not installed: {exc}. Install with: pip install 'llmstudio[train]'."
            self.store.set_status(job_id, JobStatus.FAILED, error=msg)
            self.bus.error(job_id, msg)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            log.exception("Job %s failed", job_id)
            self.store.set_status(job_id, JobStatus.FAILED, error=str(exc))
            self.bus.error(job_id, f"Training failed: {exc}")
        finally:
            control.clear()
            self._controls.pop(job_id, None)

    def _register_completed(
        self,
        job_id: str,
        model_id: str,
        cfg: TrainingConfig,
        result,
        export_dir: Path,
    ) -> None:
        job = self.store.get(job_id)
        name = (job.name if job else None) or model_id
        kind = result.export_kind or _EXPORT_TO_KIND.get(cfg.export_format.value, "lora")
        quant = "qlora" if cfg.method == FinetuneMethod.QLORA else "lora"
        record = self.registry.register(
            name=name,
            model_id=model_id,
            path=result.exported_path or str(export_dir),
            base_model_key=cfg.base_model_key,
            base_repo=cfg.base_repo or "",
            artifact_kind=kind,
            quantization=quant,
            dataset_id=cfg.dataset_id,
            job_id=job_id,
            metrics=result.final_metrics,
            config=cfg.to_dict(),
        )
        self.store.update(
            job_id,
            status=JobStatus.COMPLETED,
            registered_model_id=record.id,
            metrics=result.final_metrics,
            last_checkpoint=result.last_checkpoint or "",
            finished_at=time.time(),
        )
        self.bus.status(job_id, JobStatus.COMPLETED.value, f"Done. Saved as '{name}'.")
        self.bus.log(job_id, f"Model registered: {record.id}")
