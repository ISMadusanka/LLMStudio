"""Training service: recommendations, the AI advisor, and job control.

Sits between the UI and the core :class:`JobManager`. Builds GPU-aware default
configs, asks the advisor for hyperparameters, and exposes start/pause/resume/
cancel plus live event access.
"""

from __future__ import annotations

from typing import Any, Optional

from llmstudio.config import Settings
from llmstudio.core.assistant.hyperparam_advisor import (
    AdvisorContext,
    HyperparamAdvice,
    HyperparameterAdvisor,
)
from llmstudio.core.gpu.detector import detect_gpus
from llmstudio.core.gpu.recommender import QLORA, QuantRecommendation, recommend_quantization
from llmstudio.core.models.catalog import ModelCatalog
from llmstudio.core.training.config import FinetuneMethod, TrainingConfig
from llmstudio.core.training.job import Job, JobStatus
from llmstudio.core.training.manager import JobManager
from llmstudio.core.utils.events import EventBus
from llmstudio.core.utils.logging import get_logger
from llmstudio.services.data_service import DataService

log = get_logger("services.training")


class TrainingService:
    def __init__(
        self,
        settings: Settings,
        jobs: JobManager,
        catalog: ModelCatalog,
        advisor: HyperparameterAdvisor,
        data_service: DataService,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.catalog = catalog
        self.advisor = advisor
        self.data = data_service

    @property
    def bus(self) -> EventBus:
        return self.jobs.bus

    # ----------------------------------------------------- recommendations
    def recommend(self, model_key: str, *, desired_seq_len: Optional[int] = None) -> QuantRecommendation:
        entry = self.catalog.get(model_key)
        gpu = detect_gpus()
        seq = desired_seq_len or min(self.settings.training.max_seq_length, entry.context_length)
        return recommend_quantization(entry.params_b, gpu, self.settings.gpu, desired_seq_len=seq)

    def build_default_config(self, model_key: str, dataset_id: str) -> tuple[TrainingConfig, QuantRecommendation]:
        entry = self.catalog.get(model_key)
        rec = self.recommend(model_key)
        td = self.settings.training
        cfg = TrainingConfig.from_recommendation(
            base_model_key=model_key,
            dataset_id=dataset_id,
            recommendation=rec,
            chat_template=entry.chat_template,
            defaults=dict(
                seed=td.seed,
                save_steps=td.save_steps,
                save_total_limit=td.save_total_limit,
                logging_steps=td.logging_steps,
                eval_steps=td.save_steps,
                packing=td.packing,
                export_format=td.default_export_format,
            ),
        )
        ds = self.data.get_dataset(dataset_id)
        if ds is not None:
            cfg.dataset_dir = str(ds.directory)
        return cfg, rec

    # -------------------------------------------------------------- advisor
    def advise(self, config: TrainingConfig) -> HyperparamAdvice:
        ds = self.data.get_dataset(config.dataset_id)
        token_stats = (ds.schema.stats.get("token_len", {}) if ds else {})
        gpu = detect_gpus()
        ctx = AdvisorContext(
            model_name=config.base_model_key,
            params_b=self.catalog.get(config.base_model_key).params_b if self.catalog.find(config.base_model_key) else 0.0,
            n_train=ds.n_train if ds else 0,
            n_eval=ds.n_eval if ds else 0,
            task_format=ds.task_format.value if ds else "instruction",
            mode=QLORA if config.method == FinetuneMethod.QLORA else "lora",
            load_in_4bit=config.load_in_4bit,
            median_tokens=int(token_stats.get("median", 0)),
            p95_tokens=int(token_stats.get("p95", 0)),
            max_tokens=int(token_stats.get("max", 0)),
            gpu_free_gb=round(gpu.max_free_gb, 1) if gpu.available else 0.0,
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        return self.advisor.advise(ctx)

    @staticmethod
    def apply_advice(config: TrainingConfig, advice: HyperparamAdvice) -> TrainingConfig:
        data = config.to_dict()
        data.update(advice.updates)
        return TrainingConfig.from_dict(data)

    # --------------------------------------------------------- job control
    def start(self, config: TrainingConfig, name: str) -> Job:
        active = self.active_jobs()
        if active:
            a = active[0]
            raise RuntimeError(
                f"A job is already {a.status.value} ('{a.name or a.id}'). "
                f"Pause or let it finish before starting another (single-GPU)."
            )
        if not config.dataset_dir:
            ds = self.data.get_dataset(config.dataset_id)
            if ds is None:
                raise ValueError(f"Dataset '{config.dataset_id}' not found — prepare it first.")
            config.dataset_dir = str(ds.directory)
        log.info("Submitting job '%s' (%s on %s)", name, config.method.value, config.base_model_key)
        return self.jobs.submit(config, name=name)

    def pause(self, job_id: str) -> Optional[Job]:
        return self.jobs.pause(job_id)

    def resume(self, job_id: str) -> Optional[Job]:
        return self.jobs.resume(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        return self.jobs.cancel(job_id)

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return self.jobs.list()

    def active_jobs(self) -> list[Job]:
        return [j for j in self.jobs.list() if j.is_active]

    # ------------------------------------------------------------- events
    def metrics(self, job_id: str) -> list[dict[str, Any]]:
        return self.bus.metrics_frame(job_id)

    def log_lines(self, job_id: str, limit: int = 200) -> list[str]:
        from llmstudio.core.utils.events import LOG

        events = self.bus.history(job_id, kinds={LOG})[-limit:]
        return [f"{e.data.get('message', '')}" for e in events]
