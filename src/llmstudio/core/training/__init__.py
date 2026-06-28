"""Training subsystem: config · engine · callbacks · job state · manager.

The :class:`JobManager` is the entry point the services layer uses::

    manager = JobManager(settings, on_before_train=assistant.unload)
    manager.recover()                      # reconcile interrupted runs on boot
    job = manager.submit(config, name="support-bot")
    manager.pause(job.id); manager.resume(job.id); manager.cancel(job.id)
"""

from llmstudio.core.training.config import (
    DEFAULT_TARGET_MODULES,
    ExportFormat,
    FinetuneMethod,
    TrainingConfig,
)
from llmstudio.core.training.engine import TrainingResult, UnslothEngine, latest_checkpoint
from llmstudio.core.training.job import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    Job,
    JobControl,
    JobStatus,
    JobStore,
    make_job_id,
)
from llmstudio.core.training.manager import JobManager

__all__ = [
    "ACTIVE_STATUSES",
    "DEFAULT_TARGET_MODULES",
    "TERMINAL_STATUSES",
    "ExportFormat",
    "FinetuneMethod",
    "Job",
    "JobControl",
    "JobManager",
    "JobStatus",
    "JobStore",
    "TrainingConfig",
    "TrainingResult",
    "UnslothEngine",
    "latest_checkpoint",
    "make_job_id",
]
