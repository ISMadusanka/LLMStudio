"""Services layer — the orchestration façade the UI (and CLI) build on.

Construct one :class:`Studio` for the process. It wires the core components
together, crucially registering ``assistant.unload`` as the job manager's
``on_before_train`` hook so the advisor model is evicted from VRAM the moment a
real fine-tune begins.
"""

from __future__ import annotations

import threading
from typing import Optional

from llmstudio.config import Settings, get_settings
from llmstudio.core.assistant.data_assistant import DataAssistant
from llmstudio.core.assistant.hyperparam_advisor import HyperparameterAdvisor
from llmstudio.core.assistant.llm import AssistantLLM
from llmstudio.core.models.catalog import ModelCatalog
from llmstudio.core.models.registry import ModelRegistry
from llmstudio.core.training.job import JobStore
from llmstudio.core.training.manager import JobManager
from llmstudio.core.utils.db import get_database
from llmstudio.core.utils.logging import get_logger
from llmstudio.services.assistant_service import AssistantService
from llmstudio.services.data_service import DataService
from llmstudio.services.inference_service import InferenceService
from llmstudio.services.registry_service import RegistryService
from llmstudio.services.system_service import SystemService
from llmstudio.services.training_service import TrainingService

log = get_logger("services.studio")

__all__ = [
    "AssistantService",
    "DataService",
    "InferenceService",
    "RegistryService",
    "Studio",
    "SystemService",
    "TrainingService",
    "get_studio",
    "reset_studio",
]


class Studio:
    """Aggregates every service and shares heavy singletons (assistant, jobs)."""

    def __init__(self, settings: Optional[Settings] = None, *, recover: bool = True) -> None:
        self.settings = (settings or get_settings()).ensure_dirs()
        self.catalog = ModelCatalog.load()
        self.db = get_database()

        # Shared assistant (one model, used by both advisors + chat).
        token = self.settings.hf_token()
        self.assistant = AssistantLLM(self.settings.assistant, hf_token=token)
        self.advisor = HyperparameterAdvisor(self.assistant)
        self.data_assistant = DataAssistant(self.assistant)

        # Persistence + job orchestration.
        self.registry = ModelRegistry(self.db)
        self.job_store = JobStore(self.db)
        self.jobs = JobManager(
            self.settings,
            catalog=self.catalog,
            store=self.job_store,
            registry=self.registry,
            on_before_train=self.assistant.unload,  # <-- free VRAM before training
        )
        self.bus = self.jobs.bus

        # Services consumed by the UI.
        self.data = DataService(self.settings, self.data_assistant)
        self.training = TrainingService(self.settings, self.jobs, self.catalog, self.advisor, self.data)
        self.inference = InferenceService(self.settings, self.registry, self.jobs)
        self.models = RegistryService(self.registry, self.settings)
        self.assistant_chat = AssistantService(self.assistant)
        self.system = SystemService(self.settings, self.catalog, self.assistant)

        if recover:
            self.jobs.recover()
        log.info("Studio ready (workspace: %s)", self.settings.home_root)

    def shutdown(self) -> None:
        """Release GPU-resident models. Call on app exit."""
        try:
            self.inference.unload()
        finally:
            self.assistant.unload()


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_STUDIO: Optional[Studio] = None
_LOCK = threading.Lock()


def get_studio() -> Studio:
    global _STUDIO
    if _STUDIO is None:
        with _LOCK:
            if _STUDIO is None:
                _STUDIO = Studio()
    return _STUDIO


def reset_studio() -> None:
    global _STUDIO
    with _LOCK:
        if _STUDIO is not None:
            _STUDIO.shutdown()
        _STUDIO = None
