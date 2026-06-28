"""The local instruct model (Qwen by default) that powers in-app guidance.

Resource-managed: it lazily loads on first use, automatically picks a smaller
fallback model on low-VRAM machines, and exposes ``unload()`` so the training
manager can reclaim VRAM the instant a real fine-tune begins.
"""

from __future__ import annotations

import threading
from typing import Optional

from llmstudio.config import AssistantConfig
from llmstudio.core.inference.engine import GenerationParams, InferenceEngine, Messages
from llmstudio.core.utils.logging import get_logger

log = get_logger("assistant.llm")


def _deps_available() -> bool:
    try:
        import torch  # noqa: F401

        try:
            import unsloth  # noqa: F401
        except Exception:
            import transformers  # noqa: F401
        return True
    except Exception:
        return False


class AssistantLLM:
    """Wraps an :class:`InferenceEngine` with assistant-specific behavior."""

    def __init__(self, config: AssistantConfig, *, hf_token: Optional[str] = None) -> None:
        self.config = config
        self.hf_token = hf_token
        self._engine: Optional[InferenceEngine] = None
        self._model_id: Optional[str] = None
        self._lock = threading.Lock()

    # --------------------------------------------------------- availability
    def available(self) -> bool:
        """True if the assistant is enabled and the runtime deps are importable."""
        return bool(self.config.enabled) and _deps_available()

    def resolve_model_id(self) -> str:
        """Pick the primary model, or the fallback on low-VRAM machines."""
        if not self.config.fallback_model_id:
            return self.config.model_id
        try:
            from llmstudio.core.gpu.detector import detect_gpus

            report = detect_gpus()
            free = report.max_free_gb if report.available else 0.0
            if free and free < self.config.min_vram_gb_for_primary:
                log.info(
                    "Free VRAM %.1f GB < %.1f GB → using fallback assistant model %s",
                    free,
                    self.config.min_vram_gb_for_primary,
                    self.config.fallback_model_id,
                )
                return self.config.fallback_model_id
        except Exception:
            pass
        return self.config.model_id

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None and self._engine.is_loaded

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    # ----------------------------------------------------------- lifecycle
    def load(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            model_id = self.resolve_model_id()
            log.info("Loading assistant model: %s", model_id)
            self._engine = InferenceEngine(
                model_id,
                load_in_4bit=self.config.load_in_4bit,
                max_seq_length=4096,
                hf_token=self.hf_token,
            )
            self._engine.load()
            self._model_id = model_id

    def unload(self) -> None:
        """Release the assistant from VRAM. Safe to call when not loaded."""
        with self._lock:
            if self._engine is not None:
                self._engine.unload()
                self._engine = None
                log.info("Assistant model unloaded.")

    # ---------------------------------------------------------- generation
    def chat(
        self,
        messages: Messages,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a reply for a list of role/content messages."""
        self.load()
        assert self._engine is not None
        temp = self.config.temperature if temperature is None else temperature
        params = GenerationParams(
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temp,
            top_p=self.config.top_p,
            do_sample=temp > 0,
        )
        reply = self._engine.generate(messages, params)
        if not self.config.keep_resident:
            # Default: don't hold VRAM between calls.
            self.unload()
        return reply

    def complete(self, system: str, user: str, **kwargs) -> str:
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
