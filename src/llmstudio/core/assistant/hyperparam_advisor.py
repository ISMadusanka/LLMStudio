"""Recommend a starting hyperparameter configuration.

Uses the LLM assistant when available, and always has a deterministic heuristic
fallback so the feature works even with the assistant disabled or unavailable.
The output is a whitelist of :class:`TrainingConfig` field updates plus a
rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from llmstudio.core.assistant.llm import AssistantLLM
from llmstudio.core.assistant.prompts import (
    HYPERPARAM_KEYS,
    HYPERPARAM_SYSTEM,
    build_hyperparam_prompt,
    extract_json,
)
from llmstudio.core.utils.logging import get_logger

log = get_logger("assistant.advisor")

_ALLOWED = set(HYPERPARAM_KEYS)


@dataclass
class AdvisorContext:
    model_name: str
    params_b: float
    n_train: int
    n_eval: int = 0
    task_format: str = "instruction"
    mode: str = "qlora"
    load_in_4bit: bool = True
    median_tokens: int = 0
    p95_tokens: int = 0
    max_tokens: int = 0
    gpu_free_gb: float = 0.0
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class HyperparamAdvice:
    updates: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = "heuristic"  # "llm" | "heuristic"


class HyperparameterAdvisor:
    def __init__(self, llm: Optional[AssistantLLM] = None) -> None:
        self.llm = llm

    def advise(self, ctx: AdvisorContext) -> HyperparamAdvice:
        if self.llm is not None and self.llm.available():
            try:
                advice = self._advise_llm(ctx)
                if advice.updates:
                    return advice
            except Exception as exc:  # pragma: no cover - fall back gracefully
                log.warning("LLM advisor failed (%s); using heuristic.", exc)
        return self._advise_heuristic(ctx)

    # ------------------------------------------------------------- backends
    def _advise_llm(self, ctx: AdvisorContext) -> HyperparamAdvice:
        prompt = build_hyperparam_prompt(ctx.as_dict())
        raw = self.llm.complete(HYPERPARAM_SYSTEM, prompt, max_new_tokens=512, temperature=0.3)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("advisor did not return a JSON object")
        rationale = str(data.pop("rationale", "")).strip()
        updates = self._sanitize({k: v for k, v in data.items() if k in _ALLOWED})
        # Blend in heuristic defaults for anything the model omitted.
        base = self._advise_heuristic(ctx)
        merged = {**base.updates, **updates}
        return HyperparamAdvice(updates=merged, rationale=rationale or base.rationale, source="llm")

    def _advise_heuristic(self, ctx: AdvisorContext) -> HyperparamAdvice:
        n = max(1, ctx.n_train)
        # Epochs: smaller datasets benefit from more passes.
        if n < 200:
            epochs, neftune = 4.0, 5.0
        elif n < 1000:
            epochs, neftune = 3.0, 5.0
        elif n < 10000:
            epochs, neftune = 2.0, None
        else:
            epochs, neftune = 1.0, None

        # LoRA rank scales mildly with data volume.
        if n < 1000:
            rank = 16
        elif n < 20000:
            rank = 32
        else:
            rank = 64

        updates: dict[str, Any] = {
            "num_train_epochs": epochs,
            "learning_rate": 2e-4,
            "lora_r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
        }
        if neftune is not None:
            updates["neftune_noise_alpha"] = neftune

        rationale = (
            f"{n} training examples → {epochs:g} epoch(s) and LoRA rank {rank}. "
            f"Learning rate 2e-4 with 5% warmup is a robust LoRA default. "
            + (f"Enabled NEFTune (α={neftune:g}) to regularize a small dataset. " if neftune else "")
            + "Adjust epochs down if you see the eval loss rising (overfitting)."
        )
        return HyperparamAdvice(updates=self._sanitize(updates), rationale=rationale, source="heuristic")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _sanitize(updates: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        numeric_int = {"lora_r", "lora_alpha", "per_device_train_batch_size",
                       "gradient_accumulation_steps", "max_seq_length"}
        numeric_float = {"num_train_epochs", "learning_rate", "lora_dropout",
                         "warmup_ratio", "weight_decay", "neftune_noise_alpha"}
        for key, value in updates.items():
            try:
                if key in numeric_int:
                    clean[key] = int(value)
                elif key in numeric_float:
                    clean[key] = float(value)
                else:
                    clean[key] = value
            except (TypeError, ValueError):
                continue
        # Clamp obviously bad values.
        if "learning_rate" in clean:
            clean["learning_rate"] = min(max(clean["learning_rate"], 1e-6), 1e-2)
        if "lora_dropout" in clean:
            clean["lora_dropout"] = min(max(clean["lora_dropout"], 0.0), 0.9)
        return clean
