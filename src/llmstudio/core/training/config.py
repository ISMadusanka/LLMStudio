"""The full training configuration — every hyperparameter the UI exposes.

Field descriptions double as UI tooltips and as context for the LLM advisor.
Defaults are good general-purpose values for instruction tuning with Unsloth.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class FinetuneMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"


class ExportFormat(str, Enum):
    LORA = "lora"  # adapter only (smallest, needs base model at inference)
    MERGED_16BIT = "merged_16bit"  # standalone fp16 model
    MERGED_4BIT = "merged_4bit"  # standalone 4-bit model
    GGUF = "gguf"  # llama.cpp / Ollama


class TrainingConfig(BaseModel):
    """All knobs for a fine-tuning run."""

    model_config = {"extra": "ignore", "validate_assignment": True}

    # -- identity / model ---------------------------------------------------
    base_model_key: str = Field(..., description="Catalog key of the base model.")
    base_repo: Optional[str] = Field(None, description="Explicit HF repo to use (overrides catalog resolution).")
    chat_template: str = Field("chatml", description="Chat template family applied to the tokenizer.")

    # -- data ---------------------------------------------------------------
    dataset_id: str = Field(..., description="ID of the prepared dataset to train on.")
    dataset_dir: Optional[str] = Field(None, description="Absolute path to the prepared dataset dir.")
    max_seq_length: int = Field(2048, ge=64, le=131072, description="Max tokens per example; longer are truncated.")
    packing: bool = Field(False, description="Pack multiple short examples into one sequence for efficiency.")
    train_on_responses_only: bool = Field(
        True, description="Mask the prompt so loss is computed only on the assistant's response."
    )

    # -- method / quantization ---------------------------------------------
    method: FinetuneMethod = Field(FinetuneMethod.QLORA, description="LoRA (16-bit base) or QLoRA (4-bit base).")
    load_in_4bit: bool = Field(True, description="Load base weights in 4-bit (QLoRA). Synced with `method`.")

    # -- LoRA adapter -------------------------------------------------------
    lora_r: int = Field(16, ge=1, le=512, description="LoRA rank. Higher = more capacity & memory.")
    lora_alpha: int = Field(16, ge=1, le=1024, description="LoRA scaling. A common rule is alpha = rank (or 2×rank).")
    lora_dropout: float = Field(0.0, ge=0.0, le=0.9, description="Dropout on LoRA layers. 0 is fastest/optimized.")
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_MODULES), description="Projection layers to adapt.")
    use_rslora: bool = Field(False, description="Rank-stabilized LoRA (helps at higher ranks).")
    use_gradient_checkpointing: Union[bool, str] = Field("unsloth", description="'unsloth' uses the memory-efficient kernel.")

    # -- optimization -------------------------------------------------------
    num_train_epochs: float = Field(1.0, gt=0, le=100, description="Passes over the dataset (ignored if max_steps > 0).")
    max_steps: int = Field(-1, ge=-1, description="Hard cap on optimizer steps. -1 = use epochs.")
    per_device_train_batch_size: int = Field(2, ge=1, le=256, description="Examples per step per GPU.")
    gradient_accumulation_steps: int = Field(4, ge=1, le=256, description="Accumulate N micro-batches before an update.")
    learning_rate: float = Field(2e-4, gt=0, le=1.0, description="Peak learning rate. 1e-4–3e-4 typical for LoRA.")
    lr_scheduler_type: str = Field("linear", description="linear | cosine | constant | constant_with_warmup | cosine_with_restarts")
    warmup_ratio: float = Field(0.05, ge=0.0, le=0.5, description="Fraction of steps to warm up over.")
    warmup_steps: int = Field(0, ge=0, description="Absolute warmup steps (overrides ratio if > 0).")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="L2 regularization.")
    optim: str = Field("adamw_8bit", description="Optimizer. adamw_8bit saves memory.")
    max_grad_norm: float = Field(1.0, gt=0, description="Gradient clipping threshold.")
    neftune_noise_alpha: Optional[float] = Field(None, description="NEFTune embedding noise; 5 can help small datasets.")
    group_by_length: bool = Field(False, description="Batch similar-length examples to reduce padding.")
    seed: int = Field(3407, description="Global RNG seed for reproducibility.")

    # -- precision ----------------------------------------------------------
    fp16: Optional[bool] = Field(None, description="Force fp16 (None = auto-detect).")
    bf16: Optional[bool] = Field(None, description="Force bf16 (None = auto-detect; preferred on Ampere+).")

    # -- logging / eval / checkpointing ------------------------------------
    logging_steps: int = Field(1, ge=1, description="Emit a metric every N steps.")
    eval_strategy: str = Field("steps", description="no | steps | epoch")
    eval_steps: int = Field(50, ge=1, description="Evaluate every N steps (when eval_strategy='steps').")
    save_steps: int = Field(50, ge=1, description="Write a checkpoint every N steps.")
    save_total_limit: int = Field(3, ge=1, description="Keep only the N most recent checkpoints.")

    # -- output -------------------------------------------------------------
    export_format: ExportFormat = Field(ExportFormat.LORA, description="What to save when training finishes.")
    gguf_quantization: str = Field("q4_k_m", description="GGUF quant method (export_format=gguf).")
    dataset_num_proc: int = Field(2, ge=1, le=32, description="Worker processes for dataset tokenization.")

    # ----------------------------------------------------------------------
    @model_validator(mode="after")
    def _sync_method_and_quant(self) -> "TrainingConfig":
        # Keep method and load_in_4bit consistent.
        object.__setattr__(self, "load_in_4bit", self.method == FinetuneMethod.QLORA)
        if self.lora_alpha <= 0:
            object.__setattr__(self, "lora_alpha", self.lora_r)
        return self

    # -- helpers ------------------------------------------------------------
    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def estimate_total_steps(self, n_train_examples: int) -> int:
        """Estimate optimizer steps for progress reporting."""
        if self.max_steps and self.max_steps > 0:
            return self.max_steps
        per_epoch = max(1, math.ceil(n_train_examples / max(1, self.effective_batch_size)))
        return max(1, int(per_epoch * self.num_train_epochs))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls.model_validate(d)

    @classmethod
    def from_recommendation(
        cls,
        *,
        base_model_key: str,
        dataset_id: str,
        recommendation,  # gpu.recommender.QuantRecommendation
        chat_template: str = "chatml",
        defaults: Optional[dict[str, Any]] = None,
    ) -> "TrainingConfig":
        """Build a config seeded from a GPU quantization recommendation."""
        from llmstudio.core.gpu.recommender import QLORA

        defaults = defaults or {}
        method = FinetuneMethod.QLORA if recommendation.mode == QLORA else FinetuneMethod.LORA
        return cls(
            base_model_key=base_model_key,
            dataset_id=dataset_id,
            chat_template=chat_template,
            method=method,
            max_seq_length=recommendation.max_seq_length,
            per_device_train_batch_size=recommendation.per_device_batch_size,
            gradient_accumulation_steps=recommendation.gradient_accumulation_steps,
            **defaults,
        )
