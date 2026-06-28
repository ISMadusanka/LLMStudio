"""Recommend LoRA vs QLoRA and fitting hyperparameters for the detected GPU.

The VRAM math is a deliberately *conservative heuristic*, not a simulator — its
job is to keep beginners out of OOM territory and to pick a sane batch size. It
errs toward QLoRA / smaller batches when uncertain. Users can always override.

Rough model (per GPU, gradient checkpointing assumed ON):
    weights      ≈ params_b * (2.0 GB for bf16 | 0.55 GB for 4-bit)
    activations  ≈ params_b * seq_len * batch * 1.5e-4 GB   (checkpointed)
    adapter+opt  ≈ small, scales with LoRA rank
    overhead     ≈ ~1.5 GB CUDA context + buffers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from llmstudio.config import GpuPolicy
from llmstudio.core.gpu.detector import GpuReport
from llmstudio.core.utils.logging import get_logger

log = get_logger("gpu.recommender")

LORA = "lora"
QLORA = "qlora"

# Heuristic constants
_BF16_GB_PER_B = 2.0
_4BIT_GB_PER_B = 0.55
_ACT_COEF = 1.5e-4  # GB per (params_b * seq_len * batch) with checkpointing
_OVERHEAD_GB = 1.5
_MAX_BATCH_SEARCH = 32


@dataclass
class VramEstimate:
    mode: str
    weights_gb: float
    activations_gb: float
    adapter_optimizer_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return round(
            self.weights_gb + self.activations_gb + self.adapter_optimizer_gb + self.overhead_gb,
            2,
        )


@dataclass
class QuantRecommendation:
    mode: str  # "lora" | "qlora"
    load_in_4bit: bool
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    estimate: VramEstimate
    usable_vram_gb: float
    feasible: bool
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def headline(self) -> str:
        name = "QLoRA (4-bit)" if self.mode == QLORA else "LoRA (16-bit)"
        if not self.feasible:
            return f"⚠️ {name} — may not fit; review the warnings"
        return (
            f"✅ {name} · batch {self.per_device_batch_size}×{self.gradient_accumulation_steps} "
            f"(eff. {self.effective_batch_size}) · ~{self.estimate.total_gb:.1f} GB"
        )


def estimate_vram(
    params_b: float,
    mode: str,
    seq_len: int,
    batch_size: int,
    lora_rank: int = 16,
) -> VramEstimate:
    """Estimate peak training VRAM for the given setup (GB)."""
    weights = params_b * (_4BIT_GB_PER_B if mode == QLORA else _BF16_GB_PER_B)
    activations = params_b * seq_len * max(batch_size, 1) * _ACT_COEF
    # LoRA adapter params + 8-bit/paged optimizer state; grows mildly with rank.
    adapter_optimizer = 0.3 + (lora_rank / 16.0) * 0.4
    return VramEstimate(
        mode=mode,
        weights_gb=round(weights, 2),
        activations_gb=round(activations, 2),
        adapter_optimizer_gb=round(adapter_optimizer, 2),
        overhead_gb=_OVERHEAD_GB,
    )


def _largest_fitting_batch(
    params_b: float, mode: str, seq_len: int, usable_gb: float, lora_rank: int
) -> int:
    best = 0
    for bs in range(1, _MAX_BATCH_SEARCH + 1):
        if estimate_vram(params_b, mode, seq_len, bs, lora_rank).total_gb <= usable_gb:
            best = bs
        else:
            break
    return best


def recommend_quantization(
    params_b: float,
    gpu: GpuReport,
    policy: Optional[GpuPolicy] = None,
    *,
    desired_seq_len: int = 2048,
    target_effective_batch: int = 16,
    lora_rank: int = 16,
) -> QuantRecommendation:
    """Pick LoRA vs QLoRA and a batch/accumulation plan that fits the GPU."""
    policy = policy or GpuPolicy()
    rationale: list[str] = []
    warnings: list[str] = []

    # --- No GPU: training is impractical; recommend QLoRA defaults + warn. ---
    if not gpu.available or gpu.primary is None:
        est = estimate_vram(params_b, QLORA, desired_seq_len, 1, lora_rank)
        warnings.append(
            "No CUDA GPU detected. Fine-tuning on CPU is not practical — run this "
            "on the GPU machine. Showing QLoRA defaults for planning only."
        )
        return QuantRecommendation(
            mode=QLORA,
            load_in_4bit=True,
            per_device_batch_size=1,
            gradient_accumulation_steps=target_effective_batch,
            max_seq_length=desired_seq_len,
            estimate=est,
            usable_vram_gb=0.0,
            feasible=False,
            rationale=["No GPU detected; defaulting to the most memory-efficient mode."],
            warnings=warnings,
        )

    free_gb = gpu.primary.free_gb
    usable_gb = free_gb * policy.vram_safety_factor
    rationale.append(
        f"Primary GPU: {gpu.primary.name} with {free_gb:.1f} GB free "
        f"(using {policy.vram_safety_factor:.0%} = {usable_gb:.1f} GB for the fit check)."
    )

    lora_min = estimate_vram(params_b, LORA, desired_seq_len, 1, lora_rank).total_gb
    qlora_min = estimate_vram(params_b, QLORA, desired_seq_len, 1, lora_rank).total_gb

    # Decide the mode.
    if free_gb >= policy.qlora_threshold_gb and usable_gb >= lora_min:
        mode = LORA
        rationale.append(
            f"≥ {policy.qlora_threshold_gb:.0f} GB free and LoRA fits "
            f"(needs ~{lora_min:.1f} GB at batch 1) → full-precision LoRA for best quality/speed."
        )
    elif usable_gb >= qlora_min:
        mode = QLORA
        if free_gb < policy.qlora_threshold_gb:
            rationale.append(
                f"< {policy.qlora_threshold_gb:.0f} GB free → QLoRA (4-bit) to save memory."
            )
        else:
            rationale.append(
                f"LoRA needs ~{lora_min:.1f} GB which exceeds the budget → QLoRA (4-bit)."
            )
    else:
        mode = QLORA
        warnings.append(
            f"Even QLoRA needs ~{qlora_min:.1f} GB at batch 1, but only ~{usable_gb:.1f} GB "
            f"is usable. Try a smaller model, a shorter max sequence length, or free up VRAM."
        )

    seq_len = desired_seq_len
    batch = _largest_fitting_batch(params_b, mode, seq_len, usable_gb, lora_rank)

    # If nothing fits at this seq_len, progressively shorten it before giving up.
    while batch == 0 and seq_len > 256:
        seq_len //= 2
        batch = _largest_fitting_batch(params_b, mode, seq_len, usable_gb, lora_rank)
        if batch > 0:
            warnings.append(
                f"Reduced max sequence length to {seq_len} so the run fits in VRAM."
            )

    feasible = batch > 0
    if not feasible:
        batch = 1
        warnings.append("Could not find a configuration that fits; using batch 1 — expect OOM risk.")

    grad_accum = max(1, math.ceil(target_effective_batch / batch))
    est = estimate_vram(params_b, mode, seq_len, batch, lora_rank)
    rationale.append(
        f"Chose batch {batch} with {grad_accum}× gradient accumulation "
        f"(effective batch {batch * grad_accum}); estimated peak ~{est.total_gb:.1f} GB."
    )

    if gpu.device_count > 1:
        warnings.append(
            f"{gpu.device_count} GPUs detected. Unsloth fine-tuning uses a single GPU; "
            f"the others will be idle (multi-GPU is on the roadmap)."
        )

    return QuantRecommendation(
        mode=mode,
        load_in_4bit=(mode == QLORA),
        per_device_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=seq_len,
        estimate=est,
        usable_vram_gb=round(usable_gb, 2),
        feasible=feasible,
        rationale=rationale,
        warnings=warnings,
    )
