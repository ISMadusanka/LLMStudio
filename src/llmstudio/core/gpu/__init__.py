"""GPU detection and quantization (LoRA vs QLoRA) recommendation."""

from llmstudio.core.gpu.detector import GpuDevice, GpuReport, detect_gpus
from llmstudio.core.gpu.recommender import (
    LORA,
    QLORA,
    QuantRecommendation,
    VramEstimate,
    estimate_vram,
    recommend_quantization,
)

__all__ = [
    "LORA",
    "QLORA",
    "GpuDevice",
    "GpuReport",
    "QuantRecommendation",
    "VramEstimate",
    "detect_gpus",
    "estimate_vram",
    "recommend_quantization",
]
