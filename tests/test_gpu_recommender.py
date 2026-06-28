"""Tests for VRAM estimation and LoRA/QLoRA recommendation."""

from __future__ import annotations

from llmstudio.config import GpuPolicy
from llmstudio.core.gpu.detector import GpuDevice, GpuReport
from llmstudio.core.gpu.recommender import LORA, QLORA, estimate_vram, recommend_quantization


def _report(free_gb: float, total_gb: float = 24.0, n: int = 1) -> GpuReport:
    devices = [
        GpuDevice(index=i, name="Test GPU", total_gb=total_gb, free_gb=free_gb, used_gb=total_gb - free_gb)
        for i in range(n)
    ]
    return GpuReport(available=True, devices=devices, source="test")


def test_estimate_qlora_cheaper_than_lora():
    lora = estimate_vram(8.0, LORA, 2048, 1)
    qlora = estimate_vram(8.0, QLORA, 2048, 1)
    assert qlora.total_gb < lora.total_gb


def test_recommend_qlora_on_small_gpu():
    rec = recommend_quantization(8.0, _report(free_gb=10.0), GpuPolicy(), desired_seq_len=2048)
    assert rec.mode == QLORA
    assert rec.load_in_4bit is True
    assert rec.per_device_batch_size >= 1


def test_recommend_lora_on_big_gpu():
    rec = recommend_quantization(3.0, _report(free_gb=48.0, total_gb=48.0), GpuPolicy(), desired_seq_len=2048)
    assert rec.mode == LORA
    assert rec.load_in_4bit is False


def test_no_gpu_is_infeasible():
    rec = recommend_quantization(7.0, GpuReport(available=False), GpuPolicy())
    assert rec.feasible is False
    assert rec.mode == QLORA


def test_effective_batch_reaches_target():
    rec = recommend_quantization(3.0, _report(free_gb=24.0), GpuPolicy(), target_effective_batch=16)
    assert rec.effective_batch_size >= 16
