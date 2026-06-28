"""Tests for TrainingConfig validation and recommendation seeding."""

from __future__ import annotations

from llmstudio.core.gpu.detector import GpuDevice, GpuReport
from llmstudio.core.gpu.recommender import recommend_quantization
from llmstudio.core.training.config import FinetuneMethod, TrainingConfig


def test_method_syncs_load_in_4bit():
    qlora = TrainingConfig(base_model_key="x", dataset_id="d", method=FinetuneMethod.QLORA)
    assert qlora.load_in_4bit is True
    lora = TrainingConfig(base_model_key="x", dataset_id="d", method=FinetuneMethod.LORA)
    assert lora.load_in_4bit is False


def test_effective_batch_and_steps():
    cfg = TrainingConfig(
        base_model_key="x", dataset_id="d",
        per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=2,
    )
    assert cfg.effective_batch_size == 16
    # 1000 examples / 16 = 63 steps/epoch * 2 epochs
    assert cfg.estimate_total_steps(1000) == 126


def test_roundtrip_dict():
    cfg = TrainingConfig(base_model_key="m", dataset_id="d", lora_r=32, learning_rate=1e-4)
    again = TrainingConfig.from_dict(cfg.to_dict())
    assert again.lora_r == 32
    assert again.learning_rate == 1e-4


def test_from_recommendation():
    report = GpuReport(available=True, source="test",
                       devices=[GpuDevice(0, "g", 24.0, 24.0, 0.0)])
    rec = recommend_quantization(3.0, report, desired_seq_len=1024)
    cfg = TrainingConfig.from_recommendation(
        base_model_key="qwen2.5-3b-instruct", dataset_id="d", recommendation=rec, chat_template="qwen-2.5"
    )
    assert cfg.base_model_key == "qwen2.5-3b-instruct"
    assert cfg.max_seq_length == rec.max_seq_length
    assert cfg.chat_template == "qwen-2.5"
