"""Core library — UI-agnostic building blocks.

Subpackages:
    gpu/        VRAM detection and LoRA/QLoRA recommendation
    data/       ingestion, schema, validation, formatting, dataset preparation
    models/     base-model catalog, on-demand downloader, fine-tuned registry
    training/   training config, Unsloth engine, callbacks, job + manager
    inference/  load a model/checkpoint and generate
    assistant/  local LLM (Qwen) for hyperparameter & data-prep guidance
    utils/      logging, resource release, db, event bus

Nothing here imports Gradio. Heavy ML imports (torch/transformers/unsloth) are
deferred to call time so this package imports cheaply on any machine.
"""
