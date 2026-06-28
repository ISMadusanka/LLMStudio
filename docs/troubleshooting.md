# Troubleshooting

Run `llmstudio doctor` first — it checks PyTorch, CUDA, Unsloth, GPU, HF token,
and the workspace.

## Install / import

**`Training stack not installed` / ImportError for torch/unsloth**
Install the GPU stack: `pip install -e ".[train]"`. Install a CUDA-matched
PyTorch and Unsloth per the [Unsloth guide](https://docs.unsloth.ai) first.

**`Reading PDF/DOCX/XLSX needs …`**
`pip install -e ".[data]"`.

**Gradio errors about `Timer` / component args**
Use Gradio ≥ 4.44 (`pip install -U 'gradio<6'`). Live auto-refresh needs
`gr.Timer` (4.39+); without it, use the **🔄** button on the Train tab.

## Models / downloads

**Gated model error (Llama, Gemma)**
Accept the license on the model's Hugging Face page, then set `HF_TOKEN` in `.env`
and retry. The downloader will also try open Unsloth mirrors automatically.

**Download is slow / fills the disk**
Point `HF_HOME` and `LLMSTUDIO_HOME` at a large, fast disk. Base models download
once and are cached; only the selected model is fetched.

**A pre-quantized repo 404s**
The downloader falls back to the official repo with on-the-fly 4-bit quantization
(`download.allow_official_fallback`).

## Training

**CUDA out of memory (OOM)**
- Lower `max_seq_length` (biggest lever).
- Lower `per_device_train_batch_size` and raise `gradient_accumulation_steps`.
- Use **QLoRA** instead of LoRA.
- Keep `gradient_checkpointing: unsloth` and `optim: adamw_8bit`.
- Close other GPU processes; the assistant is auto-unloaded before training.

**Loss is NaN or not decreasing**
- Lower the learning rate (try `1e-4`).
- Ensure outputs are non-empty and consistent (check the validation report).
- For tiny datasets, increase epochs and enable NEFTune (α ≈ 5).

**Eval loss rising while train loss falls (overfitting)**
- Fewer epochs, more data, or a lower LoRA rank.

**Training won't start — "Another job is using the GPU"**
Only one run trains at a time. Wait, pause, or cancel the active job.

## Pause / resume / crash recovery

**Resume isn't offered after a crash**
Resume needs at least one checkpoint. Lower `save_steps` so checkpoints appear
sooner. On restart, interrupted runs with a checkpoint become **Resumable**.

**Inference says "pause the run first"**
Inference and training share the GPU. Pause the run, then load its checkpoint.

## Where are my files?
Everything lives under `LLMSTUDIO_HOME` (default `./workspace`): `datasets/`,
`runs/<job_id>/checkpoints/`, `models/<model_id>/`, and `registry.sqlite`.
