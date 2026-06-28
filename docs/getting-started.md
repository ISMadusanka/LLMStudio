# Getting Started

This guide takes you from a fresh clone to a fine-tuned model you can chat with.

## 1. Install

On your **GPU machine** (Linux + NVIDIA recommended):

```bash
git clone <your-fork-url> llmstudio && cd llmstudio
python -m venv .venv && source .venv/bin/activate

pip install -e .            # core (UI, data prep, registry)
pip install -e ".[train]"   # torch / transformers / trl / peft / unsloth
pip install -e ".[data]"    # optional: PDF / DOCX / XLSX parsing
```

> **Install order matters for the GPU stack.** Install a CUDA-matched PyTorch and
> Unsloth following the [official Unsloth guide](https://docs.unsloth.ai/get-started/installing-+-updating)
> before/with `.[train]`. Unsloth pins compatible `transformers`/`trl`/`peft`.

## 2. Configure (optional)

```bash
cp .env.example .env
```

Set `HF_TOKEN` if you'll use **gated** models (Llama, Gemma). Point
`LLMSTUDIO_HOME` at a roomy disk for datasets, checkpoints, and downloads.

## 3. Setup

```bash
llmstudio doctor   # verify torch, CUDA, Unsloth, GPU, HF token
llmstudio setup    # create the workspace and download the assistant model
```

`setup` downloads **only** the assistant model (Qwen2.5-Instruct). Base models
download later, on demand, when you start a run.

## 4. Launch

```bash
llmstudio ui       # → http://localhost:7860
```

## 5. The workflow in the UI

### 📤 Data
1. Upload one or more files (`csv, json, jsonl, xlsx, txt, pdf, docx`).
2. Click **Load files** to preview them.
3. Map your columns to **instruction / input / output** (or **Suggest mapping with AI**).
4. Name the dataset and click **Validate & Prepare**. Read the validation report.

### ⚙️ Configure
1. Pick your **dataset** and **base model**.
2. Click **Analyze GPU & recommend settings** — the studio chooses **LoRA or
   QLoRA** and a batch plan that fits your VRAM.
3. Tweak hyperparameters, or click **Ask the Advisor** for a tailored config.
4. Click **Save configuration**.

### 🚀 Train
1. Give the run a name and click **Start training**.
2. Watch the **live loss / learning-rate charts** and logs.
3. **Pause** to free the GPU and probe a checkpoint; **Resume** to continue.
   Checkpoints are written automatically, so a crash is recoverable.

### 💬 Inference
- Load a **finished model** or a **paused run's checkpoint** and chat with it.

### 📦 Models
- Browse everything you've trained, inspect metrics/config, and manage models.

## CLI alternative

Every status command is also available headless:

```bash
llmstudio models      # base-model catalog
llmstudio datasets    # prepared datasets
llmstudio jobs        # training jobs + state
```

Next: [data-format.md](data-format.md) · [gpu-and-quantization.md](gpu-and-quantization.md) · [configuration.md](configuration.md)
