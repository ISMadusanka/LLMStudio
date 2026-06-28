<div align="center">

# 🧪 LLM Studio

**A no-code studio for fine-tuning open-source LLMs — upload data, click train, ship a model.**

Fine-tune Llama, Qwen, Gemma, Mistral & Phi with **LoRA/QLoRA** (powered by [Unsloth](https://github.com/unslothai/unsloth)),
guided end-to-end by a built-in **LLM assistant** — all from a friendly web GUI.

</div>

---

## Why LLM Studio?

Fine-tuning an LLM normally means wrangling CUDA, `transformers`, `peft`, dataset
formats, chat templates, and a dozen hyperparameters. **LLM Studio collapses that
into a guided workflow** that a non-technical user can drive:

> **Upload your data → let the studio structure & validate it → pick a base model →
> get AI-recommended hyperparameters → train with live charts → test checkpoints →
> save and reuse your model.**

It is built for **organizations that want private, in-house fine-tuned models** on
their own GPU box, with their own data, without writing training code.

---

## ✨ Features

| Stage | What you get |
|------|---------------|
| **1. Data upload** | Drop in `csv`, `json`, `jsonl`, `txt`, `xlsx`, `pdf`, `docx`. No schema required up front. |
| **2. Structure & validate** | Map your columns to an **instruction / input / output** schema, auto-detect format, and get a full **validation report** (empty rows, length outliers, duplicates, token stats) before training. An LLM assistant can **infer the mapping for you**. |
| **3. Configure** | Choose a base model from a curated **catalog dropdown**. Tune **every hyperparameter** (LR, epochs, batch size, LoRA rank/alpha/dropout, scheduler, warmup, weight decay, packing, …). A one-click **AI Advisor** recommends a strong starting config for your data + GPU. |
| **GPU-aware** | Detects your GPUs and **recommends LoRA vs QLoRA** automatically, with batch-size and sequence-length suggestions that fit your VRAM. |
| **4. Train** | One-click training on Unsloth. **Live loss / learning-rate / throughput charts**, streaming logs, and **automatic checkpointing** so a crashed server can **resume** from the last checkpoint. |
| **5. Pause & probe** | **Pause** a run and **run inference on the latest checkpoint** to see how it's doing — then resume or stop. |
| **6. Registry & inference** | Every finished model is saved to a **registry** with its config, metrics, and dataset lineage. Reload any model later and **chat with it** or export it (LoRA adapter, merged 16-bit, or GGUF). |
| **🤖 Built-in assistant** | A local **Qwen2.5-Instruct** model guides hyperparameter selection and data prep. It is **automatically unloaded from VRAM before training starts** so it never competes with your fine-tune for memory. |
| **🔒 Local & private** | Runs entirely on your hardware. Models download on demand; your data never leaves the box. |

---

## 🏗️ Architecture

LLM Studio is layered so the heavy ML stack stays isolated and the UI is swappable.

```
            ┌─────────────────────────────────────────────────────┐
            │                  Gradio Web UI (ui/)                 │
            │  Home · Data · Configure · Train · Inference · Models│
            └───────────────────────────┬─────────────────────────┘
                                         │ calls
            ┌───────────────────────────▼─────────────────────────┐
            │                Services layer (services/)            │
            │  data · training · inference · assistant · registry  │
            │  (orchestration, resource release, event streaming)  │
            └───────────────────────────┬─────────────────────────┘
                                         │ uses
   ┌─────────────────────────────────────▼──────────────────────────────────────┐
   │                                Core (core/)                                  │
   │                                                                              │
   │  data/      gpu/         models/        training/      inference/  assistant/│
   │  loaders    detector     catalog        config         engine      llm       │
   │  schema     recommender  registry       engine(unsloth)            advisor   │
   │  validator               downloader     job + manager              data-prep │
   │  formatter                              callbacks                            │
   │  preparer                               (checkpoint/resume/pause)            │
   └──────────────────────────────────────────────────────────────────────────────┘
                 │                                   │
        torch / transformers / trl / peft     SQLite registry + filesystem
        / bitsandbytes / unsloth  (lazy)       (jobs, models, checkpoints)
```

The **core** library has no UI dependencies and lazily imports torch/unsloth, so
you can `import llmstudio` and browse the catalog or validate data on a laptop
with no GPU. See [`docs/architecture.md`](docs/architecture.md).

---

## 🚀 Quickstart (on your GPU box)

> **Prereqs:** Linux + NVIDIA GPU (recommended), CUDA-matched PyTorch, Python 3.10–3.12.

```bash
# 1) Clone & create an environment
git clone <your-fork-url> llmstudio && cd llmstudio
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)

# 2) Install core, then the GPU training stack
pip install -e .
#   Install a CUDA-matched torch + Unsloth per https://docs.unsloth.ai
pip install -e ".[train]"
#   (optional) richer document parsing: PDF / DOCX / XLSX
pip install -e ".[data]"

# 3) One-time setup: environment check + download the in-app assistant model
llmstudio setup

# 4) Launch the studio
llmstudio ui
#   → open http://localhost:7860
```

Set `HF_TOKEN` (in `.env` or your shell) before using **gated** models such as
Llama or Gemma. Copy [`.env.example`](.env.example) → `.env` to customize paths,
ports, and the assistant model.

### CLI cheatsheet

```bash
llmstudio setup      # check env, detect GPU, download assistant model
llmstudio doctor     # diagnose the environment (GPU, deps, paths, HF auth)
llmstudio ui         # launch the web studio
llmstudio models     # list the base-model catalog
llmstudio jobs       # list training jobs (and their resumable state)
llmstudio version
```

---

## 🧭 The 6-step workflow

1. **Upload** raw data on the **Data** tab. Preview it instantly.
2. **Map & validate** — pick which fields are the *instruction*, optional *input*,
   and *output* (or let the **AI assistant infer it**). Fix any issues the
   validator flags, then **Prepare dataset** (formats to the model's chat template
   and splits train/eval).
3. **Configure** on the **Configure** tab — choose a base model, review the
   **GPU recommendation (LoRA/QLoRA)**, and either set hyperparameters yourself or
   click **Ask the Advisor** for a tailored starting config.
4. **Train** on the **Train** tab. The base model downloads (once), the assistant
   is **unloaded to free VRAM**, and training starts with **live charts**.
   Checkpoints are written automatically.
5. **Pause & probe** anytime — pause the run and **chat with the latest checkpoint**
   on the **Inference** tab, then resume or finish.
6. **Reuse** — finished models land in the **Models** registry with full metadata.
   Reload, chat, compare, or export them whenever you like.

See [`docs/getting-started.md`](docs/getting-started.md) for a guided walkthrough
and [`docs/data-format.md`](docs/data-format.md) for accepted data shapes.

---

## 📦 Supported base models

Curated in [`config/models.yaml`](config/models.yaml) and shown as a dropdown in
the UI. Out of the box:

- **Llama 3.2** 1B / 3B Instruct, **Llama 3.1** 8B Instruct
- **Qwen2.5** 0.5B / 1.5B / 3B / 7B / 14B Instruct (+ **Coder 7B**)
- **Gemma 2** 2B / 9B Instruct
- **Mistral 7B Instruct v0.3**
- **Phi-3.5 Mini Instruct**
- **TinyLlama 1.1B** (pipeline testing)

Add your own by editing `config/models.yaml` — no code changes needed.

---

## 🧮 LoRA vs QLoRA — chosen for you

LLM Studio detects your GPU's free VRAM, estimates the model's footprint, and
recommends:

- **LoRA** (16-bit base) when you have headroom — fastest, highest quality.
- **QLoRA** (4-bit base) when VRAM is tight — trains big models on small GPUs.

You can always override the recommendation. Details and the VRAM math live in
[`docs/gpu-and-quantization.md`](docs/gpu-and-quantization.md).

---

## 🗂️ Project layout

```
llmstudio/
├── config/                  # default.yaml (settings) + models.yaml (catalog)
├── docs/                    # architecture, getting-started, data-format, GPU guide…
├── src/llmstudio/
│   ├── config/              # pydantic-settings, path management
│   ├── core/
│   │   ├── gpu/             # VRAM detection + LoRA/QLoRA recommender
│   │   ├── data/            # loaders · schema · validator · formatter · preparer
│   │   ├── models/          # catalog · registry (SQLite) · on-demand downloader
│   │   ├── training/        # config · unsloth engine · callbacks · job · manager
│   │   ├── inference/       # checkpoint/model loading + generation
│   │   ├── assistant/       # Qwen advisor (load/unload, hyperparams, data prep)
│   │   └── utils/           # logging · resource release · db · event bus
│   ├── services/            # orchestration the UI calls into
│   ├── ui/                  # Gradio app: app.py + pages/ + theme
│   └── cli.py               # `llmstudio` command
├── tests/
├── pyproject.toml
└── requirements*.txt
```

Each major package carries its own `README.md` explaining its responsibilities.

---

## 🛟 Resilience: checkpoints, pause/resume, crash recovery

- Checkpoints are written every `save_steps` and the **N most recent are kept**.
- **Pause** cleanly stops after the next checkpoint; **Resume** continues from it.
- If the server dies mid-run, the job is persisted as *resumable* — on restart the
  studio offers to **continue from the last checkpoint** (`resume_from_checkpoint`).

---

## ⚙️ Configuration

All settings have sane defaults in [`config/default.yaml`](config/default.yaml) and
can be overridden via environment variables (`LLMSTUDIO_…`) or a user
`config.yaml` under your `LLMSTUDIO_HOME`. Full reference:
[`docs/configuration.md`](docs/configuration.md).

---

## 🧪 Status & roadmap

This is an actively evolving framework. Implemented: guided data prep, GPU-aware
LoRA/QLoRA, Unsloth training with checkpoint/pause/resume, AI hyperparameter
advisor, model registry + inference, Gradio UI. On the roadmap: multi-GPU /
distributed training, DPO/ORPO preference tuning, experiment comparison dashboards,
scheduled jobs, and a REST API.

> ⚠️ **Licensing:** downloaded models carry their own licenses (Llama, Gemma terms,
> Apache-2.0, MIT, …). Review and accept them on Hugging Face before fine-tuning or
> redistributing derived weights.

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE).
