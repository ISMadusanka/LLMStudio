# Architecture

LLM Studio is layered so the GPU-heavy code is isolated from the UI and the core
library imports cheaply even on a machine with no GPU.

```
 UI (Gradio)  ──calls──▶  Services  ──uses──▶  Core  ──▶  torch/unsloth + SQLite
 src/llmstudio/ui         services/            core/
```

## Layers

### `core/` — UI-agnostic building blocks
No Gradio imports. Heavy ML imports (`torch`, `unsloth`, `trl`, `transformers`,
`datasets`) are **deferred to call time**, so `import llmstudio.core...` works
without the training stack.

| Package | Responsibility |
|--------|----------------|
| `core/gpu` | Detect GPUs/VRAM (`pynvml`/`torch`); recommend LoRA vs QLoRA + batch plan |
| `core/data` | Load many formats → normalize to a schema → validate → split → persist |
| `core/models` | Base-model **catalog**, on-demand **downloader**, fine-tuned **registry** |
| `core/training` | `TrainingConfig`, Unsloth **engine**, **callbacks**, **job** state, **manager** |
| `core/inference` | Load a model/checkpoint and generate (streaming) |
| `core/assistant` | Local LLM (Qwen) + hyperparameter & data-prep advisors |
| `core/utils` | Logging, GPU resource release, SQLite/SQLAlchemy, event bus |

### `services/` — orchestration
Thin classes that compose core pieces into the operations the UI needs, and own
process-wide policy. The `Studio` façade builds them all and wires the critical
**resource-release hook**:

```python
JobManager(..., on_before_train=assistant.unload)
```

so the advisor model is evicted from VRAM the instant a real fine-tune starts.

### `ui/` — Gradio app
One module per tab under `ui/pages/`. Pages call only the services layer. Shared
cross-tab `gr.State` carries the saved config (Configure → Train) and the active
job id (Train → Inference).

## Key runtime flows

### Training job lifecycle
```
submit → PENDING → DOWNLOADING → (unload assistant) → RUNNING ─┬─▶ COMPLETED → registry
                                                               ├─▶ PAUSED (resumable)
                                                               └─▶ FAILED / CANCELLED
```
- The **manager** runs each job in a background thread, serialized by a global
  lock (single-GPU assumption).
- The **callback** streams metrics to the event bus, persists step/checkpoint
  progress to SQLite, and polls a `JobControl` for cooperative pause/stop.
- **Crash recovery:** on boot, `JobManager.recover()` turns interrupted runs with
  a checkpoint into `RESUMABLE`.

### Event streaming
`core/utils/events.py` is an in-process pub/sub bus. The training callback
publishes `log` / `metric` / `status` events; the Train tab polls history (and a
`gr.Timer`) to render live charts and logs.

### Persistence
- **SQLite** (`core/utils/db.py`, SQLAlchemy 2.0): the job store and model
  registry. Swappable to Postgres via a connection URL.
- **Filesystem** (under `LLMSTUDIO_HOME`): uploads, prepared datasets (JSONL +
  `dataset_info.json`), run checkpoints, and exported models.

## Design principles
1. **Boots anywhere.** Defaults are baked into the settings models; heavy imports
   are lazy; missing optional parsers degrade gracefully.
2. **Swappable UI.** All logic lives below `services/`; the Gradio layer is thin.
3. **Resource-aware.** One model on the GPU at a time; the assistant yields to
   training; inference refuses to load while a job is active.
4. **Resilient.** Frequent checkpoints + persisted job state = recoverable runs.
