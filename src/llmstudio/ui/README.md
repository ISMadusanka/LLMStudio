# `ui` — Gradio web studio

```
app.py            build_app() / launch(); shared cross-tab gr.State (config, job id)
theme.py          Soft theme + CSS
components/        stream_task (live log streaming), status_badge, metrics_dataframe
pages/
  home.py         welcome · system status · setup · doctor
  data.py         upload → map fields → validate → prepare
  configure.py    model + dataset · GPU recommendation · hyperparameters · advisor
  train.py        start · live loss/LR charts · logs · pause/resume/cancel
  inference.py    load model/checkpoint · streaming chat
  registry.py     browse · inspect · delete fine-tuned models
```

Pages call **only** the services layer (`studio.data`, `studio.training`, …),
never core directly — keeping the UI thin and swappable.

## Cross-tab state
- `cfg_state` — the saved `TrainingConfig` flows **Configure → Train**.
- `job_state` — the active job id flows **Train → Inference** (probe checkpoints).

## Live updates
The Train tab auto-refreshes via `gr.Timer` (Gradio ≥ 4.39) and also offers a
manual **🔄** refresh. Charts read metric history from the event bus.

Run it: `llmstudio ui` (or `from llmstudio.ui import launch; launch()`).
