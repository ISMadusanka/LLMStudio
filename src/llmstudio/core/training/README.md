# `core.training` — fine-tuning engine & job control

```
config.py     TrainingConfig — every hyperparameter (validated, with defaults)
engine.py     UnslothEngine — load → LoRA → format → SFTTrainer → export (lazy ML imports)
callbacks.py  StudioCallback — stream metrics, persist progress, honor pause/stop
job.py        JobStatus/JobRow/Job + JobStore (SQLite) + JobControl (pause/stop signal)
manager.py    JobManager — background threads, pause/resume/cancel, crash recovery
```

## Lifecycle
```
submit → DOWNLOADING → (on_before_train: unload assistant) → RUNNING
         ├─ COMPLETED → exported + registered
         ├─ PAUSED    → resumable from last checkpoint
         └─ FAILED / CANCELLED
```

## Highlights
- **Checkpoints** every `save_steps`; keep `save_total_limit` most recent.
- **Pause/Resume**: pause requests a checkpoint + clean stop; resume passes
  `resume_from_checkpoint`.
- **Crash recovery**: `JobManager.recover()` marks interrupted-with-checkpoint
  runs `RESUMABLE` on boot.
- **Resource release**: `on_before_train` (wired to `assistant.unload`) frees VRAM
  before the trainable model loads.
- **Export formats**: LoRA adapter, merged 16-bit, merged 4-bit, GGUF.

> All heavy imports live inside `engine.run()`, so this package imports without a
> GPU. The TRL/Unsloth API is version-sensitive — see inline notes in `engine.py`.
