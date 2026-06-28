# `services` — orchestration façade

The `Studio` object composes the core components into the operations the UI and
CLI call. Build one per process via `get_studio()`.

```
Studio                aggregates everything; wires on_before_train=assistant.unload; recover()
data_service.py       stage uploads · preview · suggest mapping · prepare · list datasets
training_service.py   GPU recommendation · advisor · start/pause/resume/cancel · live events
inference_service.py  load registry model / paused checkpoint · chat / stream · GPU guard
registry_service.py   list/inspect/delete fine-tuned models for the UI
assistant_service.py  free-form "ask the assistant" chat + status
system_service.py     GPU/memory status · first-run setup · environment doctor
```

## The important wiring
```python
self.jobs = JobManager(..., on_before_train=self.assistant.unload)
```
This is what guarantees the advisor model leaves VRAM before a fine-tune begins.

`InferenceService` refuses to load a model while a job is actively using the GPU —
the intended flow is **pause the run, then probe its checkpoint**.

```python
from llmstudio.services import get_studio
studio = get_studio()
studio.system.setup()                 # download assistant
prepared = studio.data.prepare(...)   # build a dataset
cfg, rec = studio.training.build_default_config(model_key, prepared.dataset_id)
job = studio.training.start(cfg, name="my-run")
```
