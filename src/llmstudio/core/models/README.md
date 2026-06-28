# `core.models` — catalog · downloader · registry

```
catalog.py     ModelCatalog/Entry — the fine-tunable base models (from config/models.yaml)
downloader.py  on-demand HF download with candidate fallback (4bit → mirror → official)
registry.py    ModelRegistry — SQLite store of fine-tuned models + lineage/metrics
```

## Catalog
Loaded from `config/models.yaml`. Each entry knows its official repo, Unsloth
mirror, and Unsloth `bnb-4bit` repo. `candidate_repos(load_in_4bit, …)` yields an
ordered download fallback list.

## Downloader
`download_model(entry, load_in_4bit=…)` tries each candidate until one succeeds.
Gated repos raise `GatedModelError` with guidance (accept license + set
`HF_TOKEN`). Base models are fetched **only when a run starts**.

## Registry
`ModelRegistry.register(...)` records name, base model, artifact kind
(`lora` / `merged_16bit` / `merged_4bit` / `gguf`), quantization, on-disk path,
dataset + job lineage, final metrics, and the full training config. The UI lists,
inspects, loads, and deletes from here.
