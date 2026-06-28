# `core.data` — data pipeline

Turns uploaded files into a validated, train-ready dataset.

```
load_paths()        loaders.py     read csv/json/jsonl/xlsx/txt/pdf/docx → RawDataset
guess_mapping()     schema.py      Example / FieldMapping / DatasetSchema + name heuristics
normalize_records() formatter.py   raw record → normalized Example (chat or completion)
validate_examples() validator.py   ValidationReport (errors/warnings + token stats)
DataPreparer        preparer.py    normalize → validate → split → write train/eval JSONL
```

## Flow
```python
from llmstudio.core.data import load_paths, guess_mapping, DataPreparer

raw = load_paths(["data.csv"])
mapping = guess_mapping(raw.columns)            # or build a FieldMapping by hand
prepared = DataPreparer(datasets_root).prepare(raw, mapping, name="my-set")
print(prepared.report.to_markdown())
```

## Schemas (`TaskFormat`)
- `instruction` — instruction (+ optional input) → output (Alpaca).
- `chat` — a column of `{role, content}` turns (OpenAI or ShareGPT style).
- `completion` — a single `text` column.

Everything is normalized to `Example` (chat `messages` or `text`). The model's
chat template is applied later by the training engine with the real tokenizer.

See [docs/data-format.md](../../../../docs/data-format.md).
