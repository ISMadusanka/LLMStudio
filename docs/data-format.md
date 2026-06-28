# Data Formats

LLM Studio accepts messy, real-world files and normalizes them into one of three
training schemas. You choose the schema (or let the assistant infer it) and map
your columns on the **Data** tab.

## Accepted file types

| Type | Notes |
|------|-------|
| `.csv`, `.tsv` | One row per example. |
| `.json` | A list of objects, or `{ "data": [...] }` / `examples` / `rows`. |
| `.jsonl`, `.ndjson` | One JSON object per line (recommended for large sets). |
| `.xlsx`, `.xls` | First sheet, one row per example. Needs `llmstudio[data]`. |
| `.parquet` | One row per example. |
| `.txt`, `.md` | Split into paragraphs → `text` records (completion/QA-synthesis). |
| `.pdf` | One record per page (extracted text). Needs `llmstudio[data]`. |
| `.docx` | One record per paragraph. Needs `llmstudio[data]`. |

## The three schemas

### 1. Instruction (Alpaca-style) — most common
Map columns to **instruction**, optional **input**, and **output**.

```json
{"instruction": "Summarize the text.", "input": "Long article…", "output": "A short summary."}
{"instruction": "Translate to French.", "output": "Bonjour"}
```

Internally normalized to chat messages:
```
[system?] → user: instruction (+ input) → assistant: output
```

### 2. Chat (multi-turn)
Point **messages** at a column holding a list of turns. Both OpenAI-style
(`role`/`content`) and ShareGPT-style (`from`/`value`) are understood; roles like
`human`/`gpt` are mapped automatically.

```json
{"messages": [
  {"role": "system", "content": "You are concise."},
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello! How can I help?"}
]}
```

### 3. Completion (raw text)
Point **text** at a single column. Good for continued pretraining or
domain-adaptation on unstructured corpora (e.g. extracted from PDFs/TXT).

```json
{"text": "Any free-form text to learn the style/voice of…"}
```

## Optional fields
- **System column** or a **static system prompt** applied to every example.
- **Input template** controls how instruction + input merge into the user turn
  (default: `"{instruction}\n\n{input}"`).

## Validation report
On **Prepare**, the validator flags the issues that most often hurt training:

- ❌ **too_few** — fewer than 10 examples (10–50 is small; 50+ recommended).
- ❌ **empty_output** — examples with a blank response (dropped from training).
- ⚠️ **tiny_output** — suspiciously short responses.
- ⚠️ **exceeds_seq_len** — examples longer than `max_seq_length` (will truncate).
- ⚠️ **duplicates** — repeated examples.
- ⚠️ **unparsed_rows** — raw rows skipped during structuring.

It also reports estimated tokens/example (median, p95, max), which informs the
recommended `max_seq_length`.

## On-disk layout
Each prepared dataset lives under `LLMSTUDIO_HOME/datasets/<dataset_id>/`:
```
train.jsonl          # one normalized Example per line
eval.jsonl           # held-out split
dataset_info.json    # schema, counts, mapping, stats
validation.json      # full report
preview.txt          # a few rendered examples
```

## Tips
- **Quality > quantity.** A few hundred clean, consistent examples beat thousands
  of noisy ones.
- **Be consistent** in tone/format of outputs — the model learns the pattern.
- For unstructured docs, you can have the assistant **synthesize Q/A pairs** from
  the text before training (see the data assistant).
