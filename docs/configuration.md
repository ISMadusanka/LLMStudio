# Configuration Reference

Settings resolve in this order (highest priority first):

1. Constructor/init kwargs
2. Environment variables — prefix `LLMSTUDIO_`, nested with `__`
3. `.env` file
4. `config/default.yaml` (+ optional `$LLMSTUDIO_HOME/config.yaml`)
5. Built-in defaults

Because defaults are baked into the code, the app boots even with no YAML — the
files just document and override.

## Environment variable mapping

Nested keys use `__`. Examples:

| Setting | Env var |
|--------|---------|
| `server.port` | `LLMSTUDIO_SERVER__PORT` |
| `server.share` | `LLMSTUDIO_SERVER__SHARE` |
| `assistant.model_id` | `LLMSTUDIO_ASSISTANT__MODEL_ID` |
| `assistant.enabled` | `LLMSTUDIO_ASSISTANT__ENABLED` |
| `gpu.qlora_threshold_gb` | `LLMSTUDIO_GPU__QLORA_THRESHOLD_GB` |
| `log_level` | `LLMSTUDIO_LOG_LEVEL` |

Special, non-prefixed env vars:
- `LLMSTUDIO_HOME` — workspace root (overrides `paths.home`).
- `LLMSTUDIO_CONFIG_DIR` — directory containing `default.yaml`/`models.yaml`.
- `LLMSTUDIO_MODELS_CATALOG` — explicit catalog file path.
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — for gated model downloads.

## Sections (see `config/default.yaml` for all defaults)

### `paths`
Workspace layout. Relative paths resolve under `home` (which itself resolves
under `LLMSTUDIO_HOME` or the repo root). Subdirs: `uploads`, `datasets`, `runs`,
`models`, `hf_cache`, `logs`, plus the `registry_db` file.

### `server`
`host`, `port`, `share` (public Gradio tunnel), `auth` (`"user:password"`),
`show_error`.

### `assistant`
The in-app advisor model.
- `enabled` — turn the LLM assistant on/off (heuristics still work).
- `model_id` / `fallback_model_id` — primary and low-VRAM fallback.
- `min_vram_gb_for_primary` — below this free VRAM, use the fallback.
- `load_in_4bit`, `max_new_tokens`, `temperature`, `top_p`.
- `keep_resident` — keep loaded between calls (always unloaded before training).

### `training`
Defaults seeded into new runs: `seed`, `max_seq_length`, `eval_ratio`,
`logging_steps`, `save_steps`, `save_total_limit`, `default_export_format`,
`packing`, `gradient_checkpointing`.

### `gpu`
`vram_safety_factor`, `qlora_threshold_gb`, `cpu_offload_allowed`.

### `download`
`prefer_unsloth_4bit` (use Unsloth `bnb-4bit` repos for QLoRA),
`allow_official_fallback` (fall back to the official repo + on-the-fly 4-bit).

## User config override
Drop a `config.yaml` in your `LLMSTUDIO_HOME`; it is deep-merged over the
defaults. Handy for server deployments that share a workspace.

## Adding base models
Edit `config/models.yaml` — add an entry with `key`, `name`, `family`, `hf_id`,
`params_b`, `chat_template`, and (optionally) `unsloth_4bit_id`. No code change
needed; it appears in the UI dropdown immediately.
