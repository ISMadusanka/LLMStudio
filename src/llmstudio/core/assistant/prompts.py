"""Prompt templates and JSON parsing for the LLM assistant."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Hyperparameter advisor
# ---------------------------------------------------------------------------
HYPERPARAM_SYSTEM = (
    "You are an expert ML engineer specializing in parameter-efficient fine-tuning "
    "(LoRA/QLoRA) of open-source LLMs with Unsloth. Given a dataset and GPU profile, "
    "you recommend a strong STARTING hyperparameter configuration. Be pragmatic and "
    "conservative: prefer settings that converge reliably and fit in memory. "
    "Respond with ONLY a single JSON object — no prose, no markdown fences."
)

HYPERPARAM_KEYS = [
    "num_train_epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler_type",
    "neftune_noise_alpha",
    "max_seq_length",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
]


def build_hyperparam_prompt(ctx: dict[str, Any]) -> str:
    return (
        "Recommend fine-tuning hyperparameters for this run.\n\n"
        f"- Base model: {ctx.get('model_name')} (~{ctx.get('params_b')}B params)\n"
        f"- Method: {ctx.get('mode')} (load_in_4bit={ctx.get('load_in_4bit')})\n"
        f"- Task format: {ctx.get('task_format')}\n"
        f"- Training examples: {ctx.get('n_train')} (eval: {ctx.get('n_eval')})\n"
        f"- Token length per example: median {ctx.get('median_tokens')}, p95 {ctx.get('p95_tokens')}, max {ctx.get('max_tokens')}\n"
        f"- GPU free VRAM: {ctx.get('gpu_free_gb')} GB\n"
        f"- Current max_seq_length: {ctx.get('max_seq_length')}, batch {ctx.get('per_device_train_batch_size')}"
        f"×{ctx.get('gradient_accumulation_steps')}\n\n"
        "Return JSON with these keys (only include keys you want to change): "
        f"{HYPERPARAM_KEYS} plus a short 'rationale' string. "
        "Guidance: smaller datasets need MORE epochs (2-4) and can use NEFTune (alpha ~5); "
        "large datasets use 1 epoch. Keep learning_rate in 1e-4..3e-4 for LoRA. "
        "Set lora_alpha to roughly equal lora_r. Do not exceed the GPU budget."
    )


# ---------------------------------------------------------------------------
# Data-prep assistant: field mapping
# ---------------------------------------------------------------------------
MAPPING_SYSTEM = (
    "You map raw tabular columns onto a fine-tuning schema. The schema is one of: "
    "'instruction' (instruction[+input]->output, Alpaca-style), 'chat' (a column "
    "holding a list of role/content messages), or 'completion' (a single text column). "
    "Respond with ONLY a single JSON object — no prose."
)


def build_mapping_prompt(columns: list[str], sample_rows: list[dict]) -> str:
    sample = json.dumps(sample_rows[:3], ensure_ascii=False, indent=2)[:2000]
    return (
        f"Columns: {columns}\n\n"
        f"Sample rows:\n{sample}\n\n"
        "Decide the best schema and which columns fill each role. Return JSON:\n"
        '{"task_format": "instruction|chat|completion", '
        '"instruction_field": null, "input_field": null, "output_field": null, '
        '"system_field": null, "text_field": null, "messages_field": null, '
        '"system_prompt": null, "rationale": "..."}'
    )


# ---------------------------------------------------------------------------
# Data-prep assistant: synthesize QA pairs from unstructured text
# ---------------------------------------------------------------------------
SYNTH_SYSTEM = (
    "You create high-quality instruction/answer training pairs from a source document. "
    "Questions must be answerable SOLELY from the text. Answers must be faithful and "
    "self-contained. Respond with ONLY a JSON array of objects."
)


def build_synth_prompt(text: str, n: int = 3) -> str:
    return (
        f"From the following text, write {n} diverse instruction/answer pairs.\n\n"
        f"TEXT:\n\"\"\"\n{text[:4000]}\n\"\"\"\n\n"
        'Return JSON: [{"instruction": "...", "output": "..."}, ...]'
    )


# ---------------------------------------------------------------------------
# Robust JSON extraction from an LLM response
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of a possibly chatty response."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the first balanced {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None
