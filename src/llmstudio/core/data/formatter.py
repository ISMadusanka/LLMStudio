"""Turn raw records into normalized :class:`Example` objects, and render
human-readable previews.

The *actual* chat-template formatting used for training is applied later by the
training engine with the real model tokenizer (``apply_chat_template``). The
renderer here is only for previews in the UI and for completion-format text.
"""

from __future__ import annotations

from typing import Any, Optional

from llmstudio.core.data.schema import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    Example,
    FieldMapping,
    TaskFormat,
)
from llmstudio.core.utils.logging import get_logger

log = get_logger("data.formatter")

# ShareGPT-style role aliases -> canonical roles.
_ROLE_ALIASES = {
    "human": ROLE_USER,
    "user": ROLE_USER,
    "prompter": ROLE_USER,
    "question": ROLE_USER,
    "gpt": ROLE_ASSISTANT,
    "assistant": ROLE_ASSISTANT,
    "bot": ROLE_ASSISTANT,
    "answer": ROLE_ASSISTANT,
    "system": ROLE_SYSTEM,
}


class NormalizationError(ValueError):
    """A single record could not be normalized (skipped, with a reason)."""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_messages_value(raw: Any) -> list[dict[str, str]]:
    """Coerce a messages/conversation column value into canonical messages."""
    if isinstance(raw, str):
        # Sometimes the column holds a JSON string.
        import json

        try:
            raw = json.loads(raw)
        except Exception as exc:
            raise NormalizationError(f"messages field is a non-JSON string: {exc}")
    if not isinstance(raw, list):
        raise NormalizationError("messages field is not a list")

    messages: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            raise NormalizationError("conversation turn is not an object")
        role = turn.get("role") or turn.get("from") or turn.get("speaker")
        content = turn.get("content")
        if content is None:
            content = turn.get("value") or turn.get("text") or turn.get("message")
        role = _ROLE_ALIASES.get(_clean(role).lower(), _clean(role).lower() or ROLE_USER)
        messages.append({"role": role, "content": _clean(content)})
    if not messages:
        raise NormalizationError("conversation has no turns")
    return messages


def normalize_record(record: dict[str, Any], mapping: FieldMapping) -> Example:
    """Normalize one raw record to an :class:`Example`. Raises on unusable rows."""
    fmt = mapping.task_format

    if fmt == TaskFormat.COMPLETION:
        text = _clean(record.get(mapping.text_field)) if mapping.text_field else ""
        if not text:
            raise NormalizationError("empty text")
        return Example(text=text)

    if fmt == TaskFormat.CHAT:
        if not mapping.messages_field:
            raise NormalizationError("no messages field configured")
        messages = _normalize_messages_value(record.get(mapping.messages_field))
        sys_value = _clean(record.get(mapping.system_field)) if mapping.system_field else ""
        sys_value = sys_value or (mapping.system_prompt or "")
        if sys_value and not any(m["role"] == ROLE_SYSTEM for m in messages):
            messages = [{"role": ROLE_SYSTEM, "content": sys_value}, *messages]
        return Example(messages=messages)

    # INSTRUCTION
    instruction = _clean(record.get(mapping.instruction_field)) if mapping.instruction_field else ""
    output = _clean(record.get(mapping.output_field)) if mapping.output_field else ""
    extra_input = _clean(record.get(mapping.input_field)) if mapping.input_field else ""
    if not instruction:
        raise NormalizationError("empty instruction")
    if not output:
        raise NormalizationError("empty output")

    user_content = (
        mapping.input_template.format(instruction=instruction, input=extra_input).strip()
        if extra_input
        else instruction
    )
    messages = []
    sys_value = _clean(record.get(mapping.system_field)) if mapping.system_field else ""
    sys_value = sys_value or (mapping.system_prompt or "")
    if sys_value:
        messages.append({"role": ROLE_SYSTEM, "content": sys_value})
    messages.append({"role": ROLE_USER, "content": user_content})
    messages.append({"role": ROLE_ASSISTANT, "content": output})
    return Example(messages=messages)


def normalize_records(
    records: list[dict[str, Any]], mapping: FieldMapping
) -> tuple[list[Example], list[tuple[int, str]]]:
    """Normalize many records. Returns (examples, [(row_index, reason), ...])."""
    examples: list[Example] = []
    dropped: list[tuple[int, str]] = []
    for i, rec in enumerate(records):
        try:
            examples.append(normalize_record(rec, mapping))
        except NormalizationError as exc:
            dropped.append((i, str(exc)))
    if dropped:
        log.info("Normalization dropped %d/%d rows", len(dropped), len(records))
    return examples, dropped


# ---------------------------------------------------------------------------
# Preview rendering (ChatML-ish, model-agnostic). NOT used for training.
# ---------------------------------------------------------------------------
def render_preview(example: Example, max_chars: int = 1200) -> str:
    if not example.is_chat:
        text = example.text or ""
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    parts: list[str] = []
    for m in example.messages or []:
        role = m.get("role", "user").upper()
        parts.append(f"<{role}>\n{m.get('content', '')}")
    rendered = "\n\n".join(parts)
    return rendered[:max_chars] + ("…" if len(rendered) > max_chars else "")
