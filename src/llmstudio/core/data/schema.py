"""Dataset schema: the normalized representation of training examples.

Everything the user uploads is normalized into :class:`Example` objects that are
either *chat* (a list of role/content messages) or *completion* (raw text). A
:class:`FieldMapping` records how the user's raw columns map onto that schema.
The on-disk prepared dataset is JSONL of ``Example.to_record()`` dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class TaskFormat(str, Enum):
    """Supported fine-tuning data shapes."""

    INSTRUCTION = "instruction"  # instruction (+ optional input) -> output  (Alpaca)
    CHAT = "chat"  # multi-turn list of {role, content} messages
    COMPLETION = "completion"  # raw text / continued pretraining


@dataclass
class Example:
    """A single normalized training example."""

    messages: Optional[list[dict[str, str]]] = None
    text: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    def system(self) -> Optional[str]:
        if not self.messages:
            return None
        for m in self.messages:
            if m.get("role") == ROLE_SYSTEM:
                return m.get("content")
        return None

    def last_assistant(self) -> Optional[str]:
        if not self.messages:
            return None
        for m in reversed(self.messages):
            if m.get("role") == ROLE_ASSISTANT:
                return m.get("content")
        return None

    def to_record(self) -> dict[str, Any]:
        if self.is_chat:
            rec: dict[str, Any] = {"messages": self.messages}
        else:
            rec = {"text": self.text}
        if self.meta:
            rec["meta"] = self.meta
        return rec

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Example":
        if "messages" in rec and rec["messages"] is not None:
            return cls(messages=list(rec["messages"]), meta=rec.get("meta", {}))
        return cls(text=rec.get("text"), meta=rec.get("meta", {}))


@dataclass
class FieldMapping:
    """How raw columns map onto the schema for a chosen :class:`TaskFormat`."""

    task_format: TaskFormat
    instruction_field: Optional[str] = None
    input_field: Optional[str] = None
    output_field: Optional[str] = None
    system_field: Optional[str] = None
    text_field: Optional[str] = None
    messages_field: Optional[str] = None
    system_prompt: Optional[str] = None  # static system prompt applied to all rows
    # How to merge an instruction with its optional input into one user turn.
    input_template: str = "{instruction}\n\n{input}"

    def required_fields(self) -> list[str]:
        if self.task_format == TaskFormat.INSTRUCTION:
            return [f for f in (self.instruction_field, self.output_field) if f]
        if self.task_format == TaskFormat.CHAT:
            return [f for f in (self.messages_field,) if f]
        return [f for f in (self.text_field,) if f]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_format": self.task_format.value,
            "instruction_field": self.instruction_field,
            "input_field": self.input_field,
            "output_field": self.output_field,
            "system_field": self.system_field,
            "text_field": self.text_field,
            "messages_field": self.messages_field,
            "system_prompt": self.system_prompt,
            "input_template": self.input_template,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FieldMapping":
        return cls(
            task_format=TaskFormat(d.get("task_format", "instruction")),
            instruction_field=d.get("instruction_field"),
            input_field=d.get("input_field"),
            output_field=d.get("output_field"),
            system_field=d.get("system_field"),
            text_field=d.get("text_field"),
            messages_field=d.get("messages_field"),
            system_prompt=d.get("system_prompt"),
            input_template=d.get("input_template", "{instruction}\n\n{input}"),
        )


@dataclass
class DatasetSchema:
    """Metadata describing a prepared dataset (written to dataset_info.json)."""

    dataset_id: str
    name: str
    task_format: TaskFormat
    n_examples: int
    n_train: int
    n_eval: int
    has_system: bool
    source_columns: list[str]
    mapping: dict[str, Any]
    source_files: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["task_format"] = self.task_format.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetSchema":
        d = dict(d)
        d["task_format"] = TaskFormat(d["task_format"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Heuristic column-name detection (the LLM assistant can refine this).
# ---------------------------------------------------------------------------
_INSTRUCTION_NAMES = ("instruction", "prompt", "question", "query", "task", "input_prompt")
_INPUT_NAMES = ("input", "context", "passage", "document", "source", "background")
_OUTPUT_NAMES = ("output", "response", "answer", "completion", "target", "label", "reply", "result")
_TEXT_NAMES = ("text", "content", "body", "document", "passage")
_MESSAGES_NAMES = ("messages", "conversation", "conversations", "dialog", "dialogue", "chat", "turns")
_SYSTEM_NAMES = ("system", "system_prompt", "system_message", "persona")


def _first_match(columns_lower: dict[str, str], names: tuple[str, ...]) -> Optional[str]:
    for n in names:
        if n in columns_lower:
            return columns_lower[n]
    # substring fallback
    for n in names:
        for low, original in columns_lower.items():
            if n in low:
                return original
    return None


def guess_mapping(columns: list[str]) -> FieldMapping:
    """Best-effort guess of a :class:`FieldMapping` from column names alone."""
    lower = {c.lower().strip(): c for c in columns}

    messages = _first_match(lower, _MESSAGES_NAMES)
    if messages:
        return FieldMapping(task_format=TaskFormat.CHAT, messages_field=messages,
                            system_field=_first_match(lower, _SYSTEM_NAMES))

    instruction = _first_match(lower, _INSTRUCTION_NAMES)
    output = _first_match(lower, _OUTPUT_NAMES)
    if instruction and output:
        return FieldMapping(
            task_format=TaskFormat.INSTRUCTION,
            instruction_field=instruction,
            input_field=_first_match(lower, _INPUT_NAMES),
            output_field=output,
            system_field=_first_match(lower, _SYSTEM_NAMES),
        )

    text = _first_match(lower, _TEXT_NAMES)
    if text:
        return FieldMapping(task_format=TaskFormat.COMPLETION, text_field=text)

    # Last resort: assume the first two columns are instruction/output.
    if len(columns) >= 2:
        return FieldMapping(
            task_format=TaskFormat.INSTRUCTION,
            instruction_field=columns[0],
            output_field=columns[1],
        )
    if columns:
        return FieldMapping(task_format=TaskFormat.COMPLETION, text_field=columns[0])
    return FieldMapping(task_format=TaskFormat.COMPLETION)
