"""Tests for data schema, normalization, and validation (no heavy deps)."""

from __future__ import annotations

from llmstudio.core.data.formatter import normalize_record, normalize_records
from llmstudio.core.data.schema import FieldMapping, TaskFormat, guess_mapping
from llmstudio.core.data.validator import validate_examples


def test_guess_mapping_instruction():
    m = guess_mapping(["instruction", "input", "output"])
    assert m.task_format == TaskFormat.INSTRUCTION
    assert m.instruction_field == "instruction"
    assert m.input_field == "input"
    assert m.output_field == "output"


def test_guess_mapping_chat():
    m = guess_mapping(["messages"])
    assert m.task_format == TaskFormat.CHAT
    assert m.messages_field == "messages"


def test_guess_mapping_completion():
    m = guess_mapping(["text"])
    assert m.task_format == TaskFormat.COMPLETION
    assert m.text_field == "text"


def test_normalize_instruction_with_input():
    mapping = FieldMapping(
        task_format=TaskFormat.INSTRUCTION,
        instruction_field="q",
        input_field="ctx",
        output_field="a",
    )
    ex = normalize_record({"q": "Summarize", "ctx": "Long text", "a": "Short"}, mapping)
    assert ex.is_chat
    roles = [m["role"] for m in ex.messages]
    assert roles == ["user", "assistant"]
    assert "Long text" in ex.messages[0]["content"]
    assert ex.last_assistant() == "Short"


def test_normalize_sharegpt_chat():
    mapping = FieldMapping(task_format=TaskFormat.CHAT, messages_field="conv")
    rec = {"conv": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
    ex = normalize_record(rec, mapping)
    assert [m["role"] for m in ex.messages] == ["user", "assistant"]


def test_validator_flags_small_and_empty():
    mapping = FieldMapping(task_format=TaskFormat.INSTRUCTION, instruction_field="i", output_field="o")
    records = [{"i": f"q{k}", "o": "a long enough answer here"} for k in range(5)]
    examples, dropped = normalize_records(records, mapping)
    report = validate_examples(examples, max_seq_length=2048, dropped=dropped)
    assert report.n_examples == 5
    # < 10 examples is an error
    assert report.has_errors
    assert any(i.code == "too_few" for i in report.issues)
