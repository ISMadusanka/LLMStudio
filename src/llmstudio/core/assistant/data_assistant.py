"""LLM-assisted data preparation: infer the field mapping, advise on cleaning,
and (optionally) synthesize instruction/answer pairs from unstructured text.

All methods degrade gracefully to heuristics when the assistant is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from llmstudio.core.assistant.llm import AssistantLLM
from llmstudio.core.assistant.prompts import (
    MAPPING_SYSTEM,
    SYNTH_SYSTEM,
    build_mapping_prompt,
    build_synth_prompt,
    extract_json,
)
from llmstudio.core.data.schema import FieldMapping, TaskFormat, guess_mapping
from llmstudio.core.utils.logging import get_logger

log = get_logger("assistant.data")


@dataclass
class MappingSuggestion:
    mapping: FieldMapping
    rationale: str = ""
    source: str = "heuristic"  # "llm" | "heuristic"


class DataAssistant:
    def __init__(self, llm: Optional[AssistantLLM] = None) -> None:
        self.llm = llm

    # --------------------------------------------------------- field mapping
    def suggest_mapping(self, columns: list[str], sample_rows: list[dict]) -> MappingSuggestion:
        if self.llm is not None and self.llm.available():
            try:
                return self._suggest_mapping_llm(columns, sample_rows)
            except Exception as exc:  # pragma: no cover
                log.warning("LLM mapping failed (%s); using heuristic.", exc)
        return MappingSuggestion(mapping=guess_mapping(columns), source="heuristic",
                                 rationale="Matched common column names.")

    def _suggest_mapping_llm(self, columns: list[str], sample_rows: list[dict]) -> MappingSuggestion:
        raw = self.llm.complete(
            MAPPING_SYSTEM, build_mapping_prompt(columns, sample_rows),
            max_new_tokens=400, temperature=0.1,
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("mapping assistant did not return JSON")

        # Validate that referenced columns actually exist; drop the rest.
        valid = set(columns)

        def col(name: str) -> Optional[str]:
            v = data.get(name)
            return v if v in valid else None

        try:
            fmt = TaskFormat(str(data.get("task_format", "instruction")))
        except ValueError:
            fmt = TaskFormat.INSTRUCTION

        mapping = FieldMapping(
            task_format=fmt,
            instruction_field=col("instruction_field"),
            input_field=col("input_field"),
            output_field=col("output_field"),
            system_field=col("system_field"),
            text_field=col("text_field"),
            messages_field=col("messages_field"),
            system_prompt=data.get("system_prompt") or None,
        )
        # If the LLM's choice is incomplete, fall back to the heuristic.
        if not mapping.required_fields():
            return MappingSuggestion(mapping=guess_mapping(columns), source="heuristic",
                                     rationale="Assistant mapping was incomplete; used heuristic instead.")
        return MappingSuggestion(mapping=mapping, source="llm",
                                 rationale=str(data.get("rationale", "")).strip())

    # ------------------------------------------------------ cleaning advice
    def advise_cleaning(self, report_markdown: str) -> str:
        if self.llm is None or not self.llm.available():
            return ""
        try:
            return self.llm.complete(
                "You are a data-quality advisor for LLM fine-tuning. Give 3-5 concise, "
                "actionable bullet points to improve this dataset. No preamble.",
                f"Validation report:\n{report_markdown}",
                max_new_tokens=400,
                temperature=0.3,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Cleaning advice failed: %s", exc)
            return ""

    # ------------------------------------------- synthesize QA from text
    def synthesize_pairs(self, text: str, n: int = 3) -> list[dict[str, str]]:
        """Generate instruction/output pairs from a chunk of unstructured text."""
        if self.llm is None or not self.llm.available():
            return []
        try:
            raw = self.llm.complete(SYNTH_SYSTEM, build_synth_prompt(text, n),
                                    max_new_tokens=900, temperature=0.6)
            data = extract_json(raw)
            pairs: list[dict[str, str]] = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("instruction") and item.get("output"):
                        pairs.append({"instruction": str(item["instruction"]), "output": str(item["output"])})
            return pairs
        except Exception as exc:  # pragma: no cover
            log.warning("QA synthesis failed: %s", exc)
            return []
