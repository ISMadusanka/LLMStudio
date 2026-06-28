"""Validate normalized examples and produce a human-readable quality report.

Catches the issues that most often wreck a fine-tune: too few examples, empty
fields, duplicates, degenerate lengths, and examples that exceed the model's
sequence length. Token counts use a fast char/4 estimate unless a tokenizer is
supplied.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from llmstudio.core.data.schema import Example
from llmstudio.core.utils.logging import get_logger

log = get_logger("data.validator")

ERROR = "error"
WARNING = "warning"
INFO = "info"

# Thresholds (tunable)
MIN_EXAMPLES_ERROR = 10
MIN_EXAMPLES_WARN = 50
SHORT_OUTPUT_CHARS = 2
TINY_OUTPUT_WARN_CHARS = 10
MAX_PREVIEW_ROWS = 5


@dataclass
class Issue:
    level: str
    code: str
    message: str
    count: int = 0
    rows: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "rows": self.rows[:MAX_PREVIEW_ROWS],
        }


@dataclass
class ValidationReport:
    n_examples: int
    n_valid: int
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.level == ERROR for i in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.level == WARNING)

    @property
    def ok(self) -> bool:
        return not self.has_errors and self.n_valid > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_valid": self.n_valid,
            "ok": self.ok,
            "has_errors": self.has_errors,
            "issues": [i.to_dict() for i in self.issues],
            "stats": self.stats,
        }

    def to_markdown(self) -> str:
        icon = {ERROR: "❌", WARNING: "⚠️", INFO: "ℹ️"}
        lines = [f"**Examples:** {self.n_examples} · **Usable:** {self.n_valid}"]
        if self.stats:
            tl = self.stats.get("token_len", {})
            if tl:
                lines.append(
                    f"**Est. tokens/example:** median {tl.get('median', 0)}, "
                    f"max {tl.get('max', 0)}, p95 {tl.get('p95', 0)}"
                )
        if not self.issues:
            lines.append("\n✅ No issues found.")
            return "\n".join(lines)
        lines.append("\n**Findings:**")
        for issue in self.issues:
            suffix = f" _(e.g. rows {issue.rows[:MAX_PREVIEW_ROWS]})_" if issue.rows else ""
            count = f" ×{issue.count}" if issue.count else ""
            lines.append(f"- {icon.get(issue.level, '•')} **{issue.code}**{count}: {issue.message}{suffix}")
        return "\n".join(lines)


def _est_tokens(text: str, tokenizer: Optional[Callable[[str], int]]) -> int:
    if tokenizer is not None:
        try:
            return tokenizer(text)
        except Exception:
            pass
    return max(1, round(len(text) / 4))  # ~4 chars/token heuristic


def _example_text(ex: Example) -> str:
    if ex.is_chat:
        return "\n".join(m.get("content", "") for m in (ex.messages or []))
    return ex.text or ""


def _example_output_len(ex: Example) -> int:
    if ex.is_chat:
        return len(ex.last_assistant() or "")
    return len(ex.text or "")


def validate_examples(
    examples: list[Example],
    *,
    max_seq_length: int = 2048,
    dropped: Optional[list[tuple[int, str]]] = None,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> ValidationReport:
    """Run all checks and return a :class:`ValidationReport`."""
    n = len(examples)
    report = ValidationReport(n_examples=n, n_valid=n)
    issues = report.issues

    # Rows dropped during normalization.
    if dropped:
        issues.append(
            Issue(
                level=WARNING,
                code="unparsed_rows",
                message=f"{len(dropped)} raw row(s) were skipped during structuring (empty/invalid).",
                count=len(dropped),
                rows=[i for i, _ in dropped],
            )
        )

    # Count / size sanity.
    if n == 0:
        issues.append(Issue(ERROR, "no_examples", "No usable examples after structuring."))
        report.n_valid = 0
        return report
    if n < MIN_EXAMPLES_ERROR:
        issues.append(Issue(ERROR, "too_few", f"Only {n} examples — need at least {MIN_EXAMPLES_ERROR} to train."))
    elif n < MIN_EXAMPLES_WARN:
        issues.append(Issue(WARNING, "small_dataset", f"{n} examples is small; expect limited gains. {MIN_EXAMPLES_WARN}+ is better."))

    # Per-example checks.
    empty_out_rows: list[int] = []
    tiny_out_rows: list[int] = []
    too_long_rows: list[int] = []
    token_lens: list[int] = []
    seen: dict[str, int] = {}
    dup_rows: list[int] = []

    for idx, ex in enumerate(examples):
        full_text = _example_text(ex)
        out_len = _example_output_len(ex)
        tok = _est_tokens(full_text, tokenizer)
        token_lens.append(tok)

        if out_len <= SHORT_OUTPUT_CHARS:
            empty_out_rows.append(idx)
        elif out_len < TINY_OUTPUT_WARN_CHARS:
            tiny_out_rows.append(idx)
        if tok > max_seq_length:
            too_long_rows.append(idx)

        key = full_text.strip()
        if key in seen:
            dup_rows.append(idx)
        else:
            seen[key] = idx

    if empty_out_rows:
        issues.append(Issue(ERROR, "empty_output", "Examples with an empty/near-empty response.", len(empty_out_rows), empty_out_rows))
    if tiny_out_rows:
        issues.append(Issue(WARNING, "tiny_output", "Very short responses (<10 chars) — verify these are intentional.", len(tiny_out_rows), tiny_out_rows))
    if too_long_rows:
        issues.append(Issue(WARNING, "exceeds_seq_len", f"Examples longer than max_seq_length ({max_seq_length}) will be truncated.", len(too_long_rows), too_long_rows))
    if dup_rows:
        issues.append(Issue(WARNING, "duplicates", "Duplicate examples found; consider de-duplicating.", len(dup_rows), dup_rows))

    # Stats block.
    if token_lens:
        srt = sorted(token_lens)
        p95 = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
        report.stats["token_len"] = {
            "min": srt[0],
            "median": int(statistics.median(srt)),
            "mean": round(statistics.fmean(srt), 1),
            "p95": p95,
            "max": srt[-1],
        }
    report.stats["duplicate_count"] = len(dup_rows)
    report.stats["empty_output_count"] = len(empty_out_rows)

    report.n_valid = n - len(empty_out_rows)
    log.info("Validation: %d examples, %d errors, %d warnings",
             n, sum(1 for i in issues if i.level == ERROR), report.n_warnings)
    return report
