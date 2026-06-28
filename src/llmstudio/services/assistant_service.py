"""Assistant service: a thin chat wrapper + status for the help panel.

The advisors (hyperparameter / data) call the assistant directly; this service
is for free-form "ask the assistant" interactions in the UI.
"""

from __future__ import annotations

from typing import Optional

from llmstudio.core.assistant.llm import AssistantLLM
from llmstudio.core.utils.logging import get_logger

log = get_logger("services.assistant")

_STUDIO_SYSTEM = (
    "You are the LLM Studio assistant. You help non-technical users fine-tune "
    "open-source LLMs: preparing data, choosing models, and setting hyperparameters. "
    "Be concise, friendly, and practical."
)


class AssistantService:
    def __init__(self, assistant: AssistantLLM) -> None:
        self.assistant = assistant

    def available(self) -> bool:
        return self.assistant.available()

    def status(self) -> str:
        if not self.assistant.config.enabled:
            return "Assistant disabled in settings."
        if not self.assistant.available():
            return "Assistant unavailable (training stack not installed)."
        loaded = " (loaded)" if self.assistant.is_loaded else ""
        return f"Ready · model: {self.assistant.config.model_id}{loaded}"

    def chat(self, message: str, history: Optional[list[dict[str, str]]] = None) -> str:
        if not self.assistant.available():
            return (
                "The assistant model isn't available. Install the training stack "
                "(`pip install 'llmstudio[train]'`) and run `llmstudio setup` to enable it."
            )
        messages = [{"role": "system", "content": _STUDIO_SYSTEM}]
        for turn in history or []:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})
        return self.assistant.chat(messages)

    def unload(self) -> None:
        self.assistant.unload()
