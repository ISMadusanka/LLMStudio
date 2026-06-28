"""Inference engine: load a fine-tuned model (or an intermediate checkpoint)
and generate text.

Used for two things in the UI:
  * **Probe a paused run** — point it at the latest ``checkpoint-N`` directory.
  * **Chat with a registry model** — point it at a saved LoRA adapter or merged
    model directory.

Loading is expensive, so an engine instance keeps one model resident and offers
``unload()`` to release VRAM (important on a single GPU shared with training).
Unsloth is used when available; otherwise it falls back to transformers + PEFT.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from llmstudio.core.data.schema import ROLE_ASSISTANT, ROLE_USER
from llmstudio.core.utils.logging import get_logger
from llmstudio.core.utils.resources import free_gpu_memory

log = get_logger("inference.engine")

Messages = list[dict[str, str]]


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    do_sample: bool = True

    def to_generate_kwargs(self) -> dict:
        kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        if self.do_sample:
            kwargs["temperature"] = max(1e-4, self.temperature)
        return kwargs


class InferenceEngine:
    """Holds (at most) one loaded model and generates from it."""

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        load_in_4bit: bool = True,
        max_seq_length: int = 4096,
        chat_template: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.chat_template = chat_template
        self.hf_token = hf_token
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress: Optional[Callable[[str], None]] = None) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return

            def emit(message: str) -> None:
                log.info(message)
                if progress:
                    try:
                        progress(message)
                    except Exception:
                        pass

            emit("Loading tokenizer & weights (first load may download several GB)…")
            try:
                self._load_unsloth()
            except ImportError:
                emit("Unsloth unavailable — using transformers + PEFT…")
                self._load_transformers()
            emit("Optimizing and finishing up…")

    def _load_unsloth(self) -> None:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
            token=self.hf_token,
        )
        FastLanguageModel.for_inference(model)
        self._apply_template(tokenizer)
        self._model, self._tokenizer = model, tokenizer

    def _load_transformers(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        path = Path(self.model_path)
        adapter_cfg = path / "adapter_config.json"
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.hf_token)

        if adapter_cfg.exists():
            import json

            from peft import PeftModel  # type: ignore

            base = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
            base_model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype="auto", device_map="auto", token=self.hf_token
            )
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto", token=self.hf_token
            )
        model.eval()
        self._apply_template(tokenizer)
        self._model, self._tokenizer = model, tokenizer

    def _apply_template(self, tokenizer) -> None:
        if self.chat_template and getattr(tokenizer, "chat_template", None) in (None, ""):
            try:
                from unsloth.chat_templates import get_chat_template  # type: ignore

                get_chat_template(tokenizer, chat_template=self.chat_template)
            except Exception:
                pass

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
        free_gpu_memory()
        log.info("Unloaded inference model.")

    # ---------------------------------------------------------- generation
    def _prepare_inputs(self, prompt_or_messages: Union[str, Messages]):
        tok = self._tokenizer
        if isinstance(prompt_or_messages, str):
            messages = [{"role": ROLE_USER, "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        try:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            # No chat template: join into a simple prompt.
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + f"\n{ROLE_ASSISTANT}:"
            input_ids = tok(text, return_tensors="pt").input_ids
        return input_ids.to(self._model.device)

    def generate(self, prompt_or_messages: Union[str, Messages], params: Optional[GenerationParams] = None) -> str:
        """Generate a full completion (blocking)."""
        self.load()
        params = params or GenerationParams()
        import torch  # type: ignore

        input_ids = self._prepare_inputs(prompt_or_messages)
        eos = self._tokenizer.eos_token_id
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                pad_token_id=eos,
                **params.to_generate_kwargs(),
            )
        new_tokens = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_stream(
        self, prompt_or_messages: Union[str, Messages], params: Optional[GenerationParams] = None
    ) -> Iterator[str]:
        """Yield the completion incrementally as it is generated."""
        self.load()
        params = params or GenerationParams()
        from transformers import TextIteratorStreamer  # type: ignore

        input_ids = self._prepare_inputs(prompt_or_messages)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            input_ids=input_ids,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
            **params.to_generate_kwargs(),
        )
        thread = threading.Thread(target=self._model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()
