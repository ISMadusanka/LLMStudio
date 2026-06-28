"""The Unsloth-backed fine-tuning engine.

All heavy imports (torch, unsloth, trl, transformers, datasets) live inside
``run()`` so importing this module never requires the GPU stack. The engine:

  1. loads the (already-downloaded) base model + tokenizer with Unsloth,
  2. attaches a LoRA adapter,
  3. formats the prepared dataset with the model's chat template,
  4. trains via TRL's SFTTrainer with checkpointing + live callbacks,
  5. on clean completion, exports the model in the requested format.

NOTE: TRL/Unsloth APIs drift between releases. The version-sensitive spots are
flagged inline; verify against your installed versions on the GPU box.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llmstudio.core.training.callbacks import build_studio_callback
from llmstudio.core.training.config import ExportFormat, TrainingConfig
from llmstudio.core.training.job import JobControl, JobStore
from llmstudio.core.utils.events import EventBus
from llmstudio.core.utils.logging import get_logger

log = get_logger("training.engine")

# Per-template markers for "train on responses only" (mask the prompt tokens).
# Falls back to disabling the feature for unknown templates.
_RESPONSE_PARTS = {
    "llama-3.1": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama-3": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "qwen-2.5": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gemma2": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "mistral": ("[INST]", "[/INST]"),
    "phi-3": ("<|user|>\n", "<|assistant|>\n"),
}


def latest_checkpoint(checkpoints_dir: Path) -> Optional[str]:
    """Return the path to the highest-step ``checkpoint-N`` dir, if any."""
    if not Path(checkpoints_dir).exists():
        return None
    best: Optional[str] = None
    best_step = -1
    for path in glob.glob(os.path.join(str(checkpoints_dir), "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", path)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best = path
    return best


def _safe_construct(factory, kwargs: dict):
    """Construct ``factory(**kwargs)``, dropping kwargs the installed version of
    the class doesn't accept (TRL/transformers/Unsloth APIs vary a lot).

    Repeatedly catches ``TypeError: ... unexpected keyword argument 'X'`` and
    drops ``X``. If the offending kwarg isn't one we passed, it's almost always a
    stale Unsloth compiled cache — surface that with an actionable hint.
    """
    kwargs = dict(kwargs)
    last_exc: Optional[Exception] = None
    for _ in range(60):
        try:
            return factory(**kwargs)
        except TypeError as exc:
            last_exc = exc
            match = re.search(r"unexpected keyword argument '([\w]+)'", str(exc))
            if not match:
                raise
            bad = match.group(1)
            if bad in kwargs:
                log.warning("Dropping training arg '%s' (unsupported by this version).", bad)
                kwargs.pop(bad)
                continue
            raise TypeError(
                f"{exc}. This usually means Unsloth's compiled cache is stale for your "
                f"installed transformers/trl versions. Delete the 'unsloth_compiled_cache' "
                f"folder in your project and restart, then retry."
            ) from exc
    raise last_exc  # pragma: no cover


@dataclass
class TrainingResult:
    completed: bool = False
    stopped: bool = False
    stop_reason: str = ""  # "" | "pause" | "stop"
    output_dir: str = ""
    checkpoints_dir: str = ""
    last_checkpoint: Optional[str] = None
    final_metrics: dict[str, Any] = field(default_factory=dict)
    exported_path: Optional[str] = None
    export_kind: Optional[str] = None


class UnslothEngine:
    def __init__(
        self,
        *,
        config: TrainingConfig,
        model_repo: str,
        run_dir: Path,
        export_dir: Path,
        train_path: Path,
        eval_path: Optional[Path],
        bus: EventBus,
        store: JobStore,
        control: JobControl,
        job_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        self.cfg = config
        self.model_repo = model_repo
        self.run_dir = Path(run_dir)
        self.export_dir = Path(export_dir)
        self.train_path = Path(train_path)
        self.eval_path = Path(eval_path) if eval_path else None
        self.bus = bus
        self.store = store
        self.control = control
        self.job_id = job_id
        self.hf_token = hf_token
        self.checkpoints_dir = self.run_dir / "checkpoints"

    # ------------------------------------------------------------------ run
    def run(self, resume_from: Optional[str] = None) -> TrainingResult:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        result = TrainingResult(
            output_dir=str(self.run_dir),
            checkpoints_dir=str(self.checkpoints_dir),
        )
        self._log("Loading base model — this can take a while on first download.")

        model, tokenizer = self._load_model_and_tokenizer()
        model = self._attach_lora(model)
        tokenizer = self._apply_chat_template(tokenizer)
        train_ds, eval_ds = self._build_datasets(tokenizer)
        trainer = self._build_trainer(model, tokenizer, train_ds, eval_ds)
        trainer = self._maybe_train_on_responses_only(trainer)

        # Resume from an explicit checkpoint, or auto-detect one in the run dir.
        resume = resume_from or latest_checkpoint(self.checkpoints_dir)
        if resume:
            self._log(f"Resuming from checkpoint: {resume}")

        self._log("Starting training loop.")
        train_output = trainer.train(resume_from_checkpoint=resume)

        # Distinguish a clean finish from a pause/stop request.
        if self.control.stop_requested():
            result.stopped, result.stop_reason = True, "stop"
        elif self.control.pause_requested():
            result.stopped, result.stop_reason = True, "pause"
        else:
            result.completed = True

        result.last_checkpoint = latest_checkpoint(self.checkpoints_dir)
        try:
            result.final_metrics = dict(getattr(train_output, "metrics", {}) or {})
        except Exception:
            result.final_metrics = {}

        if result.completed:
            self._log("Training complete — exporting model.")
            exported, kind = self._export(model, tokenizer)
            result.exported_path, result.export_kind = exported, kind
        else:
            self._log(f"Training halted ({result.stop_reason}); latest checkpoint retained.")

        self._cleanup(model, tokenizer, trainer)
        return result

    # ----------------------------------------------------------- internals
    def _load_model_and_tokenizer(self):
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_repo,
            max_seq_length=self.cfg.max_seq_length,
            dtype=None,  # auto (bf16 on Ampere+, else fp16)
            load_in_4bit=self.cfg.load_in_4bit,
            token=self.hf_token,
        )
        return model, tokenizer

    def _attach_lora(self, model):
        from unsloth import FastLanguageModel  # type: ignore

        return FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.target_modules,
            bias="none",
            use_gradient_checkpointing=self.cfg.use_gradient_checkpointing,
            random_state=self.cfg.seed,
            use_rslora=self.cfg.use_rslora,
        )

    def _apply_chat_template(self, tokenizer):
        if self.cfg.chat_template in ("none", "", "completion"):
            return tokenizer
        try:
            from unsloth.chat_templates import get_chat_template  # type: ignore

            return get_chat_template(tokenizer, chat_template=self.cfg.chat_template)
        except Exception as exc:  # pragma: no cover
            self._log(f"Could not apply chat template '{self.cfg.chat_template}' ({exc}); using tokenizer default.")
            return tokenizer

    def _build_datasets(self, tokenizer):
        from datasets import load_dataset  # type: ignore

        def format_chat(batch):
            texts = []
            for messages in batch["messages"]:
                texts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                )
            return {"text": texts}

        train_ds = load_dataset("json", data_files=str(self.train_path), split="train")
        is_chat = "messages" in train_ds.column_names
        if is_chat:
            train_ds = train_ds.map(format_chat, batched=True, num_proc=self.cfg.dataset_num_proc)

        eval_ds = None
        if (
            self.eval_path
            and self.eval_path.exists()
            and self.eval_path.stat().st_size > 0
            and self.cfg.eval_strategy != "no"
        ):
            eval_ds = load_dataset("json", data_files=str(self.eval_path), split="train")
            if "messages" in eval_ds.column_names:
                eval_ds = eval_ds.map(format_chat, batched=True, num_proc=self.cfg.dataset_num_proc)
        return train_ds, eval_ds

    def _training_args(self, has_eval: bool):
        # Auto precision selection.
        bf16 = self.cfg.bf16
        fp16 = self.cfg.fp16
        if bf16 is None and fp16 is None:
            try:
                from unsloth import is_bfloat16_supported  # type: ignore

                bf16 = bool(is_bfloat16_supported())
            except Exception:
                bf16 = False
            fp16 = not bf16

        common = dict(
            output_dir=str(self.checkpoints_dir),
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            warmup_steps=self.cfg.warmup_steps,
            warmup_ratio=self.cfg.warmup_ratio if not self.cfg.warmup_steps else 0.0,
            num_train_epochs=self.cfg.num_train_epochs if self.cfg.max_steps <= 0 else 1,
            max_steps=self.cfg.max_steps if self.cfg.max_steps > 0 else -1,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            weight_decay=self.cfg.weight_decay,
            optim=self.cfg.optim,
            max_grad_norm=self.cfg.max_grad_norm,
            seed=self.cfg.seed,
            fp16=bool(fp16),
            bf16=bool(bf16),
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            save_total_limit=self.cfg.save_total_limit,
            eval_strategy=(self.cfg.eval_strategy if has_eval else "no"),
            eval_steps=self.cfg.eval_steps if has_eval else None,
            report_to="none",
        )
        if self.cfg.neftune_noise_alpha:
            common["neftune_noise_alpha"] = self.cfg.neftune_noise_alpha
        # Only pass these niche/version-fragile args when actually requested;
        # _safe_construct drops them if the installed version doesn't accept them.
        if self.cfg.group_by_length:
            common["group_by_length"] = True

        # Newer TRL puts SFT-specific fields in SFTConfig; fall back to
        # transformers.TrainingArguments for older installs. _safe_construct makes
        # both resilient to version-specific unsupported kwargs.
        try:
            from trl import SFTConfig  # type: ignore

            sft_kwargs = dict(
                dataset_text_field="text",
                max_seq_length=self.cfg.max_seq_length,
                packing=self.cfg.packing,
                dataset_num_proc=self.cfg.dataset_num_proc,
                **common,
            )
            return _safe_construct(SFTConfig, sft_kwargs), "sftconfig"
        except ImportError:
            from transformers import TrainingArguments  # type: ignore

            return _safe_construct(TrainingArguments, common), "trainingargs"

    def _build_trainer(self, model, tokenizer, train_ds, eval_ds):
        from trl import SFTTrainer  # type: ignore

        args, kind = self._training_args(has_eval=eval_ds is not None)
        kwargs: dict[str, Any] = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=args,
        )
        # Older SFTTrainer needs these as direct kwargs (newer reads from SFTConfig).
        if kind == "trainingargs":
            kwargs.update(
                dataset_text_field="text",
                max_seq_length=self.cfg.max_seq_length,
                packing=self.cfg.packing,
            )
        trainer = SFTTrainer(**kwargs)
        trainer.add_callback(
            build_studio_callback(self.job_id, bus=self.bus, store=self.store, control=self.control)
        )
        return trainer

    def _maybe_train_on_responses_only(self, trainer):
        if not self.cfg.train_on_responses_only:
            return trainer
        parts = _RESPONSE_PARTS.get(self.cfg.chat_template)
        if not parts:
            self._log(f"train_on_responses_only unsupported for template '{self.cfg.chat_template}'; training on full text.")
            return trainer
        try:
            from unsloth.chat_templates import train_on_responses_only  # type: ignore

            return train_on_responses_only(trainer, instruction_part=parts[0], response_part=parts[1])
        except Exception as exc:  # pragma: no cover
            self._log(f"Could not enable train_on_responses_only ({exc}); continuing.")
            return trainer

    def _export(self, model, tokenizer) -> tuple[str, str]:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        out = str(self.export_dir)
        fmt = self.cfg.export_format
        try:
            if fmt == ExportFormat.LORA:
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
            elif fmt == ExportFormat.MERGED_16BIT:
                model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
            elif fmt == ExportFormat.MERGED_4BIT:
                model.save_pretrained_merged(out, tokenizer, save_method="merged_4bit")
            elif fmt == ExportFormat.GGUF:
                model.save_pretrained_gguf(out, tokenizer, quantization_method=self.cfg.gguf_quantization)
            else:  # safety net
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
        except Exception as exc:
            # Don't lose the run if export of a fancy format fails — fall back to adapter.
            self._log(f"Export as {fmt.value} failed ({exc}); saving LoRA adapter instead.")
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            return out, ExportFormat.LORA.value
        return out, fmt.value

    def _cleanup(self, *objs) -> None:
        from llmstudio.core.utils.resources import free_gpu_memory

        for _ in objs:
            pass
        free_gpu_memory()

    def _log(self, message: str) -> None:
        self.bus.log(self.job_id, message)
        log.info("[%s] %s", self.job_id, message)
