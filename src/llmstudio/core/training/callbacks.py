"""Training callbacks that bridge the HF/TRL trainer to LLM Studio.

``build_studio_callback`` returns a ``transformers.TrainerCallback`` instance
(the base class is imported lazily so this module is safe to import without the
training stack). The callback:

  * streams loss / lr / grad-norm to the event bus for live charts,
  * persists step/epoch/checkpoint progress to the job store (crash recovery),
  * polls :class:`JobControl` and cleanly stops + checkpoints on pause/stop.
"""

from __future__ import annotations

from typing import Any, Optional

from llmstudio.core.training.job import JobControl, JobStatus, JobStore
from llmstudio.core.utils.events import EventBus
from llmstudio.core.utils.logging import get_logger

log = get_logger("training.callbacks")


def build_studio_callback(
    job_id: str,
    *,
    bus: EventBus,
    store: JobStore,
    control: JobControl,
):
    """Construct the studio TrainerCallback (lazy transformers import)."""
    from transformers import TrainerCallback  # type: ignore

    class StudioCallback(TrainerCallback):
        def __init__(self) -> None:
            self.job_id = job_id
            self.bus = bus
            self.store = store
            self.control = control
            self._halting = False

        # -- lifecycle ------------------------------------------------------
        def on_train_begin(self, args, state, control_obj, **kwargs):  # noqa: ANN001
            total = int(getattr(state, "max_steps", 0) or 0)
            self.store.update(self.job_id, total_steps=total)
            self.bus.status(self.job_id, JobStatus.RUNNING.value, f"Training started ({total} steps).")
            self.bus.log(self.job_id, f"Training started — {total} optimizer steps planned.")
            return control_obj

        def on_log(self, args, state, control_obj, logs=None, **kwargs):  # noqa: ANN001
            logs = logs or {}
            step = int(getattr(state, "global_step", 0))
            values: dict[str, Any] = {}
            for key in ("loss", "eval_loss", "learning_rate", "grad_norm", "mean_token_accuracy"):
                if key in logs and logs[key] is not None:
                    try:
                        values[key] = float(logs[key])
                    except (TypeError, ValueError):
                        pass
            epoch = float(logs.get("epoch", getattr(state, "epoch", 0.0)) or 0.0)
            if values:
                self.bus.metric(self.job_id, step, {"epoch": round(epoch, 3), **values})
            # Persist a light snapshot for resume / UI on refresh.
            self.store.update(
                self.job_id,
                current_step=step,
                current_epoch=round(epoch, 3),
                metrics=values,
            )
            return control_obj

        def on_save(self, args, state, control_obj, **kwargs):  # noqa: ANN001
            import os

            step = int(getattr(state, "global_step", 0))
            ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
            self.store.update(self.job_id, last_checkpoint=ckpt, resumable=True)
            self.bus.log(self.job_id, f"Checkpoint saved at step {step}.")
            return control_obj

        def on_step_end(self, args, state, control_obj, **kwargs):  # noqa: ANN001
            # Cooperative pause/stop: ask the trainer to checkpoint then stop.
            if not self._halting and self.control.should_halt():
                self._halting = True
                reason = "stop" if self.control.stop_requested() else "pause"
                self.bus.log(self.job_id, f"{reason.capitalize()} requested — saving checkpoint and halting…")
                self.store.set_status(
                    self.job_id,
                    JobStatus.PAUSING if reason == "pause" else JobStatus.RUNNING,
                )
                control_obj.should_save = True
                control_obj.should_training_stop = True
            return control_obj

    return StudioCallback()
