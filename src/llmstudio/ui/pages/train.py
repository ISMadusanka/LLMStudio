"""Train tab: launch, live metric tiles + charts/logs, and pause/resume/cancel."""

from __future__ import annotations

import gradio as gr

from llmstudio.core.training.config import TrainingConfig
from llmstudio.ui.components import ui_kit as ui
from llmstudio.ui.components.common import metrics_dataframe, status_badge
from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.train")

_EMPTY_STATUS = ui.callout("Select a job below, or start a new run.", kind="info")


def render(studio, cfg_state: gr.State, job_state: gr.State) -> None:
    gr.HTML(ui.section("Train", "Launch the configuration you saved, then watch it train live.", eyebrow="Step 3"))
    gr.HTML(
        ui.callout(
            "On start, the base model downloads once, the assistant is unloaded to free VRAM, "
            "and checkpoints are saved automatically so a crash is recoverable.",
            kind="tip",
        )
    )

    with gr.Row():
        run_name = gr.Textbox(label="Run name", placeholder="e.g. support-bot-v1", scale=3)
        start_btn = gr.Button("Start training", variant="primary", scale=1)
    start_msg = gr.HTML(visible=False)

    with gr.Row():
        job_dd = gr.Dropdown(choices=_job_choices(studio), label="Active / past jobs", scale=4)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0, min_width=80)

    status_html = gr.HTML(_EMPTY_STATUS)
    tiles_html = gr.HTML(_tiles_html(None, []))
    progress_html = gr.HTML(ui.progress_bar(0.0, "step 0", "0%"))

    with gr.Row():
        loss_plot = gr.LinePlot(x="step", y="loss", title="Training loss", height=280, overlay_point=True)
        lr_plot = gr.LinePlot(x="step", y="learning_rate", title="Learning rate", height=280)

    logs_box = gr.Textbox(label="Logs", lines=12, interactive=False, max_lines=12, autoscroll=True, show_copy_button=True)

    with gr.Row():
        pause_btn = gr.Button("Pause")
        resume_btn = gr.Button("Resume")
        cancel_btn = gr.Button("Cancel", variant="stop")

    refresh_outputs = [status_html, tiles_html, progress_html, loss_plot, lr_plot, logs_box]

    # -- start --------------------------------------------------------------
    def on_start(cfg_dict, name):
        if not cfg_dict:
            return gr.update(value=ui.callout("Save a configuration on the <strong>Configure</strong> tab first.", "warning"), visible=True), None, gr.update()
        try:
            cfg = TrainingConfig.from_dict(cfg_dict)
            job = studio.training.start(cfg, (name or "").strip() or cfg.base_model_key)
        except Exception as exc:
            return gr.update(value=ui.callout(f"Could not start: {exc}", "danger"), visible=True), None, gr.update()
        return (
            gr.update(value=ui.callout(f"Started <strong>{job.name}</strong> (<code>{job.id}</code>).", "success"), visible=True),
            job.id,
            gr.update(choices=_job_choices(studio), value=job.id),
        )

    start_evt = start_btn.click(on_start, inputs=[cfg_state, run_name], outputs=[start_msg, job_state, job_dd])

    # -- select / refresh ---------------------------------------------------
    job_dd.change(lambda jid: jid, inputs=job_dd, outputs=job_state)
    refresh_btn.click(lambda: gr.update(choices=_job_choices(studio)), outputs=job_dd)

    # -- controls -----------------------------------------------------------
    pause_btn.click(lambda jid: _act(studio.training.pause, jid), inputs=job_state, outputs=status_html)
    resume_btn.click(lambda jid: _act(studio.training.resume, jid), inputs=job_state, outputs=status_html)
    cancel_btn.click(lambda jid: _act(studio.training.cancel, jid), inputs=job_state, outputs=status_html)

    # -- live refresh -------------------------------------------------------
    def refresh(job_id):
        if not job_id:
            return (_EMPTY_STATUS, _tiles_html(None, []), ui.progress_bar(0.0, "step 0", "0%"),
                    metrics_dataframe([], "loss"), metrics_dataframe([], "learning_rate"), "")
        job = studio.training.get_job(job_id)
        if job is None:
            return (ui.callout("Job not found.", "danger"), _tiles_html(None, []),
                    ui.progress_bar(0.0), metrics_dataframe([], "loss"), metrics_dataframe([], "learning_rate"), "")
        metrics = studio.training.metrics(job_id)
        logs = "\n".join(studio.training.log_lines(job_id, limit=200))
        pct = int(job.progress * 100)
        return (
            _status_html(job),
            _tiles_html(job, metrics),
            ui.progress_bar(job.progress, f"step {job.current_step}/{job.total_steps or '?'}", f"{pct}%"),
            metrics_dataframe(metrics, "loss"),
            metrics_dataframe(metrics, "learning_rate"),
            logs,
        )

    refresh_btn.click(refresh, inputs=job_state, outputs=refresh_outputs)
    job_dd.change(refresh, inputs=job_state, outputs=refresh_outputs)
    start_evt.then(refresh, inputs=job_state, outputs=refresh_outputs)

    if hasattr(gr, "Timer"):
        timer = gr.Timer(2.0)
        timer.tick(refresh, inputs=job_state, outputs=refresh_outputs)


# --------------------------------------------------------------------------- helpers
def _job_choices(studio) -> list[tuple[str, str]]:
    return [(f"{j.name or j.id}  ·  {j.status.value}", j.id) for j in studio.training.list_jobs()]


def _status_html(job) -> str:
    badge = status_badge(job.status)
    head = f'<div style="display:flex;align-items:center;gap:10px;margin:2px 0 6px;"><strong style="font-size:1.1rem;">{job.name or job.id}</strong>{badge}</div>'
    extra = ""
    if job.error:
        extra += ui.callout(str(job.error), "danger")
    if job.registered_model_id:
        extra += ui.callout(
            f"Saved as model <code>{job.registered_model_id}</code> — chat with it on the <strong>Inference</strong> tab.",
            "success",
        )
    return head + extra


def _tiles_html(job, metrics) -> str:
    last = metrics[-1] if metrics else {}
    loss = last.get("loss")
    lr = last.get("learning_rate")
    loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
    lr_s = f"{lr:.2e}" if isinstance(lr, (int, float)) else "—"
    step_s = f"{job.current_step}/{job.total_steps or '?'}" if job else "0/?"
    epoch_s = f"{job.current_epoch:.2f}" if job else "0.00"
    cards = [
        ui.stat_card("Loss", loss_s, "latest", "📉", "indigo"),
        ui.stat_card("Learning rate", lr_s, "current", "🎛️", "violet"),
        ui.stat_card("Step", step_s, "optimizer steps", "👣", "blue"),
        ui.stat_card("Epoch", epoch_s, "progress", "🔁", "green"),
    ]
    return ui.stat_grid(cards)


def _act(fn, job_id) -> str:
    if not job_id:
        return ui.callout("Select a job first.", "warning")
    job = fn(job_id)
    if job is None:
        return ui.callout("Job not found.", "danger")
    return _status_html(job)
