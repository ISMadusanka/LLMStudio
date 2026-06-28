"""Home tab: dashboard — hero, workflow stepper, live stats, setup & doctor."""

from __future__ import annotations

import gradio as gr

from llmstudio.ui.components import ui_kit as ui
from llmstudio.ui.components.common import stream_task

_STEPS = [
    ("Data", "Upload & structure"),
    ("Configure", "Model & hyperparams"),
    ("Train", "Launch & monitor"),
    ("Inference", "Test checkpoints"),
    ("Models", "Reuse & manage"),
]


def render(studio, header: "gr.HTML | None" = None) -> None:
    gr.HTML(
        ui.hero(
            "Welcome to LLM Studio",
            "Fine-tune your own open-source LLM in five guided steps — no code required.",
        )
    )
    gr.HTML(ui.stepper(_STEPS, active=0))

    stats = gr.HTML(_stats_html(studio))

    gr.HTML(
        ui.section(
            "Get set up",
            "Create the workspace and download the assistant model. This is a one-time step.",
            eyebrow="Setup",
        )
    )
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(
                ui.callout(
                    "Run <strong>Setup</strong> once to prepare folders and fetch the assistant model "
                    "(Qwen2.5-Instruct). Base models download later, on demand, when you start a run.",
                    kind="tip",
                )
            )
        with gr.Column(scale=2):
            with gr.Row():
                setup_btn = gr.Button("Run setup", variant="primary")
                doctor_btn = gr.Button("Doctor")
            refresh_btn = gr.Button("Refresh status", size="sm")

    setup_log = gr.Textbox(label="Setup / Doctor output", lines=10, interactive=False, show_copy_button=True)

    # -- handlers -----------------------------------------------------------
    def _refresh():
        from llmstudio.ui.app import _status_pills
        from llmstudio.version import __version__

        return ui.header_bar(__version__, _status_pills(studio)), _stats_html(studio)

    refresh_targets = [header, stats] if header is not None else [stats]

    def _refresh_outputs():
        return _refresh() if header is not None else (_stats_html(studio),)

    refresh_btn.click(_refresh_outputs, outputs=refresh_targets)

    def _run_setup():
        yield from stream_task(lambda progress: studio.system.setup(progress=progress))

    setup_evt = setup_btn.click(_run_setup, outputs=setup_log)
    setup_evt.then(_refresh_outputs, outputs=refresh_targets)

    def _run_doctor():
        checks = studio.system.doctor()
        lines = ["Environment doctor", ""]
        for c in checks:
            lines.append(f"[{'OK ' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        return "\n".join(lines)

    doctor_btn.click(_run_doctor, outputs=setup_log)


def _stats_html(studio) -> str:
    # GPU
    try:
        gpu = studio.system.gpu_report()
        if gpu.available and gpu.primary is not None:
            gpu_card = ui.stat_card("GPU", f"{gpu.max_free_gb:.0f} GB", gpu.primary.name, "", "green")
        else:
            gpu_card = ui.stat_card("GPU", "—", "No CUDA device", "", "amber")
    except Exception:
        gpu_card = ui.stat_card("GPU", "?", "unavailable", "", "amber")

    try:
        n_datasets = len(studio.data.list_datasets())
    except Exception:
        n_datasets = 0
    try:
        n_models = len(studio.models.list_models())
    except Exception:
        n_models = 0
    try:
        jobs = studio.training.list_jobs()
        n_jobs = len(jobs)
        n_active = sum(1 for j in jobs if j.is_active)
        jobs_sub = f"{n_active} active" if n_active else "none running"
    except Exception:
        n_jobs, jobs_sub = 0, "—"

    assistant_ready = False
    try:
        assistant_ready = studio.assistant.available()
    except Exception:
        pass

    cards = [
        gpu_card,
        ui.stat_card("Datasets", n_datasets, "prepared", "", "indigo"),
        ui.stat_card("Models", n_models, "fine-tuned", "", "violet"),
        ui.stat_card("Jobs", n_jobs, jobs_sub, "", "blue"),
        ui.stat_card(
            "Assistant",
            "Ready" if assistant_ready else "Off",
            studio.settings.assistant.model_id.split("/")[-1] if assistant_ready else "install [train]",
            "",
            "green" if assistant_ready else "amber",
        ),
    ]
    return ui.stat_grid(cards)
