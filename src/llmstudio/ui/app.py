"""Builds and launches the LLM Studio Gradio app."""

from __future__ import annotations

from typing import Optional

from llmstudio.core.utils.logging import get_logger
from llmstudio.version import __version__

log = get_logger("ui.app")


def _status_pills(studio) -> str:
    """Live status pills (GPU + assistant) for the header bar."""
    from llmstudio.ui.components import ui_kit as ui

    pills = []
    try:
        gpu = studio.system.gpu_report()
        if gpu.available and gpu.primary is not None:
            pills.append(ui.pill(f"{gpu.primary.name} · {gpu.max_free_gb:.0f} GB free", "ok"))
        else:
            pills.append(ui.pill("No GPU detected", "off"))
    except Exception:
        pills.append(ui.pill("GPU: unknown", "off"))
    try:
        if studio.assistant.available():
            pills.append(ui.pill("Assistant ready", "ok"))
        elif studio.settings.assistant.enabled:
            pills.append(ui.pill("Assistant: install [train]", "warn"))
        else:
            pills.append(ui.pill("Assistant off", "off"))
    except Exception:
        pills.append(ui.pill("Assistant off", "off"))
    return "".join(pills)


def build_app(studio=None):
    """Construct the Gradio Blocks app. Pass a Studio or one is created."""
    import gradio as gr

    from llmstudio.services import get_studio
    from llmstudio.ui.components import ui_kit as ui
    from llmstudio.ui.pages import configure, data, home, inference, registry, train
    from llmstudio.ui.theme import CUSTOM_CSS, studio_theme

    studio = studio or get_studio()

    with gr.Blocks(theme=studio_theme(), css=CUSTOM_CSS, title="LLM Studio", fill_height=True) as demo:
        # Shared cross-tab session state.
        cfg_state = gr.State(None)   # TrainingConfig dict (Configure → Train)
        job_state = gr.State(None)   # current job id (Train → Inference)

        header = gr.HTML(ui.header_bar(__version__, _status_pills(studio)))

        with gr.Tabs():
            with gr.Tab("Home"):
                home.render(studio, header)
            with gr.Tab("Data"):
                data.render(studio)
            with gr.Tab("Configure"):
                configure.render(studio, cfg_state)
            with gr.Tab("Train"):
                train.render(studio, cfg_state, job_state)
            with gr.Tab("Inference"):
                inference.render(studio, job_state)
            with gr.Tab("Models"):
                registry.render(studio)

        gr.HTML(
            '<div style="text-align:center;color:var(--body-text-color-subdued);'
            'font-size:.8rem;padding:18px 0 6px;">LLM Studio · fine-tune open-source LLMs locally '
            '· LoRA/QLoRA via Unsloth</div>'
        )

    return demo


def launch(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    share: Optional[bool] = None,
    studio=None,
) -> None:
    """Build and serve the app, honoring settings unless overridden."""
    from llmstudio.services import get_studio
    from llmstudio.ui.components import ui_kit as ui

    studio = studio or get_studio()
    server = studio.settings.server
    demo = build_app(studio)
    demo.queue()  # enable streaming + concurrent events
    favicon = ui.logo_path()
    try:
        demo.launch(
            server_name=host or server.host,
            server_port=port or server.port,
            share=server.share if share is None else share,
            auth=server.auth_tuple(),
            show_error=server.show_error,
            favicon_path=str(favicon) if favicon else None,
            inbrowser=False,
        )
    finally:
        studio.shutdown()
