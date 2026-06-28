"""Models tab: browse the registry, inspect details, and delete models."""

from __future__ import annotations

import gradio as gr

from llmstudio.ui.components import ui_kit as ui
from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.registry")


def render(studio) -> None:
    gr.HTML(ui.section("Your fine-tuned models", "Browse, inspect, and manage everything you've trained.", eyebrow="Step 5"))
    gr.HTML(
        ui.callout(
            "Each model is saved with its base model, dataset lineage, metrics, and full training config. "
            "Load any model on the <strong>Inference</strong> tab to chat with it.",
            kind="info",
        )
    )

    table = gr.Dataframe(
        headers=["Name", "Base", "Kind", "Quant", "Created", "ID"],
        value=studio.models.table_rows(),
        interactive=False,
        wrap=True,
    )
    with gr.Row():
        model_dd = gr.Dropdown(choices=studio.models.choices(), label="Select a model", scale=3)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0)

    details = gr.JSON(label="Details")

    with gr.Accordion("⚠️ Danger zone", open=False):
        remove_files = gr.Checkbox(value=False, label="Also delete model files from disk")
        delete_btn = gr.Button("Delete model", variant="stop")
        delete_msg = gr.Markdown(visible=False)

    def on_select(model_id):
        if not model_id:
            return {}
        return studio.models.details(model_id)

    model_dd.change(on_select, inputs=model_dd, outputs=details)

    def on_refresh():
        return (
            gr.update(value=studio.models.table_rows()),
            gr.update(choices=studio.models.choices()),
        )

    refresh_btn.click(on_refresh, outputs=[table, model_dd])

    def on_delete(model_id, rm):
        if not model_id:
            return gr.update(value="❌ Select a model first.", visible=True), gr.update(), gr.update(), {}
        ok = studio.models.delete(model_id, remove_files=bool(rm))
        msg = f"{'✅ Deleted' if ok else '❌ Could not delete'} `{model_id}`."
        return (
            gr.update(value=msg, visible=True),
            gr.update(value=studio.models.table_rows()),
            gr.update(choices=studio.models.choices(), value=None),
            {},
        )

    delete_btn.click(on_delete, inputs=[model_dd, remove_files], outputs=[delete_msg, table, model_dd, details])
