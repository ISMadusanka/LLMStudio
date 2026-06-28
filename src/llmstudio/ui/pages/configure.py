"""Configure tab: model choice, GPU-aware LoRA/QLoRA plan, hyperparameters, advisor.

Produces a validated TrainingConfig stored in the shared ``cfg_state`` for the
Train tab to launch.
"""

from __future__ import annotations

import gradio as gr

from llmstudio.core.training.config import ExportFormat, FinetuneMethod, TrainingConfig
from llmstudio.ui.components import ui_kit as ui
from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.configure")

# Order MUST match the inputs/outputs lists below.
HP_ORDER = [
    "method", "max_seq_length", "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_train_epochs", "max_steps", "learning_rate", "lr_scheduler_type", "warmup_ratio",
    "weight_decay", "lora_r", "lora_alpha", "lora_dropout", "use_rslora", "optim",
    "neftune_noise_alpha", "packing", "train_on_responses_only", "save_steps", "save_total_limit",
    "logging_steps", "eval_strategy", "eval_steps", "export_format", "gguf_quantization", "seed",
]


def render(studio, cfg_state: gr.State) -> None:
    gr.HTML(ui.section("Configure your fine-tune", "Choose a model & dataset, get a GPU-aware plan, then tune (or let the Advisor).", eyebrow="Step 2"))
    gr.HTML(
        ui.callout(
            "Pick your dataset and base model, then click <strong>Analyze GPU &amp; recommend settings</strong> "
            "for a LoRA/QLoRA plan that fits your VRAM. New here? Use <strong>Ask the Advisor</strong> and keep the defaults.",
            kind="info",
        )
    )

    with gr.Row():
        dataset_dd = gr.Dropdown(choices=studio.data.dataset_choices(), label="Dataset", scale=2)
        model_dd = gr.Dropdown(choices=studio.catalog.choices(), label="Base model", scale=2)
        refresh_btn = gr.Button("Refresh", size="sm", scale=0)
    recommend_btn = gr.Button("Analyze GPU & recommend settings", variant="primary")
    rec_md = gr.Markdown(visible=False)

    with gr.Accordion("🎛️ Hyperparameters", open=True):
        with gr.Row():
            method = gr.Dropdown([m.value for m in FinetuneMethod], value="qlora", label="Method")
            max_seq_length = gr.Number(value=2048, label="Max seq length", precision=0)
            per_device_train_batch_size = gr.Number(value=2, label="Batch size / device", precision=0)
            gradient_accumulation_steps = gr.Number(value=4, label="Grad accumulation", precision=0)
        with gr.Row():
            num_train_epochs = gr.Number(value=1.0, label="Epochs")
            max_steps = gr.Number(value=-1, label="Max steps (-1 = use epochs)", precision=0)
            learning_rate = gr.Number(value=2e-4, label="Learning rate")
            lr_scheduler_type = gr.Dropdown(
                ["linear", "cosine", "constant", "constant_with_warmup", "cosine_with_restarts"],
                value="linear", label="LR scheduler",
            )
        with gr.Row():
            warmup_ratio = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Warmup ratio")
            weight_decay = gr.Number(value=0.01, label="Weight decay")
            optim = gr.Dropdown(
                ["adamw_8bit", "paged_adamw_8bit", "adamw_torch", "adamw_torch_fused"],
                value="adamw_8bit", label="Optimizer",
            )
        with gr.Row():
            lora_r = gr.Slider(1, 256, value=16, step=1, label="LoRA rank (r)")
            lora_alpha = gr.Number(value=16, label="LoRA alpha", precision=0)
            lora_dropout = gr.Slider(0.0, 0.9, value=0.0, step=0.01, label="LoRA dropout")
            use_rslora = gr.Checkbox(value=False, label="Use rsLoRA")
        with gr.Row():
            neftune_noise_alpha = gr.Number(value=0, label="NEFTune α (0 = off)")
            packing = gr.Checkbox(value=False, label="Sequence packing")
            train_on_responses_only = gr.Checkbox(value=True, label="Train on responses only")
        with gr.Row():
            save_steps = gr.Number(value=50, label="Save every N steps", precision=0)
            save_total_limit = gr.Number(value=3, label="Keep N checkpoints", precision=0)
            logging_steps = gr.Number(value=1, label="Log every N steps", precision=0)
        with gr.Row():
            eval_strategy = gr.Dropdown(["steps", "epoch", "no"], value="steps", label="Eval strategy")
            eval_steps = gr.Number(value=50, label="Eval every N steps", precision=0)
        with gr.Row():
            export_format = gr.Dropdown([f.value for f in ExportFormat], value="lora", label="Export format")
            gguf_quantization = gr.Dropdown(["q4_k_m", "q5_k_m", "q8_0", "f16"], value="q4_k_m", label="GGUF quant")
            seed = gr.Number(value=3407, label="Seed", precision=0)

    with gr.Row():
        advise_btn = gr.Button("Ask the Advisor for hyperparameters", elem_classes=["ls-ai-btn"])
        save_btn = gr.Button("Save configuration", variant="primary")
    advisor_md = gr.Markdown(visible=False)
    save_md = gr.Markdown(visible=False)

    hp_components = [
        method, max_seq_length, per_device_train_batch_size, gradient_accumulation_steps,
        num_train_epochs, max_steps, learning_rate, lr_scheduler_type, warmup_ratio,
        weight_decay, lora_r, lora_alpha, lora_dropout, use_rslora, optim,
        neftune_noise_alpha, packing, train_on_responses_only, save_steps, save_total_limit,
        logging_steps, eval_strategy, eval_steps, export_format, gguf_quantization, seed,
    ]

    # -- refresh dropdowns --------------------------------------------------
    refresh_btn.click(
        lambda: (gr.update(choices=studio.data.dataset_choices()), gr.update(choices=studio.catalog.choices())),
        outputs=[dataset_dd, model_dd],
    )

    # -- recommend ----------------------------------------------------------
    def on_recommend(model_key, dataset_id):
        if not model_key or not dataset_id:
            return (gr.update(value="❌ Choose a dataset and a base model first.", visible=True),
                    None, *[gr.update() for _ in hp_components])
        cfg, rec = studio.training.build_default_config(model_key, dataset_id)
        md = _recommendation_markdown(rec)
        return (gr.update(value=md, visible=True), cfg.to_dict(), *_config_updates(cfg))

    recommend_btn.click(
        on_recommend,
        inputs=[model_dd, dataset_dd],
        outputs=[rec_md, cfg_state, *hp_components],
    )

    # -- advisor ------------------------------------------------------------
    def on_advise(model_key, dataset_id, *values):
        if not model_key or not dataset_id:
            return gr.update(value="❌ Choose a dataset and a base model first.", visible=True), None, *[gr.update() for _ in hp_components]
        cfg = _build_config(studio, model_key, dataset_id, values)
        advice = studio.training.advise(cfg)
        new_cfg = studio.training.apply_advice(cfg, advice)
        note = f"**Advisor ({advice.source}):** {advice.rationale}"
        return gr.update(value=note, visible=True), new_cfg.to_dict(), *_config_updates(new_cfg)

    advise_btn.click(
        on_advise,
        inputs=[model_dd, dataset_dd, *hp_components],
        outputs=[advisor_md, cfg_state, *hp_components],
    )

    # -- save ---------------------------------------------------------------
    def on_save(model_key, dataset_id, *values):
        if not model_key or not dataset_id:
            return gr.update(value="❌ Choose a dataset and a base model first.", visible=True), None
        try:
            cfg = _build_config(studio, model_key, dataset_id, values)
        except Exception as exc:
            return gr.update(value=f"❌ Invalid configuration: {exc}", visible=True), None
        msg = (
            f"✅ Configuration saved for **{cfg.base_model_key}** on dataset `{cfg.dataset_id}` "
            f"({cfg.method.value}, effective batch {cfg.effective_batch_size}). "
            f"Go to the **🚀 Train** tab to launch."
        )
        return gr.update(value=msg, visible=True), cfg.to_dict()

    save_btn.click(
        on_save,
        inputs=[model_dd, dataset_dd, *hp_components],
        outputs=[save_md, cfg_state],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_config(studio, model_key: str, dataset_id: str, values) -> TrainingConfig:
    vals = dict(zip(HP_ORDER, values))
    entry = studio.catalog.find(model_key)
    data: dict = {
        "base_model_key": model_key,
        "dataset_id": dataset_id,
        "chat_template": entry.chat_template if entry else "chatml",
        "use_gradient_checkpointing": "unsloth",
    }
    for key, value in vals.items():
        if key == "neftune_noise_alpha" and (value is None or float(value) <= 0):
            continue
        data[key] = value
    ds = studio.data.get_dataset(dataset_id)
    if ds is not None:
        data["dataset_dir"] = str(ds.directory)
    return TrainingConfig.from_dict(data)


def _config_updates(cfg: TrainingConfig) -> list:
    d = cfg.to_dict()
    updates = []
    for attr in HP_ORDER:
        v = d.get(attr)
        if attr == "neftune_noise_alpha" and v is None:
            v = 0
        updates.append(gr.update(value=v))
    return updates


def _recommendation_markdown(rec) -> str:
    lines = [f"### {rec.headline()}", ""]
    for r in rec.rationale:
        lines.append(f"- {r}")
    for w in rec.warnings:
        lines.append(f"- ⚠️ {w}")
    est = rec.estimate
    lines.append(
        f"\n**VRAM estimate:** weights {est.weights_gb:.1f} + activations {est.activations_gb:.1f} "
        f"+ optimizer {est.adapter_optimizer_gb:.1f} + overhead {est.overhead_gb:.1f} = "
        f"**~{est.total_gb:.1f} GB** (usable budget {rec.usable_vram_gb:.1f} GB)."
    )
    return "\n".join(lines)
