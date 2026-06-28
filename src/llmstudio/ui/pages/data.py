"""Data tab: upload → map fields → validate → prepare a training-ready dataset."""

from __future__ import annotations

import gradio as gr

from llmstudio.core.data import SUPPORTED_EXT, FieldMapping, TaskFormat
from llmstudio.ui.components import ui_kit as ui
from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.data")

_NONE = "(none)"


def _col_choices(columns: list[str]) -> list[str]:
    return [_NONE, *columns]


def _val(choice: str):
    return None if choice in (None, _NONE, "") else choice


def render(studio) -> None:
    raw_state = gr.State(None)

    gr.HTML(ui.section("Prepare your data", "Upload, map your columns, then validate & prepare a training-ready dataset.", eyebrow="Step 1"))
    gr.HTML(
        ui.callout(
            "Supported: <code>csv, json, jsonl, xlsx, txt, pdf, docx</code>. Don't worry about formatting — "
            "map your columns to <strong>instruction / input / output</strong>, or let the assistant infer it.",
            kind="tip",
        )
    )

    files = gr.File(
        label="Upload data files",
        file_count="multiple",
        file_types=[e for e in sorted(SUPPORTED_EXT)],
        type="filepath",
    )
    load_btn = gr.Button("Load files", variant="primary")

    preview_md = gr.Markdown(visible=True)
    sample_df = gr.Dataframe(label="Sample rows", interactive=False, wrap=True, visible=False)

    gr.HTML(ui.section("Field mapping", "Tell the studio which columns are which — or let the assistant infer it."))
    task_format = gr.Dropdown(
        choices=[f.value for f in TaskFormat],
        value=TaskFormat.INSTRUCTION.value,
        label="Task format",
    )
    with gr.Row():
        instruction_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="Instruction column")
        input_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="Input column (optional)")
        output_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="Output column")
    with gr.Row():
        system_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="System column (optional)")
        text_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="Text column (completion)")
        messages_field = gr.Dropdown(choices=[_NONE], value=_NONE, label="Messages column (chat)")
    system_prompt = gr.Textbox(label="Static system prompt (optional)", lines=2)
    suggest_btn = gr.Button("Suggest mapping with AI", size="sm", elem_classes=["ls-ai-btn"])
    mapping_note = gr.Markdown(visible=False)

    with gr.Row():
        ds_name = gr.Textbox(label="Dataset name", placeholder="e.g. support-bot-v1", scale=2)
        eval_ratio = gr.Slider(0.0, 0.5, value=studio.settings.training.eval_ratio, step=0.01, label="Eval split")
        max_seq = gr.Number(value=studio.settings.training.max_seq_length, label="Max sequence length", precision=0)
    prepare_btn = gr.Button("Validate & Prepare dataset", variant="primary")
    report_md = gr.Markdown(visible=False)

    with gr.Accordion("Prepared datasets", open=False):
        datasets_table = gr.Dataframe(
            headers=["Name", "Format", "Train", "Eval", "ID"],
            interactive=False,
            value=_dataset_rows(studio),
        )
        refresh_ds = gr.Button("Refresh", size="sm")

    field_dropdowns = [instruction_field, input_field, output_field, system_field, text_field, messages_field]

    # -- load ---------------------------------------------------------------
    def on_load(file_paths):
        if not file_paths:
            return (None, gr.update(value="_No files selected._"), gr.update(visible=False),
                    *[gr.update() for _ in field_dropdowns], gr.update())
        upload_dir, raw = studio.data.stage_uploads(file_paths)
        info = studio.data.preview(raw)
        cols = info["columns"]
        guess = studio.data.suggest_mapping(raw).mapping
        md = f"**Loaded {info['n_records']} records** · columns: `{', '.join(cols) or 'n/a'}`"
        if info["notes"]:
            md += "\n\n" + "\n".join(f"- ⚠️ {n}" for n in info["notes"])
        import pandas as pd

        df = pd.DataFrame(info["sample"])
        return (
            raw,
            gr.update(value=md),
            gr.update(value=df, visible=True),
            gr.update(choices=_col_choices(cols), value=guess.instruction_field or _NONE),
            gr.update(choices=_col_choices(cols), value=guess.input_field or _NONE),
            gr.update(choices=_col_choices(cols), value=guess.output_field or _NONE),
            gr.update(choices=_col_choices(cols), value=guess.system_field or _NONE),
            gr.update(choices=_col_choices(cols), value=guess.text_field or _NONE),
            gr.update(choices=_col_choices(cols), value=guess.messages_field or _NONE),
            gr.update(value=guess.task_format.value),
        )

    load_btn.click(
        on_load,
        inputs=files,
        outputs=[raw_state, preview_md, sample_df, *field_dropdowns, task_format],
    )

    # -- AI suggest ---------------------------------------------------------
    def on_suggest(raw):
        if raw is None:
            return (*[gr.update() for _ in field_dropdowns], gr.update(value="_Load files first._", visible=True), gr.update())
        suggestion = studio.data.suggest_mapping(raw)
        m = suggestion.mapping
        note = f"**Mapping source:** {suggestion.source}" + (f" — {suggestion.rationale}" if suggestion.rationale else "")
        return (
            gr.update(value=m.instruction_field or _NONE),
            gr.update(value=m.input_field or _NONE),
            gr.update(value=m.output_field or _NONE),
            gr.update(value=m.system_field or _NONE),
            gr.update(value=m.text_field or _NONE),
            gr.update(value=m.messages_field or _NONE),
            gr.update(value=note, visible=True),
            gr.update(value=m.task_format.value),
        )

    suggest_btn.click(
        on_suggest,
        inputs=raw_state,
        outputs=[*field_dropdowns, mapping_note, task_format],
    )

    # -- prepare ------------------------------------------------------------
    def on_prepare(raw, fmt, instr, inp, out, sysf, textf, msgsf, sysprompt, name, ratio, seq):
        if raw is None:
            return gr.update(value="❌ Load files first.", visible=True), gr.update()
        if not name or not name.strip():
            return gr.update(value="❌ Please enter a dataset name.", visible=True), gr.update()
        mapping = FieldMapping(
            task_format=TaskFormat(fmt),
            instruction_field=_val(instr),
            input_field=_val(inp),
            output_field=_val(out),
            system_field=_val(sysf),
            text_field=_val(textf),
            messages_field=_val(msgsf),
            system_prompt=(sysprompt or None),
        )
        try:
            prepared = studio.data.prepare(
                raw, mapping, name=name.strip(), eval_ratio=float(ratio), max_seq_length=int(seq)
            )
        except Exception as exc:
            return gr.update(value=f"❌ Preparation failed: {exc}", visible=True), gr.update()
        report = prepared.report.to_markdown() if prepared.report else ""
        status = "✅" if (prepared.report is None or prepared.report.ok) else "⚠️"
        md = (
            f"{status} **Prepared `{prepared.name}`** — {prepared.n_train} train / {prepared.n_eval} eval "
            f"(`{prepared.dataset_id}`)\n\n{report}"
        )
        return gr.update(value=md, visible=True), gr.update(value=_dataset_rows(studio))

    prepare_btn.click(
        on_prepare,
        inputs=[raw_state, task_format, *field_dropdowns, system_prompt, ds_name, eval_ratio, max_seq],
        outputs=[report_md, datasets_table],
    )

    refresh_ds.click(lambda: gr.update(value=_dataset_rows(studio)), outputs=datasets_table)


def _dataset_rows(studio) -> list[list[str]]:
    rows = []
    for ds in studio.data.list_datasets():
        rows.append([ds.name, ds.task_format.value, str(ds.n_train), str(ds.n_eval), ds.dataset_id])
    return rows
