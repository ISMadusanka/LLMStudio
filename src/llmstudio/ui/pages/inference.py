"""Inference tab: load a registry model or a paused run's checkpoint, then chat."""

from __future__ import annotations

import threading
import time

import gradio as gr

from llmstudio.core.inference.engine import GenerationParams
from llmstudio.ui.components import ui_kit as ui
from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.inference")

_SRC_MODEL = "Registry model"
_SRC_CKPT = "Run checkpoint"

_GREETING = (
    "## How can I help?\n\n"
    "Load a model in **Model & checkpoint** above, then start chatting.\n\n"
    "Your fine-tuned model's responses stream in as they're generated."
)


def render(studio, job_state: gr.State) -> None:
    gr.HTML(ui.section("Inference", "Chat with a finished model, or pause a run and probe its latest checkpoint.", eyebrow="Step 4"))

    user_av, bot_av = ui.chat_avatars()

    # Model / checkpoint selection — tucked into an accordion so the chat is the hero.
    with gr.Accordion("Model & checkpoint", open=True):
        gr.HTML(
            ui.callout(
                "Inference shares the GPU with training — if a job is active, "
                "<strong>pause it first</strong>, then load its checkpoint here.",
                kind="info",
            )
        )
        with gr.Row():
            source = gr.Radio([_SRC_MODEL, _SRC_CKPT], value=_SRC_MODEL, label="Source", scale=1)
            model_dd = gr.Dropdown(choices=studio.models.choices(), label="Registry model", scale=2, visible=True)
            job_dd = gr.Dropdown(choices=_ckpt_jobs(studio), label="Run (with checkpoint)", scale=2, visible=False)
        with gr.Row():
            load_btn = gr.Button("Load", variant="primary")
            unload_btn = gr.Button("Unload", size="sm")
            refresh_btn = gr.Button("Refresh", size="sm", scale=0)
        load_status = gr.HTML(_loaded_status(studio))

    with gr.Accordion("Generation settings", open=False):
        with gr.Row():
            max_new = gr.Slider(16, 2048, value=256, step=16, label="Max new tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k")
            rep = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty")

    # Conversation — centered column, bubble layout, avatars, empty-state greeting.
    with gr.Column(elem_classes=["ls-chat-wrap"]):
        chatbot = gr.Chatbot(
            type="messages",
            height=540,
            show_label=False,
            layout="bubble",
            avatar_images=(user_av, bot_av),
            show_copy_button=True,
            render_markdown=True,
            placeholder=_GREETING,
            elem_classes=["ls-chat"],
        )
        with gr.Row(elem_classes=["ls-composer"]):
            msg = gr.Textbox(
                placeholder="Message your model…",
                show_label=False,
                container=False,
                lines=1,
                max_lines=6,
                autofocus=True,
                scale=9,
            )
            send = gr.Button("Send", variant="primary", scale=1, min_width=92, elem_classes=["ls-send"])
        with gr.Row(elem_classes=["ls-chat-actions"]):
            gr.Examples(
                examples=[
                    ["Who are you, and what can you help me with?"],
                    ["Summarize the following in two sentences: "],
                    ["Write a short, friendly reply to a customer asking about refunds."],
                ],
                inputs=msg,
                label="Try an example",
            )
            clear = gr.Button("Clear chat", size="sm", scale=0)

    # -- source toggle ------------------------------------------------------
    def on_source(src):
        return gr.update(visible=src == _SRC_MODEL), gr.update(visible=src == _SRC_CKPT)

    source.change(on_source, inputs=source, outputs=[model_dd, job_dd])

    refresh_btn.click(
        lambda: (gr.update(choices=studio.models.choices()), gr.update(choices=_ckpt_jobs(studio))),
        outputs=[model_dd, job_dd],
    )

    # -- load / unload ------------------------------------------------------
    def on_load(src, model_id, job_id):
        # Validate selection up front.
        if src == _SRC_MODEL and not model_id:
            yield ui.callout("Pick a model to load first.", "warning")
            return
        if src == _SRC_CKPT and not job_id:
            yield ui.callout("Pick a run to load first.", "warning")
            return

        # Friendly name for the in-progress message.
        if src == _SRC_MODEL:
            rec = studio.models.get(model_id)
            name = rec.name if rec else model_id
        else:
            job = studio.training.get_job(job_id)
            name = (job.name or job_id) if job else job_id

        stages: list[str] = []
        result: dict = {}

        def progress(message: str) -> None:
            stages.append(message)

        def work() -> None:
            try:
                if src == _SRC_MODEL:
                    result["label"] = studio.inference.load_registered(model_id, progress=progress)
                else:
                    result["label"] = studio.inference.load_checkpoint(job_id, progress=progress)
            except Exception as exc:  # surfaced to the UI below
                result["error"] = str(exc)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        start = time.time()
        dots = 0
        # Stream a live, ticking loader while the model loads in the background.
        while worker.is_alive():
            dots = (dots % 3) + 1
            elapsed = int(time.time() - start)
            stage = stages[-1] if stages else "preparing…"
            yield ui.callout(
                f"Loading <strong>{_esc_html(name)}</strong>{'.' * dots}<br>"
                f"<span class='ls-muted'>{_esc_html(stage)} · {elapsed}s elapsed</span>",
                "info",
            )
            time.sleep(0.5)
        worker.join()

        if result.get("error"):
            yield ui.callout(_esc_html(result["error"]), "danger")
        else:
            yield ui.callout(
                f"Loaded <strong>{_esc_html(result.get('label', name))}</strong> — ready to chat.",
                "success",
            )

    def on_unload():
        studio.inference.unload()
        return '<div class="ls-muted">Model unloaded.</div>'

    load_btn.click(on_load, inputs=[source, model_dd, job_dd], outputs=load_status)
    unload_btn.click(on_unload, outputs=load_status)

    # -- chat ---------------------------------------------------------------
    def respond(message, history, mn, temp, tp, tk, rp):
        history = list(history or [])
        if not message or not message.strip():
            yield history, ""
            return
        history.append({"role": "user", "content": message})
        params = GenerationParams(
            max_new_tokens=int(mn), temperature=float(temp), top_p=float(tp),
            top_k=int(tk), repetition_penalty=float(rp), do_sample=float(temp) > 0,
        )
        model_messages = [{"role": h["role"], "content": h["content"]} for h in history]
        history.append({"role": "assistant", "content": ""})
        try:
            acc = ""
            for chunk in studio.inference.generate_stream(model_messages, params):
                acc += chunk
                history[-1]["content"] = acc
                yield history, ""
            if not acc:
                history[-1]["content"] = "_(empty response)_"
                yield history, ""
        except Exception as exc:
            history[-1]["content"] = f"⚠️ {exc}"
            yield history, ""

    inputs = [msg, chatbot, max_new, temperature, top_p, top_k, rep]
    send.click(respond, inputs=inputs, outputs=[chatbot, msg])
    msg.submit(respond, inputs=inputs, outputs=[chatbot, msg])
    clear.click(lambda: [], outputs=chatbot)


def _ckpt_jobs(studio) -> list[tuple[str, str]]:
    out = []
    for j in studio.training.list_jobs():
        if j.last_checkpoint:
            out.append((f"{j.name or j.id} @ {j.current_step} steps [{j.status.value}]", j.id))
    return out


def _loaded_status(studio) -> str:
    label = studio.inference.current_label
    if label:
        return ui.callout(f"Loaded <strong>{_esc_html(label)}</strong> — ready to chat.", "success")
    return '<div class="ls-muted">No model loaded.</div>'


def _esc_html(text: object) -> str:
    import html

    return html.escape(str(text))
