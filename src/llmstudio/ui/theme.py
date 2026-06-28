"""Gradio theme + design system (CSS) for LLM Studio.

The CSS is built on Gradio's runtime theme variables (``var(--...)``) so the
custom components adapt to both light and dark mode. Custom classes are prefixed
``ls-`` and styled directly (we avoid styling Gradio's internal classes, which
change between versions).
"""

from __future__ import annotations

# System font stack — no external fonts (CSP / offline friendly).
_FONT_STACK = (
    'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, "Noto Sans", sans-serif'
)
_MONO_STACK = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace'


def studio_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        # Component labels: neutral, NOT button-colored. The indigo/violet accent
        # is reserved for buttons & actions only.
        block_label_background_fill="*neutral_100",
        block_label_background_fill_dark="*neutral_800",
        block_label_text_color="*neutral_600",
        block_label_text_color_dark="*neutral_300",
        block_label_border_width="0px",
        block_label_text_weight="600",
        block_title_text_color="*neutral_600",
        block_title_text_color_dark="*neutral_300",
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(15,23,42,0.04)",
        button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
        button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
        button_primary_text_color="white",
        button_large_radius="*radius_lg",
        input_border_width="1px",
    )


CUSTOM_CSS = """
:root {
  --ls-radius: 16px;
  --ls-radius-sm: 10px;
  --ls-gap: 16px;
  /* Build surfaces from translucent tints of the text color so they always
     contrast with the page background in BOTH light and dark mode (the theme's
     --background-fill-primary can render light even in dark mode). */
  --ls-card-bg: color-mix(in srgb, var(--body-text-color) 6%, transparent);
  --ls-card-elev: color-mix(in srgb, var(--body-text-color) 10%, transparent);
  --ls-card-border: color-mix(in srgb, var(--body-text-color) 18%, transparent);
  --ls-muted: var(--body-text-color-subdued);
  --ls-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
  --ls-shadow-lg: 0 10px 30px -12px rgba(15,23,42,0.25);
  --ls-indigo: #6366f1;
  --ls-violet: #8b5cf6;
  --ls-green: #22c55e;
  --ls-amber: #f59e0b;
  --ls-red: #ef4444;
  --ls-blue: #0ea5e9;
}

/* Layout polish */
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }
.ls-fade { animation: ls-fade-in .35s ease both; }
@keyframes ls-fade-in { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: none; } }

/* ---------------- Header bar ---------------- */
.ls-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 18px 22px; margin-bottom: 6px;
  border-radius: var(--ls-radius);
  background: linear-gradient(100deg, rgba(99,102,241,0.14), rgba(139,92,246,0.14));
  border: 1px solid var(--ls-card-border);
}
.ls-brand { display: flex; align-items: center; gap: 14px; }
.ls-logo {
  width: 46px; height: 46px; border-radius: 13px; flex: none;
  display: grid; place-items: center; font-size: 24px;
  background: linear-gradient(135deg, var(--ls-indigo), var(--ls-violet));
  box-shadow: 0 6px 18px -6px rgba(99,102,241,0.7);
}
.ls-logo-img {
  height: 58px; width: auto; max-width: 220px; flex: none;
  border-radius: 12px; display: block; object-fit: contain;
  background: #fff; padding: 4px;
  box-shadow: 0 6px 18px -8px rgba(15,23,42,0.55);
}
.ls-brand h1 { margin: 0; font-size: 1.45rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.1; }
.ls-brand p { margin: 2px 0 0; color: var(--ls-muted); font-size: .86rem; }
.ls-header-status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }

/* ---------------- Pills / badges ---------------- */
.ls-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  border: 1px solid var(--ls-card-border); background: var(--ls-card-elev);
  color: var(--body-text-color); white-space: nowrap;
}
.ls-pill .ls-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ls-muted); }
.ls-pill--ok .ls-dot { background: var(--ls-green); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
.ls-pill--warn .ls-dot { background: var(--ls-amber); box-shadow: 0 0 0 3px rgba(245,158,11,.18); }
.ls-pill--off .ls-dot { background: var(--ls-muted); }
.ls-badge {
  display: inline-block; padding: 3px 9px; border-radius: 999px;
  font-size: .72rem; font-weight: 700; letter-spacing: .01em; color: var(--body-text-color);
}
.ls-badge--neutral { background: color-mix(in srgb, var(--body-text-color) 16%, transparent); }
.ls-badge--indigo  { background: rgba(99,102,241,.30); }
.ls-badge--green   { background: rgba(34,197,94,.30); }
.ls-badge--amber   { background: rgba(245,158,11,.32); }
.ls-badge--red     { background: rgba(239,68,68,.30); }

/* ---------------- Section headers ---------------- */
/* Titles use neutral tones — the indigo/violet accent is reserved for buttons/actions. */
.ls-section { margin: 6px 0 12px; padding-left: 14px; border-left: 3px solid color-mix(in srgb, var(--body-text-color) 24%, transparent); }
.ls-eyebrow { font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.ls-section h2 { margin: 2px 0 0; font-size: 1.3rem; font-weight: 720; letter-spacing: -0.01em; }
.ls-section p { margin: 4px 0 0; color: var(--ls-muted); font-size: .9rem; max-width: 70ch; }

/* ---------------- Cards & grids ---------------- */
.ls-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--ls-gap); }
.ls-card {
  background: var(--ls-card-bg); border: 1px solid var(--ls-card-border);
  border-radius: var(--ls-radius); padding: 18px; box-shadow: var(--ls-shadow);
  color: var(--body-text-color);
}
.ls-card--hero {
  background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(139,92,246,0.06));
}

/* Stat tiles */
.ls-stat { position: relative; overflow: hidden; }
.ls-stat .ls-stat-icon {
  position: absolute; top: 14px; right: 14px; font-size: 1.3rem; opacity: .9;
  width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center;
  background: color-mix(in srgb, var(--ls-indigo) 14%, transparent);
}
.ls-stat .ls-stat-label { font-size: .8rem; font-weight: 600; color: var(--ls-muted); }
.ls-stat .ls-stat-value { font-size: 1.9rem; font-weight: 760; line-height: 1.1; margin-top: 4px; letter-spacing: -0.02em; color: var(--body-text-color); }
.ls-stat .ls-stat-sub { font-size: .78rem; color: var(--ls-muted); margin-top: 4px; }
.ls-stat--indigo .ls-stat-icon { background: rgba(99,102,241,.16); }
.ls-stat--green  .ls-stat-icon { background: rgba(34,197,94,.16); }
.ls-stat--amber  .ls-stat-icon { background: rgba(245,158,11,.18); }
.ls-stat--violet .ls-stat-icon { background: rgba(139,92,246,.16); }
.ls-stat--blue   .ls-stat-icon { background: rgba(14,165,233,.16); }

/* ---------------- Callouts ---------------- */
.ls-callout {
  display: flex; gap: 12px; padding: 13px 15px; border-radius: var(--ls-radius-sm);
  border: 1px solid var(--ls-card-border); border-left-width: 4px;
  background: var(--ls-card-bg); color: var(--body-text-color); font-size: .9rem; margin: 8px 0;
}
.ls-callout .ls-callout-ico { font-size: 1.1rem; line-height: 1.3; }
.ls-callout .ls-callout-body { line-height: 1.45; }
.ls-callout .ls-callout-body strong { font-weight: 700; }
.ls-callout--info    { border-left-color: var(--ls-blue); }
.ls-callout--tip     { border-left-color: var(--ls-violet); }
.ls-callout--success { border-left-color: var(--ls-green); }
.ls-callout--warning { border-left-color: var(--ls-amber); }
.ls-callout--danger  { border-left-color: var(--ls-red); }

/* ---------------- Stepper ---------------- */
.ls-stepper { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
.ls-step {
  flex: 1 1 130px; display: flex; align-items: center; gap: 10px;
  padding: 11px 13px; border-radius: var(--ls-radius-sm);
  border: 1px solid var(--ls-card-border); background: var(--ls-card-bg);
}
.ls-step .ls-step-no {
  width: 26px; height: 26px; border-radius: 50%; flex: none; display: grid; place-items: center;
  font-size: .82rem; font-weight: 700; background: color-mix(in srgb, var(--ls-muted) 16%, transparent);
  color: var(--body-text-color);
}
.ls-step .ls-step-txt { font-size: .84rem; font-weight: 600; line-height: 1.15; color: var(--body-text-color); }
.ls-step .ls-step-txt small { display: block; font-weight: 500; color: var(--ls-muted); font-size: .74rem; }
.ls-step--active { border-color: var(--ls-indigo); box-shadow: 0 0 0 3px rgba(99,102,241,.14); }
.ls-step--active .ls-step-no { background: linear-gradient(135deg, var(--ls-indigo), var(--ls-violet)); color: #fff; }

/* ---------------- Progress bar ---------------- */
.ls-progress-wrap { background: color-mix(in srgb, var(--ls-muted) 18%, transparent); border-radius: 999px; height: 12px; overflow: hidden; }
.ls-progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--ls-indigo), var(--ls-violet)); transition: width .4s ease; }
.ls-progress-meta { display: flex; justify-content: space-between; font-size: .8rem; color: var(--ls-muted); margin-top: 6px; }

/* ---------------- Tabs ---------------- */
.tab-nav button { font-weight: 600 !important; }
.tab-nav button.selected { color: var(--ls-indigo) !important; }

/* ---------------- AI-accent buttons ---------------- */
.ls-ai-btn, .ls-ai-btn button {
  background-image: linear-gradient(90deg, #8b5cf6 0%, #6366f1 32%, #06b6d4 64%, #8b5cf6 100%) !important;
  background-size: 220% 100% !important;
  color: #fff !important;
  border: none !important;
  font-weight: 650 !important;
  box-shadow: 0 4px 18px -5px rgba(139,92,246,0.65) !important;
  animation: ls-ai-shimmer 5s linear infinite;
  transition: box-shadow .2s ease, transform .2s ease, filter .2s ease;
}
.ls-ai-btn::before { content: "✨"; margin-right: 7px; }
.ls-ai-btn:hover, .ls-ai-btn button:hover {
  box-shadow: 0 8px 26px -5px rgba(139,92,246,0.9) !important;
  transform: translateY(-1px); filter: brightness(1.06);
}
@keyframes ls-ai-shimmer {
  0%   { background-position: 0% 50%; }
  100% { background-position: 220% 50%; }
}
@media (prefers-reduced-motion: reduce) { .ls-ai-btn { animation: none; } }

/* ---------------- Chat (ChatGPT / Claude-like) ---------------- */
.ls-chat-wrap { max-width: 860px; margin: 0 auto; width: 100%; }
.ls-chat { border: none !important; background: transparent !important; }
.ls-composer {
  border: 1px solid var(--ls-card-border) !important;
  border-radius: 22px !important;
  background: var(--ls-card-bg) !important;
  padding: 6px 6px 6px 14px !important;
  box-shadow: var(--ls-shadow);
  align-items: flex-end !important;
  gap: 8px !important;
  transition: border-color .15s ease, box-shadow .15s ease;
}
.ls-composer:focus-within {
  border-color: var(--ls-indigo) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.16);
}
.ls-composer textarea {
  border: none !important; background: transparent !important;
  box-shadow: none !important; outline: none !important; resize: none !important;
  padding: 9px 4px !important; font-size: .98rem !important;
}
.ls-send button {
  border-radius: 16px !important;
  background-image: linear-gradient(90deg, var(--ls-indigo), var(--ls-violet)) !important;
  color: #fff !important; border: none !important; font-weight: 650 !important;
}
.ls-chat-actions { margin-top: 6px; align-items: center; }

/* Tidy up dataframes a touch */
.ls-muted { color: var(--ls-muted); font-size: .88rem; }
"""
