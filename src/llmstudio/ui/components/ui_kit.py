"""HTML building blocks for a consistent, professional look.

Each function returns an HTML string to drop into a ``gr.HTML`` (or as the
markup of an updating component). Styling lives in ``ui/theme.py`` (the ``ls-*``
classes).
"""

from __future__ import annotations

import base64
import functools
import html
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Optional

from llmstudio.core.utils.logging import get_logger

log = get_logger("ui.assets")

# ui/assets (this file lives at ui/components/ui_kit.py → parents[1] == ui)
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_LOGO_NAMES = ("logo.png", "logo.svg", "logo.jpg", "logo.jpeg", "logo.webp")


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _logo_search_dirs() -> list[Path]:
    """All directories we look in for a logo, most-specific first.

    Robust to install mode: the packaged ``ui/assets`` works for editable
    installs; the repo root and the workspace home cover non-editable installs
    and let users drop a logo without touching the package.
    """
    dirs: list[Path] = [ASSETS_DIR]
    try:
        from llmstudio.config.paths import find_repo_root

        root = find_repo_root()
        dirs += [root / "assets", root / "src" / "llmstudio" / "ui" / "assets"]
    except Exception:
        pass
    try:
        from llmstudio.config import get_settings

        dirs.append(get_settings().home_root)  # $LLMSTUDIO_HOME (or ./workspace)
    except Exception:
        pass
    # De-dupe, preserve order.
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def logo_path() -> Optional[Path]:
    """Locate a logo file. Override with the LLMSTUDIO_LOGO env var (full path)."""
    env = os.environ.get("LLMSTUDIO_LOGO")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        log.warning("LLMSTUDIO_LOGO is set to %s but that file does not exist.", p)
    for d in _logo_search_dirs():
        for name in _LOGO_NAMES:
            p = d / name
            if p.exists():
                return p
    return None


@functools.lru_cache(maxsize=1)
def logo_data_uri() -> Optional[str]:
    """Base64 data URI for the logo (cached). None if no logo file present."""
    p = logo_path()
    if p is None:
        log.info(
            "No logo found. Add 'logo.png' to one of: %s (or set LLMSTUDIO_LOGO). Using default mark.",
            ", ".join(str(d) for d in _logo_search_dirs()),
        )
        return None
    try:
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        log.info("Using logo: %s (%.0f KB)", p, p.stat().st_size / 1024)
        return f"data:{mime};base64,{encoded}"
    except Exception as exc:
        log.warning("Failed to read logo %s: %s", p, exc)
        return None


def chat_avatars() -> tuple[Optional[str], Optional[str]]:
    """(user_avatar_path, assistant_avatar_path) for the chat. None → Gradio default."""
    user = ASSETS_DIR / "avatar-user.svg"
    assistant = ASSETS_DIR / "avatar-assistant.svg"
    return (
        str(user) if user.exists() else None,
        str(assistant) if assistant.exists() else None,
    )


# --------------------------------------------------------------------------- header
def header_bar(version: str, status_pills: str = "") -> str:
    uri = logo_data_uri()
    if uri:
        mark = f'<img class="ls-logo-img" src="{uri}" alt="LLM Studio" />'
        title = ""  # the logo image already contains the wordmark
    else:
        mark = '<div class="ls-logo">🧪</div>'
        title = "<h1>LLM Studio</h1>"
    return f"""
<div class="ls-header ls-fade">
  <div class="ls-brand">
    {mark}
    <div>
      {title}
      <p>No-code fine-tuning for open-source LLMs · v{_esc(version)}</p>
    </div>
  </div>
  <div class="ls-header-status">{status_pills}</div>
</div>
"""


def pill(label: str, state: str = "neutral") -> str:
    """state: ok | warn | off | neutral"""
    cls = {"ok": "ls-pill--ok", "warn": "ls-pill--warn", "off": "ls-pill--off"}.get(state, "")
    return f'<span class="ls-pill {cls}"><span class="ls-dot"></span>{_esc(label)}</span>'


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="ls-badge ls-badge--{kind}">{_esc(text)}</span>'


# --------------------------------------------------------------------------- sections
def section(title: str, subtitle: str = "", eyebrow: str = "") -> str:
    eb = f'<div class="ls-eyebrow">{_esc(eyebrow)}</div>' if eyebrow else ""
    sub = f"<p>{_esc(subtitle)}</p>" if subtitle else ""
    return f'<div class="ls-section ls-fade">{eb}<h2>{_esc(title)}</h2>{sub}</div>'


def hero(title: str, subtitle: str = "") -> str:
    sub = f'<p class="ls-muted" style="margin-top:6px;font-size:.95rem;">{_esc(subtitle)}</p>' if subtitle else ""
    return (
        f'<div class="ls-card ls-card--hero ls-fade">'
        f'<div style="font-size:1.5rem;font-weight:760;letter-spacing:-0.02em;">{_esc(title)}</div>{sub}</div>'
    )


# --------------------------------------------------------------------------- stats
def stat_card(label: str, value: object, sub: str = "", icon: str = "", accent: str = "indigo") -> str:
    ico = f'<div class="ls-stat-icon">{icon}</div>' if icon else ""
    subhtml = f'<div class="ls-stat-sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="ls-card ls-stat ls-stat--{accent} ls-fade">{ico}'
        f'<div class="ls-stat-label">{_esc(label)}</div>'
        f'<div class="ls-stat-value">{_esc(value)}</div>{subhtml}</div>'
    )


def stat_grid(cards: Iterable[str]) -> str:
    return f'<div class="ls-grid">{"".join(cards)}</div>'


# --------------------------------------------------------------------------- callouts
def callout(text: str, kind: str = "info", icon: Optional[str] = None) -> str:
    """kind: info | tip | success | warning | danger"""
    default_icons = {"info": "ℹ️", "tip": "💡", "success": "✅", "warning": "⚠️", "danger": "⛔"}
    ico = icon if icon is not None else default_icons.get(kind, "ℹ️")
    # text may contain simple inline HTML (e.g. <strong>); callers control it.
    return (
        f'<div class="ls-callout ls-callout--{kind} ls-fade">'
        f'<div class="ls-callout-ico">{ico}</div>'
        f'<div class="ls-callout-body">{text}</div></div>'
    )


# --------------------------------------------------------------------------- stepper
def stepper(steps: list[tuple[str, str]], active: int) -> str:
    """steps: list of (title, subtitle). active is 0-based index."""
    items = []
    for i, (title, sub) in enumerate(steps):
        state = "ls-step--active" if i == active else ("ls-step--done" if i < active else "")
        no = "✓" if i < active else str(i + 1)
        items.append(
            f'<div class="ls-step {state}"><div class="ls-step-no">{no}</div>'
            f'<div class="ls-step-txt">{_esc(title)}<small>{_esc(sub)}</small></div></div>'
        )
    return f'<div class="ls-stepper ls-fade">{"".join(items)}</div>'


# --------------------------------------------------------------------------- progress
def progress_bar(fraction: float, left: str = "", right: str = "") -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    meta = ""
    if left or right:
        meta = f'<div class="ls-progress-meta"><span>{_esc(left)}</span><span>{_esc(right)}</span></div>'
    return (
        f'<div class="ls-fade"><div class="ls-progress-wrap">'
        f'<div class="ls-progress-bar" style="width:{pct}%;"></div></div>{meta}</div>'
    )
