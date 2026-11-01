"""Reusable dashboard components. Each returns an HTML string; every piece of dynamic text is
escaped, so data (news titles, memo text, tickers) can never inject markup.

Icons are inline SVGs from the Lucide set (ISC licence), stroke-based to match the reference design.
"""

from __future__ import annotations

import html
import math
from typing import Any, Iterable

_ICON_PATHS = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/>'
              '<polyline points="2 12 12 17 22 12"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
            '<path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/>'
                '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "news": '<path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9'
            'c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5'
              '-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
                '<path d="M3 12A9 3 0 0 0 21 12"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/>'
           '<path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    "list": '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "bookmark": '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>',
    "spark": '<path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/>'
             '<path d="m5.6 5.6 2.8 2.8"/><path d="m15.6 15.6 2.8 2.8"/><path d="m5.6 18.4 2.8-2.8"/><path d="m15.6 8.4 2.8-2.8"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "plug": '<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
}


def icon(name: str, size: int = 16) -> str:
    path = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


LOGO = ('<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="28" height="28" rx="8" fill="#C5F65A"/>'
        '<path d="M7 18.5l4.2-4.6 3.1 2.9L21 9.5" fill="none" stroke="#101211" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/><circle cx="21" cy="9.5" r="1.9" fill="#101211"/></svg>')


def brand() -> str:
    return f'<div class="fs-brand">{LOGO}<span>finsight</span></div>'


def page_header(title: str, subtitle: str = "", back: str = "") -> str:
    b = f'<div class="fs-back">&larr; {esc(back)}</div>' if back else ""
    s = f'<div class="fs-sub">{esc(subtitle)}</div>' if subtitle else ""
    return f'{b}<div class="fs-title">{esc(title)}</div>{s}'


def panel_head(title: str, icon_name: str = "chart", tone: str = "", right_html: str = "") -> str:
    return (f'<div class="fs-panel-head"><div class="fs-panel-title"><span class="fs-icon {tone}">'
            f'{icon(icon_name)}</span>{esc(title)}</div><div>{right_html}</div></div>')


def kpi(value: Any, label: str, icon_name: str, tone: str = "", sub_html: str = "") -> str:
    return (f'<div class="fs-kpi"><span class="fs-icon lg {tone}">{icon(icon_name, 20)}</span><div>'
            f'<div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>{sub_html}</div></div>')


TONES = {"ok": "ok", "warn": "warn", "bad": "bad", "neutral": "neutral", "info": "info"}


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="fs-pill {TONES.get(tone, "neutral")}">{esc(text)}</span>'


def chips(items: Iterable[str]) -> str:
    return "".join(f'<span class="fs-chip">{esc(i)}</span>' for i in items if i)


def change_html(change: float | None, pct: float | None, digits: int = 2) -> str:
    if change is None or pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return '<span class="fs-muted">n/a</span>'
    cls = "fs-up" if change >= 0 else "fs-down"
    sign = "+" if change >= 0 else ""
    return f'<span class="{cls}">{sign}{change:,.{digits}f} ({sign}{pct:.2f}%)</span>'


def pct_html(pct: float | None) -> str:
    if pct is None:
        return '<span class="fs-muted">n/a</span>'
    return f'<span class="{"fs-up" if pct >= 0 else "fs-down"}">{"+" if pct >= 0 else ""}{pct:.2f}%</span>'


def fmt_num(v: Any, digits: int = 2, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    try:
        return f"{float(v):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return esc(v)


def fmt_money(v: Any, currency: str | None = None) -> str:
    """Large amounts in crore for INR, billions otherwise; never invents a currency."""
    if v is None:
        return "n/a"
    v = float(v)
    cur = (currency or "").upper()
    if cur == "INR":
        return f"₹{v / 1e7:,.0f} cr"
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}.get(cur, f"{cur} " if cur else "")
    return f"{sym}{v / 1e9:,.2f} bn"


def timeline(items: list[dict[str, Any]]) -> str:
    """items: name, desc, status (completed|running|queued|flags|revised|failed|skipped), pill, detail."""
    dot = {"completed": "", "running": "run", "queued": "idle", "flags": "warn", "revised": "warn",
           "failed": "bad", "skipped": "idle", "rejected": "bad"}
    tone = {"completed": "ok", "running": "info", "queued": "neutral", "flags": "warn", "revised": "warn",
            "failed": "bad", "skipped": "neutral", "rejected": "bad"}
    rows = []
    for it in items:
        st = it.get("status", "queued")
        rows.append(
            f'<div class="fs-tl-item"><span class="fs-tl-dot {dot.get(st, "idle")}"></span>'
            f'<div><div class="fs-tl-name">{esc(it["name"])}</div><div class="fs-tl-desc">{esc(it.get("desc", ""))}'
            f'</div></div><div>{pill(it.get("pill") or st.title(), tone.get(st, "neutral"))}</div></div>')
    return f'<div class="fs-tl">{"".join(rows)}</div>'


def kv_rows(rows: list[tuple[str, str]]) -> str:
    """Label/value rows; values are pre-formatted *plain* strings and get escaped here."""
    return "".join(f'<div class="fs-row"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>'
                   for k, v in rows)


def table(columns: list[tuple[str, str]], rows: list[dict[str, str]], html_cols: set[str] = frozenset()) -> str:
    """columns: (key, header) where a header starting with '#' is right-aligned numeric. Cells in
    ``html_cols`` must already be safe HTML built by this module; everything else is escaped."""
    head = "".join(f'<th class="{"num" if h.startswith("#") else ""}">{esc(h.lstrip("#"))}</th>' for _, h in columns)
    body = []
    for r in rows:
        cells = []
        for k, h in columns:
            v = r.get(k, "")
            cells.append(f'<td class="{"num" if h.startswith("#") else ""}">{v if k in html_cols else esc(v)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="fs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def bullets(texts: Iterable[str]) -> str:
    return "".join(f'<div class="fs-bullet"><span>{esc(t)}</span></div>' for t in texts)


def check_line(text: str, ok: bool = True) -> str:
    color = "var(--lime)" if ok else "var(--coral)"
    return (f'<div class="fs-check"><span style="color:{color}">{icon("check" if ok else "x", 18)}</span>'
            f'<span>{esc(text)}</span></div>')


def progress(value: float, label: str = "") -> str:
    v = max(0.0, min(1.0, value))
    lab = f'<div class="fs-small fs-muted" style="display:flex;justify-content:space-between"><span>{esc(label)}</span>' \
          f'<span>{v * 100:.0f}%</span></div>' if label else ""
    return f'{lab}<div class="fs-bar"><span style="width:{v * 100:.1f}%"></span></div>'


def empty(text: str) -> str:
    return f'<div class="fs-empty">{esc(text)}</div>'


def note(text: str) -> str:
    return f'<div class="fs-note">{esc(text)}</div>'


def news_item(title: str, publisher: str, when: str, link: str = "") -> str:
    safe_link = esc(link) if link.startswith(("https://", "http://")) else ""
    t = f'<a href="{safe_link}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>' if safe_link else esc(title)
    ext = f'<a href="{safe_link}" target="_blank" rel="noopener noreferrer" class="fs-muted">{icon("external", 15)}</a>' \
        if safe_link else ""
    return (f'<div class="fs-list-item"><span class="fs-icon">{icon("news", 15)}</span><div style="flex:1">'
            f'<div style="font-size:13.5px;font-weight:500">{t}</div><div class="fs-xs fs-muted">{esc(publisher)}'
            f'{" · " if publisher and when else ""}{esc(when)}</div></div>{ext}</div>')


def avatar(initials: str) -> str:
    return f'<div class="fs-avatar">{esc((initials or "FS")[:2].upper())}</div>'
