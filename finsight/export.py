"""Export a memo as standalone HTML (citations become hover tooltips) or PDF.

The memo is model-written text, so it is HTML-escaped first and then converted with a small,
safe Markdown subset (headings, bold, bullets, paragraphs). Nothing in a memo can inject markup.
PDF export uses fpdf2 (pip install "finsight[pdf]").
"""

from __future__ import annotations

import html
import re
from finsight.state import AnalysisState

CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def _evidence_tip(state: AnalysisState, ev_id: str, limit: int = 400) -> str:
    ev = state.evidence.get(ev_id)
    if ev is None:
        return "unknown evidence"
    parts = []
    for k, v in ev.data.items():
        if isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}: {v}")
    return (f"{ev.title} ({ev.source}). " + "; ".join(parts))[:limit]


def _inline(text: str, state: AnalysisState | None) -> str:
    """Escape, then apply bold and citation tooltips. Input is raw memo text."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)

    def cite(m: re.Match) -> str:
        ids = [i.strip() for i in m.group(1).split(",")]
        if state is None:
            return m.group(0)
        tags = [f'<abbr class="cite" title="{html.escape(_evidence_tip(state, i), quote=True)}">{i}</abbr>' for i in ids]
        return "[" + ", ".join(tags) + "]"

    return CITE.sub(cite, out)


def memo_to_html_fragment(memo: str, state: AnalysisState | None = None) -> str:
    lines, out, in_list = memo.splitlines(), [], False
    for raw in lines:
        line = raw.rstrip()
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(bullet.group(1), state)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        head = re.match(r"^(#{1,3})\s+(.*)$", line)
        if head:
            level = len(head.group(1))
            out.append(f"<h{level}>{_inline(head.group(2), state)}</h{level}>")
        elif line.strip():
            out.append(f"<p>{_inline(line, state)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def to_html(state: AnalysisState) -> str:
    body = memo_to_html_fragment(state.memo, state)
    evidence = "\n".join(
        f"<li id='{html.escape(e.id)}'><b>{html.escape(e.id)}</b> {html.escape(e.title)} "
        f"<code>{html.escape(e.source)}</code></li>" for e in state.evidence.items.values())
    status = "Approved by the Critic" if state.approved else "Not approved by the Critic"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(state.ticker)} research memo · FinSight</title>
<style>
  :root {{ --ink:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --accent:#2457c5; --bg:#fff; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --ink:#e8eaed; --muted:#9aa3ad; --line:#30363d; --accent:#7aa2ff; --bg:#0f1115; }} }}
  body {{ font: 16px/1.6 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: var(--bg);
         max-width: 780px; margin: 40px auto; padding: 0 16px; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid var(--line); }}
  .meta {{ color: var(--muted); font-size: .9rem; }}
  abbr.cite {{ color: var(--accent); text-decoration: none; border-bottom: 1px dotted var(--accent); cursor: help; }}
  code {{ color: var(--muted); }} footer {{ margin-top: 3rem; font-size: .85rem; color: var(--muted); }}
  @media print {{ abbr.cite {{ border: none; }} }}
</style></head><body>
<div class="meta">FinSight · {html.escape(status)} · {state.revisions} revision(s) · hover a citation to see its evidence</div>
{body}
<footer><h3>Evidence</h3><ul>{evidence}</ul></footer>
</body></html>"""


def to_pdf(state: AnalysisState) -> bytes:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as e:  # pragma: no cover
        raise RuntimeError('PDF export needs fpdf2: pip install "finsight[pdf]"') from e

    def clean(s: str) -> str:  # core PDF fonts are Latin-1; map common Unicode punctuation
        table = {"‑": "-", "‐": "-", "–": "-", "—": "-", "−": "-", "’": "'",
                 "‘": "'", "“": '"', "”": '"', "…": "...", "₹": "INR ", "→": "->",
                 " ": " ", "•": "-"}
        s = "".join(table.get(ch, ch) for ch in s)
        return s.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(h: float, text: str) -> None:
        # fpdf2 leaves the cursor at the right margin after multi_cell by default; return to the left
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(110, 110, 110)
    line(5, clean(f"FinSight research memo - {'approved' if state.approved else 'not approved'} "
                               f"by the Critic - educational research, not financial advice"))
    pdf.set_text_color(0, 0, 0)
    for raw in state.memo.splitlines():
        line = clean(raw.rstrip())
        if not line.strip():
            pdf.ln(2)
            continue
        head = re.match(r"^(#{1,3})\s+(.*)$", line)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", head.group(2) if head else line)
        if head:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", {1: 16, 2: 13, 3: 11}[len(head.group(1))])
        else:
            pdf.set_font("Helvetica", size=10)
            text = re.sub(r"^\s*[-*]\s+", "- ", text)
        line(6 if head else 5, text)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    line(6, "Evidence")
    pdf.set_font("Helvetica", size=8)
    for e in state.evidence.items.values():
        line(4, clean(f"[{e.id}] {e.title} ({e.source})"))
    return bytes(pdf.output())


def export(state: AnalysisState, fmt: str) -> tuple[bytes, str]:
    """Return (content, file extension) for 'md', 'html' or 'pdf'."""
    if fmt == "html":
        return to_html(state).encode("utf-8"), "html"
    if fmt == "pdf":
        return to_pdf(state), "pdf"
    return state.memo.encode("utf-8"), "md"


__all__ = ["to_html", "to_pdf", "export", "memo_to_html_fragment"]
