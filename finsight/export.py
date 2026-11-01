"""Export a memo as standalone HTML (citations become hover tooltips) or PDF.

The memo is model-written text, so it is HTML-escaped first and then converted with a small,
safe Markdown subset (headings, bold, bullets, paragraphs). Nothing in a memo can inject markup.
PDF export uses ReportLab (pip install "finsight[pdf]").
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


def to_pdf(state: AnalysisState, verification: dict | None = None) -> bytes:
    """Memo, Critic verification report and evidence list as a PDF (ReportLab)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as e:  # pragma: no cover
        raise RuntimeError('PDF export needs reportlab: pip install "finsight[pdf]"') from e
    import io

    from finsight.verify import verify_memo

    verification = verification or verify_memo(state.memo, state.evidence)
    ss = getSampleStyleSheet()
    lime = colors.HexColor("#6C8F1F")
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=lime, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))

    def para(text: str, style: ParagraphStyle = body) -> Paragraph:
        safe = html.escape(text)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        return Paragraph(safe, style)

    story = [para(f"FinSight research memo · {'approved' if state.approved else 'NOT approved'} by the Critic · "
                  f"{state.revisions} revision(s) · educational research, not financial advice", small), Spacer(1, 4)]
    for raw in state.memo.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        head = re.match(r"^(#{1,3})\s+(.*)$", line)
        if head:
            story.append(para(head.group(2), h1 if len(head.group(1)) == 1 else h2))
        elif re.match(r"^\s*[-*]\s+", line):
            story.append(para("• " + re.sub(r"^\s*[-*]\s+", "", line)))
        else:
            story.append(para(line))

    story += [Spacer(1, 8), para("Critic verification report", h2),
              para(f"{verification['stats'].get('claims_verified', 0)} of {verification['stats'].get('claims', 0)} "
                   f"claims verified against cited evidence; {verification['stats'].get('unsupported_numbers', 0)} "
                   f"unsupported figure(s).", small)]
    rows = [["Claim", "Outcome", "Cites", "Explanation"]]
    for c in verification.get("claims", [])[:60]:
        rows.append([Paragraph(html.escape(c["text"][:160]), small), c["outcome"], ", ".join(c["citations"]),
                     Paragraph(html.escape(c["explanation"][:140]), small)])
    if len(rows) > 1:
        tbl = Table(rows, colWidths=[78 * mm, 22 * mm, 18 * mm, 52 * mm], repeatRows=1)
        tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                                 ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
                                 ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(tbl)
    story += [Spacer(1, 8), para("Evidence", h2)]
    story += [para(f"[{e.id}] {e.title} ({e.source})", small) for e in state.evidence.items.values()]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title=f"{state.ticker} research memo", author="FinSight").build(story)
    return buf.getvalue()


def export(state: AnalysisState, fmt: str) -> tuple[bytes, str]:
    """Return (content, file extension) for 'md', 'html' or 'pdf'."""
    if fmt == "html":
        return to_html(state).encode("utf-8"), "html"
    if fmt == "pdf":
        return to_pdf(state), "pdf"
    return state.memo.encode("utf-8"), "md"


__all__ = ["to_html", "to_pdf", "export", "memo_to_html_fragment"]
