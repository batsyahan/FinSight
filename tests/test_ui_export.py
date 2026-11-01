from pathlib import Path

import pytest

from finsight.config import Settings
from finsight.export import export, memo_to_html_fragment, to_html
from finsight.graph.workflow import run_analysis

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_html_export_has_citation_tooltips_and_escapes_model_output():
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    page = to_html(state)
    assert page.startswith("<!doctype html>") and '<abbr class="cite" title="' in page
    state.memo += '\n\nInjected <script>alert(1)</script> <img src=x onerror=alert(2)> **bold** [S1]'
    frag = memo_to_html_fragment(state.memo, state)
    assert "<script>" not in frag and "<img" not in frag and "&lt;script&gt;" in frag
    assert "<strong>bold</strong>" in frag


def test_markdown_export_is_the_memo():
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    content, ext = export(state, "md")
    assert ext == "md" and content.decode() == state.memo


def test_pdf_export():
    pytest.importorskip("reportlab")
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    state.memo += "\nUnicode check: 52‑week, ₹1,185, −3.2% [S1]."
    content, ext = export(state, "pdf")
    assert ext == "pdf" and content[:5] == b"%PDF-" and len(content) > 1500
