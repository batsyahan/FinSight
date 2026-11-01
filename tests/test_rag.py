from pathlib import Path

import pytest

from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.rag.index import BM25, FilingsIndex, chunk_pages, tokenize
from finsight.rag.loaders import expand_paths, load
from finsight.state import EvidenceStore
from finsight.verify import verify_memo

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)
SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "filings"


def test_loaders_split_pages_for_each_format(tmp_path):
    (tmp_path / "a.txt").write_text("page one text\fpage two text", encoding="utf-8")
    (tmp_path / "b.html").write_text("<html><script>x()</script><p>Revenue &amp; margin</p></html>", encoding="utf-8")
    pages = load(tmp_path / "a.txt")
    assert [(p.page, p.text) for p in pages] == [(1, "page one text"), (2, "page two text")]
    assert load(tmp_path / "b.html")[0].text.split() == ["Revenue", "&", "margin"]
    md = load(SAMPLE / "DEMO_annual_report.md")
    assert [p.page for p in md] == [1, 2, 3, 4]
    assert expand_paths(str(tmp_path)) == [tmp_path / "a.txt", tmp_path / "b.html"]
    with pytest.raises(FileNotFoundError):
        expand_paths("nope/missing.pdf")


def test_chunking_overlaps_and_keeps_page_numbers():
    from finsight.rag.loaders import Page

    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_pages([Page("d", 7, words)], size=180, overlap=40)
    assert len(chunks) == 3 and all(c.page == 7 for c in chunks)
    assert chunks[1].text.split()[0] == "w140"  # 40-word overlap


def test_bm25_ranks_relevant_chunk_first():
    docs = [tokenize(t) for t in ["dividend and share buyback approved", "revenue growth guidance for next year",
                                   "risks include currency and regulation"]]
    scores = BM25(docs).scores(tokenize("What is the revenue guidance?"))
    assert scores.index(max(scores)) == 1


def test_filings_index_search_and_cache(tmp_path):
    ix = FilingsIndex.build(str(SAMPLE), cache_dir=tmp_path)
    hits = ix.search("key risks and challenges", k=2)
    assert hits[0]["page"] == 3 and hits[0]["doc"] == "DEMO_annual_report.md"
    assert list(tmp_path.glob("filings_*.json"))  # cached
    again = FilingsIndex.build(str(SAMPLE), cache_dir=tmp_path)
    assert [c.text for c in again.chunks] == [c.text for c in ix.chunks]


def test_figures_quoted_from_filing_text_can_be_verified():
    ev = EvidenceStore()
    ev.add("search_filings", "Filings search", {"results": [{"text": "Revenue grew 23.4 percent in FY2026."}]})
    assert verify_memo("Management reported revenue growth of 23.4% [S1].", ev)["passed"]
    assert not verify_memo("Management reported revenue growth of 31.4% [S1].", ev)["passed"]


def test_pipeline_uses_filings_only_when_provided():
    with_docs = run_analysis("DEMO", "What did management say about guidance?", settings=OFFLINE,
                             engine_name="builtin", filings=str(SAMPLE))
    assert "filings" in with_docs.plan and "## From the Filings" in with_docs.memo and with_docs.approved
    without = run_analysis("DEMO", "What did management say about guidance?", settings=OFFLINE,
                           engine_name="builtin")
    assert "filings" not in without.plan and "## From the Filings" not in without.memo
    forced = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", analysts=["filings", "quant"])
    assert forced.plan == ["filings", "quant"] and "search_filings" not in {e.source for e in forced.evidence.items.values()}


def _tiny_pdf(text_by_page: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page (no PDF library needed)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in text_by_page:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_id = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_filings_keep_page_numbers(tmp_path):
    pytest.importorskip("pypdf")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(_tiny_pdf(["Dividend policy unchanged this year", "Guidance revenue growth of 12 percent"]))
    pages = load(pdf)
    assert [p.page for p in pages] == [1, 2] and "Guidance" in pages[1].text
    hit = FilingsIndex.build(str(pdf), cache_dir=None).search("revenue guidance")[0]
    assert hit["page"] == 2 and hit["doc"] == "report.pdf"
