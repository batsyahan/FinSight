"""Load filings into pages: PDF (via pypdf), plain text / Markdown (pages split on form feeds
or '--- page N ---' markers), and HTML (tags stripped)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}
_PAGE_MARK = re.compile(r"^\s*-{2,}\s*page\s+(\d+)\s*-{2,}\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Page:
    doc: str
    page: int
    text: str


def expand_paths(spec: str | list[str]) -> list[Path]:
    """Accept a comma-separated string, a list, files or directories."""
    items = spec.split(",") if isinstance(spec, str) else spec
    out: list[Path] = []
    for it in items:
        p = Path(str(it).strip())
        if not str(it).strip():
            continue
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED)
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(f"filing not found: {p}")
    return out


def load(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PDF support needs pypdf: pip install 'finsight[rag]'") from e
        reader = PdfReader(str(path))
        return [Page(path.name, i + 1, (pg.extract_text() or "").strip()) for i, pg in enumerate(reader.pages)]
    text = path.read_text(encoding="utf-8", errors="replace")
    if ext in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    if "\f" in text:
        parts = text.split("\f")
        return [Page(path.name, i + 1, p.strip()) for i, p in enumerate(parts) if p.strip()]
    marks = list(_PAGE_MARK.finditer(text))
    if marks:
        pages = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()
            if body:
                pages.append(Page(path.name, int(m.group(1)), body))
        return pages
    return [Page(path.name, 1, text.strip())]
