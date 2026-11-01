"""Chunking and retrieval.

BM25 (pure Python, no ML dependencies) is the default. If FINSIGHT_EMBED_MODEL is set, chunks are
also embedded through an OpenAI-compatible /embeddings endpoint and the two rankings are merged
with reciprocal rank fusion (hybrid search). Indexes are cached on disk by content hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finsight.rag.loaders import Page, expand_paths, load

STOP = set("""a an the and or of to in on for with by from at as is are was were be been being this that these those
it its our we us you your their they them he she his her not no but if then than so such into over under about
which who whom whose what when where why how all any each other some more most can could may might will would
shall should has have had do does did also per""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    text: str


def chunk_pages(pages: list[Page], size: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pg in pages:
        words = pg.text.split()
        step = max(1, size - overlap)
        for start in range(0, max(1, len(words)), step):
            piece = " ".join(words[start:start + size])
            if piece.strip():
                chunks.append(Chunk(len(chunks), pg.doc, pg.page, piece))
            if start + size >= len(words):
                break
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(map(len, docs)) / self.N if self.N else 0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


def _embed(texts: list[str]) -> list[list[float]] | None:
    model = os.getenv("FINSIGHT_EMBED_MODEL", "").strip()
    if not model:
        return None
    from finsight.llm.base import post_with_retry

    base = os.getenv("FINSIGHT_EMBED_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    key = os.getenv("FINSIGHT_EMBED_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    headers = {"content-type": "application/json", **({"authorization": f"Bearer {key}"} if key else {})}
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 64):
        data = post_with_retry(f"{base}/embeddings", headers, {"model": model, "input": texts[i:i + 64]})
        vectors += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return vectors


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class FilingsIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vectors = vectors

    @classmethod
    def build(cls, paths: str | list[str], cache_dir: str | Path | None = ".finsight_cache") -> "FilingsIndex":
        files = expand_paths(paths)
        if not files:
            raise ValueError("no supported filings found (pdf, txt, md, html)")
        h = hashlib.sha256()
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        h.update(os.getenv("FINSIGHT_EMBED_MODEL", "").encode())
        cache = Path(cache_dir) / f"filings_{h.hexdigest()[:16]}.json" if cache_dir else None
        if cache and cache.exists():
            raw = json.loads(cache.read_text(encoding="utf-8"))
            return cls([Chunk(**c) for c in raw["chunks"]], raw.get("vectors"))
        chunks = chunk_pages([pg for f in files for pg in load(f)])
        vectors = _embed([c.text for c in chunks])
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"chunks": [asdict(c) for c in chunks], "vectors": vectors}), encoding="utf-8")
        return cls(chunks, vectors)

    @property
    def docs(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        bm = self.bm25.scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: bm[i], reverse=True)
        method = "bm25"
        if self.vectors:
            qv = _embed([query])
            if qv:
                sims = [_cos(qv[0], v) for v in self.vectors]
                dense = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
                rrf: dict[int, float] = {}
                for ranking in (order, dense):
                    for rank, i in enumerate(ranking[:50]):
                        rrf[i] = rrf.get(i, 0.0) + 1 / (60 + rank)
                order = sorted(rrf, key=rrf.get, reverse=True)
                method = "hybrid (bm25 + embeddings, rrf)"
        hits = [i for i in order if bm[i] > 0 or method != "bm25"][:k]
        return [{"doc": self.chunks[i].doc, "page": self.chunks[i].page, "score": round(bm[i], 3),
                 "text": self.chunks[i].text, "method": method} for i in hits]
