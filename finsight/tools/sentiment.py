"""News sentiment: a transparent finance word-list scorer.

Deterministic and explainable: every headline's score comes with the words that drove it,
so the Sentiment agent (an LLM) interprets scores rather than inventing them. Handles simple
negation ("no growth", "not weak") and treats a few phrases as single terms.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "gain", "gains",
    "growth", "grow", "grows", "record", "strong", "stronger", "upgrade", "upgrades", "upgraded", "outperform",
    "profit", "profits", "expansion", "expands", "expand", "win", "wins", "won", "deal", "contract", "approval",
    "approves", "approved", "buyback", "dividend", "raise", "raises", "raised", "robust", "boost", "boosts",
    "rebound", "recovers", "recovery", "optimistic", "bullish", "accelerate", "accelerates", "milestone",
    "partnership", "invests", "launch", "launches",
}
NEGATIVE = {
    "miss", "misses", "missed", "fall", "falls", "fell", "drop", "drops", "plunge", "plunges", "slump", "slumps",
    "decline", "declines", "loss", "losses", "weak", "weaker", "downgrade", "downgrades", "downgraded", "cut",
    "cuts", "layoff", "layoffs", "probe", "investigation", "fraud", "penalty", "fine", "fined", "lawsuit", "sues",
    "notice", "warning", "warns", "default", "resigns", "exit", "exits", "slowdown", "concern",
    "concerns", "bearish", "volatile", "pressure", "delay", "delays", "scrutiny", "raid",
}
PHRASES = {"steps down": -1.0, "price target raised": 1.0, "price target cut": -1.0, "guidance raised": 1.0,
           "guidance cut": -1.0, "share buyback": 1.0, "rate cut": 0.5}
NEGATORS = {"no", "not", "never", "without", "despite", "fails", "lack"}
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")


def score_headline(text: str) -> dict[str, Any]:
    t = text.lower().replace("[sample]", "")
    hits: list[str] = []
    score = 0.0
    for phrase, w in PHRASES.items():
        if phrase in t:
            score += w
            hits.append(phrase)
            t = t.replace(phrase, " ")
    words = _WORD.findall(t)
    for i, w in enumerate(words):
        polarity = 1.0 if w in POSITIVE else -1.0 if w in NEGATIVE else 0.0
        if not polarity:
            continue
        if any(x in NEGATORS for x in words[max(0, i - 2):i]):
            polarity = -polarity
            w = f"not {w}"
        score += polarity
        hits.append(w)
    norm = max(-1.0, min(1.0, score / 2))  # two clear signals saturate the scale
    label = "positive" if norm > 0.15 else "negative" if norm < -0.15 else "neutral"
    return {"score": round(norm, 2), "label": label, "terms": hits}


def _when(item: dict[str, Any]) -> datetime | None:
    d = item.get("date")
    if d is None:
        return None
    try:
        if isinstance(d, (int, float)):
            return datetime.fromtimestamp(d)
        return datetime.fromisoformat(str(d).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def news_sentiment(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for it in items:
        s = score_headline(it.get("title", ""))
        scored.append({"title": it.get("title", "")[:140], "date": str(it.get("date", ""))[:10], **s,
                       "_when": _when(it)})
    if not scored:
        return {"headlines": 0, "net_sentiment": 0.0, "note": "no headlines available"}
    n = len(scored)
    pos = sum(1 for s in scored if s["label"] == "positive")
    neg = sum(1 for s in scored if s["label"] == "negative")
    dated = sorted([s for s in scored if s["_when"]], key=lambda s: s["_when"])
    trend = None
    if len(dated) >= 4:
        half = len(dated) // 2
        older = sum(s["score"] for s in dated[:half]) / half
        newer = sum(s["score"] for s in dated[half:]) / (len(dated) - half)
        trend = {"older_avg": round(older, 2), "recent_avg": round(newer, 2),
                 "direction": "improving" if newer - older > 0.1 else "worsening" if older - newer > 0.1 else "stable"}
    for s in scored:
        s.pop("_when")
    return {
        "headlines": n,
        "positive": pos,
        "negative": neg,
        "neutral": n - pos - neg,
        "net_sentiment": round((pos - neg) / n, 2),
        "avg_score": round(sum(s["score"] for s in scored) / n, 2),
        "trend": trend,
        "scored": scored,
        "method": "finance word list with negation handling; score in [-1, 1] per headline",
    }
