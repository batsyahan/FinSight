"""Deterministic grounding checks for a memo, used by the Critic agent.

Rules:
1. Every citation like [S3] or [S1, S4] must refer to evidence that exists.
2. Every number in the memo body must match some number in the evidence
   (allowing rounding and unit scaling such as crore/billion).
3. Every sentence that contains a number must carry at least one citation.

Headings, years, small counters and window labels ("52-week") are exempt. This is what makes FinSight's
"every claim is cited" promise checkable instead of a hope.
"""

from __future__ import annotations

import re
from typing import Any

from finsight.state import EvidenceStore

CITE_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
NUM_RE = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?")
CURRENCY_PATTERNS = {
    "USD": r"(?:US)?\$\s?\d|\bUSD\b",
    "INR": r"₹|\bINR\b|\bRs\.?\s?\d",
    "EUR": r"€|\bEUR\b",
    "GBP": r"£|\bGBP\b",
}
PERIOD_LABEL_RE = re.compile(r"[- ]?(?:days?|weeks?|months?|years?|yrs?|d|w|m|y)\b", re.IGNORECASE)
SCALES =[1, 1e2, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12]  # incl. lakh (1e5) and crore (1e7)


def _parse_num(tok: str) -> float | None:
    try:
        return float(tok.replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def _is_exempt(value: float, tok: str) -> bool:
    # years and tiny list/section counters
    if "." not in tok and 1900 <= abs(value) <= 2100:
        return True
    if "." not in tok and abs(value) <= 10 and value == int(value):
        return True
    return False


def number_supported(value: float, evidence_numbers: list[float], rel_tol: float = 0.01) -> bool:
    for ev in evidence_numbers:
        for scale in SCALES:
            for sign in (1, -1):  # "fell 12%" vs evidence -12.0
                target = sign * value * scale
                if ev == 0 and target == 0:
                    return True
                denom = max(abs(ev), 1e-9)
                if abs(target - ev) / denom <= rel_tol or abs(target - ev) <= 0.011 * scale:
                    return True
    return False


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(*_-])|\n+", text)  # "[S1]" after a full stop stays attached
    return [p.strip() for p in parts if p.strip()]


_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                         "\u2212": "-", "\u00a0": " ", "\u202f": " "})


def verify_memo(memo: str, evidence: EvidenceStore) -> dict[str, Any]:
    issues: list[str] = []
    memo = memo.translate(_DASHES)  # models often emit non-breaking hyphens ("52\u2011week") and minus signs
    ev_numbers = evidence.all_numbers()
    # Headings are skipped; every other line is checked, including the disclaimer,
    # so figures cannot be slipped in below a heading that looks exempt.
    body = "\n".join(line for line in memo.splitlines() if not line.strip().startswith("#"))

    cited = set()
    for m in CITE_RE.finditer(memo):
        for cid in re.split(r"\s*,\s*", m.group(1)):
            cited.add(cid)
            if evidence.get(cid) is None:
                issues.append(f"Citation [{cid}] does not match any evidence item.")

    checked = unsupported = uncited = 0
    for sent in split_sentences(body):
        text_wo_cites = CITE_RE.sub("", sent)
        nums = []
        for m in NUM_RE.finditer(text_wo_cites):
            tok = m.group(0)
            v = _parse_num(tok)
            if v is None or _is_exempt(v, tok):
                continue
            # window labels like "52-week high" or "200-day average" are not data points
            if PERIOD_LABEL_RE.match(text_wo_cites, m.end()):
                continue
            nums.append((tok, v))
        if not nums:
            continue
        has_cite = bool(CITE_RE.search(sent))
        if not has_cite:
            uncited += 1
            issues.append(f"Sentence with figures has no citation: \"{sent[:120]}\"")
        for tok, v in nums:
            checked += 1
            if not number_supported(v, ev_numbers):
                unsupported += 1
                issues.append(f"Figure {tok} in \"{sent[:100]}\" is not supported by any evidence.")

    # currency sanity: a currency used in the memo must be one the evidence actually reports
    ev_currencies = {str(v).upper() for ev in evidence.items.values()
                     for k, v in ev.data.items() if "currency" in k and isinstance(v, str) and len(v) == 3}
    if ev_currencies:
        for code, pattern in CURRENCY_PATTERNS.items():
            if code not in ev_currencies and re.search(pattern, body):
                issues.append(f"Memo uses {code} but the evidence reports figures only in "
                              f"{', '.join(sorted(ev_currencies))}.")

    if not cited and evidence.items:
        issues.append("Memo contains no citations at all.")

    # de-duplicate while keeping order
    seen, uniq = set(), []
    for i in issues:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return {
        "passed": not uniq,
        "issues": uniq,
        "stats": {
            "numbers_checked": checked,
            "unsupported_numbers": unsupported,
            "uncited_sentences": uncited,
            "citations_used": sorted(cited, key=lambda s: int(s[1:])),
        },
    }
