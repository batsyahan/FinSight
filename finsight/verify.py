"""Deterministic grounding checks for a memo, used by the Critic agent.

Rules:
1. Every citation like [S3] or [S1, S4] must refer to evidence that exists.
2. Every number in a sentence must match a number in the evidence *that sentence cites*,
   to the precision written (unit words such as crore/billion allow scaling).
3. Every sentence that contains a number must carry at least one citation.

Headings, years, small counters and window labels ("52-week") are exempt. This is what makes FinSight's
"every claim is cited" promise checkable instead of a hope.
"""

from __future__ import annotations

import re
from typing import Any

from finsight.guardrails import advice_issues
from finsight.state import EvidenceStore

CITE_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
NUM_RE = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?")
CURRENCY_PATTERNS = {
    "USD": r"(?:US)?\$\s?\d|\bUSD\b",
    "INR": r"₹|\bINR\b|\bRs\.?\s?\d",
    "EUR": r"€|\bEUR\b",
    "GBP": r"£|\bGBP\b",
}
PERIOD_LABEL_RE = re.compile(r"[- ]?(?:days?|weeks?|months?|years?|yrs?|d|w|y)\b", re.IGNORECASE)
# A figure may only be scaled when the memo names the unit right after it ("20.16 billion",
# "4,117 crore"). Allowing every scale for every number made random figures match by chance.
UNIT_SCALES = {
    "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
    "million": 1e6, "mn": 1e6, "m": 1e6, "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12, "t": 1e12,
}
UNIT_RE = re.compile(r"\s*(" + "|".join(sorted(UNIT_SCALES, key=len, reverse=True)) + r")\b", re.IGNORECASE)


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


def number_supported(value: float, evidence_numbers: list[float], scale: float = 1.0, decimals: int = 2) -> bool:
    """True if ``value`` (as written, in units of ``scale``) matches some evidence number to the
    precision it was written with: one unit in the last written digit ("37.7" matches 37.6-37.8,
    "48%" matches 47-49). This allows rounding or truncation but not nearby, unrelated figures."""
    tol = 10.0 ** (-decimals) + 1e-9
    for ev in evidence_numbers:
        shown = ev / scale
        for sign in (1, -1):  # "fell 12%" vs evidence -12.0
            if abs(sign * value - shown) <= tol:
                return True
    return False


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(*_-])|\n+", text)  # "[S1]" after a full stop stays attached
    return [p.strip() for p in parts if p.strip()]


_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                         "\u2212": "-", "\u00a0": " ", "\u202f": " "})


def _claim(i: int, sentence: str, cites: list[str], figures: list[dict[str, Any]], outcome: str,
           explanation: str) -> dict[str, Any]:
    """One checkable claim: a memo sentence with its citations and per-figure results."""
    kind = "quantitative" if figures else "qualitative"
    return {"id": f"C{i + 1}", "text": CITE_RE.sub("", sentence).strip(" -*")[:400], "type": kind,
            "citations": cites, "figures": figures, "outcome": outcome, "explanation": explanation}


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
    claims: list[dict[str, Any]] = []
    for sent in split_sentences(body):
        text_wo_cites = CITE_RE.sub("", sent)
        sent_ids_all = [c for grp in CITE_RE.findall(sent) for c in re.split(r"\s*,\s*", grp)]
        nums = []
        for m in NUM_RE.finditer(text_wo_cites):
            tok = m.group(0)
            v = _parse_num(tok)
            if v is None or _is_exempt(v, tok):
                continue
            # window labels like "52-week high" or "200-day average" are not data points
            if PERIOD_LABEL_RE.match(text_wo_cites, m.end()):
                continue
            unit = UNIT_RE.match(text_wo_cites, m.end())
            decimals = len(tok.split(".")[1]) if "." in tok else 0
            nums.append((tok, v, UNIT_SCALES[unit.group(1).lower()] if unit else 1.0, decimals))
        if not nums:
            if sent_ids_all:  # qualitative claim with a citation: only the LLM critic can judge it
                bad = [c for c in sent_ids_all if evidence.get(c) is None]
                claims.append(_claim(len(claims), sent, sent_ids_all, [],
                                     "bad_citation" if bad else "qualitative",
                                     f"cites unknown evidence {', '.join(bad)}" if bad else
                                     "no figures; cited, reviewed by the LLM critic"))
            continue
        sent_ids = sent_ids_all
        scoped = [evidence.get(c) for c in sent_ids if evidence.get(c) is not None]
        if not sent_ids:
            uncited += 1
            issues.append(f"Sentence with figures has no citation: \"{sent[:120]}\"")
        # Citation-scoped grounding: a figure must appear in the evidence its own sentence cites.
        # (Matching against all evidence lets random figures pass by coincidence as evidence grows.)
        scope_numbers = [n for ev in scoped for n in ev.numbers()] if scoped else ev_numbers
        figures: list[dict[str, Any]] = []
        for tok, v, scale, decimals in nums:
            checked += 1
            if number_supported(v, scope_numbers, scale=scale, decimals=decimals):
                figures.append({"figure": tok, "supported": True})
                continue
            unsupported += 1
            elsewhere = [ev.id for ev in evidence.items.values()
                         if number_supported(v, ev.numbers(), scale=scale, decimals=decimals)]
            if scoped and elsewhere:
                why = (f"not in the cited evidence [{', '.join(e.id for e in scoped)}]; "
                       f"it appears in [{', '.join(elsewhere[:3])}]")
                issues.append(f"Figure {tok} in \"{sent[:100]}\" is {why}. Fix the citation.")
            else:
                why = "not supported by any evidence"
                issues.append(f"Figure {tok} in \"{sent[:100]}\" is {why}.")
            figures.append({"figure": tok, "supported": False, "reason": why})
        failed = [f for f in figures if not f["supported"]]
        if not sent_ids:
            outcome, why = "uncited", "contains figures but no citation"
        elif failed:
            outcome, why = "unsupported", "; ".join(f"{f['figure']}: {f['reason']}" for f in failed)
        else:
            outcome, why = "verified", "every figure found in the cited evidence"
        claims.append(_claim(len(claims), sent, sent_ids, figures, outcome, why))

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
    issues += advice_issues(memo)

    # de-duplicate while keeping order
    seen, uniq = set(), []
    for i in issues:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return {
        "passed": not uniq,
        "issues": uniq,
        "claims": claims,
        "stats": {
            "claims": len(claims),
            "claims_verified": sum(c["outcome"] == "verified" for c in claims),
            "numbers_checked": checked,
            "unsupported_numbers": unsupported,
            "uncited_sentences": uncited,
            "citations_used": sorted(cited, key=lambda s: int(s[1:])),
        },
    }
