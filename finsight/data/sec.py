"""SEC EDGAR adapter (official endpoints only) for US-listed companies.

SEC requires a descriptive User-Agent with contact details: set SEC_USER_AGENT, e.g.
"Finsight research tool you@example.com". Requests are few, cached and time-limited, well
within SEC's fair-access guidance (max 10 requests/second).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_cache: dict[str, tuple[float, Any]] = {}
TTL = 24 * 3600


def is_configured() -> bool:
    return bool(os.getenv("SEC_USER_AGENT", "").strip())


def is_us_ticker(ticker: str) -> bool:
    return "." not in ticker and not ticker.startswith("^") and "=" not in ticker


def _get(url: str) -> Any:
    if not is_configured():
        raise RuntimeError("SEC_USER_AGENT is not set")
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    r = requests.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"], "Accept": "application/json"},
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    _cache[url] = (time.time(), data)
    return data


def ticker_map() -> dict[str, int]:
    data = _get(TICKERS_URL)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def recent_filings(ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), limit: int = 8) -> list[dict[str, Any]]:
    cik = ticker_map().get(ticker.upper())
    if cik is None:
        return []
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).get("filings", {}).get("recent", {})
    out = []
    for form, date, acc, doc in zip(recent.get("form", []), recent.get("filingDate", []),
                                    recent.get("accessionNumber", []), recent.get("primaryDocument", [])):
        if form in forms:
            out.append({"form": form, "date": date, "accession": acc,
                        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
                        "source": "SEC EDGAR"})
        if len(out) >= limit:
            break
    return out
