"""Peer comparison: find a company's sector peers and compare them on the same metrics.

Peer groups are a curated list of NSE large caps by sector (plus a few US mega caps).
Users can override with explicit tickers. Offline sample data gets synthetic peers.
"""

from __future__ import annotations

import hashlib
from statistics import median
from typing import Any

from finsight.data.providers import MarketDataProvider
from finsight.tools.quant import fundamental_ratios, price_stats

PEER_GROUPS: dict[str, list[str]] = {
    "Indian IT services": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Indian private & PSU banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                                   "INDUSINDBK.NS"],
    "Indian FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "TATACONSUM.NS"],
    "Indian beverages": ["VBL.NS", "TATACONSUM.NS", "UBL.NS", "RADICO.NS"],
    "Indian autos": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS"],
    "Indian pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "Indian energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "Indian telecom": ["BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS"],
    "Indian cement": ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS"],
    "Indian NBFCs": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS"],
    "US mega-cap tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
}

SAMPLE_UNIVERSE = ["ACME", "GLOBEX", "INITECH", "UMBRELLA", "HOOLI", "STARK", "WAYNE", "DEMO"]

# metric -> True if higher is better (used to rank the subject against peers)
COMPARE_METRICS: dict[str, bool] = {
    "return_1y_pct": True,
    "ann_volatility_pct": False,
    "sharpe": True,
    "max_drawdown_pct": True,  # less negative is better
    "revenue_growth_pct": True,
    "net_margin_pct": True,
    "roe_pct": True,
    "debt_to_equity": False,
    "pe_ratio": False,  # cheaper is "better" for this ranking; context still matters
}


def _base(t: str) -> str:
    return t.upper().replace(".NS", "").replace(".BO", "")


def find_peers(ticker: str, provider: MarketDataProvider, limit: int = 4) -> dict[str, Any]:
    t = ticker.upper()
    if provider.name == "sample":
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        pool = [x for x in SAMPLE_UNIVERSE if x != t]
        start = seed % len(pool)
        peers = (pool[start:] + pool[:start])[:limit]
        return {"ticker": t, "group": "Synthetic sample peers", "peers": peers, "source": "sample universe"}
    for group, members in PEER_GROUPS.items():
        bases = [_base(m) for m in members]
        if _base(t) in bases:
            suffix = ".BO" if t.endswith(".BO") else ""
            peers = [m if not suffix else _base(m) + suffix for m in members if _base(m) != _base(t)]
            return {"ticker": t, "group": group, "peers": peers[:limit], "source": "curated sector list"}
    return {"ticker": t, "group": None, "peers": [],
            "source": "no curated group; pass peer tickers to compare_peers explicitly"}


def compare_peers(ticker: str, peers: list[str], provider: MarketDataProvider) -> dict[str, Any]:
    """Fetch the same metrics for the subject and each peer; rank the subject per metric."""
    rows: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    for t in [ticker.upper(), *[p.upper() for p in peers if p.upper() != ticker.upper()]]:
        try:
            ps = price_stats(provider.prices(t, "2y"))  # same window as the Quant, so figures agree
            ratios = fundamental_ratios(provider.fundamentals(t))
        except Exception as e:  # one bad peer should not sink the comparison
            failed[t] = f"{type(e).__name__}: {e}"[:120]
            continue
        rows[t] = {k: (ps | ratios).get(k) for k in COMPARE_METRICS}
    if ticker.upper() not in rows:
        raise ValueError(f"could not fetch data for {ticker}: {failed.get(ticker.upper())}")

    subject = rows[ticker.upper()]
    peer_rows = {k: v for k, v in rows.items() if k != ticker.upper()}
    summary: dict[str, Any] = {}
    for metric, higher_better in COMPARE_METRICS.items():
        vals = [r[metric] for r in peer_rows.values() if r.get(metric) is not None]
        sv = subject.get(metric)
        if sv is None or not vals:
            continue
        everyone = sorted([sv, *vals], reverse=higher_better)
        summary[metric] = {
            "subject": sv,
            "peer_median": round(median(vals), 2),
            "rank": everyone.index(sv) + 1,
            "out_of": len(everyone),
        }
    return {"ticker": ticker.upper(), "peers": list(peer_rows), "table": rows, "vs_peers": summary,
            "failed": failed, "rank_note": "rank 1 = best; lower volatility, debt and P/E rank higher"}
