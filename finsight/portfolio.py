"""Portfolio mode: risk analytics for a set of holdings, plus a grounded LLM summary.

    finsight portfolio "INFY.NS:40,HDFCBANK.NS:30,ITC.NS:30"

Metrics: return, CAGR, volatility, Sharpe, max drawdown, 1-day historical and parametric
Value at Risk (95%/99%), Expected Shortfall (CVaR), correlation matrix, diversification
ratio, each holding's share of portfolio risk, and sector concentration (HHI).
The summary is written by the LLM from this evidence and checked by the same grounding
verifier as single-stock memos.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from finsight.data.providers import MarketDataProvider
from finsight.state import EvidenceStore
from finsight.tools.quant import price_stats
from finsight.verify import verify_memo

TRADING_DAYS = 252


def parse_holdings(spec: str) -> dict[str, float]:
    """'INFY.NS:40, ITC.NS:30' -> weights summing to 1. Missing weights mean equal weight."""
    raw: dict[str, float | None] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            t, w = part.rsplit(":", 1)
            raw[t.strip().upper()] = float(w)
        else:
            raw[part.upper()] = None
    if not raw:
        raise ValueError("no holdings given")
    if any(v is None for v in raw.values()):
        return {t: round(1 / len(raw), 6) for t in raw}
    if any(v < 0 for v in raw.values()):  # type: ignore[operator]
        raise ValueError("weights must be non-negative (long-only)")
    total = sum(raw.values())  # type: ignore[arg-type]
    if total <= 0:
        raise ValueError("weights must sum to more than zero")
    return {t: v / total for t, v in raw.items()}  # type: ignore[operator]


def analyze_portfolio(holdings: dict[str, float], provider: MarketDataProvider, period: str = "2y",
                      value: float = 1_000_000, risk_free: float = 0.06) -> dict[str, Any]:
    prices, failed, sectors = {}, {}, {}
    for t in holdings:
        try:
            prices[t] = provider.prices(t, period)
            sectors[t] = (provider.profile(t) or {}).get("sector") or "Unknown"
        except Exception as e:
            failed[t] = f"{type(e).__name__}: {e}"[:120]
    if not prices:
        raise ValueError(f"no price data for any holding: {failed}")
    w = pd.Series({t: holdings[t] for t in prices})
    w = w / w.sum()  # renormalise if a holding failed

    px = pd.DataFrame(prices).dropna()
    rets = px.pct_change().dropna()
    port = rets @ w  # daily rebalanced to target weights
    equity = (1 + port).cumprod()

    years = len(port) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1)
    vol = float(port.std() * math.sqrt(TRADING_DAYS))
    sharpe = (float(port.mean()) * TRADING_DAYS - risk_free) / vol if vol else 0.0
    mdd = float((equity / equity.cummax() - 1).min())

    def var_block(conf: float) -> dict[str, float]:
        q = float(np.quantile(port, 1 - conf))
        tail = port[port <= q]
        z = NormalDist().inv_cdf(1 - conf)
        param = float(port.mean() + z * port.std())
        return {
            "confidence_pct": round(conf * 100),
            "historical_var_pct": round(-q * 100, 2),
            "parametric_var_pct": round(-param * 100, 2),
            "expected_shortfall_pct": round(-float(tail.mean()) * 100, 2) if len(tail) else None,
            "historical_var_amount": round(-q * value, 0),
        }

    cov = rets.cov() * TRADING_DAYS
    port_var = float(w @ cov @ w)
    contrib = (w * (cov @ w)) / port_var if port_var else w * 0
    indiv_vol = rets.std() * math.sqrt(TRADING_DAYS)
    corr = rets.corr()
    n = len(w)
    avg_corr = float((corr.values.sum() - n) / (n * (n - 1))) if n > 1 else 1.0

    sector_w: dict[str, float] = {}
    for t, wt in w.items():
        sector_w[sectors[t]] = sector_w.get(sectors[t], 0.0) + float(wt)
    hhi = sum(v * v for v in sector_w.values())

    holdings_out = {}
    for t in w.index:
        st = price_stats(px[t], risk_free)
        holdings_out[t] = {
            "weight_pct": round(float(w[t]) * 100, 2),
            "sector": sectors[t],
            "return_1y_pct": st["return_1y_pct"],
            "ann_volatility_pct": st["ann_volatility_pct"],
            "risk_contribution_pct": round(float(contrib[t]) * 100, 2),
        }

    flags = []
    top = max(holdings_out.items(), key=lambda kv: kv[1]["weight_pct"])
    if top[1]["weight_pct"] > 40:
        flags.append({"id": "single_name_concentration", "severity": "medium", "limit": 40,
                      "message": f"{top[0]} is more than 40% of the portfolio."})
    big_sector = max(sector_w.items(), key=lambda kv: kv[1])
    if n > 1 and big_sector[1] > 0.5:
        flags.append({"id": "sector_concentration", "severity": "medium", "limit": 50,
                      "message": f"{big_sector[0]} is more than 50% of the portfolio."})
    if n > 1 and avg_corr > 0.7:
        flags.append({"id": "high_correlation", "severity": "medium", "limit": 0.7,
                      "message": "Holdings move closely together, so diversification is limited."})
    v95 = var_block(0.95)
    if v95["historical_var_pct"] > 3:
        flags.append({"id": "high_var", "severity": "high", "limit": 3,
                      "message": "1-day 95% VaR exceeds 3% of portfolio value."})

    return {
        "holdings": holdings_out,
        "failed": failed,
        "period_start": str(px.index[0].date()),
        "period_end": str(px.index[-1].date()),
        "portfolio_value": value,
        "metrics": {
            "total_return_pct": round(total * 100, 2),
            "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2) if years > 0 else None,
            "ann_volatility_pct": round(vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(mdd * 100, 2),
            "avg_pairwise_correlation": round(avg_corr, 2),
            "diversification_ratio": round(float(w @ indiv_vol) / math.sqrt(port_var), 2) if port_var else None,
            "sector_hhi": round(hhi, 3),
            "risk_free_pct": round(risk_free * 100, 2),
        },
        "var_1d_95": v95,
        "var_1d_99": var_block(0.99),
        "sector_weights_pct": {k: round(v * 100, 2) for k, v in sorted(sector_w.items(), key=lambda kv: -kv[1])},
        "correlation": {a: {b: round(float(corr.loc[a, b]), 2) for b in corr.columns} for a in corr.index},
        "flags": flags,
    }


PORTFOLIO_WRITER = """You are a portfolio risk analyst. Write a short Markdown summary of the portfolio.

Sections: ## Overview, ## Risk (VaR, drawdown, volatility), ## Diversification (correlation,
sectors, risk contribution), ## Flags, ## Disclaimer.

Hard rules: every sentence with a figure ends with a citation like [S1]; use only figures from the
evidence, written the same way; do not do arithmetic; no buy/sell instructions. 150-250 words."""


def portfolio_summary(analysis: dict[str, Any], llm: Any, retries: int = 1) -> dict[str, Any]:
    """LLM-written summary, verified against the analysis like any memo."""
    ev = EvidenceStore()
    ev.add("portfolio_metrics", "Portfolio return and risk metrics",
           analysis["metrics"] | {"period_start": analysis["period_start"], "period_end": analysis["period_end"]})
    ev.add("portfolio_var", "1-day Value at Risk and Expected Shortfall",
           {"var_95": analysis["var_1d_95"], "var_99": analysis["var_1d_99"],
            "portfolio_value": analysis["portfolio_value"]})
    ev.add("portfolio_holdings", "Holdings, weights, sectors and risk contribution",
           {"holdings": analysis["holdings"], "sector_weights_pct": analysis["sector_weights_pct"]})
    ev.add("portfolio_flags", "Portfolio risk flags", {"flags": analysis["flags"]})
    feedback = ""
    for attempt in range(retries + 1):
        reply = llm.complete(PORTFOLIO_WRITER, [{"role": "user", "content": f"Evidence:\n{ev.to_prompt()}{feedback}"}],
                             role="portfolio", context={"analysis": analysis, "evidence": ev}, max_tokens=1200)
        check = verify_memo(reply, ev)
        if check["passed"]:
            break
        feedback = "\n\nYour previous summary failed these checks; fix them:\n- " + "\n- ".join(check["issues"][:8])
    return {"summary": reply.strip(), "grounding": check, "attempts": attempt + 1,
            "evidence": {k: {"title": v.title, "data": v.data} for k, v in ev.items.items()}}


def portfolio_markdown(analysis: dict[str, Any], summary: dict[str, Any] | None = None) -> str:
    m = analysis["metrics"]
    v95, v99 = analysis["var_1d_95"], analysis["var_1d_99"]
    L = [f"# Portfolio report ({analysis['period_start']} to {analysis['period_end']})", ""]
    if summary:
        L += [summary["summary"], "", "---", ""]
    L += ["## Holdings", "", "| Ticker | Weight % | Sector | 1y return % | Volatility % | Share of risk % |",
          "|---|---|---|---|---|---|"]
    for t, h in analysis["holdings"].items():
        L.append(f"| {t} | {h['weight_pct']} | {h['sector']} | {h['return_1y_pct']} | {h['ann_volatility_pct']} | "
                 f"{h['risk_contribution_pct']} |")
    L += ["", "## Metrics", "", "| Metric | Value |", "|---|---|",
          f"| Total return | {m['total_return_pct']}% |", f"| CAGR | {m['cagr_pct']}% |",
          f"| Volatility | {m['ann_volatility_pct']}% |", f"| Sharpe | {m['sharpe']} |",
          f"| Max drawdown | {m['max_drawdown_pct']}% |",
          f"| 1-day VaR 95% (historical / parametric) | {v95['historical_var_pct']}% / {v95['parametric_var_pct']}% |",
          f"| 1-day VaR 99% (historical) | {v99['historical_var_pct']}% |",
          f"| Expected shortfall 95% | {v95['expected_shortfall_pct']}% |",
          f"| Avg pairwise correlation | {m['avg_pairwise_correlation']} |",
          f"| Diversification ratio | {m['diversification_ratio']} |", f"| Sector HHI | {m['sector_hhi']} |", ""]
    if analysis["flags"]:
        L += ["## Flags", ""] + [f"- **{f['severity']}**: {f['message']}" for f in analysis["flags"]] + [""]
    if analysis["failed"]:
        L += ["Could not load: " + ", ".join(analysis["failed"]), ""]
    return "\n".join(L)
