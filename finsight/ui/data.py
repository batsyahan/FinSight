"""Data services for the dashboard (no Streamlit imports, so they are unit-testable).

Every value carries its provenance: source, as-of date, currency, and whether it is synthetic sample
data. Integrations are only reported as "connected" after a real, successful check.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from finsight.config import Settings
from finsight.data.providers import MarketDataProvider, get_provider
from finsight.guardrails import sanitize, validate_ticker
from finsight.store import Store
from finsight.tools.quant import fundamental_ratios

RANGES = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "5Y": 1260}
BENCHMARKS = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "NIFTY BANK": "^NSEBANK", "S&P 500": "^GSPC",
              "NASDAQ 100": "^NDX"}

# Names for search (facts about listings, not data). Anything else can be searched by ticker.
COMPANIES = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services", "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra", "LTIM.NS": "LTIMindtree", "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "AXISBANK.NS": "Axis Bank", "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC", "HINDUNILVR.NS": "Hindustan Unilever", "LT.NS": "Larsen & Toubro", "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical", "BAJFINANCE.NS": "Bajaj Finance", "ASIANPAINT.NS": "Asian Paints",
    "NESTLEIND.NS": "Nestle India", "TITAN.NS": "Titan Company", "ULTRACEMCO.NS": "UltraTech Cement",
    "VBL.NS": "Varun Beverages", "ONGC.NS": "Oil and Natural Gas Corporation", "M&M.NS": "Mahindra & Mahindra",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta Platforms",
}


def effective_settings(store: Store) -> Settings:
    """Environment variables (and secrets) first, then choices saved on the Settings page."""
    base = Settings.from_env()
    saved = store.get_setting("app_settings", {}) or {}
    allowed = {k: v for k, v in saved.items() if k in ("llm", "model", "data", "max_revisions") and v not in (None, "")}
    return replace(base, **allowed)


def provider_for(settings: Settings) -> MarketDataProvider:
    return get_provider(settings.data)


def is_sample(provider: MarketDataProvider) -> bool:
    return getattr(provider, "name", "") == "sample"


def search(query: str, limit: int = 8) -> list[tuple[str, str]]:
    q = query.strip().lower()
    if not q:
        return []
    hits = [(t, n) for t, n in COMPANIES.items() if q in t.lower() or q in n.lower()]
    hits.sort(key=lambda tn: (not tn[0].lower().startswith(q), not tn[1].lower().startswith(q), tn[0]))
    if not hits:
        try:
            hits = [(validate_ticker(query), "ticker as typed")]
        except ValueError:
            hits = []
    return hits[:limit]


def company_name(ticker: str, provider: MarketDataProvider | None = None) -> str:
    if ticker in COMPANIES:
        return COMPANIES[ticker]
    if provider is not None:
        try:
            return provider.profile(ticker).get("name") or ticker
        except Exception:
            pass
    return ticker


def price_window(prices: pd.Series, rng: str) -> pd.Series:
    n = RANGES.get(rng, 21)
    return prices.iloc[-(n + 1):]


def quote(provider: MarketDataProvider, ticker: str, period: str = "1y") -> dict[str, Any]:
    p = provider.prices(ticker, period).dropna()
    if len(p) < 2:
        raise ValueError(f"not enough price history for {ticker}")
    last, prev = float(p.iloc[-1]), float(p.iloc[-2])
    currency = None
    try:
        currency = provider.profile(ticker).get("price_currency")
    except Exception:
        pass
    return {"ticker": ticker, "price": last, "change": last - prev, "change_pct": (last / prev - 1) * 100,
            "as_of": str(p.index[-1].date()), "currency": currency, "source": _source(provider),
            "sample": is_sample(provider), "series": p}


def _source(provider: MarketDataProvider) -> str:
    return "Synthetic sample data (offline)" if is_sample(provider) else "Yahoo Finance via yfinance (third-party)"


def currency_symbol(code: str | None) -> str:
    return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get((code or "").upper(), "")


def ratios(provider: MarketDataProvider, ticker: str) -> dict[str, Any]:
    f = provider.fundamentals(ticker)
    return fundamental_ratios(f) | {"_fundamentals": f}


def news(provider: MarketDataProvider, ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    items, _ = sanitize(provider.news(ticker, limit))  # untrusted text: strip injection attempts
    return items


def watchlist_rows(store: Store, provider: MarketDataProvider) -> list[dict[str, Any]]:
    flags = store.flags(status="open")
    memos = {m["ticker"]: m for m in reversed(store.memos(limit=500))}
    rows = []
    for t in store.watchlist():
        row: dict[str, Any] = {"ticker": t, "name": company_name(t), "error": None}
        try:
            q = quote(provider, t)
            row.update(price=q["price"], change=q["change"], change_pct=q["change_pct"], currency=q["currency"],
                       as_of=q["as_of"])
        except Exception as e:
            row.update(price=None, change=None, change_pct=None, currency=None, error=f"{type(e).__name__}")
        try:
            row["pe"] = ratios(provider, t).get("pe_ratio")
        except Exception:
            row["pe"] = None
        m = memos.get(t)
        row["research"] = "Not run" if m is None else ("Verified" if m["verified"] else "Unverified")
        tf = [f for f in flags if f["ticker"] == t]
        row["risk"] = ("High" if any(f["severity"] == "high" for f in tf) else
                       "Medium" if any(f["severity"] == "medium" for f in tf) else
                       "Low" if tf else ("None" if m else "Unknown"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- agent pipeline
AGENT_INFO = {
    "planner": ("Planner", "Chooses analysts and the memo focus from the question"),
    "researcher": ("Researcher", "Company profile, financial statements and news"),
    "quant": ("Quant", "Returns, risk, ratios and the strategy lab"),
    "peers": ("Peers", "Ranks the company against its sector peers"),
    "sentiment": ("Sentiment", "Scores recent headlines and their trend"),
    "filings": ("Filings", "Searches annual reports and transcripts (RAG)"),
    "risk": ("Risk", "Rule-based flags plus judgement"),
    "writer": ("Writer", "Drafts the cited investment memo"),
    "critic": ("Critic", "Verifies every figure against cited evidence"),
}


def pipeline_from_run(run: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Timeline entries rebuilt from a persisted run's trace: what actually happened, nothing simulated."""
    if not run:
        return []
    trace = run.get("trace") or []
    plan = run.get("plan") or []
    findings = run.get("findings") or []
    usage = (run.get("usage") or {}).get("node_seconds", {})
    items = []
    for node in ["planner", *[a for a in ("researcher", "quant", "peers", "sentiment", "filings") if a in plan],
                 "risk", "writer", "critic"]:
        name, desc = AGENT_INFO[node]
        events = [t for t in trace if t.get("agent") == node]
        secs = usage.get(node)
        status, pill_text = ("completed", "Completed") if (events or secs is not None) else ("skipped", "Not run")
        n_find = sum(1 for f in findings if f["agent"] == node)
        tools = [t for t in events if t.get("event") == "tool"]
        if node in ("researcher", "quant", "peers", "sentiment", "filings"):
            failed = [t for t in tools if not re.fullmatch(r"S\d+", str(t.get("detail", {}).get("result", "")))]
            desc = f"{len(tools)} tool call(s), {n_find} cited finding(s)" + (f", {len(failed)} failed" if failed else "")
            if failed:
                status, pill_text = "flags", "Partial"
        elif node == "planner":
            p = next((t["detail"] for t in events if t.get("event") == "plan"), {})
            desc = "Analysts: " + ", ".join(p.get("analysts", plan)) if p else desc
        elif node == "risk":
            rules = next((t["detail"] for t in events if t.get("event") == "rules"), {})
            n = len(rules.get("flags", []))
            desc = f"{n} rule flag(s) raised"
            if n:
                status, pill_text = "flags", "Flags raised"
        elif node == "writer":
            drafts = sum(1 for t in events if t.get("event") == "draft")
            desc = f"{drafts} draft(s) written"
        elif node == "critic":
            reviews = [t["detail"] for t in events if t.get("event") == "review"]
            rejected = sum(1 for r in reviews if not r.get("approved"))
            if run.get("approved"):
                desc = f"Approved after {rejected} rejection(s)" if rejected else "Approved first draft"
                status, pill_text = ("revised", "Revised") if rejected else ("completed", "Approved")
            elif reviews:
                desc = f"Not approved after {len(reviews)} review(s); unresolved claims remain"
                status, pill_text = "rejected", "Unverified"
        if secs is not None:
            desc += f" · {secs:.1f}s"
        items.append({"key": node, "name": name, "desc": desc, "status": status, "pill": pill_text})
    return items


# --------------------------------------------------------------------------- data-source health
HEALTH_LABELS = {"connected": "ok", "degraded": "warn", "unavailable": "bad", "not configured": "neutral",
                 "not checked": "info", "local": "ok"}


def health_static(settings: Settings, store: Store) -> list[dict[str, Any]]:
    """Cheap status without network calls. Live services stay 'not checked' until a check succeeds."""
    checks = store.get_setting("health_checks", {}) or {}

    def last(name: str, default: tuple[str, str]) -> dict[str, Any]:
        c = checks.get(name)
        if c:
            return {"name": name, "status": c["status"], "detail": c["detail"], "checked_at": c["checked_at"]}
        return {"name": name, "status": default[0], "detail": default[1], "checked_at": None}

    market = last("Market data", ("local", "Synthetic sample provider (offline)") if settings.data == "sample"
                  else ("not checked", "Yahoo Finance via yfinance; run a check to verify"))
    if settings.data == "sample":
        market = {"name": "Market data", "status": "local", "detail": "Synthetic sample provider (offline, labelled)",
                  "checked_at": None}
    llm_cfg = _llm_configured(settings)
    llm = {"name": "LLM provider", "status": "local", "detail": "Offline mock model (no real LLM calls)",
           "checked_at": None} if settings.llm == "mock" else (
        last("LLM provider", ("not checked", f"{settings.llm} / {settings.model or 'default model'}")) if llm_cfg
        else {"name": "LLM provider", "status": "not configured", "detail": f"{settings.llm}: API key missing",
              "checked_at": None})
    sec = last("SEC EDGAR", ("not checked", "US filings; run a check to verify")) if os.getenv("SEC_USER_AGENT") \
        else {"name": "SEC EDGAR", "status": "not configured",
              "detail": "Set SEC_USER_AGENT (name + email), as SEC requires", "checked_at": None}
    return [
        market,
        sec,
        {"name": "NSE data", "status": "not configured",
         "detail": "No official public NSE API is configured; Indian prices come from Yahoo Finance (third-party)",
         "checked_at": None},
        {"name": "Retrieval index", "status": "local",
         "detail": "BM25 over uploaded filings (pure Python); optional embeddings via FINSIGHT_EMBED_MODEL",
         "checked_at": None},
        llm,
        {"name": "Backtest engine", "status": "local",
         "detail": "Predefined strategies only; no user code is executed", "checked_at": None},
    ]


def _llm_configured(s: Settings) -> bool:
    if s.llm == "anthropic":
        return bool(s.anthropic_api_key)
    if s.llm == "openai":
        return bool(s.openai_api_key) or "localhost" in s.openai_base_url or "127.0.0.1" in s.openai_base_url
    return True


def run_health_checks(settings: Settings, store: Store) -> dict[str, dict[str, Any]]:
    """Live checks, run only on explicit user action. Results are persisted with timestamps."""
    results: dict[str, dict[str, Any]] = {}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record(name: str, ok: bool, detail: str, degraded: bool = False) -> None:
        results[name] = {"status": "connected" if ok and not degraded else "degraded" if ok else "unavailable",
                         "detail": detail, "checked_at": ts}

    if settings.data != "sample":
        t0 = time.time()
        try:
            q = quote(provider_for(settings), "^NSEI", "6mo")
            age = (datetime.now().date() - datetime.fromisoformat(q["as_of"]).date()).days
            record("Market data", True, f"NIFTY 50 last close {q['as_of']} ({time.time() - t0:.1f}s)", degraded=age > 5)
        except Exception as e:
            record("Market data", False, f"{type(e).__name__}: {str(e)[:120]}")
    if settings.llm != "mock" and _llm_configured(settings):
        from finsight.llm.providers import get_llm

        t0 = time.time()
        try:
            reply = get_llm(settings).complete("Reply with the single word: ok", [{"role": "user", "content": "ping"}],
                                               max_tokens=20)
            record("LLM provider", bool(reply.strip()), f"{settings.llm} / {settings.model or 'default'} replied "
                                                        f"({time.time() - t0:.1f}s)")
        except Exception as e:
            record("LLM provider", False, f"{type(e).__name__}: {str(e)[:120]}")
    if os.getenv("SEC_USER_AGENT"):
        from finsight.data import sec

        try:
            n = len(sec.ticker_map())
            record("SEC EDGAR", n > 0, f"company ticker index reachable ({n:,} companies)")
        except Exception as e:
            record("SEC EDGAR", False, f"{type(e).__name__}: {str(e)[:120]}")
    saved = store.get_setting("health_checks", {}) or {}
    saved.update(results)
    store.set_setting("health_checks", saved)
    return results


def finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
