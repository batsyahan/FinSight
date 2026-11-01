"""Watchlist alerts: cheap daily snapshots (no LLM calls) compared with the previous run.

    finsight watch INFY.NS,TCS.NS,HDFCBANK.NS
    finsight watch --file watchlist.txt --webhook $SLACK_WEBHOOK_URL

Alerts fire when a risk flag appears or clears, the price moves more than a threshold since
the last snapshot, or news sentiment swings. State is kept in a JSON file so a scheduled job
(see .github/workflows/watchlist.yml) can compare against yesterday.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from finsight.data.providers import MarketDataProvider
from finsight.guardrails import validate_ticker
from finsight.tools.quant import fundamental_ratios, price_stats
from finsight.tools.risk import evaluate_risks, risk_score
from finsight.tools.sentiment import news_sentiment


def snapshot(ticker: str, provider: MarketDataProvider) -> dict[str, Any]:
    stats = price_stats(provider.prices(ticker, "2y"))
    ratios = fundamental_ratios(provider.fundamentals(ticker))
    sent = news_sentiment(provider.news(ticker, 10))
    metrics = stats | ratios | ({"net_sentiment": sent["net_sentiment"]} if sent.get("headlines") else {})
    flags = evaluate_risks(metrics)
    return {
        "ticker": ticker,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price": stats["last_price"],
        "return_1y_pct": stats["return_1y_pct"],
        "net_sentiment": sent.get("net_sentiment"),
        "risk_score": risk_score(flags),
        "flags": sorted(f["id"] for f in flags),
    }


def diff(prev: dict[str, Any] | None, cur: dict[str, Any], move_pct: float = 5.0,
         sentiment_swing: float = 0.4) -> list[dict[str, Any]]:
    t = cur["ticker"]
    if prev is None:
        return [{"ticker": t, "kind": "baseline", "message": f"{t}: first snapshot (risk score {cur['risk_score']})."}]
    alerts = []
    for f in sorted(set(cur["flags"]) - set(prev["flags"])):
        alerts.append({"ticker": t, "kind": "flag_raised", "message": f"{t}: new risk flag '{f}'."})
    for f in sorted(set(prev["flags"]) - set(cur["flags"])):
        alerts.append({"ticker": t, "kind": "flag_cleared", "message": f"{t}: risk flag '{f}' cleared."})
    if prev.get("price"):
        change = (cur["price"] / prev["price"] - 1) * 100
        if abs(change) >= move_pct:
            alerts.append({"ticker": t, "kind": "price_move",
                           "message": f"{t}: price {'up' if change > 0 else 'down'} {abs(change):.1f}% since last check "
                                      f"({prev['price']} → {cur['price']})."})
    if prev.get("net_sentiment") is not None and cur.get("net_sentiment") is not None:
        swing = cur["net_sentiment"] - prev["net_sentiment"]
        if abs(swing) >= sentiment_swing:
            alerts.append({"ticker": t, "kind": "sentiment_swing",
                           "message": f"{t}: news sentiment {'improved' if swing > 0 else 'worsened'} "
                                      f"({prev['net_sentiment']} → {cur['net_sentiment']})."})
    return alerts


def run_watch(tickers: list[str], provider: MarketDataProvider, state_path: str | Path,
              move_pct: float = 5.0) -> dict[str, Any]:
    path = Path(state_path)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current, alerts, errors = {}, [], {}
    for raw in tickers:
        t = validate_ticker(raw)
        try:
            snap = snapshot(t, provider)
        except Exception as e:  # keep watching the others
            errors[t] = f"{type(e).__name__}: {e}"[:160]
            if t in previous:
                current[t] = previous[t]
            continue
        current[t] = snap
        alerts += diff(previous.get(t), snap, move_pct=move_pct)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return {"alerts": alerts, "snapshots": current, "errors": errors}


def alerts_markdown(result: dict[str, Any]) -> str:
    lines = [f"# FinSight watchlist · {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    real = [a for a in result["alerts"] if a["kind"] != "baseline"]
    lines += [f"- {a['message']}" for a in real] or ["No changes worth alerting on."]
    lines += ["", "| Ticker | Price | 1y return % | Sentiment | Risk score | Flags |", "|---|---|---|---|---|---|"]
    for t, s in result["snapshots"].items():
        lines.append(f"| {t} | {s['price']} | {s['return_1y_pct']} | {s['net_sentiment']} | {s['risk_score']} | "
                     f"{', '.join(s['flags']) or '-'} |")
    for t, e in result["errors"].items():
        lines.append(f"\n⚠️ {t}: {e}")
    return "\n".join(lines) + "\n"


def post_webhook(url: str, result: dict[str, Any]) -> bool:
    """Slack- and Discord-compatible: both accept a JSON body with a text/content field."""
    real = [a["message"] for a in result["alerts"] if a["kind"] != "baseline"]
    if not real:
        return False
    text = "FinSight watchlist alerts:\n" + "\n".join(f"• {m}" for m in real)
    r = requests.post(url, json={"text": text, "content": text[:1900]}, timeout=15)
    return r.status_code < 300
