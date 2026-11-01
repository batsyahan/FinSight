"""Evaluation harness: run FinSight over a universe of stocks and measure it.

    finsight eval --universe nifty-it
    finsight eval-compare reports/eval_*.json

Metrics per run: approved, revisions, grounding of the final memo (figures checked vs.
unsupported, uncited sentences), evidence gathered, failed tools, findings dropped for
missing citations, tokens, cost and latency. The summary turns these into rates you can
quote ("97% of figures grounded across 10 NSE stocks") and compare across models.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from finsight import __version__
from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.obs import usage_summary
from finsight.state import AnalysisState
from finsight.verify import verify_memo

UNIVERSES: dict[str, list[str]] = {
    "nifty-it": ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "nifty-top10": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
                    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS"],
    "banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "demo": ["DEMO", "ACME", "GLOBEX", "INITECH", "UMBRELLA"],  # synthetic, works offline
}


def run_metrics(state: AnalysisState) -> dict[str, Any]:
    events = [t["event"] for t in state.trace]
    tool_events = [t["detail"] for t in state.trace if t["event"] == "tool"]
    check = verify_memo(state.memo, state.evidence) if state.memo else {"passed": False, "stats": {}}
    st = check.get("stats", {})
    usage = usage_summary(state)["totals"]
    return {
        "ticker": state.ticker,
        "ok": True,
        "approved": state.approved,
        "revisions": state.revisions,
        "first_draft_approved": state.approved and state.revisions == 0,
        "grounding_passed": check["passed"],
        "numbers_checked": st.get("numbers_checked", 0),
        "unsupported_numbers": st.get("unsupported_numbers", 0),
        "uncited_sentences": st.get("uncited_sentences", 0),
        "citations_used": len(st.get("citations_used", [])),
        "evidence_items": len(state.evidence.items),
        "findings": len(state.findings),
        "dropped_uncited_findings": events.count("dropped_uncited_finding"),
        "failed_tools": sum(1 for d in tool_events if not re.fullmatch(r"S\d+", str(d.get("result", "")))),
        "provider_rejections": events.count("provider_rejected_call"),
        "risk_flags": len(state.risk_flags),
        "llm_calls": usage["llm_calls"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": usage["cost_usd"],
        "seconds": usage["wall_seconds"],
        "errors": list(state.errors),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in runs if r.get("ok")]
    checked = sum(r["numbers_checked"] for r in done)
    unsupported = sum(r["unsupported_numbers"] for r in done)

    def rate(key: str) -> float:
        return round(100 * sum(1 for r in done if r[key]) / len(done), 1) if done else 0.0

    return {
        "runs": len(runs),
        "completed": len(done),
        "completion_rate_pct": round(100 * len(done) / len(runs), 1) if runs else 0.0,
        "approval_rate_pct": rate("approved"),
        "first_draft_approval_pct": rate("first_draft_approved"),
        "grounding_pass_rate_pct": rate("grounding_passed"),
        "grounded_figures_pct": round(100 * (1 - unsupported / checked), 1) if checked else 0.0,
        "figures_checked": checked,
        "avg_revisions": round(mean(r["revisions"] for r in done), 2) if done else 0.0,
        "avg_evidence_items": round(mean(r["evidence_items"] for r in done), 1) if done else 0.0,
        "dropped_uncited_findings": sum(r["dropped_uncited_findings"] for r in done),
        "failed_tools": sum(r["failed_tools"] for r in done),
        "avg_seconds": round(mean(r["seconds"] for r in done), 1) if done else 0.0,
        "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in done),
        "avg_tokens_per_run": round(mean(r["input_tokens"] + r["output_tokens"] for r in done)) if done else 0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in done), 4),
    }


def run_eval(
    tickers: list[str],
    *,
    settings: Settings | None = None,
    llm_factory: Callable[[], Any] | None = None,
    pause: float = 0.0,
    engine_name: str | None = None,
    on_ticker: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    s = settings or Settings.from_env()
    runs = []
    for i, ticker in enumerate(tickers):
        if i and pause:
            time.sleep(pause)  # be gentle with free-tier rate limits
        t0 = time.perf_counter()
        try:
            state = run_analysis(ticker, settings=s, llm=llm_factory() if llm_factory else None,
                                 engine_name=engine_name)
            rec = run_metrics(state)
        except Exception as e:  # one bad ticker must not stop the benchmark
            rec = {"ticker": ticker.upper(), "ok": False, "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(limit=3), "seconds": round(time.perf_counter() - t0, 1)}
        runs.append(rec)
        if on_ticker:
            on_ticker(ticker, rec)
    return {
        "meta": {"finsight_version": __version__, "llm": s.llm, "model": s.model or "(default)", "data": s.data,
                 "max_revisions": s.max_revisions, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "tickers": tickers},
        "runs": runs,
        "summary": summarize(runs),
    }


def to_markdown(result: dict[str, Any]) -> str:
    m, s = result["meta"], result["summary"]
    lines = [
        f"# FinSight evaluation: {m['llm']} / {m['model']}",
        "",
        f"{m['date']} · data: {m['data']} · {s['runs']} tickers · max revisions {m['max_revisions']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Completed runs | {s['completed']}/{s['runs']} ({s['completion_rate_pct']}%) |",
        f"| Critic approval rate | {s['approval_rate_pct']}% |",
        f"| Approved on first draft | {s['first_draft_approval_pct']}% |",
        f"| Figures grounded in evidence | {s['grounded_figures_pct']}% of {s['figures_checked']} |",
        f"| Memos passing all grounding checks | {s['grounding_pass_rate_pct']}% |",
        f"| Avg revisions | {s['avg_revisions']} |",
        f"| Avg evidence items | {s['avg_evidence_items']} |",
        f"| Uncited findings dropped | {s['dropped_uncited_findings']} |",
        f"| Failed tool calls | {s['failed_tools']} |",
        f"| Avg time per run | {s['avg_seconds']}s |",
        f"| Avg tokens per run | {s['avg_tokens_per_run']:,} |",
        f"| Total est. cost | ${s['total_cost_usd']} |",
        "",
        "## Per ticker",
        "",
        "| Ticker | Approved | Revisions | Figures checked | Unsupported | Evidence | Seconds | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["runs"]:
        if not r.get("ok"):
            lines.append(f"| {r['ticker']} | error: {r['error'][:60]} | | | | | {r['seconds']} | |")
            continue
        lines.append(f"| {r['ticker']} | {'✅' if r['approved'] else '❌'} | {r['revisions']} | {r['numbers_checked']} | "
                     f"{r['unsupported_numbers']} | {r['evidence_items']} | {r['seconds']} | "
                     f"{r['input_tokens'] + r['output_tokens']:,} |")
    return "\n".join(lines) + "\n"


def compare_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| LLM / model | Data | Runs | Approved | 1st draft | Grounded figures | Avg revisions | Avg s | Avg tokens | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, s = r["meta"], r["summary"]
        lines.append(f"| {m['llm']} / {m['model']} | {m['data']} | {s['runs']} | {s['approval_rate_pct']}% | "
                     f"{s['first_draft_approval_pct']}% | {s['grounded_figures_pct']}% | {s['avg_revisions']} | "
                     f"{s['avg_seconds']} | {s['avg_tokens_per_run']:,} | ${s['total_cost_usd']} |")
    return "\n".join(lines) + "\n"


def save(result: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = result["meta"]
    slug = f"{m['llm']}_{m['model']}".replace("/", "-").replace("(", "").replace(")", "").replace(" ", "")
    stamp = m["date"][:19].replace(":", "").replace("-", "")
    j = out / f"eval_{slug}_{stamp}.json"
    md = out / f"eval_{slug}_{stamp}.md"
    j.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md.write_text(to_markdown(result), encoding="utf-8")
    return j, md


def with_overrides(s: Settings, **kw: Any) -> Settings:
    return replace(s, **{k: v for k, v in kw.items() if v is not None})
