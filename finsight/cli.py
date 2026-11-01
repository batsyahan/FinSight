"""Command-line interface.

    finsight analyze INFY.NS --llm anthropic --data yfinance
    finsight analyze DEMO                      # fully offline demo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.llm.base import LLMError
from finsight.obs import usage_summary

ICONS = {"planner": "🗺️ ", "researcher": "🔎", "quant": "📊", "peers": "👥", "risk": "⚠️ ", "writer": "✍️ ",
         "critic": "🧐"}


def _on_step(name: str, state) -> None:
    detail = ""
    if name == "planner":
        detail = "analysts: " + ", ".join(state.plan)
    elif name in ("researcher", "quant", "peers"):
        detail = f"{len(state.findings_for(name))} findings"
    elif name == "risk":
        detail = f"{len(state.risk_flags)} rule flags"
    elif name == "writer":
        detail = f"draft {state.revisions + 1} ({len(state.memo)} chars)"
    elif name == "critic":
        n = len(state.critique.get("issues", []))
        detail = "approved ✅" if state.approved else f"rejected: {n} issue(s)"
    print(f"  {ICONS.get(name, '•')} {name:<10} {detail}", flush=True)


def cmd_analyze(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    overrides = {k: v for k, v in {"llm": args.llm, "model": args.model, "data": args.data,
                                   "max_revisions": args.max_revisions}.items() if v is not None}
    s = replace(s, **overrides)
    print(f"FinSight → {args.ticker.upper()}  (llm={s.llm}, data={s.data})")
    t0 = time.time()
    try:
        state = run_analysis(args.ticker, args.question, settings=s, engine_name=args.engine, on_step=_on_step,
                             analysts=_csv(args.analysts), peers=_csv(args.peers))
    except LLMError as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        msg = str(e)
        if "429" in msg:
            print("Hint: the provider's rate limit was hit even after waiting. Free tiers allow few tokens per\n"
                  "minute; wait a minute and retry, use a smaller model, or set FINSIGHT_MAX_REVISIONS=1.",
                  file=sys.stderr)
        elif "401" in msg or "403" in msg:
            print("Hint: the API key was rejected. Check the key in .env (no quotes or spaces), that it\n"
                  "belongs to the provider in OPENAI_BASE_URL, and that it has not been revoked.", file=sys.stderr)
        elif "404" in msg or "model_not_found" in msg:
            print("Hint: check FINSIGHT_MODEL is a model your provider currently serves.", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = state.ticker.replace(".", "_")
    memo_path = out / f"{stem}_memo.md"
    memo_path.write_text(state.memo + _appendix(state), encoding="utf-8")
    trace_path = out / f"{stem}_trace.json"
    trace_path.write_text(json.dumps({
        "ticker": state.ticker, "approved": state.approved, "revisions": state.revisions,
        "critique": state.critique, "errors": state.errors, "usage": usage_summary(state), "trace": state.trace,
        "llm_calls": state.llm_calls,
        "evidence": {k: {"source": v.source, "title": v.title, "data": v.data} for k, v in state.evidence.items.items()},
    }, indent=2, default=str), encoding="utf-8")

    failed = [t["detail"] for t in state.trace
              if t["event"] == "tool" and not re.fullmatch(r"S\d+", str(t["detail"].get("result", "")))]
    if failed:
        print("\nTool calls that failed:")
        for d in failed:
            print(f"  - {d.get('tool')}: {d.get('result')}")
    tot = usage_summary(state)["totals"]
    approx = "~" if tot["tokens_estimated"] else ""
    cost = f", est. cost ${tot['cost_usd']}" if tot["cost_usd"] else ""
    print(f"\nLLM usage: {tot['llm_calls']} calls, {approx}{tot['input_tokens']:,} in / "
          f"{approx}{tot['output_tokens']:,} out tokens{cost}")
    status = "APPROVED" if state.approved else "NOT APPROVED (best draft saved)"
    print(f"\n{status} after {state.revisions} revision(s) in {time.time() - t0:.1f}s")
    print(f"Memo:  {memo_path}\nTrace: {trace_path}")
    if not state.approved:
        for issue in state.critique.get("issues", [])[:5]:
            print(f"  - {issue}")
    if args.print:
        print("\n" + state.memo)
    return 0 if state.approved else 2


def cmd_eval(args: argparse.Namespace) -> int:
    from finsight import eval as ev

    s = ev.with_overrides(Settings.from_env(), llm=args.llm, model=args.model, data=args.data,
                          max_revisions=args.max_revisions)
    if args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    else:
        tickers = ev.UNIVERSES[args.universe]
    if s.data == "sample" and args.universe != "demo" and not args.tickers:
        print("Note: FINSIGHT_DATA=sample, so these tickers get synthetic data. Use --data yfinance for real data.")
    print(f"Evaluating {len(tickers)} tickers with llm={s.llm} model={s.model or '(default)'} data={s.data}")

    def progress(ticker, rec):
        if rec.get("ok"):
            print(f"  {'✅' if rec['approved'] else '❌'} {ticker:<14} rev={rec['revisions']} "
                  f"figures={rec['numbers_checked']} unsupported={rec['unsupported_numbers']} {rec['seconds']}s")
        else:
            print(f"  ⚠️  {ticker:<14} {rec['error'][:100]}")

    result = ev.run_eval(tickers, settings=s, pause=args.pause, engine_name=args.engine, on_ticker=progress)
    j, md = ev.save(result, args.out)
    sm = result["summary"]
    print(f"\nApproval {sm['approval_rate_pct']}% · grounded figures {sm['grounded_figures_pct']}% of "
          f"{sm['figures_checked']} · avg {sm['avg_seconds']}s/run")
    print(f"Report: {md}\nData:   {j}")
    return 0


def cmd_eval_compare(args: argparse.Namespace) -> int:
    from finsight import eval as ev

    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files]
    table = ev.compare_markdown(results)
    print(table)
    if args.out:
        Path(args.out).write_text("# FinSight model comparison\n\n" + table, encoding="utf-8")
        print(f"Saved {args.out}")
    return 0


def _csv(value: str | None) -> list[str] | None:
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


def _appendix(state) -> str:
    lines = ["", "", "---", "", "### Evidence", ""]
    for ev in state.evidence.items.values():
        lines.append(f"- **[{ev.id}]** {ev.title} (`{ev.source}`)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", message=r".*inplace keyword.*")  # pandas deprecations raised inside yfinance
    p = argparse.ArgumentParser(prog="finsight", description="Multi-agent AI equity research analyst")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="research a ticker and write a cited memo")
    a.add_argument("ticker", help="e.g. INFY.NS, RELIANCE.NS, AAPL, or any word for sample data")
    a.add_argument("-q", "--question", default=None, help="investor question to focus the memo")
    a.add_argument("--llm", choices=["mock", "anthropic", "openai"], default=None)
    a.add_argument("--model", default=None)
    a.add_argument("--data", choices=["sample", "yfinance"], default=None)
    a.add_argument("--engine", choices=["auto", "langgraph", "builtin"], default=None)
    a.add_argument("--max-revisions", type=int, default=None)
    a.add_argument("--analysts", default=None,
                   help="comma-separated analysts to run, e.g. quant,peers (default: the planner decides)")
    a.add_argument("--peers", default=None, help="comma-separated peer tickers, e.g. TCS.NS,WIPRO.NS")
    a.add_argument("--out", default="reports")
    a.add_argument("--print", action="store_true", help="also print the memo")
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("eval", help="benchmark FinSight over many tickers")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--universe", default="demo", help="demo, nifty-it, nifty-top10 or banks (default: demo)")
    g.add_argument("--tickers", help="comma-separated tickers, e.g. INFY.NS,TCS.NS")
    e.add_argument("--llm", choices=["mock", "anthropic", "openai"], default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--data", choices=["sample", "yfinance"], default=None)
    e.add_argument("--engine", choices=["auto", "langgraph", "builtin"], default=None)
    e.add_argument("--max-revisions", type=int, default=None)
    e.add_argument("--pause", type=float, default=0.0, help="seconds between tickers (helps free tiers)")
    e.add_argument("--out", default="reports")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("eval-compare", help="compare saved eval results across models")
    c.add_argument("files", nargs="+", help="eval_*.json files")
    c.add_argument("--out", default=None, help="write the comparison table to this .md file")
    c.set_defaults(func=cmd_eval_compare)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
