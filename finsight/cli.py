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

ICONS = {"researcher": "🔎", "quant": "📊", "risk": "⚠️ ", "writer": "✍️ ", "critic": "🧐"}


def _on_step(name: str, state) -> None:
    detail = ""
    if name in ("researcher", "quant"):
        detail = f"{len(state.findings_for(name))} findings, {len(state.evidence.items)} evidence items"
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
        state = run_analysis(args.ticker, args.question, settings=s, engine_name=args.engine, on_step=_on_step)
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
        "critique": state.critique, "errors": state.errors, "trace": state.trace,
        "evidence": {k: {"source": v.source, "title": v.title, "data": v.data} for k, v in state.evidence.items.items()},
    }, indent=2, default=str), encoding="utf-8")

    failed = [t["detail"] for t in state.trace
              if t["event"] == "tool" and not re.fullmatch(r"S\d+", str(t["detail"].get("result", "")))]
    if failed:
        print("\nTool calls that failed:")
        for d in failed:
            print(f"  - {d.get('tool')}: {d.get('result')}")
    status = "APPROVED" if state.approved else "NOT APPROVED (best draft saved)"
    print(f"\n{status} after {state.revisions} revision(s) in {time.time() - t0:.1f}s")
    print(f"Memo:  {memo_path}\nTrace: {trace_path}")
    if not state.approved:
        for issue in state.critique.get("issues", [])[:5]:
            print(f"  - {issue}")
    if args.print:
        print("\n" + state.memo)
    return 0 if state.approved else 2


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
    a.add_argument("--out", default="reports")
    a.add_argument("--print", action="store_true", help="also print the memo")
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
