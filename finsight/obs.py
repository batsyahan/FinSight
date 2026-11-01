"""Observability: per-agent token usage, latency and estimated cost.

``instrument(llm, state)`` wraps any LLM so every call is recorded in ``state.llm_calls``.
Prices are per million tokens and come from FINSIGHT_PRICE_IN / FINSIGHT_PRICE_OUT
(defaults 0, e.g. for free tiers), so cost estimates work for any provider.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from finsight.state import AnalysisState


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb when the provider reports nothing


def _prices() -> tuple[float, float]:
    try:
        return float(os.getenv("FINSIGHT_PRICE_IN", "0") or 0), float(os.getenv("FINSIGHT_PRICE_OUT", "0") or 0)
    except ValueError:
        return 0.0, 0.0


class _Instrumented:
    def __init__(self, inner: Any, state: AnalysisState) -> None:
        self.inner, self.state = inner, state
        self.name = getattr(inner, "name", "llm")

    def _record(self, role: str, kind: str, started: float, prompt_text: str, reply_text: str, ok: bool) -> None:
        usage = inner_usage(self.inner)
        inp = usage["input_tokens"] if usage else _estimate_tokens(prompt_text)
        out = usage["output_tokens"] if usage else _estimate_tokens(reply_text)
        p_in, p_out = _prices()
        rec = {
            "agent": role or "?", "kind": kind, "ok": ok,
            "seconds": round(time.perf_counter() - started, 3),
            "input_tokens": inp, "output_tokens": out, "estimated": usage is None,
            "cost_usd": round(inp / 1e6 * p_in + out / 1e6 * p_out, 6),
        }
        with self.state._lock:
            self.state.llm_calls.append(rec)

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
        t0 = time.perf_counter()
        prompt = system + "".join(str(m.get("content", "")) for m in messages)
        try:
            out = self.inner.complete(system, messages, role=role, context=context, max_tokens=max_tokens)
        except Exception:
            self._record(role, "complete", t0, prompt, "", ok=False)
            raise
        self._record(role, "complete", t0, prompt, out, ok=True)
        return out


class _InstrumentedTools(_Instrumented):
    def chat_tools(self, system, messages, tools, *, tool_choice="auto", role="", context=None, max_tokens=1500):
        t0 = time.perf_counter()
        prompt = system + "".join(str(m.get("content", "")) for m in messages) + str(tools)
        try:
            out = self.inner.chat_tools(system, messages, tools, tool_choice=tool_choice, role=role,
                                        context=context, max_tokens=max_tokens)
        except Exception:
            self._record(role, "chat_tools", t0, prompt, "", ok=False)
            raise
        self._record(role, "chat_tools", t0, prompt, str(out.get("content", "")) + str(out.get("tool_calls", "")), ok=True)
        return out


def inner_usage(llm: Any) -> dict[str, int] | None:
    fn = getattr(llm, "last_usage", None)
    return fn() if callable(fn) else None


def instrument(llm: Any, state: AnalysisState) -> Any:
    if isinstance(llm, _Instrumented):
        llm = llm.inner
    cls = _InstrumentedTools if hasattr(llm, "chat_tools") else _Instrumented
    return cls(llm, state)


def usage_summary(state: AnalysisState) -> dict[str, Any]:
    per_agent: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                                                  "llm_seconds": 0.0, "cost_usd": 0.0})
    for c in state.llm_calls:
        a = per_agent[c["agent"]]
        a["calls"] += 1
        a["input_tokens"] += c["input_tokens"]
        a["output_tokens"] += c["output_tokens"]
        a["llm_seconds"] = round(a["llm_seconds"] + c["seconds"], 3)
        a["cost_usd"] = round(a["cost_usd"] + c["cost_usd"], 6)
    totals = {
        "llm_calls": len(state.llm_calls),
        "input_tokens": sum(c["input_tokens"] for c in state.llm_calls),
        "output_tokens": sum(c["output_tokens"] for c in state.llm_calls),
        "cost_usd": round(sum(c["cost_usd"] for c in state.llm_calls), 6),
        "tokens_estimated": any(c["estimated"] for c in state.llm_calls),
        "wall_seconds": round(sum(state.node_seconds.values()), 3),
    }
    return {"totals": totals, "per_agent": dict(per_agent), "node_seconds": dict(state.node_seconds)}
