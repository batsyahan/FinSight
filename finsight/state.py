"""Shared state passed between agents, plus the evidence store that makes citations checkable.

Every tool result an agent relies on is registered as an ``Evidence`` item with an id
like ``S1``. The Writer must cite those ids, and the Critic verifies that every cited id
exists and every number in the memo can be traced back to some piece of evidence.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    id: str
    source: str  # tool or data source name, e.g. "price_stats"
    title: str  # short human description
    data: dict[str, Any]  # flat-ish key -> value payload

    def numbers(self) -> list[float]:
        """All numeric values in this evidence (recursively), for grounding checks."""
        out: list[float] = []

        def walk(v: Any) -> None:
            if isinstance(v, bool):
                return
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)

        walk(self.data)
        return out


@dataclass
class EvidenceStore:
    items: dict[str, Evidence] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, source: str, title: str, data: dict[str, Any]) -> Evidence:
        with self._lock:  # agents may run in parallel branches
            ev = Evidence(id=f"S{len(self.items) + 1}", source=source, title=title, data=data)
            self.items[ev.id] = ev
        return ev

    def get(self, ev_id: str) -> Evidence | None:
        return self.items.get(ev_id)

    def all_numbers(self) -> list[float]:
        return [n for ev in self.items.values() for n in ev.numbers()]

    def to_prompt(self, max_chars: int = 6000) -> str:
        """Compact text form of the evidence for LLM prompts."""
        lines = []
        for ev in self.items.values():
            lines.append(f"[{ev.id}] {ev.title} ({ev.source}): {_compact(ev.data)}")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "\n...(truncated)"


def _compact(data: dict[str, Any]) -> str:
    parts = []
    for k, v in data.items():
        if isinstance(v, float):
            v = round(v, 4)
        if isinstance(v, (list, dict)) and len(str(v)) > 300:
            v = str(v)[:300] + "..."
        parts.append(f"{k}={v}")
    return "; ".join(parts)


@dataclass
class Finding:
    agent: str
    text: str
    citations: list[str] = field(default_factory=list)


@dataclass
class AnalysisState:
    ticker: str
    question: str = "Should a long-term investor consider this stock?"
    plan: list[str] = field(default_factory=list)  # analyst agents to run (set by the planner)
    peers: list[str] = field(default_factory=list)  # optional user-chosen peer tickers
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    findings: list[Finding] = field(default_factory=list)
    risk_flags: list[dict[str, Any]] = field(default_factory=list)
    memo: str = ""
    critique: dict[str, Any] = field(default_factory=dict)
    revisions: int = 0
    approved: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)  # step-by-step agent log
    errors: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)  # telemetry: one record per LLM call
    node_seconds: dict[str, float] = field(default_factory=dict)  # wall time per graph node
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def log(self, agent: str, event: str, detail: Any = None) -> None:
        with self._lock:
            self.trace.append({"agent": agent, "event": event, "detail": detail})

    def add_findings(self, findings: list[Finding]) -> None:
        with self._lock:
            self.findings.extend(findings)

    def findings_for(self, agent: str) -> list[Finding]:
        return [f for f in self.findings if f.agent == agent]
