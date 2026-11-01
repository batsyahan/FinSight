"""The five FinSight agents. Each is a function ``(state) -> state`` so any graph engine can run it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from finsight.agents import prompts
from finsight.agents.react import _parse_findings, run_react_agent
from finsight.llm.base import extract_json
from finsight.state import AnalysisState
from finsight.tools.registry import ToolRegistry
from finsight.tools.risk import evaluate_risks, risk_score
from finsight.verify import verify_memo


@dataclass
class Team:
    llm: Any
    tools: ToolRegistry
    max_revisions: int = 2

    # ------------------------------------------------------------------ researcher
    def researcher(self, state: AnalysisState) -> AnalysisState:
        findings = run_react_agent(
            agent="researcher",
            role_prompt=prompts.RESEARCHER,
            task=f"Research {state.ticker}. Investor question: {state.question}",
            llm=self.llm,
            tools=self.tools.subset(["get_company_profile", "get_fundamentals", "get_news"]),
            state=state,
            max_steps=5,
        )
        state.findings.extend(findings)
        return state

    # ------------------------------------------------------------------ quant
    def quant(self, state: AnalysisState) -> AnalysisState:
        findings = run_react_agent(
            agent="quant",
            role_prompt=prompts.QUANT,
            task=f"Analyse {state.ticker} quantitatively.",
            llm=self.llm,
            tools=self.tools.subset(["get_price_stats", "run_backtest", "compute_ratios"]),
            state=state,
            max_steps=5,
        )
        state.findings.extend(findings)
        return state

    # ------------------------------------------------------------------ risk
    def risk(self, state: AnalysisState) -> AnalysisState:
        metrics = self._latest("get_price_stats", state) | self._latest("compute_ratios", state)
        if not metrics:  # quant failed: fetch what the rules need directly
            for name in ("get_price_stats", "compute_ratios"):
                res = self.tools.call(name)
                metrics |= res.get("data", {})
        flags = evaluate_risks(metrics)
        ev = state.evidence.add(
            "risk_rules", f"{state.ticker} rule-based risk flags",
            {"risk_score": risk_score(flags), "risk_score_max": 100, "flags": [
                {"id": f["id"], "metric": f["metric"], "value": f["value"], "threshold": f["threshold"], "limit": f["limit"],
                 "severity": f["severity"]} for f in flags]},
        )
        state.risk_flags = flags
        state.log("risk", "rules", {"evidence": ev.id, "flags": [f["id"] for f in flags]})

        user = (
            f"Ticker: {state.ticker}\nRisk flags (evidence {ev.id}):\n{json.dumps(flags, default=str)}\n\n"
            f"Team findings:\n{_findings_text(state)}\n\nEvidence:\n{state.evidence.to_prompt()}"
        )
        reply = self.llm.complete(prompts.RISK, [{"role": "user", "content": user}], role="risk", context=state)
        try:
            state.findings.extend(_parse_findings("risk", extract_json(reply), state))
        except ValueError:
            state.errors.append("risk: could not parse LLM reply")
        return state

    # ------------------------------------------------------------------ writer
    def writer(self, state: AnalysisState) -> AnalysisState:
        feedback = ""
        if state.critique.get("issues"):
            feedback = (
                "\n\nYour previous draft was rejected. Fix every issue below and return the full corrected memo:\n- "
                + "\n- ".join(state.critique["issues"][:12])
                + f"\n\nPrevious draft:\n{state.memo}"
            )
            state.revisions += 1
        user = (
            f"Ticker: {state.ticker}\nInvestor question: {state.question}\n\n"
            f"Team findings:\n{_findings_text(state)}\n\nEvidence:\n{state.evidence.to_prompt()}{feedback}"
        )
        memo = self.llm.complete(prompts.WRITER, [{"role": "user", "content": user}], role="writer",
                                 context=state, max_tokens=2500)
        state.memo = _strip_fences(memo)
        state.log("writer", "draft", {"revision": state.revisions, "chars": len(state.memo)})
        return state

    # ------------------------------------------------------------------ critic
    def critic(self, state: AnalysisState) -> AnalysisState:
        check = verify_memo(state.memo, state.evidence)
        user = (
            f"Automated check: {json.dumps(check)}\n\nEvidence:\n{state.evidence.to_prompt()}\n\n"
            f"Draft memo:\n{state.memo}"
        )
        reply = self.llm.complete(prompts.CRITIC, [{"role": "user", "content": user}], role="critic", context=state)
        try:
            review = extract_json(reply)
        except ValueError:
            review = {"approved": False, "issues": ["Critic reply could not be parsed."]}
        issues = list(check["issues"]) + [str(i) for i in review.get("issues", []) if i]
        approved = bool(review.get("approved")) and check["passed"]
        state.critique = {"approved": approved, "issues": [] if approved else issues, "check": check["stats"]}
        state.approved = approved
        state.log("critic", "review", {"approved": approved, "issues": len(issues)})
        return state

    def route_after_critic(self, state: AnalysisState) -> str:
        if state.approved or state.revisions >= self.max_revisions:
            return "done"
        return "revise"

    @staticmethod
    def _latest(source: str, state: AnalysisState) -> dict[str, Any]:
        for ev in reversed(list(state.evidence.items.values())):
            if ev.source == source:
                return dict(ev.data)
        return {}


def _findings_text(state: AnalysisState) -> str:
    return "\n".join(
        f"- ({f.agent}) {f.text} [{', '.join(f.citations)}]" if f.citations else f"- ({f.agent}) {f.text}"
        for f in state.findings
    ) or "(none)"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
