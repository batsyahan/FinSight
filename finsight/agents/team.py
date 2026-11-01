"""The five FinSight agents. Each is a function ``(state) -> state`` so any graph engine can run it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from finsight.agents import prompts
from finsight.agents.react import _parse_findings, run_react_agent
from finsight.llm.base import extract_json
from finsight.state import AnalysisState
from finsight.tools.registry import ToolRegistry
from finsight.tools.risk import evaluate_risks, risk_score
from finsight.verify import verify_memo


ANALYST_INFO = {
    "researcher": "company profile, financial statements and recent news (what the business is and does)",
    "quant": "price returns, volatility, drawdown, Sharpe, strategy backtest and fundamental ratios",
    "peers": "side-by-side comparison with sector peers (relative growth, margins, valuation, returns)",
    "sentiment": "tone of recent news headlines and whether it is improving or worsening",
}
ANALYSTS = list(ANALYST_INFO)  # agents that can run in parallel after the planner
DEFAULT_PLAN = list(ANALYSTS)

# keyword fallback used when no LLM plan is available or parseable
_KEYWORDS = {
    "researcher": ["business", "company", "news", "management", "strategy", "product", "outlook", "what does"],
    "quant": ["valuation", "valued", "price", "return", "risk", "volatil", "drawdown", "backtest", "momentum",
              "cheap", "expensive", "p/e", "pe ", "growth", "margin", "debt"],
    "peers": ["peer", "compare", "competitor", "versus", " vs", "sector", "industry", "relative", "valuation"],
    "sentiment": ["sentiment", "news", "headline", "mood", "market think", "perception", "media"],
}


def keyword_plan(question: str) -> list[str]:
    q = f" {question.lower()} "
    picked = [a for a, words in _KEYWORDS.items() if any(w in q for w in words)]
    if not picked or any(w in q for w in ("should", "consider", "invest", "buy", "overall")):
        return list(DEFAULT_PLAN)
    if "quant" not in picked:
        picked.append("quant")  # the risk agent needs price and ratio metrics
    return [a for a in ANALYSTS if a in picked]

ReviewFn = Callable[[AnalysisState], dict[str, Any]]


@dataclass
class Team:
    llm: Any
    tools: ToolRegistry
    max_revisions: int = 2
    reviewer: ReviewFn | None = None  # human-in-the-loop callback; None = fully automatic
    max_human_rounds: int = 3

    # ------------------------------------------------------------------ planner
    def planner(self, state: AnalysisState) -> AnalysisState:
        if state.plan:  # chosen explicitly by the user
            state.log("planner", "plan", {"analysts": state.plan, "source": "user"})
            return state
        menu = "\n".join(f"- {k}: {v}" for k, v in ANALYST_INFO.items())
        source = "llm"
        try:
            reply = self.llm.complete(prompts.PLANNER.format(analysts=menu),
                                      [{"role": "user", "content": f"Ticker: {state.ticker}\nQuestion: {state.question}"}],
                                      role="planner", context=state, max_tokens=400)
            obj = extract_json(reply)
            plan = [a for a in ANALYSTS if a in {str(x).strip().lower() for x in obj.get("analysts", [])}]
            state.focus = str(obj.get("focus", "")).strip()[:300]
            reason = str(obj.get("reason", "")).strip()[:300]
        except Exception as e:  # planning must never block the analysis
            plan, reason = [], f"LLM plan unavailable ({type(e).__name__})"
        if not plan:
            plan, source = keyword_plan(state.question), "keywords"
        if "quant" not in plan:
            plan.append("quant")  # risk rules need price and ratio metrics
        state.plan = [a for a in ANALYSTS if a in plan]
        state.log("planner", "plan", {"analysts": state.plan, "focus": state.focus, "reason": reason, "source": source})
        return state

    def route_analysts(self, state: AnalysisState) -> list[str]:
        return [a for a in ANALYSTS if a in state.plan]

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
            seed=[("get_company_profile", {}), ("get_fundamentals", {}), ("get_news", {"limit": 6})],
        )
        state.add_findings(findings)
        return state

    # ------------------------------------------------------------------ peers
    def peers(self, state: AnalysisState) -> AnalysisState:
        findings = run_react_agent(
            agent="peers",
            role_prompt=prompts.PEERS,
            task=f"Compare {state.ticker} with its sector peers.",
            llm=self.llm,
            tools=self.tools.subset(["find_peers", "compare_peers"]),
            state=state,
            max_steps=3,
            seed=[("find_peers", {}), ("compare_peers", {})],
        )
        state.add_findings(findings)
        return state

    # ------------------------------------------------------------------ sentiment
    def sentiment(self, state: AnalysisState) -> AnalysisState:
        findings = run_react_agent(
            agent="sentiment",
            role_prompt=prompts.SENTIMENT,
            task=f"Assess the tone of recent news about {state.ticker}.",
            llm=self.llm,
            tools=self.tools.subset(["score_news_sentiment"]),
            state=state,
            max_steps=2,
            seed=[("score_news_sentiment", {})],
        )
        state.add_findings(findings)
        return state

    # ------------------------------------------------------------------ quant
    def quant(self, state: AnalysisState) -> AnalysisState:
        findings = run_react_agent(
            agent="quant",
            role_prompt=prompts.QUANT,
            task=f"Analyse {state.ticker} quantitatively.",
            llm=self.llm,
            tools=self.tools.subset(["get_price_stats", "run_backtest", "compute_ratios", "run_strategy_lab"]),
            state=state,
            max_steps=5,
            seed=[("get_price_stats", {"period": "2y"}), ("run_backtest", {}), ("compute_ratios", {}),
                  ("run_strategy_lab", {})],
        )
        state.add_findings(findings)
        return state

    # ------------------------------------------------------------------ risk
    def risk(self, state: AnalysisState) -> AnalysisState:
        metrics = self._latest("get_price_stats", state) | self._latest("compute_ratios", state)
        if (sent := self._latest("score_news_sentiment", state)).get("headlines"):
            metrics["net_sentiment"] = sent["net_sentiment"]
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
            state.add_findings(_parse_findings("risk", extract_json(reply), state))
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
            f"Ticker: {state.ticker}\nInvestor question: {state.question}\n"
            + (f"Planner focus: {state.focus}\n" if state.focus else "")
            + "\n"
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
        if not state.approved and state.revisions < self.max_revisions:
            return "revise"
        if self.reviewer and state.human_rounds < self.max_human_rounds:
            return "review"
        return "done"

    # ------------------------------------------------------------------ human in the loop
    def human_review(self, state: AnalysisState) -> AnalysisState:
        """Ask a human to approve, request changes, or edit the memo directly.

        The reviewer returns {"action": "approve" | "revise" | "edit", "feedback": str, "memo": str}.
        Edited memos go back through the Critic, so human edits are held to the same evidence rules.
        """
        state.human_rounds += 1
        decision = self.reviewer(state) if self.reviewer else {"action": "approve"}
        action = str(decision.get("action", "approve")).lower()
        state.log("human", "review", {"round": state.human_rounds, "action": action,
                                      "feedback": decision.get("feedback", "")[:300]})
        if action == "revise" and decision.get("feedback"):
            state.critique = {"approved": False, "issues": [f"Human reviewer: {decision['feedback']}"],
                              "check": state.critique.get("check", {})}
            state.approved = False
            state.human_approved = False
        elif action == "edit" and decision.get("memo"):
            state.memo = str(decision["memo"]).strip()
            state.human_approved = False
        else:
            state.human_approved = True
        state.log("human", "decision", {"next": self.route_after_human(state)})
        return state

    def route_after_human(self, state: AnalysisState) -> str:
        if state.human_approved:
            return "approve"
        last = next((t for t in reversed(state.trace) if t["event"] == "review" and t["agent"] == "human"), None)
        return "recheck" if last and last["detail"]["action"] == "edit" else "revise"

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
