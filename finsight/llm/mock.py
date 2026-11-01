"""Deterministic offline "LLM" so FinSight runs end to end with no API key.

It follows the exact same JSON protocols as a real model, but decides from the live
AnalysisState instead of from language understanding. Good for demos, tests and CI;
use a real model for genuine analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from finsight.state import AnalysisState

PLANS = {
    "researcher": [("get_company_profile", {}), ("get_fundamentals", {}), ("get_news", {"limit": 5})],
    "quant": [("get_price_stats", {"period": "2y"}), ("run_backtest", {"fast": 50, "slow": 200}), ("compute_ratios", {})],
}


class MockLLM:
    name = "mock"

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500) -> str:
        state: AnalysisState = context
        if role in PLANS:
            return self._react(role, messages, state)
        if role == "risk":
            return json.dumps({"findings": self._risk_findings(state)})
        if role == "writer":
            return self._memo(state)
        if role == "critic":
            return self._critic(messages)
        return json.dumps({"final": {"findings": []}})

    # ---------------------------------------------------------------- ReAct agents
    def _react(self, role: str, messages: list[dict[str, str]], state: AnalysisState) -> str:
        done = set()
        for m in messages:
            if m["role"] == "user":
                hit = re.match(r"(?:Observation from|Tool) (\w+)", m["content"])
                if hit:
                    done.add(hit.group(1))
        for tool, args in PLANS[role]:
            if tool not in done:
                return json.dumps({"thought": f"I need {tool} to answer this.", "action": tool, "args": args})
        findings = self._researcher_findings(state) if role == "researcher" else self._quant_findings(state)
        return json.dumps({"thought": "I have what I need.", "final": {"findings": findings}})

    @staticmethod
    def _ev(state: AnalysisState, source: str):
        for ev in reversed(list(state.evidence.items.values())):
            if ev.source == source:
                return ev
        return None

    def _researcher_findings(self, s: AnalysisState) -> list[dict[str, Any]]:
        out = []
        if p := self._ev(s, "get_company_profile"):
            d = p.data
            out.append({"text": f"{d.get('name')} operates in {d.get('industry')} ({d.get('sector')}).",
                        "citations": [p.id]})
        if n := self._ev(s, "get_news"):
            heads = [i["title"] for i in n.data.get("items", [])[:3]]
            out.append({"text": "Recent headlines: " + "; ".join(heads) + ".", "citations": [n.id]})
        return out

    def _quant_findings(self, s: AnalysisState) -> list[dict[str, Any]]:
        out = []
        if ps := self._ev(s, "get_price_stats"):
            d = ps.data
            out.append({"text": f"1-year return {d['return_1y_pct']}%, volatility {d['ann_volatility_pct']}%, "
                                f"Sharpe {d['sharpe']}, max drawdown {d['max_drawdown_pct']}%.", "citations": [ps.id]})
        if bt := self._ev(s, "run_backtest"):
            d = bt.data
            out.append({"text": f"{d['strategy']} returned {d['strategy_return_pct']}% vs buy-and-hold "
                                f"{d['buy_hold_return_pct']}%.", "citations": [bt.id]})
        if r := self._ev(s, "compute_ratios"):
            d = r.data
            out.append({"text": f"Revenue growth {d['revenue_growth_pct']}%, net margin {d['net_margin_pct']}%, "
                                f"debt/equity {d['debt_to_equity']}.", "citations": [r.id]})
        return out

    def _risk_findings(self, s: AnalysisState) -> list[dict[str, Any]]:
        ev = self._ev(s, "risk_rules")
        if not ev:
            return []
        if not s.risk_flags:
            return [{"text": "No rule-based risk thresholds were breached.", "citations": [ev.id]}]
        return [{"text": f"{f['severity'].title()} risk: {f['message']}", "citations": [ev.id]} for f in s.risk_flags[:5]]

    # ---------------------------------------------------------------- writer
    def _memo(self, s: AnalysisState) -> str:
        e = lambda src: self._ev(s, src)  # noqa: E731
        prof, fund, news = e("get_company_profile"), e("get_fundamentals"), e("get_news")
        ps, bt, rat, risk = e("get_price_stats"), e("run_backtest"), e("compute_ratios"), e("risk_rules")
        name = prof.data.get("name") if prof else s.ticker
        cur = (prof.data.get("price_currency") if prof else None) or ""
        stmt_cur = (fund.data.get("statement_currency") if fund else None) or cur
        synthetic = any("SYNTHETIC" in str(ev.data.get("data_note", "")) for ev in s.evidence.items.values())
        L: list[str] = [f"# {name} ({s.ticker}): Research Memo", "", "## Summary", ""]
        if synthetic:
            L.append("This memo was generated from SYNTHETIC SAMPLE DATA for demonstration, not real market data.")
        if ps and rat:
            L.append(f"Over the analysed period the stock returned {ps.data['return_total_pct']}% with annualised "
                     f"volatility of {ps.data['ann_volatility_pct']}% [{ps.id}], while revenue growth was "
                     f"{rat.data['revenue_growth_pct']}% [{rat.id}].")
        if risk:
            L.append(f"The rule-based risk score is {risk.data['risk_score']} out of 100 [{risk.id}].")

        L += ["", "## Business Overview", ""]
        if prof:
            mc = prof.data.get("market_cap")
            L.append(f"{name} operates in {prof.data.get('industry')} within the {prof.data.get('sector')} sector"
                     + (f", with a market capitalisation of about {mc / 1e7:,.2f} crore {cur}" if mc else "")
                     + f" [{prof.id}].")
        if news and news.data.get("items"):
            heads = [i["title"].replace("[SAMPLE] ", "") for i in news.data["items"][:3]]
            L.append(f"Recent news includes: {'; '.join(heads)} [{news.id}].")

        L += ["", "## Financial Health", ""]
        if fund:
            rev = fund.data.get("revenue")
            if rev:
                L.append(f"Latest annual revenue was about {rev / 1e7:,.2f} crore {stmt_cur} [{fund.id}].")
        if rat:
            d = rat.data
            parts = [f"net margin of {d['net_margin_pct']}%" if d.get("net_margin_pct") is not None else None,
                     f"free-cash-flow margin of {d['fcf_margin_pct']}%" if d.get("fcf_margin_pct") is not None else None,
                     f"return on equity of {d['roe_pct']}%" if d.get("roe_pct") is not None else None]
            L.append("The company reports a " + ", ".join(p for p in parts if p) + f" [{rat.id}].")
            parts2 = [f"debt-to-equity of {d['debt_to_equity']}" if d.get("debt_to_equity") is not None else None,
                      f"a current ratio of {d['current_ratio']}" if d.get("current_ratio") is not None else None,
                      f"a P/E ratio of {d['pe_ratio']}" if d.get("pe_ratio") is not None else None]
            if any(parts2):
                L.append("Balance sheet and valuation: " + ", ".join(p for p in parts2 if p) + f" [{rat.id}].")

        L += ["", "## Price Performance and Backtest", ""]
        if ps:
            d = ps.data
            L.append(f"The one-year return was {d['return_1y_pct']}% and the compound annual growth rate was "
                     f"{d['cagr_pct']}% [{ps.id}].")
            L.append(f"The Sharpe ratio was {d['sharpe']} against a {d['risk_free_pct']}% risk-free rate, and the "
                     f"maximum drawdown was {d['max_drawdown_pct']}% [{ps.id}].")
        if bt:
            d = bt.data
            L.append(f"A {d['strategy']} strategy returned {d['strategy_return_pct']}% versus "
                     f"{d['buy_hold_return_pct']}% for buy-and-hold, spending {d['time_in_market_pct']}% of days "
                     f"invested [{bt.id}].")

        L += ["", "## Key Risks", ""]
        if s.risk_flags and risk:
            for f in s.risk_flags[:5]:
                L.append(f"- **{f['severity'].title()}**: {f['message'].rstrip('.')} [{risk.id}].")
        elif risk:
            L.append(f"No rule-based risk thresholds were breached [{risk.id}].")

        L += ["", "## Outlook", ""]
        high = sum(f["severity"] == "high" for f in s.risk_flags)
        view = "Cautious" if high >= 2 else "Neutral" if high == 1 or len(s.risk_flags) >= 3 else "Positive"
        L.append(f"**{view}.** This view weighs the growth and return figures above against the flagged risks; "
                 "investors should form their own judgement and consider their own circumstances.")
        L += ["", "## Disclaimer", "",
              "This memo is automated educational research generated by FinSight. It is not financial advice."]
        return "\n".join(L)

    # ---------------------------------------------------------------- critic
    @staticmethod
    def _critic(messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        m = re.search(r"Automated check: (\{.*?\})\n\n", content, re.DOTALL)
        passed = False
        if m:
            try:
                passed = bool(json.loads(m.group(1)).get("passed"))
            except json.JSONDecodeError:
                pass
        return json.dumps({"approved": passed, "issues": [] if passed else ["Fix the automated check issues."]})
