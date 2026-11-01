import json

import pytest

from finsight.agents.react import run_react_agent
from finsight.config import Settings
from finsight.data.providers import SampleProvider
from finsight.graph import engine
from finsight.graph.workflow import run_analysis
from finsight.llm.base import extract_json
from finsight.llm.mock import MockLLM
from finsight.state import AnalysisState, EvidenceStore
from finsight.tools.registry import build_tools
from finsight.verify import verify_memo

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


# --------------------------------------------------------------------------- verifier
def _store():
    ev = EvidenceStore()
    ev.add("get_price_stats", "stats", {"return_1y_pct": 23.39, "sharpe": 1.12})
    ev.add("get_fundamentals", "fund", {"revenue": 412_345_678_900.0})
    return ev


def test_verifier_accepts_grounded_memo_with_unit_scaling():
    memo = ("## Summary\nThe stock returned 23.39% over one year [S1]. Revenue was about "
            "41,234.57 crore [S2]. Its 52-week range is wide [S1].\n## Disclaimer\nEducational research, not advice.")
    res = verify_memo(memo, _store())
    assert res["passed"], res["issues"]


def test_verifier_flags_fabricated_number_bad_citation_and_missing_citation():
    memo = "The stock returned 31.7% [S1]. Sharpe was 1.12 [S9]. Margins were 18.2%."
    res = verify_memo(memo, _store())
    assert not res["passed"]
    text = " ".join(res["issues"])
    assert "31.7" in text and "[S9]" in text and "no citation" in text
    assert res["stats"]["unsupported_numbers"] == 2


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    assert extract_json('thinking... {"x": 1} trailing') == {"x": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


# --------------------------------------------------------------------------- react loop
class FlakyLLM:
    """Bad JSON, then an unknown tool, then a real tool, then a final answer."""

    name = "flaky"

    def __init__(self):
        self.replies = [
            "I think I should look at prices",
            json.dumps({"action": "teleport", "args": {}}),
            json.dumps({"action": "get_price_stats", "args": {"period": "1y"}}),
            json.dumps({"final": {"findings": [{"text": "Volatile.", "citations": ["S1", "S99"]}]}}),
        ]

    def complete(self, system, messages, **kw):
        return self.replies.pop(0)


def test_react_loop_recovers_from_bad_json_and_unknown_tools():
    state = AnalysisState("DEMO")
    tools = build_tools(SampleProvider(), state)
    findings = run_react_agent(agent="quant", role_prompt="x", task="go", llm=FlakyLLM(), tools=tools, state=state)
    assert [f.text for f in findings] == ["Volatile."]
    assert findings[0].citations == ["S1"]  # hallucinated S99 dropped
    events = [t["event"] for t in state.trace]
    assert "parse_error" in events and events.count("tool") == 2


# --------------------------------------------------------------------------- full graph
def test_offline_pipeline_produces_approved_cited_memo():
    steps = []
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", on_step=lambda n, s: steps.append(n))
    assert steps == ["researcher", "quant", "risk", "writer", "critic"]
    assert state.approved and state.revisions == 0
    assert "## Key Risks" in state.memo and "[S" in state.memo
    assert "SYNTHETIC" in state.memo
    assert not state.errors


class HallucinatingWriterLLM(MockLLM):
    """First draft invents a figure; the Critic must reject it and the Writer must fix it."""

    def __init__(self):
        self.drafts = 0

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
        if role == "writer":
            self.drafts += 1
            memo = super().complete(system, messages, role=role, context=context)
            if self.drafts == 1:
                memo = memo.replace("## Outlook", "Analysts expect 47.5% upside next year.\n\n## Outlook")
            return memo
        return super().complete(system, messages, role=role, context=context)


def test_critic_rejects_hallucination_and_writer_revises():
    llm = HallucinatingWriterLLM()
    steps = []
    state = run_analysis("DEMO", llm=llm, settings=OFFLINE, engine_name="builtin",
                         on_step=lambda n, s: steps.append(n))
    assert steps[-4:] == ["writer", "critic", "writer", "critic"]
    assert state.revisions == 1 and state.approved
    assert "47.5" not in state.memo


def test_revision_budget_is_enforced():
    class AlwaysBad(MockLLM):
        def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
            out = super().complete(system, messages, role=role, context=context)
            return out + "\nUpside of 99.9% is likely." if role == "writer" else out

    state = run_analysis("DEMO", llm=AlwaysBad(), settings=OFFLINE, engine_name="builtin")
    assert not state.approved and state.revisions == 2
    assert any("99.9" in i for i in state.critique["issues"])


def test_engine_guards_against_infinite_loops():
    g = engine.StateGraph()
    g.add_node("a", lambda s: s)
    g.add_edge(engine.START, "a")
    g.add_conditional_edges("a", lambda s: "again", {"again": "a"})
    with pytest.raises(RuntimeError):
        g.compile(max_steps=5).invoke({})


def test_langgraph_engine_matches_builtin():
    pytest.importorskip("langgraph")
    a = run_analysis("INFY", settings=OFFLINE, engine_name="builtin")
    b = run_analysis("INFY", settings=OFFLINE, engine_name="langgraph")
    assert a.memo == b.memo and b.approved
