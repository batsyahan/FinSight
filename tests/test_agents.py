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
    assert steps[0] == "planner" and set(steps[1:4]) == {"researcher", "quant", "peers"}
    assert steps[4:] == ["risk", "writer", "critic"]
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
    steps = []
    b = run_analysis("INFY", settings=OFFLINE, engine_name="langgraph", on_step=lambda n, s: steps.append(n))
    # parallel branches may finish in a different order, so evidence ids can differ; compare structure
    heads = lambda m: [line for line in m.splitlines() if line.startswith("#")]  # noqa: E731
    assert b.approved and heads(a.memo) == heads(b.memo)
    assert sorted(e.source for e in a.evidence.items.values()) == sorted(e.source for e in b.evidence.items.values())
    assert steps[0] == "planner" and set(steps[1:4]) == {"researcher", "quant", "peers"}


def test_langgraph_human_review_loop():
    pytest.importorskip("langgraph")
    rounds = []

    def reviewer(state):
        rounds.append(1)
        return {"action": "revise", "feedback": "Shorter summary please."} if len(rounds) == 1 else {"action": "approve"}

    s = run_analysis("DEMO", settings=OFFLINE, engine_name="langgraph", reviewer=reviewer)
    assert s.human_approved and s.human_rounds == 2 and s.revisions == 1


# --------------------------------------------------------------------------- real-model failure modes
class AnswersFromMemoryLLM:
    """Tries to skip the tools and answer from memory first, like gpt-oss did on INFY.NS."""

    name = "lazy"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, **kw):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"final": {"findings": [{"text": "Revenue is $19bn.", "citations": []}]}})
        if self.calls == 2:
            return json.dumps({"action": "get_company_profile", "args": {}})
        return json.dumps({"final": {"findings": [
            {"text": "Profiled the company.", "citations": ["S1"]},
            {"text": "Uncited claim from memory.", "citations": []}]}})


def test_react_rejects_final_answer_without_tool_evidence():
    state = AnalysisState("DEMO")
    findings = run_react_agent(agent="researcher", role_prompt="x", task="go", llm=AnswersFromMemoryLLM(),
                               tools=build_tools(SampleProvider(), state), state=state)
    assert [f.text for f in findings] == ["Profiled the company."]
    events = [t["event"] for t in state.trace]
    assert "rejected_final" in events and "dropped_uncited_finding" in events


def test_retry_delay_reads_provider_hints():
    from types import SimpleNamespace

    from finsight.llm.base import retry_delay

    groq = SimpleNamespace(headers={}, text='{"error":{"message":"... Please try again in 22.68s. Need more"}}')
    assert retry_delay(groq, 0) == pytest.approx(23.68)
    assert retry_delay(SimpleNamespace(headers={"retry-after": "7"}, text=""), 0) == 8
    assert retry_delay(SimpleNamespace(headers={}, text="try again in 1m30s"), 0) == 90  # capped
    assert retry_delay(None, 2) == 4


def test_openai_client_handles_rate_limit_and_reasoning_models(monkeypatch):
    import finsight.llm.base as base
    from finsight.llm.providers import OpenAICompatLLM

    class Resp:
        def __init__(self, code, body, text=""):
            self.status_code, self._body, self.text, self.headers = code, body, text, {}

        def json(self):
            return self._body

    sent, replies = [], [
        Resp(429, None, "Rate limit reached. Please try again in 0.01s."),
        Resp(200, {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}]}),
        Resp(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}),
    ]
    monkeypatch.setattr(base.requests, "post", lambda url, **kw: sent.append(kw["json"]) or replies.pop(0))
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    llm = OpenAICompatLLM("k", "openai/gpt-oss-120b", "https://x/v1")
    assert llm.complete("s", [{"role": "user", "content": "hi"}]) == '{"ok": 1}'
    assert sent[0]["reasoning_effort"] == "low" and sent[0]["max_tokens"] >= 3000
    with pytest.raises(base.LLMError, match="finish_reason=length"):
        llm.complete("s", [{"role": "user", "content": "hi"}])


def test_verifier_normalises_unicode_dashes():
    ev = EvidenceStore()
    ev.add("get_price_stats", "stats", {"high_52w": 1654.01, "return_total_pct": -30.83})
    memo = "The 52‑week high was 1654.01 and the 2‑year return was −30.83% [S1]."
    assert verify_memo(memo, ev)["passed"]


# --------------------------------------------------------------------------- native tool calling
class NativeMock(MockLLM):
    """Native tool-calling model. Mixes a from-memory claim in with real findings, the way
    gpt-oss did on INFY.NS. Answers via submit_findings."""

    def __init__(self):
        self.calls = 0

    def chat_tools(self, system, messages, tools, *, tool_choice="auto", role="", context=None, max_tokens=1500):
        self.calls += 1
        f = self.findings_for_role(role, context)
        f = f + [{"text": "From memory: revenue was $19bn.", "citations": []}]
        return {"content": "", "tool_calls": [{"id": "done", "name": "submit_findings", "args": {"findings": f}}]}


def test_native_loop_seeds_core_evidence_and_drops_memory_claims():
    llm = NativeMock()
    steps = []
    state = run_analysis("INFY", llm=llm, settings=OFFLINE, engine_name="builtin",
                         on_step=lambda n, s: steps.append((n, len(s.evidence.items))))
    sources = [ev.source for ev in state.evidence.items.values()]
    assert {"get_company_profile", "get_price_stats", "compare_peers"} <= set(sources)
    assert llm.calls == 3  # one LLM call per ReAct agent (researcher, quant, peers)
    assert state.approved and all("From memory" not in f.text for f in state.findings)
    seeded = [t for t in state.trace if t["event"] == "tool" and t["detail"]["seeded"]]
    assert len(seeded) == 8  # 3 researcher + 3 quant + 2 peers


class TextAnswerMock(MockLLM):
    """Ignores tools and answers in markdown, like gpt-oss's failed_generation."""

    def chat_tools(self, system, messages, tools, *, tool_choice="auto", role="", context=None, max_tokens=1500):
        ids = [m["content"].split()[1] for m in messages if m["role"] == "tool"]
        return {"content": "**Findings**\n\n"
                           f"1. **Profile** – company profiled from data [{ids[0]}]\n"
                           f"2. Price and fundamentals reviewed ({ids[1]}, {ids[-1]})\n"
                           "3. Strong brand loyalty from memory\n", "tool_calls": []}


def test_native_loop_accepts_markdown_findings_with_citations():
    state = run_analysis("INFY", llm=TextAnswerMock(), settings=OFFLINE, engine_name="builtin")
    texts = [f.text for f in state.findings_for("researcher")]
    assert texts == ["Profile – company profiled from data.", "Price and fundamentals reviewed."]
    cited = state.findings_for("researcher")[1].citations
    assert [state.evidence.get(c).source for c in cited] == ["get_fundamentals", "get_news"]


def test_native_loop_salvages_groq_failed_generation():
    """Groq HTTP 400 tool_use_failed carries the model's text in failed_generation."""
    from finsight.llm.base import LLMError

    class GroqRejects(MockLLM):
        def chat_tools(self, system, messages, tools, **kw):
            body = json.dumps({"error": {"code": "tool_use_failed", "message": "Tool choice is required...",
                                         "failed_generation": '{"name": "JSON", "arguments": {"findings": ['
                                                              '{"text": "Data reviewed.", "citations": ["S1"]}]}}'}})
            raise LLMError("LLM request failed: HTTP 400: tool_use_failed", body=body)

    state = run_analysis("INFY", llm=GroqRejects(), settings=OFFLINE, engine_name="builtin")
    assert [f.text for f in state.findings_for("researcher")] == ["Data reviewed."]
    assert any(t["event"] == "provider_rejected_call" for t in state.trace)


def test_openai_chat_tools_round_trip(monkeypatch):
    import finsight.llm.base as base
    from finsight.llm.providers import OpenAICompatLLM

    sent = []

    class Resp:
        status_code, text, headers = 200, "", {}

        def json(self):
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_price_stats", "arguments": '{"period": "1y"}'}}]}}]}

    monkeypatch.setattr(base.requests, "post", lambda url, **kw: sent.append(kw["json"]) or Resp())
    llm = OpenAICompatLLM("k", "openai/gpt-oss-120b", "https://x/v1")
    history = [{"role": "user", "content": "go"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "name": "get_news", "args": {}}]},
               {"role": "tool", "tool_call_id": "c0", "name": "get_news", "content": "Evidence S1 ..."}]
    schema = build_tools(SampleProvider(), AnalysisState("X")).subset(["get_price_stats"]).schemas()
    out = llm.chat_tools("sys", history, schema, tool_choice="required")
    assert out["tool_calls"] == [{"id": "c1", "name": "get_price_stats", "args": {"period": "1y"}}]
    p = sent[0]
    assert p["tool_choice"] == "required" and p["tools"][0]["function"]["name"] == "get_price_stats"
    assert p["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert p["messages"][3] == {"role": "tool", "tool_call_id": "c0", "content": "Evidence S1 ..."}


def test_anthropic_message_conversion_alternates_roles():
    from finsight.llm.providers import _to_anthropic

    out = _to_anthropic([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "a", "args": {}},
                                                            {"id": "t2", "name": "b", "args": {"x": 1}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "a", "content": "r1"},
        {"role": "tool", "tool_call_id": "t2", "name": "b", "content": "r2"},
        {"role": "user", "content": "now answer"},
    ])
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t2", "name": "b", "input": {"x": 1}}


def test_verifier_flags_currency_not_in_evidence():
    ev = EvidenceStore()
    ev.add("get_fundamentals", "fund", {"statement_currency": "INR", "price_currency": "INR", "eps": 75.1})
    assert not verify_memo("EPS was $75.1 [S1].", ev)["passed"]
    assert verify_memo("EPS was ₹75.1 [S1].", ev)["passed"]
    ev2 = EvidenceStore()
    ev2.add("get_fundamentals", "fund", {"statement_currency": "USD", "price_currency": "INR", "eps": 75.1})
    assert verify_memo("EPS was ₹75.1 [S1].", ev2)["passed"]


def test_verifier_only_scales_figures_with_a_unit_word():
    ev = EvidenceStore()
    ev.add("get_fundamentals", "fund", {"revenue": 377_000_000_000.0})
    assert verify_memo("Revenue was 37,700 crore [S1].", ev)["passed"]
    assert verify_memo("Revenue was 377 billion [S1].", ev)["passed"]
    # without a unit, 37.7 must not match 3.77e11 by silently scaling it
    assert not verify_memo("Targets imply 37.7% upside [S1].", ev)["passed"]
