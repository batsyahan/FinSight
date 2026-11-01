import json

import pytest

from finsight import eval as ev
from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.llm.mock import MockLLM
from finsight.obs import instrument, usage_summary
from finsight.state import AnalysisState

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


def test_telemetry_records_every_llm_call_per_agent():
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    s = usage_summary(state)
    assert set(s["per_agent"]) == {"researcher", "quant", "peers", "risk", "writer", "critic"}
    assert s["totals"]["llm_calls"] == len(state.llm_calls) > 0
    assert s["totals"]["tokens_estimated"] is True  # the mock reports no usage
    assert set(s["node_seconds"]) == {"planner", "researcher", "quant", "peers", "risk", "writer", "critic"}


def test_telemetry_uses_provider_token_counts_and_prices(monkeypatch):
    import finsight.llm.base as base
    from finsight.llm.providers import OpenAICompatLLM

    class Resp:
        status_code, text, headers = 200, "", {}

        def json(self):
            return {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}

    monkeypatch.setattr(base.requests, "post", lambda url, **kw: Resp())
    monkeypatch.setenv("FINSIGHT_PRICE_IN", "2")   # $ per million input tokens
    monkeypatch.setenv("FINSIGHT_PRICE_OUT", "8")
    state = AnalysisState("X")
    llm = instrument(OpenAICompatLLM("k", "some-model", "https://x/v1"), state)
    llm.complete("sys", [{"role": "user", "content": "q"}], role="writer")
    rec = state.llm_calls[0]
    assert (rec["input_tokens"], rec["output_tokens"], rec["estimated"]) == (1000, 500, False)
    assert rec["cost_usd"] == pytest.approx(1000 / 1e6 * 2 + 500 / 1e6 * 8)
    assert hasattr(llm, "chat_tools")  # tool-capable models keep native tool calling


def test_instrument_preserves_capabilities():
    state = AnalysisState("X")
    assert not hasattr(instrument(MockLLM(), state), "chat_tools")


def test_eval_on_demo_universe(tmp_path):
    result = ev.run_eval(ev.UNIVERSES["demo"], settings=OFFLINE, engine_name="builtin")
    s = result["summary"]
    assert s["runs"] == 5 and s["approval_rate_pct"] == 100.0 and s["grounded_figures_pct"] == 100.0
    j, md = ev.save(result, tmp_path)
    assert json.loads(j.read_text())["summary"] == s
    assert "| DEMO | ✅ |" in md.read_text()


class Hallucinator(MockLLM):
    """Adds an unsupported figure to every draft, so the Critic can never approve."""

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
        out = super().complete(system, messages, role=role, context=context)
        return out + "\nTargets imply 37.7% upside [S1]." if role == "writer" else out


def test_eval_measures_ungrounded_memos_and_survives_crashes():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("provider down")
        return Hallucinator()

    result = ev.run_eval(["DEMO", "ACME", "GLOBEX"], settings=OFFLINE, llm_factory=factory, engine_name="builtin")
    s = result["summary"]
    assert s["runs"] == 3 and s["completed"] == 2 and s["approval_rate_pct"] == 0.0
    assert s["grounded_figures_pct"] < 100.0 and s["avg_revisions"] == 2
    assert result["runs"][1]["error"].startswith("RuntimeError")
    table = ev.compare_markdown([result, ev.run_eval(["DEMO"], settings=OFFLINE, engine_name="builtin")])
    assert table.count("| mock / (default) |") == 2
