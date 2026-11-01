import time

import pytest

from finsight.config import Settings
from finsight.data.providers import SampleProvider
from finsight.graph import engine
from finsight.graph.workflow import run_analysis
from finsight.tools.peers import compare_peers, find_peers

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


class FakeYahoo(SampleProvider):
    name = "yfinance"  # pretend to be live data so curated groups are used


def test_find_peers_uses_curated_sector_groups():
    out = find_peers("INFY.NS", FakeYahoo())
    assert out["group"] == "Indian IT services" and "INFY.NS" not in out["peers"]
    assert out["peers"][0] == "TCS.NS"
    assert find_peers("INFY.BO", FakeYahoo())["peers"][0] == "TCS.BO"  # keeps the exchange suffix
    assert find_peers("UNKNOWNCO.NS", FakeYahoo())["peers"] == []
    assert len(find_peers("DEMO", SampleProvider())["peers"]) == 4


def test_compare_peers_ranks_subject_and_tolerates_bad_peer():
    class OnePeerBroken(SampleProvider):
        def prices(self, ticker, period="2y"):
            if ticker == "BROKEN":
                raise ValueError("no data")
            return super().prices(ticker, period)

    out = compare_peers("DEMO", ["ACME", "GLOBEX", "BROKEN"], OnePeerBroken())
    assert out["peers"] == ["ACME", "GLOBEX"] and "BROKEN" in out["failed"]
    r = out["vs_peers"]["return_1y_pct"]
    assert 1 <= r["rank"] <= r["out_of"] == 3
    vals = sorted([row["return_1y_pct"] for row in out["table"].values()], reverse=True)
    assert vals.index(out["table"]["DEMO"]["return_1y_pct"]) + 1 == r["rank"]


def test_pipeline_includes_peer_section_and_respects_overrides():
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", peers=["HOOLI", "STARK"])
    assert state.approved and "## Peer Comparison" in state.memo
    ev = [e for e in state.evidence.items.values() if e.source == "compare_peers"][0]
    assert ev.data["peers"] == ["HOOLI", "STARK"]


def test_analyst_selection_skips_unrequested_agents():
    steps = []
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", analysts=["quant"],
                         on_step=lambda n, s: steps.append(n))
    assert steps == ["planner", "quant", "risk", "writer", "critic"]
    assert "## Peer Comparison" not in state.memo and state.approved
    with pytest.raises(ValueError):
        run_analysis("DEMO", settings=OFFLINE, analysts=["astrologer"])


def test_builtin_engine_runs_branches_concurrently():
    g = engine.StateGraph()
    seen = []
    for n in ("a", "b", "c"):
        g.add_node(n, lambda s, n=n: (time.sleep(0.3), seen.append(n), s)[2])
    g.add_node("join", lambda s: s)
    g.add_conditional_edges(engine.START, lambda s: ["a", "b", "c"])
    for n in ("a", "b", "c"):
        g.add_edge(n, "join")
    t0 = time.perf_counter()
    g.compile().invoke({})
    assert time.perf_counter() - t0 < 0.6  # three 0.3s branches in parallel, not 0.9s in sequence
    assert sorted(seen) == ["a", "b", "c"]


def test_sample_periods_are_slices_of_one_path_and_peer_figures_match_quant():
    p = SampleProvider()
    one, two = p.prices("DEMO", "1y"), p.prices("DEMO", "2y")
    assert (two.iloc[-len(one):].values == one.values).all()
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    stats = next(e for e in state.evidence.items.values() if e.source == "get_price_stats").data
    peer = next(e for e in state.evidence.items.values() if e.source == "compare_peers").data
    assert peer["table"]["DEMO"]["return_1y_pct"] == stats["return_1y_pct"]
