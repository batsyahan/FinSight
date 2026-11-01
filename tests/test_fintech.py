import numpy as np
import pandas as pd
import pytest

from finsight.config import Settings
from finsight.data.providers import SampleProvider
from finsight.graph.workflow import run_analysis
from finsight.llm.mock import MockLLM
from finsight.portfolio import analyze_portfolio, parse_holdings, portfolio_markdown, portfolio_summary
from finsight.state import EvidenceStore
from finsight.tools.sentiment import news_sentiment, score_headline
from finsight.tools.strategies import backtest, momentum, rsi, strategy_lab, walk_forward
from finsight.verify import verify_memo

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


def series(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- grounding scope
def test_figures_must_come_from_the_evidence_their_sentence_cites():
    ev = EvidenceStore()
    ev.add("a", "a", {"margin_pct": 18.52})
    ev.add("b", "b", {"return_pct": -30.83})
    assert verify_memo("Margin was 18.52% [S1].", ev)["passed"]
    res = verify_memo("Margin was 18.52% [S2].", ev)
    assert not res["passed"] and "appears in [S1]" in res["issues"][0]


# --------------------------------------------------------------------------- sentiment
def test_headline_scoring_and_negation():
    assert score_headline("Infosys beats estimates, raises guidance")["label"] == "positive"
    assert score_headline("Regulator issues notice; shares plunge")["label"] == "negative"
    assert score_headline("Company reports no growth this quarter")["score"] < 0  # negated positive
    assert score_headline("CEO steps down unexpectedly")["terms"] == ["steps down"]
    assert score_headline("Board meeting scheduled for Tuesday")["label"] == "neutral"


def test_news_sentiment_aggregates_and_detects_trend():
    items = [{"title": "Shares plunge on probe", "date": "2026-10-01"},
             {"title": "Losses widen, outlook weak", "date": "2026-10-05"},
             {"title": "Company wins large deal", "date": "2026-10-20"},
             {"title": "Profit surges to record", "date": "2026-10-28"}]
    s = news_sentiment(items)
    assert (s["positive"], s["negative"], s["net_sentiment"]) == (2, 2, 0.0)
    assert s["trend"]["direction"] == "improving"
    assert news_sentiment([])["headlines"] == 0


# --------------------------------------------------------------------------- strategies
def test_strategies_cannot_see_the_future():
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    a = series(base)
    b = series(np.r_[base[:-1], base[-1] * 3])  # only the last day differs (a huge jump)
    pa, pb = momentum(a, 20), momentum(b, 20)
    assert (pa.iloc[:-1] == pb.iloc[:-1]).all()  # earlier positions unaffected by a future price
    # and the jump day's return is only earned if the position was already held the day before
    assert backtest(b, pb)["total_return_pct"] != backtest(b, pb.shift(-1).fillna(0))["total_return_pct"]


def test_costs_reduce_returns():
    p = SampleProvider().prices("DEMO", "2y")
    pos = momentum(p, 20)
    free, costly = backtest(p, pos, cost_bps=0), backtest(p, pos, cost_bps=50)
    assert costly["total_return_pct"] < free["total_return_pct"] and costly["costs_paid_pct"] > 0


def test_rsi_bounds_and_walk_forward():
    p = SampleProvider().prices("ACME", "5y")
    r = rsi(p)
    assert r.between(0, 100).all()
    wf = walk_forward(p, "momentum")
    assert wf["folds"] >= 4 and len(wf["params_chosen"]) == wf["folds"]
    assert wf["overfit_gap"] == pytest.approx(wf["in_sample_sharpe"] - wf["out_of_sample_sharpe"], abs=0.02)
    lab = strategy_lab(p)
    assert set(lab["strategies"]) == {"sma_crossover", "momentum", "rsi_reversion", "bollinger_reversion"}
    assert lab["best_out_of_sample"] in lab["walk_forward"]


# --------------------------------------------------------------------------- portfolio
def test_parse_holdings():
    assert parse_holdings("A:60, B:40") == {"A": 0.6, "B": 0.4}
    assert parse_holdings("a,b,c,d") == {t: 0.25 for t in "ABCD"}
    with pytest.raises(ValueError):
        parse_holdings("A:-1,B:2")


def test_portfolio_risk_math():
    p = SampleProvider()
    a = analyze_portfolio({"DEMO": 0.5, "ACME": 0.3, "GLOBEX": 0.2}, p)
    v95, v99 = a["var_1d_95"], a["var_1d_99"]
    assert v99["historical_var_pct"] >= v95["historical_var_pct"] > 0
    assert v95["expected_shortfall_pct"] >= v95["historical_var_pct"]  # ES is the average beyond VaR
    assert sum(h["risk_contribution_pct"] for h in a["holdings"].values()) == pytest.approx(100, abs=0.1)
    assert a["correlation"]["DEMO"]["DEMO"] == 1.0
    assert sum(a["sector_weights_pct"].values()) == pytest.approx(100, abs=0.1)
    assert any(f["id"] == "single_name_concentration" for f in a["flags"])  # DEMO is 50%


def test_identical_holdings_have_no_diversification():
    class Twins(SampleProvider):
        def prices(self, ticker, period="2y"):
            return super().prices("DEMO", period)

    a = analyze_portfolio({"X": 0.5, "Y": 0.5}, Twins())
    assert a["metrics"]["avg_pairwise_correlation"] == 1.0
    assert a["metrics"]["diversification_ratio"] == pytest.approx(1.0, abs=0.01)
    assert any(f["id"] == "high_correlation" for f in a["flags"])


def test_portfolio_summary_is_grounded():
    a = analyze_portfolio(parse_holdings("DEMO:40,ACME:30,GLOBEX:30"), SampleProvider())
    out = portfolio_summary(a, MockLLM())
    assert out["grounding"]["passed"], out["grounding"]["issues"]
    md = portfolio_markdown(a, out)
    assert "| DEMO | 40.0 |" in md and "## Risk" in md


# --------------------------------------------------------------------------- pipeline
def test_sentiment_and_strategy_lab_reach_the_memo():
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    assert state.approved
    assert "## News Sentiment" in state.memo and "walk-forward" in state.memo
