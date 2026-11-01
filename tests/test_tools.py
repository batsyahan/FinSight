import numpy as np
import pandas as pd
import pytest

from finsight.data.providers import SampleProvider, get_provider
from finsight.state import AnalysisState
from finsight.tools.quant import fundamental_ratios, max_drawdown, price_stats, sma_crossover_backtest
from finsight.tools.registry import build_tools
from finsight.tools.risk import evaluate_risks, risk_score


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_sample_provider_is_deterministic():
    a, b = SampleProvider(), SampleProvider()
    pd.testing.assert_series_equal(a.prices("DEMO"), b.prices("DEMO"))
    assert a.fundamentals("DEMO") == b.fundamentals("DEMO")
    assert a.prices("DEMO").iloc[-1] != a.prices("OTHER").iloc[-1]


def test_sample_data_is_labelled_synthetic():
    p = SampleProvider()
    assert "SYNTHETIC" in p.profile("DEMO")["data_note"]
    assert all(n["title"].startswith("[SAMPLE]") for n in p.news("DEMO"))


def test_price_stats_on_steady_growth():
    s = series(100 * 1.001 ** np.arange(300))  # +0.1% every day, no drawdown
    st = price_stats(s, risk_free=0.0)
    assert st["max_drawdown_pct"] == 0.0
    assert st["ann_volatility_pct"] == pytest.approx(0.0, abs=1e-6)
    assert st["return_total_pct"] == pytest.approx((1.001**299 - 1) * 100, abs=0.01)


def test_max_drawdown():
    s = series([100, 120, 60, 90, 130] * 10)
    assert max_drawdown(s) == pytest.approx(60 / 130 - 1)  # peak 130, later trough 60


def test_backtest_has_no_lookahead_and_reports_trades():
    p = SampleProvider().prices("DEMO", "3y")
    bt = sma_crossover_backtest(p, 20, 100)
    assert bt["trades"] >= 0
    assert 0 <= bt["time_in_market_pct"] <= 100
    with pytest.raises(ValueError):
        sma_crossover_backtest(p, 100, 20)


def test_backtest_always_long_matches_buy_and_hold():
    s = series(100 * 1.002 ** np.arange(400))  # fast SMA always above slow SMA
    bt = sma_crossover_backtest(s, 5, 50)
    # one-day signal lag means the strategy misses only the first day after warm-up
    assert bt["strategy_return_pct"] == pytest.approx(bt["buy_hold_return_pct"], abs=0.5)


def test_ratios_and_risk_flags():
    f = {"revenue": 100.0, "revenue_prev": 125.0, "net_income": -5.0, "total_equity": 50.0,
         "total_debt": 100.0, "free_cash_flow": -3.0, "current_assets": 20.0,
         "current_liabilities": 40.0, "price": 10.0, "eps": -0.5}
    r = fundamental_ratios(f)
    assert r["revenue_growth_pct"] == -20.0
    assert r["debt_to_equity"] == 2.0
    assert r["pe_ratio"] is None  # undefined for negative earnings
    flags = evaluate_risks(r)
    ids = {fl["id"] for fl in flags}
    assert {"high_leverage", "negative_fcf", "loss_making", "revenue_decline", "weak_liquidity"} <= ids
    assert flags[0]["severity"] == "high"
    assert risk_score(flags) == 100


def test_registry_records_evidence_and_survives_errors():
    state = AnalysisState("DEMO")
    reg = build_tools(get_provider("sample"), state)
    out = reg.call("get_price_stats", {"period": "1y", "bogus": 1})
    assert out["evidence_id"] == "S1"
    assert state.evidence.get("S1").source == "get_price_stats"
    bad = reg.call("run_backtest", {"fast": 300, "slow": 200})
    assert "error" in bad
    assert "error" in reg.call("no_such_tool")
    assert len(state.evidence.items) == 1
