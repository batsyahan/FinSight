import numpy as np
import pandas as pd
import pytest

from finsight.config import Settings
from finsight.data.providers import SampleProvider
from finsight.graph.workflow import run_analysis
from finsight.store import Store
from finsight.tools.backtest import (UNAVAILABLE_STRATEGIES, BacktestConfig, run_backtest, run_strategy,
                                     validate_prices)
from finsight.tools.quant import fundamental_ratios

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


def series(values, start="2022-01-03"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- store
def test_store_persists_runs_memos_flags_and_survives_restart(tmp_path):
    db = tmp_path / "f.db"
    s = Store(db)
    assert s.watchlist() == ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    s.remove_from_watchlist("TCS.NS")
    s.add_to_watchlist("wipro.ns")
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    rid = s.save_run(state, llm="mock", data="sample")
    s2 = Store(db)  # "restart"
    assert "TCS.NS" not in s2.watchlist() and "WIPRO.NS" in s2.watchlist()
    run = s2.run(rid)
    assert run["approved"] == 1 and run["plan"] == state.plan and "S1" in run["evidence"]
    memo = s2.memos("DEMO")[0]
    assert memo["version"] == 1 and memo["verified"] == 1 and memo["verification"]["claims"]
    flags = s2.flags(ticker="DEMO")
    assert len(flags) == len(state.risk_flags) and all(f["evidence_id"] for f in flags)
    s2.set_flag_status(flags[0]["id"], "reviewed")
    assert s2.flags(ticker="DEMO", status="open") and len(s2.flags(ticker="DEMO", status="reviewed")) == 1
    with pytest.raises(ValueError):
        s2.set_flag_status(flags[0]["id"], "deleted")
    s2.save_run(run_analysis("DEMO", settings=OFFLINE, engine_name="builtin"))
    assert s2.memos("DEMO")[0]["version"] == 2
    assert len(s2.flags(ticker="DEMO")) == len(state.risk_flags)  # only the latest run's flags are "current"
    k = s2.kpis()
    assert k["runs_total"] == 2 and k["companies_tracked"] == 5 and k["open_flags"] >= 1


def test_store_rejects_bad_tickers_and_records_failures(tmp_path):
    s = Store(tmp_path / "f.db")
    with pytest.raises(ValueError):
        s.add_to_watchlist("../x")
    rid = s.save_failed_run("DEMO", "q", "LLMError: down")
    assert s.run(rid)["status"] == "failed" and s.latest_run("DEMO") is None


# --------------------------------------------------------------------------- backtest engine
def test_backtest_cash_accounting_and_costs():
    p = series(np.linspace(100, 200, 300))
    always = series(np.ones(300))
    free = run_backtest(p, always, BacktestConfig(capital=10_000, cost_bps=0, slippage_bps=0))
    # bought at day 1's close (signal from day 0), held to the end: equity tracks price exactly
    assert free.equity.iloc[-1] == pytest.approx(10_000 * 200 / p.iloc[1], rel=1e-9)
    costly = run_backtest(p, always, BacktestConfig(capital=10_000, cost_bps=50, slippage_bps=50))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1] and costly.metrics["costs_paid"] > 0
    assert costly.trades[-1].open and any("still open" in w for w in costly.warnings)


def test_backtest_trades_and_win_rate():
    p = series([100, 100, 110, 120, 120, 90, 80, 80, 100, 100] * 10)
    sig = series([1, 1, 1, 0, 0, 1, 1, 0, 0, 0] * 10)
    r = run_backtest(p, sig, BacktestConfig(cost_bps=0, slippage_bps=0))
    closed = [t for t in r.trades if not t.open]
    assert r.metrics["trades"] == len(closed) == 20
    assert r.metrics["win_rate_pct"] == pytest.approx(100 * sum(t.return_pct > 0 for t in closed) / 20)


def test_no_look_ahead_in_execution():
    rng = np.random.default_rng(1)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    a, b = series(base), series(np.r_[base[:-1], base[-1] * 2])
    ra = run_strategy(a, "momentum", {"lookback": 20}, BacktestConfig(cost_bps=0, slippage_bps=0))
    rb = run_strategy(b, "momentum", {"lookback": 20}, BacktestConfig(cost_bps=0, slippage_bps=0))
    pd.testing.assert_series_equal(ra.equity.iloc[:-1], rb.equity.iloc[:-1])  # the future can't change the past


def test_backtest_validation_and_benchmark():
    with pytest.raises(ValueError):
        validate_prices(series([1.0] * 10))
    with pytest.raises(ValueError):
        run_strategy(SampleProvider().prices("DEMO"), "sma_crossover", {"fast": 100, "slow": 50})
    with pytest.raises(ValueError, match="point-in-time"):
        run_strategy(SampleProvider().prices("DEMO"), "value_screen")
    assert "value_screen" in UNAVAILABLE_STRATEGIES
    p = SampleProvider().prices("DEMO", "5y")
    r = run_strategy(p, "sma_crossover", benchmark=SampleProvider().prices("^NSEI", "5y"))
    assert r.benchmark is not None and r.benchmark_metrics["cumulative_return_pct"] is not None
    assert r.drawdown.max() <= 0 and r.metrics["max_drawdown_pct"] <= 0
    gappy = p.copy()
    gappy.iloc[10:13] = np.nan
    assert "forward-filled" in validate_prices(gappy)[1][0]


# --------------------------------------------------------------------------- ratios
def test_ratios_never_turn_missing_data_into_zero():
    r = fundamental_ratios({"revenue": 100.0, "net_income": None, "price": 10.0, "eps": -1.0})
    assert r["net_margin_pct"] is None and "net income" in r["unavailable"]["net_margin_pct"]
    assert r["pe_ratio"] is None and "negative" in r["unavailable"]["pe_ratio"]


def test_cross_currency_ratios_are_withheld():
    f = SampleProvider().fundamentals("DEMO") | {"statement_currency": "USD", "price_currency": "INR"}
    r = fundamental_ratios(f)
    assert r["pb_ratio"] is None and r["ev_to_ebitda"] is None
    assert "USD" in r["unavailable"]["ev_to_ebitda"]
    same = fundamental_ratios(SampleProvider().fundamentals("DEMO"))
    assert same["pb_ratio"] is not None and same["roce_pct"] is not None


def test_pdf_contains_verification_report():
    pytest.importorskip("reportlab")
    from finsight.export import to_pdf

    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin")
    pdf = to_pdf(state)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 3000
