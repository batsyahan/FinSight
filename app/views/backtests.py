"""Backtests: configure a predefined strategy, run it with cash accounting, costs and slippage."""

import streamlit as st

from finsight.guardrails import validate_ticker
from finsight.tools.backtest import STRATEGY_CATALOG, UNAVAILABLE_STRATEGIES, BacktestConfig, run_strategy
from finsight.ui import charts
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()

st.html(C.page_header("Backtests", "Test predefined strategies on historical data with explicit execution, "
                                   "cost and slippage assumptions"))

col_list, col_main, col_metrics = st.columns([0.9, 2, 1], gap="medium")
with col_list:
    with st.container(key="glass_strats"):
        st.html(C.panel_head("Strategies", "layers"))
        keys = list(STRATEGY_CATALOG)
        strategy = st.radio("Strategy", keys, format_func=lambda k: STRATEGY_CATALOG[k]["label"], key="bt_strategy",
                            label_visibility="collapsed")
        st.html(C.note(STRATEGY_CATALOG[strategy]["rule"]))
        for k, why in UNAVAILABLE_STRATEGIES.items():
            st.html(f'<div style="margin-top:12px">{C.pill("Unavailable", "neutral")} '
                    f'<span class="fs-small">{C.esc(k.replace("_", " ").title())}</span></div>' + C.note(why))
        st.html(C.note("Only these predefined strategies run. Finsight never executes user-supplied code, so "
                       "there is no sandbox pretending to be safe."))
    with st.container(key="glass_bt_cfg"):
        st.html(C.panel_head("Configuration", "list"))
        ticker = st.selectbox("Ticker", list(dict.fromkeys([st.session_state.get("bt_ticker") or "INFY.NS",
                                                            *store.watchlist(), *D.COMPANIES])),
                              key="bt_tk", accept_new_options=True)
        period = st.segmented_control("History", ["1y", "2y", "5y"], default="5y", key="bt_period") or "5y"
        bench_name = st.selectbox("Benchmark", list(D.BENCHMARKS), key="bt_bench")
        capital = st.number_input("Starting capital", 10_000.0, 1e10, 1_000_000.0, step=100_000.0, key="bt_cap")
        cost = st.number_input("Commission (bps per trade)", 0.0, 500.0, 10.0, step=1.0, key="bt_cost")
        slip = st.number_input("Slippage (bps)", 0.0, 500.0, 5.0, step=1.0, key="bt_slip")
        rf = st.number_input("Risk-free rate for Sharpe (% a year)", 0.0, 20.0, 0.0, step=0.5, key="bt_rf")
        params = {}
        for name, (default, lo, hi) in STRATEGY_CATALOG[strategy]["params"].items():
            if isinstance(default, float):
                params[name] = st.slider(name, float(lo), float(hi), float(default), 0.1, key=f"bt_p_{strategy}_{name}")
            else:
                params[name] = st.slider(name, int(lo), int(hi), int(default), 1, key=f"bt_p_{strategy}_{name}")
        go = st.button("Run backtest", type="primary", icon=":material/play_arrow:", key="bt_run", use_container_width=True)

if go:
    try:
        t = validate_ticker(str(ticker).split(" · ")[0])
        cfg = BacktestConfig(capital=capital, cost_bps=cost, slippage_bps=slip, risk_free=rf / 100)
        px = shell.prices(s.data, t, period)
        try:
            bench = shell.prices(s.data, D.BENCHMARKS[bench_name], period)
        except Exception:
            bench = None
        res = run_strategy(px, strategy, params, cfg, bench)
        store.save_backtest(t, strategy, params, {"period": period, "benchmark": bench_name, "capital": capital,
                                                  "cost_bps": cost, "slippage_bps": slip, "risk_free_pct": rf,
                                                  "data": s.data}, res.metrics)
        st.session_state["bt_result"] = {"ticker": t, "strategy": strategy, "bench": bench_name, "res": res,
                                         "data": s.data}
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Backtest failed: {type(e).__name__}: {e}")

out = st.session_state.get("bt_result")
with col_main:
    with st.container(key="glass_bt_chart"):
        if not out:
            st.html(C.panel_head("Results", "trend"))
            st.html(C.empty("Configure a strategy and press Run backtest."))
        else:
            r = out["res"]
            m = r.metrics
            st.html(C.panel_head(f"{STRATEGY_CATALOG[out['strategy']]['label']} · {out['ticker']}", "trend",
                                 right_html=f'<span class="fs-small">Strategy {C.pct_html(m["cumulative_return_pct"])} · '
                                            f'{C.esc(out["bench"])} '
                                            f'{C.pct_html((r.benchmark_metrics or {}).get("cumulative_return_pct"))}</span>'))
            st.plotly_chart(charts.compare(r.equity, r.benchmark), use_container_width=True, config=charts.CONFIG,
                            key="bt_cmp")
            if out["data"] == "sample":
                st.html(C.note("Synthetic sample prices (offline mode): results illustrate the engine, not a real stock."))
    if out:
        a, b = st.columns([1, 1.5], gap="medium")
        with a:
            with st.container(key="glass_trades"):
                m = out["res"].metrics
                closed = [t for t in out["res"].trades if not t.open]
                st.html(C.panel_head("Trade Summary", "list"))
                st.html(C.kv_rows([("Total trades", str(m["trades"])),
                                   ("Profitable trades", f"{sum(1 for t in closed if t.return_pct > 0)} "
                                                         f"({C.fmt_num(m['win_rate_pct'], 1, '%')})"),
                                   ("Avg trade return", C.fmt_num(m["avg_trade_return_pct"], 2, "%")),
                                   ("Best trade", C.fmt_num(m["best_trade_pct"], 2, "%")),
                                   ("Worst trade", C.fmt_num(m["worst_trade_pct"], 2, "%")),
                                   ("Time in market", C.fmt_num(m["time_in_market_pct"], 1, "%")),
                                   ("Costs paid", C.fmt_num(m["costs_paid"], 0))]))
        with b:
            with st.container(key="glass_dd"):
                st.html(C.panel_head("Equity Curve & Drawdown", "alert", "coral"))
                st.plotly_chart(charts.drawdown(out["res"].drawdown), use_container_width=True, config=charts.CONFIG,
                                key="bt_dd")
        with st.expander("Trade log"):
            st.html(C.table([("e", "Entry"), ("ep", "#Entry px"), ("x", "Exit"), ("xp", "#Exit px"), ("r", "#Return %")],
                            [{"e": t.entry_date, "ep": C.fmt_num(t.entry_price), "x": t.exit_date or "open",
                              "xp": C.fmt_num(t.exit_price), "r": C.fmt_num(t.return_pct)} for t in out["res"].trades])
                    if out["res"].trades else C.empty("No trades."))

with col_metrics:
    with st.container(key="glass_perf"):
        st.html(C.panel_head("Performance Metrics", "chart"))
        if out:
            m, bm = out["res"].metrics, out["res"].benchmark_metrics or {}
            st.html(C.table([("k", "Metric"), ("v", "#Strategy"), ("b", "#Benchmark")], [
                {"k": "Total return", "v": C.fmt_num(m["cumulative_return_pct"], 2, "%"),
                 "b": C.fmt_num(bm.get("cumulative_return_pct"), 2, "%")},
                {"k": "Annualised return", "v": C.fmt_num(m["annualised_return_pct"], 2, "%"),
                 "b": C.fmt_num(bm.get("annualised_return_pct"), 2, "%")},
                {"k": "Volatility (ann.)", "v": C.fmt_num(m["annualised_volatility_pct"], 2, "%"),
                 "b": C.fmt_num(bm.get("annualised_volatility_pct"), 2, "%")},
                {"k": "Sharpe ratio", "v": C.fmt_num(m["sharpe"]), "b": C.fmt_num(bm.get("sharpe"))},
                {"k": "Max drawdown", "v": C.fmt_num(m["max_drawdown_pct"], 2, "%"),
                 "b": C.fmt_num(bm.get("max_drawdown_pct"), 2, "%")},
                {"k": "Win rate", "v": C.fmt_num(m["win_rate_pct"], 1, "%"), "b": "n/a"},
            ]))
            for w in out["res"].warnings:
                st.html(C.note(w))
        else:
            st.html(C.empty("Metrics appear after a run."))
    with st.container(key="glass_assump"):
        st.html(C.panel_head("Assumptions", "shield", "slate"))
        st.html(C.bullets([
            "Signals use data up to each day's close; trades execute at the next close (no look-ahead).",
            "Buys fill at close plus slippage, sells at close minus slippage; commission on traded notional.",
            "Long-only, fully invested when long, fractional shares, no leverage or shorting.",
            "252 trading days a year; Sharpe uses the risk-free rate you set.",
            "Price return only: dividends are not reinvested.",
        ]))
        st.warning("Historical backtest performance does not guarantee future results.")

with st.expander("Recent backtests"):
    recent = store.backtests(15)
    st.html(C.table([("t", "Ticker"), ("s", "Strategy"), ("r", "#Return %"), ("sh", "#Sharpe"), ("w", "When")],
                    [{"t": b["ticker"], "s": b["strategy"].replace("_", " "),
                      "r": C.fmt_num(b["metrics"].get("cumulative_return_pct")),
                      "sh": C.fmt_num(b["metrics"].get("sharpe")), "w": b["created_at"][:16].replace("T", " ")}
                     for b in recent]) if recent else C.empty("None yet."))
