"""Backtesting engine with explicit cash and position accounting.

Execution model (documented, deliberately simple):
* Signals are computed from data up to and including day t's close.
* A change in the target position is executed at day t+1's close (no look-ahead).
* Buys fill at close x (1 + slippage); sells fill at close x (1 - slippage).
* Commission is charged on traded notional (cost_bps) and paid from cash.
* Long-only, all-in when long, fractional shares allowed, no leverage, no shorting.
* Annualisation uses 252 trading days. Sharpe = (mean daily return x 252 - risk-free) / annualised vol.

Strategies are the predefined ones in ``finsight.tools.strategies``. User-supplied code is never
executed: there is no sandbox pretending to be safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finsight.tools.strategies import bollinger_reversion, momentum, rsi_reversion, sma_crossover

TRADING_DAYS = 252

STRATEGY_CATALOG: dict[str, dict[str, Any]] = {
    "sma_crossover": {"label": "Moving-average crossover", "fn": sma_crossover,
                      "params": {"fast": (50, 5, 200), "slow": (200, 20, 400)},
                      "rule": "Long when the fast simple moving average is above the slow one; flat otherwise."},
    "momentum": {"label": "Time-series momentum", "fn": momentum,
                 "params": {"lookback": (126, 20, 252)},
                 "rule": "Long when the trailing lookback-day return is positive; flat otherwise."},
    "rsi_reversion": {"label": "RSI mean reversion", "fn": rsi_reversion,
                      "params": {"low": (30, 5, 45), "exit_at": (55, 40, 80)},
                      "rule": "Buy when 14-day RSI drops below `low`; sell when it rises above `exit_at`."},
    "bollinger_reversion": {"label": "Bollinger mean reversion", "fn": bollinger_reversion,
                            "params": {"window": (20, 5, 60), "k": (2.0, 1.0, 3.0)},
                            "rule": "Buy below the lower band (mean - k x std); sell back at the moving average."},
}
UNAVAILABLE_STRATEGIES = {
    "value_screen": "Value screening needs point-in-time historical fundamentals (what was known on each date). "
                    "The configured providers only give recent statements, so a value backtest would use future "
                    "information. It is disabled rather than faked.",
}


@dataclass
class BacktestConfig:
    capital: float = 1_000_000.0
    cost_bps: float = 10.0  # commission per trade, basis points of notional
    slippage_bps: float = 5.0  # execution price penalty, basis points
    risk_free: float = 0.0  # annual, for Sharpe

    def validate(self) -> None:
        if self.capital <= 0:
            raise ValueError("starting capital must be positive")
        if not (0 <= self.cost_bps <= 500 and 0 <= self.slippage_bps <= 500):
            raise ValueError("costs and slippage must be between 0 and 500 bps")


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str | None
    exit_price: float | None
    shares: float
    return_pct: float | None  # net of costs, None while open
    open: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series | None
    drawdown: pd.Series
    trades: list[Trade]
    metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)


def validate_prices(prices: pd.Series, min_obs: int = 60) -> tuple[pd.Series, list[str]]:
    warnings = []
    p = pd.to_numeric(prices, errors="coerce")
    n_missing = int(p.isna().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing price(s) forward-filled")
    p = p.ffill().dropna()
    if (p <= 0).any():
        raise ValueError("prices must be positive")
    if not p.index.is_monotonic_increasing:
        p = p.sort_index()
    if p.index.has_duplicates:
        p = p[~p.index.duplicated(keep="last")]
        warnings.append("duplicate dates removed")
    if len(p) < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {len(p)}")
    return p, warnings


def _stats(equity: pd.Series, risk_free: float) -> dict[str, Any]:
    rets = equity.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0] - 1
    vol = float(rets.std() * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = (float(rets.mean()) * TRADING_DAYS - risk_free) / vol if vol > 0 else None
    dd = equity / equity.cummax() - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else None
    return {
        "cumulative_return_pct": round(total * 100, 2),
        "annualised_return_pct": round(cagr * 100, 2) if cagr is not None else None,
        "annualised_volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


def run_backtest(prices: pd.Series, target: pd.Series, config: BacktestConfig | None = None,
                 benchmark: pd.Series | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    cfg.validate()
    p, warnings = validate_prices(prices)
    tgt = target.reindex(p.index).fillna(0.0).clip(0, 1)
    slip, comm = cfg.slippage_bps / 1e4, cfg.cost_bps / 1e4

    cash, shares = cfg.capital, 0.0
    equity, trades, open_trade = [], [], None
    costs_paid = 0.0
    dates = p.index
    for i, d in enumerate(dates):
        price = float(p.iloc[i])
        desired = tgt.iloc[i - 1] if i > 0 else 0.0  # yesterday's signal, executed today
        if desired >= 0.5 and shares == 0:
            fill = price * (1 + slip)
            spend = cash / (1 + comm)
            shares = spend / fill
            fee = spend * comm
            costs_paid += fee + spend * slip / (1 + slip)
            cash -= spend + fee
            open_trade = {"entry_date": str(d.date()), "entry_price": fill, "shares": shares, "cost": spend + fee}
        elif desired < 0.5 and shares > 0:
            fill = price * (1 - slip)
            proceeds = shares * fill
            fee = proceeds * comm
            costs_paid += fee + shares * price * slip
            cash += proceeds - fee
            trades.append(Trade(open_trade["entry_date"], round(open_trade["entry_price"], 4), str(d.date()),
                                round(fill, 4), round(shares, 6),
                                round(((proceeds - fee) / open_trade["cost"] - 1) * 100, 2)))
            shares, open_trade = 0.0, None
        equity.append(cash + shares * price)
    if open_trade:
        mark = float(p.iloc[-1])
        trades.append(Trade(open_trade["entry_date"], round(open_trade["entry_price"], 4), None, None,
                            round(shares, 6), round((shares * mark / open_trade["cost"] - 1) * 100, 2), open=True))
        warnings.append("last position still open; its return is marked to the final close, before exit costs")

    eq = pd.Series(equity, index=dates, name="strategy")
    m = _stats(eq, cfg.risk_free)
    closed = [t for t in trades if not t.open]
    wins = [t for t in closed if (t.return_pct or 0) > 0]
    m.update({
        "trades": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if closed else None,
        "avg_trade_return_pct": round(float(np.mean([t.return_pct for t in closed])), 2) if closed else None,
        "best_trade_pct": max((t.return_pct for t in closed), default=None),
        "worst_trade_pct": min((t.return_pct for t in closed), default=None),
        "time_in_market_pct": round(float(tgt.shift(1).fillna(0).ge(0.5).mean()) * 100, 2),
        "costs_paid": round(costs_paid, 2),
        "starting_capital": cfg.capital,
        "cost_bps": cfg.cost_bps,
        "slippage_bps": cfg.slippage_bps,
        "risk_free_pct": round(cfg.risk_free * 100, 2),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
    })
    if not closed:
        warnings.append("no completed trades: win rate and average trade return are undefined")

    bench_eq, bench_m = None, None
    if benchmark is not None:
        b = pd.to_numeric(benchmark, errors="coerce").reindex(dates).ffill().dropna()
        if len(b) >= 2:
            bench_eq = (b / b.iloc[0] * cfg.capital).rename("benchmark")
            bench_m = _stats(bench_eq, cfg.risk_free)
        else:
            warnings.append("benchmark has no overlapping dates; comparison skipped")
    dd = (eq / eq.cummax() - 1).rename("drawdown")
    return BacktestResult(eq, bench_eq, dd, trades, m, bench_m, warnings)


def run_strategy(prices: pd.Series, strategy: str, params: dict[str, Any] | None = None,
                 config: BacktestConfig | None = None, benchmark: pd.Series | None = None) -> BacktestResult:
    if strategy not in STRATEGY_CATALOG:
        raise ValueError(UNAVAILABLE_STRATEGIES.get(strategy, f"unknown strategy {strategy!r}"))
    spec = STRATEGY_CATALOG[strategy]
    clean: dict[str, Any] = {}
    for name, (default, lo, hi) in spec["params"].items():
        v = (params or {}).get(name, default)
        v = type(default)(v)
        if not lo <= v <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
        clean[name] = v
    if strategy == "sma_crossover" and clean["fast"] >= clean["slow"]:
        raise ValueError("fast window must be shorter than slow window")
    p, _ = validate_prices(prices)
    return run_backtest(p, spec["fn"](p, **clean), config, benchmark)
