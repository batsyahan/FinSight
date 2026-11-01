"""Strategy lab: rule-based trading strategies, backtested net of costs, with walk-forward
validation to expose overfitting.

Every strategy produces a daily position (0 = flat, 1 = long) that is lagged one day before
it earns returns, so no strategy can trade on information it did not have yet. Costs
(commission + slippage, in basis points) are charged on every change of position.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- indicators
def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Stateful long/flat position from entry and exit signals."""
    pos, out = 0, []
    for e, x in zip(entry.to_numpy(), exit_.to_numpy()):
        if pos == 0 and e:
            pos = 1
        elif pos == 1 and x:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=entry.index, dtype=float)


# --------------------------------------------------------------------------- strategies
def sma_crossover(p: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    return (p.rolling(fast).mean() > p.rolling(slow).mean()).astype(float)


def momentum(p: pd.Series, lookback: int = 126) -> pd.Series:
    return (p.pct_change(lookback) > 0).astype(float)


def rsi_reversion(p: pd.Series, low: int = 30, exit_at: int = 55, window: int = 14) -> pd.Series:
    r = rsi(p, window)
    return _hold(r < low, r > exit_at)


def bollinger_reversion(p: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    mid = p.rolling(window).mean()
    lower = mid - k * p.rolling(window).std()
    return _hold(p < lower, p > mid)


STRATEGIES: dict[str, tuple[Callable[..., pd.Series], dict[str, Any], dict[str, list[Any]]]] = {
    # name: (function, default params, walk-forward grid)
    "sma_crossover": (sma_crossover, {"fast": 50, "slow": 200}, {"fast": [20, 50], "slow": [100, 200]}),
    "momentum": (momentum, {"lookback": 126}, {"lookback": [63, 126, 252]}),
    "rsi_reversion": (rsi_reversion, {"low": 30, "exit_at": 55}, {"low": [25, 30, 35], "exit_at": [50, 60]}),
    "bollinger_reversion": (bollinger_reversion, {"window": 20, "k": 2.0}, {"window": [20], "k": [1.5, 2.0, 2.5]}),
}


# --------------------------------------------------------------------------- backtest
def _metrics(daily: pd.Series) -> dict[str, float]:
    daily = daily.fillna(0.0)
    equity = (1 + daily).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    years = len(daily) / TRADING_DAYS
    vol = float(daily.std() * math.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    return {"total_return_pct": round(total * 100, 2), "cagr_pct": round(cagr * 100, 2),
            "sharpe": round(sharpe, 2), "max_drawdown_pct": round(dd * 100, 2)}


def backtest(prices: pd.Series, position: pd.Series, cost_bps: float = 10.0) -> dict[str, Any]:
    p = prices.dropna().astype(float)
    pos = position.reindex(p.index).fillna(0.0).shift(1).fillna(0.0)  # trade on yesterday's signal
    rets = p.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = turnover * cost_bps / 1e4
    strat = pos * rets - costs
    out = _metrics(strat)
    out.update({
        "trades": int(turnover.gt(0).sum()),
        "time_in_market_pct": round(float(pos.mean()) * 100, 2),
        "costs_paid_pct": round(float(costs.sum()) * 100, 2),
    })
    return out


def walk_forward(prices: pd.Series, name: str, train: int = 504, test: int = 126,
                 cost_bps: float = 10.0) -> dict[str, Any]:
    """Pick parameters on each training window (in-sample), then trade them on the following
    unseen window (out-of-sample). A big in-sample vs out-of-sample gap signals overfitting."""
    fn, _, grid = STRATEGIES[name]
    p = prices.dropna().astype(float)
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    oos_daily, is_sharpes, chosen = [], [], []
    start = 0
    while start + train + test <= len(p):
        tr = p.iloc[start:start + train]
        best, best_sh = None, -1e9
        for params in combos:
            if name == "sma_crossover" and params["fast"] >= params["slow"]:
                continue
            sh = backtest(tr, fn(tr, **params), cost_bps)["sharpe"]
            if sh > best_sh:
                best, best_sh = params, sh
        # signals for the test window are computed with history up to it (no look-ahead)
        hist = p.iloc[: start + train + test]
        pos = fn(hist, **best).shift(1).fillna(0.0).iloc[-test:]
        r = hist.pct_change().fillna(0.0).iloc[-test:]
        turnover = pos.diff().abs().fillna(pos.abs())
        oos_daily.append(pos * r - turnover * cost_bps / 1e4)
        is_sharpes.append(best_sh)
        chosen.append(best)
        start += test
    if not oos_daily:
        return {"strategy": name, "folds": 0, "note": f"need at least {train + test} days of prices"}
    oos = _metrics(pd.concat(oos_daily))
    is_sharpe = round(float(np.mean(is_sharpes)), 2)
    return {
        "strategy": name,
        "folds": len(chosen),
        "in_sample_sharpe": is_sharpe,
        "out_of_sample_sharpe": oos["sharpe"],
        "out_of_sample_return_pct": oos["total_return_pct"],
        "overfit_gap": round(is_sharpe - oos["sharpe"], 2),
        "params_chosen": chosen,
    }


def strategy_lab(prices: pd.Series, cost_bps: float = 10.0) -> dict[str, Any]:
    p = prices.dropna().astype(float)
    hold = _metrics(p.pct_change().fillna(0.0))
    results, wf = {}, {}
    for name, (fn, defaults, _) in STRATEGIES.items():
        results[name] = backtest(p, fn(p, **defaults), cost_bps) | {"params": defaults}
        wf[name] = walk_forward(p, name, cost_bps=cost_bps)
    ranked = sorted((n for n in wf if wf[n].get("folds")), key=lambda n: wf[n]["out_of_sample_sharpe"], reverse=True)
    return {
        "period_start": str(p.index[0].date()),
        "period_end": str(p.index[-1].date()),
        "cost_bps": cost_bps,
        "buy_and_hold": hold,
        "strategies": results,
        "walk_forward": wf,
        "best_out_of_sample": ranked[0] if ranked else None,
        "note": "positions lag signals by one day; costs charged on every position change; "
                "walk-forward picks parameters in-sample and reports unseen out-of-sample results",
    }
