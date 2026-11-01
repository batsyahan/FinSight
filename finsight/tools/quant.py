"""Deterministic quant functions. Agents call these as tools; no LLM does the arithmetic.

Percent-style outputs end in ``_pct`` and are already multiplied by 100 and rounded,
so the numbers the Writer quotes ("32.41%") match the evidence exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _pct(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x) * 100, 2)


def _r(x: float | None, nd: int = 2) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def price_stats(prices: pd.Series, risk_free: float = 0.06) -> dict[str, Any]:
    """Return, risk and drawdown statistics for a close-price series.

    ``risk_free`` is annual (default 6%, roughly the Indian 10y G-sec yield range).
    """
    p = prices.dropna().astype(float)
    if len(p) < 30:
        raise ValueError("Need at least 30 price points")
    rets = p.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    total = p.iloc[-1] / p.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = rets.std() * math.sqrt(TRADING_DAYS)
    excess = rets.mean() * TRADING_DAYS - risk_free
    sharpe = excess / vol if vol > 0 else float("nan")
    one_year = p.iloc[-1] / p.iloc[-TRADING_DAYS] - 1 if len(p) > TRADING_DAYS else total
    running_max = p.cummax()
    dd = p / running_max - 1
    trough = dd.idxmin()
    return {
        "start_date": str(p.index[0].date()),
        "end_date": str(p.index[-1].date()),
        "last_price": _r(p.iloc[-1]),
        "high_52w": _r(p.iloc[-TRADING_DAYS:].max()),
        "low_52w": _r(p.iloc[-TRADING_DAYS:].min()),
        "return_1y_pct": _pct(one_year),
        "return_total_pct": _pct(total),
        "cagr_pct": _pct(cagr),
        "ann_volatility_pct": _pct(vol),
        "sharpe": _r(sharpe),
        "max_drawdown_pct": _pct(dd.min()),
        "max_drawdown_date": str(trough.date()),
        "risk_free_pct": _pct(risk_free),
    }


def sma_crossover_backtest(prices: pd.Series, fast: int = 50, slow: int = 200) -> dict[str, Any]:
    """Long when fast SMA > slow SMA, flat otherwise. Signal is lagged one day (no look-ahead)."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    p = prices.dropna().astype(float)
    if len(p) < slow + 20:
        raise ValueError(f"Need at least {slow + 20} prices for a {fast}/{slow} backtest")
    fast_ma = p.rolling(fast).mean()
    slow_ma = p.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int).shift(1).fillna(0)
    rets = p.pct_change().fillna(0)
    window = slice(slow, None)  # only evaluate once both averages exist
    strat = (1 + rets[window] * signal[window]).prod() - 1
    hold = p.iloc[-1] / p.iloc[slow] - 1
    trades = int(signal[window].diff().abs().sum())
    return {
        "strategy": f"SMA {fast}/{slow} crossover",
        "fast_window": fast,
        "slow_window": slow,
        "evaluated_from": str(p.index[slow].date()),
        "strategy_return_pct": _pct(strat),
        "buy_hold_return_pct": _pct(hold),
        "outperformance_pct": _pct(strat - hold),
        "trades": trades,
        "time_in_market_pct": _pct(signal[window].mean()),
        "currently_long": bool(fast_ma.iloc[-1] > slow_ma.iloc[-1]),
    }


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def fundamental_ratios(f: dict[str, Any]) -> dict[str, Any]:
    rev, prev = f.get("revenue"), f.get("revenue_prev")
    ni, eq, debt = f.get("net_income"), f.get("total_equity"), f.get("total_debt")
    fcf = f.get("free_cash_flow")
    price, eps = f.get("price"), f.get("eps")
    growth = _div(rev, prev)
    pe = _div(price, eps) if eps and eps > 0 else None
    return {
        "revenue_growth_pct": _pct(growth - 1) if growth is not None else None,
        "net_margin_pct": _pct(_div(ni, rev)),
        "fcf_margin_pct": _pct(_div(fcf, rev)),
        "roe_pct": _pct(_div(ni, eq)) if eq and eq > 0 else None,
        "debt_to_equity": _r(_div(debt, eq)) if eq and eq > 0 else None,
        "current_ratio": _r(_div(f.get("current_assets"), f.get("current_liabilities"))),
        "pe_ratio": _r(pe),
        "eps": _r(eps),
    }


def max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna().astype(float)
    return float((p / p.cummax() - 1).min())


__all__ = ["price_stats", "sma_crossover_backtest", "fundamental_ratios", "max_drawdown"]
