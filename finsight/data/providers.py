"""Market data providers.

``SampleProvider`` generates deterministic synthetic data so the whole system runs offline
(demos, tests, CI). ``YFinanceProvider`` pulls real data from Yahoo Finance via yfinance.
Both return the same shapes, so agents never care which one is in use.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from datetime import date, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd


class MarketDataProvider(Protocol):
    name: str

    def profile(self, ticker: str) -> dict[str, Any]: ...
    def prices(self, ticker: str, period: str = "2y") -> pd.Series: ...
    def fundamentals(self, ticker: str) -> dict[str, Any]: ...
    def news(self, ticker: str, limit: int = 8) -> list[dict[str, Any]]: ...


_PERIOD_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}


# --------------------------------------------------------------------------- sample


class SampleProvider:
    """Deterministic synthetic market data. Every value is fake and labelled as such."""

    name = "sample"

    _SECTORS = [
        ("Technology", "Software—Application"),
        ("Financial Services", "Banks—Regional"),
        ("Consumer Defensive", "Beverages—Non-Alcoholic"),
        ("Healthcare", "Drug Manufacturers"),
        ("Industrials", "Specialty Industrial Machinery"),
    ]

    def _rng(self, ticker: str, salt: str = "") -> np.random.Generator:
        seed = int(hashlib.sha256((ticker.upper() + salt).encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed)

    def profile(self, ticker: str) -> dict[str, Any]:
        rng = self._rng(ticker, "profile")
        sector, industry = self._SECTORS[int(rng.integers(len(self._SECTORS)))]
        mcap = float(rng.uniform(5e9, 4e11))
        return {
            "ticker": ticker.upper(),
            "name": f"{ticker.upper()} Sample Corp (synthetic)",
            "sector": sector,
            "industry": industry,
            "country": "India",
            "price_currency": "INR",
            "market_cap": round(mcap, 0),
            "summary": (
                f"{ticker.upper()} Sample Corp is a SYNTHETIC company generated for offline demos. "
                f"It operates in {industry.lower()} within the {sector.lower()} sector."
            ),
            "data_note": "SYNTHETIC SAMPLE DATA, not real market data",
        }

    def prices(self, ticker: str, period: str = "2y") -> pd.Series:
        # One 5-year path per ticker; shorter periods are slices of it, so 1y and 2y agree.
        full = _PERIOD_DAYS["5y"]
        n = _PERIOD_DAYS.get(period, 504)
        rng = self._rng(ticker, "prices")
        mu = rng.uniform(-0.05, 0.25) / 252  # daily drift
        sigma = rng.uniform(0.15, 0.45) / math.sqrt(252)
        shocks = rng.normal(mu - 0.5 * sigma**2, sigma, full)
        start = float(rng.uniform(100, 3000))
        values = start * np.exp(np.cumsum(shocks))
        idx = pd.bdate_range(end=date(2026, 10, 30), periods=full)
        return pd.Series(np.round(values, 2), index=idx, name="close").iloc[-n:]

    def fundamentals(self, ticker: str) -> dict[str, Any]:
        rng = self._rng(ticker, "fund")
        revenue = float(rng.uniform(2e10, 5e11))
        growth = float(rng.uniform(-0.08, 0.3))
        prev_revenue = revenue / (1 + growth)
        margin = float(rng.uniform(-0.03, 0.25))
        net_income = revenue * margin
        equity = revenue * float(rng.uniform(0.3, 1.2))
        debt = equity * float(rng.uniform(0.05, 2.4))
        ocf = net_income * float(rng.uniform(0.6, 1.5)) + revenue * 0.02
        capex = revenue * float(rng.uniform(0.02, 0.12))
        cur_assets = revenue * float(rng.uniform(0.2, 0.6))
        cur_liab = cur_assets / float(rng.uniform(0.7, 2.5))
        shares = float(rng.uniform(2e8, 5e9))
        price = float(self.prices(ticker).iloc[-1])
        eps = net_income / shares
        return {
            "fiscal_year": 2026,
            "statement_currency": "INR",
            "price_currency": "INR",
            "revenue": round(revenue, 0),
            "revenue_prev": round(prev_revenue, 0),
            "net_income": round(net_income, 0),
            "total_debt": round(debt, 0),
            "total_equity": round(equity, 0),
            "operating_cash_flow": round(ocf, 0),
            "capex": round(capex, 0),
            "free_cash_flow": round(ocf - capex, 0),
            "current_assets": round(cur_assets, 0),
            "current_liabilities": round(cur_liab, 0),
            "shares_outstanding": round(shares, 0),
            "eps": round(eps, 2),
            "price": round(price, 2),
            "data_note": "SYNTHETIC SAMPLE DATA, not real market data",
        }

    def news(self, ticker: str, limit: int = 8) -> list[dict[str, Any]]:
        rng = self._rng(ticker, "news")
        templates = [
            "{t} reports quarterly results; revenue {dir} year on year",
            "{t} announces expansion into new regional markets",
            "Analysts revise {t} price targets after earnings call",
            "{t} board approves dividend and share buyback plan",
            "Regulator issues notice to {t} over disclosure practices",
            "{t} signs multi-year supply agreement with major partner",
            "{t} CFO steps down; interim replacement named",
            "{t} invests in AI-driven automation across operations",
        ]
        picks = rng.permutation(len(templates))[:limit]
        today = date(2026, 10, 30)
        out = []
        for i, k in enumerate(picks):
            out.append(
                {
                    "title": "[SAMPLE] "
                    + templates[int(k)].format(t=ticker.upper(), dir="up" if rng.random() > 0.4 else "down"),
                    "publisher": "Synthetic Newswire",
                    "date": (today - timedelta(days=int(3 * i + rng.integers(0, 3)))).isoformat(),
                    "link": "",
                }
            )
        return out


# --------------------------------------------------------------------------- yfinance


class YFinanceProvider:
    """Real data via yfinance. Install with ``pip install finsight[market]``.

    Indian stocks use Yahoo suffixes: ``.NS`` (NSE) or ``.BO`` (BSE), e.g. ``INFY.NS``.
    """

    name = "yfinance"

    def __init__(self) -> None:
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("yfinance is not installed: pip install 'finsight[market]'") from e
        self._yf = yf
        self._cache: dict[str, Any] = {}
        # yfinance triggers pandas "inplace keyword" deprecation warnings; they are not ours to fix
        warnings.filterwarnings("ignore", message=r".*inplace keyword.*")

    def _t(self, ticker: str):
        if ticker not in self._cache:
            self._cache[ticker] = self._yf.Ticker(ticker)
        return self._cache[ticker]

    def profile(self, ticker: str) -> dict[str, Any]:
        info = self._t(ticker).info or {}
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "price_currency": info.get("currency"),
            "market_cap": info.get("marketCap"),
            "summary": (info.get("longBusinessSummary") or "")[:1200],
        }

    def prices(self, ticker: str, period: str = "2y") -> pd.Series:
        hist = self._t(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            raise ValueError(f"No price history for {ticker}")
        s = hist["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.rename("close")

    @staticmethod
    def _row(df: pd.DataFrame | None, names: list[str], col: int = 0) -> float | None:
        if df is None or df.empty or df.shape[1] <= col:
            return None
        for n in names:
            if n in df.index:
                v = df.loc[n].iloc[col]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    return float(v)
        return None

    def fundamentals(self, ticker: str) -> dict[str, Any]:
        t = self._t(ticker)
        inc, bal, cf = t.financials, t.balance_sheet, t.cashflow
        info = t.info or {}
        r = self._row
        ocf = r(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = r(cf, ["Capital Expenditure", "Capital Expenditures"])
        fcf = r(cf, ["Free Cash Flow"])
        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is reported as a negative number
        return {
            "fiscal_year": str(inc.columns[0].year) if inc is not None and not inc.empty else None,
            # Statements can be reported in a different currency from the share price
            # (e.g. an NSE listing quoted in INR whose statements Yahoo reports in USD).
            "statement_currency": info.get("financialCurrency") or info.get("currency"),
            "price_currency": info.get("currency"),
            "currency_note": "revenue, income, debt, equity and cash-flow figures are in statement_currency; "
                             "price and eps are in price_currency",
            "revenue": r(inc, ["Total Revenue", "Operating Revenue"]),
            "revenue_prev": r(inc, ["Total Revenue", "Operating Revenue"], col=1),
            "net_income": r(inc, ["Net Income", "Net Income Common Stockholders"]),
            "total_debt": r(bal, ["Total Debt"]),
            "total_equity": r(bal, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
            "operating_cash_flow": ocf,
            "capex": abs(capex) if capex is not None else None,
            "free_cash_flow": fcf,
            "current_assets": r(bal, ["Current Assets", "Total Current Assets"]),
            "current_liabilities": r(bal, ["Current Liabilities", "Total Current Liabilities"]),
            "shares_outstanding": info.get("sharesOutstanding"),
            "eps": info.get("trailingEps"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }

    def news(self, ticker: str, limit: int = 8) -> list[dict[str, Any]]:
        items = self._t(ticker).news or []
        out = []
        for it in items[:limit]:
            c = it.get("content", it)  # newer yfinance nests fields under "content"
            provider = c.get("provider") or {}
            url = c.get("canonicalUrl") or {}
            out.append(
                {
                    "title": c.get("title", ""),
                    "publisher": provider.get("displayName") if isinstance(provider, dict) else c.get("publisher"),
                    "date": c.get("pubDate") or c.get("providerPublishTime"),
                    "link": url.get("url") if isinstance(url, dict) else c.get("link", ""),
                }
            )
        return out


def get_provider(name: str = "sample") -> MarketDataProvider:
    name = (name or "sample").lower()
    if name == "yfinance":
        return YFinanceProvider()
    if name == "sample":
        return SampleProvider()
    raise ValueError(f"Unknown data provider: {name}")
