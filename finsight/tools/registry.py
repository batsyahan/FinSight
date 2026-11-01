"""Tool registry. Each tool is a plain Python function with a JSON-ish parameter spec.

Calling a tool through the registry automatically records its output in the evidence
store, so anything an agent learns from a tool gets a citable id (S1, S2, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from finsight.data.providers import MarketDataProvider
from finsight.state import AnalysisState
from finsight.tools.peers import compare_peers, find_peers
from finsight.tools.quant import fundamental_ratios, price_stats, sma_crossover_backtest
from finsight.tools.sentiment import news_sentiment
from finsight.tools.strategies import strategy_lab


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., dict[str, Any] | list[Any]]
    params: dict[str, str] = field(default_factory=dict)  # name -> description
    title: str = ""

    def schema(self) -> dict[str, Any]:
        """JSON-schema description used for native (API-level) tool calling."""
        props = {
            k: {"type": "integer" if v.lower().startswith("int") else "string", "description": v}
            for k, v in self.params.items()
        }
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": []}}

    def signature(self) -> str:
        ps = ", ".join(f"{k}: {v}" for k, v in self.params.items()) or "no arguments"
        return f"- {self.name}({ps}): {self.description}"


class ToolRegistry:
    def __init__(self, state: AnalysisState) -> None:
        self.state = state
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def subset(self, names: list[str]) -> "ToolRegistry":
        sub = ToolRegistry(self.state)
        for n in names:
            sub.register(self.tools[n])
        return sub

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def describe(self) -> str:
        return "\n".join(t.signature() for t in self.tools.values())

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a tool. Returns {"evidence_id", "data"} on success or {"error"} on failure."""
        if name not in self.tools:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(self.tools)}"}
        tool = self.tools[name]
        args = {k: v for k, v in (args or {}).items() if k in tool.params}
        try:
            out = tool.fn(**args)
        except Exception as e:  # tools must never crash the agent loop
            return {"error": f"{type(e).__name__}: {e}"}
        data = out if isinstance(out, dict) else {"items": out}
        ev = self.state.evidence.add(source=name, title=tool.title or name, data=data)
        return {"evidence_id": ev.id, "data": data}


def build_tools(provider: MarketDataProvider, state: AnalysisState, filings: Any = None) -> ToolRegistry:
    t = state.ticker
    reg = ToolRegistry(state)
    cache: dict[str, Any] = {}

    def prices(period: str = "2y"):
        key = f"prices:{period}"
        if key not in cache:
            cache[key] = provider.prices(t, period)
        return cache[key]

    def fundamentals():
        if "fund" not in cache:
            cache["fund"] = provider.fundamentals(t)
        return cache["fund"]

    reg.register(Tool(
        "get_company_profile", "Company name, sector, industry, market cap and business summary.",
        lambda: provider.profile(t), title=f"{t} company profile"))
    reg.register(Tool(
        "get_fundamentals", "Latest annual financial statement figures (revenue, profit, debt, cash flow).",
        fundamentals, title=f"{t} financial statements"))
    reg.register(Tool(
        "get_news", "Recent news headlines about the company.",
        lambda limit=6: provider.news(t, int(limit)), params={"limit": "int, max headlines (default 6)"},
        title=f"{t} recent news"))
    reg.register(Tool(
        "get_price_stats", "Returns, CAGR, volatility, Sharpe ratio and max drawdown from price history.",
        lambda period="2y": price_stats(prices(period)),
        params={"period": "one of 6mo, 1y, 2y, 3y, 5y (default 2y)"}, title=f"{t} price statistics"))
    reg.register(Tool(
        "run_backtest", "Backtest an SMA crossover strategy versus buy-and-hold.",
        lambda fast=50, slow=200, period="2y": sma_crossover_backtest(prices(period), int(fast), int(slow)),
        params={"fast": "int (default 50)", "slow": "int (default 200)", "period": "price period (default 2y)"},
        title=f"{t} SMA crossover backtest"))
    reg.register(Tool(
        "compute_ratios", "Growth, margins, ROE, debt/equity, current ratio and P/E from fundamentals.",
        lambda: fundamental_ratios(fundamentals()), title=f"{t} fundamental ratios"))

    reg.register(Tool(
        "score_news_sentiment", "Score recent headlines positive/negative/neutral with a finance word list; "
        "returns net sentiment and its trend.",
        lambda limit=10: news_sentiment(provider.news(t, int(limit))),
        params={"limit": "int, headlines to score (default 10)"}, title=f"{t} news sentiment"))
    reg.register(Tool(
        "run_strategy_lab", "Backtest SMA crossover, momentum, RSI and Bollinger strategies net of costs, with "
        "walk-forward (out-of-sample) validation, versus buy-and-hold.",
        lambda cost_bps=10, period="5y": strategy_lab(prices(period), float(cost_bps)),
        params={"cost_bps": "int, round-trip cost per trade in basis points (default 10)",
                "period": "price history (default 5y)"},
        title=f"{t} strategy lab"))

    if filings is not None:
        state.filings = filings.docs
        reg.register(Tool(
            "search_filings", "Search the company's filings (annual reports, earnings-call transcripts) and return "
            "the most relevant excerpts with document and page number.",
            lambda query, k=4: {"query": str(query), "results": filings.search(str(query), int(k))},
            params={"query": "what to look for, in plain words", "k": "int, excerpts to return (default 4)"},
            title="Filings search"))

    def _peer_list(tickers: str = "") -> list[str]:
        chosen = [x.strip() for x in str(tickers).split(",") if x.strip()]
        return chosen or state.peers or find_peers(t, provider)["peers"]

    reg.register(Tool(
        "find_peers", "Find the company's sector peer group.", lambda: find_peers(t, provider),
        title=f"{t} sector peer group"))
    reg.register(Tool(
        "compare_peers", "Compare 1-year return, risk, growth, margins, leverage and P/E against peers; "
        "ranks the company on each metric.",
        lambda tickers="": compare_peers(t, _peer_list(tickers), provider),
        params={"tickers": "comma-separated peer tickers (optional; defaults to the sector peers)"},
        title=f"{t} vs peers"))
    return reg


def to_json(obj: Any, limit: int = 2500) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"
