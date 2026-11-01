"""Rule-based risk flags. Transparent thresholds the Risk agent starts from before adding judgement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class RiskRule:
    id: str
    metric: str
    check: Callable[[float], bool]
    threshold: str
    limit: float  # numeric threshold, recorded in evidence so the memo can cite it
    severity: str  # low | medium | high
    message: str


RULES: list[RiskRule] = [
    RiskRule("high_leverage", "debt_to_equity", lambda v: v > 1.5, "> 1.5", 1.5, "high",
             "Debt is more than 1.5x equity; balance sheet is highly leveraged."),
    RiskRule("elevated_leverage", "debt_to_equity", lambda v: 1.0 < v <= 1.5, "1.0 to 1.5", 1.0, "medium",
             "Debt exceeds equity; leverage is elevated."),
    RiskRule("negative_fcf", "fcf_margin_pct", lambda v: v < 0, "< 0%", 0, "high",
             "Free cash flow is negative; the business is consuming cash."),
    RiskRule("loss_making", "net_margin_pct", lambda v: v < 0, "< 0%", 0, "high",
             "The company is loss-making at the net level."),
    RiskRule("revenue_decline", "revenue_growth_pct", lambda v: v < 0, "< 0%", 0, "medium",
             "Revenue shrank versus the previous year."),
    RiskRule("weak_liquidity", "current_ratio", lambda v: v < 1.0, "< 1.0", 1.0, "medium",
             "Current liabilities exceed current assets; short-term liquidity is tight."),
    RiskRule("high_volatility", "ann_volatility_pct", lambda v: v > 40, "> 40%", 40, "medium",
             "Annualised volatility is above 40%; expect large price swings."),
    RiskRule("deep_drawdown", "max_drawdown_pct", lambda v: v < -35, "< -35%", -35, "medium",
             "The stock fell more than 35% from a peak within the period."),
    RiskRule("rich_valuation", "pe_ratio", lambda v: v > 60, "> 60", 60, "low",
             "P/E above 60 prices in a lot of future growth."),
    RiskRule("negative_news_sentiment", "net_sentiment", lambda v: v <= -0.3, "<= -0.3", -0.3, "medium",
             "Recent news flow is clearly negative."),
    RiskRule("poor_risk_adjusted_return", "sharpe", lambda v: v < 0, "< 0", 0, "low",
             "Returns did not beat the risk-free rate after adjusting for risk."),
]


def evaluate_risks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    flags = []
    for rule in RULES:
        v = metrics.get(rule.metric)
        if v is None:
            continue
        try:
            hit = rule.check(float(v))
        except (TypeError, ValueError):
            continue
        if hit:
            d = asdict(rule)
            d.pop("check")
            d["value"] = v
            flags.append(d)
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(flags, key=lambda f: order[f["severity"]])


def risk_score(flags: list[dict[str, Any]]) -> int:
    """0 (clean) to 100 (many severe flags)."""
    weights = {"high": 30, "medium": 15, "low": 5}
    return min(100, sum(weights[f["severity"]] for f in flags))
