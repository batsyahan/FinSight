"""Plotly charts styled to the finsight design tokens (lime line on dark glass, faint grid)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from finsight.ui.theme import TOKENS

LIME, AMBER, CORAL, MUTED, SLATE = TOKENS["lime"], TOKENS["amber"], TOKENS["coral"], TOKENS["muted"], "#9AA5B1"
CONFIG = {"displayModeBar": False, "responsive": True}


def _base(fig: go.Figure, height: int, y_title: str = "", y_fmt: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=11), showlegend=False,
        hovermode="x unified", hoverlabel=dict(bgcolor="#1E2120", bordercolor="rgba(255,255,255,0.12)",
                                               font=dict(color=TOKENS["text"], size=12)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(255,255,255,0.08)", ticks="", tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False, ticks="", tickfont=dict(size=11),
                     title=dict(text=y_title, font=dict(size=11)), tickformat=y_fmt)
    return fig


def area(series: pd.Series, name: str = "Close", height: int = 250, prefix: str = "") -> go.Figure:
    s = series.dropna()
    lo, hi = float(s.min()), float(s.max())
    pad = (hi - lo) * 0.08 or hi * 0.02
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=name, line=dict(color=LIME, width=2.2),
        fill="tozeroy", fillcolor="rgba(197,246,90,0.10)",
        hovertemplate=f"%{{x|%d %b %Y}}<br>{prefix}%{{y:,.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[s.index[-1]], y=[s.values[-1]], mode="markers", hoverinfo="skip",
                             marker=dict(color=LIME, size=7, line=dict(color="#101211", width=2))))
    _base(fig, height)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def compare(strategy: pd.Series, benchmark: pd.Series | None, height: int = 300) -> go.Figure:
    """Both series rebased to 100 at the start."""
    fig = go.Figure()
    s = strategy / strategy.iloc[0] * 100
    if benchmark is not None and len(benchmark):
        b = benchmark / benchmark.iloc[0] * 100
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Benchmark", mode="lines",
                                 line=dict(color=SLATE, width=1.6),
                                 hovertemplate="Benchmark %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Strategy", mode="lines", line=dict(color=LIME, width=2.2),
                             fill="tozeroy", fillcolor="rgba(197,246,90,0.07)",
                             hovertemplate="Strategy %{y:.1f}<extra></extra>"))
    _base(fig, height, "Growth of 100")
    lo = min(s.min(), (benchmark / benchmark.iloc[0] * 100).min() if benchmark is not None and len(benchmark) else s.min())
    hi = max(s.max(), (benchmark / benchmark.iloc[0] * 100).max() if benchmark is not None and len(benchmark) else s.max())
    fig.update_yaxes(range=[lo * 0.95, hi * 1.05])
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def drawdown(dd: pd.Series, height: int = 200) -> go.Figure:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, mode="lines", line=dict(color=CORAL, width=1.5),
                               fill="tozeroy", fillcolor="rgba(238,107,97,0.28)",
                               hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}%<extra></extra>"))
    return _base(fig, height, "Drawdown %")


def donut(parts: list[tuple[str, float, str]], center_value: str, center_label: str, height: int = 190) -> go.Figure:
    labels = [p[0] for p in parts]
    values = [p[1] for p in parts]
    colors = [p[2] for p in parts]
    if not any(values):
        values, colors = [1], ["rgba(255,255,255,0.08)"]
        labels = ["none"]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.72, marker=dict(colors=colors, line=dict(width=0)),
                           sort=False, textinfo="none", hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.add_annotation(text=f"<b style='font-size:22px;color:{TOKENS['text']}'>{center_value}</b><br>"
                            f"<span style='font-size:11px'>{center_label}</span>", showarrow=False, font=dict(color=MUTED))
    return _base(fig, height)


def history_bars(rows: list[dict[str, Any]], height: int = 260) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color, label in (("revenue", "rgba(197,246,90,0.85)", "Revenue"), ("net_income", SLATE, "Net income"),
                              ("free_cash_flow", AMBER, "Free cash flow")):
        if col in df and df[col].notna().any():
            fig.add_trace(go.Bar(x=df["year"].astype(str), y=df[col] / 1e9, name=label, marker=dict(color=color),
                                 hovertemplate=f"{label} %{{y:,.2f}} bn<extra></extra>"))
    _base(fig, height, "Billions (statement currency)")
    fig.update_layout(barmode="group", showlegend=True, legend=dict(orientation="h", y=1.1, x=0),
                      hovermode="closest")
    return fig


def hbar(labels: list[str], values: list[float], height: int = 220, suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=LIME),
                           hovertemplate=f"%{{y}}: %{{x:.2f}}{suffix}<extra></extra>"))
    _base(fig, height)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig
