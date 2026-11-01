"""Wires the agent team into a graph:

    researcher -> quant -> risk -> writer -> critic --(revise)--> writer
                                                    \\--(done)---> END
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable

from finsight.agents.team import Team
from finsight.config import Settings
from finsight.data.providers import MarketDataProvider, get_provider
from finsight.graph import engine
from finsight.llm.providers import get_llm
from finsight.obs import instrument
from finsight.state import AnalysisState
from finsight.tools.registry import build_tools

PIPELINE = ["researcher", "quant", "risk", "writer", "critic"]
StepCallback = Callable[[str, AnalysisState], None]


def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


def timed(name: str, fn: Callable[[AnalysisState], AnalysisState]) -> Callable[[AnalysisState], AnalysisState]:
    """Record wall time per node (summed across revisions) in state.node_seconds."""

    @functools.wraps(fn)
    def run(state: AnalysisState) -> AnalysisState:
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            with state._lock:
                state.node_seconds[name] = round(state.node_seconds.get(name, 0.0) + time.perf_counter() - t0, 3)

    return run


def _wire(graph: Any, team: Team, start: Any, end: Any, wrap: Callable) -> None:
    for name in PIPELINE:
        graph.add_node(name, wrap(timed(name, getattr(team, name))))
    graph.add_edge(start, "researcher")
    for a, b in zip(PIPELINE, PIPELINE[1:]):
        graph.add_edge(a, b)
    graph.add_conditional_edges("critic", wrap(team.route_after_critic, router=True),
                                {"revise": "writer", "done": end})


def run_builtin(team: Team, state: AnalysisState, on_step: StepCallback | None) -> AnalysisState:
    g: engine.StateGraph[AnalysisState] = engine.StateGraph()
    _wire(g, team, engine.START, engine.END, lambda fn, router=False: fn)
    return g.compile().invoke(state, on_step=on_step)


def run_langgraph(team: Team, state: AnalysisState, on_step: StepCallback | None) -> AnalysisState:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class GraphState(TypedDict):
        s: AnalysisState

    def wrap(fn, router=False):
        if router:
            return lambda gs: fn(gs["s"])

        def node(gs: GraphState) -> GraphState:
            out = fn(gs["s"])
            if on_step:
                on_step(fn.__name__, out)
            return {"s": out}

        node.__name__ = fn.__name__
        return node

    g = StateGraph(GraphState)
    _wire(g, team, START, END, wrap)
    app = g.compile()
    return app.invoke({"s": state}, config={"recursion_limit": 50})["s"]


def run_analysis(
    ticker: str,
    question: str | None = None,
    *,
    llm: Any = None,
    provider: MarketDataProvider | None = None,
    settings: Settings | None = None,
    engine_name: str | None = None,
    on_step: StepCallback | None = None,
) -> AnalysisState:
    s = settings or Settings.from_env()
    provider = provider or get_provider(s.data)
    state = AnalysisState(ticker=ticker.strip().upper())
    if question:
        state.question = question
    llm = instrument(llm or get_llm(s), state)
    team = Team(llm=llm, tools=build_tools(provider, state), max_revisions=s.max_revisions)

    choice = (engine_name or os.getenv("FINSIGHT_ENGINE", "auto")).lower()
    use_lg = choice == "langgraph" or (choice == "auto" and _langgraph_available())
    state.log("system", "start", {"engine": "langgraph" if use_lg else "builtin", "llm": getattr(llm, "name", "?"),
                                  "data": provider.name})
    return run_langgraph(team, state, on_step) if use_lg else run_builtin(team, state, on_step)
