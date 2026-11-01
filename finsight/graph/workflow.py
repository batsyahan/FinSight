"""Wires the agent team into a graph:

    START -> planner -> (analysts in parallel) -> risk -> writer -> critic --revise--> writer
                        researcher | quant | peers                          |--review--> human_review
                                                                            \\--done----> END
    human_review --approve--> END | --revise--> writer | --recheck (edited)--> critic

The planner picks which analysts run; they fan out in parallel and fan back in at risk.
Runs on LangGraph when installed (native fan-out), otherwise on the built-in engine.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Annotated, Any, Callable, TypedDict

from finsight.agents.team import ANALYSTS, Team
from finsight.config import Settings
from finsight.data.providers import MarketDataProvider, get_provider
from finsight.graph import engine
from finsight.llm.providers import get_llm
from finsight.obs import instrument
from finsight.state import AnalysisState
from finsight.tools.registry import build_tools

StepCallback = Callable[[str, AnalysisState], None]


def _same_state(_old: AnalysisState, new: AnalysisState) -> AnalysisState:
    return new  # parallel branches all mutate and return the one shared, thread-safe state


class GraphState(TypedDict):
    """LangGraph state. Defined at module level: LangGraph resolves these type hints at runtime."""

    s: Annotated[AnalysisState, _same_state]


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
    nodes = ["planner", *ANALYSTS, "risk", "writer", "critic", "human_review"]
    for name in nodes:
        graph.add_node(name, wrap(timed(name, getattr(team, name))))
    graph.add_edge(start, "planner")
    graph.add_conditional_edges("planner", wrap(team.route_analysts, router=True), {a: a for a in ANALYSTS})
    for a in ANALYSTS:
        graph.add_edge(a, "risk")
    graph.add_edge("risk", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", wrap(team.route_after_critic, router=True),
                                {"revise": "writer", "review": "human_review", "done": end})
    graph.add_conditional_edges("human_review", wrap(team.route_after_human, router=True),
                                {"approve": end, "revise": "writer", "recheck": "critic"})


def run_builtin(team: Team, state: AnalysisState, on_step: StepCallback | None) -> AnalysisState:
    g: engine.StateGraph[AnalysisState] = engine.StateGraph()
    _wire(g, team, engine.START, engine.END, lambda fn, router=False: fn)
    return g.compile().invoke(state, on_step=on_step)


def run_langgraph(team: Team, state: AnalysisState, on_step: StepCallback | None) -> AnalysisState:
    from langgraph.graph import END, START, StateGraph

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
    return app.invoke({"s": state}, config={"recursion_limit": 60})["s"]


def run_analysis(
    ticker: str,
    question: str | None = None,
    *,
    llm: Any = None,
    provider: MarketDataProvider | None = None,
    settings: Settings | None = None,
    engine_name: str | None = None,
    on_step: StepCallback | None = None,
    analysts: list[str] | None = None,
    peers: list[str] | None = None,
    reviewer: Any = None,
    filings: Any = None,
) -> AnalysisState:
    """``filings`` is a FilingsIndex, or paths (files/dirs, comma-separated) to index."""
    s = settings or Settings.from_env()
    provider = provider or get_provider(s.data)
    state = AnalysisState(ticker=ticker.strip().upper())
    if question:
        state.question = question
    if analysts:
        unknown = set(analysts) - set(ANALYSTS)
        if unknown:
            raise ValueError(f"unknown analysts {sorted(unknown)}; choose from {ANALYSTS}")
        state.plan = list(analysts)
    state.peers = [p.strip().upper() for p in (peers or []) if p.strip()]
    llm = instrument(llm or get_llm(s), state)
    if filings is not None and not hasattr(filings, "search"):
        from finsight.rag.index import FilingsIndex

        filings = FilingsIndex.build(filings)
    team = Team(llm=llm, tools=build_tools(provider, state, filings), max_revisions=s.max_revisions,
                reviewer=reviewer)

    choice = (engine_name or os.getenv("FINSIGHT_ENGINE", "auto")).lower()
    use_lg = choice == "langgraph" or (choice == "auto" and _langgraph_available())
    state.log("system", "start", {"engine": "langgraph" if use_lg else "builtin", "llm": getattr(llm, "name", "?"),
                                  "data": provider.name})
    return run_langgraph(team, state, on_step) if use_lg else run_builtin(team, state, on_step)
