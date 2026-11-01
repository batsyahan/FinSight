"""A minimal state-graph engine (nodes, edges, conditional edges, parallel fan-out).

FinSight runs on LangGraph when it is installed; this engine is the zero-dependency
fallback and mirrors LangGraph's model so the same node functions work on both:

* ``add_conditional_edges`` routers may return a node name or a *list* of names.
  A list fans out: those nodes run concurrently (threads), and execution continues at
  their common successor once all of them finish (fan-in), like LangGraph supersteps.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, TypeVar

S = TypeVar("S")
END = "__end__"
START = "__start__"


class StateGraph(Generic[S]):
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[S], S]] = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, tuple[Callable[[S], str | list[str]], dict[str, str] | None]] = {}

    def add_node(self, name: str, fn: Callable[[S], S]) -> None:
        if name in (START, END):
            raise ValueError("reserved node name")
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Callable[[S], str | list[str]],
                              mapping: dict[str, str] | None = None) -> None:
        self.cond[src] = (router, mapping)

    def compile(self, max_steps: int = 50, max_workers: int = 4) -> "CompiledGraph[S]":
        if START not in self.edges and START not in self.cond:
            raise ValueError("graph needs an edge from START")
        for src, dst in self.edges.items():
            if dst != END and dst not in self.nodes:
                raise ValueError(f"edge {src}->{dst} points to unknown node")
        return CompiledGraph(self, max_steps, max_workers)


class CompiledGraph(Generic[S]):
    def __init__(self, graph: StateGraph[S], max_steps: int, max_workers: int) -> None:
        self.g, self.max_steps, self.max_workers = graph, max_steps, max_workers

    def _next(self, node: str, state: S) -> list[str]:
        if node in self.g.cond:
            router, mapping = self.g.cond[node]
            out = router(state)
            names = out if isinstance(out, list) else [out]
            return [mapping[n] if mapping else n for n in names]
        return [self.g.edges.get(node, END)]

    def invoke(self, state: S, on_step: Callable[[str, S], None] | None = None) -> S:
        frontier = self._next(START, state)
        for _ in range(self.max_steps):
            frontier = [n for n in frontier if n != END]
            if not frontier:
                return state
            if len(frontier) == 1:
                node = frontier[0]
                state = self.g.nodes[node](state)
                if on_step:
                    on_step(node, state)
                frontier = self._next(node, state)
                continue
            # fan-out: run branches concurrently on the shared state, then fan in
            def branch(n: str) -> None:
                self.g.nodes[n](state)
                if on_step:
                    on_step(n, state)  # reported as each branch finishes

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(frontier))) as pool:
                for f in [pool.submit(branch, n) for n in frontier]:
                    f.result()  # re-raise branch errors
            successors = {s for n in frontier for s in self._next(n, state)}
            if len(successors) != 1:
                raise RuntimeError(f"parallel branches {frontier} must share one successor, got {successors}")
            frontier = list(successors)
        raise RuntimeError(f"Graph exceeded {self.max_steps} steps")
