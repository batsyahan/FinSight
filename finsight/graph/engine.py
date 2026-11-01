"""A minimal state-graph engine (nodes, edges, conditional edges).

FinSight can run on LangGraph when it is installed; this engine is the zero-dependency
fallback and mirrors LangGraph's model so the same node functions work on both.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

S = TypeVar("S")
END = "__end__"
START = "__start__"


class StateGraph(Generic[S]):
    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[S], S]] = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, tuple[Callable[[S], str], dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[S], S]) -> None:
        if name in (START, END):
            raise ValueError("reserved node name")
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Callable[[S], str], mapping: dict[str, str]) -> None:
        self.cond[src] = (router, mapping)

    def compile(self, max_steps: int = 50) -> "CompiledGraph[S]":
        if START not in self.edges:
            raise ValueError("graph needs an edge from START")
        for src, dst in self.edges.items():
            if dst != END and dst not in self.nodes:
                raise ValueError(f"edge {src}->{dst} points to unknown node")
        return CompiledGraph(self, max_steps)


class CompiledGraph(Generic[S]):
    def __init__(self, graph: StateGraph[S], max_steps: int) -> None:
        self.g, self.max_steps = graph, max_steps

    def invoke(self, state: S, on_step: Callable[[str, S], None] | None = None) -> S:
        current = self.g.edges[START]
        for _ in range(self.max_steps):
            if current == END:
                return state
            state = self.g.nodes[current](state)
            if on_step:
                on_step(current, state)
            if current in self.g.cond:
                router, mapping = self.g.cond[current]
                current = mapping[router(state)]
            else:
                current = self.g.edges.get(current, END)
        raise RuntimeError(f"Graph exceeded {self.max_steps} steps")
