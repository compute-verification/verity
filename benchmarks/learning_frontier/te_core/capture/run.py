"""A training run's circuit C as the sum of its distinct step graphs, so |C| is exact in MACs without building gates.

    capture_run(segments) -> RunGraph            segments: [(steps, StepGraph | () -> StepGraph), ...]
    schedule_updated(trainable, registered=()) -> [frozenset, ...]

Each segment is a number of steps and their step graph, or a thunk that captures it, which runs only if no earlier
segment's thunk is the same object: give one thunk to every segment of the same step, and a new one where the step's
structure changes (a schedule that changes which parameters train, a new batch shape). Segments whose graphs have the same
structure digest and the same `updated` set merge. |C| = Σ steps · MACs of each distinct step graph.

`schedule_updated` turns a schedule of per-step trainable sets into the `updated` set each step reads: a parameter differs
from θ from the step after the first one that trains it, and stays new whether or not it trains again.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Sequence

from te_core.capture.structure import StepGraph

__all__ = ["RunGraph", "capture_run", "schedule_updated"]


@dataclass
class RunGraph:
    """The run's distinct step graphs, each with its number of steps."""
    steps: list[tuple[int, StepGraph]]

    @property
    def n_steps(self) -> int:
        return sum(n for n, _ in self.steps)

    @property
    def macs(self) -> int:
        """|C| in MACs."""
        return sum(n * g.macs for n, g in self.steps)

    @property
    def tokens(self) -> int:
        return sum(n * g.tokens for n, g in self.steps)

    def role_macs(self, role: str) -> int:
        return sum(n * g.role_macs(role) for n, g in self.steps)

    def deep_macs(self, D, unit: str = "macs") -> int:
        return sum(n * g.deep_macs(D, unit) for n, g in self.steps)

    def phi(self, D, unit: str = "macs") -> Fraction:
        """The share of |C| in using GEMMs with at least D upstream (`StepGraph.depth_profile`'s unit)."""
        return Fraction(self.deep_macs(D, unit), self.macs) if self.macs else Fraction(0)

    def summary(self) -> dict:
        return {"steps": self.n_steps, "distinct": len(self.steps), "macs": self.macs, "tokens": self.tokens,
                "roles": {r: self.role_macs(r) for r in ("using", "producing", "neither")},
                "graphs": [{"steps": n, "digest": g.digest, "name": g.name, "macs": g.macs, "updated": sorted(g.updated),
                            "trained": sorted(g.trained)} for n, g in self.steps]}


def capture_run(segments: Iterable[tuple[int, StepGraph | Callable[[], StepGraph]]]) -> RunGraph:
    """The run of `segments` (module docstring)."""
    by_thunk: dict[int, StepGraph] = {}
    merged: dict[tuple[str, frozenset], list] = {}
    order: list[tuple[str, frozenset]] = []
    for n, g in segments:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"a segment's step count is a count, not {n!r}")
        if not isinstance(g, StepGraph):
            if id(g) not in by_thunk:
                by_thunk[id(g)] = g()
            g = by_thunk[id(g)]
        key = (g.digest, g.updated)
        if key not in merged:
            merged[key] = [0, g]
            order.append(key)
        merged[key][0] += n
    return RunGraph([(merged[k][0], merged[k][1]) for k in order if merged[k][0]])


def schedule_updated(trainable: Sequence[Iterable[str]], registered: Iterable[str] = ()) -> list[frozenset]:
    """For each step of a schedule of trainable sets, the parameters that differ from θ when that step reads them:
    every parameter trained in an earlier step, less `registered` (parameters re-registered at the period's start)."""
    seen: set[str] = set()
    out = []
    reg = set(registered)
    for t in trainable:
        out.append(frozenset(seen - reg))
        seen |= set(t)
    return out
