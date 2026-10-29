"""Parallel oracle programs over P3's primitives, and their costs.

A program is a generator: each `yield` sends one round of queries together (Lean's `OracleAlgorithm.round qs k`) and
receives their answers in order, and `return` is Lean's `.ret`.  `run` answers the queries and counts the cost as Lean's
`OracleAlgorithm.depth` (rounds), `OracleAlgorithm.work` (queries) and `OracleAlgorithm.width` (the largest round) do,
so a program's rounds are measured, not claimed.  Local computation between rounds is free.  `parallel` runs programs in
lockstep.

The queries are the public interface of the primitives: the tweakable permutation `P_t` and `P_t⁻¹` on labels, and
whatever the key derivation `H` exposes: `Π₂` and `Π₂⁻¹` for the overwrite chain (`primitives.OverwriteChain`), `Π` and
`Π⁻¹` for the broken sponge modes, or one monolithic `H` query for a random oracle on key queries
(`primitives.RandomOracleKey`, Lean's `p3Model'`).
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KeyQuery:
    """An element of Lean's `KQ' n wc = Bits 256 × Fin 2 × Fin n × (Fin n → Option (Lab n wc))`.

    `layer` is 0 for the lower layer and 1 for the top layer.  `parents` lists the defined entries of the partial
    assignment as `(node, label)` pairs in increasing node order; every other node maps to `none`.
    """

    tag: int
    layer: int
    node: int
    parents: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Tweak:
    """`P`'s tweak `t = (tag, layer, node)`: the segment tag, layer and node `H` takes."""

    tag: int
    layer: int
    node: int

    def to_bytes(self) -> bytes:
        """`T = tag_s || u8(layer) || u16(v)`, 35 B."""
        if not 0 <= self.tag < 1 << 256 or self.layer not in (0, 1) or not 0 <= self.node < 1 << 16:
            raise ValueError("tweak outside (32 B tag, layer 0 or 1, 16-bit node)")
        return self.tag.to_bytes(32, "little") + bytes([self.layer]) + self.node.to_bytes(2, "little")


@dataclass(frozen=True, slots=True)
class HQuery:
    """A monolithic `H` query (Lean `.inl q`)."""

    q: KeyQuery


@dataclass(frozen=True, slots=True)
class PQuery:
    """`P_t x`."""

    t: Tweak
    x: int


@dataclass(frozen=True, slots=True)
class PinvQuery:
    """`P_t⁻¹ y`."""

    t: Tweak
    y: int


@dataclass(frozen=True, slots=True)
class PiQuery:
    """`Π x`: a query to the key derivation's permutation (`Π₂` for the overwrite chain)."""

    x: int


@dataclass(frozen=True, slots=True)
class PiInvQuery:
    """`Π⁻¹ y`."""

    y: int


type Query = HQuery | PQuery | PinvQuery | PiQuery | PiInvQuery
type Prog[T] = Generator[list[Query], list[int], T]


def answer(prims: Any, q: Query) -> int:
    """The answer of `prims` (a `primitives.Primitives`) to one query."""
    if isinstance(q, PQuery):
        return prims.p.forward(q.t, q.x)
    if isinstance(q, PinvQuery):
        return prims.p.inverse(q.t, q.y)
    return prims.h.answer(q)


@dataclass(slots=True)
class Cost:
    depth: int = 0
    work: int = 0
    width: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    def add(self, queries: list[Query]) -> None:
        self.depth += 1
        self.work += len(queries)
        self.width = max(self.width, len(queries))
        for q in queries:
            name = type(q).__name__
            self.kinds[name] = self.kinds.get(name, 0) + 1

    @property
    def inverse(self) -> int:
        """`P⁻¹` and `Π⁻¹` queries."""
        return self.kinds.get("PinvQuery", 0) + self.kinds.get("PiInvQuery", 0)


type RoundHook = Callable[[int, list[Query], list[int]], None]


def run[T](prog: Prog[T], prims: Any, hook: RoundHook | None = None) -> tuple[T, Cost]:
    """Lean `OracleAlgorithm.run` with `answer`, and the path's cost.  `hook(round, queries, answers)` sees every round."""
    cost = Cost()
    try:
        queries = next(prog)
        while True:
            queries = list(queries)
            answers = [answer(prims, q) for q in queries]
            cost.add(queries)
            if hook is not None:
                hook(cost.depth, queries, answers)
            queries = prog.send(answers)
    except StopIteration as stop:
        return stop.value, cost


def pure[T](value: T) -> Prog[T]:
    """Lean `.ret value`: no rounds."""
    return value
    yield


def parallel(progs: Iterable[Prog]) -> Prog[list]:
    """The programs side by side: each round sends every live program's next round; depth is the largest depth."""
    progs = list(progs)
    results: list = [None] * len(progs)
    pending: dict[int, list[Query]] = {}
    for i, prog in enumerate(progs):
        try:
            pending[i] = list(next(prog))
        except StopIteration as stop:
            results[i] = stop.value
    while pending:
        order = list(pending)
        answers = yield [q for i in order for q in pending[i]]
        at, following = 0, {}
        for i in order:
            part = answers[at:at + len(pending[i])]
            at += len(pending[i])
            try:
                following[i] = list(progs[i].send(part))
            except StopIteration as stop:
                results[i] = stop.value
        pending = following
    return results
