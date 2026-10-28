"""Candidate cuts for the sampling planner (`optimize`), generated up a Definition DAG.

A `Shape` is one node of the DAG at its statics: an instance's ANDs, its credited work, its output width, the rows a proved
instance opens, what serving commits when the node is a proof block (its outputs, now committed wires), and its ways down: its
body's children (`parts`) and other decompositions (`ways`), its statics re-bound (a dot's K cut into chunks that carry the
accumulator) or a shared subcircuit recomputed inside each reader (a covering).

`cuts` is a dynamic program at each density of `optimize`'s sweep: a node is a proof block at its statics when its outputs fit a
public size class (`one_stage.partition.unit_class`, the one-stage partition's), or descends by its cheapest way.  A
proof block is priced as the planner prices a subcircuit, serving's commitment on every instance plus the touched statements and
proved proof blocks of its draw, so a finer cut pays for the wires it commits.
Each distinct cheapest cut over the grid becomes one alternative of the op (`Candidate.cut`), and the planner chooses among
them exactly; the program's price is approximate where one Definition appears under several parents (their proof blocks merge
into one subcircuit only afterwards).

Two hints come with the cuts (`Pressure`): `boundary`, where a proof block between a node and its dominant child's size (interpolated
log-linearly between the two) would beat both, so a Definition at that size would pay; and `width`, where a node wider than
the widest class would beat every legal cut, so the cap has a price there.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from math import log

from verity.core.protocols.sampled_proofs.one_stage.partition import UNIT_CLASS_MAX, unit_class
from verity.core.protocols.sampled_proofs.optimize import Candidate, CostModel, Objective, _k, template_cost

Parts = tuple[tuple["Shape", int], ...]


@dataclass(frozen=True)
class Shape:
    """A Definition at its statics, `id` naming it: one instance has `ands` ANDs, credits `work` and forfeits `influence`
    downstream when wrong, returns `out_bits` over `out_gates` gates, opens `rows` (bytes each) when proved, and has serving
    commit `commit_bits` in `commit_rows` rows when it is a proof block; a statement holds at most `cap` of it.  It descends into
    `parts` (its body's children, each with its count per instance) or any of `ways`, labelled."""

    id: str
    ands: int
    work: int
    out_bits: int
    out_gates: int = 1
    rows: tuple[int, ...] = ()
    commit_bits: int = 0
    commit_rows: float = 0.0
    influence: int = 0
    cap: int = 1
    parts: Parts = ()
    ways: tuple[tuple[str, Parts], ...] = ()

    def decompositions(self) -> list[tuple[str, Parts]]:
        return ([("body", self.parts)] if self.parts else []) + list(self.ways)


@dataclass(frozen=True)
class Pressure:
    """A developer hint at density `lam`: `kind` "boundary" when proof blocks of 1/`split` of `node` (between it and `child`)
    would cost `saving` (a fraction) less than the best cut there, or "width" when `node` as a proof block, wider than the widest
    class, would."""

    kind: str
    node: str
    child: str
    lam: float
    split: int
    saving: float


def _legal(s: Shape) -> bool:
    return unit_class(s.out_bits) is not None


def _candidate(op: str, s: Shape, n: int, label: str = "", replay: tuple[str, int] | None = None) -> Candidate:
    name, per = ("", 0) if replay is None else (replay[0], n // replay[1])
    return Candidate(op, f"{op}:{s.id}", n, s.work, s.out_bits, s.out_gates, label, s.influence, s.ands, s.rows,
                     s.commit_bits, s.commit_rows, s.cap, name, per)


def _price(c: Candidate, costs: CostModel, objective: Objective, lam: float) -> float:
    return costs.commit(c) + template_cost(c, costs, _k(c, objective, lam))[0]


def _unchecked(s: Shape, n: int) -> Candidate:
    """`n` of `s` as a candidate whatever its width, for the width hint only."""
    return Candidate("x", s.id, n, s.work, 0, 1, "", s.influence, s.ands, s.rows, s.commit_bits, s.commit_rows, s.cap)


class _Program:
    def __init__(self, op: str, costs: CostModel, objective: Objective, lam: float, replay: tuple[str, int] | None):
        self.op, self.costs, self.objective, self.lam, self.replay = op, costs, objective, lam, replay
        self.memo: dict[tuple[str, int], tuple[float, tuple[tuple[Shape, int], ...]]] = {}

    def unit(self, s: Shape, n: int) -> float:
        return _price(_candidate(self.op, s, n, replay=self.replay), self.costs, self.objective, self.lam)

    def best(self, s: Shape, n: int) -> tuple[float, tuple[tuple[Shape, int], ...]]:
        key = (s.id, n)
        if key in self.memo:
            return self.memo[key]
        options = []
        if _legal(s):
            options.append((self.unit(s, n), ((s, n),)))
        for _, parts in s.decompositions():
            subs = [self.best(child, n * count) for child, count in parts]
            options.append((sum(p for p, _ in subs), tuple(u for _, us in subs for u in us)))
        if not options:
            raise ValueError(f"{s.id} has no legal cut: its {s.out_bits} bits out fit no size class (at most "
                             f"{UNIT_CLASS_MAX}) and it has no parts or ways to descend into")
        self.memo[key] = min(options, key=lambda o: o[0])
        return self.memo[key]


def _merge(units: Sequence[tuple[Shape, int]]) -> tuple[tuple[Shape, int], ...]:
    out: dict[str, list] = {}
    for s, n in units:
        if s.id in out and out[s.id][0] != s:
            raise ValueError(f"two different shapes are named {s.id!r}")
        out.setdefault(s.id, [s, 0])[1] += n
    return tuple((s, n) for s, n in sorted(out.values(), key=lambda v: v[0].id))


def _lams(root: Shape, n: int, objective: Objective, grid: int) -> list[float]:
    seen, stack, harms = set(), [root], []
    while stack:
        s = stack.pop()
        if s.id in seen:
            continue
        seen.add(s.id)
        harms.append(objective.work * (s.work + s.influence) + objective.bits * s.out_bits)
        stack.extend(child for _, parts in s.decompositions() for child, _ in parts)
    positive = [h for h in harms if h > 0]
    if not positive:
        raise ValueError(f"nothing under {root.id} does harm under this objective")
    total = objective.work * (root.work + root.influence) * n + objective.bits * root.out_bits * n
    lo, hi = 1 / (10 * max(total, max(positive))), 1 / min(positive)
    return [lo * (hi / lo) ** (i / (grid - 1)) for i in range(grid)]


def _geo(a: float, b: float, f: float) -> float:
    return a * (b / a) ** f


def _between(parent: Shape, child: Shape, count: int, split: int) -> Shape:
    """A proof block of 1/`split` of `parent`: its ANDs, rows and cap log-linear between the parent's (split 1) and the child's
    (split `count`), a row only the child opens (its incoming accumulator, say) opened whole, and its commitment the
    parent's plus the share 1 - 1/split of the boundaries the child's cut adds."""
    f, b = log(split) / log(count), (1 - 1 / split) / (1 - 1 / count)
    rows = tuple(round(_geo(p, c, f)) if p and c else c for p, c in zip_longest(parent.rows, child.rows, fillvalue=0)
                 if c)
    return Shape(f"{parent.id}/{split}", round(_geo(parent.ands, child.ands, f)), parent.work // split,
                 child.out_bits, child.out_gates, rows,
                 round(parent.commit_bits + b * (child.commit_bits - parent.commit_bits)),
                 parent.commit_rows + b * (child.commit_rows - parent.commit_rows), parent.influence // split,
                 max(1, round(_geo(parent.cap, child.cap, f))))


def _pressure(prog: _Program, s: Shape, n: int, threshold: float, out: list[Pressure], seen: set) -> None:
    if (s.id, n) in seen:
        return
    seen.add((s.id, n))
    best = prog.best(s, n)[0]
    if not _legal(s):
        free = _price(_unchecked(s, n), prog.costs, prog.objective, prog.lam)
        if free < best * (1 - threshold):
            out.append(Pressure("width", s.id, "", prog.lam, 1, 1 - free / best))
    for _, parts in s.decompositions():
        child, count = max(parts, key=lambda p: p[0].ands * p[1])
        if count >= 4 and child.ands * count >= 0.9 * sum(c.ands * k for c, k in parts) and _legal(s):
            splits = [1 << i for i in range(1, count.bit_length()) if 1 << i < count]
            priced = [(prog.unit(mid, n * g), g) for g in splits
                      for mid in [_between(s, child, count, g)] if _legal(mid)]
            if priced and min(priced)[0] < best * (1 - threshold):
                price, g = min(priced)
                out.append(Pressure("boundary", s.id, child.id, prog.lam, g, 1 - price / best))
        for child, count in parts:
            _pressure(prog, child, n * count, threshold, out, seen)


def pressure(op: str, root: Shape, n: int, costs: CostModel, objective: Objective, lam: float, *,
             replay: tuple[str, int] | None = None, threshold: float = 0.1) -> list[Pressure]:
    """The hints at one density `lam` (a plan's), each saving more than `threshold`."""
    out: list[Pressure] = []
    _pressure(_Program(op, costs, objective, lam, replay), root, n, threshold, out, set())
    return _strongest(out)


def cuts(op: str, root: Shape, n: int, costs: CostModel, objective: Objective, *, grid: int = 41,
         replay: tuple[str, int] | None = None, threshold: float = 0.1
         ) -> tuple[list[Candidate], dict[str, tuple[tuple[str, int], ...]], list[Pressure]]:
    """The op's alternatives for `optimize.frontier`: `n` instances of `root`, cut by the dynamic program at each density of
    a `grid`, each distinct cheapest cut one alternative ("c0" the sparsest's, and on), its proof blocks merged by Definition.
    Also each alternative's proof blocks (Definition id, instances) and the hints that save more than `threshold`.  `replay`
    (domain, replay blocks) puts every subcircuit in that replay domain, `n` and each part's count splitting evenly."""
    found: dict[tuple, str] = {}
    out: list[Candidate] = []
    hints: list[Pressure] = []
    for lam in _lams(root, n, objective, grid):
        prog = _Program(op, costs, objective, lam, replay)
        units = _merge(prog.best(root, n)[1])
        key = tuple((s.id, k) for s, k in units)
        if key not in found:
            label = found[key] = f"c{len(found)}"
            out.extend(replace(_candidate(op, s, k, label, replay), template=f"{op}#{label}:{s.id}") for s, k in units)
        _pressure(prog, root, n, threshold, hints, set())
    return out, {label: key for key, label in found.items()}, _strongest(hints)


def _strongest(hints: Sequence[Pressure]) -> list[Pressure]:
    """Per (kind, node, child), the hint of the largest saving over the grid."""
    best: dict[tuple, Pressure] = {}
    for h in hints:
        key = (h.kind, h.node, h.child)
        if key not in best or h.saving > best[key].saving:
            best[key] = h
    return sorted(best.values(), key=lambda h: (-h.saving, h.kind, h.node))


def shape_of(defn, pieces: Mapping[str, int], *, work: Mapping[str, int] | None = None, rows: bool = True) -> Shape:
    """A specialized Definition's shape from its body: each primitive's ANDs from `pieces` (circuit-check's pins, by
    primitive id; a constant, with no parameters, has none), each child Definition a part with its count (its calls, a
    batch's members, a scan's iterations, summed), credited `work` by Definition
    id (0 elsewhere, a node's being its parts' unless named), the rows a proved instance opens one per parameter and one for
    its return (with `rows`), and serving's commitment its return's bits."""
    from verity.ml.circuits.types import Array, Record, Tuple, Value

    def bits(t) -> int:
        if isinstance(t, Value):
            return t.w
        if isinstance(t, Array):
            return t.n * bits(t.elem)
        if isinstance(t, Tuple):
            return sum(bits(f) for f in t.fields)
        if isinstance(t, Record):
            return sum(bits(f) for _, f in t.fields)
        raise TypeError(f"no width for {t!r}")

    work = work or {}
    memo: dict[str, Shape] = {}

    def walk(fn) -> Shape:
        if fn.id in memo:
            return memo[fn.id]
        params, ret = fn.signature()
        io = tuple((bits(t) + 7) // 8 for _, t in params) + ((bits(ret) + 7) // 8,) if rows else ()
        out_bits, out_gates = bits(ret), ret.leaves
        if not hasattr(fn, "body"):
            if fn.id not in pieces and params:
                raise KeyError(f"no AND count for the primitive {fn.id!r}: pin it (circuit-check) or pass it in pieces")
            s = Shape(fn.id, pieces.get(fn.id, 0), work.get(fn.id, 0), out_bits, out_gates, io, out_bits)
        else:
            counts: dict[str, list] = {}
            for node in fn.body.nodes:
                child = walk(node.fn)
                counts.setdefault(child.id, [child, 0])[1] += node.n
            parts = tuple((c, k) for c, k in counts.values())
            s = Shape(fn.id, sum(c.ands * k for c, k in parts),
                      work.get(fn.id, sum(c.work * k for c, k in parts)), out_bits, out_gates, io, out_bits,
                      parts=parts)
        memo[fn.id] = s
        return s

    return walk(defn)


__all__ = ["Pressure", "Shape", "cuts", "pressure", "shape_of"]
