"""Closure draws: the verification plan for `ncp-v2`'s partition (design note §12.2 (2′), §12.3).

The verifier draws tiles by a law over the tile strata, and proves each drawn tile together with its **closure**: its
X strip's unit, its Y strip's unit, its strip's digest unit, the tree-node units on that strip's path to D_A, and the call's
key unit, each proved once. The rest (the dequantization units) is drawn under the law's floor. Every other unit is never
drawn on its own: a node beneath no drawn tile is never proved, and a wrong one is counted through the unsound tiles
beneath it (Lean `harm_le_unsoundWork`, `closureLaw_escape`).

**The law is a seam** (`TileLaw`), so the plan switches laws without changing:
* `CountLaw` is the law on `main` today, one_stage's and #167's stratified law: one stratum per template, a uniform k_s-subset
  inside each, k_s = max(1, round(K·n_s/N)) (rounding half up), K the law's budget over the strata it draws.
* `WorkLaw` is Verity root's work law (draft #362, `Flock/Draw.lean` `workK`): k_s = min(n_s, max(1, ⌈K·W_s/W⌉)), with
  K = `RECORD_K` = 27,713 at ε = 0.1% and δ = 2⁻⁴⁰ (`record_k`). It is not of record until #362 merges; switch
  `law=WorkLaw()` when it does.

**Openings** are row-granular (the red team's X-SPC-57): A and the weight are committed one leaf per row, so a proved
unit opens exactly the rows it reads.

**The profile** is core's `Stratified` over the tile strata, whose `harm_bound` with each tile's W_ref as its harm bounds
the unsound work at δ (Lean `audit_closure`, `accountable_compute`).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, log
from typing import Mapping, Protocol, Sequence

from verity.proofs.profile import Stratified, Stratum
from verity.randomness import Key

from ..schemes.ncp import FORM_UNITS
from .ncp2 import D, Shape, rfc6962
from .partition import Unit, units

DRAW_DOMAIN = "verity/pouw/closure-draw/v1"


@dataclass(frozen=True)
class StratumSpec:
    name: str                   # the template's descriptor id
    role: str
    units: tuple[int, ...]      # its units' partition indices, ascending
    work: int                   # the stratum's total W_ref (0 for the rest)


class TileLaw(Protocol):
    """A draw law's sizes: k_s for each stratum it draws."""

    name: str

    def sizes(self, strata: Sequence[StratumSpec]) -> dict[str, int]: ...


def _round_half_up(num: int, den: int) -> int:
    return (2 * num + den) // (2 * den)


@dataclass(frozen=True)
class CountLaw:
    """The stratified law on `main` today: k_s = max(1, round(K·n_s/N)), one stratum per template."""

    k: int
    name: str = "count"

    def sizes(self, strata: Sequence[StratumSpec]) -> dict[str, int]:
        total = sum(len(s.units) for s in strata)
        return {s.name: min(len(s.units), max(1, _round_half_up(self.k * len(s.units), total))) for s in strata}


def record_k(epsilon: Fraction, delta: Fraction) -> int:
    """The smallest K with (1 − ε)^K ≤ δ, decided in exact integers: wrong units doing a fraction ε of the work escape the
    work law's draw with at most (1 − ε)^K (#362's `work_escape_le`). At ε = 0.1% and δ = 2⁻⁴⁰ it is 27,713; 27,712 misses
    (#362's `record_sizing`, `record_sizing_one_fewer`)."""
    eps, dlt = Fraction(epsilon), Fraction(delta)
    if not (0 < eps < 1 and 0 < dlt < 1):
        raise ValueError("0 < ε < 1 and 0 < δ < 1")
    q = 1 - eps

    def meets(k: int) -> bool:                              # q^k <= δ, cross-multiplied
        return q.numerator ** k * dlt.denominator <= dlt.numerator * q.denominator ** k

    k = max(1, ceil(log(float(dlt)) / log(float(q))) - 2)
    while not meets(k):
        k += 1
    while k > 1 and meets(k - 1):
        k -= 1
    return k


#: the tile draws of record: ε = 0.1%, δ = 2⁻⁴⁰
RECORD_K = 27_713


@dataclass(frozen=True)
class WorkLaw:
    """The work law (Verity root's, draft #362, `Flock/Draw.lean` `workK`): each stratum draws
    k_s = min(n_s, max(1, ⌈K·W_s/W⌉)), W_s its work, W the strata's total, 1 ≤ K ≤ N. The floor of one binds exactly for the
    strata that do no work (the dequantization units). ``k`` is the budget K (`record_k`)."""

    k: int = RECORD_K
    name: str = "work"

    def sizes(self, strata: Sequence[StratumSpec]) -> dict[str, int]:
        work = sum(s.work for s in strata)
        if work < 1 or not 1 <= self.k <= sum(len(s.units) for s in strata):
            raise ValueError("the work law needs some work and 1 <= K <= N")
        return {s.name: min(len(s.units), max(1, -(-self.k * s.work // work))) for s in strata}


@dataclass(frozen=True)
class Layout:
    """The units a plan reads: every unit, the tile and rest strata, and each tile's closure."""

    shape: Shape
    units: tuple[Unit, ...]
    strata: tuple[StratumSpec, ...]
    closure: Mapping[int, tuple[int, ...]]              # tile unit -> the units proved with it (itself first)
    per_tile: int | None = None                         # a tile's W_ref, when not `ncp-v2`'s

    def tile_work(self) -> int:
        """One tile's W_ref: its cells' share of the call's (`schemes.ncp.NCP.wref`, or the scheme's own for NCP-FP8)."""
        if self.per_tile is not None:
            return self.per_tile
        s = self.shape
        return (3 * s.m * s.k * s.n + FORM_UNITS * s.m * s.k) * D * D // (s.m * s.n)


def layout(shape: Shape, kind: str = "shift24") -> Layout:
    us = units(shape, kind)
    by_role: dict[str, list[Unit]] = {}
    for u in us:
        by_role.setdefault(u.role, []).append(u)
    # the tree's nodes in emission order, each with the strips beneath it
    spans: list[tuple[int, int]] = []

    def leaf(i: int) -> tuple[int, int]:
        return i, i + 1

    def node(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        spans.append((left[0], right[1]))
        return spans[-1]

    rfc6962(0, shape.m // D, leaf, node)
    nodes = by_role.get("node", [])
    assert len(nodes) == len(spans)
    key = by_role["key"][0].index
    digest = {u.where[0]: u.index for u in by_role["digest"]}
    form_x = {u.where[0]: u.index for u in by_role["form_x"]}
    form_y = {u.where[0]: u.index for u in by_role["form_y"]}
    closure = {}
    for t in by_role["tile"]:
        rs, cs = t.where
        path = [nodes[j].index for j, (lo, hi) in enumerate(spans) if lo <= rs < hi]
        closure[t.index] = (t.index, form_x[rs], form_y[cs], digest[rs], *path, key)
    tile = Layout(shape, us, (), {})
    strata = []
    for role in ("tile", "dequant"):
        group = by_role[role]
        work = tile.tile_work() * len(group) if role == "tile" else 0
        strata.append(StratumSpec(group[0].template, role, tuple(u.index for u in group), work))
    return Layout(shape, us, tuple(strata), closure)


def fp8_layout(shape, sums: str = "f16", lengths=None) -> Layout:
    """NCP-FP8's layout (`circuit.fp8`): each tile's closure is `ncp-v2`'s (its X and Y strips, its strip's digest, the
    nodes on that strip's path to root_A, the key) with the call's domain unit, and the **pad cone**, every pad strip and
    every node with no row beneath it.  root_A's tree is padded to a power of two, and a pad has no tile beneath it that a
    wrong one could be charged to, so the pad cone is proved whenever a tile is.  A tile's W_ref is its cells' share of
    `FP8IS.wref`, rounded up."""
    from ..protocol import Shape as PShape
    from . import fp8, fp8_reference
    from .partition import circuit_units, fp8_circuit

    lengths = fp8.Lengths(1, 1, 1) if lengths is None else lengths
    us = circuit_units(fp8_circuit(shape, sums, lengths))
    by_role: dict[str, list[Unit]] = {}
    for u in us:
        by_role.setdefault(u.role, []).append(u)
    ups = fp8.upper_nodes(shape)
    nodes = {p: u.index for p, u in zip(ups, by_role.get("node", []))}
    real = shape.m // D
    pad_cone = [u.index for u in by_role.get("pad", [])]
    pad_cone += [nodes[(lv, j)] for lv, j in ups if j << (lv - 3) >= real]
    one = [by_role["key"][0].index, by_role["domain"][0].index, *pad_cone]
    digest = {u.where[0]: u.index for u in by_role["digest"]}
    form_x = {u.where[0]: u.index for u in by_role["form_x"]}
    form_y = {u.where[0]: u.index for u in by_role["form_y"]}
    closure = {}
    for t in by_role["tile"]:
        rs, cs = t.where
        path = [nodes[(lv, j)] for lv, j in ups if j == rs >> (lv - 3)]
        closure[t.index] = (t.index, form_x[rs], form_y[cs], digest[rs], *path, *one)
    wref = fp8_reference.scheme(sums).wref(PShape(shape.m, shape.k, shape.n))
    per_tile = -(-wref * D * D // (shape.m * shape.n))
    strata = []
    for role in ("tile", "dequant"):
        group = by_role[role]
        strata.append(StratumSpec(group[0].template, role, tuple(u.index for u in group), per_tile * len(group) if role == "tile" else 0))
    return Layout(shape, us, tuple(strata), closure, per_tile)


@dataclass(frozen=True)
class Plan:
    law: str
    sizes: Mapping[str, int]
    drawn: tuple[int, ...]                   # the units the law drew (tiles and floor units)
    proved: tuple[int, ...]                  # the drawn units and every drawn tile's closure, ascending, each once
    openings: tuple[tuple[str, int], ...]    # the row leaves the proved units open: ("a", row) or ("b", weight row)


def draw(key: Key, lay: Layout, law: TileLaw) -> Plan:
    """The plan for one draw: a uniform k_s-subset of each stratum from ``key`` (the verifier's own randomness, derived under
    `DRAW_DOMAIN` after the receipt), then every drawn tile's closure."""
    sizes = law.sizes(lay.strata)
    drawn: list[int] = []
    for s in lay.strata:
        drawn += [s.units[i] for i in key.subset(sizes[s.name], len(s.units), ("stratum", s.name))]
    proved = set(drawn)
    for u in drawn:
        proved.update(lay.closure.get(u, ()))
    return Plan(law.name, sizes, tuple(sorted(drawn)), tuple(sorted(proved)), openings(lay, proved))


def openings(lay: Layout, proved) -> tuple[tuple[str, int], ...]:
    """The input rows the proved units read, one leaf per row: a digest or X unit its strip's 16 A rows, a Y unit its 16
    weight rows."""
    rows: set[tuple[str, int]] = set()
    for i in proved:
        u = lay.units[i]
        if u.role in ("digest", "form_x"):
            rows.update(("a", D * u.where[0] + r) for r in range(D))
        elif u.role == "form_y":
            rows.update(("b", D * u.where[0] + r) for r in range(D))
    return tuple(sorted(rows))


def profile(lay: Layout, sizes: Mapping[str, int]) -> Stratified:
    """Core's `Stratified` over the tile strata the plan drew."""
    return Stratified(tuple(Stratum(s.name, len(s.units), sizes[s.name]) for s in lay.strata if s.role == "tile"))


def unsound_work_bound(lay: Layout, sizes: Mapping[str, int], delta: float) -> float:
    """`harm_bound` over the tile strata at harm W_ref per tile: except with probability δ (plus the proofs' error), an
    accepted plan leaves at most this much work in unsound tiles."""
    return profile(lay, sizes).harm_bound({s.name: float(lay.tile_work()) for s in lay.strata if s.role == "tile"}, delta)


__all__ = ["DRAW_DOMAIN", "RECORD_K", "CountLaw", "Layout", "Plan", "StratumSpec", "TileLaw", "WorkLaw", "draw", "fp8_layout",
           "layout", "openings", "profile", "record_k", "unsound_work_bound"]
