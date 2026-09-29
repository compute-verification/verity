"""Closure draws: the verification plan for `ncp-v2`'s partition (design note §12.2 (2′), §12.3).

The verifier draws tiles by a law over the tile strata, and proves each drawn tile together with its **closure**: its
X strip's unit, its strip's digest unit, the tree-node units on that strip's path to D_A and the call's key unit, each
proved once, and its Y strip's unit in the weight's own Program (`NcpWeight`), proved once per plan. The rest (the dequantization units) is drawn under the law's floor. Every other unit is never
drawn on its own: a node beneath no drawn tile is never proved, and a wrong one is counted through the unsound tiles
beneath it (Lean `harm_le_unsoundWork`, `closureLaw_escape`).

**The law is a seam** (`TileLaw`), so the plan switches laws without changing:
* `CountLaw` is the law on `main` today, one_stage's and #167's stratified law: one stratum per template, a uniform k_s-subset
  inside each, k_s = max(1, round(K·n_s/N)) (rounding half up), K the law's budget over the strata it draws.
* `WorkLaw` is Verity root's work law (#362, `Flock/Draw.lean` `workK`): k_s = min(n_s, max(1, ⌈K·W_s/W⌉)), with
  K = `RECORD_K` = 27,713 at ε = 0.1% and δ = 2⁻⁴⁰ (`record_k`). K is a **window's** budget: a call draws its share,
  K_c = ⌈K·W_c/W_window⌉, capped at its units (`window_laws`), which is #362's law with strata by (call, template), so
  the window keeps its (1 − ε)^K. It is not of record until root chooses the strata (X-SPC-84): these two, or #362's one
  per template (#372).

**Openings** are row-granular (the red team's X-SPC-57): A and the weight are committed one leaf per row, so a proved
unit opens exactly the rows it reads.

**The layout is the verifier's.** `layout` derives each tile's own unit and its closure from the shape alone (through the
Program the verifier builds), never from a prover: `Traced.check_layout` refuses any other, and `draw` refuses one whose
tile is not its own closure's first unit.

**The profile** is core's `Stratified` over the tile strata, whose `harm_bound` with each tile's W_ref as its harm bounds
the unsound work at δ (Lean `audit_closure`, `accountable_compute`).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Mapping, Protocol, Sequence

from verity.proofs.profile import Stratified, Stratum
from verity.randomness import Key

from ..protocol import DomainError
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

    with localcontext() as cx:                              # the start, ⌈ln δ / ln(1 − ε)⌉, in 60-digit decimals
        cx.prec = 60
        ln = lambda f: (Decimal(f.numerator).ln() - Decimal(f.denominator).ln())  # noqa: E731
        k = max(1, int((ln(dlt) / ln(q)).to_integral_value(rounding=ROUND_CEILING)) - 2)
    while not meets(k):
        k += 1
    while k > 1 and meets(k - 1):
        k -= 1
    return k


#: the tile draws of record: ε = 0.1%, δ = 2⁻⁴⁰
RECORD_K = 27_713


@dataclass(frozen=True)
class WorkLaw:
    """The work law (Verity root's #362, `Flock/Draw.lean` `workK`): each stratum draws
    k_s = min(n_s, max(1, ⌈K·W_s/W⌉)), W_s its work, W the strata's total, 1 ≤ K ≤ N. The floor of one binds exactly for the
    strata that do no work (the dequantization units). ``k`` is the budget K (`record_k`)."""

    k: int = RECORD_K
    name: str = "work"
    window_work: int | None = None          # the window's work W_window, when the call draws its share of K

    def budget(self, strata: Sequence[StratumSpec]) -> int:
        """This call's K: its share ⌈K·W_c/W_window⌉ of the window's (all of K when it is the window), capped at its N."""
        work, n = sum(s.work for s in strata), sum(len(s.units) for s in strata)
        if work < 1 or self.k < 1:
            raise ValueError("the work law needs some work and K >= 1")
        share = self.k if self.window_work is None else -(-self.k * work // self.window_work)
        return min(n, max(1, share))

    def sizes(self, strata: Sequence[StratumSpec]) -> dict[str, int]:
        work, k = sum(s.work for s in strata), self.budget(strata)
        return {s.name: min(len(s.units), max(1, -(-k * s.work // work))) for s in strata}


def window_laws(layouts: Sequence["Layout"], k: int = RECORD_K) -> list[WorkLaw]:
    """Each call's work law in a window of calls: its share of the window's budget K (X-SPC-83). The window draws at most
    K + (its strata) units, and each call's floor draws its dequantization at least once."""
    total = sum(lay.work() for lay in layouts)
    return [WorkLaw(k, window_work=total) for _ in layouts]


@dataclass(frozen=True)
class Layout:
    """The units a plan reads: every unit, the tile and rest strata, and each tile's closure."""

    shape: Shape
    units: tuple[Unit, ...]
    strata: tuple[StratumSpec, ...]
    closure: Mapping[int, tuple[int, ...]]              # tile unit -> the call's units proved with it (itself first)
    y_strip: Mapping[int, int] = None                   # tile unit -> the weight Program's Y unit it reads (column strip)

    def tile_work(self) -> int:
        """One tile's W_ref: its cells' share of the call's (`schemes.ncp.NCP.wref`)."""
        s = self.shape
        return (3 * s.m * s.k * s.n + FORM_UNITS * s.m * s.k) * D * D // (s.m * s.n)

    def work(self) -> int:
        """The call's work W_c: its strata's."""
        return sum(st.work for st in self.strata)

    def check(self) -> None:
        """Refuse a layout whose tiles are not the partition's tile units, or whose closure of a tile doesn't hold the tile's
        own unit first: the Layout theorems (`Layout.accountable_compute`) read τ = the tile's own unit, and a tile mapped to
        its X strip is C1 again.  `draw` runs it; a verifier compares the whole layout with its own (`Traced.check_layout`)."""
        tiles = [u for st in self.strata if st.role == "tile" for u in st.units]
        if set(tiles) != set(self.closure):
            raise DomainError("the layout's closures are not its tile stratum's")
        for t in tiles:
            if self.units[t].role != "tile" or self.closure[t][:1] != (t,):
                raise DomainError(f"unit {t}: a tile is a tile unit, and its closure holds it first")


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
    closure, y_strip = {}, {}
    for t in by_role["tile"]:
        rs, cs = t.where
        path = [nodes[j].index for j, (lo, hi) in enumerate(spans) if lo <= rs < hi]
        closure[t.index] = (t.index, form_x[rs], digest[rs], *path, key)
        y_strip[t.index] = cs
    tile = Layout(shape, us, (), {})
    strata = []
    for role in ("tile", "dequant"):
        group = by_role[role]
        work = tile.tile_work() * len(group) if role == "tile" else 0
        strata.append(StratumSpec(group[0].template, role, tuple(u.index for u in group), work))
    return Layout(shape, us, tuple(strata), closure, y_strip)


@dataclass(frozen=True)
class Plan:
    law: str
    sizes: Mapping[str, int]
    drawn: tuple[int, ...]                   # the units the law drew (tiles and floor units)
    proved: tuple[int, ...]                  # the drawn units and every drawn tile's closure, ascending, each once
    openings: tuple[tuple[str, int], ...]    # the row leaves the proved units open: ("a", row) or ("b", weight row)
    proved_y: tuple[int, ...] = ()           # the weight Program's Y units the drawn tiles read (column strips), each once


def draw(key: Key, lay: Layout, law: TileLaw) -> Plan:
    """The plan for one draw: a uniform k_s-subset of each stratum from ``key`` (the verifier's own randomness, derived under
    `DRAW_DOMAIN` after the receipt), then every drawn tile's closure."""
    lay.check()
    sizes = law.sizes(lay.strata)
    drawn: list[int] = []
    for s in lay.strata:
        drawn += [s.units[i] for i in key.subset(sizes[s.name], len(s.units), ("stratum", s.name))]
    proved = set(drawn)
    ys = {lay.y_strip[u] for u in drawn if lay.y_strip and u in lay.y_strip}
    for u in drawn:
        proved.update(lay.closure.get(u, ()))
    return Plan(law.name, sizes, tuple(sorted(drawn)), tuple(sorted(proved)), openings(lay, proved, ys), tuple(sorted(ys)))


def openings(lay: Layout, proved, proved_y=()) -> tuple[tuple[str, int], ...]:
    """The input rows the proved units read, one leaf per row: a digest or X unit its strip's 16 A rows, a Y unit (the
    weight Program's, by column strip) its 16 weight rows."""
    rows: set[tuple[str, int]] = set()
    for i in proved:
        u = lay.units[i]
        if u.role in ("digest", "form_x"):
            rows.update(("a", D * u.where[0] + r) for r in range(D))
    for cs in proved_y:
        rows.update(("b", D * cs + r) for r in range(D))
    return tuple(sorted(rows))


def profile(lay: Layout, sizes: Mapping[str, int]) -> Stratified:
    """Core's `Stratified` over the tile strata the plan drew."""
    return Stratified(tuple(Stratum(s.name, len(s.units), sizes[s.name]) for s in lay.strata if s.role == "tile"))


def unsound_work_bound(lay: Layout, sizes: Mapping[str, int], delta: float) -> float:
    """`harm_bound` over the tile strata at harm W_ref per tile: except with probability δ (plus the proofs' error), an
    accepted plan leaves at most this much work in unsound tiles."""
    return profile(lay, sizes).harm_bound({s.name: float(lay.tile_work()) for s in lay.strata if s.role == "tile"}, delta)


__all__ = ["DRAW_DOMAIN", "RECORD_K", "CountLaw", "Layout", "Plan", "StratumSpec", "TileLaw", "WorkLaw", "draw", "layout", "openings",
           "profile", "record_k", "unsound_work_bound", "window_laws"]
