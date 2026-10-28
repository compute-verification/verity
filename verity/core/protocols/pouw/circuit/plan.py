"""Closure samples: the verification plan for `ncp-v2`'s partition (design note §12.2 (2′), §12.3).

The strata are the verifier's own (`Flock/Draw.lean` `strataOf`, root's ruling on X-SPC-84): one per template of the
call's partition, its units as ranges, in the order of their first unit. The law draws each stratum; the plan then proves
each drawn tile together with its **closure**: its X strip's unit, its strip's digest unit, the tree-node units on that
strip's path to D_A and the call's key unit, each proved once, and its Y strip's unit in the weight's own Program
(`NcpWeight`), once per plan. Every other template is sampled at its floor, so a wholly wrong one is caught directly
(`audit_work_whole_stratum`), and a wrong unit beneath no drawn tile is counted through the unsound tiles beneath it (Lean
`harm_le_unsoundWork`, `closureLaw_escape`).

**The law is a seam** (`TileLaw`), so the plan switches laws without changing:
* `CountLaw` is the law on `main` today, one_stage's and #167's stratified law: one stratum per template, a uniform k_s-subset
  inside each, k_s = max(1, round(K·n_s/N)) (rounding half up), K the law's budget over the strata it samples.
* `WorkLaw` is Verity root's work law (#362 with #374's floors, `Flock/Draw.lean` `workK`):
  k_s = min(n_s, max(f_s, ⌈K·W_s/W⌉)), with K = `RECORD_K` = 27,713 at ε = 0.1% and δ = 2⁻⁴⁰ (`record_k`) and f_s the
  verifier's floor. `work_table` is the verifier's `--work-table` and `law_object` the law `verify` derives from it. K is
  a **window's** budget: a call draws its share K_c = ⌈K·W_c/W_window⌉, capped at its units, and its y cells at the
  floor ⌈K_y·n_c/N⌉ of the window's N, K_y = K (`window_laws`, root's option B on X-SPC-105).

**The tile checks' layout** (`tile_layout`, `pc4` and `pc8`) is beside it: a drawn tile's closure is its rows' units
in the call (`RowA`, or `rowk`'s pieces; TM rows, or `pc8`'s short last tile row's r), and its TN rows' units in the
weight's Program (``b_rows``), each once per plan (`Plan.proved_b`); its W_ref is the caller's (the tile's cells' share of the scheme's).  It opens
nothing: its rows are C-Flock-committed ports.

**Openings** are row-granular (the red team's X-SPC-57): A and the weight are committed one leaf per row, so a proved
unit opens exactly the rows it reads.

**The layout is the verifier's.** `layout` derives each tile's own unit and its closure from the shape alone (through the
Program the verifier builds), never from a prover: `Traced.check_layout` refuses any other, and `draw` refuses one whose
tile is not its own closure's first unit.

**The profile** is core's `Stratified` over the tile strata, whose `harm_bound` with each tile's W_ref as its harm bounds
the unsound work at δ (Lean `audit_closure`, `accountable_compute`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Mapping, Protocol, Sequence

from verity.core.protocols.profile import Stratified, Stratum
from verity.ml.randomness import Key

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
    work: int                   # the stratum's total W_ref (0 for units that do no PoUW work)

    def ranges(self) -> list[list[int]]:
        """Its units as ascending half-open ranges [lo, hi), adjacent ones merged."""
        out: list[list[int]] = []
        for u in self.units:
            if out and out[-1][1] == u:
                out[-1][1] = u + 1
            else:
                out.append([u, u + 1])
        return out


class TileLaw(Protocol):
    """A sample law's sizes: k_s for each stratum it samples."""

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


def check_work_strata(strata: Sequence[StratumSpec]) -> None:
    """Refuse a call whose work is in more than one stratum, or whose tiles are of more than one template. The window pin's
    `covers_window` takes each call's work in one stratum as its hypothesis `hone`: where the cap K_c ≤ N_c binds, a
    second work stratum is under-drawn (the red team's window: ten wrong units escape with 0.156 against a claimed
    2.4·10⁻⁴). A stratum does work when 0 < w_s·n_s, its `work`. X-SPC-84 gives work to the tile template only; this
    makes that a check, not a convention."""
    worked = [s.name for s in strata if s.work > 0]
    if len(worked) > 1:
        raise ValueError(f"a call's work is in one stratum (the window pin's hone), not in {len(worked)}: {worked}")
    tiles = [s.name for s in strata if s.role == "tile"]
    if len(tiles) > 1:
        raise ValueError(f"a call's tiles share one template, not {len(tiles)}: {tiles}")


def record_k(epsilon: Fraction, delta: Fraction) -> int:
    """The smallest K with (1 − ε)^K ≤ δ, decided in exact integers: wrong units doing a fraction ε of the work escape the
    work law's sample with at most (1 − ε)^K (#362's `work_escape_le`). At ε = 0.1% and δ = 2⁻⁴⁰ it is 27,713; 27,712 misses
    (#362's `record_sizing`, `record_sizing_one_fewer`). That ε at δ is the default point the profile reports at, not a
    policy parameter: PoUW has none."""
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


#: the tile draws at the default report point, ε = 0.1% at δ = 2⁻⁴⁰
RECORD_K = 27_713


@dataclass(frozen=True)
class WorkLaw:
    """The work law (Verity root's #362, `Flock/Draw.lean` `workK`): each stratum samples
    k_s = min(n_s, max(f_s, ⌈K·W_s/W⌉)), W_s its work, W the strata's total, 1 ≤ K ≤ N, and f_s ≥ 1 its floor from the
    verifier's own table (X-SPC-81; 1 for every template it doesn't name). The floor binds for the strata that do no work
    (every template but the tile's). ``k`` is the budget K (`record_k`)."""

    k: int = RECORD_K
    name: str = "work"
    window_work: int | None = None          # the window's work W_window, when the call draws its share of K
    floors: tuple[tuple[str, int], ...] = ()    # the verifier's floors f_s by template descriptor id

    def floor(self, template: str) -> int:
        f = dict(self.floors).get(template, 1)
        if f < 1:
            raise ValueError(f"stratum {template!r}: a floor is at least one")
        return f

    def budget(self, strata: Sequence[StratumSpec]) -> int:
        """This call's K: its share ⌈K·W_c/W_window⌉ of the window's (all of K when it is the window), capped at its N. The
        call's work must be in one stratum (`check_work_strata`)."""
        check_work_strata(strata)
        work, n = sum(s.work for s in strata), sum(len(s.units) for s in strata)
        if work < 1 or self.k < 1:
            raise ValueError("the work law needs some work and K >= 1")
        share = self.k if self.window_work is None else -(-self.k * work // self.window_work)
        return min(n, max(1, share))

    def sizes(self, strata: Sequence[StratumSpec]) -> dict[str, int]:
        work, k = sum(s.work for s in strata), self.budget(strata)
        return {s.name: min(len(s.units), max(self.floor(s.name), -(-k * s.work // work))) for s in strata}


def window_laws(layouts: Sequence["Layout"], k: int = RECORD_K, floors: tuple[tuple[str, int], ...] = (),
                k_y: int | None = None) -> list[WorkLaw]:
    """Each call's work law in a window of calls: its share K_c = ⌈K·W_c/W_window⌉ of the window's budget K (X-SPC-83), capped
    at its units, over its own strata. That is the law `verify` enforces when each call is its own statement, given its own
    K_c (`--work K_c`) and work table. Within one Program (`ncp2.window`) the calls share their templates' strata and one K,
    which is what `verify` derives there instead.

    **y's floors** (root's option B on X-SPC-105): each call's dequantization stratum s has floor f_s = ⌈K_y·n_s/N⌉, n_s its
    units and N the window's dequantization units, so k_s = min(n_s, f_s) meets k_s·N ≥ K_y·n_s or proves the stratum, and a
    window samples at most K_y + C of its y cells. K_y = K = 27,713 keeps δ = 2⁻⁴⁰ over the window: a tile-bad and a y-bad
    wrong set are one family of the audit, so its sampling term is the larger of (1 − ε)^K and (1 − ε)^K_y, not their sum.
    The other zero-work templates keep their floors (1 unless ``floors`` names them, X-SPC-84). Per call these are pinned
    (`audit_work_floor`); over the window, the per-call split (X-SPC-83, 106) and B (X-SPC-105) are root's window pin
    `audit_window_split_of_record` (`FlockSoundness/Audit/Window.lean`, #418), which takes each call's work
    in one stratum (`check_work_strata`) and each call's sample under its own key context (`Traced.draw`'s `Ledger`).
    It also takes a window that does some work (0 < W, its `hW`) and has y cells (0 < N, its `hN`): a window with
    either empty is refused here, so none is ever drawn or accepted."""
    if not layouts:
        raise ValueError("a window has at least one call")
    for lay in layouts:
        check_work_strata(lay.strata)
    total = sum(lay.work() for lay in layouts)
    if total <= 0:
        raise ValueError("a window does some work (0 < W, the window pin's hW): this one does none")
    k_y = k if k_y is None else k_y
    ys = [[s for s in lay.strata if s.role == "dequant"] for lay in layouts]
    n = sum(len(s.units) for ss in ys for s in ss)
    if n <= 0:
        raise ValueError("a window has y cells (0 < N, the window pin's hN): this one has none")
    out = []
    for ss in ys:
        f = dict(floors) | {s.name: -(-k_y * len(s.units) // n) for s in ss}
        out.append(WorkLaw(k, window_work=total, floors=tuple(f.items())))
    return out


@dataclass(frozen=True)
class Layout:
    """The units a plan reads: every unit, the tile and rest strata, and each tile's closure."""

    shape: Shape
    units: tuple[Unit, ...]
    strata: tuple[StratumSpec, ...]
    closure: Mapping[int, tuple[int, ...]]              # tile unit -> the call's units proved with it (itself first)
    y_strip: Mapping[int, int] = None                   # tile unit -> the weight Program's Y unit it reads (column strip)
    b_rows: Mapping[int, tuple[int, ...]] = None        # tile unit -> the weight Program's row units it reads (`tile_layout`)
    w_ref: int | None = None                            # one tile's W_ref, where the scheme's is given (`tile_layout`)

    def tile_work(self) -> int:
        """One tile's W_ref: ``w_ref``, else its cells' share of the call's (`schemes.ncp.NCP.wref`)."""
        if self.w_ref is not None:
            return self.w_ref
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
    return Layout(shape, us, template_strata(us, Layout(shape, us, (), {}).tile_work()), closure, y_strip)


def tile_layout(shape, w_ref: int) -> Layout:
    """A tile check's layout (`pc4.Shape` or `pc8.Shape`), from the shape alone: the call's units and strata, each tile's
    closure (itself, then its rows of A's units: TM, or a short last tile row's r; and under pc8's cap, on whole tiles
    only, its strip, its TM·TN word units row by row, its TM rows' digest and tally units and its tally, so every cap word
    is proved with its tile, never sampled) and its TN rows of B's units in the weight's Program, at W_ref ``w_ref`` a
    whole tile.  A row's units are its `RowA` (`RowB`) unit, or with ``row_k`` its head, forms and tail (`rowk`).  A short
    tile is its own template's stratum at its rows' share of that, w_ref·r/TM, which must be whole; a call with whole and
    short tiles is then two tile strata, which the work law refuses (`check_work_strata`)."""
    from . import pc4, pc8

    mod = pc8 if isinstance(shape, pc8.Shape) else pc4
    us, ws = mod.units(mod.call(shape)), mod.units(mod.weight(shape))
    tm, tn = shape.tile
    row_a: dict[int, list[int]] = {}
    row_b: dict[int, list[int]] = {}
    for rows, side, units_ in ((row_a, "row_a", us), (row_b, "row_b", ws)):
        for u in units_:
            if u.role.startswith(side):
                rows.setdefault(u.where[0], []).append(u.index)
    strip = {u.where[0]: u.index for u in us if u.role == "strip"}
    word, row_digest, row_tally, tally = ({u.where: u.index for u in us if u.role == r}
                                          for r in ("word", "row_digest", "row_tally", "tally"))
    closure, b_rows = {}, {}
    for t in (u for u in us if u.role == "tile"):
        rs, cs = t.where
        closure[t.index] = (t.index, *(a for i in range(rs * tm, min(shape.m, (rs + 1) * tm)) for a in row_a[i]))
        if getattr(shape, "cap", False):
            closure[t.index] += (strip[rs], *(word[rs * tm + i, cs * tn + j] for i in range(tm) for j in range(tn)),
                                 *(row_digest[rs * tm + i, cs] for i in range(tm)),
                                 *(row_tally[rs * tm + i, cs] for i in range(tm)), tally[rs, cs])
        b_rows[t.index] = tuple(b for j in range(tn) for b in row_b[cs * tn + j])
    strata, r = template_strata(us, w_ref), shape.m % tm
    if r:
        if w_ref * r % tm:
            raise DomainError(f"a short tile's W_ref is {r}/{tm} of a whole tile's, {w_ref}: not an integer")
        short = {u.template for u in us if u.role == "tile" and u.where[0] == shape.m // tm}
        strata = tuple(replace(s, work=w_ref * r // tm * len(s.units)) if s.name in short else s for s in strata)
    return Layout(shape, us, strata, closure, None, b_rows, w_ref)


def template_strata(us: Sequence[Unit], per_tile: int) -> tuple[StratumSpec, ...]:
    """The verifier's strata (`Draw.strataOf`): one per template, in the order of its first unit, the tile's with W_ref
    ``per_tile`` a unit and every other's with none."""
    groups: dict[str, list[Unit]] = {}
    for u in us:
        groups.setdefault(u.template, []).append(u)
    return tuple(StratumSpec(t, g[0].role, tuple(u.index for u in g), per_tile * len(g) if g[0].role == "tile" else 0)
                 for t, g in groups.items())


@dataclass(frozen=True)
class Plan:
    law: str
    sizes: Mapping[str, int]
    drawn: tuple[int, ...]                   # the units the law drew (tiles and floor units)
    proved: tuple[int, ...]                  # the sampled proof blocks and every drawn tile's closure, ascending, each once
    openings: tuple[tuple[str, int], ...]    # the leaves the plan opens: ("a", row), ("b", weight row), ("y", column strip)
    proved_y: tuple[int, ...] = ()           # the weight Program's Y units the drawn tiles read (column strips), each once
    proved_b: tuple[int, ...] = ()           # a tile check's: the weight Program's row units the drawn tiles read, each once


def draw(key: Key, lay: Layout, law: TileLaw, call: Sequence[int] = ()) -> Plan:
    """The plan for one sample: a uniform k_s-subset of each stratum from ``key`` (the verifier's own randomness, derived under
    `DRAW_DOMAIN` after the receipt), then every drawn tile's closure.  ``call`` is the call's index (`anchors`'
    `call_index`), which keys each stratum's subset with its template: the strata are the call's own (one per template,
    X-SPC-84), and the calls of a window drawn under one key draw independently (X-SPC-96).  The verifier samples through
    `Traced.draw`, over its own layout."""
    lay.check()
    sizes = law.sizes(lay.strata)
    drawn: list[int] = []
    for s in lay.strata:
        context = ("call", *call, "stratum", s.name) if call else ("stratum", s.name)
        drawn += [s.units[i] for i in key.subset(sizes[s.name], len(s.units), context)]
    proved = set(drawn)
    ys = {lay.y_strip[u] for u in drawn if lay.y_strip and u in lay.y_strip}
    bs = {b for u in drawn if lay.b_rows and u in lay.b_rows for b in lay.b_rows[u]}
    for u in drawn:
        proved.update(lay.closure.get(u, ()))
    return Plan(law.name, sizes, tuple(sorted(drawn)), tuple(sorted(proved)), openings(lay, proved, ys), tuple(sorted(ys)),
                tuple(sorted(bs)))


def work_table(lay: Layout, law: WorkLaw = WorkLaw()) -> dict[str, dict[str, int]]:
    """The verifier's work table (#362's `--work-table`, `Draw.workTableOfJson`): every template of the call, with its W_ref
    per unit (0 for units that do no PoUW work) and its floor f_s (`law.floors`). It refuses a call whose work is in more
    than one stratum (`check_work_strata`, the window pin's `hone`)."""
    check_work_strata(lay.strata)
    return {s.name: {"work": s.work // len(s.units), "floor": law.floor(s.name)} for s in lay.strata}


def law_object(lay: Layout, law: WorkLaw) -> dict:
    """The law `flock-verify draw --program P --partition Q --work K --work-table F` derives for this call (#362's
    `Draw.deriveWork`, as `Law.toJson` writes it): each stratum `{template, units, work, floor, k}`, at the call's K
    (`WorkLaw.budget`)."""
    sizes, table, k = law.sizes(lay.strata), work_table(lay, law), law.budget(lay.strata)
    return {"law": "work", "k": k, "strata": [{"template": s.name, "units": s.ranges(), "work": table[s.name]["work"],
                                               "floor": table[s.name]["floor"], "k": sizes[s.name]} for s in lay.strata]}


def openings(lay: Layout, proved, proved_y=()) -> tuple[tuple[str, int], ...]:
    """The leaves a plan opens: a digest or X unit's strip's 16 A rows, one leaf per row; a Y unit's (the weight Program's,
    by column strip) 16 weight rows; and at each such column strip, ("y", strip), the call's Y and c₀ input there and the
    weight Program's output there, which the verifier compares (`Traced.check_link`, X-SPC-95)."""
    rows: set[tuple[str, int]] = set()
    for i in proved:
        u = lay.units[i]
        if u.role in ("digest", "form_x"):
            rows.update(("a", D * u.where[0] + r) for r in range(D))
    for cs in proved_y:
        rows.update(("b", D * cs + r) for r in range(D))
        rows.add(("y", cs))
    return tuple(sorted(rows))


def profile(lay: Layout, sizes: Mapping[str, int]) -> Stratified:
    """Core's `Stratified` over the tile strata the plan drew."""
    return Stratified(tuple(Stratum(s.name, len(s.units), sizes[s.name]) for s in lay.strata if s.role == "tile"))


def unsound_work_bound(lay: Layout, sizes: Mapping[str, int], delta: float) -> float:
    """`harm_bound` over the tile strata at harm W_ref per tile (its stratum's): except with probability δ (plus the
    proofs' error), an accepted plan leaves at most this much work in unsound tiles."""
    return profile(lay, sizes).harm_bound({s.name: s.work / len(s.units) for s in lay.strata if s.role == "tile"}, delta)


__all__ = ["DRAW_DOMAIN", "RECORD_K", "CountLaw", "Layout", "Plan", "StratumSpec", "TileLaw", "WorkLaw", "check_work_strata",
           "draw", "law_object",
           "layout", "openings", "profile", "record_k", "template_strata", "tile_layout", "unsound_work_bound", "window_laws",
           "work_table"]
