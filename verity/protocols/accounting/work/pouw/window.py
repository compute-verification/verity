"""PoUW's audit window as the proof service sees it: what the window makes public, the grid its tile count is padded to,
the size of the uniform draw, and the window's advice (its per-call entries and the map from a drawn index to a tile),
with the window statement's reference (`note:verity-root/20261006T0545Z-draft-daniel-one-recursive-architecture-v2`,
PoUW; the joint note's L2, L3 and L4).

**Public items** (`PUBLIC`), exactly the draft's list.  The service refuses a registration that makes anything else public;
the window's own root and W's root are salted roots, which the rule allows.

    item       kind         source     what it is
    verdict    output       outcome    every drawn tile proved and opened, or not
    work       output       window     the credited work, as the tile count N padded to a grid level (`bucket`)
    check      structure    parameter  the tile template every unit instantiates, its work per tile, the commitment scheme
    sampling   structure    parameter  ε, δ, the grid and the law subset:K′ they fix with the work level (`k_prime`)
    drawn      coins        outcome    the verifier's own draw, after the receipt
    root_W     commitment   parameter  the weights' salted root

**The grid** (`GRID_BITS` = s): the positive integers with at most s + 1 significant bits, so consecutive points above
2^(s+1) differ by a factor of at most 1 + 2^-s (9/8 at s = 3).  A window with N_real real tiles registers one level ℓ, the
largest point g_ℓ <= N_real: it is credited g_ℓ tiles' work and pads its tiles to N = g_(ℓ+1) > N_real with filler tiles,
at least one.  So N is `pad(N_real + 1)`, the least grid point at or above the real tiles and the one filler a window
always has: the private partition's unit-count bucket n_b (`note:circuits/20261006T0615Z-draft-private-partitions`,
G6) on the same grid, one bucket and not two.  Both outputs are fixed by ℓ, so the window's influenceable public output is one grid level, and over a
declared range [lo, hi] it takes `levels(lo, hi)` = q values: log2 q bits in the draft's leakage bound.  The honest cost is
under 1 − 2^-s of the credit (at most 11% at s = 3) and the factor N / g_ℓ <= 1 + 2^-s in K′.

**The draw** (L3): uniform subset:K′ over the N equal tiles, K′ the smallest K with (1 − ε·W/(N·w_max))^K <= δ, W the
credited work and w_max the most work a tile does (`k_prime`).  A wrong set holding a share ε of W has at least ε·W/w_max
tiles, so a draw of K′ without replacement misses it with probability at most δ.  Filler tiles are credited nothing.

**The advice** (L2): one entry per committed call, then one filler entry, then empty entries up to the public bound, each
committed in one window root registered before the draw.  A call entry holds its sides' roots, row counts and the rows its
tiles keep (a row the scheme refused is no tile's), and its tile range [lo, hi); its tiles are the kept rows' products,
row-major.  The filler entry holds zero rows on both sides and the range [N_real, N).  `tile_at` maps a drawn index to its
entry and rows, and `window_problem` is the window statement (L4) in Python: the ranges tile [0, N) in order, each call's
range is its kept tiles, the filler covers the rest, and the registered level is `bucket` of the real tiles.  It is not
proved yet: no Program states it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

GRID_BITS = 3

#: (item, kind, source, reason): the window's public items, in the service's `Public` form
PUBLIC = (
    ("verdict", "output", "outcome", "what the credit rests on: every drawn tile was proved and opened, or not"),
    ("work", "output", "window", "the credited work, as the tile count padded to a grid level: the draw ranges over its N "
                                 "tiles and the auditor credits its work; one level, so at most log2 q bits"),
    ("check", "structure", "parameter", "the check's identity: the tile template, its work per tile and the commitment "
                                        "scheme fix the auditor's verifier"),
    ("sampling", "structure", "parameter", "ε, δ and the grid, and the law subset:K′ they fix with the level: the guarantee "
                                           "is stated in them"),
    ("drawn", "coins", "outcome", "the verifier's own draw, after the receipt"),
    ("root_W", "commitment", "parameter", "the weights' salted root: what the tiles' weight rows are committed to, hiding"),
)
#: the items a registration carries; the verdict and the draw come after it
REGISTERED = ("work", "check", "sampling", "root_W")
KINDS = ("call", "filler", "none")


# ---- the grid ---------------------------------------------------------------------------------------------------------------

def _shift(n: int, bits: int) -> int:
    return max(0, n.bit_length() - bits - 1)


def _positive(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


def on_grid(n: Any, bits: int = GRID_BITS) -> bool:
    return _positive(n) and n & ((1 << _shift(n, bits)) - 1) == 0


def floor_point(n: int, bits: int = GRID_BITS) -> int:
    """The largest grid point at most n >= 1."""
    if not _positive(n):
        raise ValueError(f"the grid is over positive integers, not {n!r}")
    s = _shift(n, bits)
    return n >> s << s


def pad(n: int, bits: int = GRID_BITS) -> int:
    """The least grid point at least n >= 1: a unit count's public bucket."""
    g = floor_point(n, bits)
    return g if g == n else point(level(g, bits) + 1, bits)


def level(g: int, bits: int = GRID_BITS) -> int:
    """Grid point g's index, from 1 at level 0."""
    if not on_grid(g, bits):
        raise ValueError(f"{g!r} is not a point of the {bits}-bit grid")
    small = 1 << (bits + 1)
    if g < small:
        return g - 1
    e = _shift(g, bits)
    return small - 1 + (e - 1) * (1 << bits) + (g >> e) - (1 << bits)


def point(ell: int, bits: int = GRID_BITS) -> int:
    """The grid point at level ell (`level`'s inverse)."""
    if not isinstance(ell, int) or isinstance(ell, bool) or ell < 0:
        raise ValueError(f"a level is a natural number, not {ell!r}")
    small = 1 << (bits + 1)
    if ell < small - 1:
        return ell + 1
    r = ell - (small - 1)
    return ((1 << bits) + r % (1 << bits)) << (r // (1 << bits) + 1)


def levels(lo: int, hi: int, bits: int = GRID_BITS) -> int:
    """q: the grid levels a window's tile count in [lo, hi] can register."""
    return level(floor_point(hi, bits), bits) - level(floor_point(lo, bits), bits) + 1


@dataclass(frozen=True)
class Work:
    """A window's work level: it is credited `credited` tiles, point(level), and pads to `n` tiles, point(level + 1)."""
    level: int
    credited: int
    n: int

    @classmethod
    def at(cls, ell: int, bits: int = GRID_BITS) -> "Work":
        return cls(ell, point(ell, bits), point(ell + 1, bits))

    def to_json(self, tile_work: int) -> dict[str, int]:
        return {"level": self.level, "n": self.n, "credited_tiles": self.credited, "credited": self.credited * tile_work}


def bucket(real: int, bits: int = GRID_BITS) -> Work:
    """The level of a window of `real` real tiles: credited floor_point(real), padded to the next point."""
    return Work.at(level(floor_point(real, bits), bits), bits)


# ---- the draw -----------------------------------------------------------------------------------------------------------

def k_prime(eps: Fraction, delta: Fraction, credited: int, n: int, w_max: int) -> int:
    """L3's K′: the smallest K with (1 − ε·credited/(n·w_max))^K <= δ, at most n (`credited` the credited work, `n` the tiles,
    each doing at most `w_max`).  Computed in binary64 logarithms, then K stepped until the inequality holds there."""
    eps, delta = Fraction(eps), Fraction(delta)
    if not (0 < eps <= 1 and 0 < delta < 1 and _positive(n) and _positive(w_max) and 0 < credited <= n * w_max):
        raise ValueError(f"k_prime needs 0 < ε <= 1, 0 < δ < 1 and 0 < credited <= n·w_max, not {eps}, {delta}, "
                         f"{credited}, {n}, {w_max}")
    p = eps * credited / (n * w_max)
    if p == 1:
        return 1
    step, target = math.log1p(-float(p)), math.log(float(delta))
    k = max(1, math.ceil(target / step))
    while k > 1 and (k - 1) * step <= target:
        k -= 1
    while k * step > target:
        k += 1
    return min(k, n)


def sampling(eps: Fraction, delta: Fraction, work: Work, tile_work: int, bits: int = GRID_BITS) -> dict[str, Any]:
    """The `sampling` item: ε, δ, the grid and the law they fix with the level (every tile does `tile_work`)."""
    k = k_prime(eps, delta, work.credited * tile_work, work.n, tile_work)
    return {"eps": str(Fraction(eps)), "delta": str(Fraction(delta)), "grid_bits": bits, "law": {"law": "subset", "k": k}}


# ---- the advice and the window statement --------------------------------------------------------------------------------

def tiles(entry: Mapping[str, Any]) -> int:
    return len(entry["a"]["keep"]) * len(entry["b"]["keep"]) if entry["kind"] != "none" else 0


def tile_in(entry: Mapping[str, Any], t: int) -> tuple[int, int]:
    """(A row, B row) of index t in its entry's range: the kept rows' products, row-major."""
    if entry["kind"] == "none" or not entry["lo"] <= t < entry["hi"]:
        raise ValueError(f"index {t} is not in the entry's range")
    u, nb = t - entry["lo"], len(entry["b"]["keep"])
    return entry["a"]["keep"][u // nb], entry["b"]["keep"][u % nb]


def tile_at(entries: Sequence[Mapping[str, Any]], t: int) -> tuple[int, int, int]:
    """(entry, A row, B row) of drawn index t."""
    for e, entry in enumerate(entries):
        if entry["kind"] != "none" and entry["lo"] <= t < entry["hi"]:
            return (e, *tile_in(entry, t))
    raise ValueError(f"index {t} is in no entry's range")


def _keep_ok(side: Any) -> bool:
    if not isinstance(side, Mapping) or not _positive(side.get("rows")) or not isinstance(side.get("keep"), list):
        return False
    keep = side["keep"]
    return bool(keep) and all(isinstance(x, int) and 0 <= x < side["rows"] for x in keep) and keep == sorted(set(keep))


def window_problem(entries: Sequence[Mapping[str, Any]], work: Work, bound: int, bits: int = GRID_BITS) -> str | None:
    """The window statement (L4) on the advice, or None: `bound` entries, the calls, then one filler, then empty ones; the
    ranges tile [0, work.n) in order; a call's range is its kept tiles; the filler's holds the rest, within its rows; and
    the level is `bucket` of the calls' tiles.  Credited tiles are only the calls'."""
    if len(entries) != bound:
        return f"{len(entries)} entries, not the bound {bound}"
    kinds = [e.get("kind") for e in entries]
    if any(k not in KINDS for k in kinds):
        return f"an entry's kind is none of {KINDS}"
    calls = kinds.index("filler") if "filler" in kinds else -1
    if calls < 1 or kinds[:calls] != ["call"] * calls or kinds[calls + 1:] != ["none"] * (bound - calls - 1):
        return "the entries are the calls, then one filler, then empty entries"
    at = 0
    for e, entry in enumerate(entries[:calls + 1]):
        if not (_keep_ok(entry.get("a")) and _keep_ok(entry.get("b"))):
            return f"entry {e}: its sides' kept rows are not ascending rows of the side"
        if entry.get("lo") != at or not isinstance(entry.get("hi"), int) or entry["hi"] <= at:
            return f"entry {e}: its range does not start where the last ended, or is empty"
        size = entry["hi"] - entry["lo"]
        if entry["kind"] == "call" and size != tiles(entry):
            return f"entry {e}: its range is {size} tiles, not its {tiles(entry)} kept tiles"
        if entry["kind"] == "filler" and size > tiles(entry):
            return f"entry {e}: the filler's range is {size} tiles, beyond its {tiles(entry)}"
        at = entry["hi"]
    real = entries[calls]["lo"]
    if at != work.n:
        return f"the ranges end at {at}, not the padded count {work.n}"
    if bucket(real, bits) != work:
        return f"{real} real tiles are level {bucket(real, bits).level}, not the registered level {work.level}"
    return None


__all__ = ["GRID_BITS", "KINDS", "PUBLIC", "REGISTERED", "Work", "bucket", "floor_point", "k_prime", "level", "levels",
           "on_grid", "point", "sampling", "tile_at", "tile_in", "tiles", "window_problem"]
