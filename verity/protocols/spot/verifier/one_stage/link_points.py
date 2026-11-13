"""A member's link points (Lean `SampledProofs.LinkPoints`, which the difftest's `link_points` piece holds this to).

A session of a member's statement opens its link at `PT_LOCAL + nbl(g, c)` points for its `c` proved units and its
circuit's `g` units a proof block (META `vus_per_block`). A draw over several members draws a random count of each
member's units. One that runs out of the verifier's public coins fails the window (`OneStageZk.RAN_OUT`); one that ends
draws between the member's fewest and most units (`ended`, `bounds`). The registration of the per-member form refuses a
member whose fewest and most give different link points, and a law no draw of which ends (`refused`, `ends`, the
registration's `link-points`).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from verity.circuits.program import Program

from verity.protocols.spot.device.coins import law_problem
from . import draw as D, partition as P

Member = tuple[Sequence[Sequence[int]], int]


def nbl(g: int, c: int) -> int:
    """The link points beyond `PT_LOCAL` of a session of `c` proved units at `g` units a proof block:
    `max(3, ceil(log2(ceil(c / g))))`."""
    return max(3, (max((c + g - 1) // g, 1) - 1).bit_length())


def size(rs: Sequence[Sequence[int]]) -> int:
    return sum(hi - lo for lo, hi in rs)


def overlap(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int:
    """The units in both `a` and `b`, each `[lo, hi)` ranges disjoint among themselves."""
    return sum(max(0, min(hi, hi2) - max(lo, lo2)) for lo, hi in a for lo2, hi2 in b)


def ended(law: Mapping[str, Any], population: int, us: Sequence[Sequence[int]]) -> tuple[int, int] | None:
    """The fewest and most of the units `us` of `[0, population)` a draw under `law` (well formed: `coins.law_problem`)
    holds when it ends: the fewest is what is left after its units fill the rest of the population, or of each stratum,
    first, and the most `k`, or each stratum's `k_s`, where the member holds that many. None where no draw ends (`subset`
    above the population, a stratum drawing more than it has), so every window under `law` fails."""
    n = size(us)
    if law["law"] == "subset":
        k = law["k"]
        return (max(0, k - (population - n)), min(k, n)) if k <= population else None
    if law["law"] == "bernoulli":
        num, den = (int(x) for x in law["p"].split("/"))
        return (0, 0) if num == 0 else None if den == 0 else (n, n) if den <= num else (0, n)
    strata = law["strata"]
    if any(size(s["units"]) < s["k"] for s in strata):
        return None
    lo = hi = 0
    for s in strata:
        o = overlap(us, s["units"])
        lo += max(0, s["k"] - (size(s["units"]) - o))
        hi += min(s["k"], o)
    return lo, hi


def bounds(law: Mapping[str, Any], population: int, members: Sequence[Member]) -> list[tuple[int, int] | None]:
    """Each member's fewest and most drawn units when the draw ends (`ended`): a member is its units, `[lo, hi)` ranges,
    and its circuit's units a proof block. ValueError for a member with no unit, a range of none, ranges that meet, a
    unit outside the population, or no unit a proof block."""
    out = []
    for i, (us, g) in enumerate(members):
        if g == 0:
            raise ValueError(f"link points: member {i}: its vus_per_block is 0: a proof block holds at least one unit")
        rs = sorted((lo, hi) for lo, hi in us)
        if (not rs or any(hi <= lo or population < hi for lo, hi in rs)
                or any(rs[j + 1][0] < rs[j][1] for j in range(len(rs) - 1))):
            raise ValueError(f"link points: member {i}: its units are not nonempty, disjoint [lo, hi) ranges of the "
                             f"population's {population}")
        out.append(ended(law, population, us))
    return out


def ends(law: Mapping[str, Any], population: int) -> bool:
    """Whether some draw under `law` over `[0, population)` ends: whether `ended` gives counts, which reads only the law
    and the population, never the member's units."""
    return ended(law, population, ()) is not None


#: Lean `LinkPoints.NEVER_ENDS`: the refusal of a law no draw of which ends (`ends`).
NEVER_ENDS = "link points: no draw under the law ends, so each of its windows fails"


def refused(law: Mapping[str, Any], population: int, members: Sequence[Member]) -> list[int]:
    """The members whose fewest and most drawn units, over the draws that end, give different link points (`bounds`),
    by index; ValueError (`NEVER_ENDS`) when no draw under `law` ends."""
    bs = bounds(law, population, members)
    if not ends(law, population):
        raise ValueError(NEVER_ENDS)
    return [i for i, ((_, g), b) in enumerate(zip(members, bs)) if b is not None and nbl(g, b[0]) != nbl(g, b[1])]


def law_of(program: Program, query: Mapping[str, Any], law: Mapping[str, Any],
           table: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], int]:
    """The sampling design the draw runs on, the verifier's own expanded over its program and query (`draw.derive`),
    with the population; ValueError (or `partition.QueryRefused`) where it doesn't apply."""
    population = P.units(program, query)
    mine = D.derive(program, query, law, table)
    if mine.get("law") not in ("stratified", "work"):
        problem = law_problem(mine, population)
        if problem:
            raise ValueError(problem)
    return mine, population
