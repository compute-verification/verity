"""A member's link points (Lean `SampledProofs.LinkPoints`, which the difftest's `link_points` piece holds this to).

A session of a member's statement opens its link at `PT_LOCAL + nbl(g, c)` points for its `c` proved units and its
circuit's `g` units a proof block (META `vus_per_block`). A draw over several members draws a random count of each
member's units, so the registration of the per-member form refuses a member whose fewest and most drawn units give
different link points (`refused`, the registration's `link-points`): at every draw the sample can be, the deployed
draw's fallback included (one that runs out of coins draws every unit), the member's count lies between its fewest and
most (`bounds`), and its statement's whole public file, exactly its units, has the most.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from verity.ml.circuits.program import Program

from ..coins import law_problem
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


def fewest(law: Mapping[str, Any], population: int, us: Sequence[Sequence[int]]) -> int:
    """The fewest of the units `us` of `[0, population)` a draw under `law` (well formed: `coins.law_problem`) holds when
    it ends: what is left after its units fill the rest of the population, or of each stratum, first; a member's whole
    count where the draw never ends, so only the fallback's every unit is a sample."""
    n = size(us)
    if law["law"] == "subset":
        k = law["k"]
        return max(0, k - (population - n)) if k <= population else n
    if law["law"] == "bernoulli":
        num, den = (int(x) for x in law["p"].split("/"))
        return n if num != 0 and den <= num else 0
    strata = law["strata"]
    if any(size(s["units"]) < s["k"] for s in strata):
        return n
    return sum(max(0, s["k"] - (size(s["units"]) - overlap(us, s["units"]))) for s in strata)


def bounds(law: Mapping[str, Any], population: int, members: Sequence[Member]) -> list[tuple[int, int]]:
    """Each member's fewest and most drawn units: a member is its units, `[lo, hi)` ranges, and its circuit's units a
    proof block. ValueError for a member with no unit, a range of none, ranges that meet, a unit outside the population,
    or no unit a proof block."""
    out = []
    for i, (us, g) in enumerate(members):
        if g == 0:
            raise ValueError(f"link points: member {i}: its vus_per_block is 0: a proof block holds at least one unit")
        rs = sorted((lo, hi) for lo, hi in us)
        if (not rs or any(hi <= lo or population < hi for lo, hi in rs)
                or any(rs[j + 1][0] < rs[j][1] for j in range(len(rs) - 1))):
            raise ValueError(f"link points: member {i}: its units are not nonempty, disjoint [lo, hi) ranges of the "
                             f"population's {population}")
        out.append((fewest(law, population, us), size(us)))
    return out


def refused(law: Mapping[str, Any], population: int, members: Sequence[Member]) -> list[int]:
    """The members whose fewest and most drawn units give different link points (`bounds`), by index."""
    return [i for i, ((_, g), (lo, hi)) in enumerate(zip(members, bounds(law, population, members)))
            if nbl(g, lo) != nbl(g, hi)]


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
