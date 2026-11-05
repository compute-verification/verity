"""Attacks: constructions that finish counted products in a block, reported beside the overhead's lower bound.

An attack is an `Argument` on the `attack` side: `allowance(profile)` is what the construction finishes in a block with
that budget (never more than the block's compute, which `theorem.terms` caps), in the FLOP model as the classes are
(`classes.FlopBudget`). Its `cites` are the analyses it rests on (`draft:`). The best attack for an algorithm is its
best known attack (`FamilyResult.best_attack`), a construction and not an end of an interval; a bound above an attack's
achieved count means a premise is false for that algorithm at that budget, and `scenario.evaluate` reports it as a
contradiction.

Every allowance here is achieved: a lower bound on the most a block can finish, never an estimate of it. Where a
construction's count is of another set than the profile's `κ` (the tile's forward products), it is converted as a
class's is (`classes.counted`), which assumes the rest of the counted products tile at the same rate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Literal

from rho.classes import (
    Allowance,
    Argument,
    FlopBudget,
    Outside,
    counted,
    need,
)
from rho.profile import Profile


@dataclass(frozen=True)
class Comply:
    """Run the registered work as registered and train only on free compute: always available."""

    name: str = "comply"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ()

    def allowance(self, p: Profile) -> Allowance:
        return lambda budget: 0.0


@dataclass(frozen=True)
class StateFits:
    """A block whose novel input holds the whole trained state trains unthrottled (the small-state exception, N1): it
    reads the state in, runs the reference algorithm on its compute, and leaves the state behind for the next."""

    name: str = "state-fits"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ("draft:N1",)

    def allowance(self, p: Profile) -> Allowance:
        def a(budget: FlopBudget) -> float:
            return p.kappa * budget.compute / 2 if p.state_bits <= budget.inputs else 0.0
        return a


@cache
def phis() -> tuple[float, ...]:
    """`benchmarks/nci/slowdown.py`'s grid of slices."""
    return (*(10 ** (-12 + 12 * i / 2400) for i in range(2400)), 1.0)


@lru_cache(maxsize=1 << 16)
def search(state: float, splits: bool, flops: float, cut: float, data: float, decode: float, k: float, compute: float,
           shared: float = math.inf) -> float:
    """The most reference work one block does by `benchmarks/nci/slowdown.py`'s `search`: it covers a slice, a fraction
    `φ` of every token's computation, for `T` tokens, reading the slice's state (`φ U`, or all of `U` when it doesn't
    split), the `cut` bits per token crossing into it (none at `φ = 1`) and `data` novel bits per token, and decoding
    its slice (`φ decode`). Only slices within the largest contiguous run every token shares (`φ U ≤ shared`) are taken,
    so that `T` tokens hold no more of it and cross only at its two ends. 0 when no slice fits. One sequential chain of
    blocks, as there: the state carries what a block hands on inside a step (`Profile.token_state_bits`)."""
    best = 0.0
    for phi in phis() + ((k / (2 * state),) if splits and state > k / 2 else ()):
        held = phi * state if splits else state
        if held >= k or phi * decode >= compute or (splits and held > shared):
            continue
        per_token = (cut if phi < 1 else 0.0) + data
        tokens = min((k - held) / per_token if per_token > 0 else math.inf, (compute - phi * decode) / (phi * flops))
        best = max(best, phi * flops * tokens)
    return best


@dataclass(frozen=True)
class Shard:
    """`benchmarks/nci`'s construction family (`search`): the best slice of every token's computation a block can hold,
    for as many tokens as its reads and compute allow. Its counted products are `κ/2` of the reference work."""

    name: str = "shard"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ("draft:nci-search",)

    def allowance(self, p: Profile) -> Allowance:
        u, splits, flops, cut, data, decode, shared = need(
            p, "token_state_bits", "state_splits", "flops_per_token", "cut_bits", "token_bits", "decode_flops",
            "shared_state_bits")
        memo: dict[FlopBudget, float] = {}

        def a(budget: FlopBudget) -> float:
            if budget not in memo:
                memo[budget] = p.kappa / 2 * search(u, splits, flops, cut, data, decode, budget.inputs, budget.compute,
                                                     shared)
            return memo[budget]
        return a


@lru_cache(maxsize=1 << 16)
def tile(k: float, weight_bits: float, beta_x: float, psum_bits: float, rows: float, cols: float,
         grid: int = 400) -> float:
    """The column tile (E1, Proposition 3's tightness) completed with its partial sums: `r` input coordinates by `w`
    output slots of one matrix at `Q` tokens, reading `r w` weights, `r Q` activation occurrences and the `w Q` partial
    sums it adds to, `r w p + r Q β_x + w Q ψ ≤ k`, with `r ≤ rows` and `w ≤ cols`. Its `r w Q` products, maximized over
    whole `r`, `w`, `Q` (each `r` on a geometric grid, its best `w` in closed form)."""
    best = 0.0
    for i in range(grid + 1):
        r = math.floor(rows ** (i / grid))
        a, b = r * weight_bits, r * beta_x
        if a + b + psum_bits >= k:
            break
        w = min(math.floor(b / psum_bits * (math.sqrt(1 + psum_bits * k / (a * b)) - 1)), math.floor(cols))
        w = max(w, 1)
        q = math.floor((k - a * w) / (b + psum_bits * w))
        if q >= 1:
            best = max(best, r * w * q)
    return best


@dataclass(frozen=True)
class Tile:
    """Every block a tile (`tile`) of one matrix's forward products, the tiles of a matrix chained along its inputs so
    that each hands its partial sums to the next: E1's column tile at the margin, paying for its partial sums."""

    name: str = "tile"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ("draft:E1-tile",)

    def allowance(self, p: Profile) -> Allowance:
        values = need(p, "tile_weight_bits", "beta_x_marginal", "psum_bits", "tile_inputs", "tile_outputs")
        scale = counted(p, "kappa_forward")
        return lambda budget: tile(budget.inputs, *values) * scale


@dataclass(frozen=True)
class Point:
    """A measured execution's block: `bits` novel bits and `flops` FLOPs (`None`: any compute), which did the
    reference work of `flops / rho` FLOPs (`rho` of the block's own compute when `flops` is `None`)."""

    bits: float
    flops: float | None
    rho: float


@dataclass(frozen=True)
class Measured:
    """Measured executions, extended monotonically: a block runs as many of a measured block as its bits and compute
    both hold (`⌊k / k_m⌋` and `⌊C / C_m⌋`), and finishes `κ/2` of their reference work. Measured on `families` only."""

    name: str
    families: tuple[str, ...]
    points: tuple[Point, ...]
    cites: tuple[str, ...]
    side: Literal["bound", "attack"] = field(default="attack")

    def allowance(self, p: Profile) -> Allowance:
        if p.name not in self.families:
            raise Outside(f"{self.name} was measured on {', '.join(self.families)}, not {p.name}")

        def a(budget: FlopBudget) -> float:
            best = 0.0
            for pt in self.points:
                if pt.flops is None:
                    work = budget.compute / pt.rho if budget.inputs >= pt.bits else 0.0
                else:
                    work = min(budget.inputs // pt.bits, budget.compute // pt.flops) * pt.flops / pt.rho
                best = max(best, p.kappa * work / 2)
            return best
        return a


#: C1's cheapest executions at FP8 on V4's and K2's shapes, computed from B1's measured rates (not run at that scale):
#: the high end of each range, so that both models reach it
C1 = ("draft:C1-cheapest-execution",)
#: matched compute, `c = φ k` at `φ = 6.7·10⁹` FLOPs per bit (D1)
PHI = 6.7e9

ATTACKS: Mapping[str, Argument] = {a.name: a for a in (
    Comply(), StateFits(), Shard(), Tile(),
    Measured("measured-full", ("full",), (Point(2e7, PHI * 2e7, 6.8e5), Point(8e6, PHI * 8e6, 1.25e6)), C1),
    Measured("measured-frozen-prefix", ("frozen-prefix",),
             (Point(2e7, PHI * 2e7, 4.9e5), Point(8e6, PHI * 8e6, 1.22e6)), C1),
    Measured("measured-rl-elicitation", ("rl-elicitation",), (Point(7.84e4, None, 1.0),), ()),
)}
