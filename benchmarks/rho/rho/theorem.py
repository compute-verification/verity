"""The slowdown theorem's shape: what every bound, attack and datacenter instantiates.

Training is charged by its *counted products*: the multiplications of its reference computation, `κ/2` per FLOP. A
**block** is a part of the actual circuit with a **budget**: the FLOPs it runs and the novel bits that enter it (what it
didn't register). A bound caps the counted products one block can finish given its budget, an attack is a construction
that finishes that many: either is an **allowance**. A **region** is a part of the monitored fleet over the horizon (one
location over one window), and a **cut** of it is one way of covering it with blocks plus *free* FLOPs, which no budget
constrains and which count at face value. Every cut is an upper bound, so under a bound a region contributes the least
of its cuts, and never more than its own compute. An attack is a construction, so it is evaluated on the region's
*layout* instead: the blocks and free FLOPs a strategy that obeys every protocol actually gets there (no layout: the
region is unconstrained). The dark pool's share `ε_dark` of the adversary's training rate runs unencumbered:

    g = Σ_R n_R · min(κ C_R / 2, min_cuts Σ_B n_B · min(a(B), κ C_B / 2) + κ · free / 2) / Σ_R n_R · κ C_R / 2

(an attack's `g` takes the layout in place of the least cut)
    ρ̄ = μ / (ε_dark + (1 − ε_dark) · g)

where `μ` is the compute the algorithm needs to reach the target, in multiples of the reference algorithm's (the
working draft's `|C_{0,m}(G)|/T_0`), and the other unaccounted compute (`ε_undecl`, `ε_gap`, `ε_skip`, `ε_break`) is
the free FLOPs inside `g`.

This module knows no GPU, protocol or training class: `regimes` builds the partition from the actual datacenter,
`classes` and `attacks` give allowances, and `families` the algorithm's profile.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Budget:
    """What one block may use: the FLOPs it runs, the novel bits that enter it, and the bits that may cross its
    narrowest declared forward cut (`math.inf` where the policy declares none: an NCI block; an episode of the proof of
    forward computation, whose operators form a one-way chain, has `n_max` positions times the write allowance)."""

    compute: float
    inputs: float
    cut: float = math.inf


#: The most counted products one block with this budget finishes: a cap for a bound, a construction for an attack.
#: `math.inf` is no cap (a bound whose premises fail at this budget).
Allowance = Callable[[Budget], float]


@dataclass(frozen=True)
class Blocks:
    """`count` blocks of one kind in a region, each with `budget`."""

    kind: str
    budget: Budget
    count: float


@dataclass(frozen=True)
class Cut:
    """One way of covering a region: its budgeted blocks, and its free FLOPs by source, valid under `cites` (the Lean
    guarantees and premises that make every FLOP of the region either in a block or free)."""

    name: str
    blocks: tuple[Blocks, ...] = ()
    free: Mapping[str, float] = field(default_factory=dict)
    cites: tuple[str, ...] = ()


@dataclass(frozen=True)
class Region:
    """`count` regions of one kind over the horizon, each running `compute` FLOPs when unencumbered: its cuts (upper
    bounds, for bounds) and its layout (what a protocol-abiding strategy gets, for attacks; None: unconstrained)."""

    name: str
    compute: float
    count: float
    cuts: tuple[Cut, ...]
    layout: Cut | None = None


@dataclass(frozen=True)
class Partition:
    """The monitored fleet over the horizon as regions, and `ε_dark`, the dark pool's share of the adversary's training
    rate outside them."""

    regions: tuple[Region, ...]
    eps_dark: float


@dataclass(frozen=True)
class Share:
    """What one region contributes, as a share of its unencumbered products: the total, the cut that set it, and the
    cut's terms (by block kind and free source)."""

    value: float
    cut: str
    terms: Mapping[str, float]


Side = Literal["bound", "attack"]


def share(region: Region, allowance: Allowance, kappa: float, side: Side = "bound") -> Share:
    """Under a bound, the least of `region`'s cuts; under an attack, its layout; never more than its own compute."""
    full = kappa * region.compute / 2
    best = Share(1.0, "compute", {"compute": 1.0})
    if full <= 0:
        return best
    for cut in region.cuts if side == "bound" else (region.layout,) if region.layout else ():
        terms = {b.kind: b.count * min(allowance(b.budget), kappa * b.budget.compute / 2) / full for b in cut.blocks}
        for source, flops in cut.free.items():
            terms[source] = terms.get(source, 0.0) + kappa * flops / 2 / full
        total = math.fsum(terms.values())
        if total < best.value:
            best = Share(total, cut.name, terms)
    return best


@dataclass(frozen=True)
class Slowdown:
    """ρ for one algorithm under one allowance: `rho = efficiency / (ε_dark + (1 − ε_dark) g)`, with each region's share
    and `g`'s terms summed over regions (each a share of the monitored fleet's unencumbered products)."""

    rho: float
    g: float
    eps_dark: float
    efficiency: float
    regions: Mapping[str, Share]
    terms: Mapping[str, float]

    @property
    def monitored(self) -> float:
        """The share of the adversary's training done on monitored compute: the protocols do their job for this
        algorithm when it is small (`g ≪ d`), and not when it is near 1."""
        rate = self.eps_dark + (1 - self.eps_dark) * self.g
        return (1 - self.eps_dark) * self.g / rate if rate > 0 else 0.0


def slowdown(partition: Partition, allowance: Allowance, kappa: float, efficiency: float = 1.0,
             side: Side = "bound") -> Slowdown:
    weights = {r.name: r.count * r.compute for r in partition.regions}
    total = math.fsum(weights.values())
    shares = {r.name: share(r, allowance, kappa, side) for r in partition.regions}
    terms: dict[str, float] = {}
    for name, s in shares.items():
        for term, v in s.terms.items():
            terms[term] = terms.get(term, 0.0) + weights[name] * v / total
    g = math.fsum(weights[n] * s.value for n, s in shares.items()) / total if total else 0.0
    return Slowdown(rho(partition.eps_dark, g, efficiency), g, partition.eps_dark, efficiency, shares, terms)


def rho(eps_dark: float, g: float, efficiency: float = 1.0) -> float:
    """The slowdown when an `ε_dark` share of the training rate runs unencumbered and the rest at `g` of its
    unencumbered rate."""
    rate = eps_dark + (1 - eps_dark) * g
    return efficiency / rate if rate > 0 else math.inf


def least(*allowances: Allowance) -> Allowance:
    """Several bounds on the same block at once: each caps the same counted products, so the least of them does, under
    the union of their premises."""
    return lambda budget: min((a(budget) for a in allowances), default=math.inf)
