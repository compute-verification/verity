"""The exclusion bound: the serving-block bound with its local minima (the working draft's §8, which states it).

Everything here is in gates and counted weight. A **block** has a budget `(c, k)` (`Budget`): its gates and its
unregistered input bits. A training class gives a per-block cap `q_T(c, k)` (`Cap`), the most counted weight a block
with that budget holds; an attack gives a construction's count instead. A **serving block** (`ServingBlock`) is one
server between two consecutive wipes, padded to `c_S` gates; each `Cut` of it is one valid partition, its blocks plus
its uncertified gates `U` by route, and an argument of the bound's minimum:

* `serving`, one block `(c_S, k_S)`, its cap `q_T(c_S, k_S)`;
* `inference`, its `n_S` inference blocks `(c, k)` and `U_S`, so `n_S q_T(c, k) + q_U(U_S)`.

`q_U(U) = ω_max U / s_×` is the elementary uncertified cap, `u U` with `u` the most weight a gate can hold (`ω_max /
s_×`, `classes.Structure`), and no block holds more than `u c`. Under a bound each serving block contributes the least
of its cuts (the local minimum), and the dark leaves `q_U(|A[L_dark]|)`:

    Σ_v ω_v ≤ Σ_S min{q_T(c_S, k_S), n_S q_T(c, k) + q_U(S)} + q_U(dark)

With `aggregate=True` the minimum is of the global sums instead (each cut name summed over every serving block), the
bound to use where only aggregate bounds exist. Remote has no wipe or certifier, so its serving blocks are a server over
the horizon with only the inference cut. An attack is a construction, so it is evaluated on a serving block's *layout*
(what a protocol-abiding strategy gets), at the share `mfu` of the gates a construction uses, and the dark leaves at
theirs.

The overhead is `|A| / T_0` for the actual circuit `A` that reaches the target (a factor; the time to the target is
below): the bound's weight per gate of the inventory, against the reference's `w` per gate (its counted weight over its
gates, `κ / 2G`), gives

    overhead ≥ μ / ((u/w) ε_dark + (1 − ε_dark) g),   g = Σ_S n_S · share_S / Σ_S n_S

with `share_S` each serving block's local minimum over the reference's weight in it, `w c_S` (so at most `u/w`), and
`μ` the gates the algorithm needs in multiples of the reference's. `u = w` is `s_×` at the reference's gates per
counted product; a construction runs the reference's operations, so an attack is at `w` throughout. The time to the
target, wall-clock, is then at least the overhead times `T_0 / Φ_total`, the idealized normalization (the reference at
full capacity; `scenario.Scenario.seconds`).

This module knows no GPU, protocol or training class: `envelopes` builds the serving blocks from the physical model,
`classes` and `attacks` give the caps, and `families` the algorithm's profile.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Budget:
    """`(c, k)`: a block's gates and its unregistered input bits, and the bits that may cross its narrowest declared
    forward cut (`math.inf` where the policy declares none: an inference block; an episode of the proof of forward
    computation, whose operators form a one-way chain, has `n_max` positions times the write allowance)."""

    gates: float
    inputs: float
    cut: float = math.inf


#: `q_T(c, k)`: the most counted weight one block with this budget holds, a cap for a bound and a construction for an
#: attack. `math.inf` is no cap (a class whose premises fail at this budget).
Cap = Callable[[Budget], float]


@dataclass(frozen=True)
class Blocks:
    """`count` blocks of one kind in a serving block, each with `budget`."""

    kind: str
    budget: Budget
    count: float


@dataclass(frozen=True)
class Cut:
    """One valid partition of a serving block: its blocks, and its uncertified gates by route (`U`, which no block
    constrains), valid under `cites` (the Lean guarantees and premises that put every gate in a block or in `U`)."""

    name: str
    blocks: tuple[Blocks, ...] = ()
    uncertified: Mapping[str, float] = field(default_factory=dict)
    cites: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServingBlock:
    """`count` serving blocks of one kind over the horizon, each of `gates` (`c_S`, at capacity): its cuts (the
    arguments of the bound's minimum) and its layout (what a protocol-abiding strategy gets, for attacks; None:
    unconstrained), and `mfu`, the share of its gates a construction uses."""

    name: str
    gates: float
    count: float
    cuts: tuple[Cut, ...]
    layout: Cut | None = None
    mfu: float = 1.0


@dataclass(frozen=True)
class Partition:
    """The actual circuit as serving blocks, and the dark leaves: `eps_dark` of the inventory's capacity, of which a
    construction uses `dark_mfu`."""

    serving: tuple[ServingBlock, ...]
    eps_dark: float
    dark_mfu: float = 1.0


@dataclass(frozen=True)
class Share:
    """What one serving block holds, as a share of its weight at capacity (`w c_S`): the total, the cut that set it,
    and the cut's terms (by block kind and uncertified route)."""

    value: float
    cut: str
    terms: Mapping[str, float]


Side = Literal["bound", "attack"]


def terms(block: ServingBlock, cut: Cut, cap: Cap, w: float, u: float | None = None) -> dict[str, float]:
    """`cut`'s argument of the minimum, term by term, over `w c_S`: each block kind's `n · min(q(c, k), u c)` and each
    route's `q_U(U) = u U` (`u` defaults to `w`)."""
    u = w if u is None else u
    full = w * block.gates
    out = {b.kind: b.count * min(cap(b.budget), u * b.budget.gates) / full for b in cut.blocks}
    for route, gates in cut.uncertified.items():
        out[route] = out.get(route, 0.0) + u * gates / full
    return out


def local(block: ServingBlock, cap: Cap, w: float, side: Side = "bound", u: float | None = None) -> Share:
    """The serving-block bound's local minimum: under a bound, the least of `block`'s cuts; under an attack, its layout
    at `mfu`. Never more than the block's own gates hold (`u c_S`, or `mfu` of `w c_S` for a construction)."""
    u = w if u is None or side == "attack" else u
    ceiling = u / w if side == "bound" else block.mfu
    best = Share(ceiling, "capacity", {"capacity": ceiling})
    if block.gates <= 0:
        return best
    for cut in block.cuts if side == "bound" else (block.layout,) if block.layout else ():
        t = terms(block, cut, cap, w, u)
        total = math.fsum(t.values())
        if total < best.value:
            best = Share(total, cut.name, t)
    return best


def aggregate(partition: Partition, cap: Cap, w: float, u: float | None = None) -> dict[str, Share]:
    """The minimum of the global bounds: each cut name's terms summed over every serving block (a block without that
    cut, or whose cut holds more than its capacity, at its capacity), the least of those sums, as each block's share."""
    u = w if u is None else u
    capacity = Share(u / w, "capacity", {"capacity": u / w})
    weights = {b.name: b.count * b.gates for b in partition.serving}
    best: dict[str, Share] = {b.name: capacity for b in partition.serving}
    best_total = math.fsum(weights.values()) * capacity.value
    for name in sorted({c.name for b in partition.serving for c in b.cuts}):
        shares = {}
        for b in partition.serving:
            cut = next((c for c in b.cuts if c.name == name), None)
            t = terms(b, cut, cap, w, u) if cut and b.gates > 0 else {}
            v = math.fsum(t.values())
            shares[b.name] = Share(v, name, t) if cut and v < capacity.value else capacity
        s = math.fsum(weights[n] * v.value for n, v in shares.items())
        if s < best_total:
            best, best_total = shares, s
    return best


@dataclass(frozen=True)
class Overhead:
    """The overhead for one algorithm under one cap: `rho = efficiency / (dark + (1 − ε_dark) g)`, with each serving
    block's share and `g`'s terms summed over serving blocks (each a share of the reference's weight in the monitored
    fleet at capacity). `dark` is the dark leaves' weight over the reference's in the whole inventory: `(u/w) ε_dark`
    under a bound (`q_U(dark)`), and `ε_dark` times what a construction uses of them under an attack."""

    rho: float
    g: float
    eps_dark: float
    dark: float
    efficiency: float
    serving: Mapping[str, Share]
    terms: Mapping[str, float]

    @property
    def monitored(self) -> float:
        """The share of the adversary's counted weight held on monitored gates: the protocols do their job for this
        algorithm when it is small (`g ≪ ε_dark`), and not when it is near 1."""
        rate = self.dark + (1 - self.eps_dark) * self.g
        return (1 - self.eps_dark) * self.g / rate if rate > 0 else 0.0


def overhead(partition: Partition, cap: Cap, w: float, efficiency: float = 1.0, side: Side = "bound",
             aggregate_only: bool = False, u: float | None = None) -> Overhead:
    """The overhead from the serving-block bound: each serving block's local minimum (or, with `aggregate_only`, the
    minimum of the global sums), and the dark leaves. `w` is the reference's weight per gate, `u` the most a gate holds
    under a bound (`q_U`'s, default `w`)."""
    u = w if u is None or side == "attack" else u
    weights = {b.name: b.count * b.gates for b in partition.serving}
    total = math.fsum(weights.values())
    if aggregate_only and side == "bound":
        shares = aggregate(partition, cap, w, u)
    else:
        shares = {b.name: local(b, cap, w, side, u) for b in partition.serving}
    out: dict[str, float] = {}
    for name, s in shares.items():
        for term, v in s.terms.items():
            out[term] = out.get(term, 0.0) + weights[name] * v / total
    g = math.fsum(weights[n] * s.value for n, s in shares.items()) / total if total else 0.0
    dark = partition.eps_dark * (u / w if side == "bound" else partition.dark_mfu)
    return Overhead(rho(dark, g, efficiency, partition.eps_dark), g, partition.eps_dark, dark, efficiency, shares,
                    out)


def rho(dark: float, g: float, efficiency: float = 1.0, eps_dark: float | None = None) -> float:
    """The overhead when the dark leaves hold `dark` of the inventory's weight at capacity and the monitored
    `1 − ε_dark` hold `g` of theirs (`eps_dark` defaults to `dark`)."""
    rate = dark + (1 - (dark if eps_dark is None else eps_dark)) * g
    return efficiency / rate if rate > 0 else math.inf


def least(*caps: Cap) -> Cap:
    """Several bounds on the same block at once: each caps the same counted weight, so the least of them does, under
    the union of their premises."""
    return lambda budget: min((a(budget) for a in caps), default=math.inf)
