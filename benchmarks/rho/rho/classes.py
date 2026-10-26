"""Training classes: each a class of algorithms, defined by the premises it meets, with the per-block bound they give.

A class is an `Argument` on the `bound` side. `allowance(profile)` is the most counted products one block can finish
given its budget, or raises `Outside` when the algorithm isn't in the class (a premise fails, or a parameter the
statement needs is unknown). A budget at which the statement's premises fail gets `math.inf` (no cap). Each class cites
the Lean statement it instantiates first, then its named assumptions; `lean.weakest(cites)` is its status.

The statements are about inference blocks of `s` gates reading `k` novel bits; here `k` is `budget.inputs`. Adding a
class is one more `Argument` in `CLASSES`: a premise set, the profile fields it reads, and its count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from rho.profile import Profile
from rho.theorem import Allowance


class Outside(Exception):
    """The algorithm is outside an argument's class: the reason is the message."""


class Argument(Protocol):
    """A way of bounding (`bound`) or achieving (`attack`) the counted products of one block."""

    name: str
    side: Literal["bound", "attack"]
    #: the Lean statement first (or a `note:`/`art:`/`draft:` id), then the named premises
    cites: tuple[str, ...]

    def allowance(self, p: Profile) -> Allowance: ...


def need(p: Profile, *names: str) -> tuple[float, ...]:
    missing = [n for n in names if getattr(p, n) is None]
    if missing:
        raise Outside(f"{p.name} leaves {', '.join(missing)} unknown")
    return tuple(getattr(p, n) for n in names)


@dataclass(frozen=True)
class Formula:
    """A bound given by a formula in `k` and the profile's fields."""

    name: str
    cites: tuple[str, ...]
    reads: tuple[str, ...]
    count: Callable[..., float]
    side: Literal["bound", "attack"] = "bound"

    def allowance(self, p: Profile) -> Allowance:
        values = need(p, *self.reads)
        return lambda budget: self.count(budget.inputs, *values)


def cover(k: float, b: float) -> float:
    """`CoverTraining`: `α₀ b s ≤ (α²Γ + bα) size(d)`, so a block of `k` novel bits finishes at most `k²/b + k`."""
    return k * k / b + k


def chained(k: float, c_w: float, c_a: float) -> float:
    """`CoverTrainingChained`: `(k/c_w)(k/c_a) + k` products per block."""
    return (k / c_w) * (k / c_a) + k


def lookups(k: float, bcol: float, vocab: float) -> int:
    """`E`: the least number of a token's embedding lookups with `k < bcol · V · (E + 1)`."""
    return math.floor(k / (bcol * vocab))


def fanout(k: float, b: float, legs: int, aout: float, c: float, vocab: int, bcol: float) -> float:
    """`FanOutCoverBound`: `k · max(aOut, c k / ((d − E) b)) + k`, while `E < d`."""
    e = lookups(k, bcol, vocab)
    if e >= legs:
        return math.inf
    return k * max(aout, c * k / ((legs - e) * b)) + k


def fanout_read(k: float, b: float, f: float, legs: int, aout: float, c: float, vocab: int, bcol: float) -> float:
    """`FanOutReadCoverBound`: `k · max(aOut / f, c k / (4 (d − E) f b)) + k`, while `E < d`."""
    e = lookups(k, bcol, vocab)
    if e >= legs:
        return math.inf
    return k * max(aout / f, c * k / (4 * (legs - e) * f * b)) + k


def ps_rate(b: float, f: float, a: float, aout: float, pool: float, mu: float) -> float:
    """`psRate b f a aOut K μ`, exactly as Lean states it."""
    c1 = (1 + math.sqrt(mu)) ** 2 * (b * f * a / pool)
    c2 = 2 * (1 + math.sqrt(mu)) * math.sqrt(1 - mu) * math.sqrt(b * f / (3 * pool))
    return max(min(c1, c2), mu * f / max(aout, pool / (4 * b * a)))


def partial_sum(k: float, b: float, f: float, a: float, aout: float, grid: int = 1000) -> float:
    """`PartialSumCoverBound`: `r(μ) (#P − k #𝓘) ≤ 2k #𝓘` for every `μ ∈ [0, 1]`, so a block finishes at most
    `k + 2k / r` products at the best `r = max_μ psRate b f a aOut (2k) μ` (over a grid of `μ`)."""
    r = max(ps_rate(b, f, a, aout, 2 * k, i / grid) for i in range(grid + 1))
    return k + 2 * k / r if r > 0 else math.inf


def fanout_priced(k: float, aout: float, beta_x: float) -> float:
    """The draft's Proposition 3, sharpened (E1): each activation meets at most `aOut` counted products and costs a block
    `β_x` fresh bits, so a block of `k` bits finishes at most `aOut · k / β_x`. Linear in `k`, with no `E < d` premise,
    so it is the one count that stays finite at a server-window's budget."""
    return aout * k / beta_x


def threaded(k: float, b_a: float, b_w: float, d: float, e: float, c: float) -> float:
    """`ThreadedThrottled`: `b_a |M| ≤ α Λ(αΓ) size(d)`, so a block of `k` novel bits does at most `k Λ(k) / b_a` counted
    multiplications, with `Λ = max(e, c k / (4 d b_w))` (`Threading.bound`)."""
    return k * max(e, c * k / (4 * d * b_w)) / b_a


CLASSES: Mapping[str, Argument] = {a.name: a for a in (
    Formula("cover", ("Nci.Cover.Guarantees.CoverTraining", "Nci.Cover.Assumptions.UpdateHardness"), ("b",), cover),
    Formula("cover-chained", ("Nci.Cover.Guarantees.CoverTrainingChained", "Nci.Assumptions.ActivationChained",
                              "Nci.Assumptions.ParameterHard"), ("c_w", "c_a"), chained),
    Formula("fanout", ("Nci.Cover.Guarantees.FanOutCoverBound", "Nci.Cover.Assumptions.UpdateHardness",
                       "Nci.Cover.Assumptions.ColumnsIncompressible"),
            ("b", "legs", "fanout", "per_param", "vocab", "bcol"), fanout),
    Formula("fanout-read", ("Nci.Cover.Guarantees.FanOutReadCoverBound", "Nci.Cover.Assumptions.ValueReadHardness",
                            "Nci.Cover.Assumptions.ColumnsIncompressible"),
            ("b", "f", "legs", "fanout", "per_param", "vocab", "bcol"), fanout_read),
    Formula("partial-sum", ("Nci.Cover.Guarantees.PartialSumCoverBound", "Nci.Cover.Assumptions.UpdateHardness",
                            "Nci.Cover.Assumptions.ActivationReads"), ("b", "f", "width", "fanout"), partial_sum),
    Formula("threaded", ("Nci.Guarantees.ThreadedThrottled", "Nci.Assumptions.WeightsIncompressible"),
            ("b_a", "b_w", "thread_d", "thread_e", "thread_c"), threaded),
    Formula("fanout-priced", ("draft:Prop3", "note:training-20261024T2336Z-report-tvi-findings"),
            ("fanout", "beta_x"), fanout_priced),
)}
