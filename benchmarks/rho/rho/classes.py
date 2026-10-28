"""Training classes: each a class of algorithms, defined by the premises it meets, with the per-block bound they give.

A class is an `Argument` on the `bound` side. `allowance(profile)` is the most counted products one block can finish
given its budget, or raises `Outside` when the algorithm isn't in the class (a premise fails, or a parameter the
statement needs is unknown). A budget at which the statement's premises fail gets `math.inf` (no cap). Each class cites
the Lean statement it instantiates first, then its named assumptions; `lean.weakest(cites)` is its status, and
`premise_status(cites, k)` is where those assumptions stand at a block of `k` bits (`data/premises.toml`).

The statements are about inference blocks of `c` gates reading `k` novel bits; here `k` is `budget.inputs`. Adding a
class is one more `Argument` in `CLASSES`: a premise set, the profile fields it reads, and its count.

**Counted sets.** Each statement counts its own products: the covering family and the partial-sum and E1 bounds count
each (weight, token) pair of the forward pass once, `Keyed` counts Dgrad too at `c = 2`, and the cover and thread
theorems count whatever products their certificate covers. A count on a set `S` is `κ / κ_S` times as many of the
profile's counted products (`counted`), and where a premise covers only a share of `S` at `k` (the layers deep enough to
be chained, the layers whose inputs aren't cheap) the count is divided by that share, with no cap where it is 0. Both
conversions are exact over a whole cover, and per block they assume each block does a representative mix of products.

**Units.** The statements, and the attacks' constructions, are in the FLOP model: a block's compute in FLOPs
(`FlopBudget`), its count in counted products. `per_block` turns one into the exclusion bound's per-block cap
`q_T(c, k)` on gates through `hardware.Units`; `weight_per_gate` is the reference's weight per gate `w`, which the
overhead is normalized by, and `Structure` the most a gate holds under a bound, `u = ω_max / s_×`
(`data/exclusion.toml`), with `s_×` and `Units.gates_per_flop` from one source (`Structure.check`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

from rho import lean
from rho.hardware import Units
from rho.params import Registry
from rho.profile import Profile
from rho.theorem import Cap


@dataclass(frozen=True)
class FlopBudget:
    """A block's budget as the FLOP-model statements read it: `Budget` with its gates as FLOPs (`compute`)."""

    compute: float
    inputs: float
    cut: float = math.inf


#: The most counted products one block with this budget finishes, in the FLOP model: a cap for a bound, a construction
#: for an attack. `math.inf` is no cap (a bound whose premises fail at this budget).
Allowance = Callable[[FlopBudget], float]


def per_block(a: Allowance, units: Units) -> Cap:
    """`q_T(c, k)`: allowance `a` on a block of `c` gates."""
    return lambda b: a(FlopBudget(units.flops(b.gates), b.inputs, b.cut))


def weight_per_gate(p: Profile, units: Units) -> float:
    """`w = κ / 2G`, the reference's counted weight per gate (its counted products, `κ/2` a FLOP, over its gates): what
    the overhead is normalized by. The most a gate holds under a bound is `Structure.weight`."""
    return p.kappa / (2 * units.gates_per_flop)


#: `s_×` at the reference's gates per counted product, `2G / κ`: the reference's average, until the structural minimum is
#: measured
REFERENCE = "reference"


@dataclass(frozen=True)
class Structure:
    """The elementary uncertified cap `q_U(U) = ω_max U / s_×`: what an uncertified or a dark gate holds under a bound
    (`theorem`'s `u`), and the most any block holds, `u c`."""

    name: str
    #: `s_×`: the fewest gates in any counted operation's subcircuit, or `REFERENCE`
    s_times: float | str
    #: `ω_max`: the most counted weight one operation carries
    omega_max: float = 1.0
    #: where `s_×`'s gates come from, as `Units.basis` says it for `gates_per_flop` (`REFERENCE` is the FLOP model's);
    #: none needed at `s_× ≤ 1`, which holds in every design
    basis: str = ""
    #: the runs that measured `s_×` (`art:` ids), cited beside the straddling premise
    evidence: tuple[str, ...] = ()

    #: `U / s_×` counts the operations inside the uncertified gates only if none has gates on both sides; at `s_× = 1`
    #: each operation is charged to one gate, its output, and none can straddle
    STRADDLE = ("draft:no-straddle",)

    def check(self, units: Units) -> None:
        """Refuse an `s_×` and a `gates_per_flop` from different sources. The bound holds a gate at `u/w = ω_max 2G /
        (κ s_×)` of the reference's weight, so `G` cancels only when both count the same gates: a Boolean design's
        measured `s_×` beside the FLOP model's `G` moves every term counted at `u` (the dark leaves' first) with
        nothing to say so."""
        if (self.s_times != REFERENCE and float(self.s_times) <= 1) or self.basis == units.basis:
            return
        raise ValueError(
            f"structure.{self.name}: s_times ({self.s_times}) comes from {self.basis or 'no stated basis'!r} but "
            f"units.gates_per_flop ({units.gates_per_flop:g}) from {units.basis!r}; take both from one source: "
            f"units.gates_per_flop and units.basis from the design s_times was counted on (basis = "
            f"\"boolean:<design>\" on both), or structure = \"reference\" (2G / kappa, the FLOP model's)")

    def gates(self, w: float) -> float:
        """`s_×` in gates, given the reference's weight per gate `w` (`1 / w` at `REFERENCE`)."""
        return 1 / w if self.s_times == REFERENCE else float(self.s_times)

    def weight(self, w: float) -> float:
        """`u = ω_max / s_×`."""
        return self.omega_max / self.gates(w)

    def cites(self) -> tuple[str, ...]:
        return self.STRADDLE + self.evidence if self.s_times == REFERENCE or float(self.s_times) > 1 else ()

    def elementary(self) -> Structure:
        """One counted operation per gate, at most: the cap that needs no straddling premise."""
        return Structure("elementary", 1.0, self.omega_max)


def structure(registry: Registry, name: str) -> Structure:
    t = f"structure.{name}"
    return registry.fields(Structure, t, evidence=tuple(registry.get(f"{t}.evidence", ())))


class Outside(Exception):
    """The algorithm is outside an argument's class: the reason is the message."""


class Argument(Protocol):
    """A way of bounding (`bound`) or achieving (`attack`) the counted products of one block."""

    name: str
    side: Literal["bound", "attack"]
    #: the Lean statement first (or a `draft:` id), then the named premises
    cites: tuple[str, ...]

    def allowance(self, p: Profile) -> Allowance: ...


def need(p: Profile, *names: str) -> tuple[float, ...]:
    missing = [n for n in names if getattr(p, n) is None]
    if missing:
        raise Outside(f"{p.name} leaves {', '.join(missing)} unknown")
    return tuple(getattr(p, n) for n in names)


def counted(p: Profile, counts: str) -> float:
    """`κ / κ_S`: the profile's counted products per product of the set `S` a statement counts. `counts` names `S`'s
    `κ` field (`kappa` is the profile's own set); `keyed` is `Keyed`'s, the forward products at `c = 1` and with Dgrad
    on the forward's tokens at `c = 2`, which is `κ`'s."""
    if counts == "keyed":
        counts = "kappa" if need(p, "per_param")[0] >= 2 else "kappa_forward"
    if counts == "kappa":
        return 1.0
    return p.kappa / need(p, counts)[0]


@dataclass(frozen=True)
class Formula:
    """A bound given by a formula in `k` and the profile's fields: `count(k, *reads)` products of the set `counts`
    (`counted`), over the share `share(k, *share_reads)` of that set the premises cover at `k`."""

    name: str
    cites: tuple[str, ...]
    reads: tuple[str, ...]
    count: Callable[..., float]
    side: Literal["bound", "attack"] = "bound"
    counts: str = "kappa"
    share: Callable[..., float] | None = None
    share_reads: tuple[str, ...] = ()

    def allowance(self, p: Profile) -> Allowance:
        values = need(p, *self.reads)
        scale = counted(p, self.counts)
        extra = need(p, *self.share_reads)

        def a(budget) -> float:
            k = budget.inputs
            x = self.share(k, *extra) if self.share else 1.0
            return self.count(k, *values) * scale / x if x > 0 else math.inf
        return a


def upstream_share(k: float, b: float, layers: int, layer_weights: float) -> float:
    """The share of the products `MulBlocks.Upstream` certifies at depth `D` with `k < b D`: the uses of every `m`-th
    layer, each region holding its token's `m` layers below (`D = m · layer_weights`), so `m = ⌊k / (b ·
    layer_weights)⌋ + 1` and `⌊(L − 1) / m⌋` of the `L` layers count (the first layer's products have no certificate)."""
    m = math.floor(k / (b * layer_weights)) + 1
    return math.floor((layers - 1) / m) / layers


def chained_share(k: float, c_w: float, layers: int, layer_weights: float) -> float:
    """The share of the products chained at a depth `D` with `c_w D > k` (`ActivationChainedAt`): those of the layers
    whose token has more than `k / c_w` updated weights upstream, `ℓ · layer_weights > k / c_w`."""
    return max(0, layers - math.floor(k / (c_w * layer_weights)) - 1) / layers


def chained_share_one(k: float, c_w: float, c_a: float, layers: int, layer_weights: float) -> float:
    """`chained_share` at `CoveringBound`'s one cost `c = min(c_w, c_a)`, whose depth needs `c D > k`."""
    return chained_share(k, min(c_w, c_a), layers, layer_weights)


def expensive_share(k: float, layer_bits: float, layers: int) -> float:
    """`1 − cheap(k)`: the share of the products whose activation isn't computable from token ids by a block of `k`
    bits. A block that reads the first `n = ⌊k / layer_bits⌋` layers' updates computes layers `0..n`'s inputs, so
    `cheap(k) = min(1, (n + 1) / L)`, all of them once the update fits (E1, "Cheap inputs")."""
    return 1 - min(1.0, (math.floor(k / layer_bits) + 1) / layers)


def cover(k: float, b: float) -> float:
    """`CoverTraining`: `α₀ b s ≤ (α²Γ + bα) size(d)`, so a block of `k` novel bits finishes at most `k²/b + k`."""
    return k * k / b + k


def cover_aligned(k: float, b: float, f: float) -> float:
    """`CoverTrainingAligned`: `α₀ b f s ≤ (α²Γ + b f α) size(d)` on activations of whole words at least `f` bits wide,
    so a block finishes at most `k² / (b f) + k`."""
    return k * k / (b * f) + k


def chained(k: float, c_w: float, c_a: float) -> float:
    """`CoverTrainingChained`, `CoveringBoundTwoCosts`: `(k/c_w)(k/c_a) + k` products per block."""
    return (k / c_w) * (k / c_a) + k


def covering(k: float, c_w: float, c_a: float) -> float:
    """`CoveringBound`: `(k / c)² + k` at one cost `c = r · b`, which both premises must meet: `c = min(c_w, c_a)`."""
    c = min(c_w, c_a)
    return (k / c) ** 2 + k


def covering_additive(k: float, c_w: float, c_a: float) -> float:
    """`CoveringBoundAdditive`: `(k/c_w)(k/c_a)/4 + k`, the two charges from disjoint inputs (AM–GM)."""
    return (k / c_w) * (k / c_a) / 4 + k


def covering_priced(k: float, c_a: float, price: float, price_max: float) -> float:
    """`CoveringBoundPriced`: the chained blocks' prices add up to at most `k (k / c_a) + t k` per block, so at a mean
    price `p̄` a block finishes at most `(k² / c_a + t k) / p̄`. Its depth condition, any `D` weights costing more than
    `k`, is read at the price floor `c_w` (`chained_share`)."""
    return (k * k / c_a + price_max * k) / price


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


@lru_cache(maxsize=1 << 16)
def partial_sum(k: float, b: float, f: float, a: float, aout: float, grid: int = 1000) -> float:
    """`PartialSumCoverBound`: `r(μ) (#P − k #𝓘) ≤ 2k #𝓘` for every `μ ∈ [0, 1]`, so a block finishes at most
    `k + 2k / r` products at the best `r = max_μ psRate b f a aOut (2k) μ` (over a grid of `μ`)."""
    r = max(ps_rate(b, f, a, aout, 2 * k, i / grid) for i in range(grid + 1))
    return k + 2 * k / r if r > 0 else math.inf


def partial_sum_keyed(k: float, b: float, legs: int, aout: float, vocab: int, bcol: float, n: float) -> float:
    """`PartialSumKeyedCoverBound`: `PartialSumCoverBound` at `f = 1` and `a = min(d − E, n)` (`KeyedActivationReads`),
    while `E < d`."""
    e = lookups(k, bcol, vocab)
    if e >= legs:
        return math.inf
    return partial_sum(k, b, 1, min(legs - e, n), aout)


def threaded(k: float, b_a: float, b_w: float, d: float, e: float, c: float) -> float:
    """`ThreadedThrottled`: `b_a |M| ≤ α Λ(αΓ) size(d)`, so a block of `k` novel bits does at most `k Λ(k) / b_a` counted
    multiplications, with `Λ = max(e, c k / (4 d b_w))` (`Threading.bound`)."""
    return k * max(e, c * k / (4 * d * b_w)) / b_a


def threaded_checked(k: float, b_a: float, b_w: float, d: float, e: float, c: float, k_max: float) -> float:
    """`threaded` where the `Threading` certificate holds (`ReferenceThreaded`: `S ≤ k_max`), no cap beyond."""
    return threaded(k, b_a, b_w, d, e, c) if k <= k_max else math.inf


def fanout_priced(k: float, aout: float, beta_x: float) -> float:
    """E1's Proposition 3, sharpened: each activation meets at most `aOut` counted products and costs a block `β_x`
    fresh bits, so a block of `k` bits finishes at most `aOut · k / β_x` (the priced form's `min{h_w, P_col} h_x / β_x`
    at `P_col = aOut p̄ ≤ h_w`, `h_x ≈ k`). `β_x` is the replay-consistent rate, `≤ (h_tok + F) / d`."""
    return aout * k / beta_x


def fanout_split(k: float, aout: float, beta_x: float, gamma: float, price: float, price_max: float) -> float:
    """E1's Proposition 3′ (splitting allowed, under `H_γ^t`): `(1 + 2/γ) aOut k / β_x + k p_max / p̄` per block."""
    return (1 + 2 / gamma) * aout * k / beta_x + k * price_max / price


def accumulator(k: float, h_eff: float, b_ps: float, step_tokens: float) -> float:
    """E1's Corollary 4a, under (R): `M ≤ n k² / (b_ps h_eff) + n_E k / h_eff` with `n_E = M / T_step` entries, so a block
    finishes at most `k² / (b_ps h_eff) / (1 − k / (h_eff T_step))` weight-gradient products, no cap at `k ≥ h_eff T_step`."""
    rest = 1 - k / (h_eff * step_tokens)
    return k * k / (b_ps * h_eff) / rest if rest > 0 else math.inf


def output_cap(map_width_min: float, b: float, beta_x: float, beta_v: float) -> float:
    """Proposition 6b's cap `k* = 4 d β_w (h_tok + F_in) / β_v^R`, with `h_tok + F_in = β_x d` (the replay-consistent
    rate) and `β_w = b`, at the narrowest map `d = d_in`."""
    return 4 * map_width_min ** 2 * b * beta_x / beta_v


def output_premise(k: float, map_width: float, map_width_min: float, b: float, beta_x: float, beta_v: float) -> float:
    """E1's Proposition 6b: each output value a block produces costs `β_v^R` and carries `d` products, so a block
    finishes at most `d k / β_v^R` while `k ≤ k*` (`output_cap`), no cap beyond."""
    return map_width * k / beta_v if k <= output_cap(map_width_min, b, beta_x, beta_v) else math.inf


DEPTH = ("layers", "layer_weights")

CLASSES: Mapping[str, Argument] = {a.name: a for a in (
    Formula("cover", ("Nci.Cover.Guarantees.CoverTraining", "Nci.Cover.Assumptions.UpdateHardness",
                      "Nci.Cover.MulBlocks.Upstream", "Nci.Assumptions.BaselineRegistered"),
            ("b",), cover, share=upstream_share, share_reads=("b", *DEPTH)),
    Formula("cover-aligned", ("Nci.Cover.Guarantees.CoverTrainingAligned", "Nci.Cover.Assumptions.UpdateHardness",
                              "Nci.Cover.MulBlocks.Upstream", "Nci.Assumptions.BaselineRegistered"),
            ("b", "f"), cover_aligned, share=upstream_share, share_reads=("b", *DEPTH)),
    Formula("cover-chained", ("Nci.Cover.Guarantees.CoverTrainingChained", "Nci.Assumptions.ParameterHardAt",
                              "Nci.Assumptions.ActivationChainedAt", "Nci.Assumptions.BaselineRegistered"),
            ("c_w", "c_a"), chained, counts="kappa_forward", share=chained_share, share_reads=("c_w", *DEPTH)),
    Formula("covering", ("Nci.Guarantees.CoveringBound", "Nci.Assumptions.ParameterHard",
                         "Nci.Assumptions.ActivationChained"),
            ("c_w", "c_a"), covering, counts="kappa_forward", share=chained_share_one,
            share_reads=("c_w", "c_a", *DEPTH)),
    Formula("covering-two-costs", ("Nci.Guarantees.CoveringBoundTwoCosts", "Nci.Assumptions.ParameterHardAt",
                                   "Nci.Assumptions.ActivationChainedAt"),
            ("c_w", "c_a"), chained, counts="kappa_forward", share=chained_share, share_reads=("c_w", *DEPTH)),
    Formula("covering-additive", ("Nci.Cover.Guarantees.CoveringBoundAdditive",
                                  "Nci.Cover.Assumptions.ConditionalParameterHardAt",
                                  "Nci.Assumptions.ActivationChainedAt"),
            ("c_w", "c_a"), covering_additive, counts="kappa_forward", share=chained_share,
            share_reads=("c_w", *DEPTH)),
    Formula("covering-priced", ("Nci.Guarantees.CoveringBoundPriced", "Nci.Assumptions.ParameterHardPricedAt",
                                "Nci.Assumptions.ActivationChainedAt"),
            ("c_a", "price", "price_max"), covering_priced, counts="kappa_forward", share=chained_share,
            share_reads=("c_w", *DEPTH)),
    Formula("fanout", ("Nci.Cover.Guarantees.FanOutCoverBound", "Nci.Cover.Assumptions.UpdateHardness",
                       "Nci.Cover.Assumptions.ColumnsIncompressible"),
            ("b", "legs", "fanout", "per_param", "vocab", "bcol"), fanout, counts="keyed"),
    Formula("fanout-read", ("Nci.Cover.Guarantees.FanOutReadCoverBound", "Nci.Cover.Assumptions.ValueReadHardness",
                            "Nci.Cover.Assumptions.ColumnsIncompressible"),
            ("b", "f", "legs", "fanout", "per_param", "vocab", "bcol"), fanout_read, counts="keyed"),
    Formula("partial-sum", ("Nci.Cover.Guarantees.PartialSumCoverBound", "Nci.Cover.Assumptions.UpdateHardness",
                            "Nci.Cover.Assumptions.ActivationReads"),
            ("b", "f", "width", "fanout"), partial_sum, counts="kappa_forward"),
    Formula("partial-sum-keyed", ("Nci.Cover.Guarantees.PartialSumKeyedCoverBound",
                                  "Nci.Cover.Assumptions.UpdateHardness", "Nci.Cover.Assumptions.ColumnsIncompressible",
                                  "Nci.Cover.Cube.Wide"),
            ("b", "legs", "fanout", "vocab", "bcol", "entry_width"), partial_sum_keyed, counts="kappa_forward"),
    Formula("threaded", ("Nci.Guarantees.ThreadedThrottled", "Nci.Assumptions.WeightsIncompressible",
                         "Nci.Assumptions.BaselineRegistered", "Nci.Guarantees.ReferenceThreaded",
                         "Nci.Assumptions.ReferenceLayered"),
            ("b_a", "b_w", "thread_d", "thread_e", "thread_c", "thread_k"), threaded_checked),
    Formula("fanout-priced", ("draft:E1-Prop3", "draft:E1-activation-rate"),
            ("fanout", "beta_x"), fanout_priced, counts="kappa_forward", share=expensive_share,
            share_reads=("layer_bits", "layers")),
    Formula("fanout-split", ("draft:E1-Prop3-prime", "draft:E1-activation-rate", "draft:E1-H-gamma"),
            ("fanout", "beta_x", "gamma", "price", "price_max"), fanout_split, counts="kappa_forward",
            share=expensive_share, share_reads=("layer_bits", "layers")),
    Formula("accumulator", ("draft:E1-Cor4a", "draft:E1-R"),
            ("h_eff", "b_ps", "step_tokens"), accumulator, counts="kappa_wgrad"),
    Formula("output-premise", ("draft:E1-Prop6b", "draft:E1-output-premise"),
            ("map_width", "map_width_min", "b", "beta_x", "beta_v"), output_premise, counts="kappa_forward"),
)}

#: weakest last
STANDINGS = ("supported", "assumed", "contested", "refuted")


def premises(registry: Registry | None = None) -> dict[str, dict]:
    """`data/premises.toml` by the cited name (`lean`): each premise's `standing`, `beyond` and `credible_bits`."""
    reg = registry or Registry.load()
    return {reg[f"premise.{n}.lean"]: reg.table(f"premise.{n}") for n in reg.names("premise")}


def premise_status(cites: Iterable[str], k: float, registry: Registry | None = None,
                   table: Mapping[str, Mapping] | None = None) -> str:
    """The weakest standing (`STANDINGS`) of the premises among `cites` at a block of `k` bits: each premise's
    `standing` up to its `credible_bits`, its `beyond` above. Statements (`proved`, `draft:E1-Prop…`) and evidence
    don't count; an assumption `data/premises.toml` doesn't list is `assumed`. `table` is `premises(registry)`, read
    once by a caller that asks often."""
    table = premises(registry) if table is None else table
    out = "supported"
    for c in cites:
        if c in table:
            row = table[c]
            s = row["standing"] if k <= row["credible_bits"] else row["beyond"]
        elif lean.status(c) == "assumption":
            s = "assumed"
        else:
            continue
        out = max(out, s, key=STANDINGS.index)
    return out
