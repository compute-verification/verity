"""A planner for the sampling scheme: from a Program's partition candidates, a proof system's cost model and an objective,
the plans on the Pareto frontier of expected cost against certified harm, over every budget.

Nothing trusts it.  A plan is a stratified sampling design, `k_s` proof blocks of each subcircuit, the work law
`work:1` over a work table carrying the credited work and floors equal to the sizes (`work_law`:
`one_stage.draw.work_k(1, w, n, W, f)` is `min(n, f)`).  A stream's rule can't state it: a fixed count per stratum falls
against the stratum as the prover pads it, so a stream registers a rate (`work-rate:NUM/DEN`, every stratum's floor at
the rate).  Its certificate is recomputed from the sizes alone (`certify`): core's `Stratified.harm_bound` for the
objective's harm and for credited work, and `density_bound`, the closed form that `work_escape_le`'s proof extends to any
sizes.

**Cuts.**  An op type is cut into proof blocks one of several ways, each a set of subcircuits (`Candidate.cut`): one subcircuit per
op is a partition of it; a shared piece is either its own subcircuit, committed once and read by the others, or
recomputed inside each reader (a covering, `recomputed_across` in `Q_word` v2), and the planner picks per density.  A proof block
whose incorrect output can forfeit downstream work (a noise or form proof block every dot of its row reads) carries that work as
`influence`, which its harm counts.

**Cost.**  Per touched statement plus per proved proof block, as Flock does: subcircuit `s` cut into statements of `c` proof blocks costs
`touched_s(k; c) * (a0 + a1 c) + k * (b + replay)`, a statement being touched when the sample meets it.  Everything is
committed at serve time whatever the sample: every proof block's outputs (`commit_bits`, `commit_rows`),
and the fixed serve-time commitments of every scale (`Commitment`: the Commit's taps, PoUW's work blocks, the remote audit's
receipts, the bits a work block pins), so a plan's cost is the whole serve-time and proving total.  One tensor committed
at several scales is counted once per scale, and `overlaps` prices what committing it once would save.  A plan is thus a
one-stage design (`two_stage_design` prices a two-stage one, below).  A sampled proof block's witness is built by replaying the
replay block of its `ReplayGrain` it lies in from the values the prover keeps, which commits nothing: a commitment is a
hash, so unless serving retained intermediates a replay block starts at the external inputs and the registration's
constants.
A subcircuit's costs are measured (`TemplateCost`) or derived from the model's rates and its features: ANDs, and the hm96
rows a proved proof block opens (`row_ands`).

**Sweep.**  A Lagrangian sweep: one density `lam` on every subcircuit (`k_s = ceil(min(1, lam w_b v_s) n_s)`, `w_b` the
weight of the subcircuit's weight class, 1 unless the plan weighs classes), some subcircuits at their floor only, some proved
whole, and the cut per op type and the statement size each the cheapest at that `lam`.  With no statement cost and a proof block
cost proportional to work, the cheapest plan certifying `C` is `Achievability`'s rates, `q_s = min(1, ln(1/delta) w_s / C)`.
`weigh` chooses the kinds' weights at a target, under the same stratified certificate.

**Modes.**  `CostModel.commit_weight` is the weight w on serving's commitments in a plan's total: at 1 they count as spent,
and at 0 (`PROVING_ONLY`) the plan prices proving alone, the model of Lean's `GranularityOptimum` (the cheapest law costs
`c(s) · (F/C) · Λ`).  `Grain` is that model on one uniform partition, with the commitment term beside it: `n · b(s) · h`
for `n` proof blocks of `b(s)` committed bytes at `h` a hashed byte, paid whether a proof block is sampled or not.
`grain_costs` prices a ladder of partitions for the free-compute target (`Achievability`'s law, `GranularityOptimum`'s
cost) or the exfiltration target (`ExfiltrationOptimum`'s law), and `sqrt_grain` is the work target's optimum in closed
form, `s* = √(w β h C / (o Λ))`, when `c(s) = f + o s` and every proof block commits `β` bytes.

**Two-stage.**  A two-stage design (`Claims.Sampling.Design.Two`): serving commits only the replay blocks' boundaries, the
first stage samples exactly K of a verification phase's n replay blocks, uniformly (the rate q = K/n; the bound holds
at that rate), the prover commits each sampled one's interior, and the second
stage samples k_t of the n_t proof blocks of each stratum t in it with fresh coins.  It certifies exfiltration alone
(compute credit isn't certified with deferred interiors), in `CompiledTwo.Exfil`'s form: Σ over the admissible sets T of
incorrect proof blocks of 2^{b(T)}, each T charged its own blocks' committed bits.  `two_stage_exfil` bounds that count's
log2 (a wholly incorrect replay block is charged all its bits, so the bound is near Λ/q of them), `two_stage_price`
prices a sampled replay block's proving, each type's statements its own or its verification phase's
(`two_stage_statements`), and `two_stage_design` the design with the least bound at a budget, width-stratified
(the public size classes up to a threshold in one stratum, each wider one in its own; on request also each distinct
width in its own, `two_stage_strata`), uniform (one stratum) or by template (each template a stratum).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import ceil, exp, expm1, floor, fsum, inf, lgamma, log, log1p, log2, nextafter, sqrt, ulp
from pathlib import Path

from verity.protocols.spot.verifier.profile import HARM_MARGIN, IntegrityProfile, Stratified, Stratum, neg_log_escape
from verity.protocols.spot.verifier.consumers import entropy_bits, exfiltration_bound
from verity.circuits.partition_v2 import UNIT_CLASS_MAX, unit_class


def pinned_ands(definition: str, sidecar: str = "lms") -> int:
    """`definition`'s ANDs as its circuit-check pin records them (`verity.commitments.gates`' `SIDECAR.circuit-check.json`):
    the AND rows Flock proves, the copies that keep a form short (`forms.commit`) among them."""
    from verity.commitments import gates

    pins = json.loads(Path(gates.__file__).with_name(f"{sidecar}.circuit-check.json").read_text())
    return pins["definitions"][definition]["and"]


#: ANDs of one SHA-512 compression in hm96 PROTOCOL §5's ripple-carry model, no copies: a gate count to compare hashes
#: by, not what Flock proves
SHA512_ANDS = 58_120
#: the AND rows Flock proves for one SHA-512 compression, `Sha512Compress_v1`'s pin: `compress_gadget`'s ANDs and the
#: copies it commits
SHA512_ROWS = pinned_ands("Sha512Compress_v1")
#: the hiding part of an hm96-sha512 row in circuit: two private compressions (hm96 PROTOCOL §5)
HM96_ANDS = 2 * SHA512_ROWS


def row_ands(nbytes: int) -> int:
    """In-circuit AND rows to open one hm96-sha512 row of `nbytes` bytes: its digest's private blocks, the length and
    padding included, and the hiding part (hm96 PROTOCOL §6's table, at `SHA512_ROWS` a compression)."""
    return HM96_ANDS + SHA512_ROWS * ceil((nbytes + 17) / 128)


@dataclass(frozen=True)
class Candidate:
    """One subcircuit of one way (`cut`, default the subcircuit alone) to cut one op type into proof blocks, derived from the IR,
    never from the prover: `units` instances, each credited `work` (the verifier's work table), with `out_bits` over
    `out_gates` boundary gates, its committed bits, which fit a public size class (`partition_v2.unit_class`, at
    most `UNIT_CLASS_MAX`, as the one-stage partition refuses a wider proof block), and `influence`, the credited work a wrong
    one can forfeit downstream.  A proved proof block costs `ands` plus opening `rows` (bytes each) in circuit; serving commits
    `commit_bits` in `commit_rows` rows per proof block whatever the sampling design; one statement holds at most `cap` proof blocks
    (m = k_log + nbl <= 35); `n_rho` proof blocks of it lie in each replay block of replay grain `grain`; and `kind` is its
    weight class for the sample's weights (served proof blocks of one subcircuit, PoUW's tiles), the subcircuit itself when empty."""

    op: str
    template: str
    units: int
    work: int
    out_bits: int
    out_gates: int = 1
    cut: str = ""
    influence: int = 0
    ands: int = 0
    rows: tuple[int, ...] = ()
    commit_bits: int = 0
    commit_rows: float = 0.0
    cap: int = 1
    grain: str = ""
    n_rho: int = 0
    kind: str = ""

    def __post_init__(self) -> None:
        if not self.op or not self.template or self.units < 1 or self.cap < 1 or self.out_bits < 0 or self.out_gates < 0 \
                or min(self.work, self.influence, self.ands, self.commit_bits, self.commit_rows, *self.rows, 0) < 0 \
                or unit_class(self.out_bits) is None or (bool(self.grain) != (self.n_rho > 0)) \
                or self.n_rho > self.units:
            raise ValueError(f"candidate {self.template!r}: needs units >= 1, cap >= 1, non-negative sizes, outputs in a "
                             f"size class (<= {UNIT_CLASS_MAX} bits), and n_rho in [1, units] with a replay grain")

    @property
    def alternative(self) -> str:
        return self.cut or self.template

    @property
    def block(self) -> str:
        return self.kind or self.template


@dataclass(frozen=True)
class TemplateCost:
    """One subcircuit's measured costs under one proof-system version, in the caller's currency: `statement_fixed` (a0) and
    `statement_per_unit` (a1, per staged proof block) per touched statement, `unit` (b) per proved proof block, `replay` per sampled proof block,
    and `sigma`, their relative uncertainty.  A field left None is derived from the model's rates."""

    statement_fixed: float | None = None
    statement_per_unit: float | None = None
    unit: float | None = None
    replay: float = 0.0
    sigma: float = 0.0


@dataclass(frozen=True)
class ReplayGrain:
    """A replay grain: `replay_blocks` replay blocks (served requests, say, each re-run whole from its tokens and the weights),
    which the proof-block partition refines, each replayed at `cost` from the values the prover keeps to build the
    witnesses of the sampled proof blocks in it.  Under a one-stage design replay commits nothing: every value it recomputes
    was committed at serve time.  Under a two-stage design (`two_stage_design`) serving commits each replay block's
    `boundary_bytes`, the boundary commitment, and a sampled one's replay builds its interior, which the prover then
    commits."""

    replay_blocks: int
    cost: float
    boundary_bytes: float = 0.0

    def __post_init__(self) -> None:
        if self.replay_blocks < 1 or not min(self.cost, self.boundary_bytes) >= 0:
            raise ValueError("a replay grain needs replay_blocks >= 1 and a non-negative cost and boundary")


@dataclass(frozen=True)
class Commitment:
    """One tensor committed at serve time by one scale, whatever the sample: `nbytes` hashed over the window, at the model's
    per-byte rate (`CostModel.commit_per_bit`) unless `cost` is measured.  `scale` names who commits it (the Commit's taps,
    PoUW's A rows or tile leaves, the remote audit's receipts, a work block's pin) and `tensor` what; one tensor under two
    scales is an overlap (`overlaps`).  Decode is latency-bound: a decode step hashes `decode_bytes` of it, and `chain` is
    the dependent compressions its trees put on a step's critical path when its roots gate the step (a seed reads them
    before the next call), which they do when `gates`; a copy that doesn't gate trails the step."""

    scale: str
    tensor: str
    nbytes: float
    cost: float | None = None
    decode_bytes: float = 0.0
    chain: float = 0.0
    gates: bool = False

    def __post_init__(self) -> None:
        if not self.scale or not self.tensor or min(self.nbytes, self.decode_bytes, self.chain) < 0 \
                or (self.cost is not None and self.cost < 0):
            raise ValueError(f"commitment {self.scale}/{self.tensor}: needs a scale, a tensor and non-negative sizes")


@dataclass(frozen=True)
class CostModel:
    """Measured subcircuits, and the rates that derive the rest from their features, `ands` being a proof block's ANDs with its
    rows' openings (`row_ands`), as `costfit` measures them: per touched statement `statement_fixed + statement_per_and *
    ands` (the session and the verifier's parse), per staged proof block `staged_per_and * ands + staged_per_unit`, and per
    proved proof block `per_and * ands`.  When `shared`, a template's statements hold only its sampled proof blocks (`IntegrityPeriod`'s
    hidden window: one session, its k sampled proof blocks split evenly over `ceil(k / cap)` statements), so only they are staged;
    otherwise its proof blocks are cut in order into statements and every touched statement is staged whole.  Serving commits each proof block's outputs at `commit_per_bit` and `commit_per_row`, and
    `serve`, every other serve-time commitment, which every plan pays; `replay` names the replay grains that build
    witnesses; `sigma` is a derived cost's relative uncertainty; `source` is the fit's `art:` id.  A plan's total weighs
    serving's commitments, the proof blocks' and `serve`, by `commit_weight` (`PROVING_ONLY`, 0, leaves them out)."""

    templates: Mapping[str, TemplateCost] = field(default_factory=dict)
    per_and: float = 0.0
    statement_fixed: float = 0.0
    statement_per_and: float = 0.0
    staged_per_and: float = 0.0
    staged_per_unit: float = 0.0
    commit_per_bit: float = 0.0
    commit_per_row: float = 0.0
    serve: tuple[Commitment, ...] = ()
    replay: Mapping[str, ReplayGrain] = field(default_factory=dict)
    sigma: float = 0.0
    source: str = ""
    shared: bool = False
    commit_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.commit_weight >= 0:
            raise ValueError("the commitments' weight is at least 0 (0 prices proving alone)")

    def of(self, c: Candidate) -> TemplateCost:
        if c.template in self.templates:
            return self.templates[c.template]
        if not (self.per_and or self.statement_fixed or self.statement_per_and or self.staged_per_and
                or self.staged_per_unit):
            raise KeyError(f"the cost model has no template {c.template!r} and no rates to derive it: fit it or pass a "
                           f"TemplateCost")
        return TemplateCost()

    @staticmethod
    def ands(c: Candidate) -> int:
        return c.ands + sum(row_ands(r) for r in c.rows)

    def statement(self, c: Candidate) -> tuple[float, float]:
        tc = self.of(c)
        a0 = tc.statement_fixed if tc.statement_fixed is not None else \
            self.statement_fixed + self.statement_per_and * self.ands(c)
        a1 = tc.statement_per_unit if tc.statement_per_unit is not None else \
            self.staged_per_and * self.ands(c) + self.staged_per_unit
        return a0, a1

    def unit(self, c: Candidate) -> float:
        tc = self.of(c)
        return tc.unit if tc.unit is not None else self.per_and * self.ands(c)

    def commit(self, c: Candidate) -> float:
        return c.units * (self.commit_per_bit * c.commit_bits + self.commit_per_row * c.commit_rows)

    def committed(self, s: Commitment) -> float:
        return s.cost if s.cost is not None else 8 * self.commit_per_bit * s.nbytes

    def serve_cost(self) -> float:
        """Every fixed serve-time commitment, each tensor once per scale that commits it."""
        return fsum(self.committed(s) for s in self.serve)

    def sigma_of(self, c: Candidate) -> float:
        return self.templates[c.template].sigma if c.template in self.templates else self.sigma


@dataclass(frozen=True)
class Objective:
    """The harm an incorrect proof block does: `work` times its credited work and influence plus `bits` times its output bits.  No
    target: targets and budgets are queries on the frontier.  Every subcircuit samples at least `floor` proof blocks."""

    delta: float
    work: float = 1.0
    bits: float = 0.0
    floor: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1 or self.work < 0 or self.bits < 0 or not self.work + self.bits > 0 or self.floor < 1:
            raise ValueError("an objective needs 0 < delta < 1, weights >= 0 not both 0, and floor >= 1")

    def harm(self, c: Candidate) -> float:
        return self.work * (c.work + c.influence) + self.bits * c.out_bits


@dataclass(frozen=True)
class PlanBounds:
    """What a plan's sampling design certifies at `delta`, from its sizes alone: `harm` (core's `harm_bound` of the objective's harm),
    `density` (`density_bound` of it), `free_compute` (core's `harm_bound` of credited work and influence) and, when the
    objective weighs bits, `leak` (`consumers.exfiltration_bound`: `harm_bound` of output bits plus `location_bits`).  When
    not `exact`, every field is `density_bound`'s and there is no leak."""

    delta: float
    harm: float
    density: float
    free_compute: float
    leak: float | None = None
    exact: bool = True


#: `_log_miss` sums this many of its terms before it takes their closed form: a planner asks it a million times
MISS_TERMS = 64


@dataclass(frozen=True)
class Plan:
    """A cut per op type (`choice`: op -> its alternative), a statement size and a sample size per subcircuit, the subcircuits
    held at their floor and those proved whole, the expected cost by part (`serve` the fixed serve-time commitments,
    `total` everything, `commit` and `serve` at the model's `commit_weight`), each weight class's weight on the density (1
    when absent), and the certificate once certified."""

    lam: float
    choice: Mapping[str, str]
    statement_size: Mapping[str, int]
    sizes: Mapping[str, int]
    floor_only: frozenset[str]
    full: frozenset[str]
    cost: Mapping[str, float]
    weights: Mapping[str, float] = field(default_factory=dict)
    certificate: PlanBounds | None = field(default=None, compare=False)


def _log_miss(n: int, m: int, k: int) -> float:
    """log C(n - m, k) / C(n, k), that a uniform k-subset of n misses m given proof blocks: minus core's risk
    (`neg_log_escape`), which sums its t = min(m, k) terms while t <= `MISS_TERMS` and past that takes their closed form.
    The closed form puts the log over its exact value by less than s / (8 (n - t + 1)(n - s - t + 1)) at s = max(m, k),
    under 1 / (100 n) of it wherever the miss is at least 2^-40.  It prices a plan and steers `max_harm`'s search; no
    bound reads it."""
    return -inf if k > n - m else -neg_log_escape(n, k, m, MISS_TERMS)


def _miss(n: int, m: int, k: int) -> float:
    return exp(_log_miss(n, m, k))


def _hit(n: int, m: int, k: int) -> float:
    """1 - `_miss`, its digits kept when the sample is sparse."""
    return -expm1(_log_miss(n, m, k))


def touched(n: int, c: int, k: int) -> float:
    """The expected number of statements a uniform k-subset of n proof blocks meets, the proof blocks cut in order into statements of
    c proof blocks (the last one shorter)."""
    if k <= 0:
        return 0.0
    whole, rest = divmod(n, c)
    return whole * _hit(n, c, k) + (_hit(n, rest, k) if rest else 0.0)


def _statement_sizes(cap: int, units: int) -> list[int]:
    top = min(cap, units)
    sizes = {top}
    c = 1
    while c < top:
        sizes.add(c)
        c *= 2
    return sorted(sizes, reverse=True)


def template_cost(c: Candidate, costs: CostModel, k: int) -> tuple[float, int, dict[str, float]]:
    """Sampling k of the candidate's proof blocks: its expected proving cost at the cheapest statement size, that size (the larger
    one on a tie, for fewer roots), and the cost's parts.  Serving's commitments and the replay grains are the plan's."""
    a0, a1 = costs.statement(c)
    units = k * (costs.unit(c) + costs.of(c).replay)
    if costs.shared:
        count = ceil(k / c.cap)
        st = count * a0 + k * a1
        return st + units, ceil(k / count) if count else c.cap, {"statements": st, "units": units}
    best = None
    for size in _statement_sizes(c.cap, c.units):
        st = touched(c.units, size, k) * (a0 + a1 * size)
        if best is None or st < best[0] * (1 - 1e-12):
            best = (st, size)
    return best[0] + units, best[1], {"statements": best[0], "units": units}


def replay_cost(sizes: Mapping[str, int], chosen: Sequence[Candidate], costs: CostModel) -> float:
    """The expected cost of building the sampled proof blocks' witnesses, by replaying each replay block the sample meets: a
    replay grain's replay block is missed when every subcircuit's sample misses its share of it, independently across strata."""
    total = 0.0
    for name in sorted({c.grain for c in chosen if c.grain}):
        if name not in costs.replay:
            raise KeyError(f"the cost model has no replay grain {name!r}")
        rg, log_miss = costs.replay[name], 0.0
        for c in chosen:
            if c.grain == name:
                if c.units != rg.replay_blocks * c.n_rho:
                    raise ValueError(f"{c.template!r}: {c.units} units are not {rg.replay_blocks} replay blocks of {c.n_rho}")
                log_miss += _log_miss(c.units, c.n_rho, sizes[c.template])
        total += rg.replay_blocks * -expm1(log_miss) * rg.cost
    return total


def met_replay(rg: ReplayGrain, draws: float) -> float:
    """A one-stage sample's replay: the expected cost of replaying the replay blocks that `draws` proof blocks, sampled
    independently and uniformly over the grain's replay blocks, meet."""
    if draws <= 0:
        return 0.0
    r = rg.replay_blocks
    return -r * expm1(draws * log1p(-1 / r)) * rg.cost if r > 1 else rg.cost


#: `ln_upper`'s atanh terms and the dyadic it rounds up to (`Spot.Bound.lnTerms`, `lnBits`)
LN_TERMS, LN_BITS = 32, 64


def _two_atanh_upper(t: Fraction) -> Fraction:
    """At least 2 atanh(t) = ln((1 + t) / (1 - t)) for 0 <= t <= 1/3: the series to `LN_TERMS` terms, plus
    2 t^(2N+1) / ((2N + 1)(1 - t^2)), which bounds the rest."""
    s, p, t2 = Fraction(0), t, t * t
    for i in range(LN_TERMS):
        s += p / (2 * i + 1)
        p *= t2
    return 2 * (s + p / ((2 * LN_TERMS + 1) * (1 - t2)))


def ln_upper(delta: Fraction | float) -> Fraction:
    """A rational at least ln(1/delta), for 0 < delta < 1, within about 2^-LN_BITS of it: with 1/delta = 2^j z,
    1 <= z < 2, it is j ln 2 + ln z, each 2 atanh((z - 1)/(z + 1)) (ln 2 at z = 2), rounded up.  Lean's
    `Spot.Bound.lnUpper` is its definition."""
    delta = Fraction(delta)
    if not 0 < delta < 1:
        raise ValueError("0 < delta < 1")
    y = 1 / delta
    j = y.numerator.bit_length() - y.denominator.bit_length()
    if y < 2 ** j:
        j -= 1
    z = y / 2 ** j
    v = j * _two_atanh_upper(Fraction(1, 3)) + _two_atanh_upper((z - 1) / (z + 1))
    return Fraction(ceil(v * 2 ** LN_BITS), 2 ** LN_BITS)


def fill(rows: Iterable[tuple[int, int, float | Fraction]], budget: float | Fraction) -> float | Fraction:
    """The fractional knapsack's value at a risk budget: a stratum of n proof blocks with k sampled and harm h per proof block counts
    whole when k = 0 and not at all when k = n; the rest go in decreasing harm per proof block of risk h n / k, each whole
    (n - k proof blocks at risk k / n each) until the budget runs out, the last one fractionally.  Exact on Fractions, where
    Lean's `Spot.Bound.fill` is its definition."""
    out, items = 0, []
    for n, k, h in rows:
        if h <= 0 or k >= n:
            continue
        if k == 0:
            out += n * h
            continue
        items.append((h * n / k, n - k, n, k, h))
    for _, top, n, k, h in sorted(items, reverse=True):
        x = min(top, budget * n / k)
        out += x * h
        budget = (budget * n - x * k) / n
        if budget <= 0:
            break
    return out


def density_bound(rows: Iterable[tuple[int, int, float]], delta: float) -> float:
    """The most harm an incorrect set accepted with probability at least `delta` holds, when a stratum of n proof blocks with k sampled
    and harm h per proof block, holding x of the set, is accepted with probability at most (1 - x/n)^k <= exp(-k x / n), and never once
    x > n - k.  So sum_s k_s x_s / n_s <= ln(1/delta), and the bound is that fractional knapsack's value (`fill`).  In
    floats, the budget and the value raised by `HARM_MARGIN` (the budget also by 2^-60, for delta near 1), so it is at
    least `density_exact`, its definition."""
    v = fill(rows, -log(delta) * (1 + HARM_MARGIN) + 2.0 ** -60)
    return nextafter(v * (1 + HARM_MARGIN), inf) if v > 0 else 0.0


def density_exact(rows: Iterable[tuple[int, int, float | Fraction]], delta: float | Fraction) -> Fraction:
    """`density_bound` exactly, harms and delta read as rationals: `fill` at `ln_upper(delta)`.  Lean's
    `Spot.Bound.densityBound` is its definition."""
    return fill(((n, k, Fraction(h)) for n, k, h in rows), ln_upper(delta))


#: `ln_bound` works to multiples of `2^-LN_PREC` (`Spot.Bound.lnPrec`)
LN_PREC = 128


def _two_atanh_lower(t: Fraction) -> Fraction:
    """At most 2 atanh(t) for 0 <= t < 1: the series' first `LN_TERMS` terms."""
    s, p, t2 = Fraction(0), t, t * t
    for i in range(LN_TERMS):
        s += p / (2 * i + 1)
        p *= t2
    return 2 * s


_LN2 = (_two_atanh_lower(Fraction(1, 3)), _two_atanh_upper(Fraction(1, 3)))


def ln_bound(a: int, b: int, up: bool) -> Fraction:
    """ln(a / b) for a >= b > 0, a multiple of 2^-LN_PREC below it, or above it when `up`: with a / b = 2^j z,
    1 <= z < 2, j ln 2 + ln z, z first rounded to a multiple of 2^-LN_PREC.  Lean's `Spot.Bound.lnBound` is
    its definition."""
    j = a.bit_length() - b.bit_length()
    if a < b << j:
        j -= 1
    q, d = 1 << LN_PREC, b << j
    m = (a * q + d - 1) // d if up else a * q // d
    t = Fraction(m - q, m + q)
    v = j * _LN2[1] + _two_atanh_upper(t) if up else j * _LN2[0] + _two_atanh_lower(t)
    return Fraction(ceil(v * q) if up else floor(v * q), q)


def acceptance_parts(n: int, k: int, x: int) -> tuple[int, int]:
    """prod_{i < t} (n - s - i) and prod_{i < t} (n - i), t = min(x, k), s = max(x, k), whose ratio is
    C(n - x, k) / C(n, k) = C(n - k, x) / C(n, x), for x <= n - k."""
    a = b = 1
    for i in range(min(x, k)):
        a *= n - max(x, k) - i
        b *= n - i
    return a, b


def max_harm_check(rows: Iterable[tuple[int, int, float | Fraction, int, int, int]], delta: float | Fraction,
                   lam: float | Fraction) -> tuple[Fraction, Fraction]:
    """(lower, upper) around the most harm an incorrect set accepted with probability at least `delta` holds,
    `Flock.Guarantees.JointStratified`'s `StratifiedBlock.maxHarm`, from a certificate: rows (n, k, h, take, lo, hi) and
    a multiplier `lam > 0`.  `lower` is the harm of the counts `take`, whose exact acceptance probability is at least `delta`; `upper`
    the dual bound `lam ln(1/delta) + sum_s max_x (h_s x - lam c_s(x))`, `c(x) = -ln(C(n - x, k)/C(n, k))`, whose
    maximizer the row's lo <= hi place: the increments `d(j) = ln((n - j)/(n - j - k))` before `lo` cost at most `h`,
    those from `hi` on at least `h`.  Raises ValueError where the certificate fails.  Lean's
    `Spot.Bound.maxHarmCheck` is its definition."""
    delta, lam = Fraction(delta), Fraction(lam)
    if lam <= 0:
        raise ValueError("the multiplier is not positive")
    num = den = 1
    lower, upper = Fraction(0), lam * ln_upper(delta)
    for n, k, h, take, lo, hi in rows:
        h, top = Fraction(h), n - k
        if h < 0:
            raise ValueError("a row's harm is negative")
        if top < take:
            raise ValueError("a row takes more than size - k")
        if not lo <= hi <= top:
            raise ValueError("a row needs lo <= hi <= size - k")
        a, b = acceptance_parts(n, k, take)
        num, den = num * a, den * b
        lower += take * h
        if lo > 0 and h < lam * ln_bound(n - lo + 1, top - lo + 1, True):
            raise ValueError("an increment before a row's lo may cost more than its harm")
        if hi < top and lam * ln_bound(n - hi, top - hi, False) < h:
            raise ValueError("an increment from a row's hi on may cost less than its harm")
        a, b = acceptance_parts(n, k, lo)
        gain = max(Fraction(0), h - lam * ln_bound(n - lo, top - lo, False)) if lo < hi else Fraction(0)
        upper += lo * h - lam * ln_bound(b, a, False) + (hi - lo) * gain
    if num * delta.denominator < delta.numerator * den:
        raise ValueError("take escapes with probability below delta")
    return lower, upper


def _taken(n: int, k: int, h: float, lam: float) -> int:
    """How many of a stratum's increments d(j) = -ln(1 - k/(n - j)), j < n - k, cost less than h / lam, in floats."""
    if h <= 0:
        return 0
    if k == 0:
        return n
    q = -expm1(-h / lam)
    v = n - k / q if q > 0 else -inf
    return 0 if v <= 0 else min(n - k, ceil(v))


def max_harm(rows: Iterable[tuple[int, int, float | Fraction]], delta: float | Fraction
             ) -> tuple[Fraction, Fraction, Fraction, list[tuple[int, int, int]]]:
    """`max_harm_check`'s bracket around `StratifiedBlock.maxHarm` for rows (n, k, h) at delta, with the certificate
    found: (lower, upper, lam, [(take, lo, hi)]).  The multiplier is searched in floats, where each stratum's maximizer is
    in closed form, so `upper` is the linear relaxation over the exact risk to the search's precision; `take` is the
    maximizers at the least multiplier found whose counts' exact acceptance probability is at least `delta`; `lo` and `hi` are widened until they hold."""
    rows = [(n, k, Fraction(h)) for n, k, h in rows]
    floats = [(n, k, float(h)) for n, k, h in rows]
    budget, d = -log(float(delta)), Fraction(delta)

    def counts(lam: float) -> list[int]:
        return [_taken(n, k, h, lam) for n, k, h in floats]

    def used(lam: float) -> float:
        return fsum(-_log_miss(n, x, k) for (n, k, _), x in zip(floats, counts(lam)))

    def accepted(xs: list[int]) -> bool:
        num = den = 1
        for (n, k, _), x in zip(rows, xs):
            a, b = acceptance_parts(n, k, x)
            num, den = num * a, den * b
        return num * d.denominator >= d.numerator * den

    lo, hi = -1000.0, 1000.0
    if used(2.0 ** lo) <= budget:
        hi = lo
    for _ in range(200 if hi > lo else 0):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if used(2.0 ** mid) <= budget else (mid, hi)
    lam, scale, step = 2.0 ** hi, 1.0, 2.0 ** -40
    while not accepted(takes := counts(lam * scale)):
        scale, step = scale * (1 + step), step * 4
    cert, exact = [], Fraction(lam)
    for (n, k, h), x, take in zip(rows, counts(lam), takes):
        top = n - k
        a = b = n if k == 0 else 0 if h == 0 else x
        step = 1
        while a > 0 and h < exact * ln_bound(n - a + 1, top - a + 1, True):
            a, step = max(0, a - step), step * 2
        step = 1
        while b < top and exact * ln_bound(n - b, top - b, False) < h:
            b, step = min(top, b + step), step * 2
        cert.append((take, a, b))
    lower, upper = max_harm_check(((n, k, h, *c) for (n, k, h), c in zip(rows, cert)), d, exact)
    return lower, upper, exact, cert


def certify(sizes: Mapping[str, int], candidates: Sequence[Candidate], objective: Objective, *,
            exact: bool = True) -> PlanBounds:
    """The plan's guarantee, from its sizes alone, by core's bounds (and `density_bound`); by `density_bound` alone when not
    `exact`, for populations too large for core's search."""
    chosen = [c for c in candidates if c.template in sizes]
    d = objective.delta

    def dens(h) -> float:
        return density_bound(((c.units, sizes[c.template], h(c)) for c in chosen), d)

    density = dens(objective.harm)
    if not exact:
        return PlanBounds(d, density, density, dens(lambda c: float(c.work + c.influence)), None, False)
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    leak = None
    if objective.bits:
        profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, d)
        leak = exfiltration_bound(profile, {c.template: float(c.out_bits) for c in chosen})
    return PlanBounds(d, harm, density, work, leak)


def _k(c: Candidate, objective: Objective, lam: float, weight: float = 1.0) -> int:
    return min(c.units, max(objective.floor, ceil(min(1.0, lam * weight * objective.harm(c)) * c.units)))


def _plan(chosen: Sequence[Candidate], costs: CostModel, objective: Objective, lam: float,
          floor_only: frozenset[str], full: frozenset[str], weights: Mapping[str, float] | None = None) -> Plan:
    weights = dict(weights or {})
    sizes, statement, parts = {}, {}, {"statements": 0.0, "units": 0.0, "commit": 0.0}
    for c in chosen:
        k = c.units if c.template in full else min(c.units, objective.floor) if c.template in floor_only \
            else _k(c, objective, lam, weights.get(c.block, 1.0))
        _, size, p = template_cost(c, costs, k)
        sizes[c.template], statement[c.template] = k, size
        parts["statements"] += p["statements"]
        parts["units"] += p["units"]
        parts["commit"] += costs.commit(c)
    parts["replay"] = replay_cost(sizes, chosen, costs)
    parts["serve"] = costs.serve_cost()
    parts["total"] = parts["statements"] + parts["units"] + parts["replay"] \
        + costs.commit_weight * (parts["commit"] + parts["serve"])
    choice = {c.op: c.alternative for c in chosen}
    return Plan(lam, choice, statement, sizes, floor_only, full, parts, weights)


def _by_op(candidates: Sequence[Candidate]) -> dict[str, dict[str, list[Candidate]]]:
    if len({c.template for c in candidates}) != len(candidates):
        raise ValueError("candidates need distinct templates")
    ops: dict[str, dict[str, list[Candidate]]] = {}
    for c in candidates:
        ops.setdefault(c.op, {}).setdefault(c.alternative, []).append(c)
    return ops


def _plans_at(ops: Mapping[str, Mapping[str, list[Candidate]]], costs: CostModel, objective: Objective, lam: float,
              promote: float, weights: Mapping[str, float]) -> list[Plan]:
    """The plans the Lagrangian allows at one density: per op type its cheapest cut, then each prefix of the subcircuits
    ranked by cost saved per proof block of harm conceded at their floor, each with and without promoting to full the subcircuits
    whose whole proof costs at most `promote` of the plan more."""
    def k(c: Candidate) -> int:
        return _k(c, objective, lam, weights.get(c.block, 1.0))

    def at(c: Candidate) -> float:
        return costs.commit_weight * costs.commit(c) + template_cost(c, costs, k(c))[0]

    chosen = [c for alts in ops.values()
              for c in min(alts.items(), key=lambda kv: (sum(at(c) for c in kv[1]), kv[0]))[1]]

    def saving(c: Candidate) -> float:
        drop = template_cost(c, costs, k(c))[0] - template_cost(c, costs, min(c.units, objective.floor))[0]
        return drop / (c.units * objective.harm(c))

    ranked = sorted((c for c in chosen if objective.harm(c) > 0 and k(c) > objective.floor),
                    key=lambda c: (-saving(c), c.template))
    plans = []
    for i in range(len(ranked) + 1):
        skip = frozenset(c.template for c in ranked[:i])
        base = _plan(chosen, costs, objective, lam, skip, frozenset(), weights)
        plans.append(base)
        whole = frozenset(c.template for c in chosen if c.template not in skip and base.sizes[c.template] < c.units
                          and template_cost(c, costs, c.units)[0] - template_cost(c, costs, base.sizes[c.template])[0]
                          <= promote * (base.cost["total"] - costs.commit_weight * base.cost["serve"]))
        if whole:
            plans.append(_plan(chosen, costs, objective, lam, skip, whole, weights))
    return plans


def _harm(p: Plan, by: str) -> float:
    return getattr(p.certificate, by)


def _pareto(plans: Iterable[Plan], key) -> list[Plan]:
    out, best = [], float("inf")
    for p in sorted(plans, key=lambda p: (p.cost["total"], key(p))):
        if key(p) < best:
            out.append(p)
            best = key(p)
    return out


def frontier(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, *, by: str = "harm",
             grid: int = 121, promote: float = 0.05, exact: bool = True,
             weights: Mapping[str, float] | None = None) -> list[Plan]:
    """The certified Pareto frontier of (expected cost, `by`), cheapest first, `by` a `PlanBounds` field, each weight
    class's density scaled by its weight (`weights`, 1 when absent).  The sweep's plans are screened by `density_bound`
    (cheap) before the survivors are certified (`certify`, `exact` or not)."""
    weights = dict(weights or {})
    if any(not w > 0 for w in weights.values()):
        raise ValueError("a weight class's weight is positive: hold a class at its floor by its density, not a zero weight")
    ops = _by_op(candidates)
    harms = [objective.harm(c) * c.units for c in candidates if objective.harm(c) > 0]
    if not harms:
        raise ValueError("no candidate does any harm under this objective: there is nothing to sample for")
    ws = [weights.get(c.block, 1.0) for c in candidates]
    lo = 1 / (10 * max(harms) * max(ws))
    hi = 1 / (min(objective.harm(c) for c in candidates if objective.harm(c) > 0) * min(ws))
    lams = [lo * (hi / lo) ** (i / (grid - 1)) for i in range(grid)]
    seen: dict[tuple, Plan] = {}
    for lam in lams:
        for p in _plans_at(ops, costs, objective, lam, promote, weights):
            seen.setdefault((tuple(sorted(p.sizes.items())), tuple(sorted(p.statement_size.items()))), p)

    def screened(p: Plan) -> float:
        chosen = [c for c in candidates if c.template in p.sizes]
        return density_bound(((c.units, p.sizes[c.template], objective.harm(c)) for c in chosen), objective.delta)

    plans = [replace(p, certificate=certify(p.sizes, candidates, objective, exact=exact))
             for p in _pareto(seen.values(), screened)]
    return _pareto(plans, lambda p: _harm(p, by))


def for_budget(plans: Sequence[Plan], budget: float, *, candidates: Sequence[Candidate] = (),
               costs: CostModel | None = None, objective: Objective | None = None, by: str = "harm",
               exact: bool = True) -> Plan | None:
    """The frontier's least-harm plan within `budget`: its costliest affordable plan.  Given the candidates, costs and
    objective, the plans either side of the budget are moved to it by bisection on the density, each holding its cuts,
    floor-only and full subcircuits and weights, and the one certifying less is taken, so the answer doesn't hang on the
    sweep's grid.  The budget is the total, the serve-time commitments included at the model's `commit_weight`."""
    affordable = [p for p in plans if p.cost["total"] <= budget]
    if not affordable:
        return None
    if costs is None or objective is None:
        return affordable[-1]

    def at(p: Plan, lam: float) -> Plan:
        return _plan([c for c in candidates if c.template in p.sizes], costs, objective, lam, p.floor_only, p.full,
                     p.weights)

    def fits_at(p: Plan, lam: float) -> bool:
        return at(p, lam).cost["total"] <= budget

    found = []
    low = affordable[-1]
    lo, hi = low.lam, low.lam * 2
    while fits_at(low, hi) and hi < low.lam * 2.0 ** 64:
        lo, hi = hi, hi * 2
    found.append((low, lo, hi))
    high = next((p for p in plans if p.cost["total"] > budget), None)
    if high is not None:
        lo, hi = high.lam / 2, high.lam
        while not fits_at(high, lo) and lo > high.lam * 2.0 ** -64:
            lo, hi = lo / 2, lo
        if fits_at(high, lo):
            found.append((high, lo, hi))
    out = []
    for p, lo, hi in found:
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if fits_at(p, mid) else (lo, mid)
        q = at(p, lo)
        out.append(replace(q, certificate=certify(q.sizes, candidates, objective, exact=exact)))
    return min(out, key=lambda p: (_harm(p, by), p.cost["total"]))


def for_target(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, target: float, *,
               by: str = "harm", plans: Sequence[Plan] | None = None, exact: bool = True, **kw) -> Plan | None:
    """The cheapest plan certifying at most `target`: the frontier's cheapest such plan (of `plans`, a frontier by `by`,
    when given), its density then lowered by bisection, the cuts, floor-only and full subcircuits and weights held, to
    where its certificate just meets the target."""
    if plans is None:
        plans = frontier(candidates, costs, objective, by=by, exact=exact, **kw)
    ok = [p for p in plans if _harm(p, by) <= target]
    if not ok:
        return None
    best = ok[0]
    chosen = [c for c in candidates if c.template in best.sizes]

    def at(lam: float) -> Plan:
        p = _plan(chosen, costs, objective, lam, best.floor_only, best.full, best.weights)
        return replace(p, certificate=certify(p.sizes, candidates, objective, exact=exact))

    lo, hi = best.lam, best.lam
    for _ in range(64):
        lo /= 2
        if _harm(at(lo), by) > target:
            break
    else:
        return at(lo)
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if _harm(at(mid), by) > target else (lo, mid)
    return at(hi)


def weigh(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, target: float, *, by: str = "harm",
          factors: Sequence[float] = (0.25, 0.5, 2.0, 4.0), rounds: int = 3, exact: bool = True, **kw) -> Plan | None:
    """The cheapest plan certifying at most `target` when each weight class (`Candidate.block`) is sampled at its own weight
    on the density: from equal weights, a coordinate search, each kind of the current plan in turn (costliest first) taking
    the factor of `factors` on its weight that makes `for_target` cheapest, for at most `rounds` passes.  The certificate
    is the same stratified one over every subcircuit, so the weights are parameters of the sample, chosen at the target, and
    never of the guarantee; the plan carries them, and registers by its sizes (`work_law`) as any plan does."""
    def at(weights: Mapping[str, float]) -> Plan | None:
        return for_target(candidates, costs, objective, target, by=by, exact=exact, weights=weights, **kw)

    best = at({})
    if best is None:
        return None
    for _ in range(rounds):
        moved = False
        share: dict[str, float] = {}
        for c in candidates:
            if c.template in best.sizes:
                share[c.block] = share.get(c.block, 0.0) + template_cost(c, costs, best.sizes[c.template])[0]
        for kind in sorted(share, key=lambda b: (-share[b], b)):
            trials = [p for f in factors
                      if (p := at({**best.weights, kind: best.weights.get(kind, 1.0) * f})) is not None]
            cheaper = min(trials, key=lambda p: p.cost["total"], default=None)
            if cheaper is not None and cheaper.cost["total"] < best.cost["total"] * (1 - 1e-9):
                best, moved = cheaper, True
        if not moved:
            break
    return best


def overlaps(costs: CostModel, *, latency: float = 0.0) -> list[dict]:
    """Every tensor that more than one scale commits at serve time (`CostModel.serve`), ranked by what committing it once
    would save at prefill.  Per copy: its bytes and its prefill cost (throughput-bound: its bytes at the model's rate, or
    its measured cost), and at decode its bytes, its chain and, when it gates the step, its chain's latency at `latency`
    a dependent compression (latency-bound).  Committed once, the cheapest copy is kept: the saving is the others' bytes
    and prefill cost, and at decode their latency less the kept copy's chain where it has to gate in their place."""
    by: dict[str, list[Commitment]] = {}
    for s in costs.serve:
        by.setdefault(s.tensor, []).append(s)
    rows = []
    for tensor, copies in by.items():
        if len({s.scale for s in copies}) < 2:
            continue
        keep = min(copies, key=lambda s: (costs.committed(s), s.scale))
        drop = [s for s in copies if s is not keep]
        path = fsum(s.chain for s in copies if s.gates) * latency
        once = keep.chain * latency if any(s.gates for s in copies) else 0.0
        rows.append({"tensor": tensor, "kept": keep.scale,
                     "copies": [{"scale": s.scale, "bytes": s.nbytes, "prefill": costs.committed(s),
                                 "decode_bytes": s.decode_bytes, "chain": s.chain, "gates": s.gates,
                                 "decode": s.chain * latency if s.gates else 0.0} for s in copies],
                     "saving": {"bytes": fsum(s.nbytes for s in drop), "prefill": fsum(costs.committed(s) for s in drop),
                                "decode_bytes": fsum(s.decode_bytes for s in drop), "decode": path - once}})
    return sorted(rows, key=lambda r: (-r["saving"]["prefill"], r["tensor"]))


def structure(p: Plan) -> tuple:
    """A plan's discrete shape: its cuts, statement sizes, and floor-only and whole subcircuits."""
    return (tuple(sorted(p.choice.items())), tuple(sorted(p.statement_size.items())), tuple(sorted(p.floor_only)),
            tuple(sorted(p.full)))


def perturb(costs: CostModel, what: str, factor: float, candidates: Sequence[Candidate] = ()) -> CostModel:
    """The model with one group of costs scaled by `factor`: "prove" (per proved proof block), "statement" (per touched
    statement), "commit" (serving's, the proof blocks' and the fixed serve-time commitments'), "replay" (the domains' and per
    sampled proof block), or one subcircuit's (its derived costs pinned from `candidates` first)."""
    t = dict(costs.templates)
    if what == "prove":
        t = {k: replace(v, unit=None if v.unit is None else v.unit * factor) for k, v in t.items()}
        return replace(costs, templates=t, per_and=costs.per_and * factor)
    if what == "statement":
        t = {k: replace(v, statement_fixed=None if v.statement_fixed is None else v.statement_fixed * factor,
                        statement_per_unit=None if v.statement_per_unit is None else v.statement_per_unit * factor)
             for k, v in t.items()}
        return replace(costs, templates=t, statement_fixed=costs.statement_fixed * factor,
                       statement_per_and=costs.statement_per_and * factor, staged_per_and=costs.staged_per_and * factor,
                       staged_per_unit=costs.staged_per_unit * factor)
    if what == "commit":
        return replace(costs, commit_per_bit=costs.commit_per_bit * factor, commit_per_row=costs.commit_per_row * factor,
                       serve=tuple(replace(s, cost=None if s.cost is None else s.cost * factor) for s in costs.serve))
    if what == "replay":
        t = {k: replace(v, replay=v.replay * factor) for k, v in t.items()}
        return replace(costs, templates=t, replay={k: replace(d, cost=d.cost * factor) for k, d in costs.replay.items()})
    c = next((c for c in candidates if c.template == what), None)
    if c is None:
        raise KeyError(f"nothing to perturb named {what!r}: prove, statement, commit, replay or a candidate's template")
    a0, a1 = costs.statement(c)
    tc = costs.of(c)
    t[what] = TemplateCost(a0 * factor, a1 * factor, costs.unit(c) * factor, tc.replay * factor, costs.sigma_of(c))
    return replace(costs, templates=t)


def sensitivity(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, budget: float, *,
                factors: Mapping[str, Sequence[float]] | None = None, by: str = "harm",
                **kw) -> tuple[Plan | None, list[dict]]:
    """The plan at `budget` and how it moves when one group of costs is scaled: per (group, factor), the certified harm,
    the cost and whether the plan's `structure` holds.  By default each nonzero group of the model at 1/2 and 2."""
    if factors is None:
        groups = {"prove": costs.per_and or any(v.unit for v in costs.templates.values()),
                  "statement": costs.statement_fixed or costs.statement_per_and or costs.staged_per_and
                  or costs.staged_per_unit or any(v.statement_fixed or v.statement_per_unit
                                                  for v in costs.templates.values()),
                  "commit": costs.commit_per_bit or costs.commit_per_row or bool(costs.serve),
                  "replay": bool(costs.replay)}
        factors = {g: (0.5, 2.0) for g, on in groups.items() if on}
    def at(model: CostModel) -> Plan | None:
        return for_budget(frontier(candidates, model, objective, by=by, **kw), budget, candidates=candidates,
                          costs=model, objective=objective, by=by, exact=kw.get("exact", True))

    base = at(costs)
    rows = []
    for what, fs in factors.items():
        for f in fs:
            p = at(perturb(costs, what, f, candidates))
            rows.append({"what": what, "factor": f, "harm": None if p is None else _harm(p, by),
                         "cost": None if p is None else p.cost["total"],
                         "same": p is not None and base is not None and structure(p) == structure(base)})
    return base, rows


def needs_benchmark(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, budget: float, *,
                    by: str = "harm", **kw) -> list[str]:
    """The subcircuits worth benchmarking at `budget`: those whose cost, moved by its own uncertainty either way (a
    measurement's `sigma`, or the model's for a derived cost), changes the plan's structure."""
    factors = {c.template: (1 / (1 + costs.sigma_of(c)), 1 + costs.sigma_of(c)) for c in candidates
               if costs.sigma_of(c) > 0}
    if not factors:
        return []
    _, rows = sensitivity(candidates, costs, objective, budget, factors=factors, by=by, **kw)
    return sorted({r["what"] for r in rows if not r["same"]})


def work_law(plan: Plan, candidates: Sequence[Candidate]) -> dict:
    """The plan as a work law: `work:1` over a work table carrying the credited work and floors equal to the plan's
    sizes, which `one_stage.draw.work_k` samples exactly (the ceiling is at most 1 since w n <= W). It is a fixed count
    per stratum, which a stream's rule refuses."""
    chosen = [c for c in candidates if c.template in plan.sizes]
    if not any(c.work for c in chosen):
        raise ValueError("a work law needs a population that does some work")
    return {"law": "work:1", "table": {c.template: {"work": c.work, "floor": plan.sizes[c.template]} for c in chosen}}


#: `CostModel.commit_weight`'s proving-only mode, the model of Lean's `GranularityOptimum`
PROVING_ONLY = 0.0


@dataclass(frozen=True)
class Grain:
    """One uniform partition of a window, as `Partitioning.Model` states one: `units` proof blocks (n) of `work` each (s),
    each `out_bits` wide (b, what an incorrect one exports), committing `commit_bytes` at serve time whether sampled or not (b(s),
    its share of every byte the window commits under this partition), and costing `prove` when sampled (c(s), its
    statement's share included)."""

    name: str
    units: int
    work: float
    out_bits: float
    commit_bytes: float
    prove: float

    def __post_init__(self) -> None:
        if self.units < 1 or not self.work > 0 or not min(self.out_bits, self.commit_bytes, self.prove) >= 0:
            raise ValueError(f"grain {self.name!r}: needs units >= 1, work > 0, and non-negative widths and costs")


def loc_bits(n: int, k: int) -> float:
    """At least Lean's `locBits n K`, log2 of sum_{j <= K} C(n, j), to the floats' rounding: 0 at K = 0, n H2(K/n) up to
    K = n/2 (`consumers.entropy_bits`) and n past it (`consumers.location_bits`' bound, before its margin)."""
    if k <= 0:
        return 0.0
    if 2 * k >= n:
        return float(n)
    return entropy_bits(n, k)


def leak_count(g: Grain, ell: float) -> int:
    """`ExfiltrationOptimum`'s K: the most incorrect proof blocks a leak target of `ell` bits allows, the largest K <= n with
    b K + locBits(n, K) <= ell, by `loc_bits`, so the K it returns is certified."""
    if not ell >= 0:
        raise ValueError("a leak target is at least 0 bits")

    def ok(k: int) -> bool:
        return g.out_bits * k + loc_bits(g.units, k) <= ell

    lo, hi = 0, g.units
    if ok(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def work_draws(g: Grain, target: float, delta: float) -> float:
    """`Achievability`'s law on equal proof blocks for the free-compute target C: each sampled with probability
    min(1, Λ s / C), so (ns/C) Λ samples, `GranularityOptimum`'s cost over c(s), while s Λ <= C, and every one past it."""
    return g.units * min(1.0, log(1 / delta) * g.work / target)


def leak_draws(g: Grain, ell: float, delta: float) -> float:
    """`ExfiltrationOptimum`'s achieving law for a leak target of `ell` bits: n Λ / (K + 1) samples, at most n, and none once
    every proof block may be incorrect."""
    k = leak_count(g, ell)
    return 0.0 if k >= g.units else min(float(g.units), g.units * log(1 / delta) / (k + 1))


def grain_costs(grains: Iterable[Grain], delta: float, *, h: float, weight: float = 1.0, work: float | None = None,
                leak: float | None = None, replay: ReplayGrain | None = None) -> list[dict]:
    """Each partition's expected one-stage cost at one target, the free-compute target `work` (C) or the exfiltration
    target `leak` (bits), at `delta`, in the grains' order: `prove`, c(s) times the target's samples; `commit`, n b(s) h,
    paid on every proof block; `replay`, with a replay grain, the replay blocks the samples meet (`met_replay`); and
    `total`, proving and replay plus `weight` times the commitment (`PROVING_ONLY` prices proving alone, Lean's
    model).  A two-stage design is `two_stage_design`'s."""
    if (work is None) == (leak is None):
        raise ValueError("one target: work (C) or leak (bits)")
    if not 0 < delta < 1 or not weight >= 0 or not h >= 0 or (work is not None and not work > 0):
        raise ValueError("needs 0 < delta < 1, weight >= 0, h >= 0 and a positive work target")
    rows = []
    for g in grains:
        draws = work_draws(g, work, delta) if work is not None else leak_draws(g, leak, delta)
        prove, commit = draws * g.prove, g.units * g.commit_bytes * h
        row = {"grain": g.name, "work": g.work, "units": g.units, "draws": draws, "prove": prove,
               "commit": commit, "total": prove + weight * commit}
        if replay is not None:
            r = met_replay(replay, draws)
            row |= {"design": "one-stage", "replay": r, "total": row["total"] + r}
        rows.append(row)
    return rows


def grain_optimum(grains: Iterable[Grain], delta: float, **kw) -> dict:
    """`grain_costs`' cheapest partition, the first of the grains' order on a tie."""
    rows = grain_costs(grains, delta, **kw)
    if not rows:
        raise ValueError("no partition to choose from")
    return min(rows, key=lambda r: r["total"])


def sqrt_grain(beta: float, h: float, target: float, o: float, delta: float, weight: float = 1.0) -> float:
    """The free-compute target's optimal proof-block work when c(s) = f + o s and every proof block commits `beta` bytes:
    the s minimizing (f + o s)(F/C) Λ + w (F/s) β h, which is s* = √(w β h C / (o Λ)), whatever F and f, while s* Λ <= C."""
    return sqrt(weight * beta * h * target / (o * log(1 / delta)))


#: the breakpoints of `two_stage_exfil`'s chords of the escape exponent h; past the last, h is taken at its value there.
#: Dense from 3 to 8, where a wholly incorrect replay block's y = Σ_t k_t falls once there are a few strata
CHORDS = (0.0, 2.0 ** -8, 2.0 ** -4, 2.0 ** -2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
#: `two_stage_design`'s family of designs, one at each κ.  Down to 2^-10: capped at a request's served bits, B is the
#: capped charge, so the patterns' form's best designs sit at κ of 2^-9 to 2^-6
KAPPAS = (0.0, *(2.0 ** e for e in range(-10, 5)))
#: the tolerance in ln θ `two_stage_design` compares its designs at, each bound holding at any θ; the chosen one's is
#: then taken to `two_stage_exfil`'s
SCREEN = 1e-2
_LN2F = log(2.0)


def _softplus(x: float) -> float:
    return x + log1p(exp(-x)) if x > 0 else log1p(exp(x))


def _log_expm1(x: float) -> float:
    if x <= 0:
        return -inf
    return x + log1p(-exp(-x)) if x > 1 else log(expm1(x))


def _logsumexp(xs: Sequence[float]) -> float:
    top = max(xs)
    return top if top == -inf else top + log(fsum(exp(x - top) for x in xs))


@dataclass(frozen=True)
class Cell:
    """A proof-block type's blocks in each replay block, its part of `Claims.Sampling.cell` (ρ, t) for its stratum t:
    `n` of them, each committing `bits` (its summand of `bitsOf`) and `commit_bytes` once a sampled replay block's
    interior is committed, and proved at `prove` a block beside its statement's `fixed` cost, a statement holding at
    most `cap` of the type's sampled blocks.  `table` is the type's proof blocks one table of the proof system holds,
    where the cell states it: a session of more is a session of several tables."""

    n: float
    bits: float
    stratum: int = 0
    prove: float = 0.0
    fixed: float = 0.0
    cap: float = 1.0
    commit_bytes: float = 0.0
    table: float = inf

    def __post_init__(self) -> None:
        if (not self.n > 0 or not self.cap >= 1 or not self.table >= 1
                or not min(self.bits, self.prove, self.fixed, self.commit_bytes) >= 0):
            raise ValueError(f"a cell needs n > 0, cap >= 1, table >= 1, and non-negative bits and costs: {self}")


def _strata(cells: Sequence[Cell], k: Mapping[int, float]) -> dict[int, float]:
    """Each stratum's proof blocks in a replay block, n_t, refusing a `k` without k_t >= 1 in each (`WellFormed`)."""
    out: dict[int, float] = {}
    for c in cells:
        out[c.stratum] = out.get(c.stratum, 0.0) + c.n
    if not out or any(not k.get(t, 0) >= 1 for t in out):
        raise ValueError(f"a well-formed design samples k >= 1 in each stratum: k {dict(k)} over strata {sorted(out)}")
    return out


def _log_expm1_exp(z: float) -> float:
    """At least ln(e^{e^z} − 1), finite however small e^z is: ln z' + z' at z' = e^z once it is below 2^-40, as
    e^{z'} − 1 <= z' e^{z'}."""
    if z < -28:
        return z + exp(z)
    return _log_expm1(exp(z)) if z < 700 else inf


def _log_grow(gs: Iterable[tuple[float, float, float]]) -> float:
    """At least ln(∏ (1 + e^x)^n − 1) over (n, ln n, x), finite however small the e^x: once Σ n ln(1 + e^x) is below
    2^-40, ln(e^{Σ n e^x} − 1) by `_log_expm1_exp`."""
    gs = list(gs)
    total = fsum(n * _softplus(x) for n, _, x in gs)
    if total > 2.0 ** -40:
        return _log_expm1(total)
    return _log_expm1_exp(_logsumexp([ln_n + x for _, ln_n, x in gs])) if gs else -inf


def _log_head(ln_x: float, m: float) -> float:
    """ln Σ_{1<=j<=m} x^j/j!, at most e^x − 1 and, once x > m, its last term times min(m, 1/(1 − m/x)) (the terms
    rise to it, each at most m/x of the next)."""
    if m < 1:
        return -inf
    out = _log_expm1_exp(ln_x)
    if m < inf and ln_x > (ln_m := log(m)):
        out = min(out, m * ln_x - lgamma(m + 1) + min(ln_m, -log(-expm1(ln_m - ln_x))))
    return out


def _capped(g: tuple[float, ...], lw: float) -> tuple[float, float]:
    """A group of n proof blocks of b bits each (`g`, from `two_stage_exfil`) at weight w = e^{lw}: ln of Light =
    Σ_{1<=m<=m*} C(n, m)(2^b w)^m and of Heavy = Σ_{m>m*} C(n, m) w^m, at m* = ⌊S/b⌋ the most of them a set of at most
    S bits holds, each at most (1 + 2^b w)^n − 1 or (1 + w)^n − 1 and the least of its closed forms by
    C(n, m) <= n^m/m!."""
    n, b, m, ln_n, lg_m, ln_m2 = g[:6]
    light = -inf
    if m >= 1:
        ln_na = ln_n + b + lw
        light = _log_grow([(n, ln_n, b + lw)])
        if ln_na > (ln_m := log(m)):
            light = min(light, m * ln_na - lg_m + min(ln_m, -log(-expm1(ln_m - ln_na))))
    heavy = _log_grow([(n, ln_n, lw)])
    if (ln_nw := ln_n + lw) < ln_m2:
        heavy = min(heavy, (m + 1) * ln_nw - lg_m - log(m + 1) - log1p(-exp(ln_nw - ln_m2)))
    return light, heavy


def two_stage_exfil(cells: Iterable[Cell], k: Mapping[int, float], replay_blocks: int, q: float, delta: float,
                    served: float | None = None, union: bool = False, tol: float = 1e-6, fanout: int = 1) -> float:
    """`two_stage_rankin`'s bits."""
    return two_stage_rankin(cells, k, replay_blocks, q, delta, served, union, tol, fanout)[0]


def two_stage_rankin(cells: Iterable[Cell], k: Mapping[int, float], replay_blocks: int, q: float, delta: float,
                     served: float | None = None, union: bool = False, tol: float = 1e-6,
                     fanout: int = 1) -> tuple[float, float]:
    """At least log2 of `CompiledTwo.Exfil`'s count, Σ_{T ∈ admissibleTwo} 2^{b(T)}, over `replay_blocks` replay blocks
    (R) of the cells each, the first stage sampling each replay block with probability q and the second `k[t]` of
    stratum t's n_t proof blocks in a sampled one.

    The first stage's sample is Bernoulli, so T's acceptance is P̄_a = ∏_ρ (1 − q(1 − e_ρ)), and a cell's
    `escCell(n, k, m) = C(n − m, k)/C(n, k) <= (1 − m/n)^k <= e^{−k m/n}` (0 once k >= n and m >= 1): P̄_a <=
    exp(−Σ_ρ h(y_ρ)), at y_ρ = Σ r_v over T's proof blocks in ρ, r_v = k_t/n_t (∞ once k_t >= n_t), and h(y) =
    −ln(1 − q(1 − e^{−y})).  An admissible T has Σ_ρ h(y_ρ) < Λ, so Rankin's bound at any θ > 0 gives
    Σ_{T adm} 2^{b(T)} <= e^{θΛ} G^R, G = Σ_{T ⊆ ρ} 2^{b(T)} e^{−θ h(y(T))}.  h is concave, increasing and 0 at 0: past
    `CHORDS[-1]` = Y_J it is at least h(Y_J), and between two breakpoints at least its chord a_j + s_j y, a_j >= 0, so
    G − 1 <= Σ_j e^{−θ a_j}(∏_v (1 + 2^{b_v} e^{−θ s_j r_v}) − 1) + e^{−θ h(Y_J)}(∏_{r_v<∞} (1 + 2^{b_v}) − 1)
    + e^{−θ h(∞)} ∏_{r_v<∞} (1 + 2^{b_v}) (∏_{r_v=∞} (1 + 2^{b_v}) − 1), each product over a replay block's proof
    blocks, a T meeting a stratum proved whole at h(∞).  The bound takes the θ minimizing it, by golden section on ln θ
    to within `tol` (it is convex in θ, and holds at any θ), and is at most R log2 ∏_v (1 + 2^{b_v}), every T counted
    (θ = 0).  It returns the bits and that θ.

    `served`, S, caps each replay block's charge at the bits it serves (the capped form,
    `Properties.Spot.TwoStageCheckServedSound`, whose `N` is `fanout`).  `fanout`, F, is the replay grain's fan-out
    (`partition.fanout`): the requests reading in a replay block serve at most F S bits together, so S is read as F S.
    F is 1 when every replay block is closed, each recomputing from its own inputs any value another computed while
    serving; the patterns' form is priced only at closed replay blocks, so it refuses F > 1.  The sets' form
    counts Σ_{T adm} 2^{Σ_ρ min(b(T_ρ), S)}: each product's factor over a group of n blocks of b bits and one rate
    becomes Σ_m C(n, m) w^m 2^{min(m b, S)} <= 1 + Light + 2^S Heavy (`_capped`), by min(Σ_g x_g, S) <= Σ_g min(x_g, S),
    and each sum over nonempty T is also at most 2^S (∏_v (1 + w_v) − 1), so a wholly incorrect replay block of N proof
    blocks is charged about N + S bits, the sets it holds and what it serves.  The patterns' form (`union`) counts
    the served outputs' patterns instead: a replay block's T_ρ either holds at most S bits, counted as itself, or
    more, where all of them together serve at most 2^S outputs and each has y_ρ >= y_S = min_v r_v max(1, S/b_v) (as
    Σ_v min(b_v, S) >= S over its blocks), so G <= Σ_{b(T)<=S} 2^{b(T)} e^{−θ h(y(T))} + 2^S e^{−θ h(y_S)}, the first sum bounded as above with each factor
    1 + Light and also by 2^S Σ_{1<=j<=S/b_min} W^j/j!, W = Σ_v w_v."""
    cells = list(cells)
    if not 0 <= q <= 1 or not 0 < delta < 1 or replay_blocks < 1:
        raise ValueError("needs 0 <= q <= 1, 0 < delta < 1 and replay_blocks >= 1")
    if (served is None and union) or (served is not None and not served >= 0):
        raise ValueError("a served-output cap is at least 0 bits, and the patterns' form needs one")
    if not fanout >= 1 or (union and fanout > 1):
        raise ValueError(f"a fan-out is at least 1, and the patterns' form is priced only where no replay block reads "
                         f"a value another computed, so it needs fan-out 1, not {fanout}")
    served = None if served is None else served * fanout
    strata = _strata(cells, k)
    groups: dict[tuple[float, float], float] = {}
    for c in cells:
        n = strata[c.stratum]
        key = (c.bits, inf if k[c.stratum] >= n else k[c.stratum] / n)
        groups[key] = groups.get(key, 0.0) + c.n
    cap = 0.0 if served is None else served * _LN2F
    parts = []
    for (bits, r), n in groups.items():
        m = inf if served is None or bits == 0 or served >= n * bits else floor(served / bits)
        parts.append(((n, bits * _LN2F, m, log(n), lgamma(m + 1) if m < inf else 0.0, log(m + 2),
                       log(m / 2) if 0 < m < inf else -inf), r))

    def factor(g: tuple[float, ...], lw: float) -> float:
        """ln of the group's factor at weight e^{lw}: uncapped, or where the cap binds (its blocks' mean count under
        the tilted weights, n a/(1 + a), above m*/2), the least of that and its capped form."""
        la = g[1] + lw
        whole = g[0] * (sp := _softplus(la))
        if g[2] == inf or g[3] + la - sp <= g[6]:
            return whole
        light, heavy = _capped(g, lw)
        return min(whole, _softplus(light) if union else _logsumexp((0.0, light, cap + heavy)))

    narrow = min(bits for bits, _ in groups)
    most = inf if served is None or narrow == 0 else floor(served / narrow)

    def light(gs: list[tuple[tuple[float, ...], float]]) -> float:
        """ln of Σ over a replay block's nonempty T of these groups of T's charge times its weight: 2^{b(T)}, capped
        at 2^S, or under `union` over the T of at most S bits; capped, also at most 2^S (∏_v (1 + w_v) − 1)."""
        out = _log_expm1(fsum(factor(g, lw) for g, lw in gs))
        if served is not None and gs:
            out = min(out, cap + _log_grow([(g[0], g[3], lw) for g, lw in gs]))
            if union and most < inf:
                out = min(out, cap + _log_head(_logsumexp([g[3] + lw for g, lw in gs]), most))
        return out

    full = fsum(factor(g, 0.0) for g, _ in parts)
    over = served is not None and fsum(n * bits for (bits, _), n in groups.items()) > served
    if union and over:
        full = _logsumexp((_softplus(light([(g, 0.0) for g, _ in parts])), cap))
    elif served is not None:
        full = min(full, (fsum(groups.values()) + served) * _LN2F)
    every = replay_blocks * full / _LN2F
    if q == 0 or full == 0:
        return every, 0.0
    lam = log(1 / delta)

    def h_at(y: float) -> float:
        return inf if q == 1 and y == inf else -log1p(q * expm1(-y))

    hs = [h_at(y) for y in CHORDS]
    chords = [((h1 - h0) / (y1 - y0), h0 - (h1 - h0) / (y1 - y0) * y0)
              for y0, y1, h0, h1 in zip(CHORDS, CHORDS[1:], hs, hs[1:])]
    finite = [(g, r) for g, r in parts if r < inf]
    partial = fsum(factor(g, 0.0) for g, _ in finite)
    tail = light([(g, 0.0) for g, _ in finite])
    whole = partial + _log_expm1(fsum(factor(g, 0.0) for g, r in parts if r == inf))
    if served is not None:
        whole = min(whole, cap + fsum(g[0] for g, _ in finite) * _LN2F
                    + _log_expm1(fsum(g[0] for g, r in parts if r == inf) * _LN2F))
    h_whole = h_at(inf)
    heavy, h_heavy = -inf, inf
    if union and over:
        heavy, h_heavy = cap, h_at(min(r * max(1.0, served / bits) if bits > 0 else inf for bits, r in groups))

    def f(u: float) -> float:
        theta = exp(u)
        terms = [tail - theta * hs[-1], whole - theta * h_whole if whole > -inf else -inf,
                 heavy - theta * h_heavy if heavy > -inf else -inf]
        for s, a in chords:
            terms.append(light([(g, -theta * s * r) for g, r in finite]) - theta * a)
        return theta * lam + replay_blocks * _softplus(_logsumexp(terms))

    scales = [(full + 1 + log(replay_blocks)) / hs[-1]]
    scales += [(g[1] + 1 + log(g[0] * replay_blocks)) / (q * r) for g, r in finite]
    lo, hi = log(min(scales)) - 8, log(max(scales)) + 8
    gold = (sqrt(5) - 1) / 2
    c, d = hi - gold * (hi - lo), lo + gold * (hi - lo)
    fc, fd = f(c), f(d)
    while hi - lo > tol:
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - gold * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gold * (hi - lo)
            fd = f(d)
    best, u = min((fc, c), (fd, d))
    return (best / _LN2F, exp(u)) if best / _LN2F < every else (every, 0.0)


def registration_count(q: float, registration: float) -> int:
    """The replay blocks the first stage samples of a verification phase of `registration` of them at the rate q:
    exactly ⌈q·registration⌉ (`Spot.Rate.count` at one table), so at least one at any positive rate and none at 0."""
    if not 0 <= q <= 1 or not registration >= 1:
        raise ValueError(f"a count needs 0 <= q <= 1 and a verification phase of at least one replay block: "
                         f"{q}, {registration}")
    if not q > 0:
        return 0
    x = q * registration
    return max(1, round(x) if abs(x - round(x)) <= 4 * ulp(x) else ceil(x))


#: how a design's sampled proof blocks are split into sessions: `statements`, the cost model's own (a type's blocks of
#: one sampled replay block, or of a verification phase's, in as many sessions of at most `cap` blocks as they
#: need); `template`, one session a template of a verification phase whatever it holds; `table`, one session a
#: template of at most one table (`Cell.table`), the one form the verifier's second stage reads
#: (`Spot.SecondStage`: `second-template`, `second-tables`)
SESSIONS = ("statements", "template", "table")


def two_stage_statements(cells: Iterable[Cell], k: Mapping[int, float], sampled: int = 0,
                         sessions: str = "statements") -> list[dict]:
    """Each cell's statements, a sampled replay block's share of them: its stratum t's share of min(k_t, n_t) proof
    blocks (`taken`), their `marginal` proving, and `prove`, with its statements' fixed costs.  A sampled replay block's
    own statements hold at most `cap` of a type's blocks, a statement's `fixed` cost paid once (below one block, at the
    chance the type is sampled).  A verification phase that samples exactly `sampled` replay blocks may instead hold a
    type's blocks of all of them, the statement's shape fixed before any coin because the count is: split into sessions
    of at most `cap` blocks padded to one size.  A type takes the verification phase's where they cost less (`shared`),
    and `padding` is its proving of padding.  With no `sampled`, every statement is a sampled replay block's own.
    `sessions` is the type's sessions, a sampled replay block's share of them: the verifier's second stage reads one
    session of one table a template of a verification phase (`Spot.SecondStage`, `second-template` and
    `second-tables`), so a type with more than one in a phase is priced in sessions it refuses.  `largest` is the
    type's largest session in proof blocks and `tables` that session in tables of the type (`Cell.table`; 0 where the
    cell states none).

    Under `sessions` "template" or "table" (`SESSIONS`) each type's blocks of the verification phase's `sampled`
    replay blocks are one session, its `fixed` cost paid once and every block at `prove`, whatever the session holds:
    a session of more than one table is priced at the same seconds a block, which nothing measures and no prover
    builds.  "table" differs only in the designs that take it (`two_stage_design` keeps each session to one table)."""
    if sessions not in SESSIONS:
        raise ValueError(f"sessions are one of {SESSIONS}: {sessions!r}")
    if sessions != "statements" and not sampled >= 1:
        raise ValueError("one session a template is a verification phase's: it needs the count its first stage samples")
    cells = list(cells)
    strata = _strata(cells, k)
    out = []
    for c in cells:
        taken = c.n * min(1.0, k[c.stratum] / strata[c.stratum])
        marginal = taken * c.prove
        if sessions != "statements":
            blocks = ceil(sampled * taken - 1e-9)
            out.append({"taken": taken, "marginal": marginal, "prove": (c.fixed + blocks * c.prove) / sampled,
                        "shared": True, "padding": max(0.0, blocks * c.prove / sampled - marginal),
                        "sessions": 1 / sampled, "largest": blocks, "tables": blocks / c.table})
            continue
        own = taken / max(1.0, min(c.cap, taken))
        largest = max(1, ceil(min(c.cap, taken) - 1e-9))
        row = {"taken": taken, "marginal": marginal, "prove": marginal + c.fixed * own, "shared": False,
               "padding": 0.0, "sessions": own, "largest": largest, "tables": largest / c.table}
        if sampled and taken > 0:
            blocks = ceil(sampled * taken - 1e-9)
            count = ceil(blocks / c.cap)
            padded = count * ceil(blocks / count) * c.prove / sampled
            if (shared := count * c.fixed / sampled + padded) < row["prove"]:
                row |= {"prove": shared, "shared": True, "padding": max(0.0, padded - marginal),
                        "sessions": count / sampled, "largest": ceil(blocks / count),
                        "tables": ceil(blocks / count) / c.table}
        out.append(row)
    return out


def two_stage_price(cells: Iterable[Cell], k: Mapping[int, float], sampled: int = 0,
                    sessions: str = "statements") -> tuple[float, float, float]:
    """A sampled replay block's proved proof blocks, their marginal proving and its proving with its statements' fixed
    costs and padding (`two_stage_statements`)."""
    rows = two_stage_statements(cells, k, sampled, sessions)
    return fsum(r["taken"] for r in rows), fsum(r["marginal"] for r in rows), fsum(r["prove"] for r in rows)


def registration_sample(cells: list[Cell], k: Mapping[int, float], registration: float, replay: float,
                        left: float, sessions: str = "statements") -> int:
    """The most replay blocks K a verification phase of `registration` of them samples at a cost of at most `left` a
    replay block of it: the largest whole K <= `registration` with K (`replay` + `two_stage_price`'s proving at K) <=
    `left`·`registration`, that cost never falling in K, by bisection; 0 when one sampled replay block is over it.  The
    first stage samples no fraction of a replay block and, at any positive rate, at least one
    (`registration_count`), so below that cost no design fits."""
    if not registration >= 1:
        raise ValueError(f"a verification phase holds at least one replay block: {registration}")

    def fits(x: int) -> bool:
        return x * (replay + two_stage_price(cells, k, x, sessions)[2]) <= left * registration

    if not fits(1):
        return 0
    lo, hi = 1, floor(registration)
    if replay > 0:
        hi = min(hi, floor(left * registration / replay))
    if fits(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def _size_class(bits: float) -> int:
    if (w := unit_class(ceil(bits))) is None:
        raise ValueError(f"a proof block of {bits} bits fits no public size class (<= {UNIT_CLASS_MAX} bits)")
    return w


def two_stage_strata(cells: Iterable[Cell], merged: int | str | None) -> list[Cell]:
    """The cells in the strata of a `two_stage_design` row whose `merged` is this: the public size classes
    (`unit_class`) up to `merged` in stratum 0 and each wider class in the stratum of its bits; `merged` None, each
    distinct width in a stratum of its own, named by the width's place among the cells' widths in increasing order; or
    `merged` "template", a by-template design's, each cell the stratum of its position."""
    cells = list(cells)
    if merged == "template":
        return [replace(c, stratum=i) for i, c in enumerate(cells)]
    if merged is None:
        place = {bits: i for i, bits in enumerate(sorted({c.bits for c in cells}))}
        return [replace(c, stratum=place[c.bits]) for c in cells]
    return [replace(c, stratum=0 if (w := _size_class(c.bits)) <= merged else w) for c in cells]


def two_stage_design(cells: Iterable[Cell], replay_blocks: int, delta: float, *, budget: float, replay: float,
                     widths: bool = False,
                     margin: float = 0.0,
                     kappas: Sequence[float] | None = None,
                     boundary: float = 0.0, h: float = 0.0, stratify: bool | str = True, served: float | None = None,
                     union: bool = False, registration: float = 0.0, fanout: int = 1,
                     sessions: str = "statements") -> dict | None:
    """The two-stage design with the least `two_stage_exfil` bound at a cost of at most `budget`: width-stratified
    (`stratify`) or uniform (one stratum).  A width-stratified design merges the public size classes (`unit_class`) up
    to a threshold into stratum 0 and gives each wider class a stratum of its own, the threshold chosen with κ: at the
    narrowest class every class is its own stratum, and at the widest the design is the uniform one, so it is never
    worse.  A class's stratum is sampled at the rate its widest block asks, so with `widths` the family also holds the
    designs that give each distinct width a stratum of its own (`merged` None, `two_stage_strata`): a block narrower
    than its class's widest is then sampled at its own bits' rate, and the design is again never worse.  At each κ of
    `KAPPAS` it samples k_t = κ n_t (b_t + ℓ_t)/B of stratum t's n_t proof blocks, at least 1 and at
    most n_t, at b_t the stratum's widest bits, ℓ_t = log2(e R n_t) and B a replay block's bits: at κ = 1 one incorrect
    proof block of a stratum's widest, with its location, leaks as much a unit of escape as a wholly incorrect replay
    block does.  With `stratify` "template" each cell is a stratum of its own, the strata the verifier's two-stage
    rule registers (`Spot.Select.Counts`: one whole count a template of the check): the row's `k` is each cell's
    count by its position in `cells`, its `merged` "template", and `two_stage_counts` writes it by name.  Under a
    served-output cap (`served`, `union` and `fanout`, as `two_stage_exfil` takes them) b_t is at
    most the cap read at the fan-out and B is a replay block's capped charge, every T counted.

    The first stage is the one the verifier runs: of a verification phase's `registration` replay blocks it samples
    exactly K, a whole number of at least one (`registration_count`), so the design is priced and bounded at the rate
    q = K/`registration`, never at a rate between two counts.  K is the most (`registration_sample`) that what
    `boundary` (serving's boundary hashing) leaves of the budget pays for, each sampled replay block costing `replay`,
    its interior's `commit_bytes` at `h` a byte, and `two_stage_price`'s proving at K: a statement holds one sampled
    replay block's proof blocks of one Definition, or the verification phase's K sampled ones' where that costs less
    (`two_stage_statements`).  A κ at which one sampled replay block a verification phase is over the budget has no
    design.  With no `registration`, one verification phase holds every replay block and every statement is a sampled
    replay block's own.

    `sessions` (`SESSIONS`) is how the sample is split into sessions.  "statements" is the cost model's own, above.
    "template" and "table" are a by-template design's with a `registration`: each template's sampled proof blocks of
    a verification phase in one session.  Under "template" the session holds whatever the design samples, and K is
    again the most the budget pays for.  Under "table" it holds at most one table of the template's proof blocks
    (`Cell.table`), K k_t <= table_t in every template: the design searches K over the powers of two (and the most
    the smallest table allows), each count held to its table, and no K above a template's table has a design.  Both
    are priced at a proof block's seconds and one fixed cost a session (`two_stage_statements`).

    With `margin` m > 0 the family also holds, at each threshold and κ, the design that samples (1 + m) k_t, at most
    n_t, of every stratum but the one whose statements cost the most proving at those k: where one stratum is most of
    the proving a margin on the rest costs little, and the design is again never worse.  With `kappas`, some of
    `KAPPAS`, the family holds only the designs at those κ: k_t grows with κ, so a caller that holds down the proof
    blocks a sampled replay block proves pins it.

    Its row:
    `merged` (the widest size class in stratum 0, None where each width is a stratum, "template" where each cell is),
    `kappa`, `margin` (m where its k carry it, else 0), `k`, `q`, `registration`, `registration_sample` (K, the
    replay blocks a verification phase samples; 0 with no `registration`), `shared` and `padding` (their shares of
    proving), `sessions` (a verification phase's sessions, `two_stage_statements`'s, over all its sampled replay
    blocks), `form` (the `sessions` asked for), `largest` and `tables` (the largest session in proof blocks, and the
    largest in tables of its type) and `over` (the templates with a session of more than one table),
    `bound` (bits) and the Rankin `theta` it takes, `replay_sample`, `draws` (proved proof blocks), `prove`, `replay`,
    `boundary`, `interior`, `commit` (their sum), `total` and `fill` (proving over its marginal); None when the boundary
    alone is over the budget or no κ has a design.  The bound's Lean takes the first stage through
    `Claims.Sampling.AdmissibleTwoRankin` alone: it misses each set of replay blocks with probability at most the
    product of 1 − q over them, which a uniform K of n does at q = K/n."""
    cells = list(cells)
    if not margin >= 0:
        raise ValueError(f"a margin is at least 0, not {margin}")
    if kappas is not None and not (kappas and set(kappas) <= set(KAPPAS)):
        raise ValueError(f"a design's κ are some of KAPPAS, not {tuple(kappas)}")
    if not cells or not budget >= boundary:
        return None
    if stratify not in (True, False, "template"):
        raise ValueError(f"a design is width-stratified (True), uniform (False) or by template: {stratify!r}")
    if sessions not in SESSIONS:
        raise ValueError(f"sessions are one of {SESSIONS}: {sessions!r}")
    if sessions != "statements" and not (stratify == "template" and registration):
        raise ValueError("one session a template is a by-template design's, over a verification phase "
                         "(`stratify` \"template\" and a `registration`)")
    if sessions == "table" and any(c.table == inf for c in cells):
        raise ValueError("a session of one table needs each cell's `table`, its type's proof blocks a table holds")
    classes = {_size_class(c.bits) for c in cells}
    by_template = stratify == "template"
    thresholds: list[int | str | None] = ["template"] if by_template else sorted(classes) if stratify else [max(classes)]
    if stratify is True and widths:
        thresholds.append(None)
    charge = fsum(c.n * c.bits for c in cells)
    if served is not None:
        charge = two_stage_exfil(cells, {c.stratum: 1 for c in cells}, 1, 0.0, 0.5, served, union, fanout=fanout)
    interior = fsum(c.n * c.commit_bytes for c in cells) * h
    rows, seen, done = [], set(), set()
    tables: list[int] = []
    if sessions == "table":
        top = floor(min(min(c.table for c in cells), registration))
        tables = sorted({x for x in [2 ** e for e in range(top.bit_length())] + [top] if 1 <= x <= top})
    for merged, m in [(t, x) for t in thresholds for x in ((0.0, margin) if margin > 0 else (0.0,))]:
        split = two_stage_strata(cells, merged)
        strata: dict[int, float] = {}
        widest: dict[int, float] = {}
        for c in split:
            strata[c.stratum] = strata.get(c.stratum, 0.0) + c.n
            widest[c.stratum] = max(widest.get(c.stratum, 0.0), min(c.bits, inf if served is None else served * fanout))
        for kappa in KAPPAS:
            if kappas is not None and kappa not in kappas:
                continue
            k = {t: min(ceil(n), max(1, ceil(kappa * n * (widest[t] + 1 / _LN2F + log2(replay_blocks * n)) / charge)))
                 if charge > 0 else 1 for t, n in strata.items()}
            if m:
                cost: dict[int, float] = {}
                for c, r in zip(split, two_stage_statements(split, k)):
                    cost[c.stratum] = cost.get(c.stratum, 0.0) + r["prove"]
                most = max(cost, key=cost.__getitem__)
                k = {t: v if t == most else min(ceil(strata[t]), ceil((1 + m) * v)) for t, v in k.items()}
            if (key := (merged, *sorted(k.items()))) in seen:
                continue
            seen.add(key)
            left = (budget - boundary) / replay_blocks
            if sessions == "table":
                picks = [(x, held) for x in tables for held in [{c.stratum: max(1, min(k[c.stratum], floor(c.table / x)))
                                                                  for c in split}]
                         if x * (replay + interior + two_stage_price(split, held, x, sessions)[2])
                         <= left * registration]
            elif registration:
                picks = [(registration_sample(split, k, registration, replay + interior, left, sessions), k)]
            else:
                per = replay + interior + two_stage_price(split, k)[2]
                picks = [(floor(min(replay_blocks, left * replay_blocks / per)) if per > 0 else floor(replay_blocks),
                          k)]
            for count, k in picks:
                if not count or (key := (merged, count, *sorted(k.items()))) in done:
                    continue
                done.add(key)
                q = count / (registration or replay_blocks)
                st = two_stage_statements(split, k, count if registration else 0, sessions)
                blocks, marginal, prove = (fsum(r[x] for r in st) for x in ("taken", "marginal", "prove"))
                per = replay + interior + prove
                sampled = q * replay_blocks
                rows.append({"design": "template" if by_template else "stratified" if stratify else "uniform",
                             "merged": merged, "kappa": kappa, "margin": m,
                             "k": k, "q": q, "registration": registration,
                             "registration_sample": count if registration else 0,
                             "shared": fsum(r["prove"] for r in st if r["shared"]) / prove if prove > 0 else 0.0,
                             "padding": fsum(r["padding"] for r in st) / prove if prove > 0 else 0.0,
                             "sessions": count * fsum(r["sessions"] for r in st), "form": sessions,
                             "largest": max(r["largest"] for r in st), "tables": max(r["tables"] for r in st),
                             "over": sum(1 for r in st if r["tables"] > 1), "split": split,
                             "bound": two_stage_exfil(split, k, replay_blocks, q, delta, served, union, SCREEN,
                                                      fanout),
                             "replay_sample": sampled, "draws": sampled * blocks, "prove": sampled * prove,
                             "replay": sampled * replay, "boundary": boundary, "interior": sampled * interior,
                             "commit": boundary + sampled * interior, "total": boundary + sampled * per,
                             "fill": prove / marginal if marginal > 0 else 1.0})
    if not rows:
        return None
    best = min(rows, key=lambda r: (r["bound"], r["total"]))
    split = best.pop("split")
    bound, theta = two_stage_rankin(split, best["k"], replay_blocks, best["q"], delta, served, union, fanout=fanout)
    return best | {"bound": bound, "theta": theta}


def two_stage_counts(row: Mapping, names: Sequence[str]) -> dict:
    """A by-template design (`two_stage_design` at `stratify` "template") as the verifier's two-stage rule takes it:
    `sampled`, the replay blocks the first stage samples of a verification phase, and `k`, each template's whole
    count of proof blocks in a sampled replay block, by its name in `names`, the design's cells in order
    (`Spot.Select.Counts.per`, which names a template by its descriptor id).  A design of another kind has no such
    counts: a stratum of several templates samples its count among them."""
    k = row["k"]
    if row.get("design") != "template" or len(names) != len(k) or len(set(names)) != len(names):
        raise ValueError(f"the counts are a by-template design's, one name a cell: a {row.get('design')} design of "
                         f"{len(k)} strata and {len(names)} names, {len(set(names))} distinct")
    if any(k[i] != int(k[i]) or k[i] < 1 for i in range(len(names))):
        raise ValueError(f"a template's count is a whole number of at least one: {dict(k)}")
    return {"sampled": int(row["registration_sample"]), "k": {name: int(k[i]) for i, name in enumerate(names)}}


__all__ = ["Candidate", "Commitment", "CostModel", "Grain", "HM96_ANDS", "LN_BITS", "LN_PREC", "LN_TERMS",
           "Objective", "PROVING_ONLY", "Plan", "PlanBounds", "ReplayGrain", "SHA512_ANDS", "SHA512_ROWS",
           "TemplateCost",
           "CHORDS", "Cell", "KAPPAS", "acceptance_parts",
           "certify", "density_bound", "density_exact", "fill", "for_budget", "for_target",
           "frontier", "grain_costs", "grain_optimum", "leak_count", "leak_draws", "ln_bound", "ln_upper", "loc_bits",
           "max_harm", "max_harm_check", "met_replay", "needs_benchmark", "overlaps", "perturb", "pinned_ands",
           "registration_count", "registration_sample", "SESSIONS", "replay_cost", "row_ands", "sensitivity", "sqrt_grain", "structure", "template_cost",
           "touched", "two_stage_counts", "two_stage_design", "two_stage_exfil", "two_stage_price", "two_stage_rankin",
           "two_stage_statements", "two_stage_strata", "weigh",
           "work_draws", "work_law"]
