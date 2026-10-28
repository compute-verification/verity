"""A planner for the sampling scheme: from a Program's partition candidates, a proof system's cost model and an objective,
the plans on the Pareto frontier of expected cost against certified harm, over every budget.

Nothing trusts it.  A plan is a stratified sampling design, `k_s` proof blocks of each subcircuit, which registers as `work:1` over a work
table carrying the credited work and floors equal to the sizes (`work_law`: `one_stage.draw.work_k(1, w, n, W, f)` is
`min(n, f)`).  Its certificate is recomputed from the sizes alone (`certify`): core's `Stratified.harm_bound` for the
objective's harm and for credited work, and `density_bound`, the closed form that `work_escape_le`'s proof extends to any
sizes (the note's gap G1).

**Cuts.**  An op type is cut into proof blocks one of several ways, each a set of subcircuits (`Candidate.cut`): one subcircuit per
op is a partition of it; a shared piece is either its own subcircuit, committed once and read by the others, or
recomputed inside each reader (a covering, `recomputed_across` in `Q_word` v2), and the planner picks per density.  A proof block
whose wrong output can forfeit downstream work (a noise or form proof block every dot of its row reads) carries that work as
`influence`, which its harm counts.

**Cost.**  Per touched statement plus per proved proof block, as C-Flock does: subcircuit `s` cut into statements of `c` proof blocks costs
`touched_s(k; c) * (a0 + a1 c) + k * (b + replay)`, a statement being touched when the draw meets it.  Everything is
committed at serve time whatever the draw: every proof block's outputs (`commit_bits`, `commit_rows`),
and the fixed serve-time commitments of every scale (`Commitment`: the Commit's taps, PoUW's work blocks, the remote audit's
receipts, the bits a work block pins), so a plan's cost is the whole serve-time and proving total.  One tensor committed
at several scales is counted once per scale, and `overlaps` prices what committing it once would save.  A drawn proof block's
witness is built by re-executing its replay block from the last committed state (`ReplayDomain`), which commits nothing.
A subcircuit's costs are measured (`TemplateCost`) or derived from the model's rates and its features: ANDs, and the hm96
rows a proved proof block opens (`row_ands`).

**Sweep.**  The note's Lagrangian: one density `lam` on every subcircuit (`k_s = ceil(min(1, lam w_b v_s) n_s)`, `w_b` the
weight of the subcircuit's weight class, 1 unless the plan weighs classes), some subcircuits at their floor only, some proved
whole, and the cut per op type and the statement size each the cheapest at that `lam`.  With no statement cost and a proof block
cost proportional to work, the cheapest plan certifying `C` is `Achievability`'s rates, `q_s = min(1, ln(1/delta) w_s / C)`.
`weigh` chooses the kinds' weights at a target, under the same stratified certificate.

**Modes.**  `CostModel.commit_weight` is the weight w on serving's commitments in a plan's total: at 1 they count as spent,
and at 0 (`PROVING_ONLY`) the plan prices proving alone, the model of Lean's `GranularityOptimum` (the cheapest law costs
`c(s) · (F/C) · Λ`).  `Grain` is that model on one uniform partition, with the commitment term beside it: `n · b(s) · h`
for `n` proof blocks of `b(s)` committed bytes at `h` a hashed byte, paid whether a proof block is drawn or not.
`grain_costs` prices a ladder of partitions for the free-compute target (`Achievability`'s law, `GranularityOptimum`'s
cost) or the exfiltration target (`ExfiltrationOptimum`'s law), and `sqrt_grain` is the work target's optimum in closed
form, `s* = √(w β h C / (o Λ))`, when `c(s) = f + o s` and every proof block commits `β` bytes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import ceil, exp, expm1, floor, fsum, inf, lgamma, log, log1p, log2, nextafter, sqrt

from verity.core.protocols.profile import HARM_MARGIN, IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound
from verity.core.protocols.sampled_proofs.one_stage.partition import UNIT_CLASS_MAX, unit_class

#: ANDs of one SHA-512 compression in Boolean form (hm96 PROTOCOL §5)
SHA512_ANDS = 58_120
#: the hiding part of an hm96-sha512 row in circuit: two private compressions (hm96 PROTOCOL §5)
HM96_ANDS = 116_240


def row_ands(nbytes: int) -> int:
    """In-circuit ANDs to open one hm96-sha512 row of `nbytes` bytes: its digest's private blocks, the length and padding
    included, and the hiding part (hm96 PROTOCOL §6's table)."""
    return HM96_ANDS + SHA512_ANDS * ceil((nbytes + 17) / 128)


@dataclass(frozen=True)
class Candidate:
    """One subcircuit of one way (`cut`, default the subcircuit alone) to cut one op type into proof blocks, derived from the IR,
    never from the prover: `units` instances, each credited `work` (the verifier's work table), with `out_bits` over
    `out_gates` boundary gates, its committed bits, which fit a public size class (`one_stage.partition.unit_class`, at
    most `UNIT_CLASS_MAX`, as the one-stage partition refuses a wider proof block), and `influence`, the credited work a wrong
    one can forfeit downstream.  A proved proof block costs `ands` plus opening `rows` (bytes each) in circuit; serving commits
    `commit_bits` in `commit_rows` rows per proof block whatever the sampling design; one statement holds at most `cap` proof blocks
    (m = k_log + nbl <= 35); `per_replay` proof blocks of it lie in each replay block of domain `replay`; and `kind` is its weight
    class for the draw's weights (served proof blocks of one subcircuit, PoUW's tiles), the subcircuit itself when empty."""

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
    replay: str = ""
    per_replay: int = 0
    kind: str = ""

    def __post_init__(self) -> None:
        if not self.op or not self.template or self.units < 1 or self.cap < 1 or self.out_bits < 0 or self.out_gates < 0 \
                or min(self.work, self.influence, self.ands, self.commit_bits, self.commit_rows, *self.rows, 0) < 0 \
                or unit_class(self.out_bits) is None or (bool(self.replay) != (self.per_replay > 0)) \
                or self.per_replay > self.units:
            raise ValueError(f"candidate {self.template!r}: needs units >= 1, cap >= 1, non-negative sizes, outputs in a "
                             f"size class (<= {UNIT_CLASS_MAX} bits), and per_replay in [1, units] with a replay")

    @property
    def alternative(self) -> str:
        return self.cut or self.template

    @property
    def block(self) -> str:
        return self.kind or self.template


@dataclass(frozen=True)
class TemplateCost:
    """One subcircuit's measured costs under one proof-system version, in the caller's currency: `statement_fixed` (a0) and
    `statement_per_unit` (a1, per staged proof block) per touched statement, `unit` (b) per proved proof block, `replay` per drawn proof block,
    and `sigma`, their relative uncertainty.  A field left None is derived from the model's rates."""

    statement_fixed: float | None = None
    statement_per_unit: float | None = None
    unit: float | None = None
    replay: float = 0.0
    sigma: float = 0.0


@dataclass(frozen=True)
class ReplayDomain:
    """`units` replay blocks (one layer of one step each, say), each re-executed from the last committed state at `cost`
    to build the witnesses of the drawn proof blocks in it, when the draw meets any.  Replay commits nothing: every value it
    recomputes was committed at serve time."""

    units: int
    cost: float


@dataclass(frozen=True)
class Commitment:
    """One tensor committed at serve time by one scale, whatever the draw: `nbytes` hashed over the window, at the model's
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
    proved proof block `per_and * ands`.  When `shared`, a template's statements hold only its drawn proof blocks (`IntegrityPeriod`'s
    hidden window: one session, its k drawn proof blocks split evenly over `ceil(k / cap)` statements), so only they are staged;
    otherwise its proof blocks are cut in order into statements and every touched statement is staged whole.  Serving commits each proof block's outputs at `commit_per_bit` and `commit_per_row`, and
    `serve`, every other serve-time commitment, which every plan pays; `replay` names the replay domains that build
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
    replay: Mapping[str, ReplayDomain] = field(default_factory=dict)
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
    """The harm a faulty proof block does: `work` times its credited work and influence plus `bits` times its output bits.  No
    target: targets and budgets are queries on the frontier.  Every subcircuit draws at least `floor` proof blocks."""

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
class Certificate:
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


@dataclass(frozen=True)
class Plan:
    """A cut per op type (`choice`: op -> its alternative), a statement size and a draw size per subcircuit, the subcircuits
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
    certificate: Certificate | None = field(default=None, compare=False)


def _log_miss(n: int, m: int, k: int) -> float:
    """log C(n - m, k) / C(n, k), that a uniform k-subset of n misses m given proof blocks: the product
    prod_{i < t} (1 - s / (n - i)) over t = min(m, k), s = max(m, k), summed when t is small, by `lgamma` up to 2^25 proof blocks
    and past that, where `lgamma`'s differences lose their digits, by its midpoint (either within about 1e-7 relative)."""
    if k > n - m:
        return -inf
    if m == 0 or k == 0:
        return 0.0
    t, s = min(m, k), max(m, k)
    if t <= 64:
        return fsum(log1p(-s / (n - i)) for i in range(t))
    if n <= 1 << 25:
        return lgamma(n - m + 1) - lgamma(n - m - k + 1) - lgamma(n + 1) + lgamma(n - k + 1)
    return t * log1p(-s / (n - (t - 1) / 2))


def _miss(n: int, m: int, k: int) -> float:
    return exp(_log_miss(n, m, k))


def _hit(n: int, m: int, k: int) -> float:
    """1 - `_miss`, its digits kept when the draw is sparse."""
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
    """Drawing k of the candidate's proof blocks: its expected proving cost at the cheapest statement size, that size (the larger
    one on a tie, for fewer roots), and the cost's parts.  Serving's commitments and the replay domains are the plan's."""
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
    """The expected cost of building the drawn proof blocks' witnesses, by replaying each replay block the draw meets: a domain's
    replay block is missed when every subcircuit's draw misses its share of it, independently across strata."""
    total = 0.0
    for name in sorted({c.replay for c in chosen if c.replay}):
        if name not in costs.replay:
            raise KeyError(f"the cost model has no replay domain {name!r}")
        dom, log_miss = costs.replay[name], 0.0
        for c in chosen:
            if c.replay == name:
                if c.units != dom.units * c.per_replay:
                    raise ValueError(f"{c.template!r}: {c.units} units are not {dom.units} replay units of {c.per_replay}")
                log_miss += _log_miss(c.units, c.per_replay, sizes[c.template])
        total += dom.units * -expm1(log_miss) * dom.cost
    return total


#: `ln_upper`'s atanh terms and the dyadic it rounds up to (`SampledProofs.Bound.lnTerms`, `lnBits`)
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
    `SampledProofs.Bound.lnUpper` is its definition."""
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
    """The fractional knapsack's value at a risk budget: a stratum of n proof blocks with k drawn and harm h per proof block counts
    whole when k = 0 and not at all when k = n; the rest go in decreasing harm per proof block of risk h n / k, each whole
    (n - k proof blocks at risk k / n each) until the budget runs out, the last one fractionally.  Exact on Fractions, where
    Lean's `SampledProofs.Bound.fill` is its definition."""
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
    """The most harm a faulty set accepted with probability at least `delta` holds, when a stratum of n proof blocks with k drawn
    and harm h per proof block, holding x of the set, is accepted with probability at most (1 - x/n)^k <= exp(-k x / n), and never once
    x > n - k.  So sum_s k_s x_s / n_s <= ln(1/delta), and the bound is that fractional knapsack's value (`fill`).  In
    floats, the budget and the value raised by `HARM_MARGIN` (the budget also by 2^-60, for delta near 1), so it is at
    least `density_exact`, its definition."""
    v = fill(rows, -log(delta) * (1 + HARM_MARGIN) + 2.0 ** -60)
    return nextafter(v * (1 + HARM_MARGIN), inf) if v > 0 else 0.0


def density_exact(rows: Iterable[tuple[int, int, float | Fraction]], delta: float | Fraction) -> Fraction:
    """`density_bound` exactly, harms and delta read as rationals: `fill` at `ln_upper(delta)`.  Lean's
    `SampledProofs.Bound.densityBound` is its definition."""
    return fill(((n, k, Fraction(h)) for n, k, h in rows), ln_upper(delta))


#: `ln_bound` works to multiples of `2^-LN_PREC` (`SampledProofs.Bound.lnPrec`)
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
    1 <= z < 2, j ln 2 + ln z, z first rounded to a multiple of 2^-LN_PREC.  Lean's `SampledProofs.Bound.lnBound` is
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
    """(lower, upper) around the most harm a faulty set accepted with probability at least `delta` holds,
    `Flock.Guarantees.JointStratified`'s `StratifiedBlock.maxHarm`, from a certificate: rows (n, k, h, take, lo, hi) and
    a multiplier `lam > 0`.  `lower` is the harm of the counts `take`, whose exact acceptance probability is at least `delta`; `upper`
    the dual bound `lam ln(1/delta) + sum_s max_x (h_s x - lam c_s(x))`, `c(x) = -ln(C(n - x, k)/C(n, k))`, whose
    maximizer the row's lo <= hi place: the increments `d(j) = ln((n - j)/(n - j - k))` before `lo` cost at most `h`,
    those from `hi` on at least `h`.  Raises ValueError where the certificate fails.  Lean's
    `SampledProofs.Bound.maxHarmCheck` is its definition."""
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
            exact: bool = True) -> Certificate:
    """The plan's guarantee, from its sizes alone, by core's bounds (and `density_bound`); by `density_bound` alone when not
    `exact`, for populations too large for core's search."""
    chosen = [c for c in candidates if c.template in sizes]
    d = objective.delta

    def dens(h) -> float:
        return density_bound(((c.units, sizes[c.template], h(c)) for c in chosen), d)

    density = dens(objective.harm)
    if not exact:
        return Certificate(d, density, density, dens(lambda c: float(c.work + c.influence)), None, False)
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    leak = None
    if objective.bits:
        profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, d)
        leak = exfiltration_bound(profile, {c.template: float(c.out_bits) for c in chosen})
    return Certificate(d, harm, density, work, leak)


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
    """The certified Pareto frontier of (expected cost, `by`), cheapest first, `by` a `Certificate` field, each weight
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
    """The cheapest plan certifying at most `target` when each weight class (`Candidate.block`) is drawn at its own weight
    on the density: from equal weights, a coordinate search, each kind of the current plan in turn (costliest first) taking
    the factor of `factors` on its weight that makes `for_target` cheapest, for at most `rounds` passes.  The certificate
    is the same stratified one over every subcircuit, so the weights are parameters of the draw, chosen at the target, and
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
    drawn proof block), or one subcircuit's (its derived costs pinned from `candidates` first)."""
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
    """The plan as the rule the verifier registers today: `work:1` over a work table carrying the credited work and
    floors equal to the plan's sizes, which `one_stage.draw.work_k` draws exactly (the ceiling is at most 1 since
    w n <= W)."""
    chosen = [c for c in candidates if c.template in plan.sizes]
    if not any(c.work for c in chosen):
        raise ValueError("a work law needs a population that does some work")
    return {"law": "work:1", "table": {c.template: {"work": c.work, "floor": plan.sizes[c.template]} for c in chosen}}


#: `CostModel.commit_weight`'s proving-only mode, the model of Lean's `GranularityOptimum`
PROVING_ONLY = 0.0


@dataclass(frozen=True)
class Grain:
    """One uniform partition of a window, as `Partitioning.Model` states one: `units` proof blocks (n) of `work` each (s),
    each `out_bits` wide (b, what a wrong one exports), committing `commit_bytes` at serve time whether drawn or not (b(s),
    its share of every byte the window commits under this partition), and costing `prove` when drawn (c(s), its
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
    """At least Lean's `locBits n K`, log2 of sum_{j <= K} C(n, j): 0 at K = 0, n H2(K/n) up to K = n/2 and n past it
    (`consumers.location_bits`' bound)."""
    if k <= 0:
        return 0.0
    if 2 * k >= n:
        return float(n)
    x = k / n
    return n * (-x * log2(x) - (1 - x) * log2(1 - x))


def leak_count(g: Grain, ell: float) -> int:
    """`ExfiltrationOptimum`'s K: the most wrong proof blocks a leak target of `ell` bits allows, the largest K <= n with
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
    """`Achievability`'s law on equal proof blocks for the free-compute target C: each drawn with probability
    min(1, Λ s / C), so (ns/C) Λ draws, `GranularityOptimum`'s cost over c(s), while s Λ <= C, and every one past it."""
    return g.units * min(1.0, log(1 / delta) * g.work / target)


def leak_draws(g: Grain, ell: float, delta: float) -> float:
    """`ExfiltrationOptimum`'s achieving law for a leak target of `ell` bits: n Λ / (K + 1) draws, at most n, and none once
    every proof block may be wrong."""
    k = leak_count(g, ell)
    return 0.0 if k >= g.units else min(float(g.units), g.units * log(1 / delta) / (k + 1))


def grain_costs(grains: Iterable[Grain], delta: float, *, h: float, weight: float = 1.0, work: float | None = None,
                leak: float | None = None) -> list[dict]:
    """Each partition's expected cost at one target, the free-compute target `work` (C) or the exfiltration target `leak`
    (bits), at `delta`, in the grains' order: `prove`, c(s) times the target's draws; `commit`, n b(s) h, paid on every
    proof block; and `total`, proving plus `weight` times the commitment (`PROVING_ONLY` prices proving alone, Lean's
    model)."""
    if (work is None) == (leak is None):
        raise ValueError("one target: work (C) or leak (bits)")
    if not 0 < delta < 1 or not weight >= 0 or not h >= 0 or (work is not None and not work > 0):
        raise ValueError("needs 0 < delta < 1, weight >= 0, h >= 0 and a positive work target")
    rows = []
    for g in grains:
        draws = work_draws(g, work, delta) if work is not None else leak_draws(g, leak, delta)
        prove, commit = draws * g.prove, g.units * g.commit_bytes * h
        rows.append({"grain": g.name, "work": g.work, "units": g.units, "draws": draws, "prove": prove,
                     "commit": commit, "total": prove + weight * commit})
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


__all__ = ["Candidate", "Certificate", "Commitment", "CostModel", "Grain", "HM96_ANDS", "LN_BITS", "LN_PREC", "LN_TERMS",
           "Objective", "PROVING_ONLY", "Plan", "ReplayDomain", "SHA512_ANDS", "TemplateCost", "acceptance_parts",
           "certify", "density_bound", "density_exact", "fill", "for_budget", "for_target", "frontier", "grain_costs",
           "grain_optimum", "leak_count", "leak_draws", "ln_bound", "ln_upper", "loc_bits", "max_harm",
           "max_harm_check", "needs_benchmark", "overlaps", "perturb", "replay_cost", "row_ands", "sensitivity",
           "sqrt_grain", "structure", "template_cost", "touched", "weigh", "work_draws", "work_law"]
