"""A planner for the sampling scheme: from a Program's partition candidates, a proof system's cost model and an objective,
the plans on the Pareto frontier of expected cost against certified harm, over every budget
(note:proofs/20261018T0335Z-draft-sampling-optimizer-design).

Nothing trusts it.  A plan is a stratified law, `k_s` units of each template, which registers as `work:1` over a work
table carrying the credited work and floors equal to the sizes (`work_law`: `one_stage.draw.work_k(1, w, n, W, f)` is
`min(n, f)`).  Its certificate is recomputed from the sizes alone (`certify`): core's `Stratified.harm_bound` for the
objective's harm and for credited work, and `density_bound`, the closed form that `work_escape_le`'s proof extends to any
sizes (the note's gap G1).

**Cuts.**  An op type is cut into units one of several ways, each a set of templates (`Candidate.cut`): one template per
op is a partition of it; a shared subcircuit is either its own template, committed once and read by the others, or
recomputed inside each reader (a covering, `recomputed_across` in `Q_word` v2), and the planner picks per density.  A unit
whose wrong output can forfeit downstream work (a noise or form unit every dot of its row reads) carries that work as
`influence`, which its harm counts.

**Cost.**  Per touched statement plus per proved unit, as C-Flock does: template `s` cut into statements of `c` units costs
`touched_s(k; c) * (a0 + a1 c) + k * (b + replay)`, a statement being touched when the draw meets it.  Everything is
committed at serve time whatever the draw (Daniel, 18 Oct, ASK-25): every unit's outputs (`commit_bits`, `commit_rows`),
and the fixed serve-time commitments of every scale (`Commitment`: the Commit's taps, PoUW's work blocks, the remote audit's
receipts, the bits a work block pins), so a plan's cost is the whole serve-time and proving total.  One tensor committed
at several scales is counted once per scale, and `overlaps` prices what committing it once would save.  A drawn unit's
witness is built by re-executing its replay unit from the last committed state (`ReplayDomain`), which commits nothing.
A template's costs are measured (`TemplateCost`) or derived from the model's rates and its features: ANDs, and the hm96
rows a proved unit opens (`row_ands`).

**Sweep.**  The note's Lagrangian: one density `lam` on every template (`k_s = ceil(min(1, lam w_b v_s) n_s)`, `w_b` the
weight of the template's block kind, 1 unless the plan weighs kinds), some templates that change no value (no output bits)
at their floor only, some proved whole, and the cut per op type and the statement size each the cheapest at that `lam`.
With no statement cost and a unit cost proportional to work, the cheapest plan certifying `C` is `Achievability`'s rates,
`q_s = min(1, ln(1/delta) w_s / C)`.  `weigh` chooses the kinds' weights at a target, under the same stratified certificate.

**Floor.**  A template that changes values never sits at its floor: drawn `floor` of its n_s units whatever the density, a
wrong unit escapes with 1 - floor / n_s, so its undetected pool grows with the fleet (register row P5).  The default
objective weighs every unit's output bits beside its work, and a plan that would hold a value-changing template at its
floor, at zero harm or among the floor-only templates, is refused (`refuse_floor`).

**Policies.**  An objective may carry policies instead (`Policy`: a public cost per unit, an amplification and a target,
note:proofs/20261018T2305Z-draft-iop-core-objects's tier 2).  A unit's harm is then the largest share of a target it
spends, so the draw is weighted by the envelope of the policies' costs, a plan certifying harm 1 meets every policy, and
`Certificate.policies` says which one binds.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import ceil, exp, expm1, fsum, inf, lgamma, log, log1p

from verity.ml.circuits.cut import fits
from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound

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
    """One template of one way (`cut`, default the template alone) to cut one op type into units, derived from the IR,
    never from the prover: `units` instances, each credited `work` (the verifier's work table), with `out_bits` over
    `out_gates` boundary gates (the width rule, `cut.fits`), and `influence`, the credited work a wrong one can forfeit
    downstream.  A proved unit costs `ands` plus opening `rows` (bytes each) in circuit; serving commits `commit_bits` in
    `commit_rows` rows per unit whatever the law; one statement holds at most `cap` units (m = k_log + nbl <= 35);
    `per_replay` units of it lie in each replay unit of domain `replay`; and `kind` is its block kind for the draw's
    weights (served proof units of one template, PoUW's tiles), the template itself when empty."""

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
                or not fits(self.out_bits, self.out_gates) or (bool(self.replay) != (self.per_replay > 0)) \
                or self.per_replay > self.units:
            raise ValueError(f"candidate {self.template!r}: needs units >= 1, cap >= 1, non-negative sizes, outputs within "
                             f"the width rule (<= 16 bits, or one gate of <= 32), and per_replay in [1, units] with a replay")

    @property
    def alternative(self) -> str:
        return self.cut or self.template

    @property
    def block(self) -> str:
        return self.kind or self.template


@dataclass(frozen=True)
class TemplateCost:
    """One template's measured costs under one proof-system version, in the caller's currency: `statement_fixed` (a0) and
    `statement_per_unit` (a1, per staged unit) per touched statement, `unit` (b) per proved unit, `replay` per drawn unit,
    and `sigma`, their relative uncertainty.  A field left None is derived from the model's rates."""

    statement_fixed: float | None = None
    statement_per_unit: float | None = None
    unit: float | None = None
    replay: float = 0.0
    sigma: float = 0.0


@dataclass(frozen=True)
class ReplayDomain:
    """`units` replay units (one layer of one step each, say), each re-executed from the last committed state at `cost`
    to build the witnesses of the drawn units in it, when the draw meets any.  Replay commits nothing: every value it
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
    """Measured templates, and the rates that derive the rest from their features, `ands` being a unit's ANDs with its
    rows' openings (`row_ands`), as `costfit` measures them: per touched statement `statement_fixed + statement_per_and *
    ands` (the session and the verifier's parse), per staged unit `staged_per_and * ands + staged_per_unit`, and per
    proved unit `per_and * ands`.  Serving commits each unit's outputs at `commit_per_bit` and `commit_per_row`, and
    `serve`, every other serve-time commitment, which every plan pays; `replay` names the replay domains that build
    witnesses; `sigma` is a derived cost's relative uncertainty; `source` is the fit's `art:` id."""

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
class Policy:
    """One policy's bound read off the certificate (note:proofs/20261018T2305Z-draft-iop-core-objects, tier 2): a wrong
    unit costs `work` times its credited work and influence plus `bits` times its output bits plus `unit`, and the
    policy's damage is at most `amplification` times the certified cost, A_P H_c(delta), which is to stay within `target`.
    Integrity: `unit` the outputs in a unit's cone, A = 1.  Control: `bits` 1 and `unit` the location term's
    log2(|U| + 1) a wrong unit (the 23:02Z schema's), A = 1.  Skippable work: `work` 1, A = F / (1 - p)."""

    name: str
    target: float
    amplification: float = 1.0
    work: float = 0.0
    bits: float = 0.0
    unit: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.target > 0 or not self.amplification > 0 \
                or min(self.work, self.bits, self.unit) < 0 or not self.work + self.bits + self.unit > 0:
            raise ValueError(f"policy {self.name!r}: needs a name, target > 0, amplification > 0 and costs >= 0 not all 0")

    def cost(self, c) -> float:
        return self.work * (c.work + c.influence) + self.bits * c.out_bits + self.unit

    def share(self, c) -> float:
        """What a wrong `c` spends of the policy's target."""
        return self.amplification * self.cost(c) / self.target


@dataclass(frozen=True)
class Objective:
    """The harm a wrong unit does: `work` times its credited work and influence plus `bits` times its output bits, or,
    given `policies`, the largest share of a policy's target it spends, so that a plan certifying harm at most 1 meets
    every policy (the draw is weighted by the envelope of the policies' costs).  Both weigh 1 by default, so a unit that
    changes values does harm whether or not it credits work; `bits=0.0` is the work alone, for populations whose every
    value-changing template does work (`refuse_floor`).  No target: targets and budgets are queries on the frontier.
    Every template draws at least `floor` units."""

    delta: float
    work: float = 1.0
    bits: float = 1.0
    floor: int = 1
    policies: tuple[Policy, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1 or self.work < 0 or self.bits < 0 or not self.work + self.bits > 0 or self.floor < 1:
            raise ValueError("an objective needs 0 < delta < 1, weights >= 0 not both 0, and floor >= 1")
        if len({p.name for p in self.policies}) != len(self.policies):
            raise ValueError("an objective's policies need distinct names")

    def harm(self, c) -> float:
        if self.policies:
            return max(p.share(c) for p in self.policies)
        return self.work * (c.work + c.influence) + self.bits * c.out_bits

    def binding(self, c) -> str:
        """The policy whose share sets `c`'s harm ("" without policies)."""
        return max(self.policies, key=lambda p: p.share(c)).name if self.policies else ""


@dataclass(frozen=True)
class Certificate:
    """What a plan's law certifies at `delta`, from its sizes alone: `harm` (core's `harm_bound` of the objective's harm),
    `density` (`density_bound` of it), `free_compute` (core's `harm_bound` of credited work and influence) and, when the
    objective weighs bits, `leak` (`consumers.exfiltration_bound`: `harm_bound` of output bits plus `location_bits`), and
    per policy its damage bound A_P H_c(delta) (`policies`), the one nearest its target binding.  When not `exact`, every
    field is `density_bound`'s and there is no leak."""

    delta: float
    harm: float
    density: float
    free_compute: float
    leak: float | None = None
    exact: bool = True
    policies: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """A cut per op type (`choice`: op -> its alternative), a statement size and a draw size per template, the templates
    held at their floor (none with output bits) and those proved whole, the expected cost by part (`serve` the fixed serve-time commitments,
    `total` everything), each block kind's weight on the density (1 when absent), and the certificate once certified."""

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
    """log C(n - m, k) / C(n, k), that a uniform k-subset of n misses m given units: the product
    prod_{i < t} (1 - s / (n - i)) over t = min(m, k), s = max(m, k), summed when t is small, by `lgamma` up to 2^25 units
    and past that, where `lgamma`'s differences lose their digits, by its midpoint (either within about 1e-7 relative),
    its terms doubled to stay integers: past 2^53 units a float's spacing exceeds s."""
    if k > n - m:
        return -inf
    if m == 0 or k == 0:
        return 0.0
    t, s = min(m, k), max(m, k)
    if t <= 64:
        return fsum(_log_rest(n - i - s, n - i) for i in range(t))
    if n <= 1 << 25:
        return lgamma(n - m + 1) - lgamma(n - m - k + 1) - lgamma(n + 1) + lgamma(n - k + 1)
    return t * _log_rest(2 * (n - s) - (t - 1), 2 * n - (t - 1))


def _log_rest(rest: float, whole: float) -> float:
    """log(rest / whole) for 0 < rest <= whole, by `log1p` unless rest is under half of whole, where the ratio rounds to 0
    (and 1 - ratio to 1) past 2^53 units."""
    return log1p((rest - whole) / whole) if 2 * rest >= whole else log(rest) - log(whole)


def _miss(n: int, m: int, k: int) -> float:
    return exp(_log_miss(n, m, k))


def _hit(n: int, m: int, k: int) -> float:
    """1 - `_miss`, its digits kept when the draw is sparse."""
    return -expm1(_log_miss(n, m, k))


def touched(n: int, c: int, k: int) -> float:
    """The expected number of statements a uniform k-subset of n units meets, the units cut in order into statements of
    c units (the last one shorter)."""
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
    """Drawing k of the candidate's units: its expected proving cost at the cheapest statement size, that size (the larger
    one on a tie, for fewer roots), and the cost's parts.  Serving's commitments and the replay domains are the plan's."""
    a0, a1 = costs.statement(c)
    units = k * (costs.unit(c) + costs.of(c).replay)
    best = None
    for size in _statement_sizes(c.cap, c.units):
        st = touched(c.units, size, k) * (a0 + a1 * size)
        if best is None or st < best[0] * (1 - 1e-12):
            best = (st, size)
    return best[0] + units, best[1], {"statements": best[0], "units": units}


def replay_cost(sizes: Mapping[str, int], chosen: Sequence[Candidate], costs: CostModel) -> float:
    """The expected cost of building the drawn units' witnesses, by replaying each replay unit the draw meets: a domain's
    replay unit escapes when every template's draw misses its share of it, independently across strata."""
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


def density_bound(rows: Iterable[tuple[int, int, float]], delta: float) -> float:
    """The most harm a wrong set escaping with probability at least `delta` holds, when a stratum of n units with k drawn
    and harm h per unit, holding x of the set, escapes with at most (1 - x/n)^k <= exp(-k x / n), and never once
    x > n - k.  So sum_s k_s x_s / n_s <= ln(1/delta), and the bound is that fractional knapsack's value: strata in the
    order of harm per unit of risk, the last one fractionally."""
    budget = log(1 / delta)
    out, items = 0.0, []
    for n, k, h in rows:
        if h <= 0 or k >= n:
            continue
        if k == 0:
            out += n * h
            continue
        items.append((h * n / k, n - k, k / n, h))
    for _, top, rate, h in sorted(items, reverse=True):
        x = min(top, budget / rate)
        out += x * h
        budget -= x * rate
        if budget <= 0:
            break
    return out


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
        return Certificate(d, density, density, dens(lambda c: float(c.work + c.influence)), None, False,
                           {p.name: p.amplification * dens(p.cost) for p in objective.policies})
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    leak = None
    if objective.bits and not objective.policies:
        profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, d)
        leak = exfiltration_bound(profile, {c.template: float(c.out_bits) for c in chosen})
    policies = {p.name: p.amplification * law.harm_bound({c.template: p.cost(c) for c in chosen}, d)
                for p in objective.policies}
    return Certificate(d, harm, density, work, leak, True, policies)


def _k(c: Candidate, objective: Objective, lam: float, weight: float = 1.0) -> int:
    return min(c.units, max(objective.floor, ceil(min(1.0, lam * weight * objective.harm(c)) * c.units)))


def refuse_floor(chosen: Iterable[Candidate], objective: Objective, floor_only: frozenset[str] = frozenset(),
                 full: frozenset[str] = frozenset()) -> None:
    """Refuse a plan that holds a template with output bits at its floor, either among `floor_only` or at zero harm,
    where the density never reaches it, unless it is proved whole."""
    for c in chosen:
        if c.out_bits and c.template not in full and (c.template in floor_only or not objective.harm(c) > 0):
            why = "is held at its floor" if c.template in floor_only else "does no harm under this objective"
            raise ValueError(
                f"{c.template!r} changes values ({c.out_bits} output bits) but {why}, so {objective.floor} of its "
                f"{c.units} units would be drawn however many there are: draw it at its harm's density, its output "
                f"bits weighed (Objective's default bits=1.0, or a policy pricing its bits or its unit), or fold its ops "
                f"into the dot-product unit that reads them (a covering cut)")


def _plan(chosen: Sequence[Candidate], costs: CostModel, objective: Objective, lam: float,
          floor_only: frozenset[str], full: frozenset[str], weights: Mapping[str, float] | None = None) -> Plan:
    refuse_floor(chosen, objective, floor_only, full)
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
    parts["total"] = parts["statements"] + parts["units"] + parts["commit"] + parts["replay"] + parts["serve"]
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
    """The plans the Lagrangian allows at one density: per op type its cheapest cut, then each prefix of the templates
    that change no value ranked by cost saved per unit of harm conceded at their floor, each with and without promoting
    to full the templates whose whole proof costs at most `promote` of the plan more."""
    def k(c: Candidate) -> int:
        return _k(c, objective, lam, weights.get(c.block, 1.0))

    def at(c: Candidate) -> float:
        return costs.commit(c) + template_cost(c, costs, k(c))[0]

    chosen = [c for alts in ops.values()
              for c in min(alts.items(), key=lambda kv: (sum(at(c) for c in kv[1]), kv[0]))[1]]

    def saving(c: Candidate) -> float:
        drop = template_cost(c, costs, k(c))[0] - template_cost(c, costs, min(c.units, objective.floor))[0]
        return drop / (c.units * objective.harm(c))

    ranked = sorted((c for c in chosen if not c.out_bits and objective.harm(c) > 0 and k(c) > objective.floor),
                    key=lambda c: (-saving(c), c.template))
    plans = []
    for i in range(len(ranked) + 1):
        skip = frozenset(c.template for c in ranked[:i])
        base = _plan(chosen, costs, objective, lam, skip, frozenset(), weights)
        plans.append(base)
        whole = frozenset(c.template for c in chosen if c.template not in skip and base.sizes[c.template] < c.units
                          and template_cost(c, costs, c.units)[0] - template_cost(c, costs, base.sizes[c.template])[0]
                          <= promote * (base.cost["total"] - base.cost["serve"]))
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
    """The certified Pareto frontier of (expected cost, `by`), cheapest first, `by` a `Certificate` field, each block
    kind's density scaled by its weight (`weights`, 1 when absent).  The sweep's plans are screened by `density_bound`
    (cheap) before the survivors are certified (`certify`, `exact` or not)."""
    weights = dict(weights or {})
    if any(not w > 0 for w in weights.values()):
        raise ValueError("a block kind's weight is positive: hold a kind at its floor by its density, not a zero weight")
    refuse_floor(candidates, objective)
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
    floor-only and full templates and weights, and the one certifying less is taken, so the answer doesn't hang on the
    sweep's grid.  The budget is the total, the serve-time commitments included."""
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
    when given), its density then lowered by bisection, the cuts, floor-only and full templates and weights held, to
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
    """The cheapest plan certifying at most `target` when each block kind (`Candidate.block`) is drawn at its own weight
    on the density: from equal weights, a coordinate search, each kind of the current plan in turn (costliest first) taking
    the factor of `factors` on its weight that makes `for_target` cheapest, for at most `rounds` passes.  The certificate
    is the same stratified one over every template, so the weights are parameters of the draw, chosen at the target, and
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
    """A plan's discrete shape: its cuts, statement sizes, and floor-only and whole templates."""
    return (tuple(sorted(p.choice.items())), tuple(sorted(p.statement_size.items())), tuple(sorted(p.floor_only)),
            tuple(sorted(p.full)))


def perturb(costs: CostModel, what: str, factor: float, candidates: Sequence[Candidate] = ()) -> CostModel:
    """The model with one group of costs scaled by `factor`: "prove" (per proved unit), "statement" (per touched
    statement), "commit" (serving's, the units' and the fixed serve-time commitments'), "replay" (the domains' and per
    drawn unit), or one template's (its derived costs pinned from `candidates` first)."""
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
    """The templates worth benchmarking at `budget`: those whose cost, moved by its own uncertainty either way (a
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


__all__ = ["Candidate", "Certificate", "Commitment", "CostModel", "HM96_ANDS", "Objective", "Plan", "ReplayDomain",
           "SHA512_ANDS", "TemplateCost", "certify", "density_bound", "for_budget", "for_target", "frontier",
           "needs_benchmark", "overlaps", "perturb", "refuse_floor", "replay_cost", "row_ands", "sensitivity",
           "structure", "template_cost", "touched", "weigh", "work_law"]
