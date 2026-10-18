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
`touched_s(k; c) * (a0 + a1 c) + k * (b + replay)`, a statement being touched when the draw meets it.  Serving commits every
unit's outputs whatever the draw (`commit_bits`, `commit_rows`), and a replay domain re-executes each replay unit the draw
meets from the last committed state (`ReplayDomain`).  A template's costs are measured (`TemplateCost`) or derived from the
model's rates and its features: ANDs, and the hm96 rows a proved unit opens (`row_ands`).

**Sweep.**  The note's Lagrangian: one density `lam` on every template (`k_s = ceil(min(1, lam v_s) n_s)`), some templates
at their floor only, some proved whole, and the cut per op type and the statement size each the cheapest at that `lam`.
With no statement cost and a unit cost proportional to work, the cheapest plan certifying `C` is `Achievability`'s rates,
`q_s = min(1, ln(1/delta) w_s / C)`.
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
    `commit_rows` rows per unit whatever the law; one statement holds at most `cap` units (m = k_log + nbl <= 35); and
    `per_replay` units of it lie in each replay unit of domain `replay`."""

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
    when the draw meets any unit in it."""

    units: int
    cost: float


@dataclass(frozen=True)
class CostModel:
    """Measured templates, and the rates that derive the rest from their features: per touched statement
    `statement_fixed + statement_per_and * ands` (the verifier's parse) and per staged unit `staged_per_and * ands +
    staged_per_unit`; per proved unit `per_and` times its ANDs and its rows' (`row_ands`).  Serving commits at
    `commit_per_bit` and `commit_per_row`; `replay` names the replay domains; `sigma` is a derived cost's relative
    uncertainty; `source` is the fit's `art:` id."""

    templates: Mapping[str, TemplateCost] = field(default_factory=dict)
    per_and: float = 0.0
    statement_fixed: float = 0.0
    statement_per_and: float = 0.0
    staged_per_and: float = 0.0
    staged_per_unit: float = 0.0
    commit_per_bit: float = 0.0
    commit_per_row: float = 0.0
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

    def statement(self, c: Candidate) -> tuple[float, float]:
        tc = self.of(c)
        a0 = tc.statement_fixed if tc.statement_fixed is not None else self.statement_fixed + self.statement_per_and * c.ands
        a1 = tc.statement_per_unit if tc.statement_per_unit is not None else self.staged_per_and * c.ands \
            + self.staged_per_unit
        return a0, a1

    def unit(self, c: Candidate) -> float:
        tc = self.of(c)
        return tc.unit if tc.unit is not None else self.per_and * (c.ands + sum(row_ands(r) for r in c.rows))

    def commit(self, c: Candidate) -> float:
        return c.units * (self.commit_per_bit * c.commit_bits + self.commit_per_row * c.commit_rows)

    def sigma_of(self, c: Candidate) -> float:
        return self.templates[c.template].sigma if c.template in self.templates else self.sigma


@dataclass(frozen=True)
class Objective:
    """The harm a wrong unit does: `work` times its credited work and influence plus `bits` times its output bits.  No
    target: targets and budgets are queries on the frontier.  Every template draws at least `floor` units."""

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
    """What a plan's law certifies at `delta`, from its sizes alone: `harm` (core's `harm_bound` of the objective's harm),
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
    """A cut per op type (`choice`: op -> its alternative), a statement size and a draw size per template, the templates
    held at their floor and those proved whole, the expected cost by part, and the certificate once certified."""

    lam: float
    choice: Mapping[str, str]
    statement_size: Mapping[str, int]
    sizes: Mapping[str, int]
    floor_only: frozenset[str]
    full: frozenset[str]
    cost: Mapping[str, float]
    certificate: Certificate | None = field(default=None, compare=False)


def _log_miss(n: int, m: int, k: int) -> float:
    """log C(n - m, k) / C(n, k), that a uniform k-subset of n misses m given units: the product
    prod_{i < t} (1 - s / (n - i)) over t = min(m, k), s = max(m, k), summed when t is small, by `lgamma` up to 2^25 units
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
    one on a tie, for fewer roots), and the cost's parts.  Commitment and replay domains are the plan's (`plan_cost`)."""
    a0, a1 = costs.statement(c)
    units = k * (costs.unit(c) + costs.of(c).replay)
    best = None
    for size in _statement_sizes(c.cap, c.units):
        st = touched(c.units, size, k) * (a0 + a1 * size)
        if best is None or st < best[0] * (1 - 1e-12):
            best = (st, size)
    return best[0] + units, best[1], {"statements": best[0], "units": units}


def replay_cost(sizes: Mapping[str, int], chosen: Sequence[Candidate], costs: CostModel) -> float:
    """The expected cost of the replay units the draw meets: a domain's replay unit escapes when every template's draw
    misses its share of it, independently across strata."""
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
        return Certificate(d, density, density, dens(lambda c: float(c.work + c.influence)), None, False)
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work + c.influence) for c in chosen}, d)
    leak = None
    if objective.bits:
        profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, d)
        leak = exfiltration_bound(profile, {c.template: float(c.out_bits) for c in chosen})
    return Certificate(d, harm, density, work, leak)


def _k(c: Candidate, objective: Objective, lam: float) -> int:
    return min(c.units, max(objective.floor, ceil(min(1.0, lam * objective.harm(c)) * c.units)))


def _plan(chosen: Sequence[Candidate], costs: CostModel, objective: Objective, lam: float,
          floor_only: frozenset[str], full: frozenset[str]) -> Plan:
    sizes, statement, parts = {}, {}, {"statements": 0.0, "units": 0.0, "commit": 0.0}
    for c in chosen:
        k = c.units if c.template in full else min(c.units, objective.floor) if c.template in floor_only \
            else _k(c, objective, lam)
        _, size, p = template_cost(c, costs, k)
        sizes[c.template], statement[c.template] = k, size
        parts["statements"] += p["statements"]
        parts["units"] += p["units"]
        parts["commit"] += costs.commit(c)
    parts["replay"] = replay_cost(sizes, chosen, costs)
    parts["total"] = parts["statements"] + parts["units"] + parts["commit"] + parts["replay"]
    choice = {c.op: c.alternative for c in chosen}
    return Plan(lam, choice, statement, sizes, floor_only, full, parts)


def _by_op(candidates: Sequence[Candidate]) -> dict[str, dict[str, list[Candidate]]]:
    if len({c.template for c in candidates}) != len(candidates):
        raise ValueError("candidates need distinct templates")
    ops: dict[str, dict[str, list[Candidate]]] = {}
    for c in candidates:
        ops.setdefault(c.op, {}).setdefault(c.alternative, []).append(c)
    return ops


def _plans_at(ops: Mapping[str, Mapping[str, list[Candidate]]], costs: CostModel, objective: Objective, lam: float,
              promote: float) -> list[Plan]:
    """The plans the Lagrangian allows at one density: per op type its cheapest cut, then each prefix of the templates
    ranked by cost saved per unit of harm conceded at their floor, each with and without promoting to full the templates
    whose whole proof costs at most `promote` of the plan more."""
    def at(c: Candidate) -> float:
        return costs.commit(c) + template_cost(c, costs, _k(c, objective, lam))[0]

    chosen = [c for alts in ops.values()
              for c in min(alts.items(), key=lambda kv: (sum(at(c) for c in kv[1]), kv[0]))[1]]

    def saving(c: Candidate) -> float:
        drop = template_cost(c, costs, _k(c, objective, lam))[0] - template_cost(c, costs, min(c.units, objective.floor))[0]
        return drop / (c.units * objective.harm(c))

    ranked = sorted((c for c in chosen if objective.harm(c) > 0 and _k(c, objective, lam) > objective.floor),
                    key=lambda c: (-saving(c), c.template))
    plans = []
    for i in range(len(ranked) + 1):
        skip = frozenset(c.template for c in ranked[:i])
        base = _plan(chosen, costs, objective, lam, skip, frozenset())
        plans.append(base)
        whole = frozenset(c.template for c in chosen if c.template not in skip and base.sizes[c.template] < c.units
                          and template_cost(c, costs, c.units)[0] - template_cost(c, costs, base.sizes[c.template])[0]
                          <= promote * base.cost["total"])
        if whole:
            plans.append(_plan(chosen, costs, objective, lam, skip, whole))
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
             grid: int = 121, promote: float = 0.05, exact: bool = True) -> list[Plan]:
    """The certified Pareto frontier of (expected cost, `by`), cheapest first, `by` a `Certificate` field.  The sweep's
    plans are screened by `density_bound` (cheap) before the survivors are certified (`certify`, `exact` or not)."""
    ops = _by_op(candidates)
    harms = [objective.harm(c) * c.units for c in candidates if objective.harm(c) > 0]
    if not harms:
        raise ValueError("no candidate does any harm under this objective: there is nothing to sample for")
    lo = 1 / (10 * max(harms))
    hi = 1 / min(objective.harm(c) for c in candidates if objective.harm(c) > 0)
    lams = [lo * (hi / lo) ** (i / (grid - 1)) for i in range(grid)]
    seen: dict[tuple, Plan] = {}
    for lam in lams:
        for p in _plans_at(ops, costs, objective, lam, promote):
            seen.setdefault((tuple(sorted(p.sizes.items())), tuple(sorted(p.statement_size.items()))), p)

    def screened(p: Plan) -> float:
        chosen = [c for c in candidates if c.template in p.sizes]
        return density_bound(((c.units, p.sizes[c.template], objective.harm(c)) for c in chosen), objective.delta)

    plans = [replace(p, certificate=certify(p.sizes, candidates, objective, exact=exact))
             for p in _pareto(seen.values(), screened)]
    return _pareto(plans, lambda p: _harm(p, by))


def for_budget(plans: Sequence[Plan], budget: float) -> Plan | None:
    """The frontier's least-harm plan within `budget`: its costliest affordable plan."""
    affordable = [p for p in plans if p.cost["total"] <= budget]
    return affordable[-1] if affordable else None


def for_target(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, target: float, *,
               by: str = "harm", plans: Sequence[Plan] | None = None, exact: bool = True, **kw) -> Plan | None:
    """The cheapest plan certifying at most `target`: the frontier's cheapest such plan (of `plans`, a frontier by `by`,
    when given), its density then lowered by bisection, the cuts, floor-only and full templates held, to where its
    certificate just meets the target."""
    if plans is None:
        plans = frontier(candidates, costs, objective, by=by, exact=exact, **kw)
    ok = [p for p in plans if _harm(p, by) <= target]
    if not ok:
        return None
    best = ok[0]
    chosen = [c for c in candidates if c.template in best.sizes]

    def at(lam: float) -> Plan:
        p = _plan(chosen, costs, objective, lam, best.floor_only, best.full)
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


def structure(p: Plan) -> tuple:
    """A plan's discrete shape: its cuts, statement sizes, and floor-only and whole templates."""
    return (tuple(sorted(p.choice.items())), tuple(sorted(p.statement_size.items())), tuple(sorted(p.floor_only)),
            tuple(sorted(p.full)))


def perturb(costs: CostModel, what: str, factor: float, candidates: Sequence[Candidate] = ()) -> CostModel:
    """The model with one group of costs scaled by `factor`: "prove" (per proved unit), "statement" (per touched
    statement), "commit" (serving's), "replay" (the domains' and per drawn unit), or one template's (its derived costs
    pinned from `candidates` first)."""
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
        return replace(costs, commit_per_bit=costs.commit_per_bit * factor, commit_per_row=costs.commit_per_row * factor)
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
                  "commit": costs.commit_per_bit or costs.commit_per_row, "replay": bool(costs.replay)}
        factors = {g: (0.5, 2.0) for g, on in groups.items() if on}
    base = for_budget(frontier(candidates, costs, objective, by=by, **kw), budget)
    rows = []
    for what, fs in factors.items():
        for f in fs:
            p = for_budget(frontier(candidates, perturb(costs, what, f, candidates), objective, by=by, **kw), budget)
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


__all__ = ["Candidate", "Certificate", "CostModel", "HM96_ANDS", "Objective", "Plan", "ReplayDomain", "SHA512_ANDS",
           "TemplateCost", "certify", "density_bound", "for_budget", "for_target", "frontier", "needs_benchmark", "perturb",
           "replay_cost", "row_ands", "sensitivity", "structure", "template_cost", "touched", "work_law"]
