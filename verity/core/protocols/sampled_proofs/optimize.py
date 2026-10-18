"""A planner for the sampling scheme: from a Program's partition candidates, a proof system's cost model and an objective,
the plans on the Pareto frontier of expected cost against certified harm, over every budget
(note:proofs/20261018T0335Z-draft-sampling-optimizer-design).

Nothing trusts it.  A plan is a stratified law, `k_s` units of each template, which registers as `work:1` over a work
table carrying the credited work and floors equal to the sizes (`work_law`: `one_stage.draw.work_k(1, w, n, W, f)` is
`min(n, f)`).  Its certificate is recomputed from the sizes alone (`certify`): core's `Stratified.harm_bound` for the
objective's harm and for credited work, and `density_bound`, the closed form that `work_escape_le`'s proof extends to any
sizes (the note's gap G1).

The cost of a law is C-Flock's shape, per touched statement plus per proved unit: template `s` cut into statements of `c`
units costs `touched_s(k; c) * (a0 + a1 c) + k * (b + replay)`, a statement being touched when the draw meets it.  The
sweep follows the note's Lagrangian: one density `lam` on every template (`k_s = ceil(min(1, lam v_s) n_s)`), some
templates at their floor only, some proved whole, and the statement size and the candidate per op type each the cheapest
at that `lam`.  With no statement cost and a unit cost proportional to work, the cheapest plan certifying `C` is
`Achievability`'s rates, `q_s = min(1, ln(1/delta) w_s / C)`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import ceil, exp, lgamma, log

from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound


@dataclass(frozen=True)
class Candidate:
    """One way to cut one op type into units, derived from the IR, never from the prover: `units` instances of
    `template`, each credited `work` (the verifier's work table) with at most 16 output bits.  A proved unit opens
    `io_leaves` hm96 leaves holding `io_bits`, serving commits `commit_bits` per unit whatever the law, and one statement
    holds at most `cap` units (m = k_log + nbl <= 35)."""

    op: str
    template: str
    units: int
    work: int
    out_bits: int
    io_leaves: int = 0
    io_bits: int = 0
    commit_bits: int = 0
    cap: int = 1

    def __post_init__(self) -> None:
        if not self.op or not self.template or self.units < 1 or self.work < 0 or not 0 <= self.out_bits <= 16 \
                or min(self.io_leaves, self.io_bits, self.commit_bits) < 0 or self.cap < 1:
            raise ValueError(f"candidate {self.template!r}: needs units >= 1, work >= 0, out_bits in [0, 16], cap >= 1")


@dataclass(frozen=True)
class TemplateCost:
    """One template's costs under one proof-system version, in the caller's currency: `statement_fixed` (a0) and
    `statement_per_unit` (a1, per staged unit) per touched statement, `unit` (b) per proved unit or, when None, derived
    from `unit_compute` and the model's hashing terms, and `replay` per drawn unit."""

    statement_fixed: float = 0.0
    statement_per_unit: float = 0.0
    unit: float | None = None
    unit_compute: float = 0.0
    replay: float = 0.0


@dataclass(frozen=True)
class CostModel:
    """Per-template costs, the in-circuit hashing terms that derive an unmeasured unit's cost, serving's commit cost per
    committed bit, and `source`, the fit's `art:` id."""

    templates: Mapping[str, TemplateCost]
    hash_per_leaf: float = 0.0
    hash_per_bit: float = 0.0
    commit_per_bit: float = 0.0
    source: str = ""

    def of(self, c: Candidate) -> TemplateCost:
        if c.template not in self.templates:
            raise KeyError(f"the cost model has no template {c.template!r}: fit it or pass a TemplateCost")
        return self.templates[c.template]

    def unit(self, c: Candidate) -> float:
        tc = self.of(c)
        if tc.unit is not None:
            return tc.unit
        return tc.unit_compute + self.hash_per_leaf * c.io_leaves + self.hash_per_bit * c.io_bits

    def commit(self, c: Candidate) -> float:
        return self.commit_per_bit * c.commit_bits * c.units


@dataclass(frozen=True)
class Objective:
    """The harm a wrong unit does: `work` times its credited work plus `bits` times its output bits.  No target: targets
    and budgets are queries on the frontier.  Every template draws at least `floor` units."""

    delta: float
    work: float = 1.0
    bits: float = 0.0
    floor: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.delta < 1 or self.work < 0 or self.bits < 0 or not self.work + self.bits > 0 or self.floor < 1:
            raise ValueError("an objective needs 0 < delta < 1, weights >= 0 not both 0, and floor >= 1")

    def harm(self, c: Candidate) -> float:
        return self.work * c.work + self.bits * c.out_bits


@dataclass(frozen=True)
class Certificate:
    """What a plan's law certifies at `delta`, from its sizes alone: `harm` (core's `harm_bound` of the objective's harm),
    `density` (`density_bound` of it), `free_compute` (core's `harm_bound` of credited work) and, when the objective
    weighs bits, `leak` (`consumers.exfiltration_bound`: `harm_bound` of output bits plus `location_bits`)."""

    delta: float
    harm: float
    density: float
    free_compute: float
    leak: float | None = None


@dataclass(frozen=True)
class Plan:
    """A partition (`choice`: op -> template), a statement size and a draw size per template, the templates held at
    their floor and those proved whole, the expected cost by part, and the certificate once certified."""

    lam: float
    choice: Mapping[str, str]
    statement_size: Mapping[str, int]
    sizes: Mapping[str, int]
    floor_only: frozenset[str]
    full: frozenset[str]
    cost: Mapping[str, float]
    certificate: Certificate | None = field(default=None, compare=False)


def _miss(n: int, m: int, k: int) -> float:
    """C(n - m, k) / C(n, k): a uniform k-subset of n misses m given units."""
    if k > n - m:
        return 0.0
    return exp(lgamma(n - m + 1) - lgamma(n - m - k + 1) - lgamma(n + 1) + lgamma(n - k + 1))


def touched(n: int, c: int, k: int) -> float:
    """The expected number of statements a uniform k-subset of n units meets, the units cut in order into statements of
    c units (the last one shorter)."""
    if k <= 0:
        return 0.0
    whole, rest = divmod(n, c)
    return whole * (1 - _miss(n, c, k)) + (1 - _miss(n, rest, k) if rest else 0.0)


def _statement_sizes(cap: int, units: int) -> list[int]:
    top = min(cap, units)
    sizes = {top}
    c = 1
    while c < top:
        sizes.add(c)
        c *= 2
    return sorted(sizes, reverse=True)


def template_cost(c: Candidate, costs: CostModel, k: int) -> tuple[float, int, dict[str, float]]:
    """Drawing k of the candidate's units: its expected cost at the cheapest statement size, that size (the larger one
    on a tie, for fewer roots), and the cost's parts."""
    tc = costs.of(c)
    units = k * (costs.unit(c) + tc.replay)
    best = None
    for size in _statement_sizes(c.cap, c.units):
        st = touched(c.units, size, k) * (tc.statement_fixed + tc.statement_per_unit * size)
        if best is None or st < best[0] * (1 - 1e-12):
            best = (st, size)
    return best[0] + units, best[1], {"statements": best[0], "units": units}


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


def certify(sizes: Mapping[str, int], candidates: Sequence[Candidate], objective: Objective) -> Certificate:
    """The plan's guarantee, from its sizes alone, by core's bounds (and `density_bound`)."""
    chosen = [c for c in candidates if c.template in sizes]
    law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in chosen))
    d = objective.delta
    harm = law.harm_bound({c.template: objective.harm(c) for c in chosen}, d)
    work = law.harm_bound({c.template: float(c.work) for c in chosen}, d)
    leak = None
    if objective.bits:
        profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, d)
        leak = exfiltration_bound(profile, {c.template: float(c.out_bits) for c in chosen})
    density = density_bound(((c.units, sizes[c.template], objective.harm(c)) for c in chosen), d)
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
    parts["total"] = parts["statements"] + parts["units"] + parts["commit"]
    return Plan(lam, {c.op: c.template for c in chosen}, statement, sizes, floor_only, full, parts)


def _by_op(candidates: Sequence[Candidate]) -> dict[str, list[Candidate]]:
    ops: dict[str, list[Candidate]] = {}
    for c in candidates:
        ops.setdefault(c.op, []).append(c)
    if len({c.template for c in candidates}) != len(candidates):
        raise ValueError("candidates need distinct templates")
    return ops


def _plans_at(ops: Mapping[str, list[Candidate]], costs: CostModel, objective: Objective, lam: float,
              promote: float) -> list[Plan]:
    """The plans the Lagrangian allows at one density: per op type its cheapest candidate, then each prefix of the
    templates ranked by cost saved per unit of harm conceded at their floor, each with and without promoting to full
    the templates whose whole proof costs at most `promote` of the plan more."""
    chosen = [min(cs, key=lambda c: (costs.commit(c) + template_cost(c, costs, _k(c, objective, lam))[0], c.template))
              for cs in ops.values()]

    def saving(c: Candidate) -> float:
        at = template_cost(c, costs, _k(c, objective, lam))[0]
        return (at - template_cost(c, costs, min(c.units, objective.floor))[0]) / (c.units * objective.harm(c))

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
             grid: int = 121, promote: float = 0.05) -> list[Plan]:
    """The certified Pareto frontier of (expected cost, `by`), cheapest first, `by` a `Certificate` field.  The sweep's
    plans are screened by `density_bound` (cheap) before core's bounds certify the survivors."""
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

    plans = [replace(p, certificate=certify(p.sizes, candidates, objective)) for p in _pareto(seen.values(), screened)]
    return _pareto(plans, lambda p: _harm(p, by))


def for_budget(plans: Sequence[Plan], budget: float) -> Plan | None:
    """The frontier's least-harm plan within `budget`: its costliest affordable plan."""
    affordable = [p for p in plans if p.cost["total"] <= budget]
    return affordable[-1] if affordable else None


def for_target(candidates: Sequence[Candidate], costs: CostModel, objective: Objective, target: float, *,
               by: str = "harm", plans: Sequence[Plan] | None = None, **kw) -> Plan | None:
    """The cheapest plan certifying at most `target`: the frontier's cheapest such plan (of `plans`, a frontier by `by`,
    when given), its density then lowered by bisection, the partition, floor-only and full templates held, to where its
    certificate just meets the target."""
    if plans is None:
        plans = frontier(candidates, costs, objective, by=by, **kw)
    ok = [p for p in plans if _harm(p, by) <= target]
    if not ok:
        return None
    best = ok[0]
    chosen = [c for c in candidates if c.template in best.sizes]

    def at(lam: float) -> Plan:
        p = _plan(chosen, costs, objective, lam, best.floor_only, best.full)
        return replace(p, certificate=certify(p.sizes, candidates, objective))

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


def work_law(plan: Plan, candidates: Sequence[Candidate]) -> dict:
    """The plan as the rule the verifier registers today: `work:1` over a work table carrying the credited work and
    floors equal to the plan's sizes, which `one_stage.draw.work_k` draws exactly (the ceiling is at most 1 since
    w n <= W)."""
    chosen = [c for c in candidates if c.template in plan.sizes]
    if not any(c.work for c in chosen):
        raise ValueError("a work law needs a population that does some work")
    return {"law": "work:1", "table": {c.template: {"work": c.work, "floor": plan.sizes[c.template]} for c in chosen}}


__all__ = ["Candidate", "Certificate", "CostModel", "Objective", "Plan", "TemplateCost", "certify", "density_bound",
           "for_budget", "for_target", "frontier", "template_cost", "touched", "work_law"]
