"""The sampling planner (`optimize`): its statement count and closed-form bound against brute force and core's bound,
`Achievability`'s rates at zero statement cost, and the frontier under C-Flock's statement-batched cost."""

from __future__ import annotations

import random
from itertools import combinations
from math import ceil, log, ulp

import pytest

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k, work_table
from verity.core.protocols.sampled_proofs.optimize import (Candidate, CostModel, Objective, TemplateCost, certify,
                                                           density_bound, for_budget, for_target, frontier,
                                                           template_cost, touched, work_law)

SLACK = 1 + 2.0 ** -18


def at_most(core: float, bound: float) -> bool:
    """Core's bound, which rounds up (a zero harm to the least positive float), is at most `bound`."""
    return core <= max(bound * SLACK, ulp(0.0))


def test_touched_is_the_expected_number_of_statements_met():
    n, c = 7, 3
    statements = [set(range(i, min(i + c, n))) for i in range(0, n, c)]
    for k in range(n + 1):
        draws = list(combinations(range(n), k))
        exact = sum(sum(1 for s in statements if s & set(d)) for d in draws) / len(draws)
        assert touched(n, c, k) == pytest.approx(exact, abs=1e-12)


def test_density_bound_is_above_core_harm_bound():
    rng = random.Random(0)
    for _ in range(300):
        strata = []
        for i in range(rng.randint(1, 4)):
            n = rng.randint(1, 60)
            strata.append((f"s{i}", n, rng.randint(0, n), rng.choice([0.0, 1.0, rng.uniform(0, 50)])))
        delta = rng.choice([0.5, 2.0 ** -5, 2.0 ** -20])
        law = Stratified(tuple(Stratum(name, n, k) for name, n, k, _ in strata))
        core = law.harm_bound({name: h for name, _, _, h in strata}, delta)
        assert at_most(core, density_bound(((n, k, h) for _, n, k, h in strata), delta))


def test_certificate_of_a_full_law_is_zero_and_the_leak_adds_location():
    cands = [Candidate("a", "A", 50, 3, 16), Candidate("b", "B", 10, 7, 4)]
    exfil = Objective(2.0 ** -10, work=0.0, bits=1.0)
    full = certify({"A": 50, "B": 10}, cands, exfil)
    assert full.density == 0 and all(at_most(v, 0.0) for v in (full.harm, full.free_compute, full.leak))
    part = certify({"A": 5, "B": 2}, cands, exfil)
    assert part.leak > part.harm > 1 and at_most(part.harm, part.density)


def test_a_population_without_harm_has_no_frontier():
    cands = [Candidate("a", "A", 50, 0, 16)]
    with pytest.raises(ValueError, match="nothing to sample"):
        frontier(cands, CostModel({"A": TemplateCost(unit=1.0)}), Objective(0.01))


@pytest.fixture(scope="module")
def achievability():
    """Three templates of work 1, 8 and 200 at delta = 2^-10 and C = 400, each unit costing its work."""
    cands = [Candidate("a", "A", 4000, 1, 16), Candidate("b", "B", 500, 8, 16), Candidate("c", "C", 20, 200, 16)]
    costs = CostModel({c.template: TemplateCost(unit=float(c.work)) for c in cands})
    objective = Objective(2.0 ** -10)
    return cands, objective, for_target(cands, costs, objective, 400.0, by="density", grid=41)


def test_zero_statement_cost_gives_achievability_rates(achievability):
    """`Achievability`: q_s = min(1, ln(1/delta) w_s / C) certifies C; the planner draws exactly those rates, rounded up."""
    cands, objective, plan = achievability
    lam = log(1 / objective.delta) / 400.0
    assert dict(plan.sizes) == {c.template: min(c.units, max(1, ceil(min(1.0, lam * c.work) * c.units))) for c in cands}
    assert dict(plan.sizes) == {"A": 70, "B": 70, "C": 20}
    assert not plan.floor_only
    assert plan.certificate.harm <= plan.certificate.density <= 400.0


def _registered(plan, cands) -> dict[str, int]:
    """What `work:1` over `work_law`'s table draws, as the verifier reads it."""
    table = work_table(work_law(plan, cands)["table"])
    chosen = [c for c in cands if c.template in table]
    total = sum(table[c.template][0] * c.units for c in chosen)
    return {c.template: work_k(1, table[c.template][0], c.units, total, table[c.template][1]) for c in chosen}


def test_work_law_draws_the_plan_sizes(achievability, batched):
    cands, _, plan = achievability
    assert _registered(plan, cands) == dict(plan.sizes)
    cands, _, _, plans = batched
    for p in plans:
        assert _registered(p, cands) == dict(p.sizes)


@pytest.fixture(scope="module")
def batched():
    """C-Flock's shape at small scale: G, many cheap units behind a per-statement cost (a0 = 2, a1 = 0.016 per staged
    unit, at most 4096 units a statement); N, units costing 10 per unit of work and no statement cost; M, 64 heavy units
    that fit one statement."""
    cands = [Candidate("gemm", "G", 20000, 100, 16, cap=4096),
             Candidate("norm", "N", 1000, 100, 16),
             Candidate("mix", "M", 64, 50000, 16, cap=64)]
    costs = CostModel({"G": TemplateCost(2.0, 0.016, 0.07), "N": TemplateCost(unit=1000.0),
                       "M": TemplateCost(6.2, 0.0, 0.0125)})
    objective = Objective(2.0 ** -20)
    return cands, costs, objective, frontier(cands, costs, objective, grid=41)


def test_frontier_is_monotone_and_certified(batched):
    cands, _, _, plans = batched
    for a, b in zip(plans, plans[1:]):
        assert a.cost["total"] <= b.cost["total"] and a.certificate.harm > b.certificate.harm
    for p in plans:
        assert at_most(p.certificate.harm, p.certificate.density)
    assert dict(plans[0].sizes) == {"G": 1, "N": 1, "M": 1}
    assert dict(plans[-1].sizes) == {c.template: c.units for c in cands} and at_most(plans[-1].certificate.harm, 0.0)
    assert plans[-1].statement_size["G"] == 4096
    assert any("M" in p.full for p in plans)


def test_statement_size_moves_to_the_cap_as_the_draw_densifies(batched):
    cands, costs, objective, plans = batched
    sparse = for_target(cands, costs, objective, 4e5, plans=plans)
    dense = for_target(cands, costs, objective, 2e5, plans=plans)
    assert sparse.statement_size["G"] == 1 and dense.statement_size["G"] == 4096


def _uniform_cost(cands, costs, objective, target) -> float:
    """The cheapest law drawing one fraction q of every template (each at its cheapest statement size) that certifies
    `target`, by bisection on q."""
    def sizes(q: float) -> dict[str, int]:
        return {c.template: min(c.units, max(1, ceil(q * c.units))) for c in cands}

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if certify(sizes(mid), cands, objective).harm > target else (lo, mid)
    return sum(template_cost(c, costs, k)[0] for c, k in zip(cands, sizes(hi).values()))


def test_target_plan_skips_the_costly_template_and_beats_a_uniform_rate(batched):
    cands, costs, objective, plans = batched
    target = 4e5
    plan = for_target(cands, costs, objective, target, plans=plans)
    assert plan.certificate.harm <= target
    assert "N" in plan.floor_only and plan.sizes["N"] == 1 and plan.sizes["M"] == 64
    assert plan.cost["total"] <= min(p.cost["total"] for p in plans if p.certificate.harm <= target)
    assert plan.cost["total"] < _uniform_cost(cands, costs, objective, target) / 2


def test_for_budget_takes_the_least_harm_affordable_plan(batched):
    _, _, _, plans = batched
    assert for_budget(plans, plans[0].cost["total"] * 0.99) is None
    for p in plans:
        got = for_budget(plans, p.cost["total"])
        assert got.cost["total"] <= p.cost["total"] and got.certificate.harm <= p.certificate.harm
