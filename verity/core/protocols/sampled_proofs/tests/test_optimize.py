"""The sampling planner (`optimize`): its statement count, replay count and closed-form bound against brute force and core's
bound, `Achievability`'s rates at zero statement cost, the frontier under C-Flock's statement-batched cost, the choice
between a partition and a covering, serving's commitment, and the sensitivity queries."""

from __future__ import annotations

import random
from itertools import combinations, product
from math import ceil, exp, fsum, log, log1p, ulp

import pytest

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k, work_table
from verity.core.protocols.sampled_proofs.optimize import (Candidate, CostModel, Objective, ReplayDomain, TemplateCost,
                                                           _miss, certify, density_bound, for_budget, for_target, frontier,
                                                           needs_benchmark, perturb, replay_cost, row_ands,
                                                           sensitivity, template_cost, touched, work_law)

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


def test_row_ands_follows_hm96_table():
    assert [row_ands(n) for n in (4, 256, 4096)] == [174_360, 290_600, 2_034_200]


def test_candidates_follow_the_width_rule():
    Candidate("a", "A", 1, 1, 32)
    Candidate("a", "A", 1, 1, 16, out_gates=4)
    for bits, gates in ((32, 2), (17, 2), (33, 1)):
        with pytest.raises(ValueError, match="width rule"):
            Candidate("a", "A", 1, 1, bits, out_gates=gates)
    with pytest.raises(ValueError, match="per_replay"):
        Candidate("a", "A", 4, 1, 16, replay="layer")
    with pytest.raises(ValueError, match="per_replay"):
        Candidate("a", "A", 4, 1, 16, replay="layer", per_replay=5)


def test_derived_costs_read_the_features_and_perturb_scales_one_group():
    c = Candidate("a", "A", 10, 1, 16, ands=1000, rows=(256, 4), commit_bits=64, commit_rows=1.0)
    costs = CostModel(per_and=1e-6, statement_fixed=2.0, statement_per_and=1e-7, staged_per_and=1e-9,
                      staged_per_unit=0.01, commit_per_bit=1e-3, commit_per_row=0.5)
    assert costs.unit(c) == pytest.approx(1e-6 * (1000 + 290_600 + 174_360))
    assert costs.statement(c) == pytest.approx((2.0 + 1e-4, 1e-6 + 0.01))
    assert costs.commit(c) == pytest.approx(10 * (0.064 + 0.5))
    assert perturb(costs, "prove", 2.0).unit(c) == pytest.approx(2 * costs.unit(c))
    assert perturb(costs, "prove", 2.0).statement(c) == pytest.approx(costs.statement(c))
    assert perturb(costs, "statement", 3.0).statement(c) == pytest.approx(tuple(3 * a for a in costs.statement(c)))
    assert perturb(costs, "commit", 0.5).commit(c) == pytest.approx(costs.commit(c) / 2)
    pinned = perturb(costs, "A", 2.0, [c])
    assert pinned.unit(c) == pytest.approx(2 * costs.unit(c))
    assert pinned.statement(c) == pytest.approx(tuple(2 * a for a in costs.statement(c)))
    other = Candidate("b", "B", 10, 1, 16, ands=1000)
    assert pinned.unit(other) == pytest.approx(costs.unit(other))
    with pytest.raises(KeyError, match="nothing to perturb"):
        perturb(costs, "Z", 2.0, [c])
    with pytest.raises(KeyError, match="no rates"):
        CostModel(commit_per_bit=1.0).unit(c)


def test_replay_cost_is_the_expected_number_of_replay_units_met():
    """Three replay units, A holding 2 units of each and B 1: every pair of draws enumerated."""
    a = Candidate("x", "A", 6, 1, 16, replay="layer", per_replay=2)
    b = Candidate("y", "B", 3, 1, 16, replay="layer", per_replay=1)
    costs = CostModel(per_and=1.0, replay={"layer": ReplayDomain(3, 5.0)})
    for ka, kb in product(range(7), range(4)):
        draws = list(product(combinations(range(6), ka), combinations(range(3), kb)))
        met = sum(len({u // 2 for u in da} | set(db)) for da, db in draws) / len(draws)
        assert replay_cost({"A": ka, "B": kb}, [a, b], costs) == pytest.approx(5.0 * met, abs=1e-12)
    with pytest.raises(ValueError, match="replay units"):
        replay_cost({"A": 1}, [a], CostModel(per_and=1.0, replay={"layer": ReplayDomain(2, 1.0)}))
    with pytest.raises(KeyError, match="no replay domain"):
        replay_cost({"A": 1}, [a], CostModel(per_and=1.0))


def test_inexact_certificate_is_the_density_bound():
    cands = [Candidate("a", "A", 400, 3, 16, influence=5), Candidate("b", "B", 60, 7, 4)]
    objective = Objective(2.0 ** -10)
    sizes = {"A": 40, "B": 6}
    exact, loose = certify(sizes, cands, objective), certify(sizes, cands, objective, exact=False)
    assert not loose.exact and loose.leak is None and loose.harm == loose.density == exact.density
    assert at_most(exact.harm, loose.harm) and at_most(exact.free_compute, loose.free_compute)


R, J, W = 32, 64, 100


def _noised(influence: int = J * W, commit_bits: int = 0) -> list[Candidate]:
    """One op of R rows of J dots of work W, cut two ways: a form unit per row (cost 4) that every dot of its row reads, and
    dots of cost 1 (a partition), or dots of cost 2 that each re-derive the row's form (a covering)."""
    return [Candidate("dot", "Form", R, 0, 16, cut="form", influence=influence, commit_bits=commit_bits),
            Candidate("dot", "DotP", R * J, W, 16, cut="form"),
            Candidate("dot", "DotC", R * J, W, 16, cut="recompute")]


def _noised_costs(sigma: float = 0.0, commit_per_bit: float = 0.0) -> CostModel:
    return CostModel({"Form": TemplateCost(unit=4.0), "DotP": TemplateCost(unit=1.0),
                      "DotC": TemplateCost(unit=2.0, sigma=sigma)}, commit_per_bit=commit_per_bit)


def _crossing(plans) -> tuple[int, int]:
    """The covering's largest dot draw and the partition's smallest, checking the frontier switches once."""
    choices = [p.choice["dot"] for p in plans]
    assert choices == sorted(choices, key=lambda c: c == "form") and set(choices) == {"recompute", "form"}
    return (max(p.sizes["DotC"] for p in plans if p.choice["dot"] == "recompute"),
            min(p.sizes["DotP"] for p in plans if p.choice["dot"] == "form"))


def test_covering_wins_sparse_and_partition_wins_once_its_form_units_saturate():
    """Each form unit's influence (its row's work) draws all R of them early; then k dots cost 4R + k as a partition and 2k
    as a covering at the same harm, so the switch is at k = 4R."""
    plans = frontier(_noised(), _noised_costs(), Objective(2.0 ** -20), grid=61)
    rec, form = _crossing(plans)
    assert rec <= 4 * R <= form
    assert all(p.sizes["Form"] == R for p in plans if p.choice["dot"] == "form")


def test_without_influence_the_form_unit_looks_free():
    plans = frontier(_noised(influence=0), _noised_costs(), Objective(2.0 ** -20), grid=61)
    assert _crossing(plans)[1] <= 4
    assert all(p.sizes["Form"] == 1 for p in plans if p.choice["dot"] == "form")


def test_commitment_is_paid_on_every_unit_and_moves_the_switch():
    """Committing the form costs 4 per row whatever the draw, so the switch moves to k = 8R."""
    plans = frontier(_noised(commit_bits=400), _noised_costs(commit_per_bit=0.01), Objective(2.0 ** -20), grid=61)
    rec, form = _crossing(plans)
    assert rec <= 8 * R <= form
    assert {p.cost["commit"] for p in plans} == {0.0, 4.0 * R}


def test_sensitivity_and_benchmarks_flag_what_moves_the_plan():
    objective = Objective(2.0 ** -20)
    base, rows = sensitivity(_noised(), _noised_costs(), objective, 260.0, grid=61)
    assert base.choice == {"dot": "recompute"} and [r["what"] for r in rows] == ["prove", "prove"]
    assert {r["factor"]: r["same"] for r in rows} == {0.5: False, 2.0: True}
    assert needs_benchmark(_noised(), _noised_costs(0.25), objective, 260.0, grid=61) == ["DotC"]
    assert needs_benchmark(_noised(), _noised_costs(0.01), objective, 260.0, grid=61) == []
    assert needs_benchmark(_noised(), _noised_costs(0.25), objective, 2000.0, grid=61) == []


def test_miss_holds_its_digits_on_huge_populations():
    """A week of served units is about 10^16: the miss probability against the product it denotes, summed exactly."""
    for n in (2 ** 25 - 3, 2 ** 25 + 3, 10 ** 12, 10 ** 16):
        for m, k in ((1, 1), (4096, 27_713), (10 ** 6, 100), (64, 10 ** 6)):
            t, s = min(m, k), max(m, k)
            assert _miss(n, m, k) == pytest.approx(exp(fsum(log1p(-s / (n - i)) for i in range(t))), rel=1e-6)
