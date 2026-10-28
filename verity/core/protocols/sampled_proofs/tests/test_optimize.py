"""The sampling planner (`optimize`): its statement count, replay count and closed-form bound against brute force and core's
bound, `Achievability`'s rates at zero statement cost, the frontier under C-Flock's statement-batched cost, the choice
between a partition and a covering, serving's commitment (per proof block, and the fixed serve-time total with its overlaps), the
weight classes' weights, and the sensitivity queries."""

from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction
from itertools import combinations, product
from math import ceil, exp, fsum, log, log1p, ulp

import pytest

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.sampled_proofs.one_stage.draw import work_k, work_table
from verity.core.protocols.sampled_proofs.one_stage.partition import UNIT_CLASS_MAX
from verity.core.protocols.sampled_proofs.optimize import (Candidate, Commitment, CostModel, Objective, ReplayDomain,
                                                           TemplateCost, _hit, _miss, certify, density_bound, density_exact,
                                                           for_budget, for_target, frontier, ln_bound, ln_upper, max_harm,
                                                           max_harm_check, needs_benchmark,
                                                           overlaps, perturb, replay_cost, row_ands, sensitivity,
                                                           template_cost, touched, weigh, work_law)

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


def test_density_bound_is_its_exact_value_rounded_up():
    """`density_bound`'s float is at least `density_exact`, Lean's rational (the sampled proofs service's `harm_bound`
    difftest holds the two equal), and within 2^-18 of it unless delta is within 2^-40 of 1, where the budget's 2^-60
    is most of ln(1/delta)."""
    assert ln_upper(Fraction(1, 2)) == Fraction(3196577161300663915, 2 ** 62)
    assert abs(ln_upper(Fraction(1, 100)) - Fraction(log(100))) < 2.0 ** -48
    rng = random.Random(0)
    for _ in range(500):
        rows = [(n, rng.randint(0, min(n, rng.choice([1, 10, 1000, n]))),
                 rng.choice([0.0, rng.random() * 10.0 ** rng.randint(-20, 20)]))
                for n in (rng.randint(1, 10 ** rng.randint(1, 16)) for _ in range(rng.randint(0, 8)))]
        delta = rng.choice([0.5, 0.01, 1e-6, 2.0 ** -40, 1 - 2.0 ** -40, 1 - 2.0 ** -53, rng.uniform(1e-9, 0.999)])
        exact, f = density_exact(rows, delta), density_bound(rows, delta)
        assert isinstance(exact, (int, Fraction)) and exact <= Fraction(f) and (delta > 1 - 2.0 ** -40 or Fraction(f) <= exact * Fraction(SLACK))


def test_max_harm_brackets_the_exact_supremum():
    """`max_harm`'s bracket holds the brute-force supremum of the harm over every count vector whose exact acceptance probability is at
    least delta (`JointStratified`'s `maxHarm`), within a few proof blocks' harm; `ln_bound` brackets ln; a failing certificate
    is refused, as Lean's `#guard`s pin."""
    lo, hi = ln_bound(2, 1, False), ln_bound(2, 1, True)
    assert lo < hi < lo + Fraction(1, 2 ** 100) and abs(lo - Fraction(log(2))) < 2.0 ** -52
    assert ln_bound(7, 7, False) == ln_bound(7, 7, True) == 0
    assert ln_bound(100, 1, False) <= ln_upper(Fraction(1, 100)) < ln_bound(100, 1, False) + Fraction(1, 2 ** 63)
    assert max_harm_check([(10, 0, 3, 10, 0, 10)], Fraction(1, 2), 1) == (30, 30 + ln_upper(Fraction(1, 2)))
    for bad in ((10, 1, 1, 5, 0, 8), (10, 1, 1, 6, 0, 9), (10, 1, -1, 0, 0, 9), (10, 1, 1, 0, 3, 2)):
        with pytest.raises(ValueError):
            max_harm_check([bad], Fraction(1, 2), 1)
    rng = random.Random(1)
    for _ in range(300):
        rows = [(n, rng.randint(0, n), rng.choice([0, 1, 3, rng.random() * 10]))
                for n in (rng.randint(0, 7) for _ in range(rng.randint(1, 3)))]
        delta = rng.choice([0.5, 0.1, 0.01, rng.uniform(1e-4, 0.99)])
        lower, upper, lam, cert = max_harm(rows, delta)
        law = Stratified(tuple(Stratum(f"s{i}", n, k) for i, (n, k, _) in enumerate(rows)))
        best = max(sum(Fraction(h) * x for (_, _, h), x in zip(rows, xs))
                   for xs in product(*(range(n + 1) for n, _, _ in rows))
                   if law.escape({f"s{i}": x for i, x in enumerate(xs)}) >= Fraction(delta))
        assert lower <= best <= upper <= best + 4 * max(Fraction(h) for _, _, h in rows) + Fraction(1, 2 ** 40), rows
        assert max_harm_check([(*r, *c) for r, c in zip(rows, cert)], delta, lam) == (lower, upper)


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
    """Three subcircuits of work 1, 8 and 200 at delta = 2^-10 and C = 400, each proof block costing its work."""
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
    """C-Flock's shape at small scale: G, many cheap proof blocks behind a per-statement cost (a0 = 2, a1 = 0.016 per staged
    proof block, at most 4096 proof blocks a statement); N, proof blocks costing 10 per unit of work and no statement cost; M, 64 heavy proof blocks
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
    """The cheapest sampling design sampling one fraction q of every subcircuit (each at its cheapest statement size) that certifies
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


def test_for_budget_closes_the_grid_gaps(batched):
    """Between two frontier points (nine gaps spread over the frontier) the refined plan spends the budget and certifies
    less than the cheaper point."""
    cands, costs, objective, plans = batched
    pairs = list(zip(plans, plans[1:]))
    gaps = 0
    for a, b in pairs[::len(pairs) // 8]:
        budget = (a.cost["total"] + b.cost["total"]) / 2
        got = for_budget(plans, budget, candidates=cands, costs=costs, objective=objective)
        assert got.cost["total"] <= budget and got.certificate.harm <= a.certificate.harm
        assert got.certificate.harm == certify(got.sizes, cands, objective).harm
        gaps += got.certificate.harm < a.certificate.harm
    assert gaps >= 6


def test_row_ands_follows_hm96_table():
    assert [row_ands(n) for n in (4, 256, 4096)] == [174_360, 290_600, 2_034_200]


def test_candidates_fit_a_size_class():
    for bits, gates in ((16, 1), (64, 2), (2048, 128), (UNIT_CLASS_MAX, UNIT_CLASS_MAX // 16)):
        Candidate("a", "A", 1, 1, bits, out_gates=gates)
    for bits, gates in ((UNIT_CLASS_MAX + 1, 1), (UNIT_CLASS_MAX + 32, 8193)):
        with pytest.raises(ValueError, match="size class"):
            Candidate("a", "A", 1, 1, bits, out_gates=gates)
    with pytest.raises(ValueError, match="per_replay"):
        Candidate("a", "A", 4, 1, 16, replay="layer")
    with pytest.raises(ValueError, match="per_replay"):
        Candidate("a", "A", 4, 1, 16, replay="layer", per_replay=5)


def test_derived_costs_read_the_features_and_perturb_scales_one_group():
    c = Candidate("a", "A", 10, 1, 16, ands=1000, rows=(256, 4), commit_bits=64, commit_rows=1.0)
    costs = CostModel(per_and=1e-6, statement_fixed=2.0, statement_per_and=1e-7, staged_per_and=1e-9,
                      staged_per_unit=0.01, commit_per_bit=1e-3, commit_per_row=0.5)
    ands = 1000 + 290_600 + 174_360
    assert costs.unit(c) == pytest.approx(1e-6 * ands)
    assert costs.statement(c) == pytest.approx((2.0 + 1e-7 * ands, 1e-9 * ands + 0.01))
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
    """Three replay blocks, A holding 2 proof blocks of each and B 1: every pair of draws enumerated."""
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
    """One op of R rows of J dots of work W, cut two ways: a form proof block per row (cost 4) that every dot of its row reads, and
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
    """Each form proof block's influence (its row's work) draws all R of them early; then k dots cost 4R + k as a partition and 2k
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
    """Committing the form costs 4 per row whatever the sample, so the switch moves to k = 8R."""
    plans = frontier(_noised(commit_bits=400), _noised_costs(commit_per_bit=0.01), Objective(2.0 ** -20), grid=61)
    rec, form = _crossing(plans)
    assert rec <= 8 * R <= form
    assert {p.cost["commit"] for p in plans} == {0.0, 4.0 * R}


SERVE = (Commitment("commit", "x", 1000.0, decode_bytes=100.0, chain=10.0),
         Commitment("pouw-a", "x", 2000.0, decode_bytes=200.0, chain=30.0, gates=True),
         Commitment("commit", "y", 500.0),
         Commitment("commit", "kv", 400.0, chain=4.0),
         Commitment("kv-append", "kv", 400.0, cost=1.0, chain=6.0),
         Commitment("remote", "receipts", 0.0, cost=7.0))


def test_serve_time_commitments_are_in_every_plan_and_the_budget_is_the_total(batched):
    """Every plan pays the fixed serve-time commitments, each tensor once per scale that commits it, so the frontier only
    moves up by them: a total budget buys what the budget less them bought without them, and less than them buys
    nothing."""
    cands, costs, objective, plans = batched
    served = replace(costs, commit_per_bit=1e-3, serve=SERVE)
    fixed = 8e-3 * (1000 + 2000 + 500 + 400) + 1.0 + 7.0
    assert served.serve_cost() == pytest.approx(fixed)
    moved = frontier(cands, served, objective, grid=41)
    assert [dict(p.sizes) for p in moved] == [dict(p.sizes) for p in plans]
    for p, q in zip(moved, plans):
        assert p.cost["serve"] == pytest.approx(fixed) and p.cost["total"] == pytest.approx(q.cost["total"] + fixed)
    budget = (plans[3].cost["total"] + plans[4].cost["total"]) / 2
    got = for_budget(moved, budget + fixed, candidates=cands, costs=served, objective=objective)
    ref = for_budget(plans, budget, candidates=cands, costs=costs, objective=objective)
    assert dict(got.sizes) == dict(ref.sizes) and got.cost["total"] <= budget + fixed
    assert for_budget(moved, fixed * 0.999, candidates=cands, costs=served, objective=objective) is None
    assert perturb(served, "commit", 2.0).serve_cost() == pytest.approx(2 * fixed)
    with pytest.raises(ValueError, match="non-negative"):
        Commitment("commit", "x", -1.0)


def test_overlaps_count_each_scale_and_price_committing_once():
    """x is committed by the Commit and by PoUW's A rows, whose roots gate the step; kv by the Commit and the KV append
    (measured); y once.  The total counts every copy; committed once, the cheapest copy stays, saving the others' bytes and
    prefill cost and, at decode, the gating chain less the kept copy's, which now gates in its place."""
    costs = CostModel(commit_per_bit=1e-3, serve=SERVE)
    assert costs.serve_cost() == pytest.approx(8e-3 * 3900 + 8.0)
    x, kv = overlaps(costs, latency=0.5)
    assert x["tensor"] == "x" and x["kept"] == "commit" and [c["scale"] for c in x["copies"]] == ["commit", "pouw-a"]
    assert [c["prefill"] for c in x["copies"]] == pytest.approx([8.0, 16.0])
    assert [c["decode"] for c in x["copies"]] == [0.0, 15.0]
    assert x["saving"] == pytest.approx({"bytes": 2000.0, "prefill": 16.0, "decode_bytes": 200.0, "decode": 10.0})
    assert kv["tensor"] == "kv" and kv["kept"] == "kv-append"
    assert kv["saving"] == pytest.approx({"bytes": 400.0, "prefill": 3.2, "decode_bytes": 0.0, "decode": 0.0})


def test_weighing_block_kinds_pays_where_core_counts_a_small_kind_exactly():
    """512 served units of work 1 and 16 PoUW tiles of 32, a tile's statement three times a proof block's.  At equal weights the
    density draws every tile, but core's bound (a sample without replacement) certifies the target with 15: the tiles'
    weight halves and the plan is cheaper, certified by the same stratified bound and registered by its sizes."""
    cands = [Candidate("lin", "S", 512, 1, 16, kind="served"), Candidate("pouw", "P", 16, 32, 1, kind="pouw-tile")]
    costs = CostModel({"S": TemplateCost(1.0, 0.0, 0.0), "P": TemplateCost(3.0, 0.0, 0.0)})
    objective, target = Objective(2.0 ** -20), 192.0
    flat = for_target(cands, costs, objective, target, grid=21)
    plan = weigh(cands, costs, objective, target, grid=21, factors=(0.5, 2.0), rounds=2)
    assert dict(flat.sizes) == {"S": 29, "P": 16} and not flat.weights
    assert dict(plan.weights) == {"pouw-tile": 0.5} and dict(plan.sizes) == {"S": 29, "P": 15}
    assert plan.cost["total"] < flat.cost["total"] and plan.certificate.harm <= target
    assert plan.certificate == certify(plan.sizes, cands, objective)
    assert _registered(plan, cands) == dict(plan.sizes)


def test_under_the_density_bound_kinds_keep_equal_weights(batched):
    """The density bound charges the strata's largest harm per proof block of risk, so a sample in proportion to harm is already
    the cheapest: no weight moves."""
    cands, costs, objective, _ = batched
    for target in (1e6, 5e4):
        flat = for_target(cands, costs, objective, target, grid=21, exact=False)
        plan = weigh(cands, costs, objective, target, grid=21, exact=False)
        assert not plan.weights and plan.cost["total"] == flat.cost["total"]
    with pytest.raises(ValueError, match="weight is positive"):
        frontier(cands, costs, objective, weights={"G": 0.0})


def test_sensitivity_and_benchmarks_flag_what_moves_the_plan():
    """At 240 the covering's 120 dots beat the partition's 4R + 112; a covering dot 25% dearer (96 dots) loses."""
    objective = Objective(2.0 ** -20)
    base, rows = sensitivity(_noised(), _noised_costs(), objective, 240.0, grid=31)
    assert base.choice == {"dot": "recompute"} and dict(base.sizes) == {"DotC": 120}
    assert [r["what"] for r in rows] == ["prove", "prove"]
    assert {r["factor"]: r["same"] for r in rows} == {0.5: False, 2.0: True}
    assert needs_benchmark(_noised(), _noised_costs(0.25), objective, 240.0, grid=31) == ["DotC"]
    assert needs_benchmark(_noised(), _noised_costs(0.01), objective, 240.0, grid=31) == []
    assert needs_benchmark(_noised(), _noised_costs(0.25), objective, 2000.0, grid=31) == []


def test_miss_holds_its_digits_on_huge_populations():
    """A week of served proof blocks is about 10^16: the miss probability against the product it denotes, summed exactly."""
    for n in (2 ** 25 - 3, 2 ** 25 + 3, 10 ** 12, 10 ** 16):
        for m, k in ((1, 1), (4096, 27_713), (10 ** 6, 100), (64, 10 ** 6)):
            t, s = min(m, k), max(m, k)
            assert _miss(n, m, k) == pytest.approx(exp(fsum(log1p(-s / (n - i)) for i in range(t))), rel=1e-6)


def test_shared_statements_hold_only_the_drawn_units():
    """Under `shared` a sample of k pays ceil(k / cap) statements and stages k proof blocks however large n is; cut in order, a
    sparse draw of a huge population touches one statement a sampled proof block."""
    c = Candidate("x", "A", 10 ** 12, 1, 16, ands=1000, cap=64)
    costs = CostModel(per_and=1e-3, statement_fixed=5.0, statement_per_and=1e-2, staged_per_and=1e-4)
    cost, size, parts = template_cost(c, replace(costs, shared=True), 130)
    assert size == 44 and parts["statements"] == pytest.approx(3 * (5.0 + 10.0) + 130 * 0.1)
    assert cost == pytest.approx(parts["statements"] + 130 * 1.0)
    assert template_cost(c, costs, 130)[2]["statements"] == pytest.approx(130 * (15.0 + 0.1), rel=1e-6)
    assert template_cost(c, replace(costs, shared=True), 0)[0] == 0.0


def test_sparse_draws_on_huge_populations_still_touch_statements_and_replay_units():
    """At 10^17 proof blocks the miss probability of a single draw rounds to 1: what it touches is kept as 1 - miss directly."""
    n = 10 ** 17
    assert touched(n, 4, 1) == pytest.approx(1.0, rel=1e-12)
    assert touched(n, 1024, 30_000) == pytest.approx(30_000, rel=1e-6)
    assert _hit(n, 3, 2) == pytest.approx(6 / n, rel=1e-9)
    a = Candidate("x", "A", n, 1, 16, replay="layer", per_replay=10 ** 6)
    costs = CostModel(per_and=1.0, replay={"layer": ReplayDomain(n // 10 ** 6, 2.0)})
    assert replay_cost({"A": 1000}, [a], costs) == pytest.approx(2000.0, rel=1e-6)
