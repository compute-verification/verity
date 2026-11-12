"""The sampling planner (`optimize`): its statement count, replay count and closed-form bound against brute force and core's
bound, `Achievability`'s rates at zero statement cost, the frontier under Flock's statement-batched cost, the choice
between a partition and a covering, serving's commitment (per proof block, and the fixed serve-time total with its overlaps), the
weight classes' weights, the sensitivity queries, and the two-stage design: its exfiltration bound against the
admissible sets' count enumerated and the adversary's best attack at scale, uncapped and capped at the bits a replay
block serves, its proving price with its statements a replay block's or a registration's padded to its public cap, and the
width-stratified design against the uniform one."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from fractions import Fraction
from itertools import combinations, product
from decimal import Decimal, localcontext
from math import ceil, comb, fsum, inf, lgamma, log, log1p, log2, prod, ulp
from pathlib import Path

import pytest

from verity.protocols.spot.verifier.profile import Stratified, Stratum
from verity.protocols.spot.verifier.one_stage.draw import work_k, work_table
from verity.protocols.spot.optimize import (HM96_ANDS, KAPPAS, PROVING_ONLY, SESSIONS, SHA512_ANDS, SHA512_ROWS, Candidate,
                                                           Cell, Commitment, CostModel, Grain, Objective, ReplayGrain,
                                                           MISS_TERMS, TemplateCost, _hit, _log_miss, _miss, certify, density_bound, density_exact,
                                                           for_budget, for_target, frontier,
                                                           grain_costs, grain_optimum, leak_count, leak_draws, ln_bound,
                                                           ln_upper, loc_bits, max_harm, max_harm_check, met_replay,
                                                           needs_benchmark, overlaps, perturb, pinned_ands,
                                                           registration_count, registration_sample, replay_cost, row_ands,
                                                           sensitivity, sqrt_grain, template_cost, touched,
                                                           two_stage_counts, two_stage_design, two_stage_exfil,
                                                           two_stage_price, two_stage_statements, two_stage_strata, weigh,
                                                           work_law)
from verity.circuits.partition_v2 import UNIT_CLASS_MAX

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
    """`Achievability`: q_s = min(1, ln(1/delta) w_s / C) certifies C; the planner samples exactly those rates, rounded up."""
    cands, objective, plan = achievability
    lam = log(1 / objective.delta) / 400.0
    assert dict(plan.sizes) == {c.template: min(c.units, max(1, ceil(min(1.0, lam * c.work) * c.units))) for c in cands}
    assert dict(plan.sizes) == {"A": 70, "B": 70, "C": 20}
    assert not plan.floor_only
    assert plan.certificate.harm <= plan.certificate.density <= 400.0


def _registered(plan, cands) -> dict[str, int]:
    """What `work:1` over `work_law`'s table samples, as the verifier reads it."""
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
    """Flock's shape at small scale: G, many cheap proof blocks behind a per-statement cost (a0 = 2, a1 = 0.016 per staged
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
    """Two hiding compressions and the digest's blocks, the 17 bytes of padding and length included."""
    assert [row_ands(n) for n in (4, 111, 112, 256, 4096)] == [c * SHA512_ROWS for c in (3, 3, 4, 5, 35)]
    assert HM96_ANDS == 2 * SHA512_ROWS


def test_a_compression_costs_its_pin_which_is_the_gadgets_rows():
    """`SHA512_ROWS` is `Sha512Compress_v1`'s circuit-check pin, which is `compress_gadget`'s AND rows on a
    `forms.Circuit`, its committed copies included, which `SHA512_ANDS`, the ripple-carry model's count, leaves out."""
    from verity.circuits.boolean.forms import Circuit
    from verity.commitments import gates
    from verity.commitments.gates.sha512 import compress_gadget

    pins = json.loads(Path(gates.__file__).with_name("lms.circuit-check.json").read_text())["definitions"]
    assert SHA512_ROWS == pinned_ands("Sha512Compress_v1") == pins["Sha512Compress_v1"]["and"]
    C = Circuit()
    out = compress_gadget(C, [C.input(f"cv{i}", 64) for i in range(8)], [C.input(f"m{i}", 64) for i in range(16)])
    C.output("cv", [b for w in out for b in w])
    assert C.counts()["ands"] == SHA512_ROWS > SHA512_ANDS


def test_candidates_fit_a_size_class():
    for bits, gates in ((16, 1), (64, 2), (2048, 128), (UNIT_CLASS_MAX, UNIT_CLASS_MAX // 16)):
        Candidate("a", "A", 1, 1, bits, out_gates=gates)
    for bits, gates in ((UNIT_CLASS_MAX + 1, 1), (UNIT_CLASS_MAX + 32, 8193)):
        with pytest.raises(ValueError, match="size class"):
            Candidate("a", "A", 1, 1, bits, out_gates=gates)
    with pytest.raises(ValueError, match="n_rho"):
        Candidate("a", "A", 4, 1, 16, grain="layer")
    with pytest.raises(ValueError, match="n_rho"):
        Candidate("a", "A", 4, 1, 16, grain="layer", n_rho=5)


def test_derived_costs_read_the_features_and_perturb_scales_one_group():
    c = Candidate("a", "A", 10, 1, 16, ands=1000, rows=(256, 4), commit_bits=64, commit_rows=1.0)
    costs = CostModel(per_and=1e-6, statement_fixed=2.0, statement_per_and=1e-7, staged_per_and=1e-9,
                      staged_per_unit=0.01, commit_per_bit=1e-3, commit_per_row=0.5)
    ands = 1000 + row_ands(256) + row_ands(4)
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


def test_replay_cost_is_the_expected_number_of_replay_blocks_met():
    """Three replay blocks, A holding 2 proof blocks of each and B 1: every pair of samples enumerated."""
    a = Candidate("x", "A", 6, 1, 16, grain="layer", n_rho=2)
    b = Candidate("y", "B", 3, 1, 16, grain="layer", n_rho=1)
    costs = CostModel(per_and=1.0, replay={"layer": ReplayGrain(3, 5.0)})
    for ka, kb in product(range(7), range(4)):
        draws = list(product(combinations(range(6), ka), combinations(range(3), kb)))
        met = sum(len({u // 2 for u in da} | set(db)) for da, db in draws) / len(draws)
        assert replay_cost({"A": ka, "B": kb}, [a, b], costs) == pytest.approx(5.0 * met, abs=1e-12)
    with pytest.raises(ValueError, match="replay blocks"):
        replay_cost({"A": 1}, [a], CostModel(per_and=1.0, replay={"layer": ReplayGrain(2, 1.0)}))
    with pytest.raises(KeyError, match="no replay grain"):
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
    """The covering's largest dot sample and the partition's smallest, checking the frontier switches once."""
    choices = [p.choice["dot"] for p in plans]
    assert choices == sorted(choices, key=lambda c: c == "form") and set(choices) == {"recompute", "form"}
    return (max(p.sizes["DotC"] for p in plans if p.choice["dot"] == "recompute"),
            min(p.sizes["DotP"] for p in plans if p.choice["dot"] == "form"))


def test_covering_wins_sparse_and_partition_wins_once_its_form_units_saturate():
    """Each form proof block's influence (its row's work) samples all R of them early; then k dots cost 4R + k as a partition and 2k
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


@pytest.mark.parametrize("weight", [PROVING_ONLY, 0.5, 2.0])
def test_the_commitments_weight_moves_the_switch_and_proving_only_leaves_them_out(weight):
    """At weight w the partition costs 4R + w 4R + k and the covering 2k, so the switch is at k = 4R (1 + w); proving
    only (w = 0) is where it was with no commitment.  The commitment is still reported, unweighted."""
    costs = replace(_noised_costs(commit_per_bit=0.01), commit_weight=weight)
    plans = frontier(_noised(commit_bits=400), costs, Objective(2.0 ** -20), grid=61)
    rec, form = _crossing(plans)
    assert rec <= 4 * R * (1 + weight) <= form
    assert {p.cost["commit"] for p in plans} == {0.0, 4.0 * R}
    for p in plans:
        assert p.cost["total"] == pytest.approx(p.cost["statements"] + p.cost["units"] + p.cost["replay"]
                                                + weight * (p.cost["commit"] + p.cost["serve"]))
    with pytest.raises(ValueError, match="at least 0"):
        CostModel(commit_weight=-1.0)


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
    unweighed = frontier(cands, replace(served, commit_weight=PROVING_ONLY), objective, grid=41)
    assert [dict(p.sizes) for p in unweighed] == [dict(p.sizes) for p in plans]
    for p, q in zip(unweighed, plans):
        assert p.cost["serve"] == pytest.approx(fixed) and p.cost["total"] == pytest.approx(q.cost["total"])


LAMBDA = log(2.0 ** 40)


def _ladder(total: float, f: float, o: float, beta: float, ratio: float = 2 ** 0.125, lo: float = 1.0,
            hi: float = 2.0 ** 30) -> list[Grain]:
    """Partitions of the same work `total` into n = total/s proof blocks of c(s) = f + o s, each committing beta bytes."""
    out, s = [], lo
    while s <= hi:
        out.append(Grain(f"s={s:.4g}", max(1, round(total / s)), s, 16, beta, f + o * s))
        s *= ratio
    return out


def test_proving_only_puts_the_optimum_at_the_finest_partition():
    """`GranularityOptimum`: while s Λ <= C a sampling design samples (ns/C) Λ proof blocks, whatever s, so proving only costs
    c(s) (F/C) Λ and the finest partition is cheapest (`GranularityChoice`); past C/Λ every proof block is sampled."""
    total, target, delta = 2.0 ** 40, 2.0 ** 34, 2.0 ** -40
    ladder = _ladder(total, 100.0, 0.01, 4.0)
    rows = grain_costs(ladder, delta, h=1e-3, weight=PROVING_ONLY, work=target)
    for g, r in zip(ladder, rows):
        if g.work * LAMBDA <= target:
            assert r["draws"] == pytest.approx(g.units * g.work / target * LAMBDA)
            assert r["total"] == r["prove"] == pytest.approx(g.prove * g.units * g.work / target * LAMBDA)
        else:
            assert r["draws"] == g.units
        assert r["commit"] == pytest.approx(g.units * 4.0 * 1e-3)
    assert grain_optimum(ladder, delta, h=1e-3, weight=PROVING_ONLY, work=target)["grain"] == ladder[0].name


@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_with_the_commitment_the_work_optimum_grows_like_the_root_of_the_target(weight):
    """With n b(s) h = (F/s) β h beside proving, the cheapest partition is `sqrt_grain`'s √(w β h C / (o Λ)) to the
    ladder's step, and quadrupling C doubles it."""
    total, delta, f, o, beta, h = 2.0 ** 44, 2.0 ** -40, 50.0, 0.02, 4.0, 1e-4
    ladder = _ladder(total, f, o, beta)
    got = []
    for target in (2.0 ** 34, 2.0 ** 36, 2.0 ** 38):
        best = grain_optimum(ladder, delta, h=h, weight=weight, work=target)
        star = sqrt_grain(beta, h, target, o, delta, weight)
        assert star * 2 ** -0.25 <= best["work"] <= star * 2 ** 0.25 and star * LAMBDA <= target
        got.append(best["work"])
    assert got[1] / got[0] == pytest.approx(2, rel=0.2) and got[2] / got[1] == pytest.approx(2, rel=0.2)


def _loc_exact(n: int, k: int) -> float:
    return log2(sum(comb(n, j) for j in range(k + 1)))


def test_leak_count_is_certified_and_the_leak_law_draws_n_lambda_over_k_plus_one():
    """K is the most incorrect proof blocks the target allows under `loc_bits` (at least Lean's `locBits`, so b K +
    locBits n K <= ℓ holds for it), and the sampling design samples n Λ / (K + 1), at most n, none once every proof block may be
    incorrect."""
    delta = 2.0 ** -20
    for n, b, ell in ((50, 3, 40.0), (200, 16, 1000.0), (1000, 1, 300.0), (40, 8, 0.0), (64, 16, 5000.0)):
        g = Grain("g", n, 1.0, b, 0.0, 1.0)
        k = leak_count(g, ell)
        assert b * k + loc_bits(n, k) <= ell and b * k + _loc_exact(n, k) <= ell
        assert k == n or b * (k + 1) + loc_bits(n, k + 1) > ell
        assert loc_bits(n, k) >= _loc_exact(n, k) - 1e-9
        draws = leak_draws(g, ell, delta)
        assert draws == (0.0 if k >= n else min(n, n * log(1 / delta) / (k + 1)))
    with pytest.raises(ValueError, match="one target"):
        grain_costs([Grain("g", 4, 1.0, 1, 0.0, 1.0)], 0.5, h=0.0)


def test_leak_count_is_certified_at_any_population():
    """A world year's proof blocks, and targets that allow so few incorrect ones that the float 1 - K / n is 1: the
    count meets the target, and one more block does not, under the entropy in 50-digit decimals."""
    for n, b, ell in ((3348 * 10 ** 18, 16, 1e7), (10 ** 18, 64, 1e6), (10 ** 17, 1, 500.0), (10 ** 21, 4096, 1e9)):
        k = leak_count(Grain("g", n, 1.0, b, 0.0, 1.0), ell)
        with localcontext() as context:
            context.prec = 50

            def exact(count: int) -> Decimal:
                entropy = count * (Decimal(n) / count).ln() + (n - count) * (Decimal(n) / (n - count)).ln()
                return b * count + entropy / Decimal(2).ln()

            assert 0 < k and exact(k) <= Decimal(ell) * (1 + Decimal(2) ** -45) < exact(k + 1) * (1 + Decimal(2) ** -44)


def test_leak_costs_keep_the_refinement_rule():
    """`RefinementRule`: splitting each proof block into r of width b' with b + log2(K+2)/(K+1) <= b' + log2 r costs at least
    (1 - K/n)(1 - Λ/(2(K+1))) of the unsplit partition's c(s) n Λ/(K+1); a split that cuts the width by much more than
    log2 r bits is cheaper."""
    rng = random.Random(3)
    delta = 2.0 ** -40
    lam = log(1 / delta)
    checked = 0
    for _ in range(400):
        n, r, b = rng.randint(10 ** 3, 10 ** 6), rng.choice([2, 4, 16, 256]), rng.randint(16, 4096)
        f, o, s = rng.uniform(0, 10), rng.uniform(0, 1), rng.uniform(1, 1000)
        ell = rng.uniform(1, 400) * b
        whole = Grain("whole", n, s, b, 0.0, f + o * s)
        k = leak_count(whole, ell)
        if not 2 * (k + 1) <= n or lam / (k + 1) >= 1:
            continue
        floor_ = (1 - k / n) * (1 - lam / (2 * (k + 1))) * (f + o * s) * n * lam / (k + 1)
        whole_cost = grain_costs([whole], delta, h=0.0, leak=ell)[0]["total"]
        assert whole_cost == pytest.approx((f + o * s) * n * lam / (k + 1))
        narrow = max(0, ceil(b + log2(k + 2) / (k + 1) - log2(r)))
        for width in (narrow, b):
            split = Grain("split", r * n, s / r, width, 0.0, f + o * s / r)
            assert grain_costs([split], delta, h=0.0, leak=ell)[0]["total"] >= floor_ * (1 - 1e-12)
        checked += 1
    assert checked > 100
    whole = Grain("whole", 10 ** 6, 64.0, 1024, 0.0, 64.0)
    split = Grain("split", 4 * 10 ** 6, 16.0, 16, 0.0, 16.0)
    ell = 2.0 ** 20
    assert grain_costs([split], delta, h=0.0, leak=ell)[0]["total"] < grain_costs([whole], delta, h=0.0, leak=ell)[0]["total"]


def _esc(n: int, k: int, m: int) -> float:
    k = min(k, n)
    return comb(n - m, k) / comb(n, k)


def _admissible_sum(cells: list[Cell], k: dict[int, int], replay_blocks: int, q: float, delta: float,
                    served: float | None = None, union: bool = False) -> float:
    """`CompiledTwo.Exfil`'s count, Σ_{T ∈ admissibleTwo} 2^{b(T)}, enumerating every set T of incorrect proof blocks:
    T is admissible when P̄_a = ∏_ρ (1 − q(1 − ∏_t escCell(n_t, k_t, T's in (ρ, t)))) > δ.  Under `served` a replay
    block's charge is min(b(T_ρ), S); under `union` its choices are each T_ρ of at most S bits, charged 2^{b(T_ρ)}, and
    one for all the others together, charged 2^S at the best escape among them."""
    blocks = [(c.bits, c.stratum) for c in cells for _ in range(int(c.n))]
    sizes: dict[int, int] = {}
    for _, t in blocks:
        sizes[t] = sizes.get(t, 0) + 1
    choices = []
    for mask in range(1 << len(blocks)):
        counts: dict[int, int] = {}
        bits = 0.0
        for i, (b, t) in enumerate(blocks):
            if mask >> i & 1:
                counts[t] = counts.get(t, 0) + 1
                bits += b
        e = 1.0
        for t, n in sizes.items():
            e *= _esc(n, k[t], counts.get(t, 0))
        choices.append((e, bits))
    if union:
        heavy = [e for e, b in choices if b > served]
        choices = [(e, b) for e, b in choices if b <= served] + ([(max(heavy), served)] if heavy else [])
    elif served is not None:
        choices = [(e, min(b, served)) for e, b in choices]
    total = 0.0
    for picks in product(choices, repeat=replay_blocks):
        pa = 1.0
        for e, _ in picks:
            pa *= 1 - q * (1 - e)
        if pa > delta:
            total += 2.0 ** fsum(b for _, b in picks)
    return total


def _every(cells: list[Cell], replay_blocks: int) -> float:
    return replay_blocks * fsum(c.n * log2(1 + 2.0 ** c.bits) for c in cells)


def test_the_two_stage_bound_is_at_least_the_admissible_sets_count():
    """On small designs, every set of incorrect proof blocks enumerated: `two_stage_exfil` is at least log2 of
    `CompiledTwo.Exfil`'s Σ_{T admissible} 2^{b(T)}, at most every T's, and every T's when nothing is sampled."""
    rng = random.Random(11)
    checked = 0
    while checked < 300:
        replay_blocks = rng.choice([1, 2, 3])
        cells = [Cell(rng.choice([1, 2, 3]), rng.choice([0.5, 1.0, 3.0, 8.0, 20.0]), rng.choice([0, 1]))
                 for _ in range(rng.choice([1, 2]))]
        if sum(c.n for c in cells) * replay_blocks > 12:
            continue
        k = {c.stratum: rng.choice([1, 2, 3, 4]) for c in cells}
        q, delta = rng.choice([0.1, 0.5, 0.9, 1.0]), rng.choice([0.5, 0.1, 0.01, 1e-4])
        bound = two_stage_exfil(cells, k, replay_blocks, q, delta)
        assert log2(_admissible_sum(cells, k, replay_blocks, q, delta)) <= bound + 1e-9, (cells, k, q, delta)
        assert bound <= _every(cells, replay_blocks) * (1 + 1e-12)
        checked += 1
    cells = [Cell(2, 3.0), Cell(1, 8.0, 1)]
    assert two_stage_exfil(cells, {0: 1, 1: 1}, 3, 0.0, 0.01) == pytest.approx(_every(cells, 3))
    assert two_stage_exfil(cells, {0: 2, 1: 1}, 1000, 1.0, 0.01) < 0.01


def test_the_capped_bound_is_at_least_the_capped_count():
    """On small designs, every set of incorrect proof blocks enumerated: capped at S served bits a replay block, the
    bound is at least log2 of the sets' count Σ_{T adm} 2^{Σ_ρ min(b(T_ρ), S)} and of the patterns' count, and the sets'
    form is at most the uncapped bound."""
    rng = random.Random(12)
    checked = 0
    while checked < 300:
        replay_blocks = rng.choice([1, 2, 3])
        cells = [Cell(rng.choice([1, 2, 3]), rng.choice([0.0, 0.5, 1.0, 3.0, 8.0, 20.0]), rng.choice([0, 1]))
                 for _ in range(rng.choice([1, 2, 3]))]
        if sum(c.n for c in cells) * replay_blocks > 12:
            continue
        k = {c.stratum: rng.choice([1, 2, 3, 4]) for c in cells}
        q, delta = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0]), rng.choice([0.5, 0.1, 0.01, 1e-4])
        served = rng.choice([0.0, 1.0, 4.0, 10.0, 30.0])
        for union in (False, True):
            bound = two_stage_exfil(cells, k, replay_blocks, q, delta, served, union)
            count = _admissible_sum(cells, k, replay_blocks, q, delta, served, union)
            assert log2(count) <= bound + 1e-9, (cells, k, q, delta, served, union)
        plain = two_stage_exfil(cells, k, replay_blocks, q, delta)
        assert two_stage_exfil(cells, k, replay_blocks, q, delta, served) <= plain * (1 + 1e-6) + 1e-9
        checked += 1
    with pytest.raises(ValueError, match="patterns' form needs one"):
        two_stage_exfil([Cell(2, 3.0)], {0: 1}, 1, 0.5, 0.01, union=True)


def _log2_comb(n: float, m: float) -> float:
    return (lgamma(n + 1) - lgamma(m + 1) - lgamma(n - m + 1)) / log(2)


def test_the_capped_bound_meets_the_adversarys_best_attack_at_scale():
    """Capped at S = 10^5 served bits a replay block, the best attack is many partly incorrect replay blocks: in each of
    m, just over S bits in the stratum whose incorrect blocks escape most cheaply a bit, 25 of stratum 1's 4,096-bit
    blocks.  Each such replay block is caught with probability q (1 − escCell(n_1, k_1, 25)); the sets' form counts
    C(R, m) (C(n_1, 25) 2^S)^m and the patterns' C(R, m) 2^{m S}, and the bound is within 5% of each.  Proved whole,
    every incorrect replay block is caught with probability q: the sets' form counts at least C(R, m) 2^{m(N − 1 + S)}
    (half a replay block's sets hold at least S bits) and the patterns' C(R, m) (C(1000, 625) 2^S)^m, 625 of the
    16-bit blocks at S bits, and the bound is within 1% of each.  One wide block over S alone, its escape weight far
    below a float's least, still counts 2^S in each of the m replay blocks it is in."""
    delta, lam, served = 2.0 ** -20, 20 * log(2), 1e5
    cells = [Cell(10 ** 6, 16.0, 0), Cell(10 ** 4, 4096.0, 1), Cell(100, 65536.0, 2)]
    caught = 1 - _esc(10 ** 4, 2, 25)
    for replay_blocks, q in ((10 ** 9, 1e-4), (10 ** 12, 1e-2), (10 ** 13, 1e-3)):
        m = ceil(lam / -log1p(-q * caught)) - 1
        for union, each in ((False, served + _log2_comb(10 ** 4, 25)), (True, served)):
            attack = _log2_comb(replay_blocks, m) + m * each
            bound = two_stage_exfil(cells, {0: 4, 1: 2, 2: 1}, replay_blocks, q, delta, served, union)
            assert attack <= bound <= 1.05 * attack, (replay_blocks, q, union)
    cells, served = [Cell(1000, 16.0, 0), Cell(10, 4096.0, 1)], 1e4
    for replay_blocks, q in ((10 ** 6, 1e-3), (10 ** 9, 1e-4), (10 ** 12, 1e-2)):
        m = ceil(lam / -log1p(-q)) - 1
        for union, each in ((False, 1010 - 1 + served), (True, served + _log2_comb(1000, 625))):
            attack = _log2_comb(replay_blocks, m) + m * each
            bound = two_stage_exfil(cells, {0: 1000, 1: 10}, replay_blocks, q, delta, served, union)
            assert attack <= bound <= 1.01 * attack, (replay_blocks, q, union)
    cells, served, replay_blocks, q = [Cell(10 ** 5, 16.0), Cell(4, 131072.0)], 4096.0, 10 ** 8, 0.0438
    m = ceil(lam / -log1p(-q * 995 / (10 ** 5 + 4))) - 1
    for union, each in ((False, served + 2), (True, served)):
        attack = _log2_comb(replay_blocks, m) + m * each
        assert attack <= two_stage_exfil(cells, {0: 995}, replay_blocks, q, delta, served, union) <= 1.05 * attack


def test_the_two_stage_bound_meets_the_adversarys_best_attack_at_scale():
    """At scale the bound is within 1% of an attack's count.  Width-stratified with each stratum sampled: m wholly
    incorrect replay blocks, each caught with probability q, so admissible while m < Λ / −ln(1 − q), count
    C(R, m) 2^{m B} at B a replay block's bits, whatever widths the strata hold.  Uniform with k = 1: m incorrect
    blocks of the widest, in distinct replay blocks, each caught with probability q/n_ρ, count C(R, m) n_w^m 2^{m b}."""
    delta, lam = 2.0 ** -20, 20 * log(2)
    cells = [Cell(10 ** 6, 16.0, 0), Cell(10 ** 4, 4096.0, 1), Cell(100, 65536.0, 2)]
    for replay_blocks, q in ((10 ** 6, 1e-3), (10 ** 9, 1e-4), (10 ** 12, 1e-2)):
        m = ceil(lam / -log1p(-q)) - 1
        attack = _log2_comb(replay_blocks, m) + m * fsum(c.n * c.bits for c in cells)
        bound = two_stage_exfil(cells, {0: 4, 1: 2, 2: 1}, replay_blocks, q, delta)
        assert attack <= bound <= 1.01 * attack, (replay_blocks, q)
    n = sum(c.n for c in cells)
    for replay_blocks, q in ((10 ** 12, 1e-2), (10 ** 13, 1e-3)):
        m = ceil(lam / -log1p(-q / n)) - 1
        attack = _log2_comb(replay_blocks, m) + m * (65536 + log2(100))
        bound = two_stage_exfil([replace(c, stratum=0) for c in cells], {0: 1}, replay_blocks, q, delta)
        assert m < replay_blocks / 100 and attack <= bound <= 1.01 * attack, (replay_blocks, q)


def test_a_two_stage_design_samples_every_stratum():
    cells = [Cell(4, 16.0, 0), Cell(4, 32.0, 1)]
    for k in ({0: 1}, {0: 1, 1: 0}):
        with pytest.raises(ValueError, match="k >= 1 in each stratum"):
            two_stage_exfil(cells, k, 10, 0.5, 0.01)
    with pytest.raises(ValueError, match="0 <= q <= 1"):
        two_stage_exfil(cells, {0: 1, 1: 1}, 10, 1.5, 0.01)
    with pytest.raises(ValueError, match="n > 0"):
        Cell(0, 16.0)


def test_a_sampled_replay_block_proves_each_stratums_k_in_its_types_statements():
    """Stratum t's min(k_t, n_t) proof blocks, each type's share of them, a statement's fixed cost once per `cap`, and
    at the chance the type is sampled below one block."""
    cells = [Cell(8, 16.0, prove=1.0, fixed=10.0, cap=4), Cell(2, 16.0, prove=3.0, fixed=6.0, cap=4)]
    assert two_stage_price(cells, {0: 5}) == (5.0, 4.0 + 3.0, 7.0 + 10.0 + 6.0)
    assert two_stage_price(cells, {0: 20}) == (10.0, 8.0 + 6.0, 14.0 + 20.0 + 6.0)
    assert two_stage_price(cells, {0: 1}) == pytest.approx((1.0, 0.8 + 0.6, 1.4 + 8.0 + 1.2))


def test_the_first_stage_samples_a_whole_count_of_a_verification_phase():
    """`registration_count` is the verifier's count at a rate: ⌈q·n⌉ of a verification phase's n replay blocks, so at
    least one at any positive rate, and the rate K/n gives K back.  `registration_sample` is the most K whose cost a
    replay block of the verification phase is within what the budget leaves, and 0 when one is over it."""
    for q, n, want in ((0.0, 7, 0), (1e-9, 7, 1), (0.5, 7, 4), (1.0, 7, 7), (3 / 7, 7, 3), (2.5e-7, 4e6, 1),
                       (2.5e-7, 4.1e6, 2), (6 / 7219.04, 7219.04, 6), (17 / 9.02e7, 9.02e7, 17)):
        assert registration_count(q, n) == want, (q, n)
    for k, n in ((1, 3), (37, 10 ** 6), (10 ** 9, 10 ** 12)):
        assert registration_count(k / n, n) == k and registration_count((k + 0.5) / n, n) == k + 1
    for bad in ((-0.1, 5), (1.5, 5), (0.5, 0.5)):
        with pytest.raises(ValueError, match="a count needs"):
            registration_count(*bad)
    cells = [Cell(8, 16.0, prove=1.0, fixed=10.0, cap=4), Cell(2, 16.0, prove=3.0, fixed=6.0, cap=4)]
    k, n, replay = {0: 5}, 1000, 2.0

    def cost(x: int) -> float:
        return x * (replay + two_stage_price(cells, k, x)[2])

    assert registration_sample(cells, k, n, replay, cost(1) / n * (1 - 1e-9)) == 0
    for most in (1, 2, 7, 64, 999, 1000):
        assert registration_sample(cells, k, n, replay, cost(most) / n) == most
        if most < n:
            assert registration_sample(cells, k, n, replay, cost(most + 1) / n * (1 - 1e-9)) == most
    assert registration_sample(cells, k, n, replay, 1e9) == n and registration_sample(cells, k, 7.9, replay, 1e9) == 7
    assert all(cost(x) <= cost(x + 1) for x in range(1, 200))


def test_a_verification_phases_statements_hold_its_whole_sample():
    """A type's statements of a verification phase hold its blocks of all K sampled replay blocks, split into sessions
    of at most its cap padded to one size, with no padding for the count, which is exact; a type takes them only
    where they cost less than a sampled replay block's own, and with no count it never does.  Each type's `sessions`
    are a sampled replay block's share of them, so K times it is the phase's whole number of sessions where the
    statements are the phase's, and one a sampled replay block of a type whose blocks fit one statement."""
    cells = [Cell(8, 16.0, prove=1.0, fixed=10.0, cap=4), Cell(2, 16.0, prove=3.0, fixed=6.0, cap=4)]
    for k in ({0: 1}, {0: 5}, {0: 20}):
        assert two_stage_price(cells, k, 0) == two_stage_price(cells, k)
    own = [r["prove"] for r in two_stage_statements(cells, {0: 1})]
    assert own == pytest.approx([0.8 + 8.0, 0.6 + 1.2])
    for sampled in (1, 2, 5, 100, 10 ** 4):
        for c, taken, mine, r in zip(cells, (0.8, 0.2), own, two_stage_statements(cells, {0: 1}, sampled)):
            blocks = ceil(sampled * taken - 1e-9)
            sessions = ceil(blocks / c.cap)
            shared = (sessions * c.fixed + sessions * ceil(blocks / sessions) * c.prove) / sampled
            assert r["shared"] == (shared < mine) and r["prove"] == pytest.approx(min(shared, mine))
            assert r["padding"] == (pytest.approx(shared - sessions * c.fixed / sampled - taken * c.prove)
                                    if r["shared"] else 0.0)
            assert r["sessions"] * sampled == pytest.approx(sessions if r["shared"] else sampled * taken)
    assert [r["sessions"] for r in two_stage_statements(cells, {0: 10})] == [8.0 / 4, 1.0]
    assert not any(r["shared"] for r in two_stage_statements(cells, {0: 1}, 1))
    assert all(r["shared"] for r in two_stage_statements(cells, {0: 1}, 10 ** 4))


def test_a_design_samples_a_whole_count_of_each_verification_phase():
    """With a verification phase of n replay blocks, the design samples a whole K of them, the most the budget pays
    for, at the rate K/n: its proving is `two_stage_price`'s at K, one more is over the budget, and its bound is the
    one at K/n.  Its types' statements shared, it samples more replay blocks and bounds less than a statement per
    sampled replay block.  A verification phase so small that one sampled replay block is over the budget has no
    design; with no size given, one verification phase holds every replay block and each statement is a sampled replay
    block's own."""
    delta, replay_blocks, budget = 2.0 ** -20, 10 ** 8, 1e7
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, fixed=5.0, cap=1024, commit_bytes=2.0),
             Cell(4, 131072.0, prove=1e-3, fixed=5.0, cap=4, commit_bytes=2.0 ** 14)]
    kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6}
    left, interior = (budget - 1e5) / replay_blocks, 1e-6 * sum(c.n * c.commit_bytes for c in cells)
    alone = two_stage_design(cells, replay_blocks, delta, **kw)
    assert alone["shared"] == alone["padding"] == 0.0 and alone["registration_sample"] == 0
    assert alone["replay_sample"] == pytest.approx(round(alone["replay_sample"]), abs=1e-6)
    assert two_stage_design(cells, replay_blocks, delta, registration=0.0, **kw) == alone
    assert two_stage_design(cells, replay_blocks, delta, registration=10.0, **kw) is None
    for n in (10 ** 3, 10 ** 4, 10 ** 6):
        row = two_stage_design(cells, replay_blocks, delta, registration=n, **kw)
        q, count = row["q"], row["registration_sample"]
        assert count == int(count) >= 1 and q == count / n and registration_count(q, n) == count
        strata = [replace(c, stratum=0 if c.bits <= row["merged"] else 131072) for c in cells]
        assert registration_sample(strata, row["k"], n, 1.0 + interior, left) == count
        assert row["prove"] == pytest.approx(row["replay_sample"] * two_stage_price(strata, row["k"], count)[2])
        assert row["total"] <= budget * (1 + 1e-12)
        assert (count + 1) * (1.0 + interior + two_stage_price(strata, row["k"], count + 1)[2]) > left * n
        assert row["bound"] == pytest.approx(two_stage_exfil(strata, row["k"], replay_blocks, q, delta), rel=1e-6)
        assert row["sessions"] == pytest.approx(
            count * sum(r["sessions"] for r in two_stage_statements(strata, row["k"], count)))
    assert row["shared"] == 1.0 and 0 <= row["padding"] < 0.01 and row["sessions"] == round(row["sessions"]) >= 2
    assert row["q"] > alone["q"] and row["bound"] < alone["bound"]


def test_the_width_stratified_design_beats_the_uniform_one_when_proving_costs():
    """Many narrow proof blocks and a few wide ones: sampled uniformly, a wide incorrect block is met as rarely as a
    narrow one, so the uniform design proves thousands of blocks a request to price it as a wholly incorrect request,
    and the width-stratified one samples the wide size class apart.  Free proving, the two meet.  The width-stratified
    family merges the classes up to a threshold, the widest one's being the uniform design, so it is never worse; either
    spends at most the budget, and none fits when the boundaries alone are over it."""
    delta, replay_blocks, budget = 2.0 ** -20, 10 ** 8, 1e7
    for prove, apart in ((1e-3, True), (0.0, False)):
        cells = [Cell(10 ** 5, 16.0, prove=prove, commit_bytes=2.0), Cell(4, 131072.0, prove=prove, commit_bytes=2.0 ** 14)]
        rows = {s: two_stage_design(cells, replay_blocks, delta, budget=budget, replay=1.0, boundary=1e5, h=1e-6,
                                    stratify=s) for s in (True, False)}
        for s, row in rows.items():
            assert row["design"] == ("stratified" if s else "uniform") and 0 < row["q"] <= 1
            assert row["merged"] in ({16, 131072} if s else {131072})
            assert set(row["k"]) == ({0} if row["merged"] == 131072 else {0, 131072})
            assert row["total"] == pytest.approx(row["boundary"] + row["replay"] + row["interior"] + row["prove"])
            assert row["total"] <= budget * (1 + 1e-12) and row["commit"] == row["boundary"] + row["interior"]
            assert row["replay_sample"] == pytest.approx(row["q"] * replay_blocks)
            strata = [replace(c, stratum=0 if c.bits <= row["merged"] else 131072) for c in cells]
            assert row["bound"] == two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta)
        assert rows[True]["bound"] <= rows[False]["bound"]
        if apart:
            assert rows[True]["merged"] == 16 and rows[True]["bound"] < 0.5 * rows[False]["bound"]
        else:
            assert rows[True]["bound"] == pytest.approx(rows[False]["bound"], rel=1e-2)
    assert two_stage_design(cells, replay_blocks, delta, budget=1.0, replay=1.0, boundary=2.0) is None


def test_a_design_by_template_has_the_counts_the_verifiers_rule_registers():
    """The verifier's two-stage rule takes one whole count a template, each template a stratum of its own.  A design
    by template has those strata: each cell's count is a whole number of at least one and at most its blocks, its bound
    is the one at those strata, and `two_stage_counts` writes the counts by name with the first stage's K.  Two types
    of one size class, which the width-stratified design samples together at a count they share, get a count each.  A
    width-stratified design has no such counts, and neither has a design whose names don't match its cells."""
    delta, replay_blocks, n = 2.0 ** -20, 10 ** 8, 10 ** 4
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, fixed=5.0, cap=1024, commit_bytes=2.0),
             Cell(300, 12.0, prove=1e-3, fixed=5.0, cap=1024, commit_bytes=2.0),
             Cell(4, 131072.0, prove=1e-3, fixed=5.0, cap=4, commit_bytes=2.0 ** 14)]
    kw = {"budget": 1e7, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "registration": n}
    row = two_stage_design(cells, replay_blocks, delta, stratify="template", **kw)
    assert row["design"] == "template" and row["merged"] == "template" and set(row["k"]) == {0, 1, 2}
    assert all(k == int(k) and 1 <= k <= c.n for k, c in zip((row["k"][i] for i in range(3)), cells))
    strata = [replace(c, stratum=i) for i, c in enumerate(cells)]
    assert two_stage_strata(cells, row["merged"]) == strata
    assert row["bound"] == pytest.approx(two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta), rel=1e-6)
    assert row["total"] <= kw["budget"] * (1 + 1e-12) and row["q"] == row["registration_sample"] / n
    # each cell is a stratum already, so `widths` adds no design; `kappas` holds the family to those scales
    assert two_stage_design(cells, replay_blocks, delta, stratify="template", widths=True, **kw) == row
    held = two_stage_design(cells, replay_blocks, delta, stratify="template", kappas=KAPPAS[-2:], **kw)
    assert held["merged"] == "template" and held["kappa"] in KAPPAS[-2:] and held["bound"] >= row["bound"]
    names = ["narrow", "narrower", "wide"]
    assert two_stage_counts(row, names) == {"sampled": row["registration_sample"],
                                            "k": {name: row["k"][i] for i, name in enumerate(names)}}
    width = two_stage_design(cells, replay_blocks, delta, **kw)
    assert width["design"] == "stratified" and len(width["k"]) < 3
    for bad, given in ((width, names), (row, names[:2]), (row, ["narrow", "narrow", "wide"])):
        with pytest.raises(ValueError, match="by-template design's, one name a cell"):
            two_stage_counts(bad, given)
    with pytest.raises(ValueError, match="width-stratified"):
        two_stage_design(cells, replay_blocks, delta, stratify="type", **kw)


def test_a_design_with_one_session_a_template_holds_it_to_one_table_or_says_how_many():
    """The verifier's second stage reads one session of one table a template of a verification phase.  The cost
    model's own statements split a template's sample into several sessions.  With one session a template whatever
    it holds, each template pays its fixed cost once a verification phase and its session is its whole sample, here
    of several tables, which the row says.  Held to one table, no template samples more than its table holds over
    the phase, so the design samples fewer proof blocks and bounds more.  Both are a by-template design's over a
    verification phase, and the second needs each cell's table."""
    delta, replay_blocks, n = 2.0 ** -20, 10 ** 8, 10 ** 4
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, fixed=5.0, cap=1024, commit_bytes=2.0, table=1024),
             Cell(300, 12.0, prove=1e-3, fixed=5.0, cap=1024, commit_bytes=2.0, table=1024),
             Cell(4, 131072.0, prove=1e-3, fixed=5.0, cap=4, commit_bytes=2.0 ** 14, table=64)]
    kw = {"budget": 1e7, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "registration": n, "stratify": "template"}
    rows = {s: two_stage_design(cells, replay_blocks, delta, sessions=s, **kw) for s in SESSIONS}
    strata = [replace(c, stratum=i) for i, c in enumerate(cells)]
    for s, row in rows.items():
        count = row["registration_sample"]
        st = two_stage_statements(strata, row["k"], count, s)
        assert row["form"] == s and row["total"] <= kw["budget"] * (1 + 1e-12)
        assert row["bound"] == pytest.approx(two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta), rel=1e-6)
        assert row["largest"] == max(r["largest"] for r in st) and row["tables"] == max(r["tables"] for r in st)
        assert row["over"] == sum(1 for r in st if r["tables"] > 1)
        assert row["sessions"] == pytest.approx(count * sum(r["sessions"] for r in st))
    own, whole, table = (rows[s] for s in SESSIONS)
    assert own["sessions"] > 3 and own["tables"] <= 1
    assert whole["sessions"] == pytest.approx(3) and whole["over"] >= 1 and whole["tables"] > 1
    st = two_stage_statements(strata, whole["k"], whole["registration_sample"], "template")
    for c, r in zip(strata, st):
        assert r["largest"] == ceil(whole["registration_sample"] * r["taken"] - 1e-9)
        assert r["prove"] == pytest.approx((c.fixed + r["largest"] * c.prove) / whole["registration_sample"])
    assert whole["bound"] <= own["bound"]
    assert table["sessions"] == pytest.approx(3) and table["over"] == 0 and table["tables"] <= 1
    assert all(table["registration_sample"] * table["k"][i] <= c.table for i, c in enumerate(cells))
    assert table["bound"] > whole["bound"] and table["draws"] < whole["draws"]
    with pytest.raises(ValueError, match="by-template design's"):
        two_stage_design(cells, replay_blocks, delta, sessions="table", **(kw | {"stratify": True}))
    with pytest.raises(ValueError, match="by-template design's"):
        two_stage_design(cells, replay_blocks, delta, sessions="template", **(kw | {"registration": 0.0}))
    with pytest.raises(ValueError, match="needs each cell's `table`"):
        two_stage_design([replace(c, table=inf) for c in cells], replay_blocks, delta, sessions="table", **kw)
    with pytest.raises(ValueError, match="sessions are one of"):
        two_stage_design(cells, replay_blocks, delta, sessions="tables", **kw)
    with pytest.raises(ValueError, match="needs the count"):
        two_stage_statements(strata, own["k"], 0, "template")


def test_a_stratum_a_width_samples_a_narrower_block_at_its_own_rate():
    """A costly type a few bits over a size class shares the next class with a rare block of that class's whole
    width, so the class's stratum samples it at the wide block's rate.  With `widths` each width is a stratum: the
    costly type is sampled at the rate its own bits ask and the bound falls.  The family only grows, so `widths` is
    never worse: where each class holds one width it changes nothing, and a uniform design has no strata to split."""
    delta, replay_blocks, budget, served = 2.0 ** -20, 10 ** 8, 1e7, 4096.0
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, commit_bytes=2.0), Cell(2000, 2100.0, prove=1e-2, commit_bytes=300.0),
             Cell(4, 4096.0, prove=1e-3, commit_bytes=512.0)]
    assert [c.stratum for c in two_stage_strata(cells, 16)] == [0, 4096, 4096]
    assert [c.stratum for c in two_stage_strata(cells, 4096)] == [0, 0, 0]
    assert [c.stratum for c in two_stage_strata(cells[::-1], None)] == [2, 1, 0]
    kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "served": served, "union": True}
    classes = two_stage_design(cells, replay_blocks, delta, **kw)
    row = two_stage_design(cells, replay_blocks, delta, widths=True, **kw)
    assert classes["merged"] == 16 and set(classes["k"]) == {0, 4096}
    assert row["merged"] is None and set(row["k"]) == {0, 1, 2} and row["design"] == "stratified"
    strata = two_stage_strata(cells, None)
    assert row["bound"] == two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta, served, True)
    assert row["prove"] == pytest.approx(row["replay_sample"] * two_stage_price(strata, row["k"])[2])
    assert row["total"] <= budget * (1 + 1e-12)
    assert row["k"][1] / 2000 < 0.6 * classes["k"][4096] / 2004 and row["bound"] < 0.7 * classes["bound"]
    uniform = {"stratify": False, **kw}
    assert two_stage_design(cells, replay_blocks, delta, widths=True, **uniform) == two_stage_design(
        cells, replay_blocks, delta, **uniform)
    apart = [Cell(10 ** 5, 16.0, prove=1e-3, commit_bytes=2.0), Cell(4, 131072.0, prove=1e-3, commit_bytes=2.0 ** 14)]
    kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6}
    assert two_stage_design(apart, replay_blocks, delta, widths=True, **kw) == two_stage_design(
        apart, replay_blocks, delta, **kw)


def test_a_margin_samples_the_strata_that_cost_little_above_their_rule():
    """One stratum is most of the proving and the rest cost little.  With `margin` the family also holds each κ's
    design with every stratum but the costliest sampled the margin above its rule: the costliest keeps its k, the
    rest take theirs times 1 + m, rounded up and at most their blocks, and the bound falls for little proving.  The
    family only grows, so a margin is never worse; none is the planner as it was, and a uniform design has no other
    stratum to raise."""
    delta, replay_blocks, budget, served = 2.0 ** -20, 10 ** 8, 1e7, 4096.0
    cells = [Cell(1000, 16.0, prove=1e-5, commit_bytes=2.0), Cell(2000, 2100.0, prove=1e-2, commit_bytes=300.0),
             Cell(40, 512.0, prove=1e-4, commit_bytes=64.0), Cell(40, 1024.0, prove=1e-4, commit_bytes=128.0),
             Cell(40, 1536.0, prove=1e-4, commit_bytes=128.0), Cell(4, 4096.0, prove=1e-3, commit_bytes=512.0)]
    kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "served": served, "union": True, "widths": True}
    rule = two_stage_design(cells, replay_blocks, delta, **kw)
    assert rule["margin"] == 0.0 and two_stage_design(cells, replay_blocks, delta, margin=0.0, **kw) == rule
    row = two_stage_design(cells, replay_blocks, delta, margin=0.125, **kw)
    assert row["margin"] == 0.125 and (row["merged"], row["kappa"]) == (rule["merged"], rule["kappa"]) == (None, 0.5)
    strata = two_stage_strata(cells, None)
    costly, = (c.stratum for c in strata if c.bits == 2100.0)
    blocks = {c.stratum: c.n for c in strata}
    assert row["k"] == {t: k if t == costly else min(blocks[t], ceil(1.125 * k)) for t, k in rule["k"].items()}
    assert any(row["k"][t] > rule["k"][t] for t in blocks) and row["k"][costly] == rule["k"][costly]
    assert row["bound"] == two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta, served, True)
    assert row["total"] <= budget * (1 + 1e-12) and row["q"] < rule["q"]
    assert row["prove"] / row["replay_sample"] < 1.001 * rule["prove"] / rule["replay_sample"]
    assert row["bound"] < 0.9 * rule["bound"]
    one = [Cell(10 ** 5, 16.0, prove=1e-3, commit_bytes=2.0), Cell(2000, 2100.0, prove=1e-2, commit_bytes=300.0),
           Cell(4, 4096.0, prove=1e-3, commit_bytes=512.0)]
    assert two_stage_design(one, replay_blocks, delta, margin=0.125, **kw) == two_stage_design(
        one, replay_blocks, delta, **kw)
    uniform = {**kw, "stratify": False}
    assert two_stage_design(cells, replay_blocks, delta, margin=0.125, **uniform) == two_stage_design(
        cells, replay_blocks, delta, **uniform)
    with pytest.raises(ValueError, match="margin"):
        two_stage_design(cells, replay_blocks, delta, margin=-0.125, **kw)


def test_a_design_held_to_some_kappas_is_the_best_among_them():
    """`kappas` holds the family to those κ of `KAPPAS`.  All of them is the planner as it was, the best design's own
    κ gives it back, and each κ alone gives the least bound at that κ, never under the best's.  k_t grows with κ, so a
    κ below the best's proves fewer proof blocks a sampled replay block and samples more replay blocks for the same
    budget.  A κ off `KAPPAS`, or none, is refused."""
    delta, replay_blocks, served = 2.0 ** -20, 10 ** 8, 4096.0
    cells = [Cell(1000, 16.0, prove=1e-5, commit_bytes=2.0), Cell(2000, 2100.0, prove=1e-2, commit_bytes=300.0),
             Cell(40, 512.0, prove=1e-4, commit_bytes=64.0), Cell(40, 1024.0, prove=1e-4, commit_bytes=128.0),
             Cell(4, 4096.0, prove=1e-3, commit_bytes=512.0)]
    kw = {"budget": 1e7, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "served": served, "union": True, "widths": True}
    best = two_stage_design(cells, replay_blocks, delta, **kw)
    assert two_stage_design(cells, replay_blocks, delta, kappas=KAPPAS, **kw) == best
    assert two_stage_design(cells, replay_blocks, delta, kappas=(best["kappa"],), **kw) == best
    alone = {kappa: two_stage_design(cells, replay_blocks, delta, kappas=(kappa,), **kw) for kappa in KAPPAS}
    assert all(row["kappa"] == kappa and row["total"] <= kw["budget"] * (1 + 1e-12) for kappa, row in alone.items())
    assert min(row["bound"] for row in alone.values()) == best["bound"]
    assert two_stage_design(cells, replay_blocks, delta, kappas=(2.0 ** -4, 2.0 ** -3), **kw)["bound"] == min(
        alone[2.0 ** -4]["bound"], alone[2.0 ** -3]["bound"])
    low = alone[best["kappa"] / 8]
    assert low["bound"] > best["bound"] and low["q"] > best["q"]
    assert all(low["k"][t] <= k for t, k in best["k"].items()) and sum(low["k"].values()) < sum(best["k"].values()) / 4
    assert alone[0.0]["k"] == dict.fromkeys(best["k"], 1)
    for kappas in ((), (0.3,), (0.5, 3.0)):
        with pytest.raises(ValueError, match="KAPPAS"):
            two_stage_design(cells, replay_blocks, delta, kappas=kappas, **kw)


def test_a_capped_design_charges_what_a_request_serves():
    """Capped at S served bits a request, a design charges a stratum's width at most S and a request its capped count,
    every T counted (at most N + S bits in the sets' form): its bound is the capped `two_stage_exfil` at its rates, at
    most the uncapped design's, and the patterns' form's at most the sets'."""
    delta, replay_blocks, budget, served = 2.0 ** -20, 10 ** 8, 1e7, 4096.0
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, commit_bytes=2.0), Cell(4, 131072.0, prove=1e-3, commit_bytes=2.0 ** 14)]
    assert two_stage_exfil(cells, {0: 1}, 1, 0.0, 0.5, served) <= 10 ** 5 + 4 + served
    for s in (True, False):
        kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "stratify": s}
        plain = two_stage_design(cells, replay_blocks, delta, **kw)
        sets = two_stage_design(cells, replay_blocks, delta, served=served, **kw)
        patterns = two_stage_design(cells, replay_blocks, delta, served=served, union=True, **kw)
        assert patterns["bound"] <= sets["bound"] < plain["bound"]
        for row, union in ((sets, False), (patterns, True)):
            strata = [replace(c, stratum=0 if c.bits <= row["merged"] else 131072) for c in cells]
            assert row["bound"] == two_stage_exfil(strata, row["k"], replay_blocks, row["q"], delta, served, union)
            assert row["total"] <= budget * (1 + 1e-12)


def test_a_replay_block_others_read_is_charged_its_groups_served_bits():
    """A replay grain of fan-out F (`partition.fanout`), some replay block reading values another computed, is charged as
    if each replay block served F S bits, the capped form at N = F: more than at F = 1 where the served bound binds, and a
    design's bound is the capped `two_stage_exfil` at F.  At F = 1, every replay block closed, the charge is the served
    bound's alone.  The sets' form without a served bound doesn't read F, and the patterns' form, each request read in one
    replay block, refuses it."""
    delta, replay_blocks, budget, served = 2.0 ** -20, 10 ** 8, 1e7, 4096.0
    cells = [Cell(10 ** 5, 16.0, prove=1e-3, commit_bytes=2.0), Cell(4, 131072.0, prove=1e-3, commit_bytes=2.0 ** 14)]
    k, q = {0: 995}, 0.0438
    one = two_stage_exfil(cells, k, replay_blocks, q, delta, served)
    assert two_stage_exfil(cells, k, replay_blocks, q, delta, served, fanout=1) == one
    for f in (2, 8):
        shared = two_stage_exfil(cells, k, replay_blocks, q, delta, served, fanout=f)
        assert shared == two_stage_exfil(cells, k, replay_blocks, q, delta, f * served) > one
    assert two_stage_exfil(cells, k, 10, .5, delta, fanout=8) == two_stage_exfil(cells, k, 10, .5, delta)
    kw = {"budget": budget, "replay": 1.0, "boundary": 1e5, "h": 1e-6, "served": served}
    closed, read = (two_stage_design(cells, replay_blocks, delta, fanout=f, **kw) for f in (1, 4))
    assert closed == two_stage_design(cells, replay_blocks, delta, **kw) and read["bound"] > closed["bound"]
    strata = [replace(c, stratum=0 if c.bits <= read["merged"] else 131072) for c in cells]
    assert read["bound"] == two_stage_exfil(strata, read["k"], replay_blocks, read["q"], delta, served, fanout=4)
    with pytest.raises(ValueError, match="needs fan-out 1, not 2"):
        two_stage_exfil(cells, k, 10, .5, delta, served, union=True, fanout=2)
    with pytest.raises(ValueError, match="fan-out is at least 1"):
        two_stage_exfil(cells, k, 10, .5, delta, served, fanout=0)


def test_a_one_stage_sample_replays_the_replay_blocks_it_meets():
    """`met_replay`: d proof blocks sampled uniformly over R replay blocks meet R (1 - (1 - 1/R)^d) of them, each
    replayed once; and `grain_costs` adds it to a one-stage total, which is otherwise unchanged."""
    for blocks, d in ((1, 3), (3, 2), (5, 0), (40, 3)):
        met = sum(len(set(t)) for t in product(range(blocks), repeat=d)) / blocks ** d if d else 0.0
        assert met_replay(ReplayGrain(blocks, 2.0), d) == pytest.approx(2.0 * met, abs=1e-12)
    g = Grain("g", 10 ** 5, 1.0, 16, 2.0, 0.5)
    (bare,) = grain_costs([g], 2.0 ** -40, h=1e-3, work=1e4)
    (row,) = grain_costs([g], 2.0 ** -40, h=1e-3, work=1e4, replay=ReplayGrain(100, 3.0))
    assert row["replay"] == pytest.approx(met_replay(ReplayGrain(100, 3.0), bare["draws"]))
    assert row["total"] == bare["total"] + row["replay"] and row["design"] == "one-stage"
    assert {k: row[k] for k in bare} | {"total": bare["total"]} == bare


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
    density samples every tile, but core's bound (a sample without replacement) certifies the target with 15: the tiles'
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


def test_miss_holds_its_digits_at_any_population():
    """A week of served proof blocks is about 10^16: the log of the miss probability against the product it denotes, in
    exact integers.  Past `MISS_TERMS` terms it is over the exact log by less than the closed form's stated looseness,
    1 / (100 n) of it where the miss is at least 2^-40.  (A sparse sample's hit is 1 - miss: an error of 1e-7 in a log of
    1e-4 is a tenth of a percent of what the sample costs.)"""
    for n in (130, 5000, 10 ** 6, 2 ** 25 - 3, 2 ** 25 + 3, 10 ** 12, 10 ** 16, 10 ** 19):
        for m, k in ((1, 1), (65, 65), (4096, 27_713), (10 ** 6, 100), (64, 10 ** 6), (300, 2000), (1500, 70)):
            if m + k > n:
                assert _miss(n, m, k) == 0.0
                continue
            t, s = min(m, k), max(m, k)
            with localcontext() as context:
                context.prec = 50
                exact = (Decimal(prod(n - s - i for i in range(t))) / Decimal(prod(n - i for i in range(t)))).ln()
                rounding = -exact * Decimal(2) ** -38
                loose = Decimal(s) / (8 * (n - t + 1) * (n - s - t + 1)) if t > MISS_TERMS else Decimal(0)
                if exact >= -28:
                    assert loose <= -exact / (100 * n)
                assert exact - rounding <= Decimal(_log_miss(n, m, k)) <= exact + loose + rounding
                assert Decimal(_miss(n, m, k)) == pytest.approx(exact.exp(), rel=Decimal("1e-9") + loose)


def test_shared_statements_hold_only_the_drawn_units():
    """Under `shared` a sample of k pays ceil(k / cap) statements and stages k proof blocks however large n is; cut in order, a
    sparse sample of a huge population touches one statement a sampled proof block."""
    c = Candidate("x", "A", 10 ** 12, 1, 16, ands=1000, cap=64)
    costs = CostModel(per_and=1e-3, statement_fixed=5.0, statement_per_and=1e-2, staged_per_and=1e-4)
    cost, size, parts = template_cost(c, replace(costs, shared=True), 130)
    assert size == 44 and parts["statements"] == pytest.approx(3 * (5.0 + 10.0) + 130 * 0.1)
    assert cost == pytest.approx(parts["statements"] + 130 * 1.0)
    assert template_cost(c, costs, 130)[2]["statements"] == pytest.approx(130 * (15.0 + 0.1), rel=1e-6)
    assert template_cost(c, replace(costs, shared=True), 0)[0] == 0.0


def test_sparse_draws_on_huge_populations_still_touch_statements_and_replay_blocks():
    """At 10^17 proof blocks the miss probability of a single sample rounds to 1: what it touches is kept as 1 - miss directly."""
    n = 10 ** 17
    assert touched(n, 4, 1) == pytest.approx(1.0, rel=1e-12)
    assert touched(n, 1024, 30_000) == pytest.approx(30_000, rel=1e-6)
    assert _hit(n, 3, 2) == pytest.approx(6 / n, rel=1e-9)
    a = Candidate("x", "A", n, 1, 16, grain="layer", n_rho=10 ** 6)
    costs = CostModel(per_and=1.0, replay={"layer": ReplayGrain(n // 10 ** 6, 2.0)})
    assert replay_cost({"A": 1000}, [a], costs) == pytest.approx(2000.0, rel=1e-6)
