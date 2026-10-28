"""IntegrityProfile: what verification established, and the worst case a consumer's utility reads off it.

The two consumer utilities below stand in for PoUW's incorrect-unit bound and the KV touched-unit bound, which live
with their consumers, not in core.  `two_level` is sampled proofs' shape (n_r units drawn with p, then k of the n_v
inside each), and `_before_levels` is the profile's arithmetic as it stood in those four numbers (every pair of m tried), kept as the
oracle the level form must reproduce: `accept` exactly, and `worst_case` from above, within its directed margin.
`_exact_rhos` and `_exact_lp` are the high-precision reference (60-digit `Decimal`): `worst_case` is never below it."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
import random
from decimal import Decimal, localcontext
from math import comb, exp, inf, log, sqrt

import pytest

from verity.ml.circuits import Program, bind
from verity.ml.circuits import query_ast as G
from verity.ml.circuits.partition import Level, validate_partition
from verity.ml.circuits.ops import gemm
from verity.core.protocols.profile import (HARM_MARGIN, MARGIN, Bernoulli, IntegrityProfile, Stratified, Stratum, Subset, Work,
                                          WorkStratum, work_k)


def two_level(n_r, n_v, p, k, delta, **facts) -> IntegrityProfile:
    draws = {"replay": Bernoulli(p), "check": Subset(k)}
    return IntegrityProfile(("replay", "check"), (n_r, n_v), draws, delta, **facts)


def _before_levels(n_r, n_v, p, k, delta):
    """(accept, worst_case) exactly as `IntegrityProfile(n_r, n_v, p, k, delta)` computed them."""
    def accept(m):
        return 1 - Fraction(p) + Fraction(p) * Fraction(comb(n_v - m, k), comb(n_v, k))

    def rho(m):
        a = accept(m)
        return inf if a == 0 else -log(a)

    def worst_case(utility):
        budget = log(1 / delta)
        options = [(float(u), rho(m)) for m in range(1, n_v + 1) if (u := utility(m)) > 0]
        options = [(u, r) for u, r in options if r < inf]
        if not options:
            return 0.0
        best = max(u * (n_r if r == 0 else min(n_r, budget / r)) for u, r in options)
        for (ua, ra), (ub, rb) in combinations(sorted(options, key=lambda o: o[1]), 2):
            if rb > ra and ra * n_r < budget < rb * n_r:
                xb = (budget - n_r * ra) / (rb - ra)
                best = max(best, ua * (n_r - xb) + ub * xb)
        return best

    return accept, worst_case


def incorrect_units(profile: IntegrityProfile) -> float:
    return profile.worst_case(lambda m: m)


def touched_units(profile: IntegrityProfile) -> float:
    return profile.worst_case(lambda m: 1)

TOY = two_level(n_r=512, n_v=4, p=Fraction(1, 2), k=2, delta=0.01)


def test_the_prototype_toy_numbers():
    """The protocol prototype's composed toy: 26.6 incorrect VUs for PoUW, 16.0 touched RUs for a KV claim."""
    assert round(incorrect_units(TOY), 1) == 26.6
    assert round(touched_units(TOY), 1) == 16.0
    assert TOY.worst_case(lambda m: 4 * m) == pytest.approx(4 * incorrect_units(TOY))


def test_consumers_are_maximised_by_different_strategies():
    a = two_level(n_r=10_000, n_v=4, p=Fraction(1, 4), k=2, delta=0.01)
    ratio = {m: m / a.rho(m) for m in range(1, 5)}
    touched = {m: 1 / a.rho(m) for m in range(1, 5)}
    assert max(ratio, key=ratio.get) == 4      # free compute concentrates into fully incorrect replay units
    assert max(touched, key=touched.get) == 1  # KV laundering spreads one incorrect verification unit per replay unit
    assert incorrect_units(a) == pytest.approx(ratio[4] * log(100))
    assert touched_units(a) == pytest.approx(touched[1] * log(100))


def test_accept_is_exact():
    assert TOY.accept(0) == 1 and TOY.accept(1) == Fraction(3, 4) and TOY.accept(4) == Fraction(1, 2)
    assert two_level(n_r=1, n_v=4, p=Fraction(1), k=4, delta=0.5).accept(1) == 0


def test_full_check_admits_nothing_and_no_check_admits_everything():
    assert incorrect_units(two_level(n_r=100, n_v=4, p=Fraction(1), k=4, delta=0.01)) == 0
    assert incorrect_units(two_level(n_r=100, n_v=4, p=Fraction(0), k=4, delta=0.01)) == 400
    assert touched_units(two_level(n_r=100, n_v=4, p=Fraction(1, 2), k=0, delta=0.01)) == 100


def test_the_bound_uses_n_r_when_it_binds():
    few = two_level(n_r=3, n_v=4, p=Fraction(1, 100), k=1, delta=0.01)
    assert incorrect_units(few) == pytest.approx(12)


@pytest.mark.parametrize("n_r, n_v, p, k, delta", [(3, 3, Fraction(1, 2), 1, 0.3), (4, 2, Fraction(1, 3), 1, 0.2),
                                                   (3, 4, Fraction(2, 3), 2, 0.4), (5, 1, Fraction(1, 4), 1, 0.1)])
@pytest.mark.parametrize("utility", [lambda m: m, lambda m: 1, lambda m: m * m, lambda m: 3 * m])
def test_the_bound_covers_every_admitted_integer_strategy(n_r, n_v, p, k, delta, utility):
    """Brute force: every strategy accepted with probability above delta stays within the bound."""
    a = two_level(n_r=n_r, n_v=n_v, p=p, k=k, delta=delta)
    bound = a.worst_case(utility)
    best = 0.0
    for ms in product(range(n_v + 1), repeat=n_r):
        accept = 1.0
        for m in ms:
            accept *= float(a.accept(m))
        if accept > delta:
            best = max(best, sum(utility(m) for m in ms if m))
    assert best <= bound + 1e-9
    assert bound <= n_r * max(utility(m) for m in range(1, n_v + 1)) + 1e-9


def test_a_bernoulli_law_is_the_single_unit_case():
    rate = Fraction(1, 10)
    a = two_level(n_r=10_000, n_v=1, p=rate, k=1, delta=0.01)
    assert a.accept(1) == 1 - rate
    assert incorrect_units(a) == pytest.approx(log(100) / -log(0.9))
    assert exp(-a.rho(1) * incorrect_units(a)) == pytest.approx(0.01)


@pytest.mark.parametrize("kw", [dict(n_v=0), dict(k=5), dict(p=Fraction(3, 2)), dict(delta=0.0), dict(delta=1.0), dict(n_r=-1)])
def test_ill_formed_profiles_are_refused(kw):
    with pytest.raises(ValueError):
        two_level(**{**dict(n_r=1, n_v=4, p=Fraction(1, 2), k=2, delta=0.01), **kw})


def test_json_carries_the_law_and_the_exhaustive_facts():
    assert TOY.to_json() == {"programs": [], "prescribed": [], "linked": False, "covered": [],
                             "levels": [{"name": "replay", "query": None, "size": 512, "draw": {"bernoulli": "1/2"}},
                                        {"name": "check", "query": None, "size": 4, "draw": {"subset": 2}}],
                             "delta": 0.01, "drawn": {}}


def test_exhaustive_facts_travel_with_the_law():
    p = two_level(8, 4, Fraction(1, 2), 2, 0.01, programs=["ab", "cd"], prescribed=["weights", "noise"], linked=True,
                  covered=["TcDot16_v1"], drawn={"replay": 3, "check": 6})
    assert p.programs == ("ab", "cd") and p.prescribed == {"weights", "noise"} and p.linked and p.covered == {"TcDot16_v1"}
    assert p.to_json()["prescribed"] == ["noise", "weights"] and p.drawn == {"replay": 3, "check": 6}


@pytest.mark.parametrize("n_r", [0, 1, 3, 512, 10_000])
@pytest.mark.parametrize("n_v", [1, 2, 4, 6])
@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)])
@pytest.mark.parametrize("delta", [0.01, 0.3])
def test_the_two_level_case_reproduces_the_four_number_law_exactly(n_r, n_v, p, delta):
    for k in range(n_v + 1):
        accept, worst_case = _before_levels(n_r, n_v, p, k, delta)
        a = two_level(n_r, n_v, p, k, delta)
        assert all(a.accept(m) == accept(m) for m in range(n_v + 1))
        for utility in (lambda m: m, lambda m: 1, lambda m: m * m, lambda m: 4 * m):
            new, old = a.worst_case(utility), worst_case(utility)
            assert old * (1 - 1e-12) - 1e-12 <= new <= old * (1 + 4 * MARGIN) + 1e-12


def test_undrawn_levels_only_group():
    """Levels above the Bernoulli level or between it and the checked level change counts, not the law."""
    above = IntegrityProfile(("request", "replay", "check"), (32, 16, 4),
                             {"replay": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    between = IntegrityProfile(("replay", "tile", "check"), (512, 2, 2),
                               {"replay": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    for a in (above, between):
        assert [a.accept(m) for m in range(5)] == [TOY.accept(m) for m in range(5)]
        assert incorrect_units(a) == incorrect_units(TOY) and touched_units(a) == touched_units(TOY)
    assert above.count("request") == 32 and above.count("replay") == 512 and above.count("check") == 2048
    assert above.count("check", within="replay") == 4 and above.count("replay", within="request") == 16
    assert between.count("tile", within="replay") == 2 and between.count("replay", within="replay") == 1


def test_the_cap_bounds_a_touched_utility_read_above_the_replay_level():
    """16 replay units may be touched, but 8 requests of 64 replay units each hold at most 8 touched requests."""
    a = IntegrityProfile(("request", "replay", "check"), (8, 64, 4),
                         {"replay": Bernoulli(Fraction(1, 2)), "check": Subset(2)}, 0.01)
    assert round(touched_units(a), 1) == 16.0
    assert a.worst_case(lambda m: 1, cap=a.count("request")) == 8
    assert a.worst_case(lambda m: 1, cap=100) == touched_units(a)
    assert a.worst_case(lambda m: 0, cap=8) == 0
    with pytest.raises(ValueError):
        a.worst_case(lambda m: 1, cap=-1)


def test_one_bernoulli_level_and_one_subset_level_are_the_degenerate_chains():
    rate = Fraction(1, 10)
    single = IntegrityProfile(("unit",), (10_000,), {"unit": Bernoulli(rate)}, 0.01)
    assert single.accept(1) == 1 - rate and incorrect_units(single) == incorrect_units(two_level(10_000, 1, rate, 1, 0.01))
    whole = IntegrityProfile(("call",), (100,), {"call": Subset(3)}, 0.01)
    assert [whole.accept(m) for m in range(101)] == [two_level(1, 100, Fraction(1), 3, 0.01).accept(m) for m in range(101)]


def test_levels_keep_their_query_ids():
    a = IntegrityProfile((Level("replay", "q-ru"), "check"), (4, 2), {"replay": Bernoulli(1), "check": Subset(1)}, 0.1)
    assert a.levels == (Level("replay", "q-ru"), Level("check"))
    assert a.to_json()["levels"][0]["query"] == "q-ru"


@pytest.mark.parametrize("levels, sizes, draws", [
    (("a", "b"), (4, 2), {"a": Bernoulli(Fraction(1, 2)), "b": Bernoulli(Fraction(1, 2))}),  # two Bernoulli levels
    (("a", "b", "c"), (4, 2, 2), {"b": Subset(1), "c": Subset(1)}),                         # Subset above the checked level
    (("a", "b"), (4, 2), {"a": Bernoulli(Fraction(1, 2))}),                                  # checked level undrawn
    (("a", "b"), (4, 2), {"b": Subset(1), "z": Bernoulli(1)}),                               # unknown level
    (("a", "a"), (4, 2), {"a": Subset(1)}),                                                  # duplicate names
    (("a", "b"), (4,), {"b": Subset(1)}),                                                    # a size missing
    (("a", "b"), (4, 2), {"b": 1}),                                                          # not a draw
    (("a", "t", "b"), (4, 2, 2), {"a": Bernoulli(1), "b": Subset(5)}),                       # k above n = 4
    ((), (), {}),                                                                            # no levels
])
def test_ill_formed_chains_are_refused(levels, sizes, draws):
    with pytest.raises(ValueError):
        IntegrityProfile(levels, sizes, draws, 0.01)


def test_counts_name_known_levels_coarse_to_fine():
    with pytest.raises(ValueError):
        TOY.count("replay", within="check")
    with pytest.raises(KeyError):
        TOY.count("tile")
    with pytest.raises(ValueError):
        two_level(1, 4, Fraction(1, 2), 2, 0.01, drawn={"tile": 1})


def test_a_chain_read_off_a_program_gives_the_profile_its_sizes():
    """GEMM coordinates as the Bernoulli level, the two calls inside each as the checked level, both leaving out
    the Inputs (checked outside the draws); the IR supplies the counts and the query ids."""
    program = Program(bind(gemm.Gemm, K=32, N=4, **gemm.GEMM_AMPERE))
    coarse_q = G.instances("//GemmCoordinate")
    fine_q = G.each(coarse_q, G.default(32))
    coarse, fine = G.evaluate(program, coarse_q), G.evaluate(program, fine_q)
    computed = [(32 + 4 * 32, program.gates)]
    assert validate_partition(program, coarse, within=computed).status == "exact"
    assert validate_partition(program, fine, within=computed).status == "exact"
    per, rest = divmod(fine.count(), coarse.count())   # `each` nests fine in coarse, the same count in every coordinate
    assert (per, rest) == (2, 0)
    a = IntegrityProfile((Level("coordinate", G.query_id(coarse_q)), Level("call", G.query_id(fine_q))),
                         (coarse.count(), per), {"coordinate": Bernoulli(Fraction(1, 2)), "call": Subset(1)}, 0.01)
    assert a.count("call") == fine.count() == 8 and a.accept(2) == Fraction(1, 2)


def one_level(n, k, delta) -> IntegrityProfile:
    return IntegrityProfile((Level("unit"),), (n,), {"unit": Subset(k)}, delta)


#: The one-stage audits' wrong-unit bounds (lane one-stage-e2e): (n units, k drawn, delta, the exact optimum to 12 digits,
#: at most).  `worst_case` is within its margin above the exact optimum.
E2E = {"A0": (1024, 16, 0.01, 254.226320800, 254),
       "A1": (1024, 256, 2.0 ** -20, 46.9243795425, 46),
       "A3": (344, 128, 2.0 ** -20, 28.2555988015, 28),
       "A2": (183_680, 256, 2.0 ** -20, 9675.41945720, 9675)}


@pytest.mark.parametrize("run", sorted(E2E))
def test_the_one_stage_audit_bounds(run):
    """A2 is serving's real population (183,680 units); the quadratic, float `worst_case` crashed there."""
    n, k, delta, value, at_most = E2E[run]
    bound = one_level(n, k, delta).worst_case(lambda m: m)
    exact = _exact_threshold(n, k, delta)
    assert float(exact) == pytest.approx(value, rel=1e-11)
    assert exact <= Decimal(bound) <= exact * (1 + 4 * Decimal(MARGIN))
    assert int(bound) == at_most


def _largest_admitted(n, k, delta) -> int:
    """The largest K with C(n - K, k)/C(n, k) >= delta, in exact rationals: `audit_count` (`Audit/OneStage.lean`)
    bounds Pr[accept and at least K wrong] by that ratio (`Law.subset_miss`) plus the per-unit terms."""
    d, top = Fraction(delta), comb(n, k)
    lo, hi = 0, n + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if Fraction(comb(n - mid, k), top) >= d else (lo, mid)
    return lo


@pytest.mark.parametrize("run", sorted(E2E))
def test_the_count_bound_is_the_lean_curve(run):
    """For one level drawn Subset(k), `accept(K)` is the Lean count curve's sampling term C(n - K, k)/C(n, k)
    (`Law.subset_miss`), and the integer bound is the largest count it does not exclude at delta."""
    n, k, delta, _, at_most = E2E[run]
    a = one_level(n, k, delta)
    top = comb(n, k)
    for K in {0, 1, k, n // 3, at_most, at_most + 1, n - k, n - k + 1, n}:
        assert a.accept(K) == Fraction(comb(n - K, k), top)
    assert _largest_admitted(n, k, delta) == at_most == int(a.worst_case(lambda m: m))


def test_rho_is_exact_where_accept_underflows():
    """At A2's size accept(m) falls far below the smallest float; rho stays exact, from integer binomials."""
    n, k = 183_680, 256
    a = one_level(n, k, 2.0 ** -20)
    assert a.accept(n - k) > 0 and float(a.accept(n - k)) == 0.0
    assert a.rho(n - k) == pytest.approx(log(comb(n, k)), rel=1e-15)
    assert a.rho(n - k + 1) == inf and a.rho(0) == 0.0
    assert [a.rho(m) for m in (0, 1, 1000, 9675, 9676, n - k)] == [a._rhos()[m] for m in (0, 1, 1000, 9675, 9676, n - k)]
    with pytest.raises(ValueError):
        a.rho(n + 1)


def _exact_threshold(n, k, delta) -> Decimal:
    """For one level drawn Subset(k), rho is convex from rho(0) = 0, so the relaxation's optimum interpolates between the
    last admitted count and the next (one-stage-e2e's direct bound, #116 6ccb9b7b), here in 60-digit Decimal."""
    with localcontext() as ctx:
        ctx.prec = 60
        lo = _largest_admitted(n, k, delta)
        top = Decimal(comb(n, k)).ln()
        ra, rb = top - Decimal(comb(n - lo, k)).ln(), top - Decimal(comb(n - lo - 1, k)).ln()
        return lo + ((1 / Decimal(delta)).ln() - ra) / (rb - ra)


def test_worst_case_is_the_convex_threshold_for_one_subset_level():
    for n, k, delta, _, _ in E2E.values():
        exact = _exact_threshold(n, k, delta)
        assert exact <= Decimal(one_level(n, k, delta).worst_case(lambda m: m)) <= exact * (1 + 4 * Decimal(MARGIN))


def _exact_rhos(a: IntegrityProfile) -> list:
    """rho(0..n) in 60-digit Decimal, None for inf: -ln of `accept`'s exact ratio."""
    _, n, p, k = a._law()
    num, den, top = p.numerator, p.denominator, comb(n, k)
    with localcontext() as ctx:
        ctx.prec = 60
        whole = Decimal(den * top).ln()
        return [None if (x := (den - num) * top + num * comb(n - m, k)) == 0 else whole - Decimal(x).ln()
                for m in range(n + 1)]


def _exact_lp(a: IntegrityProfile, utility, rhos=None) -> Decimal:
    """The relaxation's exact optimum, independently of the envelope: members times the best value, at mean risk at most
    t = ln(1/delta)/members, of a single point or of a chord between a point below t and one above (brute force), in
    60-digit Decimal."""
    members, n, _, _ = a._law()
    rhos = _exact_rhos(a) if rhos is None else rhos
    with localcontext() as ctx:
        ctx.prec = 60
        if members == 0:
            return Decimal(0)
        t = (1 / Decimal(a.delta)).ln() / members
        pts = [(Decimal(0), Decimal(0))] + [(r, Decimal(utility(m))) for m, r in enumerate(rhos)
                                            if m >= 1 and r is not None and utility(m) > 0]
        lo, hi = [q for q in pts if q[0] <= t], [q for q in pts if q[0] > t]
        best = max(u for _r, u in lo)
        for ra, ua in lo:
            for rb, ub in hi:
                best = max(best, ua + (ub - ua) * (t - ra) / (rb - ra))
        return best * members


UTILITIES = {"m": lambda m: m, "one": lambda m: 1, "m2": lambda m: m * m, "4m": lambda m: 4 * m, "sqrt": lambda m: sqrt(m)}


def _grid(count: int, seed: int = 135):
    rng = random.Random(seed)
    for _ in range(count):
        n_v = rng.randint(1, 40)
        p = rng.choice([Fraction(0), Fraction(1, 1000), Fraction(1, 7), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)])
        yield two_level(rng.choice([0, 1, 2, 3, 7, 50, 512, 10_000]), n_v, p, rng.randint(0, n_v),
                        rng.choice([0.3, 0.01, 2.0 ** -20, 1e-9]))


def test_worst_case_is_never_below_the_exact_optimum():
    """On a seeded grid (small risks included, where a log difference would cancel), the bound is at or above the
    high-precision optimum, and within the margin of it."""
    for a in _grid(160):
        rhos = _exact_rhos(a)
        for name, u in UTILITIES.items():
            exact, got = _exact_lp(a, u, rhos), Decimal(a.worst_case(u))
            assert exact <= got <= exact * (1 + 4 * Decimal(MARGIN)) + Decimal(1e-300), (a.sizes, a.draws, a.delta, name)


def test_rho_is_accurate_to_a_few_ulps():
    """Against the 60-digit reference, including risks near 0 (p = 1/1000, k = 1) and the underflow regime."""
    cases = list(_grid(80)) + [two_level(1, 250, Fraction(1, 1000), 1, 0.01), two_level(1, 4000, Fraction(1), 256, 2.0 ** -20),
                               two_level(50, 4000, Fraction(1, 2), 256, 2.0 ** -20)]
    worst = 0.0
    for a in cases:
        for m, (r, x) in enumerate(zip(a._rhos(), _exact_rhos(a))):
            assert (r == inf) == (x is None) and r == a.rho(m)
            if x is not None and x > 0:
                worst = max(worst, abs(float((Decimal(r) - x) / x)))
    assert worst < 1e-15


def test_the_red_team_cases():
    """Where the plain log difference put the bound below the exact optimum (red-team-flock-3 on b4c9a489): n = 4,000,
    k = 256, p = 1/2, 50 members, by 8.5e-14; the exact optimum here is the envelope evaluated in Decimal."""
    a = two_level(50, 4000, Fraction(1, 2), 256, 2.0 ** -20)
    rhos = _exact_rhos(a)
    with localcontext() as ctx:
        ctx.prec = 60
        t = (1 / Decimal(a.delta)).ln() / 50
        hull = []
        for q in sorted([(Decimal(0), Decimal(0))] + [(r, Decimal(m)) for m, r in enumerate(rhos) if m and r is not None]):
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (q[1] - hull[-2][1]) >= (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0]):
                hull.pop()
            hull.append(q)
        seg = max(i for i, q in enumerate(hull) if q[0] <= t)
        (ra, ua), (rb, ub) = hull[seg], hull[seg + 1]
        exact = 50 * (ua + (ub - ua) * (t - ra) / (rb - ra))
    got = Decimal(a.worst_case(lambda m: m))
    assert exact <= got <= exact * (1 + 4 * Decimal(MARGIN))



def test_the_float_pairwise_version_crashed_where_the_profile_now_serves():
    """The profile's arithmetic before this fix: -log(float(accept)) underflows once accept < 1e-308, as it does at
    n = 4,000, k = 256 (accept(n - k) = 1/C(4000, 256), about e^-960)."""
    _, before = _before_levels(1, 4000, Fraction(1), 256, 2.0 ** -20)
    with pytest.raises(ValueError):
        before(lambda m: m)
    bound = two_level(1, 4000, Fraction(1), 256, 2.0 ** -20).worst_case(lambda m: m)
    assert int(bound) == _largest_admitted(4000, 256, 2.0 ** -20)


# ---- Stratified: independent uniform draws inside strata (docs: sampling-strategies) -------------------------------

#: A4 P6's layer 0 (one-stage-e2e): six templates, 6,771,765 units, in canonical order.
A4_SIZES = {"rms_triton": 287, "gemm2048": 6_171_648, "rope": 11_480, "rms_fused": 287, "silu_mul": 287,
            "gemm8192": 587_776}


def a4_law(ks: dict) -> Stratified:
    return Stratified(tuple(Stratum(name, size, ks[name]) for name, size in A4_SIZES.items()))


def _enumerate(law: Stratified):
    """Every unit as (stratum index, unit index), and every equally likely draw of the law."""
    units = [(i, j) for i, s in enumerate(law.strata) for j in range(s.size)]
    per = [[{(i, j) for j in c} for c in combinations(range(s.size), s.k)] for i, s in enumerate(law.strata)]
    draws = [set().union(*parts) for parts in product(*per)]
    return units, draws


def _escapes(law: Stratified):
    """(wrong set, exact escape) for every subset of the units, by counting draws."""
    units, draws = _enumerate(law)
    for r in range(len(units) + 1):
        for b in combinations(units, r):
            yield set(b), Fraction(sum(not (d & set(b)) for d in draws), len(draws))


def _counts(law: Stratified, b) -> dict:
    return {s.name: sum(1 for i, _ in b if i == idx) for idx, s in enumerate(law.strata)}


SMALL = [Stratified((Stratum("a", 4, 2), Stratum("b", 3, 1), Stratum("c", 3, 0))),
         Stratified((Stratum("a", 5, 1), Stratum("b", 5, 1))),                          # tied strata
         Stratified((Stratum("a", 6, 3), Stratum("b", 2, 2), Stratum("c", 1, 1))),
         Stratified.uniform(9, 2)]


@pytest.mark.parametrize("law", SMALL)
def test_stratified_escape_miss_and_count_bound_by_enumeration(law):
    """Escape depends only on the counts per stratum and is exact; miss(K) is the largest escape of K wrong units; the
    count bound is the largest K with miss(K) >= delta."""
    best = {}
    for b, e in _escapes(law):
        assert law.escape(_counts(law, b)) == e
        best[len(b)] = max(best.get(len(b), Fraction(0)), e)
    assert [law.miss(K) for K in range(law.population + 1)] == [best[K] for K in range(law.population + 1)]
    for delta in (0.9, 0.5, 0.3, 0.1, 0.01, 1e-6):
        assert law.count_bound(delta) == max(K for K, e in best.items() if e >= Fraction(delta))


@pytest.mark.parametrize("law", SMALL)
def test_stratified_harm_bound_brackets_the_integer_optimum(law):
    """The relaxation is at or above the best integer strategy and within one unit's harm of it (only its last unit is
    fractional); harms per stratum, and several harms inside one stratum."""
    harms = [{s.name: float(i + 1) for i, s in enumerate(law.strata)},
             {law.strata[0].name: [(5.0, 1), (2.0, law.strata[0].size - 1)]},
             {s.name: 1.0 for s in law.strata}]
    for harm in harms:
        def unit_harm(i, j):
            h = harm.get(law.strata[i].name, 0.0)
            if isinstance(h, float):
                return h
            for v, c in h:
                if j < c:
                    return v
                j -= c
            return 0.0
        top = max(unit_harm(i, j) for i, s in enumerate(law.strata) for j in range(s.size))
        for delta in (0.5, 0.1, 0.01):
            exact = max(sum(unit_harm(i, j) for i, j in b) for b, e in _escapes(law) if e >= Fraction(delta))
            bound = law.harm_bound(harm, delta)
            assert exact <= bound <= (exact + top) * (1 + 4 * HARM_MARGIN) + 1e-12, (law, harm, delta)


@pytest.mark.parametrize("run", sorted(E2E))
def test_one_stratum_is_the_uniform_subset(run):
    """One stratum drawn k of n is the one-stage audits' law: the same count bound as the profile and the Lean curve."""
    n, k, delta, _, at_most = E2E[run]
    law = Stratified.uniform(n, k)
    assert law.count_bound(delta) == at_most == _largest_admitted(n, k, delta)
    assert law.miss(at_most) == Fraction(comb(n - at_most, k), comb(n, k)) and law.inclusion("unit") == Fraction(k, n)


def test_a4_uniform_leaves_the_norms_undrawn():
    """A4 P6's law: at most 91,051 wrong units at 2^-20 (the audit's own bound), but a template of 287 units that is
    wrong in every unit escapes with probability 0.958."""
    law = Stratified.uniform(sum(A4_SIZES.values()), 1024)
    assert law.count_bound(2.0 ** -20) == 91_051
    assert 0.957 < float(law.escape({"unit": 287})) < 0.958


def test_a4_one_unit_floor_per_template():
    """Proportional allocation with one unit per template, at A4's proving cost: every template is exercised, a
    template wrong in every unit is always caught, and the count bound moves by 3%."""
    law = a4_law({"rms_triton": 1, "gemm2048": 905, "rope": 2, "rms_fused": 1, "silu_mul": 1, "gemm8192": 86})
    assert law.k == 996 and law.count_bound(2.0 ** -20) == 93_892
    worst = law.worst_counts(93_892)
    assert worst["rms_triton"] == worst["rms_fused"] == worst["silu_mul"] == worst["rope"] == 0
    for name in ("rms_triton", "rms_fused", "silu_mul"):
        assert law.escape({name: 287}) == 0 and law.stratum_bound(name, 2.0 ** -20) == 286
        assert law.escape({name: 1}) == Fraction(286, 287)


def test_a4_proportional_to_output_bits():
    """Inclusion proportional to output bits at the same proving cost: the count bound rises to 11.4% of the units."""
    law = a4_law({"rms_triton": 10, "gemm2048": 104, "rope": 12, "rms_fused": 20, "silu_mul": 39, "gemm8192": 10})
    assert law.count_bound(2.0 ** -20) == 773_981


def test_the_count_bound_is_the_unit_harm_bound_rounded_down():
    law = a4_law({"rms_triton": 1, "gemm2048": 905, "rope": 2, "rms_fused": 1, "silu_mul": 1, "gemm8192": 86})
    for delta in (2.0 ** -10, 2.0 ** -20, 2.0 ** -40):
        K = law.count_bound(delta)
        assert K <= law.harm_bound({s.name: 1 for s in law.strata}, delta) < K + 1 + 1e-6 * K


@pytest.mark.parametrize("strata", [(), (Stratum("a", 3, 1), Stratum("a", 2, 1)), ("a",)])
def test_ill_formed_stratified_laws_are_refused(strata):
    with pytest.raises(ValueError):
        Stratified(strata)


@pytest.mark.parametrize("kw", [dict(size=-1), dict(k=5), dict(k=-1), dict(name="")])
def test_ill_formed_strata_are_refused(kw):
    with pytest.raises(ValueError):
        Stratum(**{**dict(name="a", size=4, k=1), **kw})


def test_stratified_queries_name_known_strata():
    law = SMALL[0]
    with pytest.raises(KeyError):
        law.escape({"z": 1})
    with pytest.raises(ValueError):
        law.escape({"a": 5})
    with pytest.raises(ValueError):
        law.miss(law.population + 1)
    with pytest.raises(ValueError):
        law.count_bound(1.0)
    assert law.worst_counts(law.population) is None and law.miss(law.population) == 0


def test_a_profile_drawn_by_strata_reads_its_bounds_off_the_law():
    """A Stratified draw of the checked level: the profile's bound is the law's exact count bound, its stratum bounds each
    stratum's own, and its weighted bound the law's.  A one-level Subset is one stratum, and a Bernoulli level has none."""
    for law in SMALL:
        a = IntegrityProfile((Level("unit", "q"),), (law.population,), {"unit": law}, 0.1, drawn={"unit": law.k})
        assert a.strata() is law and a.bound() == law.count_bound(0.1)
        assert a.stratum_bounds() == {s.name: law.stratum_bound(s.name, 0.1) for s in law.strata}
        assert a.harm_bound(2.0) == law.harm_bound({s.name: 2.0 for s in law.strata}, 0.1)
        assert a.to_json()["levels"][0]["draw"] == law.to_json()
        with pytest.raises(ValueError):
            a.worst_case(lambda m: m)
    grouped = IntegrityProfile(("group", "unit"), (2, 5), {"unit": SMALL[1]}, 0.1)
    assert grouped.bound() == SMALL[1].count_bound(0.1)
    for n, k, delta in ((1024, 16, 0.01), (6, 2, 0.25), (344, 128, 2.0 ** -20)):
        one = one_level(n, k, delta)
        assert one.strata() == Stratified.uniform(n, k) and one.bound() == Stratified.uniform(n, k).count_bound(delta)
        assert one.bound() == int(one.worst_case(lambda m: m)) and one.stratum_bounds() == {"unit": one.bound()}
    assert TOY.strata() is None and TOY.stratum_bounds() == {} and TOY.bound() == incorrect_units(TOY)
    assert TOY.harm_bound(4) == TOY.worst_case(lambda m: 4 * m)
    with pytest.raises(ValueError):
        TOY.harm_bound({"check": 4.0})


def test_a_lone_bernoulli_draw_of_the_checked_level_bounds_exactly():
    """One Bernoulli(p) draw of the checked level: M wrong members escape with probability exactly (1 - p)^M, so the
    profile's bound is the largest M <= members with (1 - p)^M >= delta (`Bernoulli.count_bound`, by search), each of the
    search's shortcuts the scan's answer, and never above the linear relaxation it replaces."""
    def scan(n: int, p: Fraction, delta: float) -> int:
        return max(m for m in range(n + 1) if (1 - p) ** m >= Fraction(delta))
    for n in (0, 1, 7, 100, 1000):
        for p in (Fraction(0), Fraction(1, 1000), Fraction(1, 100), Fraction(1, 3), Fraction(1, 2), Fraction(999, 1000), Fraction(1)):
            for delta in (0.5, 0.1, 2.0 ** -10, 2.0 ** -40):
                law = Bernoulli(p)
                assert law.count_bound(n, delta) == scan(n, p, delta), (n, p, delta)
                for levels, sizes in ((("unit",), (n,)), (("group", "unit"), (n, 1))):
                    a = IntegrityProfile(levels, sizes, {"unit": law}, delta)
                    assert a.bound() == law.count_bound(n, delta) and a.bound() <= a.worst_case(lambda m: m)
    assert Bernoulli(Fraction(1, 2)).count_bound(1000, 2.0 ** -10) == 10
    for delta in (0.0, 1.0):
        with pytest.raises(ValueError):
            Bernoulli(Fraction(1, 2)).count_bound(10, delta)


@pytest.mark.parametrize("levels, sizes, draws", [
    (("group", "unit"), (2, 5), {"group": Bernoulli(Fraction(1, 2)), "unit": SMALL[1]}),   # not the only draw
    (("unit", "sub"), (10, 1), {"unit": SMALL[1], "sub": Subset(1)}),                     # not the checked level
    (("unit",), (9,), {"unit": SMALL[1]}),                                                # strata of 10 units, level of 9
])
def test_ill_formed_stratified_profiles_are_refused(levels, sizes, draws):
    with pytest.raises(ValueError):
        IntegrityProfile(levels, sizes, draws, 0.01)


def test_stratified_json_is_the_profile_law():
    assert SMALL[0].to_json() == {"law": "stratified", "strata": [{"name": "a", "size": 4, "k": 2},
                                                                  {"name": "b", "size": 3, "k": 1},
                                                                  {"name": "c", "size": 3, "k": 0}]}


#: `work:3` over a (4 units, work 3) and b (6 units, work 1): W = 18, so k_a = ⌈3·3·4/18⌉ = 2 and k_b = ⌈3·1·6/18⌉ = 1.
WORK = Work(3, (WorkStratum("a", 4, 2, 3, 1), WorkStratum("b", 6, 1, 1, 1)))


def test_a_work_design_is_the_stratified_one_at_its_k():
    """A `Work` draw of the checked level reads its bounds off the stratified design at its k_s, and writes the profile's
    `work` case (Lean `Core.Draw.work`), its existing cases unchanged."""
    a = IntegrityProfile((Level("unit", "q"),), (10,), {"unit": WORK}, 0.1, drawn={"unit": 3})
    law = Stratified((Stratum("a", 4, 2), Stratum("b", 6, 1)))
    assert WORK.stratified == law and a.strata() == law and a.bound() == law.count_bound(0.1)
    assert a.stratum_bounds() == {s.name: law.stratum_bound(s.name, 0.1) for s in law.strata}
    assert a.to_json()["levels"][0]["draw"] == {"law": "work", "k": 3, "strata": [
        {"name": "a", "size": 4, "k": 2, "work": 3, "floor": 1}, {"name": "b", "size": 6, "k": 1, "work": 1, "floor": 1}]}
    assert [work_k(3, w, n, 18, 1) for w, n in ((3, 4), (1, 6))] == [2, 1]
    with pytest.raises(ValueError):
        IntegrityProfile((Level("unit", "q"),), (9,), {"unit": WORK}, 0.1)


@pytest.mark.parametrize("k, strata", [
    (3, ()),                                                                             # no strata
    (0, (WorkStratum("a", 4, 2, 3, 1), WorkStratum("b", 6, 1, 1, 1))),                   # k < 1
    (11, (WorkStratum("a", 4, 4, 3, 1), WorkStratum("b", 6, 6, 1, 1))),                  # k > population
    (3, (WorkStratum("a", 4, 1, 0, 1),)),                                                # no work
    (3, (WorkStratum("a", 4, 2, 3, 1), WorkStratum("a", 6, 1, 1, 1))),                   # a name twice
    (3, (WorkStratum("a", 4, 2, 3, 0), WorkStratum("b", 6, 1, 1, 1))),                   # a floor of 0
    (3, (WorkStratum("a", 4, 1, 3, 1), WorkStratum("b", 6, 1, 1, 1))),                   # a k not the rule's
    (3, (Stratum("a", 4, 2),)),                                                          # not a work stratum
])
def test_ill_formed_work_designs_are_refused(k, strata):
    with pytest.raises(ValueError):
        Work(k, strata)
