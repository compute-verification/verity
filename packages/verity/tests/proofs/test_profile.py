"""IntegrityProfile: what verification established, and the worst case a consumer's utility reads off it.

The two consumer utilities below stand in for PoUW's incorrect-unit bound and the KV touched-unit bound, which live
with their consumers, not in core.  `two_level` is sampled proofs' shape (n_r units drawn with p, then k of the n_v
inside each), and `_before_levels` is the profile's arithmetic as it stood in those four numbers (every pair of m tried), kept as the
oracle the level form must reproduce: `accept` exactly, and `worst_case` up to float rounding (the envelope and the
pairs round differently, by about 1e-15)."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import comb, exp, inf, log

import pytest

from verity.ir import Program, bind
from verity.ir import query_ast as G
from verity.ir.partition import Level, members_per_parent, validate_partition
from verity.ml import gemm
from verity.proofs.profile import Bernoulli, IntegrityProfile, Subset


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
            assert a.worst_case(utility) == pytest.approx(worst_case(utility), rel=1e-12, abs=1e-12)


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
    per = members_per_parent(program, coarse, fine)
    assert per == (2, 2, 2, 2)
    a = IntegrityProfile((Level("coordinate", G.query_id(coarse_q)), Level("call", G.query_id(fine_q))),
                         (coarse.count(), per[0]), {"coordinate": Bernoulli(Fraction(1, 2)), "call": Subset(1)}, 0.01)
    assert a.count("call") == fine.count() == 8 and a.accept(2) == Fraction(1, 2)


def one_level(n, k, delta) -> IntegrityProfile:
    return IntegrityProfile((Level("unit"),), (n,), {"unit": Subset(k)}, delta)


#: The one-stage audits' wrong-unit bounds (lane one-stage-e2e): (n units, k drawn, delta, worst_case(m), at most).
E2E = {"A0": (1024, 16, 0.01, 254.22632080026347, 254),
       "A1": (1024, 256, 2.0 ** -20, 46.92437954253062, 46),
       "A3": (344, 128, 2.0 ** -20, 28.25559880146837, 28),
       "A2": (183_680, 256, 2.0 ** -20, 9675.419457199421, 9675)}


@pytest.mark.parametrize("run", sorted(E2E))
def test_the_one_stage_audit_bounds(run):
    """A2 is serving's real population (183,680 units); the quadratic, float `worst_case` crashed there."""
    n, k, delta, value, at_most = E2E[run]
    bound = one_level(n, k, delta).worst_case(lambda m: m)
    assert bound == pytest.approx(value, rel=1e-12)
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


def test_worst_case_is_the_convex_threshold_for_one_subset_level():
    """With one level drawn Subset(k), rho is convex from rho(0) = 0, so the relaxation's optimum interpolates between
    the last admitted count and the next (one-stage-e2e's direct bound, #116 6ccb9b7b, now served by core)."""
    for n, k, delta, _, _ in E2E.values():
        a, budget = one_level(n, k, delta), log(1 / delta)
        lo = _largest_admitted(n, k, delta)
        ra, rb = a.rho(lo), a.rho(lo + 1)
        assert a.worst_case(lambda m: m) == pytest.approx(lo + (budget - ra) / (rb - ra), rel=1e-12)



def test_the_float_pairwise_version_crashed_where_the_profile_now_serves():
    """The profile's arithmetic before this fix: -log(float(accept)) underflows once accept < 1e-308, as it does at
    n = 4,000, k = 256 (accept(n - k) = 1/C(4000, 256), about e^-960)."""
    _, before = _before_levels(1, 4000, Fraction(1), 256, 2.0 ** -20)
    with pytest.raises(ValueError):
        before(lambda m: m)
    bound = two_level(1, 4000, Fraction(1), 256, 2.0 ** -20).worst_case(lambda m: m)
    assert int(bound) == _largest_admitted(4000, 256, 2.0 ** -20)
