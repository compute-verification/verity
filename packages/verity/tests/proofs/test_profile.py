"""IntegrityProfile: what verification established, and the worst case a consumer's utility reads off it.

The two consumer utilities below stand in for PoUW's incorrect-unit bound and the KV touched-unit bound, which live
with their consumers, not in core.  `two_level` is sampled proofs' shape (n_r units drawn with p, then k of the n_v
inside each), and `_before_levels` is the profile's arithmetic as it stood in those four numbers, kept as the oracle
the level form must reproduce exactly."""

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
            assert a.worst_case(utility) == worst_case(utility)


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
