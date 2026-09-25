"""IntegrityProfile: what was verified (covered families, the two-stage law, delta), and each consumer's reader over it."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import exp, log

import pytest

from verity.resource_accounting import (IntegrityProfile, forgeable_tiles, incorrect_verification_units, touched_replay_units,
                                        worst_case)

TOY = IntegrityProfile(n_r=512, n_v=4, p=Fraction(1, 2), k=2, delta=0.01)


def test_the_prototype_toy_numbers():
    """The protocol prototype's composed toy: 26.6 incorrect VUs for PoUW, 16.0 touched RUs for a KV claim."""
    assert round(incorrect_verification_units(TOY), 1) == 26.6
    assert round(touched_replay_units(TOY), 1) == 16.0
    assert forgeable_tiles(TOY, 4) == pytest.approx(4 * incorrect_verification_units(TOY))


def test_consumers_are_maximised_by_different_strategies():
    a = IntegrityProfile(n_r=10_000, n_v=4, p=Fraction(1, 4), k=2, delta=0.01)
    ratio = {m: m / a.rho(m) for m in range(1, 5)}
    touched = {m: 1 / a.rho(m) for m in range(1, 5)}
    assert max(ratio, key=ratio.get) == 4      # free compute concentrates into fully incorrect replay units
    assert max(touched, key=touched.get) == 1  # KV laundering spreads one incorrect verification unit per replay unit
    assert incorrect_verification_units(a) == pytest.approx(ratio[4] * log(100))
    assert touched_replay_units(a) == pytest.approx(touched[1] * log(100))


def test_accept_is_exact():
    assert TOY.accept(0) == 1 and TOY.accept(1) == Fraction(3, 4) and TOY.accept(4) == Fraction(1, 2)
    assert IntegrityProfile(n_r=1, n_v=4, p=Fraction(1), k=4, delta=0.5).accept(1) == 0


def test_full_check_admits_nothing_and_no_check_admits_everything():
    assert incorrect_verification_units(IntegrityProfile(n_r=100, n_v=4, p=Fraction(1), k=4, delta=0.01)) == 0
    assert incorrect_verification_units(IntegrityProfile(n_r=100, n_v=4, p=Fraction(0), k=4, delta=0.01)) == 400
    assert touched_replay_units(IntegrityProfile(n_r=100, n_v=4, p=Fraction(1, 2), k=0, delta=0.01)) == 100


def test_the_bound_uses_n_r_when_it_binds():
    few = IntegrityProfile(n_r=3, n_v=4, p=Fraction(1, 100), k=1, delta=0.01)
    assert incorrect_verification_units(few) == pytest.approx(12)


@pytest.mark.parametrize("n_r, n_v, p, k, delta", [(3, 3, Fraction(1, 2), 1, 0.3), (4, 2, Fraction(1, 3), 1, 0.2),
                                                   (3, 4, Fraction(2, 3), 2, 0.4), (5, 1, Fraction(1, 4), 1, 0.1)])
@pytest.mark.parametrize("utility", [lambda m: m, lambda m: 1, lambda m: m * m, lambda m: 3 * m])
def test_the_bound_covers_every_admitted_integer_strategy(n_r, n_v, p, k, delta, utility):
    """Brute force: every strategy accepted with probability above delta stays within the bound."""
    a = IntegrityProfile(n_r=n_r, n_v=n_v, p=p, k=k, delta=delta)
    bound = worst_case(a, utility)
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
    a = IntegrityProfile(n_r=10_000, n_v=1, p=rate, k=1, delta=0.01)
    assert a.accept(1) == 1 - rate
    assert incorrect_verification_units(a) == pytest.approx(log(100) / -log(0.9))
    assert exp(-a.rho(1) * incorrect_verification_units(a)) == pytest.approx(0.01)


@pytest.mark.parametrize("kw", [dict(n_v=0), dict(k=5), dict(p=Fraction(3, 2)), dict(delta=0.0), dict(delta=1.0), dict(n_r=-1)])
def test_ill_formed_profiles_are_refused(kw):
    with pytest.raises(ValueError):
        IntegrityProfile(**{**dict(n_r=1, n_v=4, p=Fraction(1, 2), k=2, delta=0.01), **kw})


def test_forgeable_tiles_refuses_a_zero_partition_and_json_carries_the_law():
    with pytest.raises(ValueError):
        forgeable_tiles(TOY, 0)
    assert TOY.to_json() == {"covered": [], "n_r": 512, "n_v": 4, "p": "1/2", "k": 2, "delta": 0.01}


def test_covered_families_travel_with_the_law():
    p = IntegrityProfile(n_r=8, n_v=4, p=Fraction(1, 2), k=2, delta=0.01, covered=["Input", "TcDot16_v1"])
    assert p.covered == frozenset({"Input", "TcDot16_v1"}) and p.to_json()["covered"] == ["Input", "TcDot16_v1"]
