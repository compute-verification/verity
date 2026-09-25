from fractions import Fraction
from math import log

from verity.proofs import IntegrityProfile


def profile(**kw):
    base = dict(programs=("d",), prescribed=frozenset({"weights"}), linked=True, n_r=10_000, n_v=4, p=Fraction(1, 4),
                k=2, delta=0.01, replayed=(0, 0))
    return IntegrityProfile(**{**base, **kw})


def test_consumers_are_maximised_by_different_strategies():
    a = profile()
    ratio = {m: m / a.rho(m) for m in range(1, 5)}
    assert max(ratio, key=ratio.get) == 4  # free compute concentrates into fully incorrect RUs
    touched = {m: 1 / a.rho(m) for m in range(1, 5)}
    assert max(touched, key=touched.get) == 1  # laundering spreads one incorrect VU per RU
    assert a.incorrect_units() == ratio[4] * log(100)
    assert a.touched_units() == touched[1] * log(100)


def test_full_check_admits_nothing_and_no_check_admits_everything():
    assert profile(n_r=100, p=Fraction(1), k=4).incorrect_units() == 0
    assert profile(n_r=100, p=Fraction(0), k=4).incorrect_units() == 400
