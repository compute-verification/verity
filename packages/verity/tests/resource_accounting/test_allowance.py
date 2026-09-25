from fractions import Fraction
from math import log

from verity.resource_accounting import IntegrityAllowance


def test_consumers_are_maximised_by_different_strategies():
    a = IntegrityAllowance(n_r=10_000, n_v=4, p=Fraction(1, 4), k=2, delta=0.01)
    ratio = {m: m / a.rho(m) for m in range(1, 5)}
    assert max(ratio, key=ratio.get) == 4  # free compute concentrates into fully incorrect RUs
    touched = {m: 1 / a.rho(m) for m in range(1, 5)}
    assert max(touched, key=touched.get) == 1  # KV laundering spreads one incorrect VU per RU
    assert a.worst_case(lambda m: m) == ratio[4] * log(100)
    assert a.worst_case(lambda m: 1) == touched[1] * log(100)


def test_full_check_admits_nothing_and_no_check_admits_everything():
    assert IntegrityAllowance(n_r=100, n_v=4, p=Fraction(1), k=4, delta=0.01).worst_case(lambda m: m) == 0
    assert IntegrityAllowance(n_r=100, n_v=4, p=Fraction(0), k=4, delta=0.01).worst_case(lambda m: m) == 400
