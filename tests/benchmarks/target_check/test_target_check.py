"""The target check against the definitions: `Scheme.missable` and the rate by brute force over every subset of proof
blocks, the picker against every k vector, and on the GLM-5.3 census the bracket's width and the first-order count."""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.target_check import target_check as T  # noqa: E402


def test_selftest_brute_force():
    T.selftest()


def test_rate_worst_case_is_one_proof_block():
    s = T.Sub(40, 7, 3)
    theta = math.exp(T.log_rate([s]))
    for t in range(1, 34):
        m = Fraction(math.comb(40 - t, 7), math.comb(40, 7))
        assert float(m) <= theta ** (3 * t) * (1 + 1e-12)
    assert math.isclose(theta ** 3, 1 - 7 / 40, rel_tol=1e-12)


def test_census_bracket_and_first_order():
    strata, _ = T.load(T.CENSUS)
    sizes = [s["n"] for s in strata]
    N = sum(sizes)
    amts = [s["macs"] for s in strata]
    K = T.strat_k(1, 2097152, sizes, N)
    des = [T.Sub(n, T.stratum_k(K, n, N), c) for n, c in zip(sizes, amts)]
    d = 2.0 ** -40
    lb, ub = T.missable_lb(des, d), T.missable_ub(des, d)
    assert lb <= ub <= lb + max(amts) + 1
    lt = T.log_rate(des)
    _, bproofs, _, _ = T.pick_bernoulli(sizes, amts, [1] * len(sizes), lt)
    W = sum(n * c for n, c in zip(sizes, amts))
    assert math.isclose(bproofs, W * -lt, rel_tol=1e-3)
