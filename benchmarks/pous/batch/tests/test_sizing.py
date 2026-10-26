"""`sizing`: the batch deadline and the recompute capacity it allows, the Chernoff bound against the exact binomial it
bounds, f_inf = R·s/X as the floor no batch gets under, and the obligation's fraction of the demo's `C*`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sizing as Z


def test_the_deadline_and_capacity():
    t = Z.deadline_s(1024, 0.1, 20e6, 0.005)
    assert t == pytest.approx(0.1 + 1024 * (2056 + 8) / 20e6 + 0.005)
    assert Z.deadline_s(1024, 0.1, 20e6, 0.005, i=0) == pytest.approx(0.1 + 1024 * 2056 / 20e6 + 0.005)
    assert Z.capacity(2000, t) == math.floor(2000 * t)


@pytest.mark.parametrize(("b", "n"), [(10, 20), (30, 100), (64, 4096), (200, 1000)])
def test_the_repeats_bound_is_at_least_the_exact_probability(b, n):
    p = Z.distinct_pmf(b, n)
    assert sum(p) == pytest.approx(1)
    for m in range(1, b):
        exact = sum(p[: b - m + 1])
        assert exact <= math.exp(Z.log_repeats_at_least(b, n, m)) * (1 + 1e-9)


def test_drawing_with_replacement_only_helps_the_prover_and_vanishes_for_a_large_store():
    for b, f, cap in [(4096, 0.1, 300), (65536, 0.0397, 2000), (1 << 19, 0.003966, 1936)]:
        alone = Z.log_pass(b, f, cap)
        assert alone < 0
        assert alone <= Z.log_pass_drawn(b, f, cap, Z.STORE_BLOCKS) <= 0 and alone <= Z.log_pass_drawn(b, f, cap, 1 << 22)
        assert Z.log_pass_drawn(b, f, cap, 1 << 100) <= Z.log_pass(b - 1, f, cap) + 1e-6


def test_the_drawn_bound_is_at_least_the_exact_probability():
    b, n, missing, cap = 40, 60, 12, 3
    p = Z.distinct_pmf(b, n)
    exact = sum(p[d] * sum(math.comb(missing, j) * math.comb(n - missing, d - j) for j in range(min(cap, d) + 1))
                / math.comb(n, d) for d in range(b + 1))
    assert exact <= math.exp(Z.log_pass_drawn(b, missing / n, cap, n))


@pytest.mark.parametrize(("b", "f", "cap"), [(64, 0.1, 2), (256, 0.1, 12), (512, 1 / 19, 10), (1000, 0.3, 250)])
def test_the_chernoff_bound_is_at_least_the_exact_probability(b, f, cap):
    exact = Z.binomial_cdf(b, f, cap)
    assert exact <= math.exp(Z.log_pass(b, f, cap)) and exact < 1


def test_a_capacity_at_the_mean_or_past_it_passes_for_sure():
    assert Z.log_pass(100, 0.1, 10) == 0 and Z.rounds(0.0, 0.01) == math.inf


def test_no_batch_gets_under_f_inf():
    rate, x = 2000, 64e6
    fi = Z.f_inf(rate, x)
    assert fi == pytest.approx(2000 * (2056 + 8) / 64e6)
    assert Z.plan(fi * 0.99, rate, 0.1, x, 0.005, 0.01)["b"] is None
    ok = Z.plan(fi * 1.5, rate, 0.1, x, 0.005, 0.01)
    assert ok["b"] and ok["cap"] < ok["missing_mean"] and ok["log10_pass"] <= -2


def test_the_obligation_fraction_of_the_demo_store():
    f = Z.f_obligation()
    assert f == pytest.approx(1 - 0.93 * 139.64 * 2 ** 30 / 139_996_954_624) and 0.0039 < f < 0.0040
