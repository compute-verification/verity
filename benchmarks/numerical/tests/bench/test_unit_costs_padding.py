"""F6's padding table: the hypergeometric pmf and quantiles, the menus, and one tiny audit priced exactly."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb

import pytest

from verity_numerical.bench import unit_costs as U
from verity_numerical.bench import unit_costs_padding as P


def test_hyper_is_the_exact_pmf():
    N, n, K = 30, 7, 12
    k0, p = P.hyper(N, n, K)
    want = {k: comb(n, k) * comb(N - n, K - k) / comb(N, K) for k in range(n + 1)}
    assert {k0 + j: x for j, x in enumerate(p)} == pytest.approx({k: v for k, v in want.items() if v > 1e-12}, rel=1e-12)


@pytest.mark.parametrize("N, n, K", [(30, 7, 12), (2_929_277_932, 2_000_000_000, 27_713), (656_132, 4_150, 27_713)])
def test_upper_q_is_draw_counts(N, n, K):
    assert P.upper_q(N, n, K) == U.draw_count(N, n, K)[1]


def test_menu_dp():
    us = [P.Unit(str(z), "t", w, z, z) for z, w in ((1, 100), (2, 1), (10, 1), (11, 100))]
    assert P.menu_dp(us, 2) == [1, 11]
    assert P.menu_dp(us, 3) == [1, 2, 11]
    assert P.menu_dp(us, 6) == [1, 2, 10, 11]


def test_tail_split():
    us = [P.Unit("g", "gemm-coordinate", 990, 10, 10), P.Unit("r", "rope-head", 9, 100, 100),
          P.Unit("a", "attention-head", 1, 1000, 900), P.Unit("t", "token-select-greedy", 1, 5, 5)]
    body, tail, thr = P.tail_split(us, factor=16)
    assert thr == 160 and [u.label for u in body] == ["g", "r"] and [u.label for u in tail] == ["a", "t"]


def test_a_tiny_audit_exactly():
    """N = 6 units, 4 of 3 nonzeros and 2 of 10; K = 3 drawn; menu pow2 = {4, 16}; B = 2. Checked against all 20 draws."""
    units = [P.Unit("a", "t", 4, 3, 3), P.Unit("b", "t", 2, 10, 10)]
    pop = [3] * 4 + [10] * 2
    draws = list(combinations(range(6), 3))
    actual = Fraction(sum(sum(pop[i] for i in d) for d in draws), len(draws))
    padded = Fraction(sum(2 * 4 * -(-sum(pop[i] == 3 for i in d) // 2) + 2 * 16 * -(-sum(pop[i] == 10 for i in d) // 2)
                          for d in draws), len(draws))
    assert (actual, padded) == (16, Fraction(176, 5))
    e = P.evaluate(6, 3, units, P.menu_pow2(units), 2)
    assert e["buckets"] == 2 and e["actual_mean"] == pytest.approx(16)
    assert (e["zpad"], e["pad_mean"], e["round_extra"]) == pytest.approx((1.5, 2.2, 0.7))
    # public: 3 of a (2 sessions of 4), 2 of b (1 session of 16); pad_q's floor: at least 1 of a drawn
    assert (e["sessions_public"], e["public"], e["hide_extra"], e["pad_q"]) == pytest.approx((3, 3.0, 0.8, 16.0))
    assert e["sessions_mean"] == pytest.approx(1.2 + 0.8)


def test_padding_tables_a_workload():
    row = lambda label, template, n, nnz, rng=None: {  # noqa: E731
        "workload": "w", "unit_type": label, "template": template, "source": "registered", "subcircuits_per_vu": 1,
        "subcircuit": {"units": n, "population": 100_000, "per_unit": {"nnz": nnz}, **({"nnz_range": rng} if rng else {})}}
    inv = {"K": 1000, "rows": [row("g1", "gemm-coordinate", 60_000, 30), row("g2", "gemm-coordinate", 39_000, 70),
                               row("a", "attention-head", 999, 200, {"min": 150, "max": 250}),
                               {**row("t", "token-select-greedy", 1, 10 ** 6), "nnz_step_model": {"nnz": 500}}]}
    pad = P.padding(inv)
    w = pad["workloads"][0]
    assert [u["unit_type"] for u in w["tail"]["types"]] == ["t"]
    assert {m["menu"]: m["Z"] for m in w["menus"]}["natural"] == [30, 70, 250]
    assert w["tail"]["eval_step_model"]["actual_mean"] == pytest.approx(1000 * 1 / 100_000 * 500)
    assert "| w | dp2 |" in P.render(pad)
