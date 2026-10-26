"""theory/d5_v2.py (D5 v2, CMP-125): the period's dual prices sum to 1 over the capacities and price the period at its
length; the tensor pipe has a price only past X0; on v3's operands the H200's job-only prices sit on the lanes (c = 1,
the matmul unpriced: A26), and the RTX PRO 6000's public best saves exactly one digit of the peels."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import d5_v2 as D  # noqa: E402
import public_best as PB  # noqa: E402

K, N = 4096, 6144
# Llama-3.1-8B's qkv shape at CMP-118's caps, with nofold's compressions a MAC there (bbe690dfe5's v2_round)
ROUNDS = {"H200": (PB.H200, 2, 640, 2.6335869163152276e-05), "RTX PRO 6000": (PB.SM120, 3, 256, 3.438581912658015e-05)}


def _job(name):
    card, t, Lp, comps = ROUNDS[name]
    _, bit, depth = PB.BIT_COST[card.family][0]
    return card, Lp, PB.tile_job(card, t, Lp, K, N, bit, depth, PB.V2_V, digits=(2, 2), comps=comps, model_sha=True,
                                 schemes=None, lookups=False)


def _served(card, j):
    x0 = D.first_tensor_bound(card, j, N, lookups=False)
    return D.at_prices(card, j, N, D.period(card, j, N, x0 + 1e-4 * j["route"].c, lookups=False).mu, False)


@pytest.mark.parametrize("name", list(ROUNDS))
def test_duals_sum_to_one_and_price_the_period_at_its_length(name):
    card, _, j = _job(name)
    for X in (0.0, 0.5 * j["route"].c, 10 * j["route"].c):
        p = D.period(card, j, N, X, lookups=False)
        assert sum(p.mu.get(nm, 0.0) * cap for nm, _, cap in PB._rows(card)) == pytest.approx(1.0)
        assert PB.mu_cost(card, p.tot, p.mu) == pytest.approx(p.c)


@pytest.mark.parametrize("name", list(ROUNDS))
def test_the_tensor_pipe_has_a_price_only_past_x0(name):
    card, _, j = _job(name)
    x0 = D.first_tensor_bound(card, j, N, lookups=False)
    assert x0 > 0
    assert D.tensor_price(card, D.period(card, j, N, 0.9 * x0, lookups=False).mu) == 0.0
    assert D.tensor_price(card, D.period(card, j, N, x0 + 1e-3 * j["route"].c, lookups=False).mu) > 0.0


@pytest.mark.parametrize("name", list(ROUNDS))
def test_served_prices_credit_less_than_the_job_alone_at_8b_qkv(name):
    card, _, j = _job(name)
    job, served = D.at_prices(card, j, N, j["route"].mu, False), _served(card, j)
    for a in (job, served):
        assert a["pub"] <= a["opt"] * (1 + 1e-9) and a["opt"] <= a["H"] * (1 + 1e-9)
    assert served["c"] < job["c"]


def test_h200_job_only_prices_sit_on_the_lanes():
    """A26: v3's H200 rounds are lane-saturated, so at the job's own prices no route pays and the matmul is free."""
    card, _, j = _job("H200")
    a = D.at_prices(card, j, N, j["route"].mu, False)
    assert a["c"] == pytest.approx(1.0) and a["c_mm"] is None
    assert "tensor" not in j["route"].binding and "alu" in j["route"].binding
    assert D.weighted([dict(w=1.0, job=a, gamma=1.0, any=0.0)], "job")["c_mm"] is None


def test_rtx_public_best_saves_one_digit_of_the_peels():
    """The zero-high-digit peel: the matmul's dense product and 2 + 2 digits against 2 + 1, at both prices."""
    card, Lp, j = _job("RTX PRO 6000")
    for a in (D.at_prices(card, j, N, j["route"].mu, False), _served(card, j)):
        assert a["c_mm"] == pytest.approx((1 + 3 * Lp / K) / (1 + 4 * Lp / K))
