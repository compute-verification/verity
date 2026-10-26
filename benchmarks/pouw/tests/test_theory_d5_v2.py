"""theory/d5_v2.py (D5 v2, CMP-125): the period's dual prices sum to 1 over the capacities and price the period at its
length; the served prices are the honest period's, and its tensor pipe has a price only past X0; on v3's operands the
H200's job-only prices sit on the lanes (c = 1, the matmul unpriced: A26), and the RTX PRO 6000's public best saves
exactly one digit of the peels alone and takes the tile route when served."""
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


def _mu(card, j, X):
    return D.period(card, j, N, X, honest=True, lookups=False).mu


def _job_only(card, j):
    return D.at_prices(card, j, N, _mu(card, j, 0.0), False)


def _served(card, j):
    x0 = D.first_tensor_bound(card, j, N, lookups=False)
    return D.at_prices(card, j, N, _mu(card, j, x0 + 1e-4 * j["route"].c), False)


@pytest.mark.parametrize("name", list(ROUNDS))
def test_duals_sum_to_one_and_price_the_period_at_its_length(name):
    card, _, j = _job(name)
    for X in (0.0, 0.5 * j["route"].c, 10 * j["route"].c):
        for honest in (False, True):
            p = D.period(card, j, N, X, honest=honest, lookups=False)
            assert sum(p.mu.get(nm, 0.0) * cap for nm, _, cap in PB._rows(card)) == pytest.approx(1.0)
            assert PB.mu_cost(card, p.tot, p.mu) == pytest.approx(p.c)


@pytest.mark.parametrize("name", list(ROUNDS))
def test_the_tensor_pipe_has_a_price_only_past_x0(name):
    card, _, j = _job(name)
    x0 = D.first_tensor_bound(card, j, N, lookups=False)
    assert x0 > 0
    assert D.tensor_price(card, _mu(card, j, 0.9 * x0)) == 0.0
    assert D.tensor_price(card, _mu(card, j, x0 + 1e-3 * j["route"].c)) > 0.0


def test_served_prices_are_the_honest_periods():
    """Credit can't depend on the prover's methods: the served duals are the honest period's (mu°). On the H200 the
    hull's O-route frees the joint facet, so the hull's own period still leaves the tensor pipe free at twice the X
    where the honest period prices it."""
    card, _, j = _job("H200")
    X = 2 * D.first_tensor_bound(card, j, N, lookups=False)
    assert D.tensor_price(card, _mu(card, j, X)) > 0.0
    assert D.tensor_price(card, D.period(card, j, N, X, lookups=False).mu) == 0.0


@pytest.mark.parametrize("name", list(ROUNDS))
def test_the_makespan_increment_spans_the_job_and_the_tensor_only_limit(name):
    """c_inc is the job's makespan ratio with no other work, never above 1, and at 100 round-times of other tensor
    work (idle lanes past any slack the round could use) it equals c at the period's duals."""
    card, _, j = _job(name)
    T = j["route"].c
    assert D.increment(card, j, N, 0.0, False)["c_inc"] == pytest.approx(j["c"])
    for X in (0.5 * T, 3 * T):
        assert D.increment(card, j, N, X, False)["c_inc"] <= 1 + 1e-9
    big = D.increment(card, j, N, 100 * T, False)["c_inc"]
    assert big == pytest.approx(D.at_prices(card, j, N, _mu(card, j, 100 * T), False)["c"])


@pytest.mark.parametrize("name", list(ROUNDS))
def test_served_prices_credit_less_than_the_job_alone_at_8b_qkv(name):
    card, _, j = _job(name)
    job, served = _job_only(card, j), _served(card, j)
    for a in (job, served):
        assert a["pub"] <= a["opt"] * (1 + 1e-9) and a["opt"] <= a["H"] * (1 + 1e-9)
    assert served["c"] < job["c"]


def test_h200_job_only_prices_sit_on_the_lanes():
    """A26: v3's H200 rounds are lane-saturated, so at the job's own prices no route pays and the matmul is free."""
    card, _, j = _job("H200")
    a = _job_only(card, j)
    assert a["c"] == pytest.approx(1.0) and a["c_mm"] is None
    assert D.priced(D.period(card, j, N, 0.0, honest=True, lookups=False)) == ["alu"]
    row = dict(w=1.0, job=dict(a, pub_inc=j["route"].c, H_inc=j["honest"].c), gamma=1.0, any=0.0)
    assert D.weighted([row], "job")["c_mm"] is None


def test_rtx_public_best_saves_one_digit_of_the_peels_alone_and_tiles_when_served(monkeypatch):
    """Job-only, the zero-high-digit peel: the matmul's dense product and 2 + 2 digits against 2 + 1. Served, the joint
    facet's price makes the tile route (Z U^T on the ALU) the public best, with b c^T on the lanes (route iii); with
    b c^T on FP4 Strassen (v2's menu) it saves less."""
    card, Lp, j = _job("RTX PRO 6000")
    assert _job_only(card, j)["c_mm"] == pytest.approx((1 + 3 * Lp / K) / (1 + 4 * Lp / K))
    served = _served(card, j)
    assert sum(v for k, v in served["mix"].items() if k.startswith("tiles, b c^T by emulated b1")) == pytest.approx(1.0)
    monkeypatch.setattr(PB, "B1_EMULATED", ())
    card, Lp, j = _job("RTX PRO 6000")
    v2 = _served(card, j)
    assert v2["mix"].get("tiles", 0.0) == pytest.approx(1.0)
    assert served["c_mm"] < v2["c_mm"] < (1 + 3 * Lp / K) / (1 + 4 * Lp / K)


def test_a21s_fp4_route_is_in_p_hyb3s_rtx_b_ct_row():
    """A32: b c^T at 3 × 3 bits on the RTX PRO 6000 runs as one shifted FP4 product at half an INT8 product."""
    rows = D.phyb3_rows(quick=True)
    rtx = rows["RTX PRO 6000", "b c^T, 3 x 3 bits"]
    assert rtx["c"] <= 0.5 + 1e-6 and any(k.startswith("FP4") for k in rtx["mix"])
    assert not any(k.startswith("FP4") for k in rows["H200", "b c^T, 3 x 3 bits"]["mix"])
