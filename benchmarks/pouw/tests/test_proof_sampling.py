"""The proof draw's calculator reproduces SPEC's luck and Fan numbers, and its minimax draws are covers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pouw_proof_sampling", Path(__file__).parents[1] / "proof_sampling.py")
P = importlib.util.module_from_spec(spec)
sys.modules["pouw_proof_sampling"] = P
spec.loader.exec_module(P)

CELLS = P.cells()
UNIT = P.LN_INV_DELTA * P.LUCK_FACTOR / P.ROUNDS


def _fan(c, label):
    sel = np.array([lab == label for lab in c.lev_label])
    return c.fan()[sel]


def test_luck_rates_are_theorem_s():
    assert P.c_luck(1) == 0.25
    assert P.c_luck(2) == pytest.approx(math.log(4 / 3), rel=1e-6)
    assert P.c_luck(64) == pytest.approx(math.log(4 / 3), rel=1e-6)
    assert round(100 * P.eps_luck(1), 3) == 0.093
    assert round(100 * P.eps_luck(2), 3) == 0.081
    assert round(100 * UNIT, 4) == 0.0233


def test_fans_are_specs():
    c8, c70 = P.cell(P.LLAMA_8B, "prefill", 2048), P.cell(P.LLAMA_70B, "prefill", 2048)
    assert round(_fan(c8, "o→gate/up")[0], 1) == 14.0 and round(_fan(c70, "o→gate/up")[0], 1) == 14.0
    assert round(_fan(c8, "down→lm_head")[0], 1) == 17.9 and round(_fan(c70, "down→lm_head")[0], 1) == 8.9
    two_bit = P.Design("2-bit", 0.75, 1.0)
    c2 = P.cell(P.LLAMA_8B, "prefill", 2048, two_bit)
    assert round(_fan(c2, "o→gate/up")[0], 1) == 9.3 and round(_fan(c2, "down→lm_head")[0], 1) == 11.9
    assert round(_fan(c8, "o→lm_head")[0], 1) == 62.6 and round(_fan(c70, "o→lm_head")[0], 1) == 31.3
    assert round(_fan(c8, "P·V→o").max()) == 8188


def test_work_weighted_reproduces_specs_bound():
    """(2.6) at one sample: (1 + Fan)·ln(1/δ)/S·0.403, which the knapsack reaches when that lever's supply never binds."""
    c = P.cell(P.LLAMA_8B, "decode", 2048)
    for label, eps in (("o→gate/up", 0.35), ("down→lm_head", 0.44)):
        only = np.array([lab == label for lab in c.lev_label])
        r = P.best_response(c, P.work(c, 1), 1, caps=False, levers=only)
        assert round(100 * r.eps, 2) == eps
    luck = P.best_response(c, P.work(c, 1), 1, levers=np.zeros(len(c.lev_label), bool))
    assert luck.eps == pytest.approx(P.eps_luck(1))


def test_mac_shares_match_skip():
    c2, c32 = P.cell(P.LLAMA_8B, "prefill", 2048), P.cell(P.LLAMA_8B, "prefill", 32768)
    assert round(100 * c2.share("qk"), 1) == 1.9 and round(100 * c32.share("qk")) == 19
    assert round(100 * c2.share("pv"), 1) == 1.9


@pytest.mark.parametrize("name", ["optimal", "aware"])
def test_the_minimax_draws_are_covers(name):
    """Against the uncapped cheater a sample's value is L·max ratio; the closed form's weights bound every ratio by Σ x."""
    for c in CELLS[:5]:
        for s in (1, 2, 16):
            w = P.STRATEGIES[name](c, s)
            r = P.best_response(c, w, s, caps=False)
            assert r.lam <= (c.count * w).sum() / c.macs * (1 + 1e-9)


def test_aware_is_no_worse_than_its_perturbations():
    c = P.cell(P.LLAMA_8B, "prefill", 2048)
    rng = np.random.default_rng(0)
    for s in (1, 4):
        w = P.aware(c, s)
        best = P.best_response(c, w, s, caps=False).lam
        for _ in range(20):
            v = w * np.exp(0.3 * rng.standard_normal(w.size))
            assert P.best_response(c, v, s, caps=False).lam >= best * (1 - 1e-9)


@pytest.mark.parametrize("s", [1, 2])
def test_aware_is_the_exact_minimax_against_the_capped_cheater(s):
    for c in (P.cell(P.LLAMA_8B, "prefill", 2048), P.cell(P.LLAMA_70B, "decode", 2048)):
        eps, x = P.capped_optimum(c, s)
        assert P.best_response(c, x, s).eps == pytest.approx(eps, rel=1e-6)
        assert P.evaluate(c, "aware", s)[0].eps == pytest.approx(eps, rel=1e-4)


def test_aware_and_closure_bounds_hold_in_every_cell():
    """The σ-aware draw is within 25% of ε_luck at one sample; a closure leaves ε_luck alone at any s, even with its
    producers at any coordinate of the row rather than only at noise-active ones."""
    for c in CELLS:
        assert P.evaluate(c, "aware", 1)[0].eps <= 1.25 * P.eps_luck(1)
        for s in (1, 2, 16):
            for closure in ("closure", "closure_any"):
                assert P.evaluate(c, closure, s, 1)[0].eps == pytest.approx(P.eps_luck(s))


def test_today_s_work_measure_loses_to_the_new_levers():
    c = P.cell(P.LLAMA_8B, "decode", 2048)
    r = P.evaluate(c, "work", 1)[0]
    assert max(r.by_attack, key=r.by_attack.get) == "o→lm_head" and r.eps > 0.0144


def test_the_early_token_attack_steers_every_layer_of_the_first_510_tokens():
    c = P.cell(P.LLAMA_8B, "prefill", 2048)
    r = P.evaluate(c, "work", 1)[0]
    pv = np.array([lab == "P·V→o" for lab in c.lev_label]) & (r.lev_used > 0)
    assert c.lev_pos[pv].min() == 1 and c.lev_pos[pv].max() == 510
    assert r.lev_used[pv].sum() / P.ROUNDS == pytest.approx(32 * 510, rel=1e-3)


def test_the_cost_is_prf57s_and_bounds_the_draws_a_round():
    """PRF-57's sessions; the recommended draw keeps up with 12.6 s rounds in a 2-round batch, while s = 64 never does,
    because one prover a site sustains ≈ 77 whole DPs of length 4,096 a round."""
    for dps, prove, wall in P.ZK_SESSION:
        assert P.session_s(dps) == pytest.approx((prove, wall))
    assert P.session_s(0)[1] == pytest.approx(12.13, abs=0.01)
    assert P.sessions_s(512) == 84.0
    c = P.cell(P.LLAMA_8B, "prefill", 2048)
    cost = P.evaluate(c, "closure_any", 2, 1)[1]
    assert cost.dps < 16 and cost.batch() == 2 and cost.batch(today=True) <= 4
    assert P.evaluate(c, "work", 64)[1].batch() is None
