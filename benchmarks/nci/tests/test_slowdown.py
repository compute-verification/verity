import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import catalog  # noqa: E402
import slowdown as S  # noqa: E402

BASE = {"kappa": 0.667, "b_w": 4.0, "b_a": 4.0, "d": 3072.0, "e": 7168.0, "c": 2.0, "n_updated": 1e11,
        "U_bits": 4e11, "U_token_bits": 4e11, "decode_flops": 0.0, "w_cut": 49152.0, "crossings": 2.0,
        "flops_per_token": 6e11}


def test_inputs_are_the_documented_ones():
    assert set(BASE) == set(S.INPUTS)


def test_thread_bound_reproduces_the_thread_draft():
    """The draft's arithmetic: N = 1e11, alpha* = 20 / (2N) = 1e-10, e = 7168, FP4: about 3e6 per counted
    multiplication (the draft's e is false for its model; the row uses 49152)."""
    K, Gamma = 2e7, 2e17
    assert S.thread_lower(BASE | {"kappa": 1.0}, K, Gamma, 20) == pytest.approx(4 / (2 * 1e-10 * 7168))


def test_weight_rate_is_capped_by_the_whole_update():
    seeded = BASE | {"U_bits": 3.2e5, "U_token_bits": 3.2e5}
    assert S.rates(seeded, 2e7, 20)[0] == pytest.approx(3.2e5 / 1e11)


def test_activation_rate_is_capped_by_a_token():
    assert S.rates(BASE, 2e7, 2.2)[1] == 2.2


def test_a_block_that_holds_the_state_gets_no_entry_fee():
    small = BASE | {"U_bits": 1e6, "U_token_bits": 1e6}
    _, _, d, e = S.rates(small, 2e7, 20)
    assert (d, e) == (1.0, 0.667 * 6e11 / 2)


def test_chain_read_factor_matches_its_definition():
    """f(K) at K = 2e7, b_w = b_a = 4, d = 2048: the split term binds near mu = 0.25."""
    assert S.read_factor(2e7, 4, 4, 2048) == pytest.approx(3.286e-3, rel=1e-3)


def test_shard_floor_is_twice_w_cut_over_b_tok():
    """Just past half a block of state, a forward-only shard pays 2 w_cut / b_tok when Gamma is inference's."""
    K, b_tok, n_active = 2e7, 20, 3.2e10
    Gamma = K * 2 * n_active / b_tok
    i = BASE | {"U_token_bits": K / 2, "crossings": 1.0, "w_cut": 28672.0, "flops_per_token": 2 * n_active}
    assert S.shard(i, K, Gamma) == pytest.approx(2 * 28672 / b_tok)


def test_fit_is_free_for_inference_shaped_work():
    K, b_tok, n_active = 2e7, 20, 3.2e10
    Gamma = K * 2 * n_active / b_tok
    i = BASE | {"U_bits": 0.0, "U_token_bits": 0.0, "c": 1.0, "flops_per_token": 2 * n_active}
    assert S.construction_upper(i, K, Gamma, b_tok) == (pytest.approx(1.0), "fit")


def test_decode_that_does_not_fit_shards_the_decoding():
    K, Gamma = 2e7, 6.4e16
    i = BASE | {"U_bits": 3.2e5, "U_token_bits": 3.2e5, "decode_flops": 2.1e17, "crossings": 1.0}
    assert S.construction_upper(i, K, Gamma, 20, splits=False)[1] == "decode-shard"


def test_search_reproduces_the_shard_closed_form():
    """With no data and no decoding, the best split is half a block of state: the closed form."""
    K, b_tok, n_active = 2e7, 20, 3.2e10
    Gamma = K * 2 * n_active / b_tok
    U = 8e10
    c = S.search(U, True, 6 * n_active, 2 * 28672, 0.0, 0.0, K, Gamma)
    i = BASE | {"U_token_bits": U, "crossings": 2.0, "w_cut": 28672.0, "flops_per_token": 6 * n_active}
    assert c["basis"] == "shard"
    assert c["rho"] == pytest.approx(S.shard(i, K, Gamma), rel=1e-2)
    assert c["reads"]["state"] == pytest.approx(K / 2, rel=1e-2)


def test_chain_bound_is_kappa_f():
    """The slowdown >= kappa_chain f(K) / (2 alpha), the activation rate capped by the crossing over the residual's entries."""
    K, Gamma = 2e7, 6.4e16
    chain = {"kappa": 0.1, "d": 1536, "path": "x"}
    want = 0.1 * S.read_factor(K, 4.0, 4.0, 1536) / (2 * K / Gamma)
    assert S.chain_lower(BASE, chain, K, Gamma, 20, 12288) == pytest.approx(want)
    narrow = BASE | {"w_cut": 12288.0}
    assert S.chain_lower(narrow, chain, K, Gamma, 20, 12288) == pytest.approx(
        0.1 * S.read_factor(K, 4.0, 1.0, 1536) / (2 * K / Gamma))


def test_sweep_at_a_rows_own_rates_is_its_bracket():
    rows, cals, params = catalog.load("rows"), catalog.load("calibrations"), catalog.load("parameters")
    row = next(r for r in rows if r["id"] == "pretrain-full")
    cal = next(c for c in cals if c["id"] == "kimi-k2-1m-raw")
    b = S.bracket(row, cal, params)
    cell = S.sweep(row, cal, params, [b["inputs"]["b_w"]], [b["inputs"]["b_a"]])[0]
    assert (cell["thread"], cell["chain"], cell["upper"]) == pytest.approx((b["lower"], b["chain"], b["upper"]))


def test_efficiency_of_bf16_is_one_over_the_bounds():
    rows, cals, params = catalog.load("rows"), catalog.load("calibrations"), catalog.load("parameters")
    row = next(r for r in rows if r["id"] == "pretrain-full")
    cal = next(c for c in cals if c["id"] == "kimi-k2-1m-raw")
    effs = S.efficiency(row, cal, params)
    bf16 = next(x for x in effs if x["recipe"] == "bf16")
    cell = S.sweep(row, cal, params, [16], [16])[0]
    assert (bf16["ceiling"], bf16["attack"]) == pytest.approx((1 / cell["chain"], 1 / cell["upper"]))
    assert all(x["attack"] <= x["ceiling"] for x in effs)


def test_every_bracket_is_consistent():
    """A proven lower bound above an explicit construction's cost means a row's inputs break the theorem's
    assumptions; the caps in `rates` exist because two rows did."""
    for b in S.brackets(catalog.load("rows"), catalog.load("calibrations"), catalog.load("parameters")):
        assert 1 <= b["lower"] <= b["upper"] * (1 + 1e-9), b
        assert b["chain"] is None or 1 <= b["chain"] <= b["upper"] * (1 + 1e-9), b
