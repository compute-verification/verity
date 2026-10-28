import math

import pytest

from golden import model


@pytest.mark.parametrize("key", ["b57f", "b921", "a1fc"])
def test_calibration_within_ten_percent(key):
    assert abs(model.calibrate()[key]["ratio"] - 1) < 0.10


@pytest.mark.parametrize("key", list(model.VALIDATION))
def test_out_of_sample_within_ten_percent(key):
    assert abs(model.validate()[key]["ratio"] - 1) < 0.10


def test_row_ands_matches_sampled_proofs():
    from verity.core.protocols.sampled_proofs import optimize
    for n in (1, 64, 1024, 2048, 4096, 16384, 32768):
        assert model.row_ands(n) == optimize.row_ands(n)


def test_unknown_keys_refused():
    with pytest.raises(KeyError):
        model.design(commit={"rowbytes": 4096})
    with pytest.raises(KeyError):
        model.J({"comit": {}})


def test_draws_are_lambda_over_eps_whatever_the_window():
    bound = math.log(2 ** 40) / 0.01
    for steps in (1, 1000):
        r = model.J(model.design(point={"steps_per_window": steps}))
        assert r["draws"]["values"] == pytest.approx(bound, rel=1e-3)
        assert r["draws"]["pouw"] == pytest.approx(bound, rel=1e-3)


def test_h1_chunks_and_digits_match_pouw():
    from verity.core.protocols.pouw import leaf_lanes, weights
    for k in (256, 2048, 4096, 8192, 14336):
        assert model.h1_chunks(k) == weights.chunks(k)
        assert model.h1_digits(k // 32) == leaf_lanes.n_digits(k // 32)


def test_the_k2048_tile_is_refused_and_its_lanes_fit():
    """The 32 x 32 leaf at K 2,048 is one block over 2^27 in either reading of the cap; its lanes and combine fit."""
    for k_pad in (0, 1):
        tile = model.J(model.design(pouw={"scheme": "j10r"}, prover={"k_pad": k_pad}))
        assert any("k_log" in m for m in tile["infeasible"])
        lanes = model.J(model.design(pouw={"scheme": "j10r", "form": "lanes"}, prover={"k_pad": k_pad}))
        assert lanes["infeasible"] == []
        blocks = {b["name"]: b for b in lanes["proving"]["pouw"]["classes"][0]["blocks"]}
        assert blocks["lane"]["count"] == 32 and blocks["combine"]["count"] == 1
        assert blocks["lane"]["ands"] == pytest.approx(23e6, rel=0.05)


def test_chunked_lanes_fit_at_every_served_k():
    """Past K 4,096 a lane is proved in chunks of at most 4,096 entries, so at Llama-3.2-1B's K (2,048 and 8,192)
    every block stays within 2^27 even with compose's padding (k_pad 1), whose limit for a whole lane is near the
    struct worker's K ~5,041. (At K 14,336 the combine's 32 partial rows and 128 ok rows pass 2^26 ANDs.)"""
    assert 350 * 32 * 5041 < 2 ** 26 < 350 * 32 * 6144
    for k in (2048, 4096, 8192):
        r = model.J(model.design(pouw={"scheme": "j10r", "form": "lanes", "K": k}))
        assert r["infeasible"] == [], (k, r["infeasible"])
        assert all(b["k_log"] <= 27 for b in r["proving"]["pouw"]["classes"][0]["blocks"])


def test_stratified_draws_are_ln_s_over_delta_over_eps_a_class():
    r = model.J(model.design(sampling={"law": "stratified"}))["proving"]["values"]
    S = r["strata"]
    assert r["draws"] == pytest.approx(S * (math.log(S) + 40 * math.log(2)) / 0.01, rel=0.01)


def test_whole_rows_cost_at_least_exact_reads():
    assert model.opened(2048, 4096, "whole") == [4096] and model.opened(8192, 4096, "whole") == [4096, 4096]
    assert model.opened(2, 4096, "exact") == [2]
    whole = model.J(model.design())["proving"]["total_gpu_s"]
    exact = model.J(model.design(commit={"open": "exact"}))["proving"]["total_gpu_s"]
    assert whole > exact


@pytest.mark.parametrize("tile,kc", [((1, 1), 1), ((4, 8), 1), ((8, 8), 4)])
def test_tiles_cover_every_cell_once(tile, kc):
    d = model.design(partition={"tile": tile, "kchunk": kc})
    lin = [u for u in model.preset_units(d) if "tile" in u["name"]]
    for u, (_, K, N) in zip(lin, model.LINEARS):
        assert u["per_token"] * tile[0] * tile[1] == pytest.approx(model.LAYERS * N * kc)
        assert u["work"] == tile[0] * tile[1] * model.s8_coord_work(K // kc)


def test_joint_moves_pouw_into_the_values_tiles():
    r = model.J(model.design(pouw={"joint": True}))
    assert r["proving"]["pouw"]["draws"] == 0
    assert r["proving"]["values"]["draws_compute"] == pytest.approx(math.log(2 ** 40) / 0.01, rel=1e-3)


def test_inputs_override_is_scoped():
    base = model.J(model.design())["J"]
    with model.inputs(o_per_and=2 * model.I("o_per_and")):
        assert model.J(model.design())["J"] > base
    assert model.J(model.design())["J"] == base
    with pytest.raises(KeyError):
        model.inputs(no_such_input=1)


def test_unnoised_pouw_earns_no_credit():
    assert not model.J(model.design(pouw={"scheme": "j10r"}))["pouw"]["credit"]["earns_credit"]
    assert model.pouw_credit(model.design(pouw={"L": 8192}))["credited_share_of_linear_macs"] == 0
    assert model.J(model.design())["pouw"]["credit"]["earns_credit"]
