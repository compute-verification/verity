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
    """The 32 x 32 leaf at K 2,048 is one block over 2^27 in either reading of the cap; its lanes and combine fit.
    A lane is its 4 x 8 cells' MACs at h1_ands_per_mac plus its 12 operand rows and its partial row."""
    lane = model.I("h1_ands_per_mac") * 32 * 2048 + 12 * model.row_ands(2052) + model.row_ands(8 * (64 + 2))
    for k_pad in (0, 1):
        with model.inputs({"k_pad_h1_lane": k_pad}):
            tile = model.J(model.design(pouw={"scheme": "j10r"}))
            assert any("k_log" in m for m in tile["infeasible"])
            lanes = model.J(model.design(pouw={"scheme": "j10r", "form": "lanes"}))
        assert lanes["infeasible"] == []
        blocks = {b["name"]: b for b in lanes["proving"]["pouw"]["classes"][0]["blocks"]}
        assert blocks["lane"]["count"] == 32 and blocks["combine"]["count"] == 1
        assert blocks["lane"]["ands"] == pytest.approx(lane)


def test_pouw_tiles_match_the_inbox_and_small_tiles_prove_cheaper():
    """The inbox's 1 x 1 ~2.7 M and 32 x 32 ~736 M ANDs a block (19:32Z); draws stay ~Lambda / eps, so proving
    falls with the block, and openings take a larger share of a smaller block (~1.5 (1/R + 1/C))."""
    def pouw(tile, form="tile", **kw):
        return model.J(model.design(pouw={"scheme": "j10r", "tile": tile, "form": form}, **kw))["proving"]["pouw"]
    small, big = pouw((1, 1))["classes"][0], pouw((32, 32))["classes"][0]
    assert small["ands_with_openings"] == pytest.approx(2.7e6, rel=0.15)
    assert big["ands_with_openings"] == pytest.approx(736e6, rel=0.1)
    assert small["openings_share"] > 0.7 and big["openings_share"] < 0.15
    costs = [pouw(t, "lanes")["gpu_s"] for t in ((1, 1), (2, 2), (4, 8), (8, 8), (16, 16), (32, 32))]
    assert costs == sorted(costs) and costs[-1] > 100 * costs[0]
    assert pouw((1, 1))["draws"] == pytest.approx(40 * math.log(2) / 0.01, rel=0.01)


def test_a_bigger_cap_pays_prover_memory_and_cost_a_bit():
    """16 x 16 at K 2,048 is one ~2^27.5-AND block: k_log 29 past compose's 27; at k_max 29 it fits the cap but not
    the H200's memory at 2^29 x prover_bytes_per_row, and costs cap_cost_per_bit a bit past 27."""
    pouw = {"scheme": "j10r", "tile": (16, 16)}
    at27 = model.J(model.design(pouw=pouw))
    assert any("k_log 29 > 27" in m for m in at27["infeasible"])
    at29 = model.J(model.design(pouw=pouw, prover={"k_max": 29}))
    assert not any("k_log" in m for m in at29["infeasible"])
    assert any("prover memory" in m for m in at29["infeasible"])
    with model.inputs({"prover_bytes_per_row": 237}):
        assert model.J(model.design(pouw=pouw, prover={"k_max": 29}))["infeasible"] == []
        with model.inputs({"cap_cost_per_bit": 0.0}):
            flat = model.J(model.design(pouw=pouw, prover={"k_max": 29}))["proving"]["pouw"]["gpu_s"]
        dear = model.J(model.design(pouw=pouw, prover={"k_max": 29}))["proving"]["pouw"]["gpu_s"]
    assert dear > flat


def test_chunked_lanes_fit_at_every_served_k():
    """Past K 4,096 a lane is proved in chunks of at most 4,096 entries, so at Llama-3.2-1B's K (2,048 and 8,192)
    every block stays within 2^27 even with compose's padding (k_pad 1); a whole lane at K 8,192 would not. (At
    K 14,336 the combine's 32 partial rows and 128 ok rows pass 2^26 ANDs.)"""
    assert model.I("h1_ands_per_mac") * 32 * 8192 + 12 * model.row_ands(8196) > 2 ** 26
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
