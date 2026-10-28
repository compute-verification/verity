import math

import pytest

from golden import model


@pytest.mark.parametrize("key", ["b57f", "b921", "a1fc"])
def test_calibration_within_ten_percent(key):
    assert abs(model.calibrate()[key]["ratio"] - 1) < 0.10


@pytest.mark.parametrize("key", list(model.VALIDATION))
def test_out_of_sample_within_ten_percent(key):
    assert abs(model.validate()[key]["ratio"] - 1) < 0.10


def test_row_ands_is_sampled_proofs_at_c_flocks_compression():
    """optimize.row_ands at its 58,120 ANDs a compression, and 1.40x it at C-Flock's 81,402 rows."""
    from verity.core.protocols.sampled_proofs import optimize
    with model.inputs({"sha512_ands": optimize.SHA512_ANDS, "hm96_ands": optimize.HM96_ANDS}):
        for n in (1, 64, 1024, 2048, 4096, 16384, 32768):
            assert model.row_ands(n) == optimize.row_ands(n)
    assert model.row_ands(4096) / optimize.row_ands(4096) == pytest.approx(81_402 / 58_120)


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
    with model.inputs({"sha512_ands": 58_120, "hm96_ands": 116_240}):         # the inbox's openings, 454 a byte
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


def test_lanes_fit_at_k2048_and_need_k_pad_0_past_it():
    """Past K 4,096 a lane is proved in chunks of at most 4,096 entries. At C-Flock's 81,402 rows a SHA-512
    compression a 4,096-entry lane (or chunk) is ~75 M ANDs with its 12 operand rows: past 2^26, so within 2^27 only
    at k_pad 0, not the struct worker's 1. K 2,048's lane (~39 M) fits at k_pad 1."""
    def lanes(k, k_pad):
        with model.inputs({"k_pad_h1_lane": k_pad}):
            return model.J(model.design(pouw={"scheme": "j10r", "form": "lanes", "K": k}))
    assert lanes(2048, 1)["infeasible"] == []
    for k in (4096, 8192):
        assert any("k_log 28 > 27" in m for m in lanes(k, 1)["infeasible"])
        r = lanes(k, 0)
        assert r["infeasible"] == [] and all(b["k_log"] <= 27 for b in r["proving"]["pouw"]["classes"][0]["blocks"])


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
    with model.inputs(o_per_and_h200=2 * model.I("o_per_and_h200")):
        assert model.J(model.design())["J"] > base
    assert model.J(model.design())["J"] == base
    with pytest.raises(KeyError):
        model.inputs(no_such_input=1)


def test_unnoised_pouw_earns_no_credit():
    assert not model.J(model.design(pouw={"scheme": "j10r"}))["pouw"]["credit"]["earns_credit"]
    assert model.pouw_credit(model.design(pouw={"L": 8192}))["credited_share_of_linear_macs"] == 0
    assert model.J(model.design())["pouw"]["credit"]["earns_credit"]


def test_sessions_pad_draws_to_a_power_of_two_or_split():
    """GD-PROVER's session: P = max(8, 2^ceil(log2 k)) padded draws, split where a session costs less than its pad."""
    assert model.sessions(3, 4096, 10.0, 1.0) == (1, 8)
    assert model.sessions(1500, 4096, 1e6, 1.0) == (1, 2048)          # dear sessions: pad
    assert model.sessions(1500, 4096, 1.0, 1.0) == (5, 1504)          # cheap sessions: 1024 + 256 + 128 + 64 + 32
    assert model.sessions(5000, 2048, 1e6, 1.0) == (3, 5120)          # two full sessions and 904 in 1,024


def test_the_period_pools_draws():
    """Draws ~ Lambda / eps a period, so a window's proving falls as 1 / period; eps then bounds the period's work."""
    one = model.J(model.design())
    ten = model.J(model.design(sampling={"period": 10}))
    assert ten["draws"]["values"] == pytest.approx(one["draws"]["values"] / 10, rel=1e-3)
    assert ten["proving"]["gpu_s"] < one["proving"]["gpu_s"] / 5
    assert ten["serve_term"] == one["serve_term"] and ten["eps_of_a_window"] == pytest.approx(0.1)


def test_one_rep_is_cheaper_at_the_floor_and_sm120_at_its_price():
    two, one = (model.J(model.design(prover={"reps": r})) for r in (2, 1))
    assert one["proving"]["gpu_s"] < two["proving"]["gpu_s"]
    assert (two["soundness_bits"], one["soundness_bits"]) == (195.5, 100.0)
    assert model.J(model.design(prover={"device": "sm120"}))["proving"]["gpu_s"] < two["proving"]["gpu_s"] / 10


def test_hashing_beside_the_gemm_is_never_free():
    """GD-HASH: a byte hashed beside the GEMM costs it h_b / h_x of the byte's exclusive time, up to the beside
    capacity; the rest runs alone."""
    for T in (512, 8192):
        d = model.design(point={"T": T}, commit={"corun": True, "hash_eff": model.I("hash_eff_rows")})
        r = model.J(d)
        o, c = r["serve_overhead_ms"], r["commit"]
        idle = r["stock_parts_ms"]["host gap between forwards"] + r["stock_parts_ms"]["launch gaps in the forward"]
        hashed = c["hash_kernel_ms"] - min(idle, c["hash_kernel_ms"])
        assert hashed > 0 or T == 512                                  # T 512's host gap holds all its hashing
        assert (o["GEMM's loss to co-running hash"] > 0) == (hashed > 0)
        assert o["Commit hashing (exposed)"] + o["GEMM's loss to co-running hash"] >= c["beside_h_ratio"] * hashed


def test_guessable_words_cost_draws_and_small_groups_are_refused():
    eight, four = (model.J(model.design(pouw={"word_bits": b}))["proving"]["pouw"]["draws"] for b in (8, 4))
    assert four / eight == pytest.approx((1 - 2 ** -8) / (1 - 2 ** -4), rel=1e-3)
    assert any("pouw.agg 4" in m for m in model.J(model.design(pouw={"agg": 4}))["infeasible"])
    blocks = model.pouw_units(model.design(pouw={"agg": 128}))
    assert all("(8x16," in u["name"] for u in blocks)


def test_dither_serve_parts_are_gd_denoise_arms():
    """At m 8,192 and L' 128: coins 0.205, X 0.075 and O's recompute 0.202 of the credited calls' plain GEMM."""
    r = model.J(model.design(pouw={"L_prime": 128, "L": 1024}))
    o = r["serve_overhead_ms"]
    coins, x = o["dither: coins (d_A bits and V)"], o["dither: noised A (X)"]
    assert x / coins == pytest.approx(0.075 / 0.205)
    assert o["dither: O recomputed on a', b' (less the plain GEMM)"] / coins == pytest.approx(0.202 / 0.205)
