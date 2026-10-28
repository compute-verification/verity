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
    with model.inputs(h200_warm_c=2 * model.I("h200_warm_c")):
        assert model.J(model.design())["J"] > base
    assert model.J(model.design())["J"] == base
    with pytest.raises(KeyError):
        model.inputs(no_such_input=1)


@pytest.mark.parametrize("rows,k_log", [(2.66e6, 22), (6.97e6, 24), (24.2e6, 25), (93.2e6, 27)])
def test_k_pad_rows_puts_gd_provers_blocks_at_their_measured_k_log(rows, k_log):
    assert math.ceil(math.log2(rows) + model.I("k_pad_rows")) == k_log


def test_a_warm_prover_costs_less_than_a_process_a_session():
    for device in ("h200", "sm120"):
        warm, cold = (model.J(model.design(prover={"device": device, "warm": w}))["proving"]["total_gpu_s"]
                      for w in (True, False))
        assert warm < cold, device


def test_v32_credits_s_minus_one_segments():
    """L 384 at K 2,048: S 6, so 5 x 384 / 2,048 of the K 2,048 linears' MACs, not (K - L) / K."""
    d = model.design(pouw={"L": 384, "L_prime": 128, "credit": "k2048"})
    total = sum(K * N for _, K, N in model.LINEARS)
    k2048 = sum(K * N for _, K, N in model.LINEARS if K == 2048)
    assert model.pouw_credit(d)["credited_share_of_linear_macs"] == pytest.approx(k2048 / total * 5 * 384 / 2048)


def test_credit_needs_one_minus_eps_of_g_forced():
    """Un-noised calls earn 0; a noised one earns only at a forced share of at least 1 - eps (GD-SMELL MODEL-2)."""
    assert not model.J(model.design(pouw={"scheme": "j10r"}))["pouw"]["credit"]["earns_credit"]
    assert model.pouw_credit(model.design(pouw={"L": 8192}))["credited_share_of_linear_macs"] == 0
    assert not model.pouw_credit(model.design())["earns_credit"]                     # t 8: 1/8 + 3/8 of G
    assert model.pouw_credit(model.design(pouw={"L": 256, "L_prime": 128}))["earns_credit"]


def test_per_tile_pi_loses_the_floor_at_t3_and_per_segment_pi_keeps_it():
    """GD-BITS final: pi re-drawn each tile has a 2:4 route to 23/24 G at t 3 (b1 1/8), G at b1 3/16; re-drawn each
    segment closes it."""
    t3 = {"L": 384, "L_prime": 128}
    assert model.pouw_forced_frac(model.design(pouw={**t3, "noise": "permuted"})) == pytest.approx(23 / 24)
    assert model.pouw_forced_frac(model.design(pouw={**t3, "noise": "permuted", "b1": "wgmma"})) == 1.0
    assert model.pouw_forced_frac(model.design(pouw={**t3, "noise": "segment"})) == 1.0
    assert model.pouw_forced_frac(model.design(pouw={"L": 256, "L_prime": 128, "noise": "permuted"})) == 1.0


def test_word_bits_cap_the_floor_as_gd_bits_finds():
    """GD-BITS final: no 8-bit word of a whole product forces G (Strassen over Z/2^8, 7/8); t 3 with pi a segment keeps
    G from 13 low bits (11 at b1 3/16), losing 1/24 a bit below; fresh noise needs 9 bits tensor, 11 in-model."""
    def f(bits, **w):
        return model.pouw_forced_frac(model.design(pouw={"L": 384, "L_prime": 128, "noise": "segment",
                                                         "word_bits": bits, **w}))
    assert [f(b) for b in (8, 9, 12, 13, 16)] == pytest.approx([0.708, 0.833, 23 / 24, 1.0, 1.0], abs=1e-3)
    assert f(11, b1="wgmma") == 1.0 and f(10, b1="wgmma") < 0.99
    t2 = {"L": 256, "L_prime": 128}
    assert f(8, **t2) == 7 / 8 and f(8, pricing="in-model", **t2) == 0.766 and f(16, **t2) == 1.0
    assert f(8, noise="fresh") == 7 / 8 and f(9, noise="fresh") == 1.0 and f(10, noise="fresh", pricing="in-model") < 1
    assert model.pouw_forced_frac(model.design(pouw={"word_bits": 12})) == pytest.approx(0.5)      # t 8: 1/8 + 3/8
    d = model.design(pouw={**t2, "word_bits": 8})
    c = model.pouw_credit(d)
    assert not c["earns_credit"] and "card" in c["b1_price"] and "8-bit" in c["b1_price"]


def test_jr2_u_drawn_every_call_pays_its_prg_at_small_m():
    """GD-NOISE: JR2 per period 1.32 G; per call adds U's PRG, 3,492 / m G (55 G at the demo's 64 tokens)."""
    def noise(u, point):
        return model.J(model.design(point, pouw={"scheme": "jr2", "u_draw": u, "tile": (1, 1)}))["pouw"]["jr2_noise_g"]
    assert noise("period", "demo") == noise("period", "throughput") == 1.32
    assert noise("call", "demo") == pytest.approx(1.33 + 3492 / 64)
    assert noise("call", "throughput") < 2


def test_host_retention_holds_more_and_pays_the_copy():
    """VERITY_RETAIN=host: host_retain_GB bounds the window, and each step's bytes cross PCIe at d2h_GBs beside it."""
    dev, host = (model.J(model.design(sampling={"retain": w})) for w in ("device", "host"))
    part = "retain: D2H to the host, beyond the step and the DMA's HBM share"
    copy = host["commit"]["d2h_ms_a_step"]
    assert copy == pytest.approx(host["retained_GB_per_window"] / model.I("d2h_GBs") * 1e3)
    assert host["serve_overhead_ms"][part] >= max(0.0, copy - dev["t_remote_ms"]) and part not in dev["serve_overhead_ms"]
    big = {"retain_GB": None}
    many = model.design(point={"steps_per_window": 4}, sampling=big)
    assert any("on the device" in m for m in model.J(many)["infeasible"])
    assert not any("retains" in m for m in model.J(model.design(point={"steps_per_window": 2},
                                                                sampling={**big, "retain": "host"}))["infeasible"])


def test_the_census_adds_the_served_templates():
    """GD-PLANNER's 4,32,4 census: 60 templates beside the 4 linear tiles (26 with attention padded into one)."""
    base = model.J(model.design("demo", partition={"census": None}))["proving"]["values"]
    b4, bk = (model.J(model.design("demo", partition={"census": c}))["proving"]["values"] for c in ("b4", "b4-buckets"))
    assert (len(base["classes"]), len(b4["classes"]), len(bk["classes"])) == (6, 64, 30)
    assert b4["statements"] > bk["statements"] > base["statements"] and b4["gpu_s"] > bk["gpu_s"]
    assert b4["draws"] == pytest.approx(base["draws"], rel=0.01)
    assert [m.split(" (census)")[0] for m in b4["infeasible"]] == ["Int8RowQuant_v3{K=8192,R=4}"]   # 48 M, gates
    assert not model.J(model.design("demo", prover={"k_max": 28}))["infeasible"]
    with pytest.raises(ValueError):
        model.J(model.design("throughput", partition={"census": "b4"}))


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


def test_hashing_is_free_only_in_the_host_gap():
    """GD-HASH final: h = h_x - sigma h_leaf, sigma = min(1, G / B). At b921's 13.6 GB a forward and 4 KiB rows, gap
    first exposes 26.3 ms of 34.6 (sigma 0.25); under async or CUDA graphs G is 0 and every byte costs h_x."""
    def exposed(**point):
        r = model.J(model.design(point={"T": 8192, **point},
                                 commit={"corun": True, "hash_eff": model.I("hash_eff_rows")}))
        return r["serve_overhead_ms"]["Commit hashing (exposed)"], r["commit"]
    e, c = exposed(**{"async": False})
    assert c["hash_kernel_ms"] == pytest.approx(34.6, rel=0.01)
    assert e == pytest.approx(26.3, rel=0.03) and c["gap_sigma"] == pytest.approx(0.25, abs=0.02)
    e, c = exposed(**{"async": True})
    assert e == c["hash_kernel_ms"] and c["gap_sigma"] == 0
    kc = model.J(model.design(commit={"corun": True, "hash_eff": "kcommit"}))["commit"]["hash_kernel_ms"]
    assert kc == pytest.approx(1.22 * model.J(model.design(commit={"corun": True, "hash_eff": model.I("hash_eff_rows")}))
                               ["commit"]["hash_kernel_ms"], rel=1e-3)


def test_guessable_words_cost_draws_and_small_groups_are_refused():
    eight, four = (model.J(model.design(pouw={"word_bits": b}))["proving"]["pouw"]["draws"] for b in (8, 4))
    assert four / eight == pytest.approx((1 - 2 ** -8) / (1 - 2 ** -4), rel=1e-3)
    assert any("pouw.agg 4" in m for m in model.J(model.design(pouw={"agg": 4}))["infeasible"])
    blocks = model.pouw_units(model.design(pouw={"agg": 128}))
    assert all("(8x16," in u["name"] for u in blocks)


@pytest.mark.parametrize("m,whole,rel", [(64, 3.96, 0.01), (512, 2.71, 0.01), (8192, 1.47, 0.05)])
def test_dither_served_call_is_gd_denoise_whole_call(m, whole, rel):
    """GD-DENOISE final (r20261028-194021-247f): the four linears' served call at L 1,024, L' 128, over their plain GEMM,
    which at small m is the measured one (launch- and weight-read-bound), not the compute model's."""
    d = model.design(pouw={"L_prime": 128, "L": 1024, "credit": "all4"},
                     point={"T": max(m, 512), "real_tokens": m, "seqs": 1})
    parts, info = model.pouw_parts(d, model.stock_parts(d))
    call = sum(v for k, v in parts.items() if k.startswith("dither:") and "roots" not in k)
    assert call / info["dither_G_ms"] == pytest.approx(whole, rel=rel)
    compute = model.INT8_MACS_PER_TOKEN * m / (model.theta(model.I("gemm_mhz")) * model.I("gemm_eff_per_clock"))
    assert info["dither_G_ms"] >= compute * (1 - 1e-9)


def test_dither_t_s_gemm_grows_with_segments():
    o = {L: model.pouw_parts(d := model.design(pouw={"L": L, "credit": "all4"}), model.stock_parts(d))[0]
         for L in (512, 1024, 2048)}
    key = "dither: noised GEMM over (S-1) L, int32 T_s out (GD-DENOISE)"
    assert o[512][key] > o[1024][key] > o[2048][key] > 0
