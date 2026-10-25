from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configs  # noqa: E402
import rho_attack as ra  # noqa: E402
from configs import Map  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
QWEN = configs.get("qwen2.5-1.5b")
K = ra.K_GRID["1MB"]
NVFP4 = ra.Beta(beta_v=1.4, beta_x=1.4, beta_w=0.05, raw_bits=3.5)


def test_configs_match_their_published_sizes():
    sizes = {"qwen2.5-1.5b": 1.54e9, "qwen2.5-7b": 7.62e9, "qwen2.5-72b": 72.7e9, "deepseek-v4-pro": 1.57e12,
             "kimi-k2": 1.04e12}
    for name, n in sizes.items():
        assert configs.get(name).params == pytest.approx(n, rel=0.02), name
    assert configs.get("deepseek-v4-pro").active_params == pytest.approx(49e9, rel=0.01)


def test_a_small_state_carried_whole_costs_nothing_at_matched_c():
    c = K * QWEN.phi()
    a = ra.whole_state(QWEN, 1000.0, K, c, NVFP4, ra.Knobs(), "sft")
    assert a.rho == pytest.approx(1.0)


def test_a_large_c_makes_blocks_data_bound_at_the_floor():
    c = ra.C_FIXED
    a = ra.whole_state(QWEN, 1000.0, K, c, NVFP4, ra.Knobs(), "sft")
    b = ra.bound(QWEN, 1000.0, 0.0, K, c, NVFP4, ra.Knobs(), False, "sft")
    assert b["floor"] == pytest.approx(b["eta"] / 3)
    assert a.rho == pytest.approx(b["rho"], rel=1e-3)


def test_grpo_shares_the_prompt_and_generates_its_rollouts():
    """Duplicate sharing: a block brings each prompt once and a seed per rollout, so at matched c GRPO's whole-state
    layout is compute-bound and its data floor is far below SFT's."""
    c = ra.C_FIXED
    sft = ra.whole_state(QWEN, 1000.0, K, c, NVFP4, ra.Knobs(), "sft")
    grpo = ra.whole_state(QWEN, 1000.0, K, c, NVFP4, ra.Knobs(), "grpo")
    assert grpo.rho < sft.rho / 10
    assert grpo.params["rollouts_per_prompt"] == ra.Knobs().group


def test_the_two_stage_pipeline_at_s_equal_k_is_the_programs_cliff():
    """Without an accumulator, two stages each holding S/2 = k/2 pay (h + F) per token at each side: the program's
    (2/3) * 2 * F / h at eta = 1."""
    beta = ra.Beta(1.4, 1.4, 0.05, raw_bits=3.5, b_acc=0.0)
    F = 14_000.0 * QWEN.d / 7168
    c = K * QWEN.phi()
    S = 0.999 * K
    a = ra.pipeline(QWEN, S, F, K, c, beta, ra.Knobs(), "sft")
    b = ra.bound(QWEN, S, F, K, c, beta, ra.Knobs(), False)
    assert a.params["stages"] == 2
    assert a.rho / b["cliff"] == pytest.approx(1.0, abs=0.05)


def test_the_accumulator_pushes_the_cliff_below_k():
    """Under the fixed recipe a block holding less than a step carries the step's running gradient, so a state of k
    no longer fits two stages."""
    c = K * QWEN.phi()
    a = ra.pipeline(QWEN, 0.999 * K, 3000.0, K, c, NVFP4, ra.Knobs(), "sft")
    assert a.params["stages"] > 2
    assert ra.whole_state(QWEN, 0.6 * K, K, c, NVFP4, ra.Knobs(), "sft") is None


def test_tiling_one_map_is_the_loomis_whitney_count():
    """With nothing fresh per token but its id, R rows x Q tokens with R * d_in * s + 0 <= k and Q * h <= k - R d_in s
    is optimal at R = k / (2 d_in s): 4 d_in s h d_out / k^2 blocks per token."""
    m = Map("w", 1024, 4096, True, False)
    beta = ra.Beta(beta_v=1e-12, beta_x=1e-12, beta_w=1.0, raw_bits=1e9, b_acc=0.0)
    k, s = 1e6, 1.0
    bpt, par = ra._map_blocks(QWEN, m, 0, s, 0.0, k, 1e30, beta, ra.Knobs(), "exact", True, False)
    assert bpt == pytest.approx(4 * m.d_in * s * QWEN.h * m.d_out / k**2, rel=0.01)
    assert par["rows"] == pytest.approx(k / (2 * m.d_in * s), rel=0.1)


def test_the_full_training_bound_is_the_covering_count():
    c = ra.C_FIXED
    S = QWEN.params * NVFP4.beta_w
    for wl, u in (("sft", 3), ("grpo", 4)):
        b = ra.bound(QWEN, S, 3000.0, K, c, NVFP4, ra.Knobs(), True, wl)
        assert b["regime"] == "full"
        assert b["rho"] == pytest.approx(c * NVFP4.beta_v * QWEN.values_per_token / (K * u * QWEN.f))
    eta1 = ra.bound(QWEN, S, 3000.0, K, K * QWEN.phi(), NVFP4, ra.Knobs(), True)
    assert eta1["rho"] == pytest.approx(QWEN.phi() * NVFP4.beta_v / (6 * QWEN.d_eff))
    gx = replace(NVFP4, beta_gx=0.05)
    both = ra.bound(QWEN, S, 3000.0, K, K * QWEN.phi(), gx, ra.Knobs(), True)
    assert both["rho_gx"] == pytest.approx(both["rho"] * 0.05 / NVFP4.beta_v)
    cliff = ra.bound(QWEN, 2 * K, 3000.0, K, K * QWEN.phi(), gx, ra.Knobs(), False)
    assert cliff["rho_gx"] == cliff["rho"]


def test_only_recipe_changes_and_lossy_corrections_carry_their_own_broken_assumption():
    c = K * QWEN.phi()
    res = ra.rho_attack(QWEN, 2 * K, 3000.0, K, c, NVFP4)
    for name, a in res["all"].items():
        if name in ra.EXACT:
            assert a.breaks is None, name
        else:
            assert a.breaks and a.breaks[0] in "CDE", name
    assert res["all"]["lowrank_pipeline"].breaks.startswith("E or D")
    assert res["best_exact"].strategy in ra.EXACT


def test_a_rank_r_correction_under_the_measured_G_breaks_an_assumption_even_above_F():
    c = K * QWEN.phi()
    kn = ra.Knobs(ranks=(64,), grad_factor=28.0)
    a = ra.lowrank_pipeline(QWEN, 2 * K, 203.0, K, c, NVFP4, kn, "sft")
    assert 64 * kn.b_coef > 203.0
    assert a.breaks.startswith("E or D") and "gradient" in a.breaks


def test_from_b1_takes_medians_with_G_the_backward_part():
    sys.path.insert(0, str(HERE / "inputs"))
    import from_b1

    def entry(s_bits, ff, ffb, bv, gx):
        row = [bv, gx, 0.0, 4.0, 0.1, 0.1, ff, ff, ffb - ff, ffb]
        return {"fam": "A2", "S_bits": s_bits, "d": 1536, "model": "Qwen2.5-1.5B-Instruct", "fp8t": row,
                "nvfp4": [x / 2 for x in row]}

    doc = {"schema": "tvi-b1/F_beta/v2", "entries": [entry(272, 3000, 12000, 9.0, 0.4), entry(272, 4000, 14000, 8.0, 0.3),
                                                      entry(24699428864, 1000, 11000, 7.0, 0.5),
                                                      {"fam": "B3", "S_bits": 272, "fp8t": [0.0] * 10}]}
    out = from_b1.convert(doc)
    assert out["F"]["A2-34B"]["fp8"] == 3500 and out["G"]["A2-34B"]["fp8"] == 9500
    assert out["S_bits"] == {"A2-34B": 272} and "full" in out["F"]
    inp = ra.Inputs.from_json(out)
    assert inp.grad_factor("A2-34B", "fp8", ra.Knobs()) == pytest.approx(9500 / 3500)
    assert inp.beta_for("A2-34B", "fp8").gx == pytest.approx(0.35)
    assert inp.F_for("full", "nvfp4", configs.get("kimi-k2")) == pytest.approx(500 * 7168 / 1536)
    with pytest.raises(ValueError, match="schema"):
        from_b1.convert({"schema": "v1", "entries": []})


def test_s_labels():
    kn = ra.Knobs()
    assert ra.resolve_S("10kB", QWEN, NVFP4, kn) == 80_000
    assert ra.resolve_S("10kbit", QWEN, NVFP4, kn) == 10_000
    assert ra.resolve_S("1MB", QWEN, NVFP4, kn) == 8e6
    assert ra.resolve_S("tinylora", QWEN, NVFP4, kn) == 208
    assert ra.resolve_S("full", QWEN, NVFP4, kn) == QWEN.params * NVFP4.beta_w
    assert ra.resolve_S("rank16", QWEN, NVFP4, kn) == ra.lora_params(QWEN, 16) * kn.lora_bits
    assert ra.resolve_S("mine", QWEN, NVFP4, kn, {"mine": 5.0}) == 5.0
    with pytest.raises(ValueError, match="S label"):
        ra.resolve_S("lots", QWEN, NVFP4, kn)


def test_inputs_format():
    doc = {"precisions": {"fp8": {"beta_v": 3.0, "beta_x": 3.1, "beta_w": 0.2}},
           "F": {"1MB": 14000, "full": {"fp8": 7000}}, "measured_on": {"d_model": 7168}}
    inp = ra.Inputs.from_json(doc)
    assert inp.F_for("1MB", "fp8", QWEN) == pytest.approx(14000 * 1536 / 7168)
    assert inp.F_for("full", "fp8", QWEN) == pytest.approx(7000 * 1536 / 7168)
    absolute = ra.Inputs.from_json({**doc, "F_scaling": "absolute"})
    assert absolute.F_for("1MB", "fp8", QWEN) == 14000
    assert inp.beta_for("1MB", "fp8").g == 3.0
    with pytest.raises(ValueError, match="lacks beta_w"):
        ra.Inputs.from_json({**doc, "precisions": {"fp8": {"beta_v": 3.0, "beta_x": 3.1}}})
    with pytest.raises(ValueError, match="need"):
        ra.Inputs.from_json({"precisions": doc["precisions"]})
    with pytest.raises(ValueError, match="d_model"):
        ra.Inputs.from_json({k: v for k, v in doc.items() if k != "measured_on"})


@pytest.mark.parametrize("name", ["in_hand_fullrank.json", "in_hand_rank64.json"])
def test_the_in_hand_inputs_load(name):
    inp = ra.Inputs.load(HERE / "inputs" / name)
    assert set(inp.precisions) == {"fp8_s1", "fp8_s1000", "nvfp4_s1", "nvfp4_s1000"}
    assert "full" in inp.F


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory) -> Path:
    """One search over Qwen2.5-1.5B on the rank-64 inputs: its output JSON."""
    out = tmp_path_factory.mktemp("cli")
    assert ra.main(["--inputs", str(HERE / "inputs" / "in_hand_rank64.json"), "--out", str(out), "--configs",
                    "qwen2.5-1.5b", "--workloads", "sft", "--curve-points", "3", "--no-plots"]) == 0
    return out / "rho_attack_in_hand_rank64.json"


def test_cli_writes_rows_for_every_label(cli_run):
    doc = json.loads(cli_run.read_text())
    labels = list(ra.Inputs.load(HERE / "inputs" / "in_hand_rank64.json").F)
    assert len(doc["rows"]) == len(labels) * len(ra.K_GRID) * 2 * 4
    for r in doc["rows"]:
        assert r["bound"]["rho"] >= 1
        if r["attack"] is not None:
            assert math.isfinite(r["attack"]["rho"]) and r["attack"]["rho"] >= 1 - 1e-9
    assert "| qwen2.5-1.5b |" in (cli_run.parent / "rho_attack_in_hand_rank64.md").read_text()


def test_merge_joins_per_config_runs(cli_run, tmp_path):
    pytest.importorskip("matplotlib")
    doc = json.loads(cli_run.read_text())
    for item in doc["rows"] + doc["curves"]:
        item["config"] = "other"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(doc))
    assert ra.main(["--merge", str(cli_run), str(other), "--out", str(tmp_path / "all")]) == 0
    doc = json.loads((tmp_path / "all" / "rho_attack_in_hand_rank64.json").read_text())
    assert {r["config"] for r in doc["rows"]} == {"qwen2.5-1.5b", "other"}
    assert (tmp_path / "all" / "rho_attack_in_hand_rank64_nvfp4_s1.png").stat().st_size > 0


def test_readout_tables_name_both_bounds(cli_run):
    import readout

    doc = json.loads(cli_run.read_text())
    for item in doc["rows"] + doc["curves"]:
        item.update(config="kimi-k2", precision="fp8", c_mode="eta1")
        item["workload"] = item.get("workload", "sft")
    text = readout.tables(doc)
    assert "bound (beta_v count) | bound (beta_gx count)" in text
    assert "| Kimi K2 | fp8 | 10kB |" in text and "| Kimi K2 | fp8 | 2.5MB |" in text


def test_fanout_e_is_the_largest_layer_fanout_and_the_estimate_is_phi_beta_over_kappa_e():
    import fanout

    k2, v4 = configs.get("kimi-k2"), configs.get("deepseek-v4-pro")
    assert fanout.e_layers(k2) == 2 * 2048 * 9 + 384 == 37_248
    assert fanout.e_layers(v4) == 2 * 3072 * 7 + 384
    assert fanout.fanouts(k2)["attention input (normed residual)"][0] == 1536 + 576
    kappa = 3 * k2.f / fanout.products(k2)
    assert fanout.estimate(k2, 0.5, "sft") == pytest.approx(k2.phi() * 0.5 / (kappa * 37_248))
    assert fanout.frozen_estimate(k2, 0.5, 1, "sft") < fanout.estimate(k2, 0.5, "sft")


def _entry(L: int = 28, F: float = 1000.0, G: float = 8000.0, layer_bits: float = 4e6, emb: float = 5e6) -> dict:
    per = {"F": [F] * L, "G": [G] * L, "beta_w": 0.1, "embed_bits": emb,
           "layer_bits": {str(j): {"attn": layer_bits / 4, "mlp": 3 * layer_bits / 4} for j in (0, 9, 18, 27)}}
    return {"task": "gsm8k", "seed": 1, "step": 1, "eval_tokens": 2048, "embed_params": 233_373_696, "rollouts": 256,
            "prompts": 32, "n_seq": 56, "n_tok": 15_282, "clen": 237.6, "prompt_mean": 103.6, "fp8": per, "nvfp4": per}


def test_matched_block_counts_and_the_coordinators_instantiation():
    import matched_block as mb

    qwen = configs.get("qwen2.5-1.5b")
    st = mb.Step(_entry(), "fp8", qwen)
    assert (st.e, st.acts, st.prods) == (17_920, 13_568, 46_792_704)
    brep = {"Ff_min": (17 + 1008) / 1536, "mean": (17 + 4206) / 1536}
    two = mb.median_block(qwen, 0.103, 1008, 2, brep)
    assert two["beta_x_bound"] == pytest.approx(0.073, abs=5e-4) and two["Q"] == pytest.approx(10_100, rel=0.01)
    assert two["products_per_entering_coordinate"] == 2 * 30_464


def test_an_embedding_entry_block_holds_the_table_and_reads_token_ids_only():
    import matched_block as mb

    st = mb.Step(_entry(), "fp8", configs.get("qwen2.5-1.5b"))
    emb, bnd = st.block(0, 1), st.block(1, 1)
    assert emb["l_w"] == pytest.approx(5e6 + 4e6 + st.SMALL) and emb["l_in"] == pytest.approx(emb["Q"] * 17)
    assert bnd["l_in"] == pytest.approx(bnd["Q"] * (17 + 1000)) and bnd["products_per_entering_coordinate"] == 30_464
    assert emb["rollouts"] == 56 and emb["beta_x_bound"] < bnd["beta_x_bound"]
    assert emb["violation"]["Ff_min"] == pytest.approx(st.beta_rep("Ff_min") / emb["beta_x_bound"])


def test_a_step_whose_state_fits_is_two_blocks_and_a_long_rollout_is_cut_into_chunks():
    import matched_block as mb

    qwen = configs.get("qwen2.5-1.5b")
    small = mb.Step(_entry(layer_bits=3e5, emb=5e5), "fp8", qwen).step("chain")
    assert small["blocks"] == 2 and small["rho"] == pytest.approx(2 * mb.C / small["T0"])
    st = mb.Step(_entry(), "fp8", qwen)
    # one rollout's forward fits beside the state, its forward and backward together don't: two chunks per rollout
    assert st._tiles(12e6, 0.0, 3e6, 6e6, True, "chain", 0, mb.K) == 56 * 2
    assert st._tiles(12e6, 0.0, 3e6, 0.0, False, "chain", 0, mb.K) == 256 / 2
    assert st._tiles(12e6, 0.0, 9e6, 0.0, False, "chain", 0, mb.K) == math.inf


def test_the_corrected_allowance_reproduces_the_dense_optimum_and_k2s_closure():
    import matched_block as mb

    qwen, k2 = configs.get("qwen2.5-1.5b"), configs.get("kimi-k2")
    brep = {"Ff_min": 0.667, "mean": 2.75}
    rows = mb.dense_loss(qwen, 0.103, brep)
    bnd = next(r for r in rows if r["entry"] == "cheapest boundary")
    assert bnd["R_continuous"]["Ff_min"] == pytest.approx(mb.K / (4 * 0.103 * 17_920 * 1536), rel=0.01)
    assert bnd["m"] == 2 and bnd["R"]["Ff_min"] == pytest.approx(1.76, abs=0.05)
    assert bnd["R"]["mean"] == pytest.approx(bnd["R"]["Ff_min"] * 2.75 / 0.667)
    rows = {r["held"]: r for r in mb.closure(k2, 0.103, brep)}
    assert rows["1 routed expert, its tokens only"]["p"] == 3 * 2048 + 384
    assert rows["attention, 1 layer"]["p"] == pytest.approx(101_122_048 / 7168)
    assert all(r["R"]["Ff_min"] < 1 for r in rows.values() if r["entry"] == "cheapest boundary")


def test_matched_block_report_runs_on_the_extracted_inputs(tmp_path):
    import matched_block as mb

    here = Path(__file__).resolve().parents[1]
    assert mb.main(["--inputs", str(here / "inputs" / "a2_full_blocks.json"),
                    "--measured", str(here / "inputs" / "measured_b1.json"), "--out", str(tmp_path)]) == 0
    text = (tmp_path / "matched_block.md").read_text()
    assert "coordinator's instantiation, fp8" in text and "## 4. Loss factor" in text
    res = json.loads((tmp_path / "matched_block.json").read_text())
    assert res["coordinator_instantiation"]["fp8/Ff_min"]["beta_x_bound"] == pytest.approx(0.0729, abs=5e-4)
