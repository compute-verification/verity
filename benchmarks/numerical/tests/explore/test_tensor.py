"""Tensor-core cost model (``explore.tensor``): the limb price list, the INT32 accumulator bound, the TensorZKP
calibration, the design x field rows and the VOLE row are pinned.  Everything here is a property of the stated cost
model, not a measurement."""

from __future__ import annotations

import pytest
from verity_numerical.bench import contract
from verity_numerical.explore import tensor as t

UTIL = 0.172        # TensorZKP degree-2 sumcheck: fraction of the H100 INT8 dense peak (derived in tensorzkp_calibration)


def test_limb_price_list():
    assert t.limbs(64) == 8 and t.limbs(31) == 4 and t.limbs(61) == 8
    assert t.mul_price(64, 64) == 64 and t.mul_price(31, 31) == 16
    pl = t.field_price_list(-128)
    # both families meet 2^-128 with a 24-limb challenge field: Goldilocks^3 (192 bits), BabyBear^6 / M31^6 (186 bits)
    assert pl["goldilocks"]["extension_degree"] == 3 and pl["babybear"]["extension_degree"] == 6 and pl["m31"]["extension_degree"] == 6
    assert pl["goldilocks"]["challenge_limbs"] == pl["babybear"]["challenge_limbs"] == 24
    assert pl["goldilocks"]["challenge_x_challenge_macs"] == pl["babybear"]["challenge_x_challenge_macs"] == 576
    assert pl["goldilocks"]["trace_x_challenge_macs"] == 192 and pl["babybear"]["trace_x_challenge_macs"] == 96
    pl80 = t.field_price_list(-80)
    assert pl80["goldilocks"]["extension_degree"] == 2 and pl80["babybear"]["extension_degree"] == 4
    assert pl80["goldilocks"]["challenge_limbs"] == pl80["m31"]["challenge_limbs"] == 16
    # the SIMT price that kept Goldilocks: degree-6 31-bit at half cost = 8.5 vs 6 Goldilocks equivalents
    assert pl["babybear"]["simt_ext_mult_goldilocks_equiv"] == 8.5 and pl["goldilocks"]["simt_ext_mult_goldilocks_equiv"] == 6


def test_int32_accumulator_bound():
    assert t.max_inner_length(8) == 33_025                                         # TensorZKP eq. 23; they chunk at 32,768
    assert t.max_inner_length(16) == 0                                              # no 16-bit limb option
    assert t.max_inner_length(12) == 128 and t.max_inner_length(11) == 512
    assert t.LIMB_OPTIONS["uint8_int32"]["viable"] and not t.LIMB_OPTIONS["int16_int32"]["viable"]
    assert t.LIMB_OPTIONS["bf16_fp32"]["max_inner_length"] == 258 and not t.LIMB_OPTIONS["bf16_fp32"]["viable"]
    assert t.reduction_words(24) == 576


def test_sumcheck_atom_matches_tensorzkp():
    """Degree 2, two tables, 32 limbs: 20 dl^2 MACs per element and no Hadamards (Algorithm 1)."""
    sc = t.sumcheck_macs(2**25, 2, 32)
    assert sc["matmul_macs"] == 2**25 * 20 * 1024 and sc["hadamards"] == 0
    d3 = t.sumcheck_macs(1000, 3, 24)
    assert d3["matmul_macs"] == 1000 * 576 * 32 and d3["hadamards"] == 4000


def test_tensorzkp_calibration():
    cal = t.tensorzkp_calibration()
    mods = cal["modules"]
    assert 0.03 < mods["inner_product"]["utilisation_of_int8_peak"] < 0.05           # 0.85 ms: HBM-bound, ~4% of peak
    assert 0.6 < mods["inner_product"]["hbm_fraction"] < 0.9
    assert 0.16 < mods["sumcheck_degree2"]["utilisation_of_int8_peak"] < 0.18        # 4.04 ms: 17%
    assert abs(cal["realistic_utilisation"]["matmul_dense"] - UTIL) < 1e-3
    assert 0.05 < cal["realistic_utilisation"]["matmul_gather"] < 0.12             # encoder: gather-bound, ~9% of peak
    assert 150e9 < cal["merkle_implied_bytes_per_s"] < 250e9
    hp = cal["our_relation_as_hyperplonk"]
    assert hp["gates_per_vu"] == 34370 + 13059 + 29379 == 76808
    assert 4.5e7 < hp["overhead_end_to_end"] < 5.5e7                                 # the ~5e7 estimate checks out
    assert 3.5e6 < hp["overhead_zero_check_only"] < 4.5e6                            # and the ~4e6 for the sumcheck part


@pytest.mark.parametrize("field,d,macs_lo,macs_hi,committed", [
    ("goldilocks", 3, 1.0e10, 1.05e10, 181_353_221),
    ("babybear", 6, 7.4e9, 7.6e9, None),
])
def test_candidate_a_rows_2_128(field, d, macs_lo, macs_hi, committed):
    r = t.design_row(t.DESIGNS["A"], t.FIELDS[field], -128, utilisations=(1.0, UTIL))
    assert r["extension_degree"] == d and r["rounds.sequential_depth"] == 381
    assert macs_lo < r["limb_macs_per_vu"] < macs_hi
    if committed:
        assert r["census.committed_elements"] == committed                           # the fields lane's baseline, reproduced
    comps = r["limb_macs_by_component_per_vu"]
    assert comps["gkr_checker"] > comps["logup_fractional_gkr"] > comps["ligero_row_combination"]
    if field == "goldilocks":
        assert comps["gkr_checker"] > comps["encoding"] > comps["logup_fractional_gkr"]   # 64 MACs per base product
    else:
        assert comps["encoding"] < comps["logup_fractional_gkr"]                           # 16: encoding stops mattering
    h = r["times"][f"H100@{UTIL:g}"]
    assert 0.4 < h["matmul_share_of_time"] < 0.6
    assert abs(h["overhead.vs_native_peak"] - contract.overhead(h["seconds_per_vu"], 1536)) < 1e-6


def test_ranking_and_field_verdict_2_128():
    rows = {(dn, fn): t.design_row(t.DESIGNS[dn], t.FIELDS[fn], -128, utilisations=(1.0, UTIL))
            for dn in t.DESIGNS for fn in t.FIELDS}
    ovh = {k: r["times"][f"H100@{UTIL:g}"]["overhead.vs_native_peak"] for k, r in rows.items()}
    # A designs: ~1e7; B: ~1.4-2.3e7; every 31-bit row beats its Goldilocks row on tensor cores
    assert 9e6 < ovh[("A", "goldilocks")] < 1.6e7 and 2.0e7 < ovh[("B", "goldilocks")] < 2.6e7
    for dn in t.DESIGNS:
        assert ovh[(dn, "babybear")] < ovh[(dn, "goldilocks")] and ovh[(dn, "m31")] < ovh[(dn, "goldilocks")]
    # ... by a small margin for A (the shared 24-limb challenge field) and a larger one for B (encoding-dominated)
    assert 1.1 < ovh[("A", "goldilocks")] / ovh[("A", "babybear")] < 1.3
    assert 1.4 < ovh[("B", "goldilocks")] / ovh[("B", "babybear")] < 1.9
    # committed bytes: Goldilocks A 354 kB per VU, B 1.5 MB; 31-bit halves A
    assert 3.4e5 < rows[("A", "goldilocks")]["committed_bytes_per_vu"] < 3.7e5
    assert 1.4e6 < rows[("B", "goldilocks")]["committed_bytes_per_vu"] < 1.6e6
    assert 1.8 < rows[("A", "goldilocks")]["committed_bytes_per_vu"] / rows[("A", "babybear")]["committed_bytes_per_vu"] < 2.0
    hf = rows[("A-hintfree-depth630", "goldilocks")]
    assert hf["rounds.sequential_depth"] == 630 and hf["limb_macs_per_vu"] < rows[("A", "goldilocks")]["limb_macs_per_vu"]


def test_vole_row_is_designated_verifier_and_simt_bound():
    v = t.price_vole(-128)
    assert v["zk_mode"] == "designated-verifier" and v["mac_extension"] == 3 and v["mac_limbs"] == 24
    pv = v["per_vu"]
    assert pv["vole_correlations"] == 83_434 and pv["mul_checks"] == 63_749
    assert pv["matmul_limb_macs_batch_check"] == 2 * 63_749 * 576
    assert v["per_correlation"]["lpn_primal_d10_limb_macs"] == 640 and v["per_correlation"]["lpn_dual_dense_macs_F_p"] == 589_760
    h = v["times"]["H100@0.17"]
    assert h["matmul_share_of_time"] < 0.3                                           # Hadamards on SIMT dominate
    assert 1e6 < h["overhead.vs_native_peak"] < 5e6                                 # PRG (GGM) and Hadamards, not matmul
    assert h["overhead_if_dual_dense_lpn"] > 1e8                                     # dense dual LPN as a matmul is out


def test_report_is_fast_and_complete():
    rep = t.report()
    assert len(rep["rows"]) == 2 * 3 * 3 + 4
    assert set(rep["field_verdict"]) == {f"{d}@2^{tg}" for d in t.DESIGNS for tg in t.TARGETS}
    assert rep["ranking_2^-128_H100_realistic"][0]["design"].startswith("A")
    assert len(rep["uncertainties"]) >= 3 and rep["vole"]["-128"]["zk_mode"] == "designated-verifier"
    for opt in rep["limb_options"].values():
        assert "max_inner_length" in opt and "viable" in opt


def test_the_default_zk_level_is_what_tensor_md_priced_and_its_pads_are_small():
    """The default level (``zk="hvzk"``) reproduces tensor.md's rows (up to the PRG bucket, < 1%): the accountant's rows
    and pads already contained the honest-verifier layer; its pads over ``zk="none"`` are under 50k elements; an unknown
    level is refused."""
    for dn, field, pinned in (("A", "babybear", 1.07e7), ("B", "babybear", 1.43e7), ("A", "goldilocks", 1.25e7)):
        p = t.price_design(t.DESIGNS[dn], t.FIELDS[field], -128)
        tm = p.times(t.GPUS["H100"], "H100", 0.172, 50e9, fs=t.FIELDS[field])
        assert abs(tm["overhead.vs_native_peak"] / pinned - 1) < 0.01
        assert 0 < tm["t.prg"] < 0.01 * tm["t.total"]
    pa = t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="none")
    pb = t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="hvzk")
    assert 0 < pb.committed_elements - pa.committed_elements < 50_000
    with pytest.raises(ValueError):
        t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="perfect")
