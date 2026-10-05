"""Lattice scout cost model (``explore.lattice``): the witness-bit census, the exact-embedding statement, the MSIS
heuristic, the Ajtai GEMM atom and the design rows are pinned.  Everything here is a property of the stated cost model
(and the cited papers' numbers), not a measurement."""

from __future__ import annotations

import pytest
from verity_numerical.explore import lattice as lat
from verity_numerical.explore import tensor as tc
from verity_numerical.security import census


@pytest.fixture(scope="module")
def vu():
    return census.census_vu(1536)


def test_witness_bits(vu):
    wb = lat.witness_bits(vu, True)
    assert wb.hints == 129_665                                  # exact hint widths from the census
    assert wb.inputs == 2 * 1536 * 16 + 16
    assert wb.lookup_outputs == 5952 * lat.LOOKUP_OUTPUT_BITS
    assert wb.query_wires == 13_536 * lat.QUERY_WIRE_BITS
    assert wb.total_a == 707_217 and wb.digits(16, "A") == 44_202
    assert lat.witness_bits(vu, False).digits(16, "A") == 17_130
    assert wb.digits(16, "B") == 110_442


def test_exact_embedding(vu):
    e = lat.exact_embedding(vu)
    assert e["widest_hint_bits"] == 25 and e["widest_product_bits"] == 50 and e["q_bits_needed"] == 54
    assert e["q_64_exact"] and not e["q_32_exact"]
    assert e["range_predicates_per_vu"] == 27_267 and not e["norm_bound_replaces_range_checks"]


def test_msis_heuristic_monotone_and_anchored():
    b4, b16, b40 = (lat.msis_output_bits(64, b) for b in (4, 16, 40))
    assert b4 < b16 < b40                                        # larger digits need a larger output
    assert lat.msis_output_bits(32, 0) == pytest.approx(2048)      # anchored: Akita Table 5 rank-1 d=64 q=2^32 binary map = 2048 bits (147 q-bits)
    assert 3 < lat.ANCHOR_SCALE < 4                               # the raw BKZ model under-shoots the anchor by ~3.6x
    m = lat.MSIS["q64_chunk16"]
    assert m.kappa_d % 64 == 0 and 512 <= m.kappa_d <= 2048        # 896 = 14 x 64 at the anchored heuristic
    with pytest.raises(ValueError):
        lat.msis_output_bits(0, 4)


def test_ajtai_gemm_atom():
    m = lat.MSIS["q64_chunk16"]
    c = lat.ajtai_commit_macs(44_202, m)
    assert c["per_digit"] == m.kappa_d * 8 * 2                   # 8 limbs of q x 2 limbs of a 16-bit chunk
    assert c["limb_macs"] == 44_202 * c["per_digit"]
    r = lat.response_digits(44_202, 16, 64)
    assert r["coefficient_bits"] == 16 + 6 + lat.CHALLENGE_OPNORM_BITS and r["digits_per_coefficient"] == 7
    assert r["coefficients"] == -(-44_202 // 64)


def test_range_check_scales_with_alphabet():
    a = lat.digit_range_check_macs(1000, 4, 24)
    b = lat.digit_range_check_macs(1000, 8, 24)
    assert a["vanishing_degree"] == 8 and b["vanishing_degree"] == 128 and b["limb_macs"] > 10 * a["limb_macs"]


def test_design_rows_pinned():
    a32 = lat.design_lattice_pcs("q32_chunk16", "A")
    a64 = lat.design_lattice_pcs("q64_chunk16", "A")
    ref = tc.design_row(tc.DESIGNS["A"], tc.FIELDS["babybear"], -128, utilisations=(1.0, lat.UTIL))["times"]["H100@0.172"]["overhead.vs_native_peak"]
    o32 = a32["times_H100@0.172"]["overhead.vs_native_peak"]
    o64 = a64["times_H100@0.172"]["overhead.vs_native_peak"]
    assert a32["sumcheck_field"] == "babybear" and a64["sumcheck_field"] == "goldilocks"
    assert a32["hash_bytes_per_vu"] == 0.0
    # the swap is break-even (q32) to -25% (q64) against A at 50 GB/s hashing: inside the hash-rate uncertainty, never a clear win
    assert 0.9 < o32 / ref < 1.1
    ref64 = a64["reference_row"]["overhead_H100@0.172"]                    # A on Goldilocks^3 (the field the 64-bit q forces)
    assert 0.65 < o64 / ref64 < 0.85 and 0.8 < o64 / ref < 0.95
    assert a32["limb_macs_by_component_per_vu"]["gkr_checker"] > 5e9          # the checker sumcheck is untouched by the swap
    assert a32["limb_macs_by_component_per_vu"]["digit_range_check"] > a32["limb_macs_by_component_per_vu"]["ajtai_commit"]
    k16 = lat.design_lattice_pcs("q32_chunk16", "A", fold_arity=16)["times_H100@0.172"]["overhead.vs_native_peak"]
    k256 = lat.design_lattice_pcs("q32_chunk16", "A", fold_arity=256)["times_H100@0.172"]["overhead.vs_native_peak"]
    assert k16 > ref > k256                                                    # fold arity decides the sign


def test_labrador_direct_is_not_a_candidate():
    lab = lat.labrador_direct()
    assert lab["polys_per_vu"] == 63_751 and lab["r"] == round(lab["N_polys"] ** (1 / 3))
    assert lab["overhead_H100@0.172"] > 1e8                                    # 40x+ above A at one wire per ring element
    assert lab["limb_macs_by_component_per_vu"]["h_ij_aggregation"] > lab["limb_macs_by_component_per_vu"]["ajtai_commit"]


def test_report_shape():
    rep = lat.report()
    assert rep["verdict"]["complete_alternative_backend"].startswith("NO")
    assert len(rep["rows"]) == 7 and len(rep["uncertainties"]) == 3
    for r in rep["rows"]:
        assert r["B"] == 4096 and r["K"] == 1536 and r["target_log2"] == -128
    assert rep["literature"]["akita"]["prover_seconds_measured"]["2^30 x 32-bit coefficients, 8 threads"]["commit"] == 3.46
    assert rep["literature"]["latticefold_plus"]["prover_seconds_measured"] is None
