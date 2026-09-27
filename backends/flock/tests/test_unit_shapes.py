"""The unit-shape census: the RoPE calibration, the size-only composition against the exact single-circuit lowering, and
the splitter on a GEMM coordinate."""
from __future__ import annotations

import pytest

pytest.importorskip("verity_vllm.program.registry.prims")
from verity_vllm.query import word as W  # noqa: E402

from verity_flock import unit_shapes as US  # noqa: E402


def test_rope_pair_calibration():
    """PR #83's RoPE pair unit (C-Flock's grouping): 6,273 rows in an 8,192-row slot, 157,841 private nonzeros."""
    from verity_numerical.bench import lowerings, templates as TM

    low = lowerings.registry("C-Flock").module("rope-head").lowering(TM.subcircuit("rope-head", D=64))
    c = low.layout.counts
    assert c["rows"] == 6273 and US.pow2_slot(c["rows"]) == 8192
    assert c["nnz_a"] + c["nnz_b"] - 2 * sum(low.units.in_widths) - 2 == 157_841
    assert US.layout_rows(sum(low.units.in_widths), c["ands"], sum(low.units.out_widths)) == 6273


@pytest.mark.parametrize("definition, statics", [("Gemm_v1", {"K": 64, "N": 2}), ("RMSNormFusedCuda_v2", {"N": 64, "EPS": 1e-05}),
                                                 ("Attention_v3", {"NH": 2, "KVH": 1, "D": 64, "BN": 128, "T": 20})])
def test_composition_matches_the_exact_lowering(definition, statics):
    fn = W.specialization(definition, statics)
    while (sep := W._separable(fn)) is not None:
        fn = sep[0][0]
    cu = US.CallUnits(fn)
    checked = 0
    for key, cnt, mem in cu.classes():
        u = cu.compose(key, cnt, mem)
        if u.lookups or u.gathers:
            continue
        e = cu.exact(mem)
        assert u.rows == e["rows"] and u.ands == e["ands"]
        assert abs(u.z - e["z"]) <= 0.01 * e["z"]
        checked += 1
    assert checked


def test_pack_splits_a_gemm_coordinate_at_k_steps():
    """K = 256 (16 tensor-core steps) into (2^16, 2^21, 2^8) pieces: 7 steps each, one FP32 carry committed per cut."""
    fn = W.specialization("Gemm_v1", {"K": 256, "N": 1})
    [(u, _k)] = US.spec_units(fn, {})
    assert u.d == 2 * 256 + 1 and u.seq is not None
    p = US.pack(u.seq, 1 << 16, 1 << 21, 1 << 8)
    assert p["pieces"] == 3 and p["extra"] == 2 and not p["oversize"]
    assert all(US.pow2_slot(e["rows"]) <= 1 << 16 and e["d"] <= 256 for e in p["each"])
