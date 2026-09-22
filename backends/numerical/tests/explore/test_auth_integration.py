"""The authentication-integration cost model (``explore.auth_integration``) and the ``authentication`` knob of
``explore.tensor``: the bench sharing structure, the gadget constants, the per-option deltas and the complete
numbers are pinned.  Every number is a property of the stated cost model, not a measurement."""

from __future__ import annotations

import pytest
from verity_numerical.explore import auth_integration as ai
from verity_numerical.explore import tensor as t

UTIL = 0.172
FS = t.FIELDS["babybear"]
HR = t.HASH_RATES["blake3_simt"]["bytes_per_s"]


def _ovh(design: str, **kw) -> tuple[float, dict]:
    p = t.price_design(t.DESIGNS[design], FS, -128, **kw)
    tm = p.times(t.GPUS["H100"], "H100", UTIL, HR, fs=FS)
    return tm["overhead.vs_native_peak"], tm


def test_bench_sharing_structure_from_the_index():
    sh = ai.SHARINGS["bench"]
    assert (sh.x_leaves, sh.y_leaves, sh.w_matrices) == (28, 28, 18)
    assert sh.w_rows == 3079                                 # 2 598 distinct (op, j) rows scaled x 4096/3456
    assert sh.w_chunks[1] == (3079, 3079) and sh.w_chunks[4096] == (91, 325253)
    assert sh.touched_rows(64) == 28 + 28 + 84651 and sh.touched_chunks(64) == 28 + 28 + 1323
    ds = ai.SHARINGS["decode_step"]
    assert (ds.x_leaves, ds.w_rows, ds.y_leaves) == (1, 4096, 1)


def test_gadget_constants():
    g = ai.GADGETS["sha256_lookup_air"]
    assert (g.gates_a, g.lookups_a, g.committed_a) == (4384, 1728, 4384) and g.committed_b(6) == 4384 + 1728 * 6
    assert ai.GADGETS["sha256_r1cs"].gates_a == 26_170 and ai.GADGETS["sha256_r1cs"].committed_b(6) == 78_510
    p16, p24 = ai.GADGETS["poseidon2_bb16"], ai.GADGETS["poseidon2_bb24"]
    assert p16.gates_a == 4 * (8 * 16 + 13) + 21 * 16 == 900 and p16.payload_bytes == 16
    assert p24.gates_a == 4 * (8 * 24 + 21) + 29 * 24 == 1548 and p24.payload_bytes == 32


def test_v1_compressions_per_vu_after_bench_sharing():
    total, by = ai._v1_compressions(ai.SHARINGS["bench"], ("x", "w", "y"))
    # W row (50 leaf + 40 path compressions) x 3079/4096; x and y rows shared by ~146 VUs each
    assert round(by["w"], 2) == round(90 * 3079 / 4096, 2) and by["x"] < 0.5 and by["y"] < 0.6
    assert 68 < total < 69
    private_only, _ = ai._v1_compressions(ai.SHARINGS["bench"], ("x", "y"))
    assert 1.0 < private_only < 1.1                           # weights public: ~1 compression per VU


@pytest.mark.parametrize("design,opt,wp,lo,hi", [
    ("A", "in_circuit_sha256", False, 2.3, 2.9),
    ("A", "in_circuit_sha256", True, 0.02, 0.06),
    ("A", "leaf_v2_pcs_hash", False, 0.20, 0.35),
    ("A", "algebraic", False, 0.0, 0.002),
    ("B", "in_circuit_sha256", False, 3.0, 3.8),
    ("B", "in_circuit_sha256", True, 0.03, 0.08),
    ("B", "leaf_v2_pcs_hash", False, 0.6, 1.0),
    ("B", "algebraic", False, 0.0, 0.002),
])
def test_complete_overhead_deltas_vs_malicious_zk(design, opt, wp, lo, hi):
    base, tb = _ovh(design, zk="malicious")
    comp, tc = _ovh(design, zk="malicious", authentication=opt, weights_public=wp)
    rel = tc["t.total"] / tb["t.total"] - 1
    assert lo <= rel <= hi, rel
    assert comp >= base


def test_algebraic_is_free_for_the_prover_and_within_20_percent_of_numerical_only():
    for design, zk_delta in (("A", 0.02), ("B", 0.08)):
        _nonzk, t0 = _ovh(design, zk="none")
        _comp, tc = _ovh(design, zk="malicious", authentication="algebraic")
        assert tc["t.total"] / t0["t.total"] - 1 < zk_delta                 # the ZK layer is the whole delta
        assert tc["t.total"] / t0["t.total"] - 1 < 0.20                     # breakthrough criterion
    p = t.price_design(t.DESIGNS["A"], FS, -128, zk="malicious", authentication="algebraic")
    d = p.auth_delta
    assert d.depth_delta == 0 and d.committed_elements == 0 and d.prover_hash_bytes == 0
    assert 4.4e4 < d.proof_bytes / 4096 < 4.5e4                             # t=191 columns of 1 379 chunks of 64 rows + paths
    assert d.matmul_macs / 4096 < 3e6                                        # < 0.05% of A's 7.5e9 limb MACs per VU


def test_in_circuit_sha256_adds_depth_and_committed_elements_for_a():
    p = t.price_design(t.DESIGNS["A"], FS, -128, zk="malicious", authentication="in_circuit_sha256")
    assert p.depth == 382 + 2 * 128 and p.auth_delta.committed_elements > 0 and "auth_gkr" in p.matmul
    q = t.price_design(t.DESIGNS["B"], FS, -128, zk="malicious", authentication="in_circuit_sha256")
    assert q.depth == 4 and q.auth_delta.products_b > 0


def test_verifier_openings_are_not_private_and_cost_the_prover_nothing():
    d = ai.delta("verifier_openings", "A", 6, weights_public=True)
    assert d.committed_elements == 0 and d.gkr_elements == 0 and d.proof_bytes > 0 and d.verifier_hash_bytes > 0
    assert ai.PRIVACY["verifier_openings"]["verdict"].startswith("NOT")


def test_soundness_accounting_unchanged_by_the_algebraic_option():
    s = ai.soundness_with_auth()
    for cand in ("A", "B"):
        assert s[cand]["t"] == 191 and s[cand]["meets_target"]
        assert abs(s[cand]["delta_bits"]) < 0.01 and s[cand]["soundness_log2_after"] <= -129.4
        assert all(term["log2"] < -160 for term in s[cand]["added_terms"])


def test_capture_cost_per_decode_step():
    c = ai.capture_cost()
    assert c["tableau_rows_per_token"] == 392
    act = c["activations_per_step"]
    assert 0.03 < act["encode_seconds_h100_0.172"] < 0.06 and act["hash_seconds_sha256_4090"] < 0.02
    w = c["weights_per_commitment"]
    assert 0.3 < w["encode_seconds_h100_0.172"] < 0.6 and w["hash_seconds_sha256_4090"] < 0.2


def test_report_shape_and_breakthrough_flag():
    rep = ai.report()
    assert rep["best_private_per_design"]["A"]["option"] == "algebraic" and rep["best_private_per_design"]["B"]["option"] == "algebraic"
    assert rep["breakthrough_criterion"]["met"] == {"A": True, "B": True}
    assert {r["option"] for r in rep["options"]} == set(ai.OPTIONS)
    assert [row["R"] for row in rep["algebraic_chunk_size_sweep"]] == [1, 16, 64, 256, 4096]
    best_r = min(rep["algebraic_chunk_size_sweep"], key=lambda r: r["proof_bytes_per_vu"])
    assert best_r["R"] == 64


def test_unknown_knob_rejected():
    with pytest.raises(ValueError):
        t.price_design(t.DESIGNS["A"], FS, -128, authentication="poseidon")
