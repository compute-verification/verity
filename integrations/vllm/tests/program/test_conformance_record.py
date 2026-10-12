"""The conformance record and the correspondence records carry the `cos_sin_cache` qualification explicitly and in a
fixed machine-readable form; no summary reads as 'universally exact'."""

from __future__ import annotations

import json
from pathlib import Path

from verity_vllm.correspondence import capture_identities as CO
from verity_vllm.program.registry import boolean_rope, ref_prims  # noqa: F401  (registration: the GPT-J RoPE v1 and v2)
from verity_vllm.program.frontend import conformance as CF

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "fixtures/results/TA1-typed-b1-20260907"
CORR = [RESULTS / "B1/correspondence/corr-A.json", RESULTS / "B1/correspondence/corr-B1.json",
        RESULTS / "B1/correspondence/corr-B2.json", RESULTS / "B0/correspondence/corr-B0.json"]


def test_registry_record_has_the_qualifications():
    rec = CF.record()
    defs = rec["definitions"]
    assert defs["RMSNormFusedCuda_v2"]["status"] == "exact-model-tested"
    assert defs["RMSNormFusedCuda_v1"]["status"] == "superseded"
    assert "captured" in defs["RMSNormFusedCuda_v1"]["divergences"][0]
    for fid in ("RoPEHeadGptJ_v1", "RoPEGptJ_v1"):
        assert defs[fid]["status"] == "superseded"
    for fid in ("RoPEHead_v1", "RoPE_v1"):
        assert defs[fid]["input_qualifications"] == ["weights.cos_sin"]
        assert "GPU only" in defs[fid]["conformance"]
    q = rec["input_qualifications"][0]
    assert q["rebuilt_from"] == CO.COS_SIN_REBUILT_FROM == "vllm._compute_cos_sin_cache (CUDA cosf/sinf), GPU only"
    assert q["cpu_rebuild"]["B1"] == {"differing_words": 17261, "total_words": 16777216}
    assert q["cpu_rebuild"]["B0"] == {"differing_words": 6, "total_words": 524288}
    assert rec["summary"].startswith("not universally exact")
    # other tests may register extra definitions in the same process; the library's own counts are lower bounds
    c = rec["counts"]
    assert c["superseded"] == 3 and c["exact-model-tested"] >= 51 and c["model-untested"] >= 5


def test_the_fp8_block_moe_and_group_quant_forms_are_exact_model_tested():
    """Their texts (`fp8._BLOCK_DOT_CONF`, `fp8_moe.DOT_CONF`, `fp8._QUANT_CONF`) cite P1's cc 12.0 capture, run
    r20261003-220018-3f6a."""
    from verity_vllm.program.registry import fp8, fp8_moe  # noqa: F401  (registers them)
    defs = CF.record()["definitions"]
    block = ("ScaledMmFp8BlockCoordinate", "ScaledMmFp8Block", "ScaledMmFp8BlockCoordinateGivenScale",
             "ScaledMmFp8BlockSharedScaleTile", "ScaledMmFp8BlockSharedScale")
    moe = ("MoeExpertFp8BlockCoordinate", "MoeExpertFp8BlockCoordinateW", "MoeExpertFp8BlockGemm", "MoeExpertFp8BlockGemmW")
    for fid in [f"{f}_v3" for f in block + moe] + ["Fp8GroupScale_v1", "Fp8GroupQuant_v1"]:
        assert defs[fid]["status"] == "exact-model-tested", fid


def test_requalify_is_explicit_and_idempotent(tmp_path):
    rec = {"schema": "verity-ir/correspondence/v0",
           "input_gate_records": {"prompt": {}, "weights": [
               {"leaf_id": CO.COS_SIN_LEAF, "value_identity": {"input_gates": [0, 8], "scalars": 8},
                "rebuilt_from": "vLLM RotaryEmbedding._compute_cos_sin_cache re-run on cuda (derived table, not a checkpoint tensor)",
                "sha256_matches_leaf": True, "cpu_rebuild": {"sha256": "x", "differing_words": 3, "matches_leaf": False}},
               {"leaf_id": "w=layers.0.self_attn.attn._k_scale", "value_identity": None, "not_in_program": "no gate reads it"}]},
           "records": [{"capture_id": "p=x/r=y/t=1/ph=prefill/l=27/op=layers.27.post_attention_layernorm.rmsnorm/port=0",
                        "value_identity": {"function": "RMSNormFusedCuda_v1{N=1536,EPS=1e-06}"},
                        "check": {"status": "mismatch", "mismatches": 1, "words_compared": 1536}}],
           "summary": {"exact": 5, "mismatch": 1, "sampled": 2}}
    p = tmp_path / "corr.json"
    p.write_text(json.dumps(rec))
    d1 = CO.requalify(p)
    d2 = CO.requalify(p)
    assert d1 == d2
    cs = d1["input_gate_records"]["weights"][0]
    assert cs["rebuilt_from"] == "vllm._compute_cos_sin_cache (CUDA cosf/sinf), GPU only"
    assert cs["qualification"]["cpu_rebuild_differing_words"] == 3 and cs["qualification"]["total_words"] == 8
    quals = d1["summary"]["qualifications"]
    assert any(s.startswith("cos_sin_cache Input gates (8 words): rebuilt from vllm._compute_cos_sin_cache (CUDA cosf/sinf), GPU only; CPU rebuild differs in 3 words") for s in quals)
    assert any(s.startswith("1 committed weight leaves are not read") for s in quals)
    assert any(s.startswith("mismatch: l=27/op=layers.27.post_attention_layernorm.rmsnorm/port=0 (RMSNormFusedCuda_v1") for s in quals)
    assert any("coordinate-sampled" in s for s in quals)


def test_ta1_records_are_requalified():
    present = [p for p in CORR if p.exists()]
    if not present:
        return  # the results directory is not part of every checkout
    for p in present:
        d = json.loads(p.read_text())
        cs = [r for r in d["input_gate_records"]["weights"] if r["leaf_id"] == CO.COS_SIN_LEAF][0]
        assert cs["rebuilt_from"] == CO.COS_SIN_REBUILT_FROM, p
        assert cs["qualification"]["cpu_rebuild_differing_words"] == cs["cpu_rebuild"]["differing_words"]
        assert cs["qualification"]["total_words"] == cs["value_identity"]["scalars"]
        quals = d["summary"]["qualifications"]
        assert any(s.startswith("cos_sin_cache Input gates") and "GPU only" in s for s in quals), p
        if d["summary"]["mismatch"]:
            assert any(s.startswith("mismatch:") for s in quals), p
