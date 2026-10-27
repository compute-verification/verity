"""verity/ml/definitions/numerics/twins.py — every twin against the gate evaluator on synthetic small statics (laptop-fast), and
the `--check` evidence file."""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from verity.ml.definitions.numerics import kernel_registry as KR
from verity.ml.definitions.numerics import twins as TW
from verity_vllm.pipeline import cli
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind

FAMILIES = sorted(TW.FAMILIES)

# every specialization present in the two logs the sweep covers (resolve_log `by_spec`): SmolLM2-135M (M1) and
# Qwen2.5-1.5B (M6) — a twin must claim each (the CAPTURE_BIG test below evaluates M1's for real)
M1_STATICS = {
    "Gemm_v1": [{"K": 576, "N": 960}, {"K": 576, "N": 576}, {"K": 576, "N": 3072}, {"K": 1536, "N": 576}, {"K": 576, "N": 49152}],
    "RMSNormTriton_v1": [{"N": 576, "EPS": 1e-05}], "RMSNormFusedCuda_v2": [{"N": 576, "EPS": 1e-05}],
    "RoPE_v1": [{"NHEADS": 9, "D": 64}, {"NHEADS": 3, "D": 64}], "SiluMul_v1": [{"I": 1536}],
    "Embedding_v1": [{"V": 49152, "H": 576}], "TokenSelect_v1": [{"V": 49152}],
    "Attention_v3": [{"T": T, "NH": 9, "KVH": 3, "D": 64, "BN": 128} for T in range(1, 48)],
}
M6_STATICS = {
    "Gemm_v1": [{"K": 1536, "N": 2048}, {"K": 1536, "N": 1536}, {"K": 1536, "N": 17920}, {"K": 8960, "N": 1536}, {"K": 1536, "N": 151936}],
    "BiasAdd_v1": [{"N": 2048}], "RMSNormTriton_v1": [{"N": 1536, "EPS": 1e-06}], "RMSNormFusedCuda_v2": [{"N": 1536, "EPS": 1e-06}],
    "RoPE_v1": [{"NHEADS": 12, "D": 128}, {"NHEADS": 2, "D": 128}], "SiluMul_v1": [{"I": 8960}],
    "Embedding_v1": [{"V": 151936, "H": 1536}], "TokenSelect_v1": [{"V": 151936}],
    "Attention_v3": [{"T": T, "NH": 12, "KVH": 2, "D": 128, "BN": 64} for T in range(1, 48)],
}


# ---------------------------------------------------------------------------------------------------------
# twin == gates on synthetic small statics
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", FAMILIES)
def test_twin_matches_gates_on_random_small_statics(base):
    cls = TW.FAMILIES[base]
    rng = np.random.default_rng(zlib.crc32(base.encode()))
    rnd = TW.check_instances(cls, cls.small_statics, rng, 8, TW.RANDOM_KINDS)
    assert rnd["status"] == "pass" and rnd["mismatches"] == [] and rnd["n"] >= 6, rnd
    assert len(rnd["statics"]) >= 2, rnd["statics"]                       # several structural variants were drawn
    edge = TW.check_instances(cls, cls.small_statics, rng, 8, TW.EDGE_KINDS)
    assert edge["mismatches"] == [], edge
    assert edge["n"] >= 2, edge                                            # NaN / inf / subnormal / boundary words
    if base == "Attention_v3":                                             # declines (not mismatches) on non-finite words
        assert edge["n_declined"] > 0 and all(k.startswith("non-finite") for k in edge["declined"]), edge["declined"]
    else:
        assert edge["n_declined"] == 0


@pytest.mark.parametrize("model,statics", [("m1", M1_STATICS), ("m6", M6_STATICS)])
def test_twins_cover_every_logged_specialization(model, statics):
    seen = set()
    for base, lst in statics.items():
        for st in lst:
            spec = bind(TW.DEFINITIONS[base], **st)
            assert KR.covers(spec, TW.KERNEL) is None, (spec.id, KR.covers(spec, TW.KERNEL))
            assert spec.base_id == base and TW.FAMILIES[base].twin_id
            seen.add(TW.FAMILIES[base].twin_id)
    assert seen == {TW.FAMILIES[b].twin_id for b in statics}


def test_twins_decline_statics_outside_the_body():
    assert "K=24" in KR.covers(bind(TW.DEFINITIONS["Gemm_v1"], K=24, N=2), TW.KERNEL)      # K not a multiple of 16
    assert KR.covers(bind(TW.DEFINITIONS["Gemm_v1"], K=32, N=2), TW.KERNEL) is None


def test_gemm_twin_falls_back_to_the_primitives_on_non_finite_words():
    """`tc_model.cpp` has no inf/NaN extension: coordinates with a non-finite operand or result take the registered
    primitives' path and still equal the gates."""
    spec = bind(TW.DEFINITIONS["Gemm_v1"], K=32, N=3)
    tw = KR.instance(spec, TW.KERNEL)
    rng = np.random.default_rng(1)
    x = TW.rand_bf16(rng, 32, "gauss")
    w = TW.rand_bf16(rng, 3 * 32, "gauss").reshape(3, 32)
    w[0, 5] = 0x7F80                    # +inf in coordinate 0
    w[1, 7] = 0x7FC1                    # NaN payload in coordinate 1
    w[2, :] = 0x7F00                    # huge finite words in coordinate 2 (the chain may or may not overflow)
    args = [x.astype(np.uint64), w.reshape(-1).astype(np.uint64)]
    got, expect = tw(*args, **spec.statics), np.array(evaluate(spec, *(a.tolist() for a in args)), dtype=np.uint64)
    assert np.array_equal(got, expect), (got, expect)
    assert TW._nonfinite16(got.astype(np.uint16))[:2].all()


def test_attention_twin_declines_non_finite_operands():
    spec = bind(TW.DEFINITIONS["Attention_v3"], T=3, NH=1, KVH=1, D=16, BN=128)
    tw = KR.instance(spec, TW.KERNEL)
    rng = np.random.default_rng(2)
    args = TW.AttentionTwin.random_args(spec.statics, rng, "gauss")
    tw(*args, **spec.statics)                                             # finite: claimed
    args[1][0] = 0x7FC0                                                   # a NaN key word
    with pytest.raises(KR.Declined):
        tw(*args, **spec.statics)


# ---------------------------------------------------------------------------------------------------------
# the evidence file
# ---------------------------------------------------------------------------------------------------------


def test_check_writes_the_evidence_schema(tmp_path):
    out = tmp_path / "twins.json"
    assert cli.main(["twins", "--check", "--n-random", "4", "--n-edge", "4", "--seed", "7", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == TW.SCHEMA == "verity-gen/twins/v1" and rep["status"] == "pass" and rep["log"] is None
    assert set(rep["families"]) == {cls.twin_id for cls in TW.FAMILIES.values()}
    for fam in rep["families"].values():
        assert fam["status"] == "pass" and fam["n_random"] >= 3 and fam["mismatches"] == [] and fam["words"] > 0
        assert {"base_id", "n_random", "n_edge", "n_declined", "statics", "status", "seconds"} <= set(fam)
    assert set(rep["libraries"]) == {"tc_model", "fa2_model", "rms_triton_model"}
