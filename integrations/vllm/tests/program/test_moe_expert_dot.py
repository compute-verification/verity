"""The fused-MoE expert GEMMs with the target's k16 step (`registry.moe` `MoeExpertGemm_v2` / `MoeExpertGemmW_v2{E,K,N,DOT}`, bound
through `frontend.targets.moe_expert_dot`): only the families whose fused_moe_kernel correspondence ran bind them (cc 9.0 and 12.0;
cc 8.x keeps the `_v1` ids), with the Ampere step they are the `_v1` Definitions, with the Hopper step they are that step's ascending
chain (and the step is observable), and the vocabulary, the replay rows, the VU export and GP-01 read the same families."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.kernels.tc import f2fp_bf16_batch, group_sum_total_batch
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.circuits.silicon.models import HOPPER_BF16_M16N8K16
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import moe as M
from verity.ml.definitions.registry import prims as P
from verity_vllm.program.frontend import targets as T

E, K, N = 3, 64, 5


def _tp(cc):
    return TargetProfile(compute_capability=cc)


def _bf16(rng, n: int, wide: bool = False) -> np.ndarray:
    v = rng.standard_normal(n).astype(np.float32)
    if wide:
        v *= np.exp2(rng.integers(-12, 13, n)).astype(np.float32)
    return (v.view(np.uint32) >> 16).astype(np.uint16)


def test_only_the_measured_families_bind_the_dot_forms():
    for cc in ((8, 0), (8, 9)):
        assert T.moe_expert_dot(_tp(cc)) is None
    for cc in ((9, 0), (12, 0)):
        assert T.moe_expert_dot(_tp(cc)) is HopperBF16WgmmaDot16
    with pytest.raises(T.UnregisteredTarget):
        T.moe_expert_dot(_tp((10, 0)))
    assert T.moe_expert_gemm_spec(_tp((8, 9)), 64, 2048, 2048, routed=False).id == "MoeExpertGemm_v1{E=64,K=2048,N=2048}"
    assert T.moe_expert_gemm_spec(_tp((8, 9)), 64, 1024, 2048, routed=True).id == "MoeExpertGemmW_v1{E=64,K=1024,N=2048}"
    for cc in ((9, 0), (12, 0)):
        assert T.moe_expert_gemm_spec(_tp(cc), 64, 2048, 2048, routed=False).id == \
            "MoeExpertGemm_v2{E=64,K=2048,N=2048,DOT=HopperBF16WgmmaDot16_v1}"
        assert T.moe_expert_gemm_spec(_tp(cc), 64, 1024, 2048, routed=True).id == \
            "MoeExpertGemmW_v2{E=64,K=1024,N=2048,DOT=HopperBF16WgmmaDot16_v1}"


def test_a_tuned_bf16_tile_is_modelled_on_cc90_at_any_k16_multiple():
    tuned = TargetProfile(compute_capability=(9, 0), batch_invariant=False)
    for kb in (64, 128, 256):
        assert T.moe_tuned_tile_refusal(tuned, kb) is None
    assert T.moe_tuned_tile_refusal(TargetProfile(compute_capability=(9, 0), batch_invariant=True), 64) is None
    assert "multiple of the k16 step" in T.moe_tuned_tile_refusal(tuned, 40)
    assert "cc 9.0 only" in T.moe_tuned_tile_refusal(TargetProfile(compute_capability=(12, 0), batch_invariant=False), 128)


def test_the_dot_forms_with_the_ampere_step_are_the_v1_definitions():
    rng = np.random.default_rng(3)
    x, w, g = _bf16(rng, K).tolist(), _bf16(rng, E * N * K, wide=True).tolist(), [0x3F400000]
    for e in range(E):
        assert list(evaluate(bind(M.MoeExpertGemmV2, E=E, K=K, N=N, DOT=P.AmpereBF16TcDot16), x, [e], w)) == \
            list(evaluate(bind(M.MoeExpertGemm, E=E, K=K, N=N), x, [e], w))
        assert list(evaluate(bind(M.MoeExpertGemmWV2, E=E, K=K, N=N, DOT=P.AmpereBF16TcDot16), x, [e], g, w)) == \
            list(evaluate(bind(M.MoeExpertGemmW, E=E, K=K, N=N), x, [e], g, w))


def _chain(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    acc = np.zeros(rows.shape[0], np.uint32)
    X = np.broadcast_to(x, rows.shape)
    for s in range(rows.shape[1] // 16):
        acc, _ = group_sum_total_batch(HOPPER_BF16_M16N8K16, acc, X[:, 16 * s:16 * s + 16], rows[:, 16 * s:16 * s + 16])
    return acc.astype(np.uint32)


def test_the_dot_forms_with_the_sm120_step_are_its_ascending_chain():
    rng = np.random.default_rng(5)
    x, w = _bf16(rng, K, wide=True), _bf16(rng, E * N * K, wide=True)
    g = np.array([0.625], np.float32)
    for e in range(E):
        rows = w.reshape(E, N, K)[e]
        acc = _chain(x, rows)
        up = f2fp_bf16_batch(acc)[0].astype(np.uint64).tolist()
        down = f2fp_bf16_batch((acc.view(np.float32) * g).astype(np.float32).view(np.uint32))[0].astype(np.uint64).tolist()
        assert list(evaluate(bind(M.MoeExpertGemmV2, E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16), x.tolist(), [e], w.tolist())) == up
        assert list(evaluate(bind(M.MoeExpertGemmWV2, E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16), x.tolist(), [e], [int(g.view(np.uint32)[0])],
                             w.tolist())) == down
    rows = w.reshape(E * N, K)
    ampere = [evaluate(bind(b1.DotBf16, K=K), x.tolist(), r.tolist())[0] for r in rows]
    assert ampere != _chain(x, rows).tolist(), "the two k16 steps give the same accumulators: the inputs do not tell them apart"


def test_vocabulary_replay_rows_and_vu_export_read_the_dot_static():
    from verity_vllm.pipeline import vu_export as VX
    from verity_vllm.program.frontend.rules.vocab import PROFILE_B1_EAGER_V3 as V
    from verity.ml.definitions.numerics import rows

    assert V.kind("moe_expert_gemm_dot", E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16).id == f"MoeExpertGemm_v2{{E={E},K={K},N={N},DOT=HopperBF16WgmmaDot16_v1}}"
    assert V.kind("moe_expert_gemm_w_dot", E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16).id.startswith("MoeExpertGemmW_v2{")
    assert V.kind("moe_expert_gemm", E=E, K=K, N=N).id == f"MoeExpertGemm_v1{{E={E},K={K},N={N}}}"
    rng = np.random.default_rng(7)
    x, w = _bf16(rng, K), _bf16(rng, E * N * K)
    st = {"E": E, "K": K, "N": N, "DOT": "HopperBF16WgmmaDot16_v1"}
    got = rows.ROWS["MoeExpertGemm_v2"].fn(st, [x, np.array([1], np.uint32), w])["up"]
    assert got.astype(np.uint64).tolist() == list(evaluate(bind(M.MoeExpertGemmV2, E=E, K=K, N=N, DOT=HopperBF16WgmmaDot16), x.tolist(), [1],
                                                           w.tolist()))
    assert VX.template_of(f"MoeExpertGemm_v2{{E={E},K={K},N={N},DOT=HopperBF16WgmmaDot16_v1}}")[1]["dot"] == "HopperBF16WgmmaDot16_v1"
    assert VX.template_of(f"MoeExpertGemm_v1{{E={E},K={K},N={N}}}")[1]["dot"] == "AmpereBF16TcDot16_v2"


def test_gp01_reads_an_sm120_component_as_the_declared_indexed_construction(tmp_path):
    from verity_vllm.pipeline import global_program as GP
    hist = {"Gemm_v2{K=2048,N=64,DOT=HopperBF16WgmmaDot16_v1}": 16, "MoeRouterTopKOrdered_v1{E=64,TOPK=8}": 16,
            "MoeExpertGemm_v2{E=64,K=2048,N=2048,DOT=HopperBF16WgmmaDot16_v1}": 128,
            "MoeExpertGemmW_v2{E=64,K=1024,N=2048,DOT=HopperBF16WgmmaDot16_v1}": 128, "MoeSum_v1{TOPK=8,H=2048}": 16}
    d = tmp_path / "c0"
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"ok": True, "spec_histogram": hist,
                                               "target": {"declared": {"compute_capability": [12, 0], "num_sms": 188}}}))
    out = GP.workload_moe_construction({}, {(256, 31): str(d)})
    assert out["refusal"] is None and out["value"] == "indexed-read-ordered" and out["moe_block_present"] is True


def test_the_static_pair_forms_are_the_dot_forms():
    """`verity.ml.circuits.ops.moe`'s `MoePairGemm(W)_v1` (the word view of the padded static MoE form's Boolean pair units,
    `ops.boolean.moe`) is `MoeExpertGemm(W)_v2` at the same statics, on every expert id and on ids outside [0, E), through the
    pair forms' kernels and the registry's reference."""
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.circuits.ops import moe as SM
    from verity.ml.circuits.ops.prims import AmpereBF16TcDot16

    rng = np.random.default_rng(9)
    ids = np.asarray([*range(E), E, 1 << 31, (1 << 32) - 1], np.uint32)
    n = len(ids)
    x = np.stack([_bf16(rng, K, wide=True) for _ in range(n)])
    w = np.stack([_bf16(rng, E * N * K, wide=True) for _ in range(n)]).reshape(n, E, N, K)
    g = np.concatenate([rng.standard_normal(n - 1).astype(np.float32), [np.float32("nan")]]).view(np.uint32)
    for dot in (HopperBF16WgmmaDot16, AmpereBF16TcDot16):
        st = {"E": E, "K": K, "N": N, "DOT": dot}
        assert (evaluate_batch(bind(SM.MoePairGemm, **st), x, ids, w) ==
                evaluate_batch(bind(M.MoeExpertGemmV2, **st), x, ids, w, kernel="reference")).all()
        assert (evaluate_batch(bind(SM.MoePairGemmW, **st), x, ids, g, w) ==
                evaluate_batch(bind(M.MoeExpertGemmWV2, **st), x, ids, g, w, kernel="reference")).all()
