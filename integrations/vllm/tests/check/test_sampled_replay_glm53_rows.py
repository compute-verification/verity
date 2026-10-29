"""GLM-5.3 TP8's row evaluators (`kernels.glm53_rows`, and `kernels.cublas_gemm_rows` for the cuBLAS split-K pair) as the sampled
replay calls them.  Every family the per-rank no-evaluator rule named on the TP8 B1 Commit, and on B8 greedy's
replay, has one.  The replay hands a row kernel the statics `spec_statics` parses from
the Program's spec text (`{"f64":...}` floats, `true` / `False`, `{"fn":...}` or bare step ids, quoted kernel names), not the Python
values the kernel self-check binds, so each family is checked here on those spellings against its Definition's word reference."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
from verity.ml.circuits.evaluation.reference import signature_of
from verity.ml.circuits.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
from verity.ml.definitions.numerics import rows
from verity.ml.definitions.numerics.kernel_registry import Declined
from verity.ml.definitions.registry import b1, boolean_rope, cublas_bf16, cuda_rmsnorm, deepgemm, dsa, fp8_moe_experts, hopper
from verity.ml.definitions.registry import ll_bf16, moe, sparse_mla_split
from verity_vllm.query.manifest import format as RM

#: the families FIXWIT's per-rank rule failed GLM-5.3 TP8 B1's Commit on: four plane-fed MoE producers and eleven with no evaluator
TP8_B1_FAMILIES = ("Fp8GroupQuantUe8m0_v1", "MoeExpertFp8GemmW_v1", "MoeExpertFp8Gemm_v1", "MoeRouterSigmoidBias_v1",
                   "RoPEHeadGptJ_v2", "BatchedGemm_v1", "RMSNormFusedCudaVector8_v1", "SparseMlaSplit_v1", "ResidualAdd_v1",
                   "CublasSplitKGemmF32Out_v1", "CublasSplitKGemm_v1", "DsaIndexerKRow_v1", "DsaIndexerQ_v1", "LlBf16DotprodGemm_v1",
                   "RMSNormCudaVector8_v1")
#: what the same rule failed B8 greedy's replay on, all 8 ranks: vLLM's cuteDSL router gate at 5 to 16 tokens a step
TP8_B8_FAMILIES = ("LlBf16SplitKGemm_v1",)

HOP = hopper.HopperBF16WgmmaDot16
MLA = dict(DR=16, DV=16, BN=16, DOT=HOP, INV=hopper.Fa3InvSum, SCALE=0.0625)

#: (spec text as a Program spells it, the Definition bound at the same statics)
CASES = [
    ('RMSNormCudaVector8_v1{N=64,EPS={"f64":"0x1.4f8b588e368f1p-17"},BLOCK=8}', bind(cuda_rmsnorm.RMSNormCudaVector8, N=64, EPS=1e-5, BLOCK=8)),
    ('RMSNormFusedCudaVector8_v1{N=64,EPS=1e-05,BLOCK=3}', bind(cuda_rmsnorm.RMSNormFusedCudaVector8, N=64, EPS=1e-5, BLOCK=3)),
    ('LlBf16DotprodGemm_v1{N=3,K=128,BS=64,VEC=2,FMA=1}', bind(ll_bf16.LlBf16DotprodGemmWord, N=3, K=128, BS=64, VEC=2, FMA=1)),
    ('LlBf16SplitKGemm_v1{N=3,K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}',
     bind(ll_bf16.LlBf16SplitKGemmWord, N=3, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP)),
    ('LlBf16SplitKGemm_v1{N=2,K=192,SPLIT_K=4,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v1}',
     bind(ll_bf16.LlBf16SplitKGemmWord, N=2, K=192, SPLIT_K=4, TILE_K=32, WARPS=2, DOT=HOP)),
    ('CublasSplitKGemm_v1{K=192,N=3,SPLIT_K=3,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}', bind(cublas_bf16.CublasSplitKGemmWord, K=192, N=3, SPLIT_K=3, DOT=HOP)),
    ('CublasSplitKGemmF32Out_v1{K=64,N=3,SPLIT_K=2,DOT=HopperBF16WgmmaDot16_v1}', bind(cublas_bf16.CublasSplitKGemmF32OutWord, K=64, N=3, SPLIT_K=2, DOT=HOP)),
    ('Fp8GroupQuantUe8m0_v1{K=256,G=128}', bind(deepgemm.Fp8GroupQuantUe8m0, K=256, G=128)),
    ('MoeExpertFp8Gemm_v1{E=2,K=128,N=96,G=64,KB=64,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}}',
     bind(fp8_moe_experts.MoeExpertFp8Gemm, E=2, K=128, N=96, G=64, KB=64, SWAP=True, DOT=HopperE4m3QgmmaDot32)),
    ('MoeExpertFp8Gemm_v1{E=2,K=128,N=96,G=64,KB=32,SWAP=False,DOT=HopperE4m3MmaV2Dot32_v1}',
     bind(fp8_moe_experts.MoeExpertFp8Gemm, E=2, K=128, N=96, G=64, KB=32, SWAP=False, DOT=HopperE4m3MmaV2Dot32)),
    ('MoeExpertFp8GemmW_v1{E=3,K=256,N=3,G=128,KB=128,SWAP=false,DOT={"fn":"HopperE4m3MmaV2Dot32_v1"}}',
     bind(fp8_moe_experts.MoeExpertFp8GemmW, E=3, K=256, N=3, G=128, KB=128, SWAP=False, DOT=HopperE4m3MmaV2Dot32)),
    ('MoeExpertFp8GemmW_v1{E=3,K=256,N=3,G=128,KB=128,SWAP=True,DOT=HopperE4m3QgmmaDot32_v1}',
     bind(fp8_moe_experts.MoeExpertFp8GemmW, E=3, K=256, N=3, G=128, KB=128, SWAP=True, DOT=HopperE4m3QgmmaDot32)),
    ('SparseMlaSplit_v1{T=40,NS=2,NH=2,DR=16,DV=16,BN=16,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
     'SCALE={"f64":"0x1.0000000000000p-4"}}', bind(sparse_mla_split.SparseMlaSplit, T=40, NS=2, NH=2, **MLA)),
    ('SparseMlaSplit_v1{T=20,NS=1,NH=3,DR=16,DV=16,BN=16,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,SCALE=0.0625}',
     bind(sparse_mla_split.SparseMlaSplit, T=20, NS=1, NH=3, **MLA)),
    ('ResidualAdd_v1{N=16}', bind(b1.ResidualAdd, N=16)),
    ('RoPEHeadGptJ_v2{D=16}', bind(boolean_rope.RoPEHeadGptJV2, D=16)),
    ('DsaIndexerKRow_v1{D=32,ROPE=16,EPS={"f64":"0x1.0c6f7a0b5ed8dp-20"}}', bind(dsa.DsaIndexerKRow, D=32, ROPE=16, EPS=1e-6)),
    ('DsaIndexerQ_v1{H=2,D=32,ROPE=16}', bind(dsa.DsaIndexerQ, H=2, D=32, ROPE=16)),
    ("MoeRouterSigmoidBias_v1{E=8,TOPK=2,SCALE=2.5,KERNEL='block'}", bind(moe.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL="block")),
    ('MoeRouterSigmoidBias_v1{E=8,TOPK=2,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="warp"}',
     bind(moe.MoeRouterSigmoidBias, E=8, TOPK=2, SCALE=2.5, KERNEL="warp")),
]


def _words(rng: np.random.Generator, pt, scale: float) -> np.ndarray:
    shape, w = type_shape(pt), set(pt.leaf_widths()).pop()
    if w == 16:
        return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    if w == 32:
        return (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)
    return (rng.integers(0, 0x7E, shape) | (rng.integers(0, 2, shape) << 7)).astype(np.uint8)


def test_every_family_the_tp8_b1_and_b8_rule_named_has_a_row_evaluator():
    missing = [f for f in TP8_B1_FAMILIES + TP8_B8_FAMILIES if f not in rows.ROWS or f not in rows.EVALUATORS]
    assert not missing, f"no replay row evaluator for {missing}"


@pytest.mark.slow
@pytest.mark.parametrize("spec", ['LlBf16SplitKGemm_v1{N=256,K=6144,SPLIT_K=6,TILE_K=256,WARPS=4,DOT=HopperBF16WgmmaDot16_v1}',
                                  'LlBf16SplitKGemm_v1{N=256,K=6144,SPLIT_K=8,TILE_K=256,WARPS=4,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'],
                         ids=["split-k-6", "split-k-8"])
def test_the_split_k_gate_at_glm53s_served_statics_is_the_definition(spec):
    """GLM-5.3's router gate as B8 serves it (N 256, K 6144, TILE_K 256, WARPS 4; SPLIT_K 6, or 8 at M 9 and 11): 24 tiles over 6
    or 8 cluster ranks, which the small self-check statics do not reach.  The row kernel on the spec text's statics equals the
    Definition the Program binds there, every one of the 256 words, by the word reference."""
    family, statics = RM.spec_family(spec), RM.spec_statics(spec)
    d = bind(ll_bf16.LlBf16SplitKGemmWord, N=256, K=6144, SPLIT_K=int(statics["SPLIT_K"]), TILE_K=256, WARPS=4, DOT=HOP)
    (px, pw), _ = signature_of(d)
    rng = np.random.default_rng(int(statics["SPLIT_K"]))
    x, w = _words(rng, px, 1.0), _words(rng, pw, 0.02)
    got = rows.ROWS[family].fn(statics, [x, w])["0"]
    want = np.asarray(evaluate_batch(d, x[None], w[None], kernel="reference")).reshape(-1)
    assert got.dtype == np.uint32 and got.shape == (256,) and np.array_equal(got.astype(np.uint64), want.astype(np.uint64))


def test_the_split_k_gate_declines_what_splitk_rows_does_not_model():
    statics = RM.spec_statics('LlBf16SplitKGemm_v1{N=2,K=128,SPLIT_K=2,TILE_K=32,WARPS=2,DOT=HopperBF16WgmmaDot16_v1}')
    args = [np.zeros(128, np.uint16), np.zeros(256, np.uint16)]
    assert rows.ROWS["LlBf16SplitKGemm_v1"].fn(statics, args)["0"].tolist() == [0, 0]
    with pytest.raises(Declined, match="no twin for DOT=AmpereBF16TcDot16_v1"):
        rows.ROWS["LlBf16SplitKGemm_v1"].fn(statics | {"DOT": "AmpereBF16TcDot16_v1"}, args)
    with pytest.raises(Declined, match="SPLIT_K 5"):
        rows.ROWS["LlBf16SplitKGemm_v1"].fn(statics | {"SPLIT_K": 5}, args)


def test_the_split_k_gate_cites_the_h200_capture_at_every_split_k_m_it_runs():
    """The replay record's `evaluator_provenance` for LlBf16SplitKGemm_v1 is the capture that held cuteDSL's split-K launches to
    `splitk_rows`: at both gates, every M the capture runs that the dispatch table sends to the split-K kernel, at the table's
    SPLIT_K, TILE_K and WARPS, N words a token row; and the Definition's conformance cites the same record."""
    from tests.properties import glm53_bf16_gemm_capture_gpu as X
    from verity_vllm.program.frontend.conformance import status_of
    from verity.ml.definitions.numerics import batched_gemm_rows, glm53_rows
    from verity.ml.definitions.registry import bf16_gemm_check as BC

    p = rows.EVALUATOR_PROVENANCE["LlBf16SplitKGemm_v1"]
    assert p is glm53_rows.PROVENANCE["LlBf16SplitKGemm_v1"]
    same = ("property", "run", "record", "record_served_default", "art", "variants", "captured_utc")
    assert {k: p[k] for k in same} == {k: batched_gemm_rows.PROVENANCE[k] for k in same}
    assert set(p["cases"]) == {f"{m}/router/mlp.gate" for m in X.MODELS}
    for name, m in X.MODELS.items():
        c = p["cases"][f"{name}/router/mlp.gate"]
        routes = {M: BC.dispatch_sm90(M, m.hidden, m.experts) for M in m.router_m}
        split = {M: st for M, (kind, st) in routes.items() if kind == "splitk"}
        assert (c["K"], c["N"]) == (m.hidden, m.experts) and split
        assert c["SPLIT_K"] == {str(M): st["SPLIT_K"] for M, st in split.items()}
        assert all((c["TILE_K"], c["WARPS"]) == (st["TILE_K"], st["WARPS"]) for st in split.values())
        assert c["mismatch"] == 0 and c["words"] == m.experts * sum(split)
    d = ll_bf16.LlBf16SplitKGemmWord
    assert status_of(d.conformance) == "exact-model-tested"
    assert p["run"] in d.conformance and p["record"][:16] in d.conformance and p["art"]["record"] in d.conformance


@pytest.mark.parametrize("spec,d", CASES, ids=[s.split("{")[0] + f"-{i}" for i, (s, _) in enumerate(CASES)])
def test_the_row_kernel_on_spec_text_statics_is_the_definition(spec, d):
    family, statics = RM.spec_family(spec), RM.spec_statics(spec)
    params, _ = signature_of(d)
    rng = np.random.default_rng(len(spec))
    for _ in range(3):
        args = [_words(rng, p, 0.05 if family.startswith("SparseMla") else 1.0) for p in params]
        if family.startswith("MoeExpertFp8"):
            args[1] = np.abs(rng.standard_normal(args[1].shape) * 1e-2).astype(np.float32).view(np.uint32)
            args[2] = np.asarray(rng.integers(0, int(statics["E"])), dtype=np.uint32)
        if family == "MoeRouterSigmoidBias_v1":
            args[1] = (rng.standard_normal(args[1].shape) * 1e-2).astype(np.float32).view(np.uint32)
        got = rows.ROWS[family].fn(statics, args)
        want = np.asarray(evaluate_batch(d, *(a.astype(type_dtype(p)).reshape(1, *type_shape(p)) for a, p in zip(args, params)),
                                         kernel="reference")).reshape(-1)
        flat = np.concatenate([np.asarray(got[m]).reshape(-1).astype(np.uint64) for m in rows.ROWS[family].returns])
        assert flat.shape == want.shape and np.array_equal(flat, want.astype(np.uint64)), spec


@pytest.mark.parametrize("family,definition", [("MoeExpertFp8Gemm_v1", fp8_moe_experts.MoeExpertFp8Gemm),
                                               ("MoeExpertFp8GemmW_v1", fp8_moe_experts.MoeExpertFp8GemmW)])
@pytest.mark.parametrize("swap", ["true", "True"])
def test_swap_from_the_spec_text_orders_the_two_scale_products(family, definition, swap):
    """Random scales almost never tell the two orders apart after the bf16 rounding.  With sx = 2^120 and ws = 2^-120,
    FMUL(d, sx) overflows and FMUL(d, ws) does not, so SWAP read as false declines the row, and SWAP true must equal the
    Definition."""
    spec = f'{family}{{E=2,K=128,N=96,G=64,KB=64,SWAP={swap},DOT={{"fn":"HopperE4m3QgmmaDot32_v1"}}}}'
    d = bind(definition, E=2, K=128, N=96, G=64, KB=64, SWAP=True, DOT=HopperE4m3QgmmaDot32)
    statics, (params, _) = RM.spec_statics(spec), signature_of(d)
    rng = np.random.default_rng(3)
    args = [_words(rng, p, 1.0) for p in params]
    args[1] = np.full(args[1].shape, np.float32(2.0 ** 120)).view(np.uint32)
    args[2] = np.asarray(1, dtype=np.uint32)
    if family == "MoeExpertFp8GemmW_v1":
        args[3] = np.asarray(np.float32(1.0)).view(np.uint32)
    ws = 4 if family == "MoeExpertFp8GemmW_v1" else 3
    args[ws + 1] = np.full(args[ws + 1].shape, np.float32(2.0 ** -120)).view(np.uint32)
    got = rows.ROWS[family].fn(statics, args)["0"]
    want = np.asarray(evaluate_batch(d, *(a.astype(type_dtype(p)).reshape(1, *type_shape(p)) for a, p in zip(args, params)),
                                     kernel="reference")).reshape(-1)
    assert np.array_equal(got.astype(np.uint64), want.astype(np.uint64))
    with pytest.raises(Declined, match="non-finite"):
        rows.ROWS[family].fn(statics | {"SWAP": "false"}, args)
