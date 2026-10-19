"""GLM-5.3 TP8 B1's row evaluators (`kernels.glm53_rows`) as the sampled replay calls them.  Every family the per-rank no-evaluator
rule named on the TP8 Commit r20261016-213336-6c35 has one.  The replay hands a row kernel the statics `spec_statics` parses from
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
from verity_vllm.program.kernels import rows
from verity_vllm.program.kernels.kernel_registry import Declined
from verity_vllm.program.registry import b1, boolean_rope, cublas_bf16, cuda_rmsnorm, deepgemm, dsa, fp8_moe_experts, hopper
from verity_vllm.program.registry import ll_bf16, moe, sparse_mla_split
from verity_vllm.query.manifest import format as RM

#: the families FIXWIT's per-rank rule failed GLM-5.3 TP8 B1's Commit on: four plane-fed MoE producers and eleven with no evaluator
TP8_B1_FAMILIES = ("Fp8GroupQuantUe8m0_v1", "MoeExpertFp8GemmW_v1", "MoeExpertFp8Gemm_v1", "MoeRouterSigmoidBias_v1",
                   "RoPEHeadGptJ_v2", "BatchedGemm_v1", "RMSNormFusedCudaVector8_v1", "SparseMlaSplit_v1", "ResidualAdd_v1",
                   "CublasSplitKGemmF32Out_v1", "CublasSplitKGemm_v1", "DsaIndexerKRow_v1", "DsaIndexerQ_v1", "LlBf16DotprodGemm_v1",
                   "RMSNormCudaVector8_v1")

HOP = hopper.HopperBF16WgmmaDot16
MLA = dict(DR=16, DV=16, BN=16, DOT=HOP, INV=hopper.Fa3InvSum, SCALE=0.0625)

#: (spec text as a Program spells it, the Definition bound at the same statics)
CASES = [
    ('RMSNormCudaVector8_v1{N=64,EPS={"f64":"0x1.4f8b588e368f1p-17"},BLOCK=8}', bind(cuda_rmsnorm.RMSNormCudaVector8, N=64, EPS=1e-5, BLOCK=8)),
    ('RMSNormFusedCudaVector8_v1{N=64,EPS=1e-05,BLOCK=3}', bind(cuda_rmsnorm.RMSNormFusedCudaVector8, N=64, EPS=1e-5, BLOCK=3)),
    ('LlBf16DotprodGemm_v1{N=3,K=128,BS=64,VEC=2,FMA=1}', bind(ll_bf16.LlBf16DotprodGemmWord, N=3, K=128, BS=64, VEC=2, FMA=1)),
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


def test_every_family_the_tp8_b1_rule_named_has_a_row_evaluator():
    missing = [f for f in TP8_B1_FAMILIES if f not in rows.ROWS or f not in rows.EVALUATORS]
    assert not missing, f"no replay row evaluator for {missing}"


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
