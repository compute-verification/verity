"""cuBLAS's bf16 GEMMs on sm_90 with split-K off, the configuration under which each output coordinate is one ascending chain of
k16 tensor-core steps from the zero accumulator: GLM-5.3's MoE router gate past the cuteDSL tier (`torch.mm(x, W.T,
out_dtype=float32)`, `GateLinear` tier 4) and its MLA absorbs (`torch.bmm` of the queries against `W_UK_T`, of the attention output
against `W_UV`).  Its bf16-output linears (`lm_head`, the indexer's `wk_weights_proj`) are core's `Gemm_v2{K, N, DOT}`.

    GemmF32Out_v1{K, N, DOT}        (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, F32>
                                    `DotBf16_v2{K, DOT}` of x against each W row: the f32 accumulator is the output
    BatchedGemm_v1{B, K, N, DOT}    (x: Array<B, Array<K, BF16>>, w: Array<B, Array<N, Array<K, BF16>>>) -> Array<B, Array<N, BF16>>
                                    `Gemm_v2{K, N, DOT}` of member b's x against member b's W, ``w[b][n][k]`` the right operand at
                                    ``[b, k, n]``

on words, and each one's v2 the same body on bits (`DotBf16_v3`, `Gemm_v3`; its ``word`` the v1, so `boolean.boolean_version`
finds it).  `BatchedGemm_v2` is the function of `mla.MlaBmm_v1` (vLLM's batch-invariant Triton `bmm_kernel` on sm_120), which
names no word; `tests/program/test_bf16_gemm.py` checks that they agree.  `bf16_gemm_check` holds the numpy models.

The split-K-off configuration is vLLM's batch-invariant one on cc 9.0 (`enable_batch_invariant_mode`): CUBLAS_WORKSPACE_CONFIG=:16:8,
CUBLASLT_WORKSPACE_SIZE=1, `allow_bf16_reduced_precision_reduction = (False, False)`, `preferred_blas_library("cublaslt")`.  Under it the
H100's bf16 linear is `Gemm_v2{K, N, HopperBF16WgmmaDot16_v1}` word for word (`kernels.derived_rows`, the R17 probe); torch's defaults
split K at small M.  GLM-5.3's shapes on the H200 are pinned by `tests/properties/glm53_bf16_gemm_capture_gpu.py`.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.types import Array

__all__ = ["BatchedGemm", "BatchedGemmWord", "GemmF32Out", "GemmF32OutWord"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
CONFORMANCE = "not-yet-established: GLM-5.3's shapes on the H200 (tests/properties/glm53_bf16_gemm_capture_gpu.py)"


def _gemm_f32_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, X)))), Array(S.N, T))


def _batched_type(X):
    return lambda S: ((("x", Array(S.B, Array(S.K, X))), ("w", Array(S.B, Array(S.N, Array(S.K, X))))), Array(S.B, Array(S.N, X)))


def _gemm_f32(dot):
    def body(B, S, x, w):
        return B.batch(bind(dot, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))
    return body


def _batched(gemm):
    def body(B, S, x, w):
        return B.batch(bind(gemm, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))
    return body


GEMM_F32_DOC = "`x[K] @ W[N, K]^T` with the f32 accumulator as the output: `DotBf16{K, DOT}` on each W row (torch.mm out_dtype=float32)"
BATCHED_DOC = "torch.bmm on bf16: `Gemm{K, N, DOT}` of each member's x against its W (w[b][n][k] the operand at [b, k, n])"

GemmF32OutWord = CompositeDefinition("GemmF32Out", 1, ("K", "N", "DOT"), _gemm_f32_type(P.BF16, P.F32), _gemm_f32(G.DotBf16),
                                     doc=GEMM_F32_DOC, conformance=CONFORMANCE)
BatchedGemmWord = CompositeDefinition("BatchedGemm", 1, ("B", "K", "N", "DOT"), _batched_type(P.BF16), _batched(G.Gemm),
                                      doc=BATCHED_DOC, conformance=CONFORMANCE)

GemmF32Out = CompositeDefinition("GemmF32Out", 2, ("K", "N", "DOT"), _gemm_f32_type(BF16, F32), _gemm_f32(BG.DotBf16),
                                 doc=f"{GEMM_F32_DOC} (on bits)", conformance=CONFORMANCE)
BatchedGemm = CompositeDefinition("BatchedGemm", 2, ("B", "K", "N", "DOT"), _batched_type(BF16), _batched(BG.Gemm),
                                  doc=f"{BATCHED_DOC} (on bits)", conformance=CONFORMANCE)
GemmF32Out.word = lambda b: bind(GemmF32OutWord, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
BatchedGemm.word = lambda b: bind(BatchedGemmWord, B=b["B"], K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
