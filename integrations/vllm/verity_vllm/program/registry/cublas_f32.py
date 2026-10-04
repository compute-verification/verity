"""cuBLAS's f32 GEMM on sm_120 with split-K off (vLLM batch invariance's workspace config), on a bf16 activation widened to f32:
`nn.Linear(dtype=float32)` on `x.to(float32)`, GLM-4.7-Flash's router gate. cuBLAS picks the kernel by M, and its two families
reduce K differently, so each is its own dot:

    CublasGemvDot_v1{K}               (x, w: Array<K, F32>) -> F32    gemv2T_kernel_val (M = 1), gemmSN_TN_kernel (small M): thread
                                      class t = k mod 16 runs one FMA chain from +0 per tile of 16 (k = 256 g + 16 i + t, i
                                      ascending), adds its tiles' sums in g order, and the 16 class sums are added in t order
    CublasSimtDot_v1{K}               (x, w: Array<K, F32>) -> F32    cutlass_80_simt_sgemm_*: one FMA chain from +0, k ascending
    CublasF32Linear_v1{K, N, TREE}    (x: Array<K, BF16>, w: Array<N, Array<K, F32>>) -> Array<N, F32>
                                      `Bf16ToF32` of x, then TREE's dot (gemv | simt) on each W row

on words, and each one's v2 the same body on bits (its ``word`` the v1, so `boolean.boolean_version` finds it), each chain's first
step `CublasFmaFromZero_v2` (F32Fma_v3 with the constant +0 addend traced away, `F32Fma_v1(a, b, +0)` on bits). The trees are the
ones exact LCA queries recover (r20261004-070541-1622: x = 1, w = 1 except +-2^60 at a pair, M = 1, 8 and 256 at K = 2048), and
their outputs equal the sm_120 kernels' on bf16-exact x against bf16-exact, 23-bit f32 and all-negative weights at every M probed
(r20261004-163918-3519: 2..300 and nine larger M; r20261004-164622-187e: M = 1 and 3151; `cublas_f32_check`,
`tests/program/test_cublas_f32.py`).
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.types import Array
from verity.ml import fp32 as W
from verity.ml import prims as P
from verity.ml.boolean import fp
from verity.ml.boolean import scalar as BS
from verity.ml.boolean.forms import ZERO
from verity.ml.boolean.trace import definition

__all__ = ["CublasF32Linear", "CublasF32LinearWord", "CublasFmaFromZero", "CublasFmaFromZeroWord", "CublasGemvDot", "CublasGemvDotWord",
           "CublasSimtDot", "CublasSimtDotWord", "TREES"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
BF16W, F32W = P.BF16, P.F32
CLASSES = TILE = 16
TREES = ("gemv", "simt")
CONFORMANCE = "exact-model-tested: the probed GPU kernels' outputs (tests/program/test_cublas_f32.py)"


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _gemv(B, K, x, w, fma, first, add):
    _check("CublasGemvDot", K_multiple_of_256=K > 0 and K % (CLASSES * TILE) == 0)
    y = None
    for t in range(CLASSES):
        total = None
        for g in range(K // (CLASSES * TILE)):
            k0 = CLASSES * TILE * g + t
            part = first(x[k0], w[k0])
            for i in range(1, TILE):
                k = k0 + CLASSES * i
                part = B.call(fma, x[k], w[k], part)
            total = part if total is None else B.call(add, total, part)
        y = total if y is None else B.call(add, y, total)
    return y


def _simt(B, K, x, w, fma, first):
    _check("CublasSimtDot", K_positive=K > 0)
    acc = first(x[0], w[0])
    for k in range(1, K):
        acc = B.call(fma, x[k], w[k], acc)
    return acc


def _word_first(B):
    zero = B.call(P.ZERO32)
    return lambda a, b: B.call(W.F32Fma, a, b, zero)


def _bits_first(B):
    return lambda a, b: B.call(CublasFmaFromZero, a, b)


def _dot_type(T):
    return lambda S: ((("x", Array(S.K, T)), ("w", Array(S.K, T))), T)


def _linear_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, T)))), Array(S.N, T))


def _linear(dots, widen):
    def body(B, S, x, w):
        _check("CublasF32Linear", TREE_known=S.TREE in TREES)
        xf = B.batch(widen, x, axes=(0,))
        return B.batch(bind(dots[S.TREE], K=S.K), xf, w, axes=(None, 0))
    return body


GEMV_DOC = ("gemv2T_kernel_val / gemmSN_TN_kernel's coordinate: per thread class t = k mod 16 one FMA chain from +0 per tile of 16, "
            "tile sums added in order, class sums added in t order")
SIMT_DOC = "cutlass_80_simt_sgemm_*'s coordinate: one FMA chain from +0, k ascending"
LINEAR_DOC = "`x[K] (bf16, widened) @ W[N, K]^T` in f32 as cuBLAS's TREE kernel family reduces it"

# the word dots call F32Fma_v1 on ZERO32 themselves: this word is CublasFmaFromZero_v2's reference only, not a registry entry
CublasFmaFromZeroWord = CompositeDefinition("CublasFmaFromZero", 1, (), lambda S: ((("a", F32W), ("b", F32W)), F32W),
                                            lambda B, S, a, b: B.call(W.F32Fma, a, b, B.call(P.ZERO32)),
                                            doc="F32Fma_v1(a, b, +0): an FMA chain's first step", conformance=CONFORMANCE,
                                            register=False)
CublasFmaFromZero = definition("CublasFmaFromZero", 2, lambda C, a, b: fp.f32_fma(C, a, b, [ZERO] * 32), (("a", F32), ("b", F32)), F32,
                               word=bind(CublasFmaFromZeroWord), doc="F32Fma_v3 with its addend the constant +0, folded (on bits)",
                               conformance=CONFORMANCE)

CublasGemvDotWord = CompositeDefinition("CublasGemvDot", 1, ("K",), _dot_type(F32W),
                                        lambda B, S, x, w: _gemv(B, S.K, x, w, W.F32Fma, _word_first(B), W.F32Add),
                                        doc=GEMV_DOC, conformance=CONFORMANCE)
CublasSimtDotWord = CompositeDefinition("CublasSimtDot", 1, ("K",), _dot_type(F32W),
                                        lambda B, S, x, w: _simt(B, S.K, x, w, W.F32Fma, _word_first(B)),
                                        doc=SIMT_DOC, conformance=CONFORMANCE)
CublasF32LinearWord = CompositeDefinition("CublasF32Linear", 1, ("K", "N", "TREE"), _linear_type(BF16W, F32W),
                                          _linear({"gemv": CublasGemvDotWord, "simt": CublasSimtDotWord}, P.Bf16ToF32),
                                          doc=LINEAR_DOC, conformance=CONFORMANCE)

CublasGemvDot = CompositeDefinition("CublasGemvDot", 2, ("K",), _dot_type(F32),
                                    lambda B, S, x, w: _gemv(B, S.K, x, w, BS.F32Fma, _bits_first(B), BS.F32Add),
                                    doc=f"{GEMV_DOC} (on bits)", conformance=CONFORMANCE)
CublasSimtDot = CompositeDefinition("CublasSimtDot", 2, ("K",), _dot_type(F32),
                                    lambda B, S, x, w: _simt(B, S.K, x, w, BS.F32Fma, _bits_first(B)),
                                    doc=f"{SIMT_DOC} (on bits)", conformance=CONFORMANCE)
CublasF32Linear = CompositeDefinition("CublasF32Linear", 2, ("K", "N", "TREE"), _linear_type(BF16, F32),
                                      _linear({"gemv": CublasGemvDot, "simt": CublasSimtDot}, BS.Bf16ToF32),
                                      doc=f"{LINEAR_DOC} (on bits)", conformance=CONFORMANCE)
CublasGemvDot.word = lambda b: bind(CublasGemvDotWord, K=b["K"])  # type: ignore[attr-defined]
CublasSimtDot.word = lambda b: bind(CublasSimtDotWord, K=b["K"])  # type: ignore[attr-defined]
CublasF32Linear.word = lambda b: bind(CublasF32LinearWord, K=b["K"], N=b["N"], TREE=b["TREE"])  # type: ignore[attr-defined]
