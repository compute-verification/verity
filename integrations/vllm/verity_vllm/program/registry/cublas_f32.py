"""cuBLAS's f32 GEMM on sm_120 with split-K off (vLLM batch invariance's workspace config), on a bf16 activation widened to f32:
`nn.Linear(dtype=float32)` on `x.to(float32)`, GLM-4.7-Flash's router gate. cuBLAS picks the kernel by M, and its two families
reduce K differently, so each is its own dot:

    CublasGemvDot_v1{K}               (x, w: Array<K, F32>) -> F32    gemv2T_kernel_val (M = 1), gemmSN_TN_kernel (small M): thread
                                      class t = k mod 16 runs one FMA chain from +0 per tile of 16 (k = 256 g + 16 i + t, i
                                      ascending), adds its tiles' sums in g order, and the 16 class sums are added in t order
    CublasSimtDot_v1{K}               (x, w: Array<K, F32>) -> F32    cutlass_80_simt_sgemm_*: one FMA chain from +0, k ascending
    CublasF32Linear_v1{K, N, TREE}    (x: Array<K, BF16>, w: Array<N, Array<K, F32>>) -> Array<N, F32>
                                      `Bf16ToF32` of x, then TREE's dot (gemv | simt) on each W row

and their word twins (`...Word_v1`, the same bodies on words, for word Programs). The trees are the ones exact LCA queries
recover (r20261004-070541-1622: x = 1, w = 1 except +-2^60 at a pair, M = 1, 8 and 256 at K = 2048), and the outputs on random
data, f32 weights included, are the sm_120 kernels' (r20261004-065341-db1b kernel table; `tests/program/test_cublas_f32.py`).
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind, composite
from verity.ir.refs import Coll, broadcast
from verity.ir.types import Array
from verity.ml import fp32 as W
from verity.ml import prims as P
from verity.ml.boolean import scalar as BS

__all__ = ["CublasF32Linear", "CublasF32LinearWord", "CublasGemvDot", "CublasGemvDotWord", "CublasSimtDot", "CublasSimtDotWord",
           "TREES"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
BF16W, F32W = P.BF16, P.F32
CLASSES = TILE = 16
TREES = ("gemv", "simt")
CONFORMANCE = "exact-model-tested: the sm_120 kernels' outputs (tests/program/test_cublas_f32.py)"


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _gemv(B, K, x, w, fma, add, zero):
    _check("CublasGemvDot", K_multiple_of_256=K > 0 and K % (CLASSES * TILE) == 0)
    y = None
    for t in range(CLASSES):
        total = None
        for g in range(K // (CLASSES * TILE)):
            part = zero
            for i in range(TILE):
                k = CLASSES * TILE * g + CLASSES * i + t
                part = B.call(fma, x[k], w[k], part)
            total = part if total is None else B.call(add, total, part)
        y = total if y is None else B.call(add, y, total)
    return y


def _simt(B, K, x, w, fma, zero):
    _check("CublasSimtDot", K_positive=K > 0)
    acc = zero
    for k in range(K):
        acc = B.call(fma, x[k], w[k], acc)
    return acc


def _bits_zero(B) -> Coll:
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(F32, broadcast(space, idx, leaf, 32))


def _dot_type(T):
    return lambda S: ((("x", Array(S.K, T)), ("w", Array(S.K, T))), T)


def _linear_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, T)))), Array(S.N, T))


CublasGemvDot = CompositeDefinition("CublasGemvDot", 1, ("K",), _dot_type(F32),
                                    lambda B, S, x, w: _gemv(B, S.K, x, w, BS.F32Fma, BS.F32Add, _bits_zero(B)),
                                    doc="gemv2T_kernel_val / gemmSN_TN_kernel's coordinate: per thread class t = k mod 16 one FMA "
                                        "chain from +0 per tile of 16, tile sums added in order, class sums added in t order (on bits)",
                                    conformance=CONFORMANCE)
CublasSimtDot = CompositeDefinition("CublasSimtDot", 1, ("K",), _dot_type(F32),
                                    lambda B, S, x, w: _simt(B, S.K, x, w, BS.F32Fma, _bits_zero(B)),
                                    doc="cutlass_80_simt_sgemm_*'s coordinate: one FMA chain from +0, k ascending (on bits)",
                                    conformance=CONFORMANCE)


def _linear(dots, widen):
    def body(B, S, x, w):
        _check("CublasF32Linear", TREE_known=S.TREE in TREES)
        xf = B.batch(widen, x, axes=(0,))
        return B.batch(bind(dots[S.TREE], K=S.K), xf, w, axes=(None, 0))
    return body


CublasF32Linear = CompositeDefinition("CublasF32Linear", 1, ("K", "N", "TREE"), _linear_type(BF16, F32),
                                      _linear({"gemv": CublasGemvDot, "simt": CublasSimtDot}, BS.Bf16ToF32),
                                      doc="`x[K] (bf16, widened) @ W[N, K]^T` in f32 as cuBLAS's TREE kernel family reduces it (on bits)",
                                      conformance=CONFORMANCE)


@composite("CublasGemvDotWord", 1, ["K"], _dot_type(F32W), conformance=CONFORMANCE)
def CublasGemvDotWord(B, S, x, w):
    """`CublasGemvDot_v1` on words."""
    return _gemv(B, S.K, x, w, W.F32Fma, W.F32Add, B.call(P.ZERO32))


@composite("CublasSimtDotWord", 1, ["K"], _dot_type(F32W), conformance=CONFORMANCE)
def CublasSimtDotWord(B, S, x, w):
    """`CublasSimtDot_v1` on words."""
    return _simt(B, S.K, x, w, W.F32Fma, B.call(P.ZERO32))


CublasF32LinearWord = CompositeDefinition("CublasF32LinearWord", 1, ("K", "N", "TREE"), _linear_type(BF16W, F32W),
                                          _linear({"gemv": CublasGemvDotWord, "simt": CublasSimtDotWord}, P.Bf16ToF32),
                                          doc="`CublasF32Linear_v1` on words", conformance=CONFORMANCE)
