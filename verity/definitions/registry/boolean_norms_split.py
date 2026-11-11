"""`boolean_norms`' CUDA vector-8 paths cut at their scale, as `cuda_rmsnorm_split` cuts the word kernels: separate units on bits.

    RMSNormSumSqCudaVector8_v2, RMSNormFusedSumSqCudaVector8_v2{N, EPS, BLOCK, RSQRT}
                                         the path up to s (`boolean_norms`' lanes, warps, `RMSNormArgCuda_v1` and RSQRT)
    RMSNormApplyCudaVector8_v2, RMSNormFusedApplyCudaVector8_v2{G}
                                         the output row on G elements (`RMSNormOutRowCuda_v1{G}`), from s

An apply unit holds no square, so it computes each element's decode alone (`RMSNormDecodeCuda_v1`), and the fused one redoes
its residual adds. Their word views are `cuda_rmsnorm_split`'s units of the same names, `_v1`.
"""
from __future__ import annotations

from typing import Callable

from verity.circuits.defs import bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.refs import Coll, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_norms as BN
from verity.definitions.registry.boolean_norms import BF16, CUDA_WORD, F32

__all__ = ["RMSNormApplyCudaVector8", "RMSNormFusedApplyCudaVector8", "RMSNormFusedSumSqCudaVector8", "RMSNormSumSqCudaVector8"]

RMSNormDecodeCuda = BN._traced("RMSNormDecodeCuda", 1, (), lambda S: ((("z", BF16),), BN._bits(BN._dec_bits("cuda"))),
                               lambda S: BN._out_parts("cuda")[1],
                               "z's decode for `RMSNormOutCuda_v1` without its square: an apply unit's, whose squares another "
                               "unit computes")


def _apply_word(kind: str) -> Callable[[dict], object]:
    return lambda b: bind(BN._CUDA_VECTOR8_WORDS[kind], G=b["G"])


@BN._on_bits("RMSNormSumSqCudaVector8", 2, ["N", "EPS", "BLOCK", "RSQRT"],
             lambda S: ((("x", Array(S.N, BF16)),), F32), BN._cuda_vector8_word("sumsq"), word_statics=CUDA_WORD)
def RMSNormSumSqCudaVector8(B, S, x):
    """`RMSNormSumSqCudaVector8_v1` on bits: `RMSNormCudaVector8_v2` up to its scale."""
    return BN._vector8_scale(B, S, x)[1]


@BN._on_bits("RMSNormFusedSumSqCudaVector8", 2, ["N", "EPS", "BLOCK", "RSQRT"],
             lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16))), F32), BN._cuda_vector8_word("fused_sumsq"),
             word_statics=CUDA_WORD)
def RMSNormFusedSumSqCudaVector8(B, S, x, res):
    """`RMSNormFusedSumSqCudaVector8_v1` on bits: `RMSNormFusedCudaVector8_v2` up to its scale."""
    return BN._vector8_scale(B, S, x, res)[1]


def _apply_row(B, G: int, r: Coll, w: Coll, scale: Coll) -> Coll:
    rd = B.batch(RMSNormDecodeCuda, r, axes=(0,))
    return B.call(bind(BN.RMSNormOutRowCuda, N=G), r, rd, scale, B.call(BN.RMSNormScaleCuda, scale), w)


@BN._on_bits("RMSNormApplyCudaVector8", 2, ["G"],
             lambda S: ((("x", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("scale", F32)), Array(S.G, BF16)),
             _apply_word("apply"))
def RMSNormApplyCudaVector8(B, S, x, w, scale):
    """`RMSNormApplyCudaVector8_v1` on bits: each element's decode (`RMSNormDecodeCuda_v1`), then `RMSNormOutRowCuda_v1{G}`."""
    return _apply_row(B, S.G, x, w, scale)


@BN._on_bits("RMSNormFusedApplyCudaVector8", 2, ["G"],
             lambda S: ((("x", Array(S.G, BF16)), ("res", Array(S.G, BF16)), ("w", Array(S.G, BF16)), ("scale", F32)),
                        Tuple(Array(S.G, BF16), Array(S.G, BF16))),
             _apply_word("fused_apply"))
def RMSNormFusedApplyCudaVector8(B, S, x, res, w, scale):
    """`RMSNormFusedApplyCudaVector8_v1` on bits: the residual adds, then `RMSNormApplyCudaVector8_v2`'s row on their sums."""
    r = B.batch(EW.Bf16AddF2fp, x, res, axes=(0, 0))
    return tuple_of(_apply_row(B, S.G, r, w, scale), r)
