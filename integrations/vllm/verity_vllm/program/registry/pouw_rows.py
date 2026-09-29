"""PoUW's `ncp-v2` in the model Program: `NcpLinearRow_v1{K,N}`, one row of a PoUW linear's served output.

A PoUW linear's call is its own `NcpLinear_v1` Program call (`verity_pouw.circuit`), and its served y depends on x and W
alone: the noise cancels in Z = A·Bᵀ, which is exact. So one row of the model Program at a PoUW site is this Definition.
It quantizes the row and each weight row with the adapter's quantizer (`protocol_options.pouw.quantize`,
``verity.pouw.symmetric-int7-rowmax-rne-f32/v1``):
    scale = RN_f32(rowmax |v| · RN_f32(1/63)), 1 for a zero row;  q = clamp(RNE(RN_f32(v / scale)), −63, 63).
It then takes the exact int32 dot product in ascending k from 0, and applies the circuit's dequantization (`NcpDequant`)
with no bias: y_j = bf16(fp32(s)·fp32(t_j)·Z_j).

The quantizer's max is `NcpAbsMax` rather than `F32Max_v1`, whose operands are flushed to zero as the fast-math FA2 build
does: a row of subnormals has a nonzero max here, as it does in numpy.
"""
from __future__ import annotations

import math
import struct

from verity.ir.defs import composite, primitive
from verity.ir.refs import array_of
from verity.ir.types import Array, Value
from verity.ml import fp32 as F
from verity.ml import prims as P
from verity.ml import scalar as SC
from verity_pouw.circuit.ncp2 import NcpDequant

BF16, F32, U32, I7, BIT = Value(16), Value(32), Value(32), Value(7), Value(1)
A16I7 = Array(16, I7)
M32 = (1 << 32) - 1


def _inv63_bits() -> int:
    import numpy as np
    return int((np.float32(1.0) / np.float32(63.0)).view(np.uint32))


ONE32 = P.const(32, 0x3F800000)
ZERO32 = P.const(32, 0)
ZERO16 = P.const(16, 0)
INV63_32 = P.const(32, _inv63_bits())


def _sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


@primitive("NcpAbsMax", 1, [("m", F32), ("x", F32)], F32, conformance="protocol-defined")
def NcpAbsMax(m: int, x: int) -> int:
    """max(m, |x|) on encodings, compared as unsigned words: the order of non-negative FP32 values (no flush to zero)."""
    a = x & 0x7FFFFFFF
    return a if a > (m & 0xFFFFFFFF) else m & 0xFFFFFFFF


@primitive("NcpRneClampI7", 1, [("q", F32)], I7, conformance="protocol-defined")
def NcpRneClampI7(q: int) -> int:
    """clamp(RNE(q), −63, 63) as an int7 encoding; a NaN is 0 (the quantizer refuses non-finite input before this)."""
    v = struct.unpack("<f", struct.pack("<I", q & M32))[0]
    if math.isnan(v):
        return 0
    r = 63 if v >= 63.5 else (-63 if v <= -63.5 else round(v))                 # Python's round is to nearest even
    return max(-63, min(63, int(r))) & 0x7F


@primitive("NcpDot16I7", 1, [("acc", U32), ("a", A16I7), ("b", A16I7)], U32, conformance="protocol-defined")
def NcpDot16I7(acc: int, *ab: int) -> int:
    """acc + Σ a_i·b_i over 16 int7 products, mod 2^32."""
    a, b = ab[:16], ab[16:]
    return (acc + sum(_sx(x, 7) * _sx(y, 7) for x, y in zip(a, b))) & M32


def _quant(B, K, row):
    """(the int7 entries, the FP32 scale) of one bf16 row of K."""
    xf = [B.call(P.Bf16ToF32, row[i]) for i in range(K)]
    m = B.call(ZERO32)
    for v in xf:
        m = B.call(NcpAbsMax, m, v)
    scale = B.call(F.F32Mul, m, B.call(INV63_32))
    scale = B.call(SC.SelectF32, B.call(SC.F32Eq, scale, B.call(ZERO32)), B.call(ONE32), scale)
    return [B.call(NcpRneClampI7, B.call(F.F32Div, v, scale)) for v in xf], scale


@composite("NcpLinearRow", 1, ["K", "N"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="protocol-defined")
def NcpLinearRow(B, S, x, w):
    """One row of a PoUW linear's served y: `x[K]` and `W[N,K]` quantized per row, the exact dot product, `NcpDequant` with no bias."""
    if S.K % 16:
        raise TypeError(f"NcpLinearRow: K = {S.K} is not a multiple of 16")
    a, s = _quant(B, S.K, x)
    ys = []
    for j in range(S.N):
        b, t = _quant(B, S.K, w[j])
        z = B.call(ZERO32)
        for c in range(S.K // 16):
            z = B.call(NcpDot16I7, z, array_of(a[16 * c:16 * c + 16]), array_of(b[16 * c:16 * c + 16]))
        ys.append(B.call(NcpDequant, z, s, t, B.call(ZERO16)))
    return array_of(ys)


__all__ = ["NcpAbsMax", "NcpDot16I7", "NcpLinearRow", "NcpRneClampI7"]
