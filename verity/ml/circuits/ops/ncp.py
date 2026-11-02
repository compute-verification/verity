"""Registered `ncp-v2` dequantization: `PouwI32ToF32Rn_v1`, cvt.rn.f32.s32, and `NcpDequant_v1`, a credited Z's served
BF16 y = bf16(fp32(s)·fp32(t)·Z) + bias, the unit PoUW's `ncp2.linear` ends each output entry with and a served PoUW
linear's row (`NcpLinearRow_v1`) ends with too.

Provenance.  PoUW registered both in `verity.ml.protocols.pouw.circuit` (`words.PouwI32ToF32Rn`, `ncp2.NcpDequant`: same
ids, signatures, conformance status and bodies), which re-exports them from here.
"""
from __future__ import annotations

from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind, primitive
from verity.ml.circuits.types import Value

import verity.ml.circuits.ops.fp32  # registers F32Mul_v2, Bf16Add_v1
import verity.ml.circuits.ops.prims  # noqa: F401  (registers F32ToBf16Rn_v1)

U32, F32, BF16 = Value(32), Value(32), Value(16)


def i32_to_f32_rn(z: int) -> int:
    """cvt.rn.f32.s32: an int32 as the nearest FP32, ties to even, as its encoding."""
    v = (z & 0xFFFFFFFF) - (1 << 32) if z & 0x80000000 else z & 0xFFFFFFFF
    if v == 0:
        return 0
    sign = 0x80000000 if v < 0 else 0
    m = -v if v < 0 else v
    e = m.bit_length() - 1                                  # m = 1.f × 2^e, e <= 31
    if e > 23:
        shift = e - 23
        q, r = m >> shift, m & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):
            q += 1
        if q >> 24:                                         # rounding carried into a new binade
            q >>= 1
            e += 1
        mant = q & 0x7FFFFF
    else:
        mant = (m << (23 - e)) & 0x7FFFFF
    return sign | ((e + 127) << 23) | mant


@primitive("PouwI32ToF32Rn", 1, [("z", U32)], F32, conformance="protocol-defined")
def PouwI32ToF32Rn(z: int) -> int:
    """cvt.rn.f32.s32: Z as FP32, round to nearest even."""
    return i32_to_f32_rn(z)


def _dequant_body(B, S, z, s, t, bias):
    st = B.call(REGISTRY["F32Mul_v2"], s, t)
    v = B.call(REGISTRY["F32Mul_v2"], st, B.call(PouwI32ToF32Rn, z))
    return B.call(REGISTRY["Bf16Add_v1"], B.call(REGISTRY["F32ToBf16Rn_v1"], v), bias)


NcpDequantDef = CompositeDefinition(
    "NcpDequant", 1, (), lambda S: ((("z", U32), ("s", F32), ("t", F32), ("bias", BF16)), BF16), _dequant_body,
    doc="y = bf16(fp32(s_i)·fp32(t_j)·Z) + bias_j: a width-rule unit after the Z separator.", conformance="protocol-defined")

NcpDequant = bind(NcpDequantDef)


__all__ = ["NcpDequant", "NcpDequantDef", "PouwI32ToF32Rn", "i32_to_f32_rn"]
