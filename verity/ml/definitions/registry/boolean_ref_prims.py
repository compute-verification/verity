"""`ref_prims`' scalar primitives on bits: version 2 of each, the same function on every word (`verity.ml.circuits.boolean.trace`).

    I64ToI32_v2     (t: Array<64, Value<1>>) -> Array<32, Value<1>>        word view: I64ToI32_v1
    F32MaxRef_v2    (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>     word view: F32MaxRef_v1
    F32Sqrt_v2      (x: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: F32Sqrt_v1

`I64ToI32_v2` keeps t's low word when bits 31 to 63 are all equal (t fits in an i32), else gives 0x80000000. `F32MaxRef_v2` orders
the operands by `scalar.f32_key` (one key for +0 and -0) and takes b only when its key is strictly greater, so a tie keeps a; a NaN
on either side gives 0x7FC00000. `F32Sqrt_v2` is `sqrt.f32_sqrt_rn` (`PouwF32SqrtRn_v2`'s circuit: IEEE nearest even, subnormals,
√-0 = -0, √+inf = +inf) with its NaN word 0x7FFFFFFF made 0x7FC00000: no other root has bits 22 to 30 all set, so those bits'
AND marks it and clears bits 0 to 21.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean.scalar import f32_key, gt_u
from verity.ml.circuits.ops.boolean.sqrt import f32_sqrt_rn
from verity.ml.circuits.types import Array

from . import ref_prims as RP

__all__ = ["F32MaxRef", "F32Sqrt", "I64ToI32", "f32_max_ref", "f32_sqrt", "i64_to_i32"]

BIT = BL.BIT
F32 = Array(32, BIT)
I32 = Array(32, BIT)
I64 = Array(64, BIT)
_SENTINEL = 0x80000000


def i64_to_i32(C, t):
    """``I64ToI32_v1``: t's low 32 bits when bits 31 to 63 agree, else 0x80000000."""
    fits = NOT(or_reduce(C, [b ^ t[31] for b in t[32:]]))
    return mux(C, fits, list(t[:32]), const_bits(_SENTINEL, 32))


def f32_max_ref(C, a, b):
    """``F32MaxRef_v1``: 0x7FC00000 when a or b is a NaN, else b when b > a, else a (+0 and -0 equal)."""
    na, ka = f32_key(C, a)
    nb, kb = f32_key(C, b)
    r = mux(C, gt_u(C, kb, ka), list(b), list(a))
    nan = NOT(C.AND(NOT(na), NOT(nb)))
    return mux(C, nan, const_bits(RP.F32_NAN, 32), r)


def f32_sqrt(C, x):
    """``F32Sqrt_v1``: `f32_sqrt_rn` with its NaN word 0x7FFFFFFF as 0x7FC00000."""
    r = f32_sqrt_rn(C, x)
    keep = NOT(and_reduce(C, r[22:31]))
    return [C.AND(keep, b) for b in r[:22]] + list(r[22:])


I64ToI32 = definition("I64ToI32", 2, i64_to_i32, (("t", I64),), I32, word=RP.I64ToI32,
                      doc="I64ToI32_v1 on bits: t when -2^31 <= t < 2^31, else 0x80000000")
F32MaxRef = definition("F32MaxRef", 2, f32_max_ref, (("a", F32), ("b", F32)), F32, word=RP.F32MaxRef,
                       doc="F32MaxRef_v1 on bits: NaN-propagating max (0x7FC00000), a on a tie")
F32Sqrt = definition("F32Sqrt", 2, f32_sqrt, (("x", F32),), F32, word=RP.F32Sqrt,
                     doc="F32Sqrt_v1 on bits: sqrt.rn.f32, a NaN or negative nonzero operand 0x7FC00000")
