"""The registry's scalar primitives (`registry.prims`, `registry.moe`, `registry.ref_prims`) as Boolean Definitions: the next
version of each word primitive, the same function on bits (`verity.circuits.boolean.trace`).

    DivFullScaleA_v4   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: DivFullScaleA_v2
    F32Sub_v3          (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32Sub_v2
    F32Sub_v4          (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32Sub_v1 (ref_prims)
    F32FmaRm_v3        (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32FmaRm_v1
    F32FmaRm_v4        (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32FmaRm_v2
    F32GtStrict_v2     (a, b: Array<32, Value<1>>) -> Value<1>                  word view: F32GtStrict_v1
    F32MaxRef_v2       (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32MaxRef_v1 (ref_prims)
    F32Sqrt_v2         (x: Array<32, Value<1>>) -> Array<32, Value<1>>          word view: F32Sqrt_v1 (ref_prims)
    I64ToI32_v2        (t: Array<64, Value<1>>) -> Array<32, Value<1>>          word view: I64ToI32_v1 (ref_prims)

`DivFullScaleA_v4` is `verity.definitions.ops.boolean.mufu.div_full_scale_a_gpu` (`DivFullScaleA_v3`'s scalings, each an FMUL whose
NaN is 0x7FFFFFFF); `F32Sub_v3` is FADD of a and -b with every NaN result 0x7FFFFFFF, so no operand NaN's payload or sign
survives and numpy's quieting rule does not arise, and `F32Sub_v4` is the same FADD with every NaN 0x7FC00000 (`ref_prims.F32Sub_v1`
canonicalizes numpy's NaN); the `fma.rm.f32` pair is `verity.definitions.silicon.fp.f32_fma`
rounded toward -infinity, every NaN 0x7FC00000 (v3) or 0x7FFFFFFF (v4); `F32GtStrict_v2` is
`verity.definitions.ops.boolean.softmax.f32_gt` (NaN false, +0 == -0).

`F32MaxRef_v2` orders the operands by `verity.definitions.ops.boolean.scalar.f32_key` (one key for +0 and -0) and takes b only
when its key is strictly greater, so a tie keeps a; a NaN on either side gives 0x7FC00000. `F32Sqrt_v2` is
`verity.definitions.ops.boolean.sqrt.f32_sqrt_rn` (`PouwF32SqrtRn_v2`'s circuit: nearest even, subnormals, √-0 = -0, √+inf =
+inf) with its NaN word 0x7FFFFFFF made 0x7FC00000: no other root has bits 22 to 30 all set, so their AND marks it and clears
bits 0 to 21; `benchmarks/f32_exhaustive/run.py boolean F32Sqrt` holds it to its word on all 2^32 words. `I64ToI32_v2` keeps
t's low word when bits 31 to 63 are all equal (t fits in an i32), else gives 0x80000000.
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, and_reduce, const_bits, mux, or_reduce
from verity.circuits.boolean.trace import definition
from verity.definitions.ops.boolean import mufu as BM
from verity.definitions.ops.boolean import softmax as SM
from verity.definitions.ops.boolean.scalar import f32_key, gt_u
from verity.definitions.ops.boolean.sqrt import f32_sqrt_rn
from verity.definitions.silicon import fp
from verity.circuits.types import Array

from . import moe as M
from . import prims as P
from . import ref_prims as R

__all__ = ["DivFullScaleA", "F32FmaRm", "F32FmaRmV4", "F32GtStrict", "F32MaxRef", "F32Sqrt", "F32Sub", "F32SubRef", "I64ToI32"]

F32 = Array(32, BL.BIT)
_AB = (("a", F32), ("b", F32))
_ABC = (("a", F32), ("b", F32), ("c", F32))

DivFullScaleA = definition("DivFullScaleA", 4, BM.div_full_scale_a_gpu, _AB, F32, word=P.DivFullScaleAV2,
                           doc="div.full.f32's numerator range scaling on bits: DivFullScaleA_v2, each scaling's NaN 0x7FFFFFFF")
F32Sub = definition("F32Sub", 3, lambda C, a, b: fp.f32_add(C, a, fp.f32_neg(b), nan_word=fp.F32_NAN_PTX), _AB, F32,
                    word=P.F32SubV2, doc="FSUB on bits: F32Sub_v2, RN(a - b), subnormals kept, every NaN 0x7FFFFFFF")
F32FmaRm = definition("F32FmaRm", 3, lambda C, a, b, c: fp.f32_fma(C, a, b, c, rm=True), _ABC, F32, word=M.F32FmaRm,
                      doc="fma.rm.f32 on bits: F32FmaRm_v1, a * b + c rounded once toward -infinity, every NaN 0x7FC00000")
F32FmaRmV4 = definition("F32FmaRm", 4, lambda C, a, b, c: fp.f32_fma(C, a, b, c, rm=True, nan_word=fp.F32_NAN_PTX), _ABC, F32,
                        word=M.F32FmaRmV2, doc="fma.rm.f32 on bits: F32FmaRm_v2, every NaN 0x7FFFFFFF")
F32GtStrict = definition("F32GtStrict", 2, SM.f32_gt, _AB, BL.BIT, word=R.F32GtStrict,
                         doc="a > b on f32 bits: F32GtStrict_v1, NaN compares false, +0 == -0")
F32SubRef = definition("F32Sub", 4, lambda C, a, b: fp.f32_sub(C, a, b), _AB, F32, word=R.F32Sub,
                       doc="FSUB on bits: F32Sub_v1 (ref_prims), RN(a - b), subnormals kept, every NaN 0x7FC00000")


def _f32_max_ref(C, a, b):
    na, ka = f32_key(C, a)
    nb, kb = f32_key(C, b)
    r = mux(C, gt_u(C, kb, ka), list(b), list(a))
    return mux(C, NOT(C.AND(NOT(na), NOT(nb))), const_bits(R.F32_NAN, 32), r)


def _f32_sqrt(C, x):
    r = f32_sqrt_rn(C, x)
    keep = NOT(and_reduce(C, r[22:31]))
    return [C.AND(keep, b) for b in r[:22]] + list(r[22:])


def _i64_to_i32(C, t):
    fits = NOT(or_reduce(C, [b ^ t[31] for b in t[32:]]))
    return mux(C, fits, list(t[:32]), const_bits(0x80000000, 32))


F32MaxRef = definition("F32MaxRef", 2, _f32_max_ref, _AB, F32, word=R.F32MaxRef,
                       doc="F32MaxRef_v1 on bits: NaN-propagating max (0x7FC00000), a on a tie")
F32Sqrt = definition("F32Sqrt", 2, _f32_sqrt, (("x", F32),), F32, word=R.F32Sqrt,
                     doc="F32Sqrt_v1 on bits: sqrt.rn.f32, a NaN or negative nonzero operand 0x7FC00000")
I64ToI32 = definition("I64ToI32", 2, _i64_to_i32, (("t", Array(64, BL.BIT)),), Array(32, BL.BIT), word=R.I64ToI32,
                      doc="I64ToI32_v1 on bits: t when -2^31 <= t < 2^31, else 0x80000000")
