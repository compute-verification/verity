"""The registry's scalar f32 primitives (`registry.prims`, `registry.moe`) as Boolean Definitions: the next version of each word
primitive, the same function on bits (`verity.ml.circuits.boolean.trace`).

    DivFullScaleA_v4   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: DivFullScaleA_v2
    F32Sub_v3          (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32Sub_v2
    F32Sub_v4          (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32Sub_v1 (ref_prims)
    F32FmaRm_v3        (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32FmaRm_v1
    F32FmaRm_v4        (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: F32FmaRm_v2
    F32GtStrict_v2     (a, b: Array<32, Value<1>>) -> Value<1>                  word view: F32GtStrict_v1

`DivFullScaleA_v4` is `verity.ml.circuits.ops.boolean.mufu.div_full_scale_a_gpu` (`DivFullScaleA_v3`'s scalings, each an FMUL whose
NaN is 0x7FFFFFFF); `F32Sub_v3` is FADD of a and -b with every NaN result 0x7FFFFFFF, so no operand NaN's payload or sign
survives and numpy's quieting rule does not arise, and `F32Sub_v4` is the same FADD with every NaN 0x7FC00000 (`ref_prims.F32Sub_v1`
canonicalizes numpy's NaN); the `fma.rm.f32` pair is `verity.ml.circuits.silicon.fp.f32_fma`
rounded toward -infinity, every NaN 0x7FC00000 (v3) or 0x7FFFFFFF (v4); `F32GtStrict_v2` is
`verity.ml.circuits.ops.boolean.softmax.f32_gt` (NaN false, +0 == -0).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import softmax as SM
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array

from . import moe as M
from . import prims as P
from . import ref_prims as R

__all__ = ["DivFullScaleA", "F32FmaRm", "F32FmaRmV4", "F32GtStrict", "F32Sub", "F32SubRef"]

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
