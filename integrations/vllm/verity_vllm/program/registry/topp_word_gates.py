"""The top-p keep word's word gates: each operation the builder (`topp_keep_builder`) applies, as a word-sized primitive or a small
composite of them.

* :data:`SIG`: each operation's operand and result kinds (f an f32 or u32 word, i a count, b a bit); :data:`WIDTH`, their bits.
* :data:`GATES`: each operation's word gate.  It is the registry's primitive where one computes the operation (F32 add / sub / mul /
  fma, compares, selects, the i32 ops, DivFullRcp, MufuEx2Ftz); else a small composite of them (`tl.exp`, `div.full`, `>=`,
  `|a - b| < 1e-9`, Python's and numpy's max / min, `<`), one body over calls (`k.call`, `k.c`) that `topp_words.IREngine` emits and
  :func:`numeric` evaluates on ints; or one of six new primitives (a bit XOR, a word XOR, a bit select, i32 -> f32, `__nv_logf`, the
  combine's `numkeep`).  A count is an i32 (the builder's 20-bit count, zero-extended: every count is below V < 2^19).
"""
from __future__ import annotations

import struct
from functools import cache

import numpy as np

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.ops.scalar import BIT, BitAnd, BitNot, BitOr, F32, F32Eq, F32Fabs, I32, I32Add, I32Eq, I32Le, SelectF32, SelectI32
from verity.ml.circuits.ops.fp32 import F32Add, F32Fma, F32Mul
from verity.ml.circuits.ops.mufu import DivFullRcp, DivFullScaleA, MufuEx2Ftz
from verity_vllm.program.registry.ref_prims import F32GtStrict, F32Sub

#: bits of a count word inside the builder (a lane count; V < 2^(CW - 1)); a count gate reads it as an i32
CW = 20
M32 = 0xFFFFFFFF


def f32_word(v) -> int:
    """The f32 nearest ``v``, as its word."""
    return int(np.float32(v).view(np.uint32))


QNAN = 0x7FFFFFFF
LOG2E = f32_word(1.4426950408889634)
M126, HALF, EPS9 = f32_word(-126.0), f32_word(0.5), f32_word(1e-9)


def _f(a: int) -> float:
    return struct.unpack("<f", struct.pack("<I", a & M32))[0]


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]


# --- the new word gates ------------------------------------------------------------------------------------------------------

def _xor(a: int, b: int) -> int:
    return (a ^ b) & 1


def _uxor(a: int, b: int) -> int:
    return (a ^ b) & M32


def _selb(c: int, a: int, b: int) -> int:
    return (a if c & 1 else b) & 1


def _i2f(a: int) -> int:
    """The i32 ``a`` as the f32 nearest it (ties to even): ``cvt.rn.f32.s32``."""
    a &= M32
    return _bits(np.float32(a - (1 << 32) if a & 0x80000000 else a))


def _numkeep(nd: int, q: int) -> int:
    """``combine``'s ``numkeep = min(max(numdup - qi, 1), numdup)`` on u32 counts, qi the f32 ``q`` truncated to a u32 (0 unless q is
    finite and > 0, 2^32 - 1 from 2^32 on); 0 when numdup is 0."""
    nd &= M32
    x = _f(q)
    qi = 0 if not (x > 0 and np.isfinite(x)) else min(int(x), M32)
    if nd == 0:
        return 0
    return nd - qi if qi < nd else 1


def _logf(a: int) -> int:
    from verity_vllm.program.registry.sampling import nv_logf
    return nv_logf(a)


def _logf_gpu(a: int) -> int:
    from verity_vllm.program.registry.sampling import nv_logf_gpu
    return nv_logf_gpu(a)


BitXor = PrimitiveDefinition("BitXor", 1, (("a", BIT), ("b", BIT)), BIT, _xor, doc="a XOR b on bits.", conformance="exact-model-tested")
U32Xor = PrimitiveDefinition("U32Xor", 1, (("a", I32), ("b", I32)), I32, _uxor, doc="a XOR b on 32-bit words.",
                             conformance="exact-model-tested")
SelectBit = PrimitiveDefinition("SelectBit", 1, (("c", BIT), ("a", BIT), ("b", BIT)), BIT, _selb, doc="c ? a : b on bits.",
                                conformance="exact-model-tested")
I32ToF32 = PrimitiveDefinition("I32ToF32", 1, (("a", I32),), F32, _i2f, doc=_i2f.__doc__, conformance="exact-model-tested")
NvLogf = PrimitiveDefinition("NvLogf", 1, (("a", F32),), F32, _logf, doc="libdevice `__nv_logf` on an fp32 word (FTZ): "
                             "`sampling.nv_logf`, `tl.log` in the top-p pipeline.", conformance="exact-model-tested")
NvLogfV2 = PrimitiveDefinition("NvLogf", 2, (("a", F32),), F32, _logf_gpu, doc="`NvLogf_v1` with every NaN result the GPU's "
                               "0x7FFFFFFF (`sampling.nv_logf_gpu`), for the re-baseline.", conformance="exact-model-tested")
TopPNumkeep = PrimitiveDefinition("TopPNumkeep", 1, (("numdup", I32), ("q", F32)), I32, _numkeep, doc=_numkeep.__doc__,
                                  conformance="exact-model-tested")


# --- the composite word gates: one body each, built as IR or evaluated on ints ------------------------------------------------

class _Num:
    """Evaluate a body on ints."""

    @staticmethod
    def call(fn, *a):
        return fn.evaluate(*a)

    @staticmethod
    def c(v, w=32):
        return v


def _tl_exp(k, x):
    """``topp_split.tl_exp``: MUFU.EX2(x * log2e), with ptxas' non-ftz path (t < -126: EX2(t / 2) squared)."""
    t = k.call(F32Mul, x, k.c(LOG2E))
    small = k.call(F32GtStrict, k.c(M126), t)
    e = k.call(MufuEx2Ftz, k.call(SelectF32, small, k.call(F32Mul, t, k.c(HALF)), t))
    return k.call(SelectF32, small, k.call(F32Mul, e, e), e)


def _tl_div(k, a, b, r):
    """``div.full.f32``'s a / b from ``r = DivFullRcp(b)``: ``F32Mul(DivFullScaleA(a, b), r)``."""
    return k.call(F32Mul, k.call(DivFullScaleA, a, b), r)


def _ge(k, a, b):
    return k.call(BitOr, k.call(F32GtStrict, a, b), k.call(F32Eq, a, b))


def _near(k, a, b):
    """``|a - b| < 1e-9`` in f32."""
    return k.call(F32GtStrict, k.c(EPS9), k.call(F32Fabs, k.call(F32Sub, a, b)))


def _py_max(k, a, b):
    """Python's ``max(a, b)``: ``b > a ? b : a``."""
    return k.call(SelectF32, k.call(F32GtStrict, b, a), b, a)


def _py_min(k, a, b):
    """Python's ``min(a, b)``: ``b < a ? b : a``."""
    return k.call(SelectF32, k.call(F32GtStrict, a, b), b, a)


def _nan(k, a, b):
    return k.call(BitOr, k.call(BitNot, k.call(F32Eq, a, a)), k.call(BitNot, k.call(F32Eq, b, b)))


def _nan_max(k, a, b):
    """numpy's ``maximum``: NaN when either is a NaN."""
    return k.call(SelectF32, _nan(k, a, b), k.c(QNAN), _py_max(k, a, b))


def _nan_min(k, a, b):
    """numpy's ``minimum``: NaN when either is a NaN."""
    return k.call(SelectF32, _nan(k, a, b), k.c(QNAN), _py_min(k, a, b))


def _ilt(k, a, b):
    return k.call(BitNot, k.call(I32Le, b, a))


#: each operation's operand and result kinds: f an f32 (or u32) word, i a count, b a bit
SIG = {
    "sub": ("ff", "f"), "add": ("ff", "f"), "mul": ("ff", "f"), "fma": ("fff", "f"), "exp": ("f", "f"), "rcp": ("f", "f"),
    "div": ("fff", "f"), "log": ("f", "f"), "gt": ("ff", "b"), "ge": ("ff", "b"), "eq": ("ff", "b"), "near": ("ff", "b"),
    "max": ("ff", "f"), "min": ("ff", "f"), "nanmax": ("ff", "f"), "nanmin": ("ff", "f"), "sel": ("bff", "f"),
    "seli": ("bii", "i"), "selb": ("bbb", "b"), "gate": ("bf", "f"), "gatei": ("bi", "i"), "and": ("bb", "b"),
    "or": ("bb", "b"), "xor": ("bb", "b"), "xorf": ("ff", "f"), "xori": ("ii", "i"), "not": ("b", "b"), "iadd": ("ii", "i"),
    "ile": ("ii", "b"), "ilt": ("ii", "b"), "i2f": ("i", "f"), "b2i": ("b", "i"), "numkeep": ("if", "i"), "weq": ("ff", "b"),
}
WIDTH = {"f": 32, "i": CW, "b": 1}
#: each operation's word gate, or its expansion into word gates (a function of the calls it makes), and the constant operands it
#: adds after the operation's (a gate as a select of 0, a bit as a select of 1 or 0)
GATES = {
    "sub": (F32Sub, ()), "add": (F32Add, ()), "mul": (F32Mul, ()), "fma": (F32Fma, ()), "exp": (_tl_exp, ()),
    "rcp": (DivFullRcp, ()), "div": (_tl_div, ()), "log": (NvLogf, ()), "gt": (F32GtStrict, ()), "ge": (_ge, ()),
    "eq": (F32Eq, ()), "near": (_near, ()), "max": (_py_max, ()), "min": (_py_min, ()), "nanmax": (_nan_max, ()),
    "nanmin": (_nan_min, ()), "sel": (SelectF32, ()), "seli": (SelectI32, ()), "selb": (SelectBit, ()),
    "gate": (SelectF32, ((0, 32),)), "gatei": (SelectI32, ((0, 32),)), "and": (BitAnd, ()), "or": (BitOr, ()),
    "xor": (BitXor, ()), "xorf": (U32Xor, ()), "xori": (U32Xor, ()), "not": (BitNot, ()), "iadd": (I32Add, ()),
    "ile": (I32Le, ()), "ilt": (_ilt, ()), "i2f": (I32ToF32, ()), "b2i": (SelectI32, ((1, 32), (0, 32))),
    "numkeep": (TopPNumkeep, ()), "weq": (I32Eq, ()),
}
@cache
def numeric(name: str):
    """The operation on ints, by its word gate (a composite's body evaluated on its primitives)."""
    fn, extra = GATES[name]
    ev = fn.evaluate if isinstance(fn, PrimitiveDefinition) else (lambda *a: fn(_Num, *a))
    return lambda *a: ev(*a, *(v for v, _w in extra))


__all__ = ["CW", "M32", "QNAN", "LOG2E", "M126", "HALF", "EPS9", "f32_word", "BitXor", "U32Xor", "SelectBit", "I32ToF32", "NvLogf",
           "NvLogfV2", "TopPNumkeep", "SIG", "WIDTH", "GATES", "numeric"]
