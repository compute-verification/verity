"""The scalar comparison and selects as Boolean Definitions: the next version of each word primitive, the same function on bits.

    Bf16GtStrict_v2   (a, b: Array<16, Value<1>>) -> Value<1>                            word view: Bf16GtStrict_v1
    SelectBf16_v2     (c: Value<1>, a, b: Array<16, Value<1>>) -> Array<16, Value<1>>    word view: SelectBf16_v1
    SelectI32_v2      (c: Value<1>, a, b: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: SelectI32_v1

Each is traced from a builder over `forms` (`trace.definition`). The builders are also pieces of larger circuits, such as an
argmax over a logits row: `bf16_key` maps a bf16 word to a NaN flag and a 16-bit key whose unsigned order is the value order
(+0 and -0 one key), `f32_key` an f32 word likewise, `gt_u` compares two keys, `select` is the bitwise multiplexer.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.types import Array

from .. import scalar as S
from .forms import NOT, OR, and_reduce, mux, or_reduce, sub
from .trace import definition

__all__ = ["Bf16GtStrict", "SelectBf16", "SelectI32", "bf16_gt_strict", "bf16_key", "f32_key", "gt_u", "select"]

BIT = BL.BIT
BF16 = Array(16, BIT)
I32 = Array(32, BIT)


def bf16_key(C, x):
    """``(nan, key)`` of a bf16 word: ``key`` (16 bits, LSB first) orders the non-NaN values as unsigned integers, with one key
    for +0 and -0: a positive word with its sign bit set, a negative nonzero word with its magnitude complemented and its sign
    bit clear (-0 is +0's 0x8000). A NaN's key is arbitrary; ``nan`` says it is one."""
    m, e, s = x[:7], x[7:15], x[15]
    e_any, m_any = or_reduce(C, e), or_reduce(C, m)
    nan = C.AND(and_reduce(C, e), m_any)
    neg = C.AND(s, OR(C, e_any, m_any))
    return nan, [b ^ neg for b in x[:15]] + [NOT(neg)]


def f32_key(C, x):
    """`bf16_key` of an f32 word: ``(nan, key)``, ``key`` 32 bits whose unsigned order is the order of the non-NaN values
    (`F32GtStrict_v1`'s: subnormals as they are, +0 and -0 one key)."""
    m, e, s = x[:23], x[23:31], x[31]
    e_any, m_any = or_reduce(C, e), or_reduce(C, m)
    nan = C.AND(and_reduce(C, e), m_any)
    neg = C.AND(s, OR(C, e_any, m_any))
    return nan, [b ^ neg for b in x[:31]] + [NOT(neg)]


def gt_u(C, a, b):
    """``a > b`` for unsigned words of one width (LSB first): the borrow of ``b - a``."""
    return sub(C, b, a, len(a))[1]


def bf16_gt_strict(C, a, b):
    """``Bf16GtStrict_v1``: a > b on bf16 values (a NaN on either side compares false; +0 and -0 are equal)."""
    na, ka = bf16_key(C, a)
    nb, kb = bf16_key(C, b)
    return [C.AND(NOT(OR(C, na, nb)), gt_u(C, ka, kb))]


def select(C, c, a, b):
    """``c ? a : b`` bitwise, ``c`` one bit (one AND per bit)."""
    return mux(C, c, a, b)


Bf16GtStrict = definition("Bf16GtStrict", 2, bf16_gt_strict, (("a", BF16), ("b", BF16)), BIT, word=S.Bf16GtStrict,
                          doc="a > b on bf16 bits: Bf16GtStrict_v1 (NaN compares false, +0 == -0)")
SelectBf16 = definition("SelectBf16", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", BF16), ("b", BF16)), BF16,
                        word=S.SelectBf16, doc="c ? a : b on bf16 bits: SelectBf16_v1")
SelectI32 = definition("SelectI32", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", I32), ("b", I32)), I32,
                       word=S.SelectI32, doc="c ? a : b on i32 bits: SelectI32_v1")
