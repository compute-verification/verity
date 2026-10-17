"""The exact s8 matmul family as Boolean Definitions: the next version of each word Definition of `ops.s8` and of its two
steps, the same function on bits.

    BlackwellS8ImmaDot32_v2   (acc: Array<32, Value<1>>, a, b: Array<32, Array<8, Value<1>>>) -> Array<32, Value<1>>
                                                                                  word view: BlackwellS8ImmaDot32_v1
    BlackwellU8S8ImmaDot32_v2 the same signature (a's codes unsigned)             word view: BlackwellU8S8ImmaDot32_v1
    S8Dot_v2{K, L, T0, C, DOT, A, B, TAIL, TDOT, NOISE, LIMBS, COMMIT}
                              S8Dot_v1's signature on bits                         word view: S8Dot_v1{..., DOT', TDOT'}

A step is traced from `_imma` (`trace.definition`): the 32·64 bit products of the operand pairs and the accumulator's bits in
carry-save columns modulo 2^34, where the exact sum fits (a two's-complement bit of negative weight, x·(-2^p), enters as
NOT(x)·2^p and the constant -2^p), one ripple, then `.satfinite`'s clamp to s32 (when bits 31..33 disagree: 0x80000000 for a
negative sum, else 0x7FFFFFFF) or the low 32 bits.  `S8Dot_v2` is `S8Dot_v1`'s body on bits over Boolean steps, with the same
statics, bind-time checks and census rule; ``DOT'`` is `DOT`'s word Definition (`trace.word_of`).
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, compress_columns, mux, or_reduce
from verity.ml.circuits.boolean.trace import definition, word_of
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, broadcast
from verity.ml.circuits.types import Array

from .. import prims as P
from .. import s8 as S8

__all__ = ["BlackwellS8ImmaDot32", "BlackwellU8S8ImmaDot32", "S8Dot"]

BIT = BL.BIT
S32 = Array(32, BIT)
CODE = Array(8, BIT)
A32_8 = Array(32, CODE)

#: the columns' width: |acc + 32 products| < 2^33 for every s8 or u8 by s8 operand
_M = 34


def _imma(C, acc, a, b, a_signed: bool, satfinite: bool):
    cols: list[list] = [[] for _ in range(_M)]
    const = 0
    for i, x in enumerate(acc):
        if i == 31:
            cols[i].append(NOT(x))
            const -= 1 << i
        else:
            cols[i].append(x)
    for ak, bk in zip(a, b):
        for i, x in enumerate(ak):
            for j, y in enumerate(bk):
                p = C.AND(x, y)
                if (i == 7 and a_signed) != (j == 7):
                    cols[i + j].append(NOT(p))
                    const -= 1 << (i + j)
                else:
                    cols[i + j].append(p)
    const %= 1 << _M
    for i in range(_M):
        if const >> i & 1:
            cols[i].append(ONE)
    s = compress_columns(C, cols)[:_M]
    if not satfinite:
        return s[:32]
    sign = s[_M - 1]
    ovf = or_reduce(C, [s[i] ^ sign for i in range(31, _M - 1)])
    return mux(C, ovf, [NOT(sign)] * 31 + [sign], s[:32])


BlackwellS8ImmaDot32 = definition(
    "BlackwellS8ImmaDot32", 2, lambda C, acc, *ab: _imma(C, acc, ab[:32], ab[32:], True, True),
    (("acc", S32), ("a", A32_8), ("b", A32_8)), S32, word=P.BlackwellS8ImmaDot32,
    doc="mma.sync.m16n8k32 .satfinite s32.s8.s8.s32 on sm_120, one accumulator coordinate, on bits: BlackwellS8ImmaDot32_v1")
BlackwellU8S8ImmaDot32 = definition(
    "BlackwellU8S8ImmaDot32", 2, lambda C, acc, *ab: _imma(C, acc, ab[:32], ab[32:], False, False),
    (("acc", S32), ("a", A32_8), ("b", A32_8)), S32, word=P.BlackwellU8S8ImmaDot32,
    doc="mma.sync.m16n8k32 s32.u8.s8.s32 on sm_120, one accumulator coordinate, on bits: BlackwellU8S8ImmaDot32_v1")


def _zero(B) -> Coll:
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(S32, broadcast(space, idx, leaf, 32))


@composite("S8Dot", 2, list(S8._DOT_STATICS), lambda S: S8._signature(S, S32, CODE), conformance="exact-model-tested")
def S8Dot(B, S, *args):
    """`S8Dot_v1`'s chain and tail on bits, over Boolean steps."""
    return S8._dot_body(B, S, args, _zero, S32, CODE)


def _word(b: dict) -> Any:
    return bind(S8.S8Dot, **b | {"DOT": word_of(b["DOT"]), "TDOT": None if b["TDOT"] is None else word_of(b["TDOT"])})


S8Dot.word = _word  # type: ignore[attr-defined]
S8Dot.counts = S8.counts  # type: ignore[attr-defined]
