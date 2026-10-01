"""Element-wise FP32, BF16 and I32 arithmetic as Boolean Definitions: the next version of each word primitive, the same function
on bits.

    F32Add_v3, F32Mul_v3, F32Div_v3   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: their _v1
    F32Fma_v3                         (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>     word view: F32Fma_v1
    Bf16AddF2fp_v2                    (a, b: Array<16, Value<1>>) -> Array<16, Value<1>>        word view: Bf16AddF2fp_v1
    I32Add_v2                         (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: I32Add_v1

Each is traced from the builder C-Flock's lowering runs for its word primitive (`trace.definition`), so its And gates are that
builder's live ANDs: `fp.f32_add` / `f32_mul` / `f32_div` with the IR's NaN words (numpy float32 on x86-64), `fp.f32_fma`, the
f32 add of the widened operands rounded by `fp.f32_to_bf16` to F2FP's NaN, and `forms.add` at 32 bits. None is flushed: a
subnormal operand or result is kept, as the `_v1` words keep it. `F32Div_v3` is the IEEE divider (`div.rn.f32`, which
`F32Div_v1` is); the MUFU pieces of `div.full.f32` (`DivFullRcp_v1`, `DivFullScaleA_v1`) are their own Calls, not inside it.

`bf16_band` is the piece a function of one bf16 word reads where it is neither exact arithmetic nor a constant: a band of
exponent fields per sign, held as a constant table of the function's own values and read with plain gates (one-hot minterms
and fixed XORs of them).
"""
from __future__ import annotations

from typing import Any, Callable

from verity.ir import boolean as BL
from verity.ir.types import Array

from .. import fp32 as W
from .. import scalar as S
from . import fp
from .forms import NOT, ZERO, add, and_reduce
from .trace import definition

__all__ = ["Bf16AddF2fp", "F32Add", "F32Div", "F32Fma", "F32Mul", "I32Add", "bf16_band"]

F32 = I32 = Array(32, BL.BIT)
BF16 = Array(16, BL.BIT)
_AB = (("a", F32), ("b", F32))


def _blocks(ks) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in ks:
        out[k >> 2] = out.get(k >> 2, 0) | 1 << (k & 3)
    return out


def bf16_band(C, g, value: Callable[[int], int], lo: int, hi: tuple[int, int]):
    """The 15 magnitude bits of ``value(g)`` for a bf16 word g (LSB first) whose exponent field lies in ``[lo, hi[sign]]``, and
    0 for every other g.

    g's mantissa and lowest exponent bit decode to 256 one-hot minterms, and each row of the band (a sign and an exponent pair
    e, e + 1) to one minterm; each output bit of a row is one AND of the row's minterm with a fixed XOR of low minterms, or its
    complement, whichever has fewer terms. A column's XOR is a chain over blocks of four in ascending order, each block's part
    a chain over its minterms, so equal parts and equal prefixes are one wire."""
    if not all(0 <= lo <= h <= 255 for h in hi):
        raise ValueError(f"not a band of exponent fields: [{lo}, {hi}]")
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    low = fp.decode(C, m + [e[0]])
    high = fp.decode(C, e[1:4] + [s])
    nibbles: dict[int, Any] = {}
    parts: dict[tuple[int, int], Any] = {}

    def part(b: int, mask: int):
        if (b, mask) not in parts:
            top = mask.bit_length() - 1
            rest = mask ^ 1 << top
            parts[(b, mask)] = part(b, rest) ^ low[4 * b + top] if rest else low[4 * b + top]
        return parts[(b, mask)]

    out = [ZERO] * 15
    for sign in (0, 1):
        for e0 in range(lo & ~1, hi[sign] + 1, 2):
            n = e0 >> 4
            if n not in nibbles:
                nibbles[n] = and_reduce(C, [e[4 + i] if n >> i & 1 else NOT(e[4 + i]) for i in range(4)])
            row = C.AND(high[(e0 & 0xF) >> 1 | sign << 3], nibbles[n])
            words = [value(sign << 15 | (e0 | k >> 7) << 7 | k & 0x7F) & 0x7FFF if lo <= e0 | k >> 7 <= hi[sign] else 0
                     for k in range(256)]
            for j in range(15):
                ones = [k for k in range(256) if words[k] >> j & 1]
                if len(ones) == 256:
                    out[j] = out[j] ^ row
                elif ones:
                    blocks, zeros = _blocks(ones), _blocks(sorted(set(range(256)) - set(ones)))
                    pick, flip = (blocks, False) if len(blocks) <= len(zeros) else (zeros, True)
                    col = None
                    for b in sorted(pick):
                        col = part(b, pick[b]) if col is None else col ^ part(b, pick[b])
                    out[j] = out[j] ^ C.AND(row, NOT(col) if flip else col)
    return out


def bf16_add_f2fp(C, a, b):
    """``Bf16AddF2fp_v1``: ``bf16_rn(f32(a) + f32(b))``, every NaN 0x7FFF."""
    return fp.f32_to_bf16(C, fp.f32_add(C, fp.bf16_to_f32(a), fp.bf16_to_f32(b)), W.BF16_NAN_F2FP)


F32Add = definition("F32Add", 3, lambda C, a, b: fp.f32_add(C, a, b, ir_nan=True), _AB, F32, word=W.F32Add,
                    doc="FADD on bits: F32Add_v1, RN(a + b), subnormals kept, numpy float32's NaN words")
F32Mul = definition("F32Mul", 3, lambda C, a, b: fp.f32_mul(C, a, b, ir_nan=True), _AB, F32, word=W.F32Mul,
                    doc="FMUL on bits: F32Mul_v1, RN(a * b), subnormals kept, numpy float32's NaN words")
F32Fma = definition("F32Fma", 3, fp.f32_fma, (("a", F32), ("b", F32), ("c", F32)), F32, word=W.F32Fma,
                    doc="FFMA on bits: F32Fma_v1, RN(a * b + c) with one rounding, subnormals kept, every NaN 0x7FC00000")
F32Div = definition("F32Div", 3, lambda C, a, b: fp.f32_div(C, a, b, ir_nan=True), _AB, F32, word=W.F32Div,
                    doc="div.rn.f32 on bits: F32Div_v1, RN(a / b), subnormals kept, numpy float32's NaN words")
Bf16AddF2fp = definition("Bf16AddF2fp", 2, bf16_add_f2fp, (("a", BF16), ("b", BF16)), BF16, word=W.Bf16AddF2fp,
                         doc="the fused_add_rms_norm residual add on bits: Bf16AddF2fp_v1, bf16_rn(f32(a) + f32(b)), NaN 0x7FFF")
I32Add = definition("I32Add", 2, lambda C, a, b: add(C, a, b, width=32), (("a", I32), ("b", I32)), I32, word=S.I32Add,
                    doc="wrapping 32-bit add on bits: I32Add_v1, a ripple-carry adder")
