"""Gemma-2's activations as Boolean Definitions: its GeGLU and its final-logit softcap, each the next version of the word Definition
the served Program calls, on bits.

    GeluTanhMulBf16_v2        (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>        word view: GeluTanhMulBf16_v1
    GeluTanhMul_v2{I}         (gu: Array<2I, bf16 bits>) -> Array<I, bf16 bits>          word view: GeluTanhMul_v1{I}
    TanhBf16_v2               (x: bf16 bits) -> bf16 bits                                word view: TanhBf16_v1
    Bf16Tanh_v2{N}            (x: Array<N, bf16 bits>) -> Array<N, bf16 bits>            word view: Bf16Tanh_v1{N}
    Bf16MulConst_v2{K}        (x: bf16 bits) -> bf16 bits                                word view: Bf16MulConst_v1{K}
    Bf16DivScalar_v2{N, C}    N Calls of Bf16MulConst_v2{K = RN32(1 / C)}                word view: Bf16DivScalar_v1{N, C}
    Bf16MulScalar_v2{N, C}    N Calls of Bf16MulConst_v2{K = f32(C)}                     word view: Bf16MulScalar_v1{N, C}

`TanhBf16_v1` and `Bf16MulConst_v1{K}` are the word composites of one element (`Bf16Tanh_v1`'s and the two scalar
composites' bodies at N = 1), so each element Definition has a word view; the served Program never calls them.  Each
composite is a batch of one element Definition per output word, so it partitions under `Q_word_v1` into the word composite's
units, one per element.

The two unary halves are functions of one bf16 word, so each is a few closed forms on the word's fields and, on a band of
exponent fields e, a constant table of the word Definition's own values (`band`, as `boolean_silu` reads silu's):

    gelu_bf16(g) = bf16(gelu_tanh(f32(g))), NaN 0x7FFF        tanh_bf16(x) = bf16(RN32(tanh(f32(x)))), NaN 0x7FFF
      g NaN, or g = -inf       0x7FFF                            x NaN               0x7FFF
      e < 118                  bf16_rn(g / 2) (sign kept)        e < 122             x
      g >= 0, e >= 130         g (+inf included)                 e >= 130            +-1 (+-inf included)
      g < 0,  e >= 130         -0                                122 <= e <= 129     x's sign, the table's magnitude
      118 <= e <= 129          g's sign, the table's magnitude

The GeGLU then multiplies as `GeluTanhMulBf16_v1` does, ``bf16(f32(gelu_bf16(g)) * (f32(u) + 0.0f))`` (`fp.f32_mul`, then
`fp.f32_to_bf16`, NaN 0x7FFF), and the scalar element is ``bf16(f32(x) * K)`` with K folded.  No word Definition calls a MUFU
(`TanhF32Rn_v1` is the libdevice `tanhf` stand-in), so no sub-Call is a word Definition.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.types import Array
from verity.ml.boolean import fp
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import NOT, ONE, ZERO, OR, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.boolean.trace import definition
from verity.ml.prims import const
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

__all__ = ["Bf16DivScalar", "Bf16MulConst", "Bf16MulConstWord", "Bf16MulScalar", "Bf16Tanh", "GELU_BAND", "GeluTanhMul",
           "GeluTanhMulBf16", "TANH_BAND", "TanhBf16", "TanhBf16Word", "band", "gelu_bf16", "gelu_table", "tanh_bf16", "tanh_table"]

BF16 = Array(16, BL.BIT)
BF16W = P.BF16
NAN_F2FP = P.BF16_NAN_F2FP

#: the exponent fields each table covers, even to odd, so a row is one exponent pair
GELU_BAND = (118, 129)
TANH_BAND = (122, 129)


def _blocks(ks) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in ks:
        out[k >> 2] = out.get(k >> 2, 0) | 1 << (k & 3)
    return out


def band(C, g, table, rows, signed: bool = True):
    """The 15 magnitude bits ``table`` gives the bf16 word g (LSB first) on the band ``rows``, zero off it.

    ``rows`` are (sign, e0) with e0 even, each the exponent pair (e0, e0 + 1) of that sign (any sign when not ``signed``), and
    ``table[m | (e mod 16) << 7 | sign << 11]`` is the magnitude at g = (sign, e, m).  The table is a multiplexer over constant
    bits: g's mantissa and lowest exponent bit decode to 256 one-hot minterms and each row to one minterm; each output bit of a row
    is one AND of the row's minterm with a fixed XOR of low minterms (or its complement, whichever has fewer terms), the XOR a
    chain over blocks of four in ascending order with each block's part a chain over its minterms, so equal parts and equal
    prefixes are one wire."""
    m, e, s = g[:7], g[7:15], g[15]
    lo = fp.decode(C, list(m) + [e[0]])
    hi = fp.decode(C, list(e[1:4]) + ([s] if signed else []))
    nibble = {n: and_reduce(C, [e[7 - i] if n >> 3 - i & 1 else NOT(e[7 - i]) for i in range(4)])
              for n in sorted({e0 >> 4 for _, e0 in rows})}
    parts: dict = {}

    def part(b: int, mask: int):
        if (b, mask) not in parts:
            top = mask.bit_length() - 1
            rest = mask ^ 1 << top
            parts[(b, mask)] = part(b, rest) ^ lo[4 * b + top] if rest else lo[4 * b + top]
        return parts[(b, mask)]

    out = [ZERO] * 15
    for sign, e0 in rows:
        h = (e0 & 0xF) >> 1 | (sign << 3 if signed else 0)
        row = C.AND(hi[h], nibble[e0 >> 4])
        for j in range(15):
            ones = [k for k in range(256) if table[h << 8 | k] >> j & 1]
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


def _tanh_word(x: int) -> int:
    return P.f32_to_bf16_rn_bits(D.tanh_f32_bits(P.bf16_to_f32_bits(x)), NAN_F2FP)


@lru_cache(maxsize=None)
def gelu_table() -> tuple[int, ...]:
    """``gelu_bf16(g) & 0x7FFF`` for every g on GeLU's band, at index ``m | (e mod 16) << 7 | sign << 11``; 0 off it."""
    out = [0] * (1 << 12)
    for sign in (0, 1):
        for e in range(GELU_BAND[0], GELU_BAND[1] + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = D.gelu_tanh_bf16_bits(sign << 15 | e << 7 | m) & 0x7FFF
    return tuple(out)


@lru_cache(maxsize=None)
def tanh_table() -> tuple[int, ...]:
    """``tanh_bf16(x) & 0x7FFF`` for every x >= 0 on tanh's band (tanh is odd), at index ``m | (e mod 16) << 7``; 0 off it."""
    out = [0] * (1 << 11)
    for e in range(TANH_BAND[0], TANH_BAND[1] + 1):
        for m in range(128):
            out[m | (e & 0xF) << 7] = _tanh_word(e << 7 | m)
    return tuple(out)


def _half(C, g, below):
    """``bf16_rn(g / 2)``'s 15 magnitude bits where ``below`` (2 <= e: the exponent field less one; e <= 1: the 8-bit significand
    halved, ties to even), zero elsewhere."""
    m, e = list(g[:7]), list(g[7:15])
    low = C.AND(below, NOT(or_reduce(C, e[1:])))
    high = below ^ low
    half_low = add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8) + [ZERO] * 7
    half_high = m + sub(C, e, [ONE], 8)[0]
    return [C.AND(low, y) ^ C.AND(high, z) for y, z in zip(half_low, half_high)]


def _set(bits, word: int, flag):
    """``bits`` with ``flag`` XORed into each bit that is set in ``word``: the constant ``word`` where every other term is zero."""
    return [x ^ flag if word >> j & 1 else x for j, x in enumerate(bits)]


def gelu_bf16(C, g):
    """``bf16(gelu_tanh(f32(g)))``, `GeluTanhMulBf16_v1`'s activation word before the multiply, for a 16-bit g (LSB first)."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))                 # NaN, and -inf: (0.5 * -inf) * (1 + -1)
    _, below = sub(C, e, const_bits(GELU_BAND[0], 8), 8)
    _, under = sub(C, e, const_bits(GELU_BAND[1] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))                      # g >= 0 above the band: g
    mag = [x ^ y ^ C.AND(keep, z) for x, y, z in zip(band(C, g, gelu_table(), _rows(GELU_BAND, (0, 1))), _half(C, g, below), g[:15])]
    return _set(mag, NAN_F2FP, nan) + [C.AND(s, NOT(nan))]


def tanh_bf16(C, x):
    """``bf16(RN32(tanh(f32(x))))``, `Bf16Tanh_v1`'s element, for a 16-bit x (LSB first)."""
    m, e, s = list(x[:7]), list(x[7:15]), x[15]
    nan = C.AND(and_reduce(C, e), or_reduce(C, m))
    _, below = sub(C, e, const_bits(TANH_BAND[0], 8), 8)
    _, under = sub(C, e, const_bits(TANH_BAND[1] + 1, 8), 8)
    one = C.AND(NOT(under), NOT(nan))                                      # e >= 130: +-1
    mag = [y ^ C.AND(below, z) for y, z in zip(band(C, x, tanh_table(), _rows(TANH_BAND, (0,)), signed=False), x[:15])]
    return _set(_set(mag, 0x3F80, one), NAN_F2FP, nan) + [C.AND(s, NOT(nan))]


def _rows(span, signs):
    return [(sign, e0) for sign in signs for e0 in range(span[0], span[1] + 1, 2)]


def gelu_mul(C, g, u):
    """`GeluTanhMulBf16_v1`: ``bf16(f32(gelu_bf16(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF; ``+ 0.0f`` turns u = -0 into +0."""
    u0 = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(gelu_bf16(C, g)), fp.bf16_to_f32(u0)), NAN_F2FP)


GeluTanhMulBf16 = definition("GeluTanhMulBf16", 2, gelu_mul, (("g", BF16), ("u", BF16)), BF16, word=D.GeluTanhMulBf16,
                             doc="vLLM act_and_mul_kernel<gelu_tanh> on bits: GeluTanhMulBf16_v1")


def _word_tanh(B, S, x):
    """`Bf16Tanh_v1`'s element: ``cvt.rn.bf16.f32(tanhf(f32(x)))``."""
    return B.call(P.F2fpBf16, B.call(D.TanhF32Rn, B.call(P.Bf16ToF32, x)))


def _word_mul_const(B, S, x):
    """`Bf16DivScalar_v1`'s and `Bf16MulScalar_v1`'s element: ``cvt.rn.bf16.f32(f32(x) * K)``, K the f32 word."""
    return B.call(P.F2fpBf16, B.call(P.F32Mul, B.call(P.Bf16ToF32, x), B.call(const(32, S.K))))


TanhBf16Word = CompositeDefinition("TanhBf16", 1, (), lambda S: ((("x", BF16W),), BF16W), _word_tanh, doc=_word_tanh.__doc__,
                                   conformance="exact-model-tested")
Bf16MulConstWord = CompositeDefinition("Bf16MulConst", 1, ("K",), lambda S: ((("x", BF16W),), BF16W), _word_mul_const,
                                       doc=_word_mul_const.__doc__, conformance="exact-model-tested")

TanhBf16 = definition("TanhBf16", 2, tanh_bf16, (("x", BF16),), BF16, word=bind(TanhBf16Word),
                      doc="aten.tanh on one bf16 word, on bits: TanhBf16_v1")


def _traced_mul_const(statics: tuple[str, ...]):
    """``Bf16MulConst_v2{K}``: for each K, the trace of ``bf16(f32(x) * K)`` with K's bits folded (made once per K)."""
    made: dict[int, TR.Traced] = {}

    def body(B, S, x):
        if S.K not in made:
            k = fp.f32_const(S.K)
            made[S.K] = TR.trace(lambda C, v: fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(v), k), NAN_F2FP), [BF16], BF16)
        return TR.emit(made[S.K], [BF16], BF16, B, x)

    d = CompositeDefinition("Bf16MulConst", 2, statics, lambda S: ((("x", BF16),), BF16), body,
                            doc="bf16(f32(x) * K) on bits, K folded: Bf16MulConst_v1{K}", conformance="exact-model-tested")
    d.word = lambda b: bind(Bf16MulConstWord, K=b["K"])  # type: ignore[attr-defined]
    return d


Bf16MulConst = _traced_mul_const(("K",))


def _gelu_rows(B, S, gu):
    """vLLM `gelu_tanh_and_mul` on bits: `out[i] = GeluTanhMulBf16_v2(gu[i], gu[I + i])`."""
    return B.batch(GeluTanhMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


def _tanh_rows(B, S, x):
    """`aten.tanh` on bits: `out[j] = TanhBf16_v2(x[j])`."""
    return B.batch(TanhBf16, x, axes=(0,))


def _div_rows(B, S, x):
    """`aten.div(x, C)` on bits: `out[j] = Bf16MulConst_v2{K = RN32(1 / C)}(x[j])`."""
    return B.batch(bind(Bf16MulConst, K=D.reciprocal_bits(float(S.C))), x, axes=(0,))


def _mul_rows(B, S, x):
    """`aten.mul(x, C)` on bits: `out[j] = Bf16MulConst_v2{K = f32(C)}(x[j])`."""
    return B.batch(bind(Bf16MulConst, K=P.bits_of(np.float32(float(S.C)))), x, axes=(0,))


def _rows_sig(n: str, k: int = 1):
    return lambda S: ((("x" if k == 1 else "gu", Array(k * getattr(S, n), BF16)),), Array(getattr(S, n), BF16))


GeluTanhMul = CompositeDefinition("GeluTanhMul", 2, ("I",), _rows_sig("I", 2), _gelu_rows, doc=_gelu_rows.__doc__,
                                  conformance="exact-model-tested")
Bf16Tanh = CompositeDefinition("Bf16Tanh", 2, ("N",), _rows_sig("N"), _tanh_rows, doc=_tanh_rows.__doc__,
                               conformance="exact-model-tested")
Bf16DivScalar = CompositeDefinition("Bf16DivScalar", 2, ("N", "C"), _rows_sig("N"), _div_rows, doc=_div_rows.__doc__,
                                    conformance="exact-model-tested")
Bf16MulScalar = CompositeDefinition("Bf16MulScalar", 2, ("N", "C"), _rows_sig("N"), _mul_rows, doc=_mul_rows.__doc__,
                                    conformance="exact-model-tested")
GeluTanhMul.word = lambda b: bind(D.GeluTanhMul, I=b["I"])  # type: ignore[attr-defined]
Bf16Tanh.word = lambda b: bind(D.Bf16Tanh, N=b["N"])  # type: ignore[attr-defined]
Bf16DivScalar.word = lambda b: bind(D.Bf16DivScalar, N=b["N"], C=b["C"])  # type: ignore[attr-defined]
Bf16MulScalar.word = lambda b: bind(D.Bf16MulScalar, N=b["N"], C=b["C"])  # type: ignore[attr-defined]
