"""Gemma-2's final-logit softcap tanh as Boolean Definitions, each the next version of a word Definition, on bits.

    TanhBf16_v2               (x: Array<16, Value<1>>) -> Array<16, Value<1>>                       word view: TanhBf16_v1
    Bf16Tanh_v2{N}            (x: Array<N, Array<16, Value<1>>>) -> Array<N, Array<16, Value<1>>>   word view: Bf16Tanh_v1{N}

`TanhBf16_v1` is `Bf16Tanh_v1`'s body at N = 1 on one word, so the element Definition has a word view; the served Program
never calls it.  `Bf16Tanh_v2{N}` is a batch of N `TanhBf16_v2` Calls, so it partitions under `Q_word_v1` into
`Bf16Tanh_v1{N}`'s units, one per element.  The softcap's divide and multiply are element-wise rows (`boolean_dense`).

tanh_bf16(x) = bf16(RN32(tanh(f32(x)))), NaN 0x7FFF, is a function of one bf16 word, so it is a few closed forms on the word's
fields and, on a band of exponent fields e, a constant table of the word Definition's own values (`band`, which `boolean_silu`
reads silu's with too):

      x NaN               0x7FFF
      e < 122             x
      e >= 130            +-1 (+-inf included)
      122 <= e <= 129     x's sign, the table's magnitude (tanh is odd)

No word Definition calls a MUFU (`TanhF32Rn_v1` is the libdevice `tanhf` stand-in), so no sub-Call is a word Definition.
"""
from __future__ import annotations

from functools import lru_cache

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

__all__ = ["Bf16Tanh", "TANH_BAND", "TanhBf16", "TanhBf16Word", "band", "tanh_bf16", "tanh_table"]

BF16 = Array(16, BL.BIT)
NAN_F2FP = P.BF16_NAN_F2FP

#: the exponent fields the table covers, even to odd, so a row is one exponent pair
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
def tanh_table() -> tuple[int, ...]:
    """``tanh_bf16(x) & 0x7FFF`` for every x >= 0 on tanh's band (tanh is odd), at index ``m | (e mod 16) << 7``; 0 off it."""
    out = [0] * (1 << 11)
    for e in range(TANH_BAND[0], TANH_BAND[1] + 1):
        for m in range(128):
            out[m | (e & 0xF) << 7] = _tanh_word(e << 7 | m)
    return tuple(out)


def _set(bits, word: int, flag):
    """``bits`` with ``flag`` XORed into each bit that is set in ``word``: the constant ``word`` where every other term is zero."""
    return [x ^ flag if word >> j & 1 else x for j, x in enumerate(bits)]


def tanh_bf16(C, x):
    """``bf16(RN32(tanh(f32(x))))``, `Bf16Tanh_v1`'s element, for a 16-bit x (LSB first)."""
    m, e, s = list(x[:7]), list(x[7:15]), x[15]
    nan = C.AND(and_reduce(C, e), or_reduce(C, m))
    _, below = sub(C, e, const_bits(TANH_BAND[0], 8), 8)
    _, under = sub(C, e, const_bits(TANH_BAND[1] + 1, 8), 8)
    one = C.AND(NOT(under), NOT(nan))                                      # e >= 130: +-1
    rows = [(0, e0) for e0 in range(TANH_BAND[0], TANH_BAND[1] + 1, 2)]
    mag = [y ^ C.AND(below, z) for y, z in zip(band(C, x, tanh_table(), rows, signed=False), x[:15])]
    return _set(_set(mag, 0x3F80, one), NAN_F2FP, nan) + [C.AND(s, NOT(nan))]


def _word_tanh(B, S, x):
    """`Bf16Tanh_v1`'s element: ``cvt.rn.bf16.f32(tanhf(f32(x)))``."""
    return B.call(P.F2fpBf16, B.call(D.TanhF32Rn, B.call(P.Bf16ToF32, x)))


TanhBf16Word = CompositeDefinition("TanhBf16", 1, (), lambda S: ((("x", P.BF16),), P.BF16), _word_tanh, doc=_word_tanh.__doc__,
                                   conformance="exact-model-tested")
TanhBf16 = definition("TanhBf16", 2, tanh_bf16, (("x", BF16),), BF16, word=bind(TanhBf16Word),
                      doc="aten.tanh on one bf16 word, on bits: TanhBf16_v1")


def _tanh_rows(B, S, x):
    """`aten.tanh` on bits: `out[j] = TanhBf16_v2(x[j])`."""
    return B.batch(TanhBf16, x, axes=(0,))


Bf16Tanh = CompositeDefinition("Bf16Tanh", 2, ("N",), lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)), _tanh_rows,
                               doc=_tanh_rows.__doc__, conformance="exact-model-tested")
Bf16Tanh.word = lambda b: bind(D.Bf16Tanh, N=b["N"])  # type: ignore[attr-defined]
