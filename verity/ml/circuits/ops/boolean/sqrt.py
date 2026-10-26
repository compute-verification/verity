"""sqrt.rn.f32 on bits, as a circuit over `forms`: `f32_sqrt_rn` (``PouwF32SqrtRn_v1``) and the digit-by-digit root it takes
(`isqrt`).  PoUW's `PouwF32SqrtRn_v2` (`verity.ml.protocols.pouw.circuit.boolean`) is traced from `f32_sqrt_rn`.

* `f32_sqrt_rn`: the significand normalized, the exponent made even by doubling it, and a digit-by-digit root (`isqrt`) of
  the 50-bit radicand to 25 bits; the last bit rounds, since no square root of a float lies halfway between two (the
  radicand is even, so its root is no odd integer).  Every root of a positive finite word is a normal word.
"""
from __future__ import annotations

from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, const_bits, lzc, mux, sub

_NAN_PTX = 0x7FFFFFFF


def isqrt(C, n):
    """⌊√n⌋ of an unsigned word of even width (LSB first), half as wide: one trial subtraction per root bit."""
    rem, root = [], []
    for i in range(len(n) // 2 - 1, -1, -1):
        rem = [n[2 * i], n[2 * i + 1]] + rem
        trial = [ONE, ZERO] + root
        w = max(len(rem), len(trial))
        d, borrow = sub(C, rem, trial, w)
        bit = NOT(borrow)
        rem = mux(C, bit, d, rem + [ZERO] * (w - len(rem)))[:len(root) + 2]
        root = [bit] + root
    return root


def f32_sqrt_rn(C, a):
    """``PouwF32SqrtRn_v1``: sqrt.rn.f32, √±0 = ±0, √+inf = +inf, a NaN or a negative nonzero operand 0x7FFFFFFF."""
    A = fp.unpack(C, a)
    _az, lz = lzc(C, [ZERO] * 8 + A.sig)
    sig = fp.shl(C, A.sig, lz)                                          # 1.f, the top bit set
    x = sub(C, add(C, A.e_eff, const_bits(127, 9), width=9), lz, 9)[0]  # e + 254 for the unbiased exponent e
    m = mux(C, x[0], [ZERO] + sig, sig + [ZERO])                        # e odd: 2 · 1.f and e - 1
    r = isqrt(C, [ZERO] * 25 + m)                                       # 25 bits, the top one set
    word = add(C, r[1:24] + x[1:9], [r[0]], width=31) + [ZERO]          # exponent field ⌊e / 2⌋ + 127
    keep = OR(C, A.zero, C.AND(A.inf, NOT(A.s)))
    nan = C.AND(NOT(keep), OR(C, A.e_max, A.s))
    return mux(C, nan, const_bits(_NAN_PTX, 32), mux(C, keep, list(a), word))


__all__ = ["f32_sqrt_rn", "isqrt"]
