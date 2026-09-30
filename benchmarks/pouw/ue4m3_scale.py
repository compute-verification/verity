"""Pearl-C4's block-scale step, exact and cheaper (CPU): candidate implementations of

    s_b = max(e4m3.rn.sat(mul.rn(amax_b, f32(1/6))), 2^-9),   r_b = div.rn(1, s_b)        (`fp4-design.md` step 4)

checked exhaustively against the reference, bit for bit:

  encode   `int_encode`: the UE4M3 byte from the FP32 bits with integer ops and one FADD (no F2FP), against `ue4m3_rn` on
           every finite non-negative FP32 input (2,139,095,040 patterns)
  recip    `lut`: r_b from a 128-entry table indexed by the byte (decode and reciprocal in one load; entry 0 holds 2^9, so
           the 2^-9 floor needs no separate op on the reciprocal side); `exp_mant`: an 8-entry mantissa table and an
           integer exponent, with the 7 subnormal bytes in 8 more entries; `newton`: FFMA only (quadratic seed, two Newton
           steps) on the decoded scale's mantissa, with the exponent reapplied as an integer
  each against RN_f32(1/s) computed in exact rational arithmetic, on all 126 valid bytes (0x01..0x7E).

    python benchmarks/pouw/ue4m3_scale.py            # all checks; prints the tables the kernels embed
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_emulation import ue4m3_rn  # noqa: E402

BYTES = range(1, 127)


def s_of(byte: int) -> Fraction:
    e, m = byte >> 3, byte & 7
    return Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)


def rn_f32(x: Fraction) -> np.float32:
    """Round a positive rational to the nearest FP32 (ties to even), normal range only."""

    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    q = x / Fraction(2) ** (e - 23)                     # in [2^23, 2^24)
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    return np.float32(float(Fraction(n) * Fraction(2) ** (e - 23)))


def f32(x) -> np.float32:
    return np.float32(x)


def fma(a, b, c) -> np.float32:
    return rn_f32_signed(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def rn_f32_signed(x: Fraction) -> np.float32:
    if x == 0:
        return np.float32(0.0)
    return rn_f32(x) if x > 0 else -rn_f32(-x)


def bits(x: np.float32) -> int:
    return int(np.array(x, np.float32).view(np.uint32))


def from_bits(u: int) -> np.float32:
    return np.array(u & 0xFFFFFFFF, np.uint32).view(np.float32)[()]


REF = {b: rn_f32(1 / s_of(b)) for b in BYTES}
LUT = [rn_f32(Fraction(2) ** 9)] + [REF[b] for b in BYTES] + [np.float32(0.0)]      # 128 entries; [0] = floor, [127] unused
MANT = [rn_f32(1 / (1 + Fraction(j, 8))) for j in range(8)]                            # RN(1/(1 + j/8)), exponent 0 or -1
SUB = [rn_f32(Fraction(2) ** 9)] + [rn_f32(Fraction(2) ** 9 / j) for j in range(1, 8)]  # subnormal bytes 0x00..0x07


def exp_mant(byte: int) -> np.float32:
    """Normal byte: RN(1/s) = MANT[j]·2^(7−e), applied to the bits as an integer; subnormal byte: SUB[j]."""

    e, j = byte >> 3, byte & 7
    if e == 0:
        return SUB[j]
    return from_bits(bits(MANT[j]) + ((7 - e) << 23))


# FFMA-only reciprocal on the normalized mantissa m in [1, 2): y0 = (c2·m + c1)·m + c0, two Newton steps, exponent back.
# The quadratic is a least-squares fit through the 8 mantissas weighted toward the ends (1 + 2(m − 1.4375)²); the plain
# fit leaves m = 15/8 one ulp off after two steps.
C2, C1, C0 = f32(0.37409148), f32(-1.5990455), f32(2.2202663)


def newton(byte: int) -> np.float32:
    s = f32(float(s_of(byte)))                         # the decode (exact)
    u = bits(s)
    m = from_bits((u & 0x007FFFFF) | 0x3F800000)
    y = fma(fma(C2, m, C1), m, C0)
    for _ in range(2):
        e = fma(-m, y, f32(1.0))
        y = fma(y, e, y)
    return from_bits(bits(y) - ((u & 0x7F800000) - (127 << 23)))


def int_encode(v: np.ndarray) -> np.ndarray:
    """e4m3.rn.satfinite of non-negative FP32 v (uint32 bits in, byte out), with integer ops and one FADD:
    normal range: RN to 3 mantissa bits by add-and-shift on the bits, carry into the exponent; below 2^-6, the FADD of
    2^14 rounds v to a multiple of 2^-9 (RN-even) and the bits' low part is the byte; saturate at 0x7E."""

    u = v.astype(np.uint32)
    lsb = (u >> 20) & 1
    nrm = ((u.astype(np.int64) - (120 << 23) + 0x7FFFF + lsb) >> 20)
    w = (from_bits_arr(u) + np.float32(2.0 ** 14)).astype(np.float32)
    sub = w.view(np.uint32).astype(np.int64) - int(bits(np.float32(2.0 ** 14)))
    out = np.where(u < bits(np.float32(2.0 ** -6)), sub, nrm)
    return np.minimum(out, 0x7E).astype(np.uint8)


def from_bits_arr(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint32).view(np.float32)


def check_encode(chunk: int = 1 << 24) -> dict:
    """Every finite non-negative FP32 pattern (+0 to FLT_MAX; the spec's input mul.rn(amax, 1/6) is finite) through
    `int_encode` and `ue4m3_rn`."""

    bad = 0
    top = 0x7F800000
    for lo in range(0, top, chunk):
        u = np.arange(lo, min(lo + chunk, top), dtype=np.uint32)
        v = from_bits_arr(u)
        ref = np.asarray(ue4m3_rn(v.astype(np.float64)), np.int64)
        got = int_encode(u).astype(np.int64)
        bad += int((ref != got).sum())
    return {"inputs": top, "mismatches": bad}


def main() -> int:
    res = {
        "lut": sum(bits(LUT[b]) != bits(REF[b]) for b in BYTES),
        "exp_mant": sum(bits(exp_mant(b)) != bits(REF[b]) for b in BYTES),
        "newton": sum(bits(newton(b)) != bits(REF[b]) for b in BYTES),
    }
    print(f"reciprocals on the 126 valid bytes, mismatches against RN_f32(1/s): {res}")
    print("LUT (hex bits, entry 0 = the 2^-9 floor's reciprocal):")
    print(", ".join(f"0x{bits(x):08x}" for x in LUT))
    print("MANT:", ", ".join(f"0x{bits(x):08x}" for x in MANT), " SUB:", ", ".join(f"0x{bits(x):08x}" for x in SUB))
    enc = check_encode()
    print(f"int_encode against ue4m3_rn on every non-negative FP32 input: {enc}")
    return 0 if not any(res.values()) and enc["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
