"""libdevice's `__nv_logf` and `__nv_log1pf` on fp32 words, each operation a single correctly rounded IEEE step in pure
Python: the evaluators of the sampler's and the top-p keep word's log primitives."""
from __future__ import annotations

import math
import struct

M32 = 0xFFFFFFFF

# libdevice polynomial coefficients (fp32 words, Horner order), from libdevice.10.bc `__nv_logf` / `__nv_log1pf`
LOGF_POLY = (0xBE055027, 0x3E1039F6, 0xBDF8CDCC, 0x3E0F2955, 0xBE2AD8B9, 0x3E4CED0B, 0xBE7FFF22, 0x3EAAAA78)
LOG1PF_POLY = (0xBD39BF78, 0x3DD80012, 0xBE0778E0, 0x3E146475, 0xBE2A68DD, 0x3E4CAF9E, 0xBE800042, 0x3EAAAAE6)
LN2_BITS = 0x3F317218
TWO_M23_BITS = 0x34000000
TWO_P23_BITS = 0x4B000000
FLT_MIN_BITS = 0x00800000
ONE_BITS = 0x3F800000
#: `__nv_logf` / `__nv_log1pf` outside their domain (a < 0, a < -1, -inf): their special case's `fma(a, inf, inf)` is -inf + inf,
#: whose NaN is pinned here rather than left to the host's default NaN (x86 and aarch64 differ in the sign)
LOG_INVALID_NAN = 0xFFC00000
#: every NaN result of the `_v2` logs (`nv_logf_gpu`, `nv_log1pf_gpu`): the GPU's canonical NaN, for an input outside the domain and a
#: NaN input alike, no payload or sign kept (measured on the L40S and the H100, every build); the `_v1` logs keep `LOG_INVALID_NAN`
#: and a NaN input's payload
LOG_NAN = 0x7FFFFFFF


def _neg_not_nan(a: int) -> bool:
    return a >= 0x80000000 and (a & 0x7FFFFFFF) <= 0x7F800000


# ---------------------------------------------------------------------------------------------------------
# fp32 helpers on encodings (pure Python; every operation is a single correctly rounded IEEE step)
# ---------------------------------------------------------------------------------------------------------


def _f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


def _u(x: float) -> int:
    """RN-even fp32 word of the double `x` (struct rounds to nearest-even; overflow -> inf)."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def _ftz(u: int) -> int:
    u &= M32
    return u & 0x80000000 if (u & 0x7F800000) == 0 and (u & 0x7FFFFF) else u


def _round_exact(s: float, e: float) -> int:
    """fp32 word of the exact value s + e (s double, |e| <= ulp(s)/2 from TwoSum), RN-even."""
    r = _u(s)
    if e == 0.0 or math.isinf(s) or math.isnan(s):
        return r
    rf = _f(r)
    if rf == s:
        return r
    lo, hi = (math.nextafter(rf, -math.inf), rf) if rf > s else (rf, math.nextafter(rf, math.inf))
    if (lo + hi) * 0.5 == s:
        return _u(hi) if e > 0 else _u(lo)
    return r


def fma_ftz(a: int, b: int, c: int) -> int:
    """`fma.rn.ftz.f32` on words."""
    fa, fb, fc = _f(_ftz(a)), _f(_ftz(b)), _f(_ftz(c))
    p = fa * fb
    if math.isinf(p) or math.isnan(p) or math.isinf(fc) or math.isnan(fc):
        return _ftz(_u(p + fc))
    s = p + fc
    bb = s - p
    e = (p - (s - bb)) + (fc - bb)
    return _ftz(_round_exact(s, e))


def mul_ftz(a: int, b: int) -> int:
    return _ftz(_u(_f(_ftz(a)) * _f(_ftz(b))))


def add_ftz(a: int, b: int) -> int:
    fa, fb = _f(_ftz(a)), _f(_ftz(b))
    s = fa + fb
    if math.isinf(s) or math.isnan(s):
        return _ftz(_u(s))
    bb = s - fa
    e = (fa - (s - bb)) + (fb - bb)
    return _ftz(_round_exact(s, e))


def add_rz_ftz(a: int, b: int) -> int:
    """`add.rz.ftz.f32`: a + b rounded TOWARD ZERO in fp32 (libdevice `__fadd_rz`) -- the v2 evaluator (F-stoch-r17-01 fix).

    The exact sum is s + e (TwoSum in double: exact for fp32 operands); r = RN32(s) is stepped ONE fp32 ulp toward zero when its
    magnitude exceeds the exact value (RN rounded away from zero, or RN landed on s while e pulls the exact value inside it); RZ never
    overflows to inf from a finite sum (it saturates at FLT_MAX); a zero sum keeps its sign.  The same rule the GPU-validated numpy
    transcription `adapters/vllm/verity_vllm/definitions/numerics/libdevice.py::add_rz_f32` applies (np.nextafter on float32 = one f32 ulp)."""
    fa, fb = _f(_ftz(a)), _f(_ftz(b))
    s = fa + fb
    if s == 0.0 or math.isnan(s):
        return _ftz(_u(s))
    if math.isinf(s):                            # only when an operand is infinite: fp32 operands cannot overflow a double sum
        return _ftz(_u(s))
    bb = s - fa
    e = (fa - (s - bb)) + (fb - bb)              # exact = s + e
    r = _u(s)
    rf = _f(r)
    if math.isinf(rf):                           # RN32 overflowed a finite sum: toward zero saturates at the largest finite word
        return (r & 0x80000000) | 0x7F7FFFFF
    if abs(rf) > abs(s) or (rf == s and e != 0.0 and (e < 0) == (s > 0)):
        if r & 0x7FFFFFFF:
            r = (r & 0x80000000) | ((r & 0x7FFFFFFF) - 1)      # one fp32 ulp toward zero
    return _ftz(r)


def _i32(u: int) -> int:
    u &= M32
    return u - (1 << 32) if u & 0x80000000 else u


def nv_logf(a: int, ftz: bool = True) -> int:
    """libdevice `__nv_logf` on an fp32 word (FTZ); with `ftz` False its build without flush-to-zero, whose 2^23 scaling of an
    input below FLT_MIN is ``mul.rn.f32`` (exact), so a subnormal's log is finite. No later operand or result of either build is
    subnormal, so the builds differ only in that scaling."""
    a &= M32
    small = _f(a) < _f(FLT_MIN_BITS)
    a2 = (mul_ftz(a, TWO_P23_BITS) if ftz else _u(_f(a) * 8388608.0)) if small else a
    i0 = 0xC1B80000 if small else 0
    ai = _i32(a2)
    e = (ai - 1059760811) & -8388608
    e_i32 = _i32(e & M32)
    m = (ai - e_i32) & M32
    i2 = fma_ftz(_u(float(e_i32)), TWO_M23_BITS, i0)
    f = add_ftz(m, 0xBF800000)
    r = fma_ftz(LOGF_POLY[0], f, LOGF_POLY[1])
    for k in LOGF_POLY[2:]:
        r = fma_ftz(r, f, k)
    r = fma_ftz(r, f, 0xBF000000)
    r = mul_ftz(r, f)
    r = fma_ftz(r, f, f)
    r = fma_ftz(i2, LN2_BITS, r)
    if a2 >= 0x7F800000:
        r = LOG_INVALID_NAN if _neg_not_nan(a2) else fma_ftz(a2, 0x7F800000, 0x7F800000)
    if (a2 & 0x7FFFFFFF) == 0:
        r = 0xFF800000
    return r


def nv_log1pf(a: int) -> int:
    """libdevice `__nv_log1pf` on an fp32 word (FTZ): `__fadd_rz(a, 1.0f)` is a TRUE toward-zero add (`add_rz_ftz`; F-stoch-r17-01 fix),
    then the binade selector and the polynomial (`_log1pf_from_sum`).  The sum feeds only the selector, so the fix moves exactly one
    word of the whole fp32 domain (a = 0x3EFFFFFF, positive) and none in the Gumbel lane's domain a = -u, u in [2^-31', 1):
    `GumbelNoiseLane_v1` is the same function it was (exhaustive, test_sampling_rows)."""
    return _log1pf_from_sum(a & M32, add_rz_ftz(a & M32, ONE_BITS))


def _gpu_nan(u: int) -> int:
    return LOG_NAN if (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) else u


def nv_logf_gpu(a: int, ftz: bool = True) -> int:
    """`nv_logf` with every NaN result `LOG_NAN`, the GPU's (`NvLogf_v2`; without `ftz`, `NvLogf_v4`); every other word is
    `nv_logf`'s."""
    return _gpu_nan(nv_logf(a, ftz))


def nv_log1pf_gpu(a: int) -> int:
    """`nv_log1pf` with every NaN result `LOG_NAN`, the GPU's; every other word is `nv_log1pf`'s."""
    return _gpu_nan(nv_log1pf(a))


def _log1pf_from_sum(a: int, s: int) -> int:
    """The body of libdevice `__nv_log1pf` after its first instruction: `s` = the `__fadd_rz(a, 1.0f)` word (v1 model or true RZ)."""
    si = _i32(s)
    e = (si - 1061158912) & -8388608
    e_i32 = _i32(e & M32)
    m1 = (_i32(a) - e_i32) & M32
    t = (1082130432 - e_i32) & M32
    r0 = fma_ftz(0x3E800000, t, 0xBF800000)
    u = add_ftz(m1, r0)
    ef = mul_ftz(_u(float(e_i32)), TWO_M23_BITS)
    r = fma_ftz(LOG1PF_POLY[0], u, LOG1PF_POLY[1])
    for k in LOG1PF_POLY[2:]:
        r = fma_ftz(r, u, k)
    r = fma_ftz(r, u, 0xBF000000)
    r = mul_ftz(r, u)
    r = fma_ftz(r, u, u)
    r = fma_ftz(ef, LN2_BITS, r)
    if a >= 0x7F800000:
        if _i32(a) > -1082130432:
            r = LOG_INVALID_NAN if _neg_not_nan(a) else fma_ftz(a, 0x7F800000, 0x7F800000)
        if (a & 0x7FFFFFFF) == 0:
            r = 0x80000000
    return r


__all__ = ["LOGF_POLY", "LOG1PF_POLY", "LN2_BITS", "TWO_M23_BITS", "TWO_P23_BITS", "FLT_MIN_BITS", "ONE_BITS", "LOG_INVALID_NAN",
           "LOG_NAN", "fma_ftz", "mul_ftz", "add_ftz", "add_rz_ftz", "nv_logf", "nv_log1pf", "nv_logf_gpu", "nv_log1pf_gpu"]
