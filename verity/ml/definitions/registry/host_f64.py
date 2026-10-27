"""CPython's float arithmetic on x86-64 Linux as Boolean builders: SSE2's f64 add, subtract, multiply and divide, the FMA3
fused multiply-add (round to nearest even, subnormals kept, overflow to infinity), the casts between f32 and f64, and glibc 2.39's
``pow`` as its FMA variant computes it (`__pow_fma`, the ifunc's choice on a CPU with FMA and AVX2), on its main path.

Every NaN result is the canonical 0x7FF8000000000000, and an f32 one 0x7FC00000: no operand's payload or sign is kept. A
Definition that reads these builders states where that differs from the host.

`pow_main` is `e_pow.c`'s ``log_inline`` then ``exp_inline`` in the order libm.so.6's FMA variant runs them (every ``vfmadd``
one `f64_fma`; the tables `glibc_pow_data`'s) for x a positive normal and y with ``|y|`` in [2^-65, 2^63), the inputs that
skip ``__pow``'s special cases. ``exp_inline``'s two special inputs are handled as follows:

* below 2^-54 it returns ``1 + ehi``, and so does `pow_main`;
* at 512 and above (``specialcase``, overflow and underflow), `pow_main` returns its main-path word, which is not pow's, and
  sets ``big``.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, const_bits, multiply, mux, or_reduce, sub
from verity.ml.circuits.silicon import fp
from verity.ml.definitions.registry import glibc_pow_data as D

__all__ = ["F64_ONE", "F64_QNAN", "f32_to_f64", "f64_add", "f64_div", "f64_fma", "f64_mul", "f64_neg", "f64_sub",
           "f64_to_f32", "pow_main", "s12_to_f64", "u_less"]

F64_QNAN = 0x7FF8000000000000
F64_INF = 0x7FF0000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
#: FLT_MAX as an f64: `c10::overflows<float, double>` refuses a finite value of larger magnitude
F64_FLT_MAX = 0x47EFFFFFE0000000
#: `log_inline`'s OFF, the start of the subinterval x's top bits index
POW_OFF = 0x3FE6955500000000


def c64(value: int) -> list:
    return const_bits(value & ((1 << 64) - 1), 64)


def unpack(C, w) -> fp.F32:
    return fp.unpack_fp(C, w, 11, 52)


def u_less(C, x, value: int):
    """``x < value`` for an unsigned word and a constant."""
    return sub(C, list(x), const_bits(value, len(x)), len(x))[1]


def specials(C, word, nan, inf, inf_sign):
    word = mux(C, inf, const_bits(F64_INF, 63) + [inf_sign], word)
    return mux(C, nan, c64(F64_QNAN), word)


def f64_neg(a):
    return list(a[:63]) + [NOT(a[63])]


def f64_add(C, a, b):
    """``RN(a + b)`` (addsd): `fp.f32_add` at binary64. ``x + (-x) = +0``, ``(-0) + (-0) = -0``."""
    A, B = unpack(C, a), unpack(C, b)
    _, a_lt_b = sub(C, a[:63], b[:63], 63)
    sx, sy = mux(C, a_lt_b, [B.s, A.s], [A.s, B.s])
    ex = mux(C, a_lt_b, B.e_eff, A.e_eff)
    ey = mux(C, a_lt_b, A.e_eff, B.e_eff)
    gx = mux(C, a_lt_b, B.sig, A.sig)
    gy = mux(C, a_lt_b, A.sig, B.sig)
    d, _ = sub(C, ex, ey, 11)
    ys, st = fp.shr_sticky(C, [ZERO] * 3 + gy, d)
    ys[0] = OR(C, ys[0], st)
    sb = sx ^ sy
    S = add(C, [ZERO] * 3 + gx + [ZERO], [v ^ sb for v in ys] + [sb], cin=sb, width=57)
    r = fp.round_fp(C, sx, ex + [ZERO], -1, S, C.AND(A.s, B.s), 11, 52, underflow=False)
    nan = OR(C, OR(C, A.nan, B.nan), C.AND(C.AND(A.inf, B.inf), A.s ^ B.s))
    return specials(C, r, nan, OR(C, A.inf, B.inf), mux(C, A.inf, [A.s], [B.s])[0])


def f64_sub(C, a, b):
    return f64_add(C, a, f64_neg(b))


def f64_mul(C, a, b):
    """``RN(a * b)`` (mulsd): the 106-bit significand product rounded once at E = ea + eb - 1022."""
    A, B = unpack(C, a), unpack(C, b)
    s = A.s ^ B.s
    P = multiply(C, fp.trim_low(A.sig)[0], fp.trim_low(B.sig)[0])
    r = fp.round_fp(C, s, add(C, A.e_eff, B.e_eff, width=13), 1022, P, s, 11, 52)
    nan = OR(C, OR(C, A.nan, B.nan), OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)))
    return specials(C, r, nan, OR(C, A.inf, B.inf), s)


def f64_fma(C, a, b, c):
    """``RN(a * b + c)`` with one rounding (vfmadd): `fp.f32_fma` at binary64. The normalized product sits at the top of
    bits [2, 108) of a 163-bit window and the addend's 53-bit significand starts at [110, 163), shifted right by
    ``ea_n + eb_n - ec - 967`` (``T - ec - 1095``, T = ea_n + eb_n + 128); bits below ``lo`` fold into a sticky. An addend
    whose LSB would sit above bit 110 stays there (the product then reaches the rounding only as a sticky). E = T - 1094,
    or ec + 1 at the clamp."""
    A, B, Cc = unpack(C, a), unpack(C, b), unpack(C, c)
    sp = A.s ^ B.s
    ga, lza, _ = fp._normalize_sig(C, A)
    gb, lzb, _ = fp._normalize_sig(C, B)
    P = multiply(C, ga, gb)
    pw = len(P)
    lo = max(0, min(108 - pw, 52) - 1)
    T = add(C, add(C, A.e_eff, B.e_eff, width=14), const_bits(128, 14), width=14)
    T, _ = sub(C, T, add(C, lza, lzb, width=8), 14)
    ec_b = add(C, Cc.e_eff, const_bits(1095, 14), width=14)
    U, clamp = sub(C, T, ec_b, 14)
    clamp = C.AND(clamp, NOT(Cc.zero))
    sh_c = [C.AND(NOT(clamp), u) for u in U]
    Pw = ([ZERO] * (108 - pw) + P + [ZERO] * 55)[lo:]
    Cs, st = fp.shr_sticky(C, ([ZERO] * 110 + Cc.sig)[lo:], sh_c)
    Cs[0] = OR(C, Cs[0], st)
    sub_ = sp ^ Cc.s
    n = 163 - lo
    Dw = add(C, Pw + [ZERO] * 2, [v ^ sub_ for v in Cs] + [sub_] * 2, cin=sub_, width=n + 2)
    neg = Dw[n + 1]
    M = fp.neg_if(C, Dw[:n + 1], neg)
    r = fp.round_fp(C, sp ^ neg, mux(C, clamp, ec_b, T), 1094, M, ZERO, 11, 52)
    pz = OR(C, A.zero, B.zero)
    zero_c = [ZERO] * 63 + [C.AND(sp, Cc.s)]
    r = mux(C, pz, mux(C, Cc.zero, zero_c, list(c)), r)
    prod_inf = OR(C, A.inf, B.inf)
    nan = OR(C, OR(C, OR(C, A.nan, B.nan), Cc.nan),
             OR(C, OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)), C.AND(C.AND(prod_inf, Cc.inf), sp ^ Cc.s)))
    return specials(C, r, nan, OR(C, prod_inf, Cc.inf), mux(C, prod_inf, [sp], [Cc.s])[0])


def f64_div(C, a, b):
    """``RN(a / b)`` (divsd): `fp.f32_div` at binary64, the normalized significands' quotient to 56 bits by restoring
    division, the remainder as sticky."""
    A, B = unpack(C, a), unpack(C, b)
    s = A.s ^ B.s
    ga, lza, _ = fp._normalize_sig(C, A)
    gb, lzb, _ = fp._normalize_sig(C, B)
    ga, gb = [ZERO] * (53 - len(ga)) + ga, [ZERO] * (53 - len(gb)) + gb
    r, q = list(ga) + [ZERO], []
    for _ in range(56):
        d, borrow = sub(C, r, gb + [ZERO], 54)
        q.append(NOT(borrow))
        r = mux(C, borrow, r, d)
        r = [ZERO] + r[:53]
    M = [or_reduce(C, r)] + q[::-1]
    # a / b = M 2^(Ea - Eb - 56), Ea = e_eff - lz: round_fp's E = Ea - Eb + 1023, carried as E + 2048
    E_b = add(C, add(C, A.e_eff, const_bits(3071, 14), width=14), lzb, width=14)
    E_b, _ = sub(C, E_b, add(C, B.e_eff, lza, width=14), 14)
    word = fp.round_fp(C, s, E_b, 2048, M, s, 11, 52)
    zero = OR(C, C.AND(B.inf, NOT(A.inf)), C.AND(A.zero, NOT(B.zero)))
    word = mux(C, zero, [ZERO] * 63 + [s], word)
    invalid = OR(C, C.AND(A.zero, B.zero), C.AND(A.inf, B.inf))
    inf = OR(C, invalid, OR(C, C.AND(A.inf, NOT(B.inf)), C.AND(B.zero, NOT(A.zero))))
    return specials(C, word, OR(C, OR(C, A.nan, B.nan), invalid), inf, s)


def f64_to_f32(C, x):
    """``(cvtsd2ss(x), refused)``: x rounded to the nearest f32 (subnormals kept, overflow to infinity), and whether torch's
    ``Scalar::to<float>()`` refuses it, x finite and ``|x| > FLT_MAX`` (`c10::checked_convert`)."""
    X = unpack(C, x)
    r = fp.round_fp(C, X.s, X.e_eff + [ZERO], 896, X.sig, X.s, 8, 23)
    r = fp.specials(C, r, X.nan, X.inf, X.s)
    big = NOT(u_less(C, x[:63], F64_FLT_MAX + 1))
    return r, C.AND(big, NOT(X.e_max))


def f32_to_f64(C, x):
    """cvtss2sd: the f32 x as the f64 of equal value (exact)."""
    X = fp.unpack(C, x)
    r = fp.round_fp(C, X.s, X.e_eff + [ZERO] * 4, -896, X.sig, X.s, 11, 52, underflow=False)
    return specials(C, r, X.nan, X.inf, X.s)


def s12_to_f64(C, k):
    """The 12-bit two's-complement k as an f64 (cvtsi2sd, exact); 0 is +0."""
    s = k[11]
    return fp.round_fp(C, s, const_bits(1034, 12), 0, fp.neg_if(C, list(k), s), ZERO, 11, 52, underflow=False)


def _cols(*columns) -> list[int]:
    return [sum(w << (64 * j) for j, w in enumerate(row)) for row in zip(*columns, strict=True)]


_LOG_TAB = _cols(D.LOG_INVC, D.LOG_LOGC, D.LOG_LOGCTAIL)
_EXP_TAB = _cols(D.EXP_TAB[0::2], D.EXP_TAB[1::2])
A0, A1, A2, A3, A4, A5, A6 = (c64(v) for v in D.LOG_POLY)
C2, C3, C4, C5 = (c64(v) for v in D.EXP_POLY)


def pow_main(C, x, y):
    """``(pow(x, y), big)`` on glibc's main path for x a positive normal and 2^-65 <= |y| < 2^63 (module docstring): ``big``
    is ``|y log x| >= 512``, where ``exp_inline`` leaves the main path and the word returned is not pow's."""
    fma, mul, fadd, fsub = (lambda *o: f64_fma(C, *o)), (lambda *o: f64_mul(C, *o)), (lambda *o: f64_add(C, *o)), \
        (lambda *o: f64_sub(C, *o))
    ix = list(x)
    tmp, _ = sub(C, ix, c64(POW_OFF), 64)
    k = tmp[52:64]
    z = ix[:52] + sub(C, ix[52:64], k, 12)[0]
    kd = s12_to_f64(C, k)
    row = fp.lookup(C, tmp[45:52], _LOG_TAB, 192)
    invc, logc, logctail = row[:64], row[64:128], row[128:]
    t1 = fma(kd, c64(D.LN2HI), logc)
    lo1 = fma(kd, c64(D.LN2LO), logctail)
    r = fma(z, invc, c64(F64_NEG_ONE))
    ar = mul(r, A0)
    q1 = fma(r, A2, A1)
    q3 = fma(r, A4, A3)
    t2 = fadd(r, t1)
    lo2 = fadd(fsub(t1, t2), r)
    ar2 = mul(r, ar)
    ar3 = mul(r, ar2)
    lo3 = fma(ar, r, f64_neg(ar2))
    hi = fadd(t2, ar2)
    q5 = fma(r, A6, A5)
    lo4 = fadd(fsub(t2, hi), ar2)
    q = fma(ar2, fma(q5, ar2, q3), q1)
    lo = fma(ar3, q, fadd(fadd(fadd(lo1, lo2), lo3), lo4))
    lhi = fadd(hi, lo)
    llo = fadd(fsub(hi, lhi), lo)
    ehi = mul(y, lhi)
    elo = fma(y, llo, fma(lhi, y, f64_neg(ehi)))
    abstop = ehi[52:63]
    tiny = u_less(C, abstop, 0x3C9)
    big = NOT(u_less(C, abstop, 0x408))
    kd0 = fma(ehi, c64(D.INVLN2N), c64(D.SHIFT))
    kdx = fsub(kd0, c64(D.SHIFT))
    rr = fadd(elo, fma(kdx, c64(D.NEGLN2LON), fma(kdx, c64(D.NEGLN2HIN), ehi)))
    erow = fp.lookup(C, kd0[:7], _EXP_TAB, 128)
    tail = erow[:64]
    scale = add(C, erow[64:], [ZERO] * 45 + kd0[:19], width=64)
    r2 = mul(rr, rr)
    a1 = fma(fma(rr, C3, C2), r2, fadd(rr, tail))
    t = fma(fma(rr, C5, C4), mul(r2, r2), a1)
    out = fma(t, scale, scale)
    return mux(C, tiny, fadd(c64(F64_ONE), ehi), out), big
