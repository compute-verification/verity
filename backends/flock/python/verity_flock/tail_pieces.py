"""Circuits for the primitives a template's units leave outside: attention's softmax (the FTZ f32 ops, ``F32Max``,
``GuardNegInfZero``, MUFU.EX2, FA2's inverse sum), the norms' row scalars (MUFU.RSQ and SQRT, ``div.full``'s reciprocal and
numerator scaling) and the sampler lane's compare, selects and integer add. Each is its IR primitive bit for bit
(``ir_lower.PIECES``; the piece test and circuit-check compare them).

A MUFU operation is a table lookup inside exponent and special-case logic (the C++ models ``fa2_model.cpp`` and
``rms_triton_model.cpp``, and ``live/src/ir_tail.rs``): the piece computes the table index, reads the pinned table
(:func:`table_read`, a ``gf2`` lookup, which a Flock statement proves with a lookup slot: M0's ``live/src/lookup.rs``) and
finishes the result. The RSQ, SQRT and RCP circuits around the read are M0's (``verity_flock.tail``, verity PR #83); EX2's and
FA2's inverse sum follow ``fa2_model.cpp``.
"""
from __future__ import annotations

from . import fp
from . import gf2
from .gf2 import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub

#: table name -> index bits (each table's sha256 is ir_lower.TABLES[name])
TABLE_BITS = {"rsq": 24, "sqrt": 24, "rcp": 23, "ex2": 23}
#: the bits of a table word a piece reads: exponent and mantissa (bit 31 is zero in every measured table)
VALUE_BITS = 31
QNAN_MUFU = 0x7FFFFFFF
INF = 0x7F800000
ONE_F = 0x3F800000
DEFAULT_NAN = 0xFFC00000
BIG = 0x7E800000       # 2^126
TINY = 0x00800000      # 2^-126
QUARTER = 0x3E800000   # 0.25
TWO24 = 0x4B800000     # 2^24


def _load_table(name: str):
    import hashlib

    import numpy as np
    from verity_vllm.program.kernels import fa2_relation, rms_relation

    from .ir_lower import TABLES
    t = fa2_relation.tables() if name == "ex2" else rms_relation.tables("cuda")
    arr = np.ascontiguousarray(getattr(t, name), dtype="<u4")
    if hashlib.sha256(arr.tobytes()).hexdigest() != TABLES[name]:
        raise ValueError(f"the {name} table is not the pinned one ({TABLES[name]})")
    return arr


for _name in TABLE_BITS:
    gf2.TABLES.setdefault(_name, lambda n=_name: _load_table(n))


def table_read(C, table: str, idx):
    """The pinned MUFU table ``table``'s word at ``idx`` (its low 31 bits)."""
    assert len(idx) == TABLE_BITS[table]
    return C.lookup(table, list(idx), VALUE_BITS)


# --- helpers ---------------------------------------------------------------------------------------------------------------

def _eq_const(C, bits, value: int):
    return and_reduce(C, [b if (value >> i) & 1 else NOT(b) for i, b in enumerate(bits)])


def _ult_const(C, w, k: int):
    """``w < k`` for an unsigned word and a constant."""
    return sub(C, w, const_bits(k, len(w)), len(w))[1]


def _fields(C, x):
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    return s, e, m, _eq_const(C, e, 0xFF), NOT(or_reduce(C, e)), or_reduce(C, m)


def _word(C, cases, default):
    """``cases`` = [(cond, word)] tried in order, else ``default``."""
    out = list(default)
    for cond, w in reversed(cases):
        out = mux(C, cond, w, out)
    return out


def _is_nan(C, w):
    return C.AND(and_reduce(C, w[23:31]), or_reduce(C, w[:23]))


def ftz(C, w):
    """A subnormal word flushed to its signed zero."""
    e_nz = or_reduce(C, w[23:31])
    return [C.AND(e_nz, m) for m in w[:23]] + list(w[23:])


# --- attention's softmax -----------------------------------------------------------------------------------------------------

def f32_add_ftz(C, a, b):
    """``F32AddFtz_v1``: ftz(RN(ftz(a) + ftz(b)))."""
    return ftz(C, fp.f32_add(C, ftz(C, a), ftz(C, b), ir_nan=True))


def f32_sub(C, a, b):
    """numpy float32's ``a - b``: RN; a's NaN quieted when a is a NaN, else b's (its own sign), else 0xFFC00000 for an invalid
    operation (inf - inf of one sign)."""
    r = fp.f32_add(C, a, fp.f32_neg(b))
    A, B = fp.unpack(C, a), fp.unpack(C, b)
    r = mux(C, C.AND(C.AND(A.inf, B.inf), NOT(A.s ^ B.s)), const_bits(DEFAULT_NAN, 32), r)
    src = mux(C, A.nan, a[:22] + a[23:], b[:22] + b[23:])
    return mux(C, OR(C, A.nan, B.nan), src[:22] + [ONE] + src[22:], r)


def f32_sub_ftz(C, a, b):
    """``F32SubFtz_v1``: ftz(RN(ftz(a) - ftz(b)))."""
    return ftz(C, f32_sub(C, ftz(C, a), ftz(C, b)))


def f32_mul_ftz(C, a, b):
    """``F32MulFtz_v1``: ftz(RN(ftz(a) * ftz(b)))."""
    return ftz(C, fp.f32_mul(C, ftz(C, a), ftz(C, b), ir_nan=True))


def f32_fma_ftz(C, a, b, c):
    """``F32FmaFtz_v1``: ftz(RN(ftz(a) * ftz(b) + ftz(c)))."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c)))


def f32_fma_sub_ftz(C, a, b, c):
    """``F32FmaSubFtz_v1``: ftz(RN(ftz(a) * ftz(b) - ftz(c)))."""
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), fp.f32_neg(ftz(C, c))))


def f32_gt(C, a, b):
    """``F32GtStrict_v1``: a > b on f32 values (NaN false, +0 == -0)."""
    _, mag_gt = sub(C, b[:31], a[:31], 31)
    _, mag_lt = sub(C, a[:31], b[:31], 31)
    both_zero = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    sa, sb = a[31], b[31]
    gt = C.AND(C.AND(NOT(sa), NOT(sb)), mag_gt) ^ C.AND(C.AND(NOT(sa), sb), NOT(both_zero)) ^ C.AND(C.AND(sa, sb), mag_lt)
    return [C.AND(C.AND(gt, NOT(_is_nan(C, a))), NOT(_is_nan(C, b)))]


def f32_max(C, a, b):
    """``F32Max_v1``: ``x > y ? x : y`` on the flushed operands (FA2's ``MaxOp``)."""
    x, y = ftz(C, a), ftz(C, b)
    return mux(C, f32_gt(C, x, y)[0], x, y)


def guard_neg_inf_zero(C, m):
    """``GuardNegInfZero_v1``: ``m == -inf ? 0 : m``."""
    keep = NOT(_eq_const(C, m, 0xFF800000))
    return [C.AND(keep, b) for b in m]


def mufu_ex2(C, x):
    """``MufuEx2Ftz_v1`` (``fa2_model.cpp`` mufu_ex2_bits): x as a signed fixed-point value with 23 fraction bits; the table at
    its fraction, the integer part added to the table word's exponent. For |x| < 2^-63 the model shifts a 64-bit word by 64 or
    more, which C++ leaves undefined and x86 wraps modulo 64; the IR evaluates it that way, and so does this piece."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    big = NOT(_ult_const(C, e, 157))                          # e - 127 >= 30
    neg = _ult_const(C, e, 127)
    # e < 127: K = mant >> ((127 - e) & 63), the model's 64-bit shift as x86 executes it (a count of 64 or more wraps)
    d = sub(C, const_bits(127, 8), e, 8)[0][:6]
    over = C.AND(neg, NOT(_ult_const(C, d, 25)))                # a right shift past the 24-bit mantissa: K = 0
    sh = mux(C, neg, sub(C, const_bits(24, 6), d, 6)[0], sub(C, e, const_bits(103, 8), 8)[0][:6])
    K = fp.shl(C, m + [ONE] + [ZERO] * 53, sh)[24:77]          # mant << (e - 127), or the right shift above; 53 bits
    K = [C.AND(NOT(over), k) for k in K]
    z = NOT(or_reduce(C, K[:23]))
    fixed = add(C, [k ^ s for k in K + [ZERO]], [C.AND(s, z)], width=54)   # s ? (z ? -K : ~K) : K
    t = table_read(C, "ex2", fixed[:23])
    n = fixed[23:54]
    re = add(C, list(t[23:31]) + [ZERO] * 25, n + [n[30]] * 2, width=33)
    neg = re[32]
    zero = OR(C, neg, NOT(or_reduce(C, re)))
    ovf = C.AND(NOT(neg), NOT(_ult_const(C, re[:32], 255)))
    inf_or_zero = mux(C, s, const_bits(0, 32), const_bits(INF, 32))
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), inf_or_zero)), (e_zero, const_bits(ONE_F, 32)),
                     (big, inf_or_zero), (ovf, const_bits(INF, 32)), (zero, const_bits(0, 32))],
                 list(t[:23]) + re[:8] + [ZERO])


def rcp_post(C, xb, t):
    """M0's (``verity_flock.tail``): MUFU.RCP of xb from the rcp table's word t at xb's mantissa."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, xb)
    re, _ = sub(C, list(t[23:31]) + [ZERO, ZERO], sub(C, e + [ZERO, ZERO], const_bits(127, 10), 10)[0], 10)
    neg = re[9]
    zero = OR(C, neg, NOT(or_reduce(C, re)))
    big = C.AND(NOT(neg), OR(C, re[8], _eq_const(C, re[:8], 0xFF)))
    body = list(t[:23]) + re[:8] + [s]
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), [ZERO] * 31 + [s])), (e_zero, const_bits(INF, 31) + [s]),
                     (zero, [ZERO] * 31 + [s]), (big, const_bits(INF, 31) + [s])], body)


def mufu_rcp(C, x):
    """``MufuRcpFtz_v1`` (``fa2_model.cpp`` mufu_rcp_bits): the rcp table at the mantissa, the exponent handled exactly."""
    return rcp_post(C, x, table_read(C, "rcp", x[:23]))


def fa2_inv_sum(C, s):
    """``Fa2InvSum_v1``: ``(sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)``."""
    one = OR(C, NOT(or_reduce(C, s[:31])), _is_nan(C, s))
    return mux(C, one, const_bits(ONE_F, 32), mufu_rcp(C, s))


# --- the norms' row scalars (M0's circuits around the table reads) -----------------------------------------------------------

def _abs_gt_big(C, x):
    """``|x| > 2^126`` as f32 compares it (false for NaN)."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    nan = C.AND(e_max, m_nz)
    ge254 = C.AND(e[7], C.AND(e[6], C.AND(e[5], C.AND(e[4], C.AND(e[3], C.AND(e[2], e[1]))))))
    return C.AND(NOT(nan), OR(C, ge254, C.AND(_eq_const(C, e, 253), m_nz)))


def _scale(C, x, k: int):
    return fp.f32_mul(C, x, const_bits(k, 32))


def _parity_index(xb):
    """``(par << 23) | mantissa``, par = (e - 127) & 1 = NOT e[0]: the rsq / sqrt tables' index."""
    return list(xb[:23]) + [NOT(xb[23])]


def _rsq_post(C, xb, v):
    s, e, m, e_max, e_zero, m_nz = _fields(C, xb)
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    re = sub(C, list(v[23:31]) + [ZERO], k[1:] + [k[8]], 9)[0]
    return _word(C, [(e_max, mux(C, m_nz, const_bits(QNAN_MUFU, 32), mux(C, s, const_bits(QNAN_MUFU, 32), const_bits(0, 32)))),
                     (e_zero, const_bits(INF, 31) + [s]), (s, const_bits(QNAN_MUFU, 32))], list(v[:23]) + re)


def rsqrt_approx(C, x):
    """``RsqrtApprox_v1``: MUFU.RSQ with the subnormal range scaling (x * 2^24, the result * 2^12)."""
    p0 = NOT(_fields(C, x)[4])
    xs = mux(C, p0, x, _scale(C, x, TWO24))
    r = _rsq_post(C, xs, table_read(C, "rsq", _parity_index(xs)))
    special = OR(C, _fields(C, r)[3], _fields(C, r)[4])
    return mux(C, OR(C, p0, special), r, list(r[:23]) + add(C, r[23:31], const_bits(12, 8), width=8) + [r[31]])


def mufu_sqrt_ftz(C, x):
    """``MufuSqrtFtz_v1``: MUFU.SQRT (``sqrt.approx.ftz.f32``)."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    v = table_read(C, "sqrt", _parity_index(x))
    k = sub(C, e + [ZERO], const_bits(127, 9), 9)[0]
    re = add(C, list(v[23:31]) + [ZERO], k[1:] + [k[8]], width=9)
    return _word(C, [(e_max, mux(C, OR(C, m_nz, s), const_bits(QNAN_MUFU, 32), const_bits(INF, 32))),
                     (e_zero, [ZERO] * 31 + [s]), (s, const_bits(QNAN_MUFU, 32))], list(v[:23]) + re)


def _div_scale(C, a, b):
    big = _abs_gt_big(C, b)
    a1, b1 = mux(C, big, _scale(C, a, QUARTER), a), mux(C, big, _scale(C, b, QUARTER), b)
    tiny = _fields(C, b1)[4]
    return mux(C, tiny, _scale(C, a1, TWO24), a1), mux(C, tiny, _scale(C, b1, TWO24), b1)


def div_full_rcp(C, b):
    """``DivFullRcp_v1``: ``div.full.f32``'s reciprocal part: range-scale b, MUFU.RCP, times the matching scaled 1."""
    big = _abs_gt_big(C, b)
    tiny = _fields(C, b)[4]
    fb = mux(C, big, _scale(C, b, QUARTER), mux(C, tiny, _scale(C, b, TWO24), b))
    a2, b2 = _div_scale(C, const_bits(ONE_F, 32), fb)
    return fp.f32_mul(C, a2, rcp_post(C, b2, table_read(C, "rcp", b2[:23])), ir_nan=True)


def div_full_scale_a(C, a, b):
    """``DivFullScaleA_v1``: ``a * 0.25`` when ``|b| > 2^126``, ``a * 2^24`` when ``|b| < 2^-126``, else a (NaN b: a)."""
    mag = b[:31]
    big = C.AND(NOT(_ult_const(C, mag, 0x7E800001)), _ult_const(C, mag, 0x7F800001))
    small = _ult_const(C, mag, 0x00800000)
    return mux(C, big, fp.f32_mul(C, a, fp.f32_const(QUARTER), ir_nan=True),
               mux(C, small, fp.f32_mul(C, a, fp.f32_const(TWO24), ir_nan=True), list(a)))


# --- the sampler: Philox, the Gumbel noise (libdevice logf / log1pf), the row's bit logic -----------------------------------

PHILOX_KEY_A, PHILOX_KEY_B = 0x9E3779B9, 0xBB67AE85
PHILOX_ROUND_A, PHILOX_ROUND_B = 0xD2511F53, 0xCD9E8D57
TL_RAND_SCALE = 0x2FFFFFFF        # uint_to_uniform_float's scale; vLLM's floor _TL_RAND_MIN is the same word
LOGF_POLY = (0xBE055027, 0x3E1039F6, 0xBDF8CDCC, 0x3E0F2955, 0xBE2AD8B9, 0x3E4CED0B, 0xBE7FFF22, 0x3EAAAA78)
LOG1PF_POLY = (0xBD39BF78, 0x3DD80012, 0xBE0778E0, 0x3E146475, 0xBE2A68DD, 0x3E4CAF9E, 0xBE800042, 0x3EAAAAE6)
LN2, TWO_M23, TWO_P23, MINUS_ONE, MINUS_HALF, QUARTER_F = 0x3F317218, 0x34000000, 0x4B000000, 0xBF800000, 0xBF000000, 0x3E800000


def _c32(v: int):
    return const_bits(v & 0xFFFFFFFF, 32)


def _fma_ftz(C, a, b, c):
    return f32_fma_ftz(C, a, _c32(b) if isinstance(b, int) else b, _c32(c) if isinstance(c, int) else c)


def _mul_ftz(C, a, b):
    return f32_mul_ftz(C, a, _c32(b) if isinstance(b, int) else b)


def _add_ftz(C, a, b):
    return f32_add_ftz(C, a, _c32(b) if isinstance(b, int) else b)


def _i32_to_f32(C, x):
    """``(float)`` of a signed 32-bit integer, RN."""
    s = x[31]
    mag = add(C, [b ^ s for b in x], [s], width=32)
    return fp.round_f32(C, s, const_bits(158, 9), 0, mag, ZERO)


def _signed_gt_const(C, x, k: int):
    """``(int32) x > k``: unsigned ``x ^ 2^31 >= (k + 1) ^ 2^31``."""
    return NOT(_ult_const(C, x[:31] + [NOT(x[31])], ((k + 1) & 0xFFFFFFFF) ^ 0x80000000))


def philox_word0(C, c0, c1, c2, c3, k0, k1):
    """Word 0 of Philox4x32-10 (Random123, Triton's ``philox_impl``) on 32-bit words."""
    for _ in range(10):
        pb = gf2.multiply(C, c2, const_bits(PHILOX_ROUND_B, 32))
        pa = gf2.multiply(C, c0, const_bits(PHILOX_ROUND_A, 32))
        c0, c1, c2, c3 = ([x ^ y ^ z for x, y, z in zip(pb[32:64], c1, k0)], pb[:32],
                          [x ^ y ^ z for x, y, z in zip(pa[32:64], c3, k1)], pa[:32])
        k0 = add(C, k0, const_bits(PHILOX_KEY_A, 32), width=32)
        k1 = add(C, k1, const_bits(PHILOX_KEY_B, 32), width=32)
    return c0


def gumbel_stream_key(C, seed, pos):
    """``GumbelStreamKey_v1``: ``tl.randint(seed, pos)``, Philox word 0 with key (seed_lo, seed_hi), counter (pos_lo, pos_hi, 0, 0)."""
    zero = [ZERO] * 32
    return philox_word0(C, pos[:32], pos[32:64], zero, zero, seed[:32], seed[32:64])


def nv_logf(C, a):
    """libdevice ``__nv_logf`` on an f32 word (FTZ), operation for operation as ``sampling.nv_logf``."""
    small = OR(C, C.AND(a[31], NOT(_is_nan(C, a))), C.AND(NOT(a[31]), _ult_const(C, a[:31], 0x00800000)))   # a < FLT_MIN
    a2 = mux(C, small, _mul_ftz(C, a, TWO_P23), a)
    i0 = mux(C, small, _c32(0xC1B80000), _c32(0))
    e = sub(C, a2, _c32(1059760811), 32)[0]
    e = [ZERO] * 23 + e[23:]
    m = sub(C, a2, e, 32)[0]
    i2 = _fma_ftz(C, _i32_to_f32(C, e), TWO_M23, i0)
    f = _add_ftz(C, m, MINUS_ONE)
    r = _fma_ftz(C, _c32(LOGF_POLY[0]), f, LOGF_POLY[1])
    for k in LOGF_POLY[2:]:
        r = _fma_ftz(C, r, f, k)
    r = _fma_ftz(C, r, f, MINUS_HALF)
    r = f32_mul_ftz(C, r, f)
    r = f32_fma_ftz(C, r, f, f)
    r = _fma_ftz(C, i2, LN2, r)
    r = mux(C, NOT(_ult_const(C, a2, 0x7F800000)), _fma_ftz(C, a2, INF, INF), r)
    return mux(C, NOT(or_reduce(C, a2[:31])), _c32(0xFF800000), r)


def nv_log1pf(C, a):
    """libdevice ``__nv_log1pf`` (FTZ): ``__fadd_rz(a, 1.0f)`` rounded toward zero, the binade selector, the polynomial, as
    ``sampling.nv_log1pf``."""
    s = ftz(C, fp.f32_add(C, ftz(C, a), _c32(ONE_F), rz=True))
    e = sub(C, s, _c32(1061158912), 32)[0]
    e = [ZERO] * 23 + e[23:]
    m1 = sub(C, a, e, 32)[0]
    t = sub(C, _c32(1082130432), e, 32)[0]
    r0 = _fma_ftz(C, _c32(QUARTER_F), t, MINUS_ONE)
    u = f32_add_ftz(C, m1, r0)
    ef = _mul_ftz(C, _i32_to_f32(C, e), TWO_M23)
    r = _fma_ftz(C, _c32(LOG1PF_POLY[0]), u, LOG1PF_POLY[1])
    for k in LOG1PF_POLY[2:]:
        r = _fma_ftz(C, r, u, k)
    r = _fma_ftz(C, r, u, MINUS_HALF)
    r = f32_mul_ftz(C, r, u)
    r = f32_fma_ftz(C, r, u, u)
    r = _fma_ftz(C, ef, LN2, r)
    big = NOT(_ult_const(C, a, 0x7F800000))
    r = mux(C, C.AND(big, _signed_gt_const(C, a, -1082130432)), _fma_ftz(C, a, INF, INF), r)
    return mux(C, C.AND(big, NOT(or_reduce(C, a[:31]))), _c32(0x80000000), r)


def gumbel_noise_lane(C, key, v):
    """``GumbelNoiseLane_v1``: ``-logf(-log1pf(-max(rand(key, v), 2^-31')))`` in f32 (FTZ); rand is Philox word 0 of counter
    (v, 0, 0, 0) under key (key, 0), made uniform by ``uint_to_uniform_float``."""
    zero = [ZERO] * 32
    w = philox_word0(C, v, zero, zero, zero, key, zero)
    xi = [b ^ w[31] for b in w[:31]] + [ZERO]                   # i32 w, or -w - 1 when negative
    u = f32_mul_ftz(C, _i32_to_f32(C, xi), _c32(TL_RAND_SCALE))
    u = mux(C, _ult_const(C, u[:31], TL_RAND_SCALE), _c32(TL_RAND_SCALE), u)
    l1 = nv_log1pf(C, u[:31] + [NOT(u[31])])
    g = nv_logf(C, l1[:31] + [NOT(l1[31])])
    return g[:31] + [NOT(g[31])]


def f32_eq(C, a, b):
    """``F32Eq_v1``: IEEE ``a == b`` (+0 == -0; NaN unequal to everything)."""
    same = and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])
    zeros = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    return [C.AND(C.AND(NOT(_is_nan(C, a)), NOT(_is_nan(C, b))), OR(C, same, zeros))]


def bit_at(C, word, i):
    """``BitAtx{W}_v1``: bit i of a W-bit word, 0 when i (signed) is outside [0, W)."""
    W = len(word)
    k = max(1, (W - 1).bit_length())
    cur = list(word) + [ZERO] * ((1 << k) - W)
    for b in range(k):
        cur = [cur[2 * j] ^ C.AND(i[b], cur[2 * j + 1] ^ cur[2 * j]) for j in range(len(cur) // 2)]
    return [C.AND(_ult_const(C, i, W), cur[0])]


#: primitive -> piece(C, *operand words)
PIECES = {
    "F32AddFtz_v1": f32_add_ftz,
    "F32SubFtz_v1": f32_sub_ftz,
    "F32MulFtz_v1": f32_mul_ftz,
    "F32FmaFtz_v1": f32_fma_ftz,
    "F32FmaSubFtz_v1": f32_fma_sub_ftz,
    "F32Max_v1": f32_max,
    "GuardNegInfZero_v1": guard_neg_inf_zero,
    "MufuEx2Ftz_v1": mufu_ex2,
    "MufuRcpFtz_v1": mufu_rcp,
    "Fa2InvSum_v1": fa2_inv_sum,
    "RsqrtApprox_v1": rsqrt_approx,
    "MufuSqrtFtz_v1": mufu_sqrt_ftz,
    "DivFullRcp_v1": div_full_rcp,
    "DivFullScaleA_v1": div_full_scale_a,
    "F32GtStrict_v1": f32_gt,
    "SelectF32_v1": lambda C, s, a, b: mux(C, s[0], a, b),
    "SelectI32_v1": lambda C, s, a, b: mux(C, s[0], a, b),
    "I32Add_v1": lambda C, a, b: add(C, a, b, width=32),
    "GumbelStreamKey_v1": gumbel_stream_key,
    "GumbelNoiseLane_v1": gumbel_noise_lane,
    "F32Eq_v1": f32_eq,
    "BitNot_v1": lambda C, a: [NOT(a[0])],
    "BitOr_v1": lambda C, a, b: [OR(C, a[0], b[0])],
}
#: primitive families whose id carries a width: pattern -> piece
FAMILIES = {r"BitAtx\d+_v1": bit_at}
