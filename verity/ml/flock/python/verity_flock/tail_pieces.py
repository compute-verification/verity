"""Circuits for the primitives a template's units leave outside: attention's softmax (MUFU.EX2, FA2's inverse sum, and the FTZ
f32 ops, ``F32Max`` and ``GuardNegInfZero`` from `verity.ml.circuits.ops.boolean.softmax`; PTX max / min, the E4M3 cast, the E4M3 widening
and the E2M1 cast from `verity.ml.circuits.ops.boolean.scalar`), the norms' row scalars (MUFU.RSQ and SQRT, ``div.full``'s reciprocal and
numerator scaling) and the sampler lane's compare, selects, bit AND and integer add. Each is its IR primitive bit for bit
(``ir_lower.PIECES``; the piece test and circuit-check compare them).

A MUFU operation is a table lookup inside exponent and special-case logic (the C++ models ``fa2_model.cpp`` and
``rms_triton_model.cpp``, and ``live/src/ir_tail.rs``): the piece computes the table index, reads the pinned table
(:func:`table_read`, a ``gf2`` lookup, which a Flock statement proves with a lookup slot: M0's ``live/src/lookup.rs``) and
finishes the result. The RSQ, SQRT and RCP circuits around the read are M0's (``verity_flock.tail``, verity PR #83); EX2's and
FA2's inverse sum follow ``fa2_model.cpp``.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.gather import gather as _bits_gather
from verity.ml.circuits.ops.boolean.scalar import e4m3_to_f32, f32_to_e2m1_sat, f32_to_e4m3_sat, fmaxmin as _fmaxmin
from verity.ml.circuits.ops.boolean.softmax import (_eq_const, _is_nan, f32_add_ftz, f32_fma_ftz, f32_fma_sub_ftz, f32_gt, f32_max, f32_mul_ftz,
                                       f32_sub, f32_sub_ftz, ftz, guard_neg_inf_zero)

from . import fp
from . import gf2
from . import unit_fp4
from .gf2 import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub

#: table name -> index bits (the MUFU tables' sha256 is ir_lower.TABLES[name], the others' TABLE_SHA256[name])
TABLE_BITS = {"rsq": 24, "sqrt": 24, "rcp": 23, "ex2": 23, "tanh_mufu": 27, "tanh_rn": 27, "gelu_tanh_bf16": 16}
#: the bits of a table word a piece reads: exponent and mantissa (bit 31 is zero in every measured table and for |tanh|)
VALUE_BITS = 31
TABLE_VALUE_BITS = {"gelu_tanh_bf16": 16}
#: the tables that are not MUFU tables of the Flock statement's tail: gemma's tanh (the measured MUFU.TANH; the IR's RN32 tanh)
#: over the 2^27 words their low 27 bits index, and the gelu-tanh activation of every bf16 word
TABLE_SHA256 = {"tanh_mufu": "4c25bacba332fda09d215e64dc5ca3ec9a7dfdc98a3ea7971f5422054b2943e0",
                "tanh_rn": "32fb35b4a1d0402a8f4b96a601c8ffc5e986266d20666a224eaf3bffeace308a",
                "gelu_tanh_bf16": "892c2fbca46fed05daf5b4fca3b49a392e93fdc6b826f8fd7bfa3c9bae0d8865"}
QNAN_MUFU = 0x7FFFFFFF
INF = 0x7F800000
ONE_F = 0x3F800000
#: every NaN result of `NvLogf_v2`'s logs (`sampling.LOG_NAN`, the GPU's canonical NaN)
LOG_NAN = 0x7FFFFFFF
BIG = 0x7E800000       # 2^126
TINY = 0x00800000      # 2^-126
QUARTER = 0x3E800000   # 0.25
TWO24 = 0x4B800000     # 2^24


def table_sha256(name: str) -> str:
    from .ir_lower import TABLES
    return TABLES.get(name) or TABLE_SHA256[name]


def _load_table(name: str):
    import hashlib

    import numpy as np

    if name in TABLE_SHA256:
        arr = np.ascontiguousarray(_DERIVED[name](), dtype="<u4")
    else:
        from verity.ml.circuits.ops import mufu
        arr = np.ascontiguousarray(mufu.table(name), dtype="<u4")
    if hashlib.sha256(arr.tobytes()).hexdigest() != table_sha256(name):
        raise ValueError(f"the {name} table is not the pinned one ({table_sha256(name)})")
    return arr


#: [identity below, saturated from) on |x| of each tanh table: the words its table holds. MUFU.TANH's are the IR's measured rules
#: (``verity.ml.circuits.ops.mufu.MUFU_TANH_RULES``); RN32(math.tanh)'s come from evaluating it on every word below 10.0 of both signs: x
#: itself below 0x39B89BA3, 1.0 from 0x41102CB4, odd everywhere
TANH_BOUNDS = {"tanh_mufu": (0x3B800001, 0x41000000), "tanh_rn": (0x39B89BA3, 0x41102CB4)}


def _tanh_index(lo: int, hi: int):
    assert 0 < hi - lo <= 1 << 27, "a contiguous region of at most 2^27 words has distinct low 27 bits"


def _mufu_tanh_table():
    """``MufuTanh_v1``'s measured table (``verity.ml.circuits.ops.mufu.mufu_tanh_shards``) at every positive word of [identity_below,
    saturate_from), by the word's low 27 bits."""
    import numpy as np
    from verity.ml.circuits.ops import mufu

    lo, hi = TANH_BOUNDS["tanh_mufu"]
    assert (lo, hi) == (mufu.MUFU_TANH_RULES["identity_below"], mufu.MUFU_TANH_RULES["saturate_from"])
    (sh,) = mufu.mufu_tanh_shards()
    out = np.zeros(1 << 27, np.uint32)
    for k in range((lo - sh.lo) // sh.block_words, (hi - 1 - sh.lo) // sh.block_words + 1):
        b = sh.block(k)
        w = sh.lo + k * sh.block_words + np.arange(len(b), dtype=np.int64)
        m = (w >= lo) & (w < hi)
        assert not (b[m] >> 31).any()
        out[w[m] & 0x7FFFFFF] = b[m]
    return out


def _tanh_rn_table():
    """``TanhF32Rn_v1`` (RN32 of ``math.tanh``) at every positive word of its table region, by the word's low 27 bits."""
    import math

    import numpy as np

    lo, hi = TANH_BOUNDS["tanh_rn"]
    out = np.zeros(1 << 27, np.uint32)
    for c0 in range(lo, hi, 1 << 22):
        w = np.arange(c0, min(c0 + (1 << 22), hi), dtype=np.uint32)
        t = np.array(list(map(math.tanh, w.view(np.float32).astype(np.float64).tolist())), np.float64).astype(np.float32)
        out[w & 0x7FFFFFF] = t.view(np.uint32)
    return out


def _gelu_table():
    """``GeluTanhMulBf16_v1``'s activation word ``dense.gelu_tanh_bf16_bits(g)`` for every bf16 word g."""
    import numpy as np
    from verity.ml.definitions.registry import dense

    return np.array([dense.gelu_tanh_bf16_bits(g) for g in range(1 << 16)], np.uint32)


_DERIVED = {"tanh_mufu": _mufu_tanh_table, "tanh_rn": _tanh_rn_table, "gelu_tanh_bf16": _gelu_table}


for _name in TABLE_BITS:
    gf2.TABLES.setdefault(_name, lambda n=_name: _load_table(n))


def table_read(C, table: str, idx):
    """The pinned table ``table``'s word at ``idx`` (its low 31 bits, or ``TABLE_VALUE_BITS``)."""
    assert len(idx) == TABLE_BITS[table]
    return C.lookup(table, list(idx), TABLE_VALUE_BITS.get(table, VALUE_BITS))


# --- helpers ---------------------------------------------------------------------------------------------------------------

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


# --- attention's softmax (the FTZ f32 ops, F32Max and GuardNegInfZero are `verity.ml.circuits.ops.boolean.softmax`'s) ---------------------

def mufu_ex2(C, x):
    """``MufuEx2Ftz_v1`` (``fa2_model.cpp`` mufu_ex2_bits): x as a signed fixed-point value with 23 fraction bits; the table at
    its fraction, the integer part added to the table word's exponent. A right shift past the mantissa is clamped (K = 0), so
    every |x| < 2^-23 reads the table's first word."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    big = NOT(_ult_const(C, e, 157))                          # e - 127 >= 30
    neg = _ult_const(C, e, 127)
    # e < 127: K = mant >> (127 - e), zero once the shift passes the 24-bit mantissa
    d8 = sub(C, const_bits(127, 8), e, 8)[0]
    d = d8[:6]
    over = C.AND(neg, NOT(_ult_const(C, d8, 25)))
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


# --- the `_v2` FP32 primitives: every NaN result the GPU's 0x7FFFFFFF (fp.F32_NAN_PTX), no operand's payload or sign ----------
_V2 = {"nan_word": fp.F32_NAN_PTX}


def f32_add_ftz_v2(C, a, b):
    return ftz(C, fp.f32_add(C, ftz(C, a), ftz(C, b), **_V2))


def f32_sub_ftz_v2(C, a, b):
    return ftz(C, fp.f32_sub(C, ftz(C, a), ftz(C, b), **_V2))


def f32_mul_ftz_v2(C, a, b):
    return ftz(C, fp.f32_mul(C, ftz(C, a), ftz(C, b), **_V2))


def f32_fma_ftz_v2(C, a, b, c):
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c), **_V2))


def f32_fma_sub_ftz_v2(C, a, b, c):
    return ftz(C, fp.f32_fma(C, ftz(C, a), ftz(C, b), fp.f32_neg(ftz(C, c)), **_V2))


def div_full_scale_a_v2(C, a, b):
    """``DivFullScaleA_v2``: as ``_v1``, each scaling FMUL's NaN 0x7FFFFFFF; an unscaled a (a NaN included) passes through."""
    mag = b[:31]
    big = C.AND(NOT(_ult_const(C, mag, 0x7E800001)), _ult_const(C, mag, 0x7F800001))
    small = _ult_const(C, mag, 0x00800000)
    return mux(C, big, fp.f32_mul(C, a, fp.f32_const(QUARTER), **_V2),
               mux(C, small, fp.f32_mul(C, a, fp.f32_const(TWO24), **_V2), list(a)))


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


def nv_logf(C, a, gpu: bool = False):
    """libdevice ``__nv_logf`` on an f32 word (FTZ), operation for operation as ``sampling.nv_logf``; with ``gpu`` every NaN result is
    ``LOG_NAN`` (``sampling.nv_logf_gpu``, ``NvLogf_v2``)."""
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
    # a2 >= 0x7F800000 (+inf, a NaN, a negative): fma(a2, inf, inf) is a2 quieted for a NaN, +inf, else the invalid NaN 0xFFC00000;
    # with gpu, +inf for +inf, else LOG_NAN
    if gpu:
        special = mux(C, _eq_const(C, a2, INF), _c32(INF), _c32(LOG_NAN))
    else:
        special = mux(C, _is_nan(C, a2), a2[:22] + [ONE] + a2[23:], mux(C, a2[31], _c32(0xFFC00000), _c32(INF)))
    r = mux(C, NOT(_ult_const(C, a2, 0x7F800000)), special, r)
    return mux(C, NOT(or_reduce(C, a2[:31])), _c32(0xFF800000), r)


def nv_log1pf(C, a, gpu: bool = False):
    """libdevice ``__nv_log1pf`` (FTZ): ``__fadd_rz(a, 1.0f)`` rounded toward zero, the binade selector, the polynomial, as
    ``sampling.nv_log1pf``; with ``gpu`` every NaN result is ``LOG_NAN`` (``sampling.nv_log1pf_gpu``)."""
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
    outside = C.AND(big, _signed_gt_const(C, a, -1082130432))                    # a < -1, +inf or a NaN
    r = mux(C, outside, mux(C, _eq_const(C, a, INF), _c32(INF), _c32(LOG_NAN)) if gpu else _fma_ftz(C, a, INF, INF), r)
    return mux(C, C.AND(big, NOT(or_reduce(C, a[:31]))), _c32(0x80000000), r)


def gumbel_noise_lane(C, key, v, gpu: bool = False):
    """``GumbelNoiseLane_v1``: ``-logf(-log1pf(-max(rand(key, v), 2^-31')))`` in f32 (FTZ); rand is Philox word 0 of counter
    (v, 0, 0, 0) under key (key, 0), made uniform by ``uint_to_uniform_float``.  With ``gpu``, ``_v2``: through the GPU-NaN logs."""
    zero = [ZERO] * 32
    w = philox_word0(C, v, zero, zero, zero, key, zero)
    xi = [b ^ w[31] for b in w[:31]] + [ZERO]                   # i32 w, or -w - 1 when negative
    u = f32_mul_ftz(C, _i32_to_f32(C, xi), _c32(TL_RAND_SCALE))
    u = mux(C, _ult_const(C, u[:31], TL_RAND_SCALE), _c32(TL_RAND_SCALE), u)
    l1 = nv_log1pf(C, u[:31] + [NOT(u[31])], gpu)
    g = nv_logf(C, l1[:31] + [NOT(l1[31])], gpu)
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


# --- the other rows' scalars: FA3's inverse sum, the greedy argmax, the FP8 linear's quantizer, the MoE router ---------------

def fa3_inv_sum(C, s):
    """``Fa3InvSum_v1``: ``(sum == 0 || sum != sum) ? 0 : MUFU.RCP(sum)``."""
    zero = OR(C, NOT(or_reduce(C, s[:31])), _is_nan(C, s))
    return [C.AND(NOT(zero), b) for b in mufu_rcp(C, s)]


def bf16_gt(C, a, b):
    """``Bf16GtStrict_v1``: a > b on bf16 values (NaN false, +0 == -0)."""
    _, mag_gt = sub(C, b[:15], a[:15], 15)
    _, mag_lt = sub(C, a[:15], b[:15], 15)
    both_zero = NOT(OR(C, or_reduce(C, a[:15]), or_reduce(C, b[:15])))
    sa, sb = a[15], b[15]
    gt = C.AND(C.AND(NOT(sa), NOT(sb)), mag_gt) ^ C.AND(C.AND(NOT(sa), sb), NOT(both_zero)) ^ C.AND(C.AND(sa, sb), mag_lt)
    nan_a = C.AND(and_reduce(C, a[7:15]), or_reduce(C, a[:7]))
    nan_b = C.AND(and_reduce(C, b[7:15]), or_reduce(C, b[:7]))
    return [C.AND(C.AND(gt, NOT(nan_a)), NOT(nan_b))]


def f32_sat(C, a):
    """``F32Sat_v1`` (``cvt.sat.f32.f32``): clamp to [+0, 1]; NaN and every negative word (-0 included) give +0."""
    r = mux(C, NOT(_ult_const(C, a[:31], ONE_F)), _c32(ONE_F), a)
    keep = NOT(OR(C, a[31], _is_nan(C, a)))
    return [C.AND(keep, x) for x in r[:31]] + [ZERO]


def i32_le(C, a, b):
    """``I32Le_v1``: signed ``a <= b``."""
    return [NOT(sub(C, b[:31] + [NOT(b[31])], a[:31] + [NOT(a[31])], 32)[1])]


def _gather(C, tok, col, fill: int):
    """``col[tok]`` for ``0 <= tok < V``, else ``fill``: a multiplexer tree over the candidates by tok's low bits (a candidate
    without a partner passes up, since selecting the missing one means tok >= V), then the range select."""
    cur = [list(c) for c in col]
    for b in range(max(1, (len(col) - 1).bit_length())):
        cur = [mux(C, tok[b], cur[i + 1], cur[i]) if i + 1 < len(cur) else cur[i] for i in range(0, len(cur), 2)]
    return mux(C, _ult_const(C, tok, len(col)), cur[0], const_bits(fill, len(cur[0])))


def gather_bf16(C, tok, *col):
    """``GatherBf16x{V}_v1``: out of range, 0x7FFF."""
    return _gather(C, tok, col, 0x7FFF)


def gather_f32(C, tok, *col):
    """``GatherF32x{V}_v1``: out of range, 0x7FC00000."""
    return _gather(C, tok, col, 0x7FC00000)


def gather_e4m3(C, tok, *col):
    """``GatherE4m3x{V}_v1`` (a routed expert's FP8 weight word): out of range, 0x7F (the E4M3 NaN)."""
    return _gather(C, tok, col, 0x7F)


def moe_select_bf16(C, e, *column):
    """``MoeSelectBf16x{E}_v1`` (`verity.ml.circuits.ops.moe`: a static MoE pair unit's select of its expert's word) by the
    construction of its Boolean form ``MoeSelectBf16x{E}_v2`` (`verity.ml.circuits.ops.boolean.moe`), gate for gate: the
    multiplexer tree over e's low ceil(log2 E) bits (`boolean.gather`, a word past E standing for 0x7FFF), then 0x7FFF where
    any higher bit is set. That form is made per E on demand, so `ir_lower._boolean_build` finds it only in a process that made
    it."""
    b = (len(column) - 1).bit_length()
    return mux(C, or_reduce(C, e[b:]), const_bits(0x7FFF, 16), _bits_gather(C, e[:b], list(column), 0x7FFF))


# --- gemma-2's tanh family: table reads over |x| between an identity and a saturation region ---------------------------------

def _tanh(C, x, table: str, nan_word: int):
    lo, hi = TANH_BOUNDS[table]
    _tanh_index(lo, hi)
    a, s = x[:31], x[31]
    t = table_read(C, table, a[:27])
    return _word(C, [(_is_nan(C, x), _c32(nan_word)), (NOT(_ult_const(C, a, hi)), const_bits(ONE_F, 31) + [s]),
                     (_ult_const(C, a, lo), list(x))], list(t) + [s])


def mufu_tanh(C, x):
    """``MufuTanh_v1`` (MUFU.TANH on sm_89): x below the identity bound, ±1 from the saturation bound, 0x7FFFFFFF for NaN, else
    the measured table at |x| with x's sign (the table's odd symmetry, measured with 0 violations)."""
    return _tanh(C, x, "tanh_mufu", QNAN_MUFU)


def tanh_rn(C, x):
    """``TanhF32Rn_v1``: RN32(tanh(x)); x below the identity bound, ±1 from the saturation bound (both found by scanning every
    word), 0x7FC00000 for NaN, else its table at |x| with x's sign."""
    return _tanh(C, x, "tanh_rn", 0x7FC00000)


def gelu_tanh_mul_bf16(C, g, u):
    """``GeluTanhMulBf16_v1``: ``bf16(f32(act(g)) * (f32(u) + 0))``, act the gelu-tanh activation word of g (a 2^16 table of the
    IR's own function), both conversions RN with the F2FP NaN 0x7FFF."""
    act = table_read(C, "gelu_tanh_bf16", list(g))
    uf = fp.f32_add(C, fp.bf16_to_f32(u), _c32(0), ir_nan=True)
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(act), uf, ir_nan=True), 0x7FFF)


def topp_numkeep(C, nd, q):
    """``TopPNumkeep_v1``: ``min(max(nd - qi, 1), nd)`` on u32 counts, qi the f32 q truncated to a u32 (0 unless q is finite and
    > 0, all ones from 2^32 on); 0 when nd is 0."""
    s, e, m = q[31], list(q[23:31]), list(q[:23])
    pos = C.AND(NOT(s), NOT(and_reduce(C, e)))
    ge1 = NOT(_ult_const(C, e, 127))
    big = NOT(_ult_const(C, e, 127 + 32))
    ip = fp.shl(C, m + [ONE] + [ZERO] * 32, sub(C, e, const_bits(127, 8), 8)[0][:5])[23:55]
    qi = [C.AND(pos, OR(C, big, C.AND(ge1, b))) for b in ip]
    lt = sub(C, qi, nd, 32)[1]
    r = mux(C, lt, sub(C, nd, qi, 32)[0], const_bits(1, 32))
    nz = or_reduce(C, nd)
    return [C.AND(nz, b) for b in r]


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
    # the top-p keep word's word gates (verity.ml.definitions.registry.topp_word_gates)
    "BitXor_v1": lambda C, a, b: [a[0] ^ b[0]],
    "U32Xor_v1": lambda C, a, b: [x ^ y for x, y in zip(a, b)],
    "SelectBit_v1": lambda C, c, a, b: mux(C, c[0], a, b),
    "I32ToF32_v1": _i32_to_f32,
    "NvLogf_v1": nv_logf,
    "NvLogf_v2": lambda C, a: nv_logf(C, a, gpu=True),
    "TopPNumkeep_v1": topp_numkeep,
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
    "GumbelNoiseLane_v2": lambda C, key, v: gumbel_noise_lane(C, key, v, gpu=True),
    "F32Eq_v1": f32_eq,
    "BitNot_v1": lambda C, a: [NOT(a[0])],
    "BitOr_v1": lambda C, a, b: [OR(C, a[0], b[0])],
    "Fa3InvSum_v1": fa3_inv_sum,
    "Bf16GtStrict_v1": bf16_gt,
    "SelectBf16_v1": lambda C, s, a, b: mux(C, s[0], a, b),
    "F32Fabs_v1": lambda C, a: list(a[:31]) + [ZERO],
    "F32Neg_v1": lambda C, a: fp.f32_neg(a),
    "F32Fmaxf_v1": lambda C, a, b: _fmaxmin(C, a, b, True),
    "F32Fminf_v1": lambda C, a, b: _fmaxmin(C, a, b, False),
    "F32ToE4m3Sat_v1": f32_to_e4m3_sat,
    # NVFP4 quantization's casts and the bit AND (PoUW's units; verity.ml.definitions.registry.nvfp4)
    "E4m3ToF32_v1": e4m3_to_f32,
    "F32ToE2m1Sat_v1": f32_to_e2m1_sat,
    "BitAnd_v1": lambda C, a, b: [C.AND(a[0], b[0])],
    "F32Sub_v1": fp.f32_sub,
    "F32FmaRm_v1": lambda C, a, b, c: fp.f32_fma(C, a, b, c, rm=True),
    "F32AddFtz_v2": f32_add_ftz_v2,
    "F32SubFtz_v2": f32_sub_ftz_v2,
    "F32MulFtz_v2": f32_mul_ftz_v2,
    "F32FmaFtz_v2": f32_fma_ftz_v2,
    "F32FmaSubFtz_v2": f32_fma_sub_ftz_v2,
    "DivFullScaleA_v2": div_full_scale_a_v2,
    "F32Sub_v2": lambda C, a, b: fp.f32_sub(C, a, b, **_V2),
    "F32FmaRm_v2": lambda C, a, b, c: fp.f32_fma(C, a, b, c, rm=True, **_V2),
    "F32Sat_v1": f32_sat,
    "F32BitsShl23_v1": lambda C, a: [ZERO] * 23 + list(a[:9]),
    "F32IsFinite_v1": lambda C, a: [NOT(and_reduce(C, a[23:31]))],
    "I32Eq_v1": lambda C, a, b: [and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])],
    "HopperE4m3QgmmaDot32_v1": lambda C, acc, *ab: fp.tc_dot_e4m3(C, acc, list(ab[:32]), list(ab[32:])),
    "BlackwellE4m3QmmaDot32_v1": lambda C, acc, *ab: fp.tc_dot_e4m3(C, acc, list(ab[:32]), list(ab[32:]), groups=(32,), W=26, F=-133),
    "BlackwellNvf4OmmaDot64_v1": lambda C, acc, *ab: unit_fp4.step_total(C, acc, ab[:64], ab[64:128], ab[128:132], ab[132:136]),
    "BlackwellMxf4OmmaDot64_v1": lambda C, acc, *ab: unit_fp4.step_total(C, acc, ab[:64], ab[64:128], ab[128:130], ab[130:132], ue8m0=True),
    "I32Le_v1": i32_le,
    "MufuTanh_v1": mufu_tanh,
    "TanhF32Rn_v1": tanh_rn,
    "GeluTanhMulBf16_v1": gelu_tanh_mul_bf16,
}
#: primitive families whose id carries a width: pattern -> piece
FAMILIES = {r"BitAtx\d+_v1": bit_at, r"GatherBf16x\d+_v1": gather_bf16, r"GatherF32x\d+_v1": gather_f32,
            r"GatherE4m3x\d+_v1": gather_e4m3, r"MoeSelectBf16x\d+_v1": moe_select_bf16}
