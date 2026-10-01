"""``veritor.reduce-mufu@1``: the FP32 scalar gates and MUFU table lookups of the vLLM reduction kernels.

The Python mirror of ``backends/sp1/common/src/reduce.rs`` (the guest's semantics of this family), transcribed on
the raw encodings -- integer arithmetic only, no host float operation -- so both agree on every word, and the
manifest is the guest's, so :func:`make_reduce_mufu_gate_set` has the digest the guest checks a statement against.

* ``f32_add`` / ``f32_mul`` / ``f32_div``: IEEE binary32, round to nearest even, subnormals kept, SSE's NaN rule
  (the first NaN operand quieted; an invalid operation gives ``0xFFC00000``).
* ``f32_fma``: one rounding of the exact ``a*b + c``, every NaN ``0x7FC00000``.
* ``bf16_to_f32``, ``f2fp_bf16`` (RNE, NaN ``0x7FFF``), ``bf16_add_f2fp``, ``rope_out``, ``rope_out_add``,
  ``bf16_gt``, ``select_*``, ``i32_*``: the registered primitives of the same names.
* ``mufu_sqrt_tab``, ``div_full_rcp``, ``rsqrt_approx``: the second argument is a row of a committed MUFU table as
  the 64-bit word ``(index << 32) | word``; the gate recomputes the index the hardware reads for its FP32 operand
  and raises (the guest rejects) when the row is any other, so only the true row can be used.  ``*_index`` give
  that index, for building the opening.
* ``in`` / ``weight``: 64-bit sources, each word decoded by its own leaf schema.
"""

from __future__ import annotations

from .gateset import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet

__all__ = [
    "REDUCE_MUFU_DIGEST",
    "REDUCE_MUFU_NAME",
    "REDUCE_MUFU_VERSION",
    "div_full_rcp_index",
    "make_reduce_mufu_gate_set",
    "mufu_rcp_index",
    "mufu_sqrt_index",
    "rsqrt_approx_index",
]

REDUCE_MUFU_NAME = "veritor.reduce-mufu"
REDUCE_MUFU_VERSION = "1"
#: ``GateSet::resolve("veritor.reduce-mufu@1", 32).digest()`` (``reduce.rs`` test ``family_digest_matches_the_python_mirror``).
REDUCE_MUFU_DIGEST = "a348a7bbe15090308af57cb7e5f1dc035263cc9092b853ba7a2dff7b9d462837"

F32_NAN = 0x7FC0_0000
F32_INDEFINITE = 0xFFC0_0000
BF16_NAN_TORCH = 0x7FC0
BF16_NAN_F2FP = 0x7FFF
_TWO_POW_126 = 0x7E80_0000
_MIN_NORMAL = 0x0080_0000
_QUARTER = 0x3E80_0000
_TWO_POW_24 = 0x4B80_0000
_TWO_POW_12 = 0x4580_0000


def _exp(u: int) -> int:
    return (u >> 23) & 0xFF


def _is_nan(u: int) -> bool:
    return _exp(u) == 0xFF and (u & 0x7F_FFFF) != 0


def _is_inf(u: int) -> bool:
    return _exp(u) == 0xFF and (u & 0x7F_FFFF) == 0


def _quiet(u: int) -> int:
    return u | 0x40_0000


def _unpack(u: int) -> tuple[bool, int, int]:
    """``(negative, magnitude, exponent)`` of a finite word: value ``(-1)^negative * magnitude * 2^exponent``."""

    negative = (u >> 31) & 1 == 1
    e, m = _exp(u), u & 0x7F_FFFF
    return (negative, m, -149) if e == 0 else (negative, m | 0x80_0000, e - 150)


def _round_pack(negative: bool, magnitude: int, exponent: int) -> int:
    """RN-even of the exact nonzero ``(-1)^negative * magnitude * 2^exponent`` to FP32 (overflow to infinity)."""

    sign = int(negative) << 31
    bits = magnitude.bit_length()
    result_exp = exponent + bits - 1
    shift = bits - 24
    if result_exp < -126:
        shift += -126 - result_exp
        result_exp = -126
    if shift <= 0:
        kept, rnd, sticky = magnitude << -shift, False, False
    else:
        kept = magnitude >> shift
        rnd = (magnitude >> (shift - 1)) & 1 == 1
        sticky = magnitude & ((1 << (shift - 1)) - 1) != 0
    if rnd and (sticky or kept & 1):
        kept += 1
    if kept >= 1 << 24:
        kept >>= 1
        result_exp += 1
    if result_exp > 127:
        return sign | 0x7F80_0000
    if kept < 1 << 23:
        return sign | kept
    return sign | ((result_exp + 127) << 23) | (kept & 0x7F_FFFF)


def _signed_sum(nh: bool, hi: int, nl: bool, lo: int) -> tuple[bool, int]:
    if nh == nl:
        return nh, hi + lo
    return (nh, hi - lo) if hi >= lo else (nl, lo - hi)


def f32_add(a: int, b: int) -> int:
    if _is_nan(a):
        return _quiet(a)
    if _is_nan(b):
        return _quiet(b)
    if _is_inf(a) and _is_inf(b):
        return a if (a >> 31) == (b >> 31) else F32_INDEFINITE
    if _is_inf(a):
        return a
    if _is_inf(b):
        return b
    na, ma, ea = _unpack(a)
    nb, mb, eb = _unpack(b)
    if ma == 0 and mb == 0:
        return 0x8000_0000 if na and nb else 0
    (mh, eh, nh), (ml, el, nl) = ((ma, ea, na), (mb, eb, nb)) if ea >= eb else ((mb, eb, nb), (ma, ea, na))
    # a gap beyond the grid only matters as sticky (the Rust caps it at 64 bits for its u128)
    gap = eh - el
    hi, lo, e = (mh << gap, ml, el) if gap <= 64 else (mh << 31, int(ml != 0), eh - 31)
    negative, magnitude = _signed_sum(nh, hi, nl, lo)
    return 0 if magnitude == 0 else _round_pack(negative, magnitude, e)


def f32_mul(a: int, b: int) -> int:
    if _is_nan(a):
        return _quiet(a)
    if _is_nan(b):
        return _quiet(b)
    sign = ((a ^ b) >> 31) & 1
    _, ma, ea = _unpack(0 if _is_inf(a) else a)
    _, mb, eb = _unpack(0 if _is_inf(b) else b)
    if _is_inf(a) and _is_inf(b):
        return (sign << 31) | 0x7F80_0000
    if _is_inf(a):
        return F32_INDEFINITE if mb == 0 else (sign << 31) | 0x7F80_0000
    if _is_inf(b):
        return F32_INDEFINITE if ma == 0 else (sign << 31) | 0x7F80_0000
    if ma == 0 or mb == 0:
        return sign << 31
    return _round_pack(sign == 1, ma * mb, ea + eb)


def f32_div(a: int, b: int) -> int:
    if _is_nan(a):
        return _quiet(a)
    if _is_nan(b):
        return _quiet(b)
    sign = ((a ^ b) >> 31) & 1
    if _is_inf(a) and _is_inf(b):
        return F32_INDEFINITE
    if _is_inf(a):
        return (sign << 31) | 0x7F80_0000
    if _is_inf(b):
        return sign << 31
    _, ma, ea = _unpack(a)
    _, mb, eb = _unpack(b)
    if mb == 0:
        return F32_INDEFINITE if ma == 0 else (sign << 31) | 0x7F80_0000
    if ma == 0:
        return sign << 31
    numerator = ma << 64
    quotient, remainder = divmod(numerator, mb)
    return _round_pack(sign == 1, (quotient << 1) | int(remainder != 0), ea - eb - 65)


def f32_fma(a: int, b: int, c: int) -> int:
    if _is_nan(a) or _is_nan(b) or _is_nan(c):
        return F32_NAN
    product_sign = ((a ^ b) >> 31) & 1 == 1
    if _is_inf(a) or _is_inf(b):
        finite_zero = (not _is_inf(a) and _unpack(a)[1] == 0) or (not _is_inf(b) and _unpack(b)[1] == 0)
        if finite_zero or (_is_inf(c) and ((c >> 31) & 1 == 1) != product_sign):
            return F32_NAN
        return 0xFF80_0000 if product_sign else 0x7F80_0000
    if _is_inf(c):
        return c
    _, ma, ea = _unpack(a)
    _, mb, eb = _unpack(b)
    nc, mc, ec = _unpack(c)
    mp, ep = ma * mb, ea + eb
    if mp == 0 and mc == 0:
        return 0x8000_0000 if product_sign and nc else 0
    if mp == 0:
        return _round_pack(nc, mc, ec)
    if mc == 0:
        return _round_pack(product_sign, mp, ep)
    (mh, eh, nh), (ml, el, nl) = ((mp, ep, product_sign), (mc, ec, nc)) if ep >= ec else ((mc, ec, nc), (mp, ep, product_sign))
    gap = eh - el
    hi, lo, e = (mh << gap, ml, el) if gap <= 72 else (mh << 55, int(ml != 0), eh - 55)
    negative, magnitude = _signed_sum(nh, hi, nl, lo)
    return 0 if magnitude == 0 else _round_pack(negative, magnitude, e)


def bf16_to_f32(b: int) -> int:
    return (b & 0xFFFF) << 16


def _f32_to_bf16_rn(u: int, nan: int) -> int:
    u &= 0xFFFF_FFFF
    if _is_nan(u):
        return nan
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def f2fp_bf16(u: int) -> int:
    return _f32_to_bf16_rn(u, BF16_NAN_F2FP)


def bf16_add_f2fp(a: int, b: int) -> int:
    return _f32_to_bf16_rn(f32_add(bf16_to_f32(a), bf16_to_f32(b)), BF16_NAN_F2FP)


def rope_out(x: int, y: int, c: int, s: int) -> int:
    t = f32_mul(bf16_to_f32(y), bf16_to_f32(s))
    return _f32_to_bf16_rn(f32_fma(bf16_to_f32(x), bf16_to_f32(c), t ^ 0x8000_0000), BF16_NAN_TORCH)


def rope_out_add(x: int, y: int, c: int, s: int) -> int:
    t = f32_mul(bf16_to_f32(x), bf16_to_f32(s))
    return _f32_to_bf16_rn(f32_fma(bf16_to_f32(y), bf16_to_f32(c), t), BF16_NAN_TORCH)


def bf16_gt(a: int, b: int) -> int:
    fa, fb = bf16_to_f32(a), bf16_to_f32(b)
    if _is_nan(fa) or _is_nan(fb):
        return 0

    def key(u: int) -> int:
        if _is_inf(u):
            mag = 1
        else:
            mag = _unpack(u)[1]
        if mag == 0:
            return 0
        return -(u & 0x7FFF_FFFF) if (u >> 31) & 1 else u & 0x7FFF_FFFF

    return int(key(fa) > key(fb))


def _table_word(t: int, expected_index: int) -> int:
    if t >> 32 != expected_index:
        raise ValueError(f"table row {t >> 32} opened where the operand reads row {expected_index}")
    return t & 0xFFFF_FFFF


def _abs_gt(x: int, threshold: int) -> bool:
    return not _is_nan(x) and (x & 0x7FFF_FFFF) > threshold


def _abs_ge_or_nan(x: int, threshold: int) -> bool:
    return _is_nan(x) or (x & 0x7FFF_FFFF) >= threshold


def mufu_rcp_index(x: int) -> int:
    """The row MUFU.RCP reads for ``x`` (its mantissa; ``2^23`` rows)."""

    return x & 0x7F_FFFF


def _mufu_rcp_word(x: int, t: int) -> int:
    word = _table_word(t, mufu_rcp_index(x))
    s, ax = x & 0x8000_0000, x & 0x7FFF_FFFF
    e = (ax >> 23) & 0xFF
    if e == 0xFF:
        return 0x7FFF_FFFF if ax & 0x7F_FFFF else s
    if e == 0:
        return s | 0x7F80_0000
    re = ((word >> 23) & 0xFF) - (e - 127)
    if re <= 0:
        return s
    if re >= 0xFF:
        return s | 0x7F80_0000
    return s | (re << 23) | (word & 0x7F_FFFF)


def mufu_sqrt_index(x: int) -> int:
    """The row MUFU.SQRT / MUFU.RSQ read for ``x``: exponent parity and mantissa (``2^24`` rows)."""

    ax = x & 0x7FFF_FFFF
    parity = (((ax >> 23) & 0xFF) - 127) & 1
    return (parity << 23) | (ax & 0x7F_FFFF)


def _sqrt_like(x: int, t: int, *, rsq: bool) -> int:
    word = _table_word(t, mufu_sqrt_index(x))
    s, ax = x & 0x8000_0000, x & 0x7FFF_FFFF
    e = (ax >> 23) & 0xFF
    if e == 0xFF:
        if ax & 0x7F_FFFF:
            return 0x7FFF_FFFF
        return 0x7FFF_FFFF if s else (0 if rsq else 0x7F80_0000)
    if e == 0:
        return s | 0x7F80_0000 if rsq else s
    if s:
        return 0x7FFF_FFFF
    k = e - 127
    half = (k - (k & 1)) // 2
    re = ((word >> 23) & 0xFF) + (-half if rsq else half)
    return (((re & 0xFFFF_FFFF) << 23) & 0xFFFF_FFFF) | (word & 0x7F_FFFF)


def mufu_sqrt_tab(x: int, t: int) -> int:
    return _sqrt_like(x, t, rsq=False)


def _div_full_scale_b(b: int) -> int:
    if _abs_gt(b, _TWO_POW_126):
        return f32_mul(b, _QUARTER)
    if _abs_ge_or_nan(b, _MIN_NORMAL):
        return b
    return f32_mul(b, _TWO_POW_24)


def div_full_rcp_index(b: int) -> int:
    return mufu_rcp_index(_div_full_scale_b(b))


def div_full_rcp(b: int, t: int) -> int:
    return _mufu_rcp_word(_div_full_scale_b(b), t)


def div_full_scale_a(a: int, b: int) -> int:
    if _abs_gt(b, _TWO_POW_126):
        return f32_mul(a, _QUARTER)
    if _abs_ge_or_nan(b, _MIN_NORMAL):
        return a
    return f32_mul(a, _TWO_POW_24)


def rsqrt_approx_index(x: int) -> int:
    """The RSQ row ``rsqrt_approx(x, .)`` reads (``|x| < 2^-126`` is scaled by ``2^24`` first)."""

    return mufu_sqrt_index(x if _abs_ge_or_nan(x, _MIN_NORMAL) else f32_mul(x, _TWO_POW_24))


def rsqrt_approx(x: int, t: int) -> int:
    if _abs_ge_or_nan(x, _MIN_NORMAL):
        return _sqrt_like(x, t, rsq=True)
    return f32_mul(_sqrt_like(f32_mul(x, _TWO_POW_24), t, rsq=True), _TWO_POW_12)


def _gate(name: str, fn, width: int, arg_widths: tuple[int, ...]) -> Gate:
    return Gate(name, len(arg_widths), width, replay_cost=1, proof_cost=1, evaluate=lambda args: fn(*args),
                arg_widths=arg_widths)


def make_reduce_mufu_gate_set() -> GateSet:
    """The gate set ``veritor.reduce-mufu@1`` (statement width 32; sources 64-bit)."""

    select = lambda c, a, b: a if c & 1 else b
    return GateSet(
        (
            _gate("bf16_add_f2fp", bf16_add_f2fp, 16, (16, 16)),
            _gate("bf16_gt", bf16_gt, 1, (16, 16)),
            _gate("bf16_to_f32", bf16_to_f32, 32, (16,)),
            _gate("div_full_rcp", div_full_rcp, 32, (32, 64)),
            _gate("div_full_scale_a", div_full_scale_a, 32, (32, 32)),
            _gate("f2fp_bf16", f2fp_bf16, 16, (32,)),
            _gate("f32_add", f32_add, 32, (32, 32)),
            _gate("f32_div", f32_div, 32, (32, 32)),
            _gate("f32_fma", f32_fma, 32, (32, 32, 32)),
            _gate("f32_mul", f32_mul, 32, (32, 32)),
            _gate("i32_add", lambda a, b: (a + b) & 0xFFFF_FFFF, 32, (32, 32)),
            _gate("i32_eq", lambda a, b: int(a == b), 1, (32, 32)),
            _gate("i32_le", lambda a, b: int((a ^ 0x8000_0000) <= (b ^ 0x8000_0000)), 1, (32, 32)),
            Gate("in", 0, 64, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            _gate("mufu_sqrt_tab", mufu_sqrt_tab, 32, (32, 64)),
            _gate("rope_out", rope_out, 16, (16, 16, 16, 16)),
            _gate("rope_out_add", rope_out_add, 16, (16, 16, 16, 16)),
            _gate("rsqrt_approx", rsqrt_approx, 32, (32, 64)),
            _gate("select_bf16", select, 16, (1, 16, 16)),
            _gate("select_i32", select, 32, (1, 32, 32)),
            Gate("weight", 0, 64, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=REDUCE_MUFU_NAME,
        version=REDUCE_MUFU_VERSION,
    )
