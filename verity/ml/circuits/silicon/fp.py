"""Exact f32 / bf16 arithmetic as GF(2) circuits over `forms` (formerly C-Flock's `fp`, its lowering's per-op pieces).

A word is a list of :class:`forms.L` linear forms, LSB first. XOR, NOT, constant shifts and bit selection are free; an AND
of two non-constant forms is one committed bit (``forms.Circuit.AND``). Constants propagate, so each piece is written once
for general f32 operands and specializes itself when called on bf16-widened words (16 constant-zero mantissa bits) or on
program constants.

Every piece that produces an f32 ends in :func:`round_f32`, the one rounding routine (round to nearest even, subnormals
kept, overflow to infinity). A NaN result is the canonical 0x7FC00000 inside composite pieces, whose final bf16 casts
canonicalize; ``f32_add`` / ``f32_mul`` with ``ir_nan`` return the IR primitives' NaN bit for bit (``F32Add_v1`` /
``F32Mul_v1`` are numpy float32 on x86-64: an operand's NaN, quieted, else the default NaN of an invalid operation), and
``f32_fma_bits``' NaN is 0x7FC00000. The ``_v2`` primitives' pieces pass ``nan_word=F32_NAN_PTX``: every NaN result the GPU's
0x7FFFFFFF, no operand's payload or sign kept.
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.ml.circuits.boolean.forms import ONE, ZERO, L, NOT, OR, add, and_reduce, const_bits, lzc, multiply, mux, or_reduce, sub

F32_QNAN = 0x7FC00000
#: every NaN result of the ``_v2`` FP32 primitives (`verity.ml.circuits.ops.scalar.F32_NAN_PTX`)
F32_NAN_PTX = 0x7FFFFFFF
F32_INF = 0x7F800000


def ceil_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def is_zero(C, bits):
    return NOT(or_reduce(C, bits))


def select(C, s, x, y):
    """``s ? x : y`` for equal-length words (one AND per bit that differs)."""
    return mux(C, s, x, y)


def _split_amount(C, x, amt):
    """The amount bits that shift within x, and one flag for 'shifts everything out' (any higher bit set)."""
    k = max(1, (len(x) - 1).bit_length())
    return list(amt[:k]), or_reduce(C, amt[k:])


def trim_low(bits):
    """``(bits without their constant-zero LSBs, how many were dropped)``: the value is ``trimmed << dropped``."""
    t = 0
    while t < len(bits) - 1 and bits[t].is_const() and bits[t].c == 0:
        t += 1
    return list(bits[t:]), t


def shl(C, x, amt):
    """``x << amt`` at the width of x (bits shifted past the top are dropped); ``amt`` unsigned, LSB first."""
    cur = list(x)
    n = len(cur)
    low, big = _split_amount(C, x, amt)
    for k, s in enumerate(low):
        sh = 1 << k
        cur = [(cur[j] ^ C.AND(s, cur[j] ^ cur[j - sh])) if j >= sh else C.AND(NOT(s), cur[j]) for j in range(n)]
    return [C.AND(NOT(big), w) for w in cur]


def shr_sticky(C, x, amt):
    """``(x >> amt, OR of the bits shifted out)`` at the width of x, for any unsigned ``amt`` (LSB first)."""
    cur = list(x)
    n = len(cur)
    sticky = ZERO
    low, big = _split_amount(C, x, amt)
    for k, s in enumerate(low):
        sh = 1 << k
        sticky = OR(C, sticky, C.AND(s, or_reduce(C, cur[:sh])))
        cur = [(cur[j] ^ C.AND(s, cur[j] ^ cur[j + sh])) if j + sh < n else C.AND(NOT(s), cur[j]) for j in range(n)]
    sticky = OR(C, sticky, C.AND(big, or_reduce(C, cur)))
    return [C.AND(NOT(big), w) for w in cur], sticky


def neg_if(C, x, s):
    """Two's complement ``-x`` when s, else x (width of x)."""
    return add(C, [b ^ s for b in x], [s], width=len(x))


def min_u(C, a, b, width):
    """``min(a, b)`` of two unsigned words at ``width``."""
    a = list(a) + [ZERO] * (width - len(a))
    b = list(b) + [ZERO] * (width - len(b))
    _, a_lt_b = sub(C, a, b, width)
    return mux(C, a_lt_b, a, b)


# --- f32 fields --------------------------------------------------------------------------------------------------------

@dataclass
class F32:
    s: object
    e: list          # exponent field, 8 bits
    m: list          # mantissa field, 23 bits
    e_nz: object     # exponent field nonzero (normal, inf or NaN)
    e_max: object    # exponent field all ones (inf or NaN)
    m_nz: object
    nan: object
    inf: object
    zero: object
    sig: list        # 24-bit significand, hidden bit on top
    e_eff: list      # the exponent field or 1 for zero / subnormal (8 bits): value = sig * 2^(e_eff - 150)


def unpack(C, w) -> F32:
    assert len(w) == 32
    m, e, s = list(w[:23]), list(w[23:31]), w[31]
    e_nz = or_reduce(C, e)
    e_max = and_reduce(C, e)
    m_nz = or_reduce(C, m)
    nan = C.AND(e_max, m_nz)
    inf = e_max ^ nan
    zero = NOT(OR(C, e_nz, m_nz))
    e_eff = [e[0] ^ NOT(e_nz)] + e[1:]     # e_nz = 0 forces e[0] = 0, so the OR is an XOR
    return F32(s, e, m, e_nz, e_max, m_nz, nan, inf, zero, m + [e_nz], e_eff)


def f32_const(value: int):
    return const_bits(value & 0xFFFFFFFF, 32)


def round_f32(C, sign, E_b, off: int, M, zero_sign, underflow: bool = True, rz: bool = False, rm: bool = False):
    """The f32 word nearest (ties to even) to ``(-1)^sign * M * 2^(E - 127 - (len(M) - 1))`` with ``E = E_b - off``: ``E`` is the
    exponent field the value would have if M's top bit were the hidden bit. ``E_b`` is unsigned (``len(E_b)`` bits wide
    enough for ``E_b + 1``). M = 0 gives the zero of sign ``zero_sign``. Overflow gives infinity. ``underflow=False`` promises
    ``E >= 1`` (no right shift into the subnormal range is ever needed), which drops that shifter. ``rz``: round toward zero
    instead (overflow then saturates at the largest finite word); ``rm``: toward -infinity (a positive overflow saturates, a
    negative one is -infinity)."""
    n = len(M)
    P = ceil_pow2(max(n, 32))
    Mp = [ZERO] * (P - n) + list(M)
    az, lz = lzc(C, Mp)
    EW = len(E_b)
    lzb = len(lz)
    diff, borrow = sub(C, E_b, const_bits((off + 1) % (1 << EW), EW), EW)   # E - 1, borrow iff E < 1
    if underflow:
        sh_l = min_u(C, diff, lz, EW)[:lzb]
        sh_l = [C.AND(NOT(borrow), b) for b in sh_l]
        y = shl(C, Mp, sh_l)
        sh_r = [C.AND(borrow, b) for b in neg_if(C, diff, ONE)]           # 1 - E when E < 1
        y, st = shr_sticky(C, y, sh_r)
        X = [C.AND(NOT(borrow), b) for b in sub(C, diff, sh_l, EW)[0]]     # field - hidden bit
    else:
        sh_l = min_u(C, diff, lz, EW)[:lzb]
        y = shl(C, Mp, sh_l)
        st = ZERO
        X = sub(C, diff, sh_l, EW)[0]
    sig = y[P - 24:]
    guard = y[P - 25]
    sticky = OR(C, or_reduce(C, y[:P - 25]), st)
    rnd = ZERO if rz else C.AND(sign, OR(C, guard, sticky)) if rm else C.AND(guard, OR(C, sticky, sig[0]))
    ex = add(C, X, [sig[23]], width=EW)                                    # the exponent field before rounding
    _, lt255 = sub(C, ex, const_bits(255, EW), EW)
    ovf = NOT(lt255)
    word = add(C, sig[:23] + ex[:8], [rnd], width=31)
    if rm:
        inf, sat = C.AND(ovf, sign), C.AND(ovf, NOT(sign))
        word = [C.AND(NOT(inf), b) for b in word[:23]] + [OR(C, inf, b) for b in word[23:31]]
        word = [OR(C, sat, b) for b in word[:23]] + [C.AND(NOT(sat), word[23])] + [OR(C, sat, b) for b in word[24:31]]
    elif rz:
        word = [OR(C, ovf, b) for b in word[:23]] + [C.AND(NOT(ovf), word[23])] + [OR(C, ovf, b) for b in word[24:31]]
    else:
        word = [C.AND(NOT(ovf), b) for b in word[:23]] + [OR(C, ovf, b) for b in word[23:31]]
    word = [C.AND(NOT(az), b) for b in word]
    return word + [mux(C, az, [zero_sign], [sign])[0]]


def specials(C, word, nan, inf, inf_sign, nan_word: int = F32_QNAN):
    """Override ``word`` with ``nan_word``, or with the infinity of sign ``inf_sign``."""
    word = mux(C, inf, const_bits(F32_INF, 31) + [inf_sign], word)
    return mux(C, nan, f32_const(nan_word), word)


def ir_specials(C, word, a, b, A, B, invalid, inf, inf_sign):
    """numpy float32 on x86-64's special results (the IR's ``F32Add_v1`` / ``F32Mul_v1``): b quieted when b is a NaN, else a
    quieted when a is, else the default NaN 0xFFC00000 of an invalid operation, else the infinity of sign ``inf_sign`` when
    an operand is infinite. An invalid operation always has an infinite operand, so it rides the infinity's select."""
    word = mux(C, inf, [ZERO] * 22 + [invalid] + [ONE] * 8 + [OR(C, invalid, inf_sign)], word)
    src = mux(C, B.nan, b[:22] + b[23:], a[:22] + a[23:])
    return mux(C, OR(C, A.nan, B.nan), src[:22] + [ONE] + src[22:], word)


# --- f32 arithmetic ----------------------------------------------------------------------------------------------------

def f32_mul(C, a, b, ir_nan: bool = False, nan_word: int = F32_QNAN):
    """``RN(a * b)`` (FMUL, no ftz); a NaN result canonical, or with ``ir_nan`` the IR primitive's (:func:`ir_specials`)."""
    A, B = unpack(C, a), unpack(C, b)
    s = A.s ^ B.s
    P = multiply(C, trim_low(A.sig)[0], trim_low(B.sig)[0])   # E does not depend on trimming zero LSBs
    E_b = add(C, A.e_eff, B.e_eff, width=10)                  # E = ea + eb - 126
    r = round_f32(C, s, E_b, 126, P, s)
    if ir_nan:
        return ir_specials(C, r, a, b, A, B, OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)), OR(C, A.inf, B.inf), s)
    nan = OR(C, OR(C, A.nan, B.nan), OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)))
    return specials(C, r, nan, OR(C, A.inf, B.inf), s, nan_word)


def f32_add(C, a, b, ir_nan: bool = False, rz: bool = False, nan_word: int = F32_QNAN):
    """``RN(a + b)`` (FADD, no ftz); IEEE zero signs: ``x + (-x) = +0``, ``(-0) + (-0) = -0``; a NaN result canonical, or with
    ``ir_nan`` the IR primitive's (:func:`ir_specials`)."""
    A, B = unpack(C, a), unpack(C, b)
    _, a_lt_b = sub(C, a[:31], b[:31], 31)
    sx, sy = mux(C, a_lt_b, [B.s, A.s], [A.s, B.s])
    ex = mux(C, a_lt_b, B.e_eff, A.e_eff)
    ey = mux(C, a_lt_b, A.e_eff, B.e_eff)
    gx = mux(C, a_lt_b, B.sig, A.sig)
    gy = mux(C, a_lt_b, A.sig, B.sig)
    d, _ = sub(C, ex, ey, 8)
    ys, st = shr_sticky(C, [ZERO] * 3 + gy, d)
    ys[0] = OR(C, ys[0], st)
    xs = [ZERO] * 3 + gx
    sb = sx ^ sy
    S = add(C, xs + [ZERO], [v ^ sb for v in ys] + [sb], cin=sb, width=28)
    r = round_f32(C, sx, ex + [ZERO], -1, S, C.AND(A.s, B.s), underflow=False, rz=rz)
    if ir_nan:
        invalid = C.AND(C.AND(A.inf, B.inf), A.s ^ B.s)
        return ir_specials(C, r, a, b, A, B, invalid, OR(C, A.inf, B.inf), mux(C, A.inf, [A.s], [B.s])[0])
    nan = OR(C, OR(C, A.nan, B.nan), C.AND(C.AND(A.inf, B.inf), A.s ^ B.s))
    inf = OR(C, A.inf, B.inf)
    return specials(C, r, nan, inf, mux(C, A.inf, [A.s], [B.s])[0], nan_word)


def f32_neg(a):
    return list(a[:31]) + [NOT(a[31])]


def f32_sub(C, a, b, nan_word: int = F32_QNAN):
    return f32_add(C, a, f32_neg(b), nan_word=nan_word)


def _normalize_sig(C, F: F32):
    """``(sig << lz, lz, trimmed zero LSBs)`` for a nonzero significand: its constant-zero LSBs trimmed, the rest shifted up
    to its top bit (padded to a power of two for the leading-zero count)."""
    sig, t = trim_low(F.sig)
    w = len(sig)
    pw = ceil_pow2(w)
    padded = [ZERO] * (pw - w) + sig
    _, lz = lzc(C, padded)
    return shl(C, padded, lz)[pw - w:], lz, t


def f32_fma(C, a, b, c, rm: bool = False, nan_word: int = F32_QNAN):
    """``RN(a * b + c)`` with one rounding (FFMA, no ftz), as ``verity_vllm``'s ``f32_fma_bits``; ``rm``: rounded toward -infinity
    (``fma.rm.f32``; an exact zero sum keeps the RN signs, as ``F32FmaRm_v1`` defines it).

    Operands a, b are pre-normalized, so a nonzero product P of width pw is in [2^(pw-2), 2^pw); it sits at bits
    [50 - pw, 50) of a 76-bit window (for 24-bit significands, [2, 50)). The addend's 24-bit significand starts at [52, 76)
    and shifts right; its bits below the window's low end lo fold into a sticky at lo, where P's bit is zero and which lies
    below every possible guard bit (the result's MSB is at least 47 whenever addend bits are dropped), so the fold is exact.
    When the addend's LSB would sit above bit 52 it stays at 52 and the product is moved up to it: with two zero bits
    between, the product then only affects rounding as a sticky, identically at either position."""
    A, B, Cc = unpack(C, a), unpack(C, b), unpack(C, c)
    sp = A.s ^ B.s
    ga, lza, ta = _normalize_sig(C, A)
    gb, lzb, tb = _normalize_sig(C, B)
    P = multiply(C, ga, gb)
    pw = len(P)
    lo = max(0, min(50 - pw, 23) - 1)
    # T = ea_n + eb_n + 64 = ea + eb + 64 - lza - lzb   (ea_n + eb_n in [-44, 508])
    T = add(C, add(C, A.e_eff, B.e_eff, width=10), const_bits(64, 10), width=10)
    T, _ = sub(C, T, add(C, lza, lzb, width=6), 10)
    # sh_c = T - ec - 164; negative -> clamp (the addend's LSB stays at 52)
    U, clamp = sub(C, T, add(C, Cc.e_eff, const_bits(164, 10), width=10), 10)
    clamp = C.AND(clamp, NOT(Cc.zero))
    sh_c = [C.AND(NOT(clamp), u) for u in U]
    Pw = ([ZERO] * (50 - pw) + P + [ZERO] * 26)[lo:]
    Cs, st = shr_sticky(C, ([ZERO] * 52 + Cc.sig)[lo:], sh_c)
    Cs[0] = OR(C, Cs[0], st)
    sub_ = sp ^ Cc.s
    n = 76 - lo
    D = add(C, Pw + [ZERO] * 2, [v ^ sub_ for v in Cs] + [sub_] * 2, cin=sub_, width=n + 2)
    neg = D[n + 1]
    M = neg_if(C, D[:n + 1], neg)
    # E = W0 + 203 (M's top bit is window bit 76): no clamp W0 = ea_n + eb_n - 302 -> E_b = T - 3 (off 160);
    # clamp W0 = ec - 202 -> E_b = ec + 161
    E_noclamp, _ = sub(C, T, const_bits(3, 10), 10)
    E_clamp = add(C, Cc.e_eff, const_bits(161, 10), width=10)
    E_b = mux(C, clamp, E_clamp, E_noclamp)
    r = round_f32(C, sp ^ neg, E_b, 160, M, ZERO, rm=rm)
    pz = OR(C, A.zero, B.zero)
    zero_c = f32_const(0)[:31] + [C.AND(sp, Cc.s)]
    r = mux(C, pz, mux(C, Cc.zero, zero_c, list(c)), r)
    prod_inf = OR(C, A.inf, B.inf)
    nan = OR(C, OR(C, OR(C, A.nan, B.nan), Cc.nan),
             OR(C, OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)), C.AND(C.AND(prod_inf, Cc.inf), sp ^ Cc.s)))
    inf = OR(C, prod_inf, Cc.inf)
    return specials(C, r, nan, inf, mux(C, prod_inf, [sp], [Cc.s])[0], nan_word)


def f32_div(C, a, b, ir_nan: bool = False, nan_word: int = F32_QNAN):
    """``RN(a / b)`` (``div.rn.f32``, no ftz): the normalized significands' quotient to 27 bits by restoring division, the
    remainder as sticky, one :func:`round_f32`. Specials: x / 0 and inf / finite are infinities, finite / inf and 0 / x zeros,
    0 / 0 and inf / inf invalid; a NaN result canonical, or with ``ir_nan`` the IR's (numpy float32 on x86-64,
    :func:`ir_specials`)."""
    A, B = unpack(C, a), unpack(C, b)
    s = A.s ^ B.s
    ga, lza, _ = _normalize_sig(C, A)
    gb, lzb, _ = _normalize_sig(C, B)
    ga, gb = [ZERO] * (24 - len(ga)) + ga, [ZERO] * (24 - len(gb)) + gb
    r, q = list(ga) + [ZERO], []
    for _ in range(27):
        d, borrow = sub(C, r, gb + [ZERO], 25)
        q.append(NOT(borrow))
        r = mux(C, borrow, r, d)
        r = [ZERO] + r[:24]
    sticky = or_reduce(C, r)
    M = [sticky] + q[::-1]
    # a / b = M * 2^(Ea - Eb - 27) with Ea = e_eff - lz: round_f32's E = Ea - Eb + 127, carried as E + 256
    E_b = add(C, add(C, A.e_eff, const_bits(383, 10), width=10), lzb, width=10)
    E_b, _ = sub(C, E_b, add(C, B.e_eff, lza, width=10), 10)
    word = round_f32(C, s, E_b, 256, M, s)
    zero = OR(C, C.AND(B.inf, NOT(A.inf)), C.AND(A.zero, NOT(B.zero)))
    word = mux(C, zero, [ZERO] * 31 + [s], word)
    invalid = OR(C, C.AND(A.zero, B.zero), C.AND(A.inf, B.inf))
    inf = OR(C, invalid, OR(C, C.AND(A.inf, NOT(B.inf)), C.AND(B.zero, NOT(A.zero))))
    if not ir_nan:
        return specials(C, word, OR(C, OR(C, A.nan, B.nan), invalid), inf, s, nan_word)
    # divss: the dividend's NaN when it is one, else the divisor's, quieted; the default NaN 0xFFC00000 when invalid
    word = mux(C, inf, [ZERO] * 22 + [invalid] + [ONE] * 8 + [OR(C, invalid, s)], word)
    src = mux(C, A.nan, a[:22] + a[23:], b[:22] + b[23:])
    return mux(C, OR(C, A.nan, B.nan), src[:22] + [ONE] + src[22:], word)


def bf16_to_f32(h):
    assert len(h) == 16
    return [ZERO] * 16 + list(h)


def f32_to_bf16(C, u, nan_word: int):
    """``bf16_rn(u)``: ``(u + 0x7FFF + lsb) >> 16`` on the encoding (``f32_to_bf16_rn_bits``), NaN -> ``nan_word``."""
    low, lsb = u[:16], u[16]
    carry = C.AND(low[15], OR(C, or_reduce(C, low[:15]), lsb))
    r = add(C, u[16:32], [carry], width=16)
    nan = C.AND(and_reduce(C, u[23:31]), or_reduce(C, u[:23]))
    return mux(C, nan, const_bits(nan_word, 16), r)


# --- lookups -------------------------------------------------------------------------------------------------------------

# --- the tensor-core k-step ------------------------------------------------------------------------------------------------

TC_NAN = 0x7FFFFFFF


def _classes(C, w, man: int, exp: int):
    """(nan, inf, zero) of a float word with ``man`` mantissa and ``exp`` exponent bits."""
    e, m = w[man:man + exp], w[:man]
    e_all, e_any, m_any = and_reduce(C, e), or_reduce(C, e), or_reduce(C, m)
    return C.AND(e_all, m_any), C.AND(e_all, NOT(m_any)), NOT(OR(C, e_any, m_any))


def _finite_or_zero(C, w, man: int, exp: int):
    """The word, its exponent and mantissa bits cleared when it is NaN or infinite (a finite stand-in)."""
    fin = NOT(and_reduce(C, w[man:man + exp]))
    return [C.AND(fin, x) for x in w[:man + exp]] + w[man + exp:]


def tc_dot16(C, acc, a, b, groups=(8, 8), W: int = 25, F: int = -132):
    """``AmpereBF16TcDot16``: ``acc + sum_k a[k] * b[k]`` as ``mma.sync.m16n8k16`` bf16 computes one accumulator coordinate, on
    every encoding (``verity.ml.circuits.silicon.total.tc_dot_total`` on ``AMPERE_BF16_M16N8K16``). The finite part is
    ``tc_step.unit`` (the binary census's circuit of ``GroupSum.step``: groups (8, 8), 25-bit truncating adder, floor -132) on
    operands whose non-finite words are replaced by zero; per group, a NaN term, or infinities of both signs (an overflowed
    group counts as its infinity), give the canonical 0x7FFFFFFF, and one-signed infinities that infinity. Any non-finite
    operand makes the result non-finite, so the stand-ins never reach the output."""
    from . import tc_step as U

    return _tc_total(C, acc, a, b, U.BF16, groups, W, F)


def tc_dot_e4m3(C, acc, a, b, groups=(32,), W: int = 14, F: int = -139):
    """``HopperE4m3QgmmaDot32``: one accumulator coordinate of ``wgmma.m64n{N}k32.f32.e4m3.e4m3``, total on every encoding
    (``verity.ml.circuits.silicon.total_fp8.tc_dot_total_e4m3`` on ``HOPPER_E4M3_WGMMA_K32``: one group of 32 exact E4M3 products and the
    accumulator through a 14-bit truncating adder, floor -139). E4M3 has no infinities: a byte is NaN when its exponent and
    mantissa are all ones, so a product is NaN exactly when an operand is."""
    from . import tc_step as U

    return _tc_total(C, acc, a, b, U.E4M3, groups, W, F)


def tc_dot_e5m2(C, acc, a, b, groups=(32,), W: int = 26, F: int = -133):
    """``BlackwellE5m2QmmaDot32``: one accumulator coordinate of ``mma.sync.m16n8k32.f32.e5m2.e5m2`` on sm_120, total on every
    encoding (``verity.ml.circuits.silicon.total_fp8.tc_dot_total_e5m2`` on ``BLACKWELL_SM120_E5M2_M16N8K32``: one group of 32 exact E5M2
    products and the accumulator through the 26-bit adder, floor -133). E5M2 has IEEE infinities and NaNs, so the special
    rule is the BF16 step's."""
    from . import tc_step as U

    return _tc_total(C, acc, a, b, U.E5M2, groups, W, F)


def tc_dot_tf32(C, acc, a, b, groups, W: int, F: int):
    """One accumulator coordinate of ``wgmma.mma_async.sync.aligned.m64n{N}k8.f32.tf32.tf32`` at the caller's ``groups``, adder
    width ``W`` and floor ``F``: TF32 operands (19-bit words, sign, 8 exponent and 10 mantissa bits) with exact products,
    ``group_sum``'s truncating adder and the BF16 step's special rule.  `instructions.py` lists the instruction unprobed, so no
    parameters are this function's own."""
    from . import tc_step as U

    return _tc_total(C, acc, a, b, U.TF32, groups, W, F)


def _tc_total(C, acc, a, b, fmt, groups, W, F):
    from . import tc_step as U

    k, n = sum(groups), fmt.bits
    assert len(a) == len(b) == k
    rescale = W - 24
    if fmt.nan_rule == "ieee":
        def cls(x):
            return _classes(C, x, fmt.man_bits, fmt.exp_bits)

        def stand_in(x):
            return _finite_or_zero(C, x, fmt.man_bits, fmt.exp_bits)
    else:
        def cls(x):
            return and_reduce(C, x[:n - 1]), ZERO, NOT(or_reduce(C, x[:n - 1]))

        def stand_in(x):
            ok = NOT(and_reduce(C, x[:n - 1]))
            return [C.AND(ok, y) for y in x[:n - 1]] + [x[n - 1]]
    pc = [(cls(x), cls(y), x[n - 1] ^ y[n - 1]) for x, y in zip(a, b)]
    p_nan = [OR(C, OR(C, ca[0], cb[0]), OR(C, C.AND(ca[1], cb[2]), C.AND(cb[1], ca[2]))) for ca, cb, _ in pc]
    p_inf = [C.AND(OR(C, ca[1], cb[1]), NOT(n)) for (ca, cb, _), n in zip(pc, p_nan)]
    c_nan, c_inf, _ = _classes(C, acc, 23, 8)
    prods = []
    for x, y in zip(a, b):
        sa, ea, siga, nza = _tc_operand(C, fmt, stand_in(x))
        sb, eb, sigb, nzb = _tc_operand(C, fmt, stand_in(y))
        S = add(C, ea, eb, width=fmt.exp_bits + 1)
        S = add(C, S, const_bits((fmt.offset + U.S_BIAS) % 1024, 10), width=10)
        mag = multiply(C, siga, sigb)[: 2 * fmt.sig_bits]
        sh = fmt.product_shift + rescale
        prods.append((sa ^ sb, S, C.AND(nza, nzb), ([ZERO] * sh + mag) if sh >= 0 else mag[-sh:]))
    c = _finite_or_zero(C, acc, 23, 8)
    cm, cf, cs = c[:23], c[23:31], c[31]
    hc = or_reduce(C, cf)
    cf_eff = [cf[0] ^ C.AND(NOT(cf[0]), NOT(hc))] + cf[1:]
    mag24 = cm + [hc]
    state = (cs, add(C, cf_eff, const_bits(127, 10), width=10), OR(C, hc, or_reduce(C, cm)),
             ([ZERO] * rescale + mag24) if rescale >= 0 else mag24[-rescale:])
    # the special value entering each group: (nan, +inf, -inf); the accumulator first
    nan, pos, neg = c_nan, C.AND(c_inf, NOT(acc[31])), C.AND(c_inf, acc[31])
    sats, start, m24 = [], 0, None
    for size in groups:
        for i in range(start, start + size):
            nan = OR(C, nan, p_nan[i])
            pos = OR(C, pos, C.AND(p_inf[i], NOT(pc[i][2])))
            neg = OR(C, neg, C.AND(p_inf[i], pc[i][2]))
        nan = OR(C, nan, C.AND(pos, neg))
        s_o, S_o, nz_o, m24 = U.group_sum(C, [state] + prods[start:start + size], W, F)
        sat = U.saturated(C, S_o, nz_o)
        sats.append((sat, s_o))
        # an overflowed finite group is an infinity of its sign (only when the group had no infinite term)
        fin_sat = C.AND(sat, NOT(OR(C, pos, neg)))
        pos, neg = OR(C, pos, C.AND(fin_sat, NOT(s_o))), OR(C, neg, C.AND(fin_sat, s_o))
        state = (s_o, S_o, nz_o, ([ZERO] * rescale + m24) if rescale >= 0 else m24[-rescale:])
        start += size
    s_o, S_o, nz_o = state[0], state[1], state[2]
    field = add(C, S_o[:8], const_bits((-128) % 256, 8), cin=m24[23], width=8)
    sign_n = C.AND(s_o, nz_o)
    field_n = [C.AND(nz_o, x) for x in field]
    sat_any, sat_sign = sats[-1]
    for sat_g, sgn in reversed(sats[:-1]):
        sat_sign = mux(C, sat_g, [sgn], [sat_sign])[0]
        sat_any = OR(C, sat_g, sat_any)
    finite = [C.AND(NOT(sat_any), x) for x in m24[:23]] + [OR(C, sat_any, x) for x in field_n] + [mux(C, sat_any, [sat_sign], [sign_n])[0]]
    inf = OR(C, pos, neg)
    special = select(C, nan, const_bits(TC_NAN, 32), const_bits(F32_INF, 31) + [neg])
    return select(C, OR(C, nan, inf), special, finite)


def _tc_operand(C, fmt, word):
    """``unit.decode_operand`` without its finiteness assertion (the operand is a finite stand-in here)."""
    m, e, s = word[: fmt.man_bits], word[fmt.man_bits: fmt.man_bits + fmt.exp_bits], word[fmt.bits - 1]
    h = or_reduce(C, e)
    return s, [e[0] ^ C.AND(NOT(e[0]), NOT(h))] + e[1:], m + [h], OR(C, h, or_reduce(C, m))


def decode(C, bits):
    """The 2^n minterms of ``bits`` (index i = the assignment with value i), each one committed bit or free."""
    n = len(bits)
    if n == 0:
        return [ONE]
    if n == 1:
        return [NOT(bits[0]), bits[0]]
    h = n // 2
    lo, hi = decode(C, bits[:h]), decode(C, bits[h:])
    return [C.AND(lo[i & ((1 << h) - 1)], hi[i >> h]) for i in range(1 << n)]


def lookup(C, idx, table, out_bits: int, lo_bits: int | None = None):
    """``table[idx]`` (``out_bits`` wide) for a k-bit index: idx splits into lo and hi, both decode to one-hot, and output bit j
    is the sum over hi minterms h of ``AND(hi_h, sum of the lo minterms l with bit j of table[h, l] set)``. Committed bits
    ≈ 2^lo + 2^hi * (out_bits + 1); the table lives in the linear forms (the matrix)."""
    k = len(idx)
    assert len(table) == 1 << k
    if lo_bits is None:
        lo_bits = min(range(k + 1), key=lambda lb: (1 << lb) + (1 << (k - lb)) * (out_bits + 1))
    lo_m = decode(C, idx[:lo_bits])
    hi_m = decode(C, idx[lo_bits:])
    out = []
    nlo = 1 << lo_bits
    for j in range(out_bits):
        acc = ZERO
        for h, hm in enumerate(hi_m):
            s: set = set()
            c = 0
            base = h * nlo
            for lo in range(nlo):
                if (table[base + lo] >> j) & 1:
                    s.symmetric_difference_update(lo_m[lo].s)
                    c ^= lo_m[lo].c
            acc = acc ^ C.AND(hm, L(frozenset(s), c))
        out.append(acc)
    return out
