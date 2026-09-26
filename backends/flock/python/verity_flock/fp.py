"""Per-op circuit pieces for C-Flock's generic IR lowering: exact f32 / bf16 arithmetic in Flock's GF(2) model.

A word is a list of :class:`gf2.L` linear forms, LSB first. XOR, NOT, constant shifts and bit selection are free; an AND
of two non-constant forms is one committed bit (``gf2.Circuit.AND``). Constants propagate, so each piece is written once
for general f32 operands and specializes itself when called on bf16-widened words (16 constant-zero mantissa bits) or on
program constants.

Every piece that produces an f32 ends in :func:`round_f32`, the one rounding routine (round to nearest even, subnormals
kept, overflow to infinity). A NaN result is the canonical 0x7FC00000: the IR primitives' NaN payloads differ (numpy's
propagate an operand, ``f32_fma_bits`` returns 0x7FC00000), and the pinned templates observe NaN only through their final
bf16 casts, which canonicalize.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gf2 import ONE, ZERO, L, NOT, OR, add, and_reduce, const_bits, lzc, multiply, mux, or_reduce, sub

F32_QNAN = 0x7FC00000
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


def round_f32(C, sign, E_b, off: int, M, zero_sign, underflow: bool = True):
    """The f32 word nearest (ties to even) to ``(-1)^sign * M * 2^(E - 127 - (len(M) - 1))`` with ``E = E_b - off``: ``E`` is the
    exponent field the value would have if M's top bit were the hidden bit. ``E_b`` is unsigned (``len(E_b)`` bits wide
    enough for ``E_b + 1``). M = 0 gives the zero of sign ``zero_sign``. Overflow gives infinity. ``underflow=False`` promises
    ``E >= 1`` (no right shift into the subnormal range is ever needed), which drops that shifter."""
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
    rnd = C.AND(guard, OR(C, sticky, sig[0]))
    ex = add(C, X, [sig[23]], width=EW)                                    # the exponent field before rounding
    _, lt255 = sub(C, ex, const_bits(255, EW), EW)
    ovf = NOT(lt255)
    word = add(C, sig[:23] + ex[:8], [rnd], width=31)
    word = [C.AND(NOT(ovf), b) for b in word[:23]] + [OR(C, ovf, b) for b in word[23:31]]
    word = [C.AND(NOT(az), b) for b in word]
    return word + [mux(C, az, [zero_sign], [sign])[0]]


def specials(C, word, nan, inf, inf_sign):
    """Override ``word`` with the canonical NaN, or with the infinity of sign ``inf_sign``."""
    word = mux(C, inf, const_bits(F32_INF, 31) + [inf_sign], word)
    return mux(C, nan, f32_const(F32_QNAN), word)


# --- f32 arithmetic ----------------------------------------------------------------------------------------------------

def f32_mul(C, a, b):
    """``RN(a * b)`` (FMUL, no ftz)."""
    A, B = unpack(C, a), unpack(C, b)
    s = A.s ^ B.s
    P = multiply(C, trim_low(A.sig)[0], trim_low(B.sig)[0])   # E does not depend on trimming zero LSBs
    E_b = add(C, A.e_eff, B.e_eff, width=10)                  # E = ea + eb - 126
    r = round_f32(C, s, E_b, 126, P, s)
    nan = OR(C, OR(C, A.nan, B.nan), OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)))
    return specials(C, r, nan, OR(C, A.inf, B.inf), s)


def f32_add(C, a, b):
    """``RN(a + b)`` (FADD, no ftz); IEEE zero signs: ``x + (-x) = +0``, ``(-0) + (-0) = -0``."""
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
    r = round_f32(C, sx, ex + [ZERO], -1, S, C.AND(A.s, B.s), underflow=False)
    nan = OR(C, OR(C, A.nan, B.nan), C.AND(C.AND(A.inf, B.inf), A.s ^ B.s))
    inf = OR(C, A.inf, B.inf)
    return specials(C, r, nan, inf, mux(C, A.inf, [A.s], [B.s])[0])


def f32_neg(a):
    return list(a[:31]) + [NOT(a[31])]


def f32_sub(C, a, b):
    return f32_add(C, a, f32_neg(b))


def _normalize_sig(C, F: F32):
    """``(sig << lz, lz, trimmed zero LSBs)`` for a nonzero significand: its constant-zero LSBs trimmed, the rest shifted up
    to its top bit (padded to a power of two for the leading-zero count)."""
    sig, t = trim_low(F.sig)
    w = len(sig)
    pw = ceil_pow2(w)
    padded = [ZERO] * (pw - w) + sig
    _, lz = lzc(C, padded)
    return shl(C, padded, lz)[pw - w:], lz, t


def f32_fma(C, a, b, c):
    """``RN(a * b + c)`` with one rounding (FFMA, no ftz), as ``verity_vllm``'s ``f32_fma_bits``.

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
    r = round_f32(C, sp ^ neg, E_b, 160, M, ZERO)
    pz = OR(C, A.zero, B.zero)
    zero_c = f32_const(0)[:31] + [C.AND(sp, Cc.s)]
    r = mux(C, pz, mux(C, Cc.zero, zero_c, list(c)), r)
    prod_inf = OR(C, A.inf, B.inf)
    nan = OR(C, OR(C, OR(C, A.nan, B.nan), Cc.nan),
             OR(C, OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf)), C.AND(C.AND(prod_inf, Cc.inf), sp ^ Cc.s)))
    inf = OR(C, prod_inf, Cc.inf)
    return specials(C, r, nan, inf, mux(C, prod_inf, [sp], [Cc.s])[0])


def f32_div_const(C, a, b_word: int):
    """``RN(a / b)`` (``div.rn.f32``) for a constant divisor that is a power of two: ``RN(a * 2^-k)`` with an exact
    reciprocal. Other divisors need the general divider, which no pinned template uses yet."""
    e, m = (b_word >> 23) & 0xFF, b_word & 0x7FFFFF
    if m or e in (0, 0xFF) or not 1 <= 254 - e <= 254:
        raise NotImplementedError(f"f32 division by 0x{b_word:08x}: only by a normal power of two whose reciprocal is normal")
    return f32_mul(C, a, f32_const((b_word & 0x80000000) | ((254 - e) << 23)))


# --- bf16 ----------------------------------------------------------------------------------------------------------------

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
