"""RN32 tanh and Gemma-2's GeGLU on bits, by arithmetic: the Boolean Definitions whose gates compute the entries of the registered
tables `tanh_rn` and `gelu_tanh_bf16` (`registry.tables`) instead of reading them.

    TanhF32Rn_v3         (x: Array<32, Value<1>>) -> Array<32, Value<1>>                          word view: TanhF32Rn_v1
    GeluTanhMulBf16_v4   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                       word view: GeluTanhMulBf16_v1
    GeluTanhMul_v4{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>    word view: GeluTanhMul_v1{I}

`TanhF32Rn_v3` is traced from `tanh_f32`: NaN -> 0x7FC00000, |x| >= HI -> ±1, |x| < LO -> x (`tables.TANH_RN_LO`/`HI`, the bounds
found by scanning every word), else x's sign on `tanh_core(|x|)`, which computes tanh|x| = (e^{2|x|} - 1) / (e^{2|x|} + 1) in
integers (F fraction bits, N greedy steps):

    Y = 2|x| 2^34 (exact; |x|'s exponent is 115..130)       k = floor(floor(Y / 2^28) 369 / 2^14) <= 2|x| / ln 2
    R = Y 2^(F-34) - k floor(ln 2 2^F)                      2|x| - k ln 2, in [0, 0.73)
    E = 2^F (1 + 2^-i) for each i = 1..N, in order, at which R >= c_i = round(ln(1 + 2^-i) 2^F) (then R -= c_i; E += floor(E / 2^i))
    E += floor(E R / 2^F)                                  e^R, to about 2^-(F-2)
    q = floor(2^36 (E - 2^(F-k)) / (E + 2^(F-k)))          tanh|x| = (e^R - 2^-k) / (e^R + 2^-k), non-restoring division

and rounds q to its 24 leading bits, ties up: ``floor((q + 1) / 2)`` at q's 25th bit is the nearest of the exact ratio, so the
only error is the ratio's distance from tanh|x|, about 2^-34 of a result ulp at `TANH_FN` = (72, 36) (every f32 word is checked
against `TanhF32Rn_v1`, and the region's positive results are `tanh_rn`'s pinned entries). `GeluTanhMulBf16_v4` is
`GeluTanhMulBf16_v1` on bits: ``a = (T)gelu_tanh(g)`` by `dense.gelu_tanh_bf16_bits`' f32 operations (`verity.ml.circuits.silicon.fp`)
with `tanh_f32` at `GELU_TANH_FN` = (34, 17), F the fewest fraction bits that hold Y whole. That tanh misses RN32 in the last bits
at some inner words, and no such miss reaches the activation: all 65,536 g give `gelu_tanh_bf16`'s words. Then
``bf16_rn(f32(a) * (f32(u) + 0.0f))`` with F2FP's NaN 0x7FFF. `GeluTanhMul_v4{I}` is a batch of I `GeluTanhMulBf16_v4` Calls
over (gu[i], gu[I + i]), so each member's 16 output bits are one unit under `Q_word_v1`. None reads a table or a MUFU, and no
sub-Call is a word Definition; with these, Gemma-2's GeGLU under `registered_values` binds `GeluTanhMul_v1` and reads no table.
"""
from __future__ import annotations

from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import (NOT, ONE, ZERO, add, compress_columns, const_bits, lzc, multiply, mux, or_reduce,
                                              shl_window, sub)
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import dense as D
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.tables import TANH_RN_HI, TANH_RN_LO

__all__ = ["GELU_TANH_FN", "GeluTanhMul", "GeluTanhMulBf16", "TANH_FN", "TanhF32Rn", "gelu_tanh_bf16", "gelu_tanh_mul", "tanh_core",
           "tanh_f32"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
NAN_F2FP = P.BF16_NAN_F2FP

#: (F, N) of `TanhF32Rn_v3`'s core, and of the tanh inside `GeluTanhMulBf16_v4`
TANH_FN = (72, 36)
GELU_TANH_FN = (34, 17)
#: |x|'s exponent field is EXP_LO..EXP_LO + 15 on [LO, HI); Y = 2|x| 2^34 = (1.m) << (e - EXP_LO)
EXP_LO = 115
#: quotient bits: tanh|x| >= tanh(LO) > 2^-12, so q's leading one is at bit 24 or above and q carries a round bit
QBITS = 36


@lru_cache(maxsize=None)
def constants(F: int, N: int) -> tuple[int, tuple[int, ...]]:
    """(floor(ln 2 2^F), (round(ln(1 + 2^-i) 2^F) for i = 1..N))."""
    with localcontext() as ctx:
        ctx.prec = 120
        s = Decimal(2) ** F
        ln2 = int((Decimal(2).ln() * s).to_integral_value(ROUND_FLOOR))
        cs = tuple(int(((1 + Decimal(2) ** -i).ln() * s).to_integral_value(ROUND_HALF_EVEN)) for i in range(1, N + 1))
    return ln2, cs


def _y(a: int) -> int:
    return ((a & 0x7FFFFF) | 0x800000) << ((a >> 23) - EXP_LO)


def _k(yt: int) -> int:
    return (yt * 369) >> 14


@lru_cache(maxsize=None)
def bounds(F: int, N: int) -> tuple[int, int, tuple[int | None, ...], int]:
    """(the largest k, the largest R, per step the largest R before it (None: no R reaches c_i, the step is dropped), the largest
    E) over [LO, HI), exactly: R's and E's widths come from these, so no bit is dropped that a word of the region sets."""
    ln2, cs = constants(F, N)
    y0, y1 = _y(TANH_RN_LO), _y(TANH_RN_HI - 1)
    kmax, rmax = 0, 0
    for yt in range(y0 >> 28, (y1 >> 28) + 1):
        lo, hi, k = max(yt << 28, y0), min(((yt + 1) << 28) - 1, y1), _k(yt)
        if (lo << (F - 34)) < k * ln2:
            raise ValueError(f"R < 0 at Y = {lo:#x}")
        kmax, rmax = max(kmax, k), max(rmax, (hi << (F - 34)) - k * ln2)
    steps, b, e = [], rmax, 1 << F
    for i, c in enumerate(cs, 1):
        if b < c:
            steps.append(None)
            continue
        steps.append(b)
        b, e = max(b - c, c - 1), e + (e >> i)
    return kmax, rmax, tuple(steps), e + ((e * b) >> F)


def _times_const(C, x, c: int):
    """x * c for the constant c: x's bits at each of c's set bits, carry-saved."""
    cols = [[] for _ in range(len(x) + c.bit_length())]
    for s in range(c.bit_length()):
        if c >> s & 1:
            for j, b in enumerate(x):
                cols[j + s].append(b)
    return compress_columns(C, cols)


def _onehot(C, k, n: int):
    """[k == v for v in range(n)] for the 5-bit k: a 2-bit and a 3-bit decoder, one AND per output."""
    lit = [(NOT(b), b) for b in k]
    lo = [C.AND(lit[0][v & 1], lit[1][v >> 1]) for v in range(4)]
    hi = [C.AND(C.AND(lit[2][v & 1], lit[3][v >> 1 & 1]), lit[4][v >> 2]) for v in range(8)]
    return [C.AND(lo[v & 3], hi[v >> 2]) for v in range(n)]


def tanh_core(C, a, F: int, N: int):
    """The 31 bits of RN32(tanh(|x|)) for the f32 magnitude ``a`` (31 bits, LSB first) in [LO, HI); any word outside it gives
    some 31 bits."""
    ln2, cs = constants(F, N)
    kmax, rmax, steps, emax = bounds(F, N)
    m, e = list(a[:23]) + [ONE], list(a[23:31])
    y = shl_window(C, m, add(C, e[:4], const_bits(-EXP_LO % 16, 4), width=4), 0, 39)
    k = _times_const(C, y[28:39], 369)[14:19]
    r, _ = sub(C, [ZERO] * (F - 34) + y, _times_const(C, k, ln2), F + 5)
    r = r[:rmax.bit_length()]
    ev, ew = 1 << F, F + 1
    big = list(const_bits(ev, ew))
    for i, (b, c) in enumerate(zip(steps, cs), 1):
        if b is None:
            continue
        w = b.bit_length()
        d, borrow = sub(C, r, const_bits(c, w), w)
        take = NOT(borrow)
        r = mux(C, take, d, r)[:max(b - c, c - 1).bit_length()]
        ev = ev + (ev >> i)
        ew = ev.bit_length()
        big = add(C, big, [C.AND(take, x) for x in big[i:]], width=ew)
    big = add(C, big, multiply(C, big, r)[F:], width=emax.bit_length())
    oh = [ZERO] * (F + 1)
    for v, bit in enumerate(_onehot(C, k, kmax + 1)):
        oh[F - v] = bit
    wd = (emax + (1 << F)).bit_length()
    num, _ = sub(C, big, oh, wd)
    den = add(C, big, oh, width=wd)
    w = wd + 2
    rem, s, q = num + [ZERO] * 2, ZERO, []
    for _ in range(QBITS):
        rem = add(C, [ZERO] + rem[:w - 1], [x ^ NOT(s) for x in den + [ZERO] * 2], cin=NOT(s), width=w)
        s = rem[w - 1]
        q.append(NOT(s))
    q = q[::-1]
    _, lz = lzc(C, q[QBITS - 16:])
    sh = shl_window(C, q, lz, QBITS - 25, QBITS)
    expo, _ = sub(C, const_bits(126, 8), lz, 8)
    return add(C, sh[1:24] + expo, [], cin=sh[0], width=31)


def tanh_f32(C, x, F: int, N: int):
    """``TanhF32Rn_v1`` on the f32 word x (32 bits, LSB first), the core at (F, N)."""
    a, s = list(x[:31]), x[31]
    nan = NOT(sub(C, a, const_bits(0x7F800001, 31), 31)[1])
    sat = NOT(sub(C, a, const_bits(TANH_RN_HI, 31), 31)[1])
    small = sub(C, a, const_bits(TANH_RN_LO, 31), 31)[1]
    mag = mux(C, small, a, tanh_core(C, a, F, N))
    mag = mux(C, sat, const_bits(0x3F800000, 31), mag)
    mag = mux(C, nan, const_bits(P.F32_NAN, 31), mag)
    return mag + [C.AND(s, NOT(nan))]


TanhF32Rn = definition("TanhF32Rn", 3, lambda C, x: tanh_f32(C, x, *TANH_FN), (("x", F32),), F32, word=D.TanhF32Rn,
                       doc="RN32(tanh(x)) by fixed-point exp and division: TanhF32Rn_v1, NaN 0x7FC00000")


def gelu_tanh_bf16(C, g):
    """``(T)gelu_tanh_kernel((float)g)``, `dense.gelu_tanh_bf16_bits`, for a 16-bit g (LSB first)."""
    f = fp.bf16_to_f32(g)
    x3 = fp.f32_mul(C, fp.f32_mul(C, f, f), f)
    inner = fp.f32_mul(C, const_bits(D.BETA_BITS, 32), fp.f32_fma(C, const_bits(D.KAPPA_BITS, 32), x3, f))
    t = tanh_f32(C, inner, *GELU_TANH_FN)
    act = fp.f32_mul(C, fp.f32_mul(C, const_bits(D.HALF_BITS, 32), f), fp.f32_add(C, const_bits(D.ONE_BITS, 32), t))
    return fp.f32_to_bf16(C, act, NAN_F2FP)


def gelu_tanh_mul(C, g, u):
    """`GeluTanhMulBf16_v1`: ``bf16_rn(f32(gelu_tanh_bf16(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF."""
    a = gelu_tanh_bf16(C, g)
    u0 = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(u0)), NAN_F2FP)


GeluTanhMulBf16 = definition("GeluTanhMulBf16", 4, gelu_tanh_mul, (("g", BF16), ("u", BF16)), BF16, word=D.GeluTanhMulBf16,
                             doc="vLLM act_and_mul_kernel<gelu_tanh> on bits, its tanh by fixed-point exp and division: "
                                 "GeluTanhMulBf16_v1, bf16(f32(bf16(gelu_tanh(g))) * (f32(u) + 0.0f)), NaN 0x7FFF")


def _gelu_tanh_mul(B, S, gu):
    """vLLM `gelu_tanh_and_mul` on bits: `out[i] = GeluTanhMulBf16_v4(gu[i], gu[I + i])`."""
    return B.batch(GeluTanhMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


GeluTanhMul = EW.on_bits("GeluTanhMul", 4, ("I",), lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
                         _gelu_tanh_mul, D.GeluTanhMul)
