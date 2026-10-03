"""Pearl-C's per-tile cap on a circuit (a prototype at small shapes; no Program calls it yet): the tile unit `Pc8Tile` with
rev1's ``capOK`` (`DeviceRev1.lean`'s `pearlCTilesDevRev1`) decided on one more output, so the verifier reads the
verdict and never the tile's codes.

    template                          reads                                            computes and returns
    Pc8CapWord{K,DEV}                 a word's A′ and B̃ rows, its certificates        C̃, its k/32 flags, its later +0
                                                                                       promotions, and good
    Pc8TileCap{K,TM,TN,N,RHO,DEV}     `Pc8Tile`'s inputs, the A rows' ok words, their  `Pc8Tile`'s C̃ and U, and the
                                      salt-dead counts, every word's certificates      verdict (an ok word)

**The word** (`Pc8CapWord`) is `pearl_c_debit.word_flags` and rev1's `laterZeroPromos` on the chain the tile already
computes, so C̃ is `pc8`'s word for word:
  - per group of G atoms the 2^G masked words from +0 (bit j: atom j passes its word through), the unmasked one the
    chain's own, so 2^G − 1 − G atom steps more per group (11 on G = 4);
  - each nonzero mask's word promoted into the total before the group (no add for the group of +0 total or the word +0);
  - **Freeze's certificates in place of the bisection** (`Freeze.lean`'s `certChain`, `keepsFast`): the prover supplies,
    per word and group g < k/(32·G) − 1, the totals [lo_g, hi_g] after group g that the later promotions carry to C̃,
    and the word checks them with four promotion adds per group: lo_g and hi_g land in the next interval, and their
    neighbours pred(lo_g) and succ(hi_g) in FP32 order land outside it. The last interval is C̃ itself. The checks pass
    only for the exact intervals (by induction from the last), and there no FP32 total lies between pred(lo_g) and lo_g
    or between hi_g and succ(hi_g), so `keepsFast` never falls back: a mask keeps iff its total is in [lo_g, hi_g];
  - atom t of group g is flagged when some mask containing it keeps; a later promotion into a +0 total is a g in
    [1, k/(32·G)) whose total after g groups is ±0;
  - ``good``: the certificates pass, and every atom output, promotion and certificate add is finite (where the scheme's
    `_step` and `_promote` raise, `good` is 0).
FP32 values compare by value with ±0 one value (`_Cmp`): two signed integer comparisons on the encodings, with the
sign and zero of each value computed once.

**The tile** (`Pc8TileCap`) is `Pc8Tile` plus, on its passing rows R (an A row's ok word, `pc4.ok_word`) and every
column C of the tile, Lean's per-tile cap at the device's prices, ρ = 1/RHO and the unit's n = N:

    (32 + add/G)·atoms + fs·(|C|/N)·elems + add·zeros  <=  ρ · creditDevRev1(m, K, N) · |R|·|C| / (m·N)

  - atoms and zeros: the flags and the later +0 promotions summed over the words of R × C;
  - elems: per row of R and atom block t (32 elements), 32 when every word of the row flags atom t, else the block's
    salt-dead count (``sd``, P3's `saltDeadP`; in the deployed design the row unit `Pc8RowA` computes it, since it reads
    α, β and x);
  - m cancels; multiplied by G·N, the inequality is RHO·(N·L + G·fs·|C|·elems) <= |R|·|C|·P over integers, with
    L = (32·G + add)·atoms + G·add·zeros and P = G·N·(credit per cell of the unit), and it is decided on 32-bit words: L
    against its largest passing value, then (N/g)·L + (G·fs·|C|/g)·elems against ⌊|R|·|C|·P/(RHO·g)⌋, g =
    gcd(N, G·fs·|C|), a table over |R| (`_cap_constants`, which refuses a shape whose sums could leave 31 bits);
  - the verdict: every passing row's words are good, and the inequality holds.
The certificates (``cert``) are the prover's, from `pearl_c_debit`'s own bisection; any other certificate fails ``good``.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import gcd

from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import array_of, tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml.fp32 import F32AddV2
from verity.ml.scalar import Bf16GtStrict, BitAnd, BitNot, BitOr, F32Fabs, F32IsFinite, I32Add, I32Eq, I32Le, SelectI32

from ..protocol import DomainError, Shape
from ..schemes import pearl_c as C
from ..schemes import pearl_c_debit as D
from ..schemes.pearl_c_device import DEVICES
from . import pc4, pc8

F32, BIT, I32, OK = pc8.F32, Value(1), Value(32), pc8.OK
SLICE = C.SLICE
_NEG_ONE, _BELOW_ZERO, _ABOVE_ZERO, _I32_TOP = 0xFFFFFFFF, 0x80000001, 0x00000001, (1 << 31) - 1


def _not(e: pc4._Emit, a):
    return 1 - a if isinstance(a, int) else e.call(BitNot, a)


def _and(e: pc4._Emit, a, b):
    if isinstance(a, int) or isinstance(b, int):
        c, other = (a, b) if isinstance(a, int) else (b, a)
        return other if c else 0
    return e.call(BitAnd, a, b)


def _or(e: pc4._Emit, a, b):
    if isinstance(a, int) or isinstance(b, int):
        c, other = (a, b) if isinstance(a, int) else (b, a)
        return 1 if c else other
    return e.call(BitOr, a, b)


def _add(e: pc4._Emit, a, b):
    if isinstance(a, int) and a == 0:
        return b
    if isinstance(b, int) and b == 0:
        return a
    return e.call(I32Add, a, b)


def _mul(e: pc4._Emit, c: int, x):
    """c·x mod 2^32 for a constant c >= 0: double and add from the top bit."""
    out = 0
    for bit in bin(c)[2:]:
        out = _add(e, out, out)
        if bit == "1":
            out = _add(e, out, x)
    return out


def _count(e: pc4._Emit, bits) -> object:
    total = 0
    for b in bits:
        total = _add(e, total, b if isinstance(b, int) else e.call(SelectI32, b, 1, 0))
    return total


def _wire(e: pc4._Emit, w: int, v):
    return e.c(w, v) if isinstance(v, int) else v


class _Cmp:
    """Value order on finite FP32 words, ±0 one value: per value its sign (signed x <= −1) and zero (|x| = 0) once, then
    x <= y is y <= x on the encodings when both are negative, else x <= y on them or both zero."""

    def __init__(self, e: pc4._Emit) -> None:
        self.e, self._memo = e, {}

    @staticmethod
    def _key(x):
        return ("c", x) if isinstance(x, int) else ("w", id(x))

    def _once(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def facts(self, x):
        e = self.e
        return self._once(("facts", self._key(x)),
                          lambda: (e.call(I32Le, x, _NEG_ONE), e.call(I32Eq, e.call(F32Fabs, x), 0)))

    def _pair(self, x, y):
        """(both negative, not both negative, both zero) of an unordered pair."""
        e = self.e
        (sx, zx), (sy, zy) = self.facts(x), self.facts(y)

        def make():
            both = _and(e, sx, sy)
            return both, _not(e, both), _and(e, zx, zy)
        return self._once(("pair", *sorted((self._key(x), self._key(y)))), make)

    def le(self, x, y):
        e = self.e
        both, other, zz = self._pair(x, y)

        def under(when, u, v, then):
            if isinstance(when, int) and not when:
                return 0
            return _and(e, when, then(self._once(("le", self._key(u), self._key(v)), lambda: e.call(I32Le, u, v))))
        return _or(e, under(both, y, x, lambda c: c), under(other, x, y, lambda c: _or(e, c, zz)))

    def zero(self, x):
        return self.facts(x)[1]

    def pred(self, x):
        """The FP32 value just below x (−2^-149 below ±0); −MAX's is −inf."""
        e, (s, z) = self.e, self.facts(x)
        return e.call(SelectI32, z, _BELOW_ZERO, e.call(SelectI32, s, e.call(I32Add, x, 1), e.call(I32Add, x, _NEG_ONE)))

    def succ(self, x):
        """The FP32 value just above x (2^-149 above ±0); MAX's is +inf."""
        e, (s, z) = self.e, self.facts(x)
        return e.call(SelectI32, z, _ABOVE_ZERO, e.call(SelectI32, s, e.call(I32Add, x, _NEG_ONE), e.call(I32Add, x, 1)))


def _finite(e: pc4._Emit, ok, x):
    return _and(e, ok, e.call(F32IsFinite, x))


def _promoted(dev: str, k: int):
    d = DEVICES[dev]
    if d.g is None or d.atom.name not in pc8.ATOMS:
        raise DomainError(f"Pearl-C's cap circuit needs a promoted device record with a core atom: not {dev!r}")
    if k % d.window or k // d.window < 2:
        raise DomainError(f"Pearl-C's cap circuit needs at least two whole promotion windows of {d.window}: k = {k}")
    return d, k // d.window


# ---- the word -------------------------------------------------------------------------------------------------------------

def _word_sig(S_):
    _d, ng = _promoted(S_.DEV, S_.K)
    k = S_.K
    return ((("a", Array(k, pc8.E4M3)), ("b", Array(k, pc8.E4M3)), ("cert", Array(ng - 1, Array(2, F32)))),
            Tuple(F32, Array(k // SLICE, BIT), I32, BIT))


def _word_body(B, S_, a, b, cert):
    e, k = pc4._Emit(B), S_.K
    d, ng = _promoted(S_.DEV, k)
    atom, g_ = pc8.ATOMS[d.atom.name], d.g
    full = (1 << g_) - 1
    cmp = _Cmp(e)
    good = 1
    groups, totals = [], [0]
    for g in range(ng):
        words = {0: 0}
        for j in range(g_):
            s = g * d.window + j * SLICE
            nxt = {}
            for mask, w in words.items():
                out = e.call(atom, w, a[s:s + SLICE], b[s:s + SLICE])
                good = _finite(e, good, out)
                nxt[mask], nxt[mask | 1 << j] = out, w
            words = nxt
        groups.append(words)
        totals.append(e.call(F32AddV2, totals[-1], words[0]))
        good = _finite(e, good, totals[-1])
    bounds = [(cert[g][0], cert[g][1]) for g in range(ng - 1)] + [(totals[-1], totals[-1])]
    for g in range(ng - 1):
        (lo, hi), (lo2, hi2), v = bounds[g], bounds[g + 1], groups[g + 1][0]
        below, above = cmp.pred(lo), cmp.succ(hi)
        for x in (lo, hi, below, above):
            good = _finite(e, good, x)
        s_lo, s_hi, s_below, s_above = (e.call(F32AddV2, x, v) for x in (lo, hi, below, above))
        for x in (s_lo, s_hi, s_below, s_above):
            good = _finite(e, good, x)
        good = _and(e, good, _and(e, _and(e, cmp.le(lo2, s_lo), cmp.le(s_hi, hi2)),
                                  _and(e, _not(e, cmp.le(lo2, s_below)), _not(e, cmp.le(s_above, hi2)))))
    flags = []
    for g in range(ng):
        lo, hi = bounds[g]
        keeps = {}
        for mask, w in groups[g].items():
            if not mask:
                continue
            if g == 0 or mask == full:
                x = totals[g] if mask == full else w
            else:
                x = e.call(F32AddV2, totals[g], w)
                good = _finite(e, good, x)
            keeps[mask] = _and(e, cmp.le(lo, x), cmp.le(x, hi))
        for j in range(g_):
            f = 0
            for mask in sorted(keeps):
                if mask >> j & 1:
                    f = _or(e, f, keeps[mask])
            flags.append(_wire(e, 1, f))
    zeros = _count(e, [cmp.zero(totals[g]) for g in range(1, ng)])
    return tuple_of(totals[-1], array_of(flags), _wire(e, 32, zeros), _wire(e, 1, good))


Pc8CapWordDef = CompositeDefinition(
    "Pc8CapWord", 1, ("K", "DEV"), _word_sig, _word_body,
    doc="A word's chain C̃ (`pc8`'s), its debit (A) flags per atom (`pearl_c_debit.word_flags`), its later promotions into "
        "a +0 total (rev1's `laterZeroPromos`), and good: Freeze's certificates of the later promotions' intervals pass "
        "and every value is finite.", conformance="protocol-defined")


# ---- the tile -------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cap_constants(k: int, tm: int, tn: int, n: int, rho: int, dev: str) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """(atom, zero, n/g, elem/g, L's largest passing value, ⌊|R|·|C|·P/(RHO·g)⌋ for |R| = 0..TM) of the cap's integer
    form (module docstring), or DomainError where a sum could leave 31 bits."""
    d, ng = _promoted(dev, k)
    if not (0 < tn <= n and n % tn == 0 and tm > 0 and rho > 0):
        raise DomainError(f"Pearl-C's cap circuit needs TN | N and ρ = 1/RHO: TM={tm}, TN={tn}, N={n}, RHO={rho}")
    p, g_ = d.prices, d.g
    big_p = g_ * C.PearlC(forming="v1", device=d).credit_of(Shape(1, k, n))
    atom, zero, elem = SLICE * g_ + p.fadd, g_ * p.fadd, g_ * p.forming_credited * tn
    gg = gcd(n, elem)
    table = tuple(int(Fraction(rows * tn * big_p, rho * gg)) for rows in range(tm + 1))
    l_max = table[-1] // (n // gg)
    words = tm * tn * (k // SLICE)
    if (atom * words + zero * tm * tn * ng > _I32_TOP or (n // gg) * l_max + (elem // gg) * tm * k > _I32_TOP
            or table[-1] > _I32_TOP):
        raise DomainError(f"Pearl-C's cap circuit decides on 31-bit sums: TM={tm}, TN={tn}, K={k}, N={n} is too large")
    return atom, zero, n // gg, elem // gg, l_max, table


def _tile_cap_sig(S_):
    k, tm, tn = S_.K, S_.TM, S_.TN
    _d, ng = _promoted(S_.DEV, k)
    _cap_constants(k, tm, tn, S_.N, S_.RHO, S_.DEV)
    params = ((("ac", pc8.CODES(tm, k)), ("bc", pc8.CODES(tn, k)), ("pa", Array(tm, pc8.P_ROW)),
               ("pb", Array(tn, pc8.P_ROW)), ("oka", Array(tm, OK)), ("sd", Array(tm, Array(k // SLICE, I32))),
               ("cert", Array(tm, Array(tn, Array(ng - 1, Array(2, F32)))))))
    return params, Tuple(Array(tm, Array(tn, F32)), Array(tm, Array(tn, F32)), OK)


def _tile_cap_body(B, S_, ac, bc, pa, pb, oka, sd, cert):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    atom, zero, n_g, elem_g, l_max, table = _cap_constants(k, S_.TM, S_.TN, S_.N, S_.RHO, dev)
    peel = pc8.PEELS[DEVICES[dev].peel.name]
    word = templates(k, dev)["word"]
    cs, us = [], []
    atoms = zeros = elems = rows = 0
    good = 1
    for i in range(S_.TM):
        ok = e.call(Bf16GtStrict, oka[i], 0)
        crow, urow, flags = [], [], []
        row_atoms = row_zeros = 0
        row_good = 1
        for j in range(S_.TN):
            out = B.call(word, ac[i], bc[j], cert[i][j])
            u = out[0]
            for s in range(0, pc8.PEEL, 16):
                u = e.call(peel, u, pa[i][s:s + 16], pb[j][s:s + 16])
            crow.append(out[0])
            urow.append(u)
            flags.append(out[1])
            row_atoms = _add(e, row_atoms, _count(e, [out[1][t] for t in range(k // SLICE)]))
            row_zeros = _add(e, row_zeros, out[2])
            row_good = _and(e, row_good, out[3])
        row_elems = 0
        for t in range(k // SLICE):
            every = flags[0][t]
            for f in flags[1:]:
                every = _and(e, every, f[t])
            row_elems = _add(e, row_elems, e.call(SelectI32, every, SLICE, sd[i][t]))
        atoms = _add(e, atoms, e.call(SelectI32, ok, row_atoms, 0))
        zeros = _add(e, zeros, e.call(SelectI32, ok, row_zeros, 0))
        elems = _add(e, elems, e.call(SelectI32, ok, row_elems, 0))
        rows = _add(e, rows, e.call(SelectI32, ok, 1, 0))
        good = _and(e, good, _or(e, _not(e, ok), row_good))
        cs.append(array_of(crow))
        us.append(array_of(urow))
    debit = _add(e, _mul(e, atom, atoms), _mul(e, zero, zeros))
    bound = table[0]
    for r, t in enumerate(table[1:], 1):
        bound = e.call(SelectI32, e.call(I32Eq, rows, r), t, bound)
    lhs = _add(e, _mul(e, n_g, debit), _mul(e, elem_g, elems))
    within = _and(e, e.call(I32Le, debit, l_max), e.call(I32Le, lhs, bound))
    return tuple_of(array_of(cs), array_of(us), pc4.ok_word(e, _and(e, good, within)))


Pc8TileCapDef = CompositeDefinition(
    "Pc8TileCap", 1, ("K", "TM", "TN", "N", "RHO", "DEV"), _tile_cap_sig, _tile_cap_body,
    doc="`Pc8Tile`'s C̃ and U, and rev1's per-tile capOK at ρ = 1/RHO on a unit of N columns as an ok word: the flagged "
        "atom-words, the later +0 promotions and the forming elements of the tile's passing rows against ρ times their "
        "cells' share of `creditDevRev1` (`pearlCTilesDevRev1`).", conformance="protocol-defined")


def certificates(a: Sequence[int], b: Sequence[int], dev: str = "sm120") -> list[int]:
    """A word's ``cert`` input, flat: (lo_g, hi_g) as FP32 words for each group g but the last, the intervals
    `pearl_c_debit.word_flags` bisects (raises `pearl_kw.Reject` where it does)."""
    d, ng = _promoted(dev, len(a))
    atoms = [(a[s:s + SLICE], b[s:s + SLICE]) for s in range(0, len(a), SLICE)]
    groups = [D._masked(atoms[g:g + d.g], d) for g in range(0, len(atoms), d.g)]
    total = 0
    for masked in groups:
        total = D._promote(total, masked[0])
    lo = hi = D._key(total)
    out = []
    for g in range(ng - 1, 0, -1):
        w = groups[g][0]
        lo, hi = (D._least(lambda x, w=w, lo=lo: D._carried(x, w) >= lo),
                  D._least(lambda x, w=w, hi=hi: D._carried(x, w) > hi) - 1)
        out[:0] = [D._word(lo), D._word(hi)]
    return out


@lru_cache(maxsize=16)
def templates(k: int, dev: str = "sm120", tile: tuple[int, int] = (1, 1), n: int | None = None,
              rho: int | None = None) -> dict[str, SpecializedDefinition]:
    """The word at depth ``k`` on ``dev``, and the tile with its cap at ``tile``, the unit's ``n`` (the tile's columns by
    default) and ρ = 1/``rho`` (the device's `cap` by default)."""
    cap = DEVICES[dev].cap if rho is None else Fraction(1, rho)
    if cap.numerator != 1:
        raise DomainError(f"Pearl-C's cap circuit needs ρ = 1/RHO: not {cap}")
    return {"word": bind(Pc8CapWordDef, K=k, DEV=dev),
            "tile": bind(Pc8TileCapDef, K=k, TM=tile[0], TN=tile[1], N=n or tile[1], RHO=cap.denominator, DEV=dev)}


__all__ = ["Pc8CapWordDef", "Pc8TileCapDef", "certificates", "templates"]
