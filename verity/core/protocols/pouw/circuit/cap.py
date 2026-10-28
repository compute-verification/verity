"""Pearl-C's per-tile cap on a circuit: rev1's ``capOK`` (`DeviceRev1.lean`'s `pearlCTilesDevRev1`) decided in gates on
public ok words, so the verifier reads the verdict and never the tile's codes, with the row unit `Pc8RowA`'s salt-dead
counts that the cap reads.  `pc8.Shape.cap` cuts the hiding call into a unit per word, two per tile row (its digests
and its sums), the tile and its tally (`Pc8CallCap`; the split below); `Pc8TileCap` is the cap in one tile unit, the
split's reference.

    template                          reads                                            computes and returns
    Pc8RowACap{K,DEV}                 `Pc8RowA`'s inputs                               `Pc8RowA`'s outputs, and the
                                                                                       row's k/32 salt-dead counts
    Pc8CapWord{K,DEV}                 a word's A′ and B̃ rows, its certificates        C̃, its k/32 flags, its later +0
                                                                                       promotions, and good
    Pc8TileCap{K,TM,TN,N,RHO,DEV}     `Pc8Tile`'s inputs, the A rows' ok words, their  `Pc8Tile`'s C̃ and U, and the
                                      salt-dead counts, every word's certificates      verdict (an ok word)

    the split's units (`unit_templates`)
    Pc8CapWordUnit{K,DEV}             A′_i ‖ P_A,i, B̃_j ‖ P_B,j, the word's            `Pc8CapWord`'s C̃ and U (the
      (one per word)                  certificates                                     peel); its flags, their count,
                                                                                       the later +0 promotions and
                                                                                       good's ok word
    Pc8CapStrip{K,TM,TN,N,RHO,DEV}    its TM A rows' ok words                          their admission bits (padded
      (one per strip of TM rows)                                                       to 64) and the cap's bound at
                                                                                       their passing count
    Pc8CapRowDigest{TN}               a tile row's TN words' C̃ and U                   its chunk digests (`leaves`'
      (one per tile row)                                                               D = 0x21 TurboSHAKE128)
    Pc8CapRowTally{K,TN}              a tile row's TN words' flags, counts,            the row's flag count, later +0
      (one per tile row)              promotions and good, its A row's salt-dead       promotions, forming elements
                                      counts                                           and good's ok word
    Pc8CapTile{TM,TN}                 its TM rows' chunk digests                       the tile digest
    Pc8CapTally{K,TM,TN,N,RHO,DEV}    its TM rows' sums, its strip's admission bits    the cap's verdict (an ok word)
                                      and bound

    the per-row cap (v2; `row_rho`, `served`)
    Pc8TileCap_v2{...,RHOR,DEV}       v1's                                             v1's, the verdict also capping
                                                                                       each passing row at ρr = 1/RHOR
    Pc8CapTally_v2{...,RHOR,DEV}      v1's                                             the same verdict from the sums
    Pc8CallCap_v2, Pc8CheckCap_v2     v1's                                             v1's, with the v2 tally

**The row** (`Pc8RowACap`) is `Pc8RowA`, node for node, plus ``sd``: per block of 32 elements, the count of elements whose
support's two ends agree (`words.PouwBlackwellE4m3SaltDead`, `PouwHopperE4m3SaltDead`: the codes e4m3(atom(f32(α·x),
[c_β]^32, [±0x43]^32)) are equal or both zeros), on the forming's own accumulator f32(α·x) and the row's flat code
c_β = e4m3(f32(β·44)).  On a one-group step (sm_120's and the H100's atoms) that count is P3's `saltDeadP` count:
`Security/Proofs/Pouw/PearlC/Cap/SaltDead.lean`'s (E) `saltDeadP_iff_ends` and (C) `saltDeadP_card_ends`, with no hypothesis on α, x or β.  Nothing here reads the code's
order in the signed count, which is monotone only for β >= 0 and antitone for β <= 0.

**The word** (`Pc8CapWord`) is `pearl_c_debit.word_flags` and rev1's `laterZeroPromos` on the chain the tile already
computes, so C̃ is `pc8`'s word for word:
  - per group of G = 4 atoms the 2^G masked words from +0 (bit j: atom j passes its word through), the unmasked one the
    chain's own: one window primitive per group (`words.PouwBlackwellE4m3Window`, `PouwHopperE4m3Window`), whose bit form
    builds each atom's products once for all the words it steps, and a constant read of each word (`PouwWindowWord`);
    every word but +0 is some atom's output, so all 15 finite is every atom output finite;
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
    salt-dead count (``sd``, P3's `saltDeadP`; the row unit `Pc8RowACap` computes it, since it reads α, β and x);
  - m cancels; multiplied by G·N, the inequality is RHO·(N·L + G·fs·|C|·elems) <= |R|·|C|·P over integers, with
    L = (32·G + add)·atoms + G·add·zeros and P = G·N·(credit per cell of the unit), and it is decided on 32-bit words: L
    against its largest passing value, then (N/g)·L + (G·fs·|C|/g)·elems against ⌊|R|·|C|·P/(RHO·g)⌋, g =
    gcd(N, G·fs·|C|), a table over |R| (`_cap_constants`, which refuses a shape whose sums could leave 31 bits);
  - the verdict: every passing row's words are good, and the inequality holds.
The certificates (``cert``) are the prover's, from `pearl_c_debit`'s own bisection; any other certificate fails ``good``.

**The per-row cap** (v2 of `Pc8TileCap`, `Pc8CapTally`, `Pc8CallCap` and `Pc8CheckCap`; v1 stays as it is): the verdict
also holds each passing row's own atoms, zeros and elems to the same inequality on that one row, |R| = 1, at ρr = 1/RHOR
(`_row_constants`; `pearlCTilesDevRev1RowCapAt` at ρ = 1/RHO and ρr), the clear path's row cap
(`clear_work.cap_failure`).  `served` binds the scheme's: under a hashing format with the row cap (-h3) ρr is
`PearlC.row_cap_at(k)`, 4ρ at the device's census ks and ρ elsewhere (`row_rho`); without one (-h0) the served cap is
v1's, the tile's alone, and the scheme refuses the census ks.

**The split** (`pc8.Shape.cap`, `Pc8CallCap`): a word is the cap's proof block, since a cap word is about 7.8M ANDs at
K = 8,192 and a 64 × 64 tile of them does not stage.  The root holds the A rows (`Pc8RowACap`) and a strip unit per TM
of them (`Pc8CapStrip`), then per strip its word units row by row, per tile row its digests (`Pc8CapRowDigest`) and its
sums (`Pc8CapRowTally`), its tile units and their tallies.  Each value is computed once: a word unit computes its own
chain (the window's unmasked words, which `Pc8CapWord` promotes into C̃) and its peel, so a row digest reads its row's
C̃ and U and computes only their chunk digests, and the tile reads its rows' chunk digests and computes only h0; the
strip reads its rows' ok words and computes their admission bits and the bound at their passing count, once for every
tile in the strip; a row tally reads its row's words' flags, their count, their later +0 promotions and good, and its
A row's salt-dead counts, and computes the row's sums; and the tally reads its rows' sums and the strip's bits and
bound, all through their committed rows (`PORTS`; the cut is `pc8.unit_cut`'s).
  - The tile and the tally are cut per tile row because C-Flock's composer gives each row a unit reads or writes an
    hm96 slot of 2^18 bits and a unit at most 2^27 bits (`circuit.K_MAX`), so at most 256 rows: a tile reading its
    words' rows directly would need 2^31 at 64 × 64.  Cut per row, no unit of a 64 × 64 tile has more than 66 rows
    (the row tally's TN cap-flags rows, its sd row and its cap-sums row).
  - The tally's verdict is `Pc8TileCap`'s: every passing row's words are good, and the inequality holds.  It is the
    tally's public output (``public`` in `PORTS`: the verifier reads it; no row holds it), beside the tile digest, and
    the verifier accepts a drawn tile only when its tile check passes (the digest is the miner's h0) and the verdict is
    ok.
  - A drawn tile's closure holds its A rows, its strip, all TM·TN of its word units, its TM row digests and row
    tallies, its tile and its tally (`plan.tile_layout`): every cap word is proved, never sampled.
  - The cap's rows: A's salt-dead counts (``sd``), each word's ``cap-word`` (C̃ ‖ U) and ``cap-flags`` (flags ‖ their
    count ‖ the later +0 promotions ‖ good) rows, each tile row's ``cap-chunks`` (its chunk digests ‖ zeros) and
    ``cap-sums`` (its sums ‖ good) rows, and each strip's ``cap-rows`` (admission bits ‖ bound ‖ zeros) row (`leaves`,
    which says why the cap-chunks and cap-rows rows are padded to 64 words).  The word unit returns its flags padded to
    whole 16-bit words with zero bits fixed in gates (as `pc4.ok_word` fixes an ok word's), so no bit of the row is
    outside its ports at any k; the row tally reads the first k/32.  A row digest and the strip return their rows' zero
    words the same way, and the tile and the tally leave them unread.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import gcd

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.fp32 import F32AddV2, F32MulV2
from verity.ml.circuits.ops.prims import F32ToE4m3Sat
from verity.ml.circuits.ops.scalar import Bf16GtStrict, BitAnd, BitNot, BitOr, F32Fabs, F32IsFinite, I32Add, I32Eq, I32Le, SelectI32

from ..protocol import DomainError, Matmul, Shape
from ..schemes import pearl_c as C
from ..schemes import pearl_c_debit as D
from ..schemes.pearl_c_device import DEVICES
from . import boolean as _boolean  # noqa: F401  registers the windows' _v2 Definitions C-Flock's lowering falls back to
from . import hashes as H
from . import leaves as L
from . import pc4, pc8
from . import words as W

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


#: the cap templates' advice (`pc8.PORTS`' class): Freeze's certificates, which the prover supplies and nothing binds; in
#: the split, the word unit's (`PORTS`)
ADVICE = ("cert",)

#: each core atom's window of masked words (`words.WINDOW`: G = 4)
WINDOWS = {"blackwell_sm120_e4m3_m16n8k32": W.PouwBlackwellE4m3Window, "hopper_e4m3_wgmma_k32": W.PouwHopperE4m3Window}
#: each core atom's salt-dead bit (`words.ends_agree`); both atoms are one-group steps, where `SaltDead`'s (E) holds
SALT_DEAD = {"blackwell_sm120_e4m3_m16n8k32": W.PouwBlackwellE4m3SaltDead,
             "hopper_e4m3_wgmma_k32": W.PouwHopperE4m3SaltDead}
#: the flat E line's entry, 44 at Pearl's norm 256 (`SaltDead.flatLine`), as an FP32 word: c_β = e4m3(f32(β·44))
FLAT_E = 0x42300000


# ---- the row --------------------------------------------------------------------------------------------------------------

def _row_a_cap_sig(S_):
    if DEVICES[S_.DEV].atom.name not in SALT_DEAD:
        raise DomainError(f"Pearl-C's cap circuit has no salt-dead bit for {S_.DEV!r}'s atom")
    params, _ret = pc8._row_sig(S_, "fb")
    return params, Tuple(Array(S_.K, pc8.E4M3), pc8.P_ROW, OK, Array(S_.K // SLICE, I32))


def _row_a_cap_body(B, S_, x, line, fa, fb):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    outs, accs, beta = pc8._row_a(e, k, dev, x, line, fa, fb)
    dead = SALT_DEAD[DEVICES[dev].atom.name]
    cb = e.call(F32ToE4m3Sat, e.call(F32MulV2, beta, FLAT_E))
    bits = [e.call(dead, acc, cb) for acc in accs]
    return tuple_of(*outs, array_of([_wire(e, 32, _count(e, bits[s:s + SLICE])) for s in range(0, k, SLICE)]))


Pc8RowACapDef = CompositeDefinition(
    "Pc8RowACap", 1, ("K", "DEV"), _row_a_cap_sig, _row_a_cap_body,
    doc="`Pc8RowA`'s outputs, and per block of 32 elements the count of salt-dead elements: those whose support's two ends "
        "agree on the forming's accumulator f32(α·x) and the row's flat code e4m3(f32(β·44)) (`SaltDead`'s `EndsAgree`).",
    conformance="protocol-defined")


def _promoted(dev: str, k: int):
    d = DEVICES[dev]
    if d.g != W.WINDOW // SLICE or d.atom.name not in WINDOWS:
        raise DomainError(f"Pearl-C's cap circuit needs a device record promoted every 4 core atoms: not {dev!r}")
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
    window, g_ = WINDOWS[d.atom.name], d.g
    full = (1 << g_) - 1
    cmp = _Cmp(e)
    good = 1
    groups, totals = [], [0]
    for g in range(ng):
        s = g * d.window
        masked = e.call(window, a[s:s + d.window], b[s:s + d.window])
        words = {m: e.call(W.PouwWindowWord, masked, m) for m in range(full)}
        for w in words.values():
            good = _finite(e, good, w)
        words[full] = 0
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


def _peel(e: pc4._Emit, dev: str, c, pa, pb):
    """U: the device's four BF16 k16 steps from C̃ over P_A,i·P_B,j (`Pc8Tile`'s)."""
    peel = pc8.PEELS[DEVICES[dev].peel.name]
    u = c
    for s in range(0, pc8.PEEL, 16):
        u = e.call(peel, u, pa[s:s + 16], pb[s:s + 16])
    return u


def _row_elems(e: pc4._Emit, flags, sd, k: int):
    """A row's forming elements: per atom block, 32 when every word of the row (``flags``, one per column) flags it, else
    its salt-dead count."""
    out = 0
    for t in range(k // SLICE):
        every = flags[0][t]
        for f in flags[1:]:
            every = _and(e, every, f[t])
        out = _add(e, out, e.call(SelectI32, every, SLICE, sd[t]))
    return out


def _masked_add(e: pc4._Emit, sums, ok, row):
    """``sums`` plus ``row``'s terms where ``ok``."""
    return tuple(_add(e, s, e.call(SelectI32, ok, r, 0)) for s, r in zip(sums, row))


def _bound(e: pc4._Emit, table, rows):
    """⌊|R|·|C|·P/(RHO·g)⌋ at |R| = ``rows``: the table's entry (`_cap_constants`)."""
    bound = table[0]
    for r, t in enumerate(table[1:], 1):
        bound = e.call(SelectI32, e.call(I32Eq, rows, r), t, bound)
    return bound


def _within(e: pc4._Emit, consts, sums, bound=None):
    """The cap's inequality on the tile's sums (atoms, zeros, elems and, without ``bound``, rows), in its integer form
    (module docstring), as a bit."""
    atom, zero, n_g, elem_g, l_max, table = consts
    atoms, zeros, elems = sums[:3]
    debit = _add(e, _mul(e, atom, atoms), _mul(e, zero, zeros))
    if bound is None:
        bound = _bound(e, table, sums[3])
    lhs = _add(e, _mul(e, n_g, debit), _mul(e, elem_g, elems))
    return _and(e, e.call(I32Le, debit, l_max), e.call(I32Le, lhs, bound))


def _row_constants(k: int, tn: int, n: int, rhor: int, dev: str) -> tuple[tuple, int]:
    """The per-row cap's integer form, `_cap_constants` on one row at ρr = 1/RHOR, and its bound at that one row."""
    consts = _cap_constants(k, 1, tn, n, rhor, dev)
    return consts, consts[-1][1]


def _row_capped(S_, sig):
    """``sig`` of a v2 template, refusing a per-row cap whose integer form leaves 31 bits."""
    _row_constants(S_.K, S_.TN, S_.N, S_.RHOR, S_.DEV)
    return sig(S_)


def _row_over(e: pc4._Emit, row_consts: tuple[tuple, int], ok, atoms, zeros, elems):
    """``ok``, and the row's own atoms, zeros and elements over the per-row cap (the tile's test on one row).  No gate reads
    ``ok`` alone: a strip's tallies all read its admission bits, so such a gate would be computed in each of them."""
    consts, bound = row_consts
    return _and(e, ok, _not(e, _within(e, consts, (atoms, zeros, elems), bound)))


def _tile_cap_body(B, S_, ac, bc, pa, pb, oka, sd, cert):
    return _tile_cap(B, S_, ac, bc, pa, pb, oka, sd, cert, None)


def _tile_row_cap_body(B, S_, ac, bc, pa, pb, oka, sd, cert):
    return _tile_cap(B, S_, ac, bc, pa, pb, oka, sd, cert, _row_constants(S_.K, S_.TN, S_.N, S_.RHOR, S_.DEV))


def _tile_cap(B, S_, ac, bc, pa, pb, oka, sd, cert, row_consts):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    consts = _cap_constants(k, S_.TM, S_.TN, S_.N, S_.RHO, dev)
    word = templates(k, dev)["word"]
    cs, us = [], []
    sums, good = (0, 0, 0, 0), 1
    for i in range(S_.TM):
        ok = e.call(Bf16GtStrict, oka[i], 0)
        crow, urow, flags = [], [], []
        row_atoms = row_zeros = 0
        row_good = 1
        for j in range(S_.TN):
            out = B.call(word, ac[i], bc[j], cert[i][j])
            crow.append(out[0])
            urow.append(_peel(e, dev, out[0], pa[i], pb[j]))
            flags.append(out[1])
            row_atoms = _add(e, row_atoms, _count(e, [out[1][t] for t in range(k // SLICE)]))
            row_zeros = _add(e, row_zeros, out[2])
            row_good = _and(e, row_good, out[3])
        row_elems = _row_elems(e, flags, sd[i], k)
        sums = (*_masked_add(e, sums[:3], ok, (row_atoms, row_zeros, row_elems)),
                _add(e, sums[3], e.call(SelectI32, ok, 1, 0)))
        good = _and(e, good, _or(e, _not(e, ok), row_good))
        if row_consts is not None:
            good = _and(e, good, _not(e, _row_over(e, row_consts, ok, row_atoms, row_zeros, row_elems)))
        cs.append(array_of(crow))
        us.append(array_of(urow))
    return tuple_of(array_of(cs), array_of(us), pc4.ok_word(e, _and(e, good, _within(e, consts, sums))))


Pc8TileCapDef = CompositeDefinition(
    "Pc8TileCap", 1, ("K", "TM", "TN", "N", "RHO", "DEV"), _tile_cap_sig, _tile_cap_body,
    doc="`Pc8Tile`'s C̃ and U, and rev1's per-tile capOK at ρ = 1/RHO on a unit of N columns as an ok word: the flagged "
        "atom-words, the later +0 promotions and the forming elements of the tile's passing rows against ρ times their "
        "cells' share of `creditDevRev1` (`pearlCTilesDevRev1`).", conformance="protocol-defined")

Pc8TileCapDefV2 = CompositeDefinition(
    "Pc8TileCap", 2, ("K", "TM", "TN", "N", "RHO", "RHOR", "DEV"), lambda S_: _row_capped(S_, _tile_cap_sig),
    _tile_row_cap_body,
    doc="`Pc8TileCap` v1 with the per-row cap: its verdict, and each passing row's own flagged atom-words, later +0 "
        "promotions and forming elements against ρr = 1/RHOR times its cells' share of `creditDevRev1` (the tile's test "
        "on one row; `pearlCTilesDevRev1RowCapAt` at ρ = 1/RHO, ρr = 1/RHOR).", conformance="protocol-defined")


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


def _rho(dev: str, rho: int | None) -> int:
    cap = DEVICES[dev].cap if rho is None else Fraction(1, rho)
    if cap.numerator != 1:
        raise DomainError(f"Pearl-C's cap circuit needs ρ = 1/RHO: not {cap}")
    return cap.denominator


def _row_rho(row_rho: int | None) -> int | None:
    if row_rho is not None and (not isinstance(row_rho, int) or row_rho <= 0):
        raise DomainError(f"Pearl-C's per-row cap needs ρr = 1/RHOR: not RHOR = {row_rho!r}")
    return row_rho


@lru_cache(maxsize=16)
def templates(k: int, dev: str = "sm120", tile: tuple[int, int] = (1, 1), n: int | None = None,
              rho: int | None = None, row_rho: int | None = None) -> dict[str, SpecializedDefinition]:
    """The row of A with its salt-dead counts and the word at depth ``k`` on ``dev``, and the tile with its cap at
    ``tile``, the unit's ``n`` (the tile's columns by default) and ρ = 1/``rho`` (the device's `cap` by default), and with
    ``row_rho`` the per-row cap at ρr = 1/``row_rho`` (`Pc8TileCap` v2; v1 caps the tile alone)."""
    tile_cap = dict(K=k, TM=tile[0], TN=tile[1], N=n or tile[1], RHO=_rho(dev, rho), DEV=dev)
    return {"row_a": bind(Pc8RowACapDef, K=k, DEV=dev), "word": bind(Pc8CapWordDef, K=k, DEV=dev),
            "tile": (bind(Pc8TileCapDef, **tile_cap) if _row_rho(row_rho) is None
                     else bind(Pc8TileCapDefV2, **tile_cap, RHOR=row_rho))}


# ---- the split ------------------------------------------------------------------------------------------------------------

def flag_bits(k: int) -> int:
    """A word unit's flags port: its k/32 flags padded to whole 16-bit words (the cap-flags row's)."""
    return -(-k // (16 * SLICE)) * 16


def _word_unit_sig(S_):
    _d, ng = _promoted(S_.DEV, S_.K)
    k = S_.K
    return ((("a", Array(k, pc8.E4M3)), ("pa", pc8.P_ROW), ("b", Array(k, pc8.E4M3)), ("pb", pc8.P_ROW),
             ("cert", Array(ng - 1, Array(2, F32)))),
            Tuple(F32, F32, Array(flag_bits(k), BIT), I32, I32, OK))


def _word_unit_body(B, S_, a, pa, b, pb, cert):
    e, k = pc4._Emit(B), S_.K
    out = B.call(templates(k, S_.DEV)["word"], a, b, cert)
    flags = [out[1][t] for t in range(k // SLICE)]
    pad = [e.c(1, 0)] * (flag_bits(k) - len(flags))
    return tuple_of(out[0], _peel(e, S_.DEV, out[0], pa, pb), array_of(flags + pad), _wire(e, 32, _count(e, flags)),
                    out[2], pc4.ok_word(e, out[3]))


Pc8CapWordUnitDef = CompositeDefinition(
    "Pc8CapWordUnit", 1, ("K", "DEV"), _word_unit_sig, _word_unit_body,
    doc="One word of the split cap: `Pc8CapWord`'s C̃ and U (`Pc8Tile`'s peel from C̃), its flags padded to whole 16-bit "
        "words with zero bits, their count, its later +0 promotions, and good as an ok word.",
    conformance="protocol-defined")


#: a tile row's cap-chunks row as 32-bit words: its chunk digests (TN <= 64, pc8's tile), then zeros
CHUNK_WORDS = L.CAP_ROW_WORDS // 2


def _columns(tn: int) -> int:
    if not 0 < tn <= L.CAP_CHUNKS * C.CHUNK:
        raise DomainError(f"Pearl-C's cap circuit takes at most {L.CAP_CHUNKS * C.CHUNK} columns a tile: TN = {tn}")
    return tn


def _row_digest_body(B, S_, c, u):
    e = H.Emit(B)
    ds = L.chunk_digests(e, c, u, S_.TN)
    return e.array(ds + [0] * (CHUNK_WORDS - len(ds)))


Pc8CapRowDigestDef = CompositeDefinition(
    "Pc8CapRowDigest", 1, ("TN",),
    lambda S_: ((("c", Array(_columns(S_.TN), F32)), ("u", Array(S_.TN, F32))), Array(CHUNK_WORDS, W.U32)),
    _row_digest_body,
    doc="The split cap's digests of one tile row: `leaves.PearlTileDigest`'s TurboSHAKE128 (D = 0x21) of each 32-column "
        "chunk of its TN words' C̃ and U, two digests (the second zero when TN <= 32), then zero words to its cap-chunks "
        "row's 64.", conformance="protocol-defined")


def _tile_sig(S_):
    _columns(S_.TN)
    return (("d", Array(S_.TM, Array(CHUNK_WORDS, W.U32))),), L.DIGEST


def _tile_body(B, S_, d):
    e, n = H.Emit(B), 8 * -(-S_.TN // C.CHUNK)
    return e.array(L.leaf_digest(e, [d[r][w] for r in range(S_.TM) for w in range(n)]))


Pc8CapTileDef = CompositeDefinition(
    "Pc8CapTile", 1, ("TM", "TN"), _tile_sig, _tile_body,
    doc="The split cap's tile: the tile digest h0 (`leaves.PearlTileDigest`'s TurboSHAKE128, D = 0x22) of its TM rows' "
        "chunk digests (`Pc8CapRowDigest`), in order.", conformance="protocol-defined")


#: a strip's admission bits in its cap-rows row: the scheme's tile rows, so TM <= 64
STRIP_BITS = pc8.TILE[0]
#: the zero words that end a cap-rows row, after its admission bits and its bound's two words
STRIP_TAIL = L.CAP_ROW_WORDS - STRIP_BITS // 16 - 2
_TAIL = Array(STRIP_TAIL, Value(16))


def _strip_sig(S_):
    _cap_constants(S_.K, S_.TM, S_.TN, S_.N, S_.RHO, S_.DEV)
    if S_.TM > STRIP_BITS:
        raise DomainError(f"Pearl-C's cap circuit takes at most {STRIP_BITS} rows a tile: TM = {S_.TM}")
    return (("oka", Array(S_.TM, OK)),), Tuple(Array(STRIP_BITS, BIT), I32, _TAIL)


def _strip_body(B, S_, oka):
    e = pc4._Emit(B)
    table = _cap_constants(S_.K, S_.TM, S_.TN, S_.N, S_.RHO, S_.DEV)[-1]
    oks = [e.call(Bf16GtStrict, oka[i], 0) for i in range(S_.TM)]
    pad = [e.c(1, 0)] * (STRIP_BITS - S_.TM)
    return tuple_of(array_of(oks + pad), _wire(e, 32, _bound(e, table, _count(e, oks))),
                    array_of([e.c(16, 0)] * STRIP_TAIL))


Pc8CapStripDef = CompositeDefinition(
    "Pc8CapStrip", 1, ("K", "TM", "TN", "N", "RHO", "DEV"), _strip_sig, _strip_body,
    doc="The split cap's strip of TM rows of A, once for every tile in it: its rows' admission bits, padded to 64 with zero "
        "bits, the cap's bound at its passing rows' count, ⌊|R|·|C|·P/(RHO·g)⌋, and zero words to its cap-rows row's 64.",
    conformance="protocol-defined")


def _row_tally_sig(S_):
    k, tn = S_.K, _columns(S_.TN)
    if k <= 0 or k % SLICE:
        raise DomainError(f"Pearl-C's cap circuit sums whole atom blocks of {SLICE}: k = {k}")
    return ((("flags", Array(tn, Array(flag_bits(k), BIT))), ("atoms", Array(tn, I32)), ("zeros", Array(tn, I32)),
             ("good", Array(tn, OK)), ("sd", Array(k // SLICE, I32))), Tuple(I32, I32, I32, OK))


def _row_tally_body(B, S_, flags, atoms, zeros, good, sd):
    e, tn = pc4._Emit(B), S_.TN
    row_atoms = row_zeros = 0
    row_good = 1
    for w in range(tn):
        row_atoms = _add(e, row_atoms, atoms[w])
        row_zeros = _add(e, row_zeros, zeros[w])
        row_good = _and(e, row_good, e.call(Bf16GtStrict, good[w], 0))
    elems = _row_elems(e, [flags[w] for w in range(tn)], sd, S_.K)
    return tuple_of(row_atoms, row_zeros, elems, pc4.ok_word(e, row_good))


Pc8CapRowTallyDef = CompositeDefinition(
    "Pc8CapRowTally", 1, ("K", "TN"), _row_tally_sig, _row_tally_body,
    doc="The split cap's sums over one tile row: its TN words' flag counts and later +0 promotions, its forming elements "
        "(per atom block 32 when every word flags it, else the row's salt-dead count), and whether every word is good, as "
        "an ok word.", conformance="protocol-defined")


def _tally_sig(S_):
    tm = S_.TM
    _cap_constants(S_.K, tm, S_.TN, S_.N, S_.RHO, S_.DEV)
    return ((("atoms", Array(tm, I32)), ("zeros", Array(tm, I32)), ("elems", Array(tm, I32)), ("good", Array(tm, OK)),
             ("mask", Array(STRIP_BITS, BIT)), ("bound", I32), ("tail", _TAIL)), OK)


def _tally_body(B, S_, atoms, zeros, elems, good, mask, bound, _tail):
    return _tally(B, S_, atoms, zeros, elems, good, mask, bound, None)


def _row_cap_tally_body(B, S_, atoms, zeros, elems, good, mask, bound, _tail):
    return _tally(B, S_, atoms, zeros, elems, good, mask, bound, _row_constants(S_.K, S_.TN, S_.N, S_.RHOR, S_.DEV))


def _tally(B, S_, atoms, zeros, elems, good, mask, bound, row_consts):
    e = pc4._Emit(B)
    consts = _cap_constants(S_.K, S_.TM, S_.TN, S_.N, S_.RHO, S_.DEV)
    sums, bad = (0, 0, 0), 0
    for i in range(S_.TM):
        ok = mask[i]
        sums = _masked_add(e, sums, ok, (atoms[i], zeros[i], elems[i]))
        bad = _or(e, bad, _and(e, ok, _not(e, e.call(Bf16GtStrict, good[i], 0))))
        if row_consts is not None:
            bad = _or(e, bad, _row_over(e, row_consts, ok, atoms[i], zeros[i], elems[i]))
    return pc4.ok_word(e, _and(e, _not(e, bad), _within(e, consts, sums, bound)))


Pc8CapTallyDef = CompositeDefinition(
    "Pc8CapTally", 1, ("K", "TM", "TN", "N", "RHO", "DEV"), _tally_sig, _tally_body,
    doc="The split cap's tally: `Pc8TileCap`'s verdict at ρ = 1/RHO on a unit of N columns as an ok word (every passing "
        "row's words are good, and the cap's inequality holds), from its rows' sums (`Pc8CapRowTally`) and its strip's "
        "admission bits and bound (`Pc8CapStrip`; its row's zero words unread).", conformance="protocol-defined")

Pc8CapTallyDefV2 = CompositeDefinition(
    "Pc8CapTally", 2, ("K", "TM", "TN", "N", "RHO", "RHOR", "DEV"), lambda S_: _row_capped(S_, _tally_sig),
    _row_cap_tally_body,
    doc="`Pc8CapTally` v1 with the per-row cap: `Pc8TileCap` v2's verdict as an ok word, v1's and every admitted row's own "
        "sums (`Pc8CapRowTally`) within ρr = 1/RHOR times its cells' share of `creditDevRev1` (the tile's test on one "
        "row).", conformance="protocol-defined")


@lru_cache(maxsize=16)
def unit_templates(k: int, dev: str = "sm120", tile: tuple[int, int] = (1, 1), n: int | None = None,
                   rho: int | None = None, row_rho: int | None = None) -> dict[str, SpecializedDefinition]:
    """The split's unit templates by role (`ROLES`): pc8's row of B, the row of A with its salt-dead counts and the word
    unit at depth ``k`` on ``dev``, and the strip, a tile row's digests and sums, the tile and its tally at ``tile``, the
    unit's ``n`` (the tile's columns by default) and ρ = 1/``rho`` (the device's `cap` by default); with ``row_rho`` the
    tally is `Pc8CapTally` v2, with the per-row cap at ρr = 1/``row_rho``."""
    tm, tn = tile
    capped = dict(K=k, TM=tm, TN=tn, N=n or tn, RHO=_rho(dev, rho), DEV=dev)
    return {"row_b": pc8.templates(k, pc8.TILE, dev)["row_b"], "row_a": bind(Pc8RowACapDef, K=k, DEV=dev),
            "strip": bind(Pc8CapStripDef, **capped), "word": bind(Pc8CapWordUnitDef, K=k, DEV=dev),
            "row_digest": bind(Pc8CapRowDigestDef, TN=tn), "row_tally": bind(Pc8CapRowTallyDef, K=k, TN=tn),
            "tile": bind(Pc8CapTileDef, TM=tm, TN=tn),
            "tally": (bind(Pc8CapTallyDef, **capped) if _row_rho(row_rho) is None
                      else bind(Pc8CapTallyDefV2, **capped, RHOR=row_rho))}


def _call_cap_units(B, k: int, tile: tuple[int, int], dev: str, a, ea, fa, fb, bc, pb, cert, row_rho: int | None = None):
    """The split call's units: (the tile digests and the tallies' verdicts, by tile row and column; ok)."""
    tm, tn = tile
    m, n = len(a), len(bc)
    cols = n // tn
    tp = unit_templates(k, dev, tile, n, row_rho=row_rho)
    rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
    ac, pa, oka, sd = (array_of([rows[i][f] for i in range(m)]) for f in range(4))
    strips = B.batch(tp["strip"], oka.reshape(Array(m // tm, Array(tm, oka.type.elem))), axes=(0,))
    digests, verdicts = [], []
    for rs in range(m // tm):
        strip = range(rs * tm, (rs + 1) * tm)
        words = {i: B.batch(tp["word"], ac[i], pa[i], bc, pb, cert[i], axes=(None, None, 0, 0, 0)) for i in strip}
        cells = lambda i, f: array_of([array_of([words[i][cs * tn + j][f] for j in range(tn)])  # noqa: E731
                                       for cs in range(cols)])
        chunks = {i: B.batch(tp["row_digest"], cells(i, 0), cells(i, 1), axes=(0, 0)) for i in strip}
        sums = {i: B.batch(tp["row_tally"], cells(i, 2), cells(i, 3), cells(i, 4), cells(i, 5), sd[i],
                           axes=(0, 0, 0, 0, None)) for i in strip}
        by_tile = lambda got, *f: array_of([array_of([got[i][cs][f[0]] if f else got[i][cs] for i in strip])  # noqa: E731
                                            for cs in range(cols)])
        tiles = B.batch(tp["tile"], by_tile(chunks), axes=(0,))
        tallies = B.batch(tp["tally"], *(by_tile(sums, f) for f in range(4)), *(strips[rs][f] for f in range(3)),
                          axes=(0, 0, 0, 0, None, None, None))
        digests.append(array_of([tiles[cs] for cs in range(cols)]))
        verdicts.append(array_of([tallies[cs] for cs in range(cols)]))
    return array_of(digests), array_of(verdicts), oka


def _verdicts(S_):
    return Array(S_.M // S_.TM, Array(S_.N // S_.TN, OK))


def _cert(S_):
    _d, ng = _promoted(S_.DEV, S_.K)
    return ("cert", Array(S_.M, Array(S_.N, Array(ng - 1, Array(2, F32)))))


Pc8CallCapDef = CompositeDefinition(
    "Pc8CallCap", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (pc8._call_sig(S_)[0] + (_cert(S_),), Tuple(pc4.DIGESTS(S_), _verdicts(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*_call_cap_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a)),
    doc="`Pc8CallHidden` with the per-tile cap, cut into a unit per word, the tile and its tally: each tile's digest of C̃ "
        "and U and its cap verdict by tile row and column (the verifier accepts a tile only when its digest is the "
        "miner's and its verdict is ok), and each A row's admission; every word's certificates are the prover's.",
    conformance="protocol-defined")


def _check_cap_body(B, S_, a, ea, b, eb, fa, fb, g, cert):
    bc, pb, ok_b = pc8._weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    return tuple_of(*_call_cap_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, cert), ok_b)


Pc8CheckCapDef = CompositeDefinition(
    "Pc8CheckCap", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (pc8._check_sig(S_)[0] + (_cert(S_),),
                Tuple(pc4.DIGESTS(S_), _verdicts(S_), Array(S_.M, OK), Array(S_.N, OK))),
    _check_cap_body, doc="`Pc8CheckHidden` with the per-tile cap: `Pc8CallCap`'s outputs, then each B row's admission.",
    conformance="protocol-defined")


def _check_row_cap_body(B, S_, a, ea, b, eb, fa, fb, g, cert):
    bc, pb, ok_b = pc8._weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    return tuple_of(*_call_cap_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, cert, S_.RHOR), ok_b)


Pc8CallCapDefV2 = CompositeDefinition(
    "Pc8CallCap", 2, ("M", "K", "N", "TM", "TN", "RHOR", "DEV"), Pc8CallCapDef.signature,
    lambda B, S_, *a: tuple_of(*_call_cap_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a, S_.RHOR)),
    doc="`Pc8CallCap` v1 with the per-row cap: each tile's tally is `Pc8CapTally` v2, which also refuses a tile with an "
        "admitted row over ρr = 1/RHOR times its cells' share of the credit.", conformance="protocol-defined")

Pc8CheckCapDefV2 = CompositeDefinition(
    "Pc8CheckCap", 2, ("M", "K", "N", "TM", "TN", "RHOR", "DEV"), Pc8CheckCapDef.signature, _check_row_cap_body,
    doc="`Pc8CheckCap` v1 with the per-row cap: `Pc8CallCap` v2's outputs, then each B row's admission.",
    conformance="protocol-defined")

#: each split unit's ports (`pc8.PORTS`' classes), and ``public``: outputs the verifier reads, in no row (the tally's
#: verdict)
PORTS = {
    "row_b": pc8.PORTS["row_b"],
    "row_a": {"rows": (("fp32", ROLE_X, ("x",)), ("fp8-p", ROLE_X, (0, 1)), ("ok", ROLE_X, (2,)), ("sd", ROLE_X, (3,))),
              "regions": {"e": "ea", "fa": "fa", "fb": "fb"}, "advice": ()},
    "word": {"rows": (("fp8-p", ROLE_X, ("a", "pa")), ("fp8-p", ROLE_W, ("b", "pb")), ("cap-word", ROLE_X, (0, 1)),
                      ("cap-flags", ROLE_X, (2, 3, 4, 5))), "regions": {}, "advice": ADVICE},
    "strip": {"rows": (("ok", ROLE_X, ("oka",)), ("cap-rows", ROLE_X, (0, 1, 2))), "regions": {}, "advice": ()},
    "row_digest": {"rows": (("cap-word", ROLE_X, ("c", "u")), ("cap-chunks", ROLE_X, (0,))), "regions": {}, "advice": ()},
    "row_tally": {"rows": (("cap-flags", ROLE_X, ("flags", "atoms", "zeros", "good")), ("sd", ROLE_X, ("sd",)),
                           ("cap-sums", ROLE_X, (0, 1, 2, 3))), "regions": {}, "advice": ()},
    "tile": {"rows": (("cap-chunks", ROLE_X, ("d",)), ("tile", ROLE_X, (0,))), "regions": {}, "advice": ()},
    "tally": {"rows": (("cap-sums", ROLE_X, ("atoms", "zeros", "elems", "good")),
                       ("cap-rows", ROLE_X, ("mask", "bound", "tail"))), "regions": {}, "advice": (), "public": (0,)},
}
#: the split's roles on each side, in the order its root body first uses them
ROLES = {"weight": ("row_b",), "call": ("row_a", "strip", "word", "row_digest", "row_tally", "tile", "tally"),
         "check": ("row_b", "row_a", "strip", "word", "row_digest", "row_tally", "tile", "tally")}
ROLE_OF = {id(Pc8RowACapDef): "row_a", id(Pc8CapStripDef): "strip", id(Pc8CapWordUnitDef): "word",
           id(Pc8CapRowDigestDef): "row_digest", id(Pc8CapRowTallyDef): "row_tally", id(Pc8CapTileDef): "tile",
           id(Pc8CapTallyDef): "tally", id(Pc8CapTallyDefV2): "tally"}


def reads(tile: tuple[int, int]) -> dict[str, dict[str, int]]:
    """Each split unit's rows read per port (`leaves.statement_rows`' ``reads``) at ``tile``."""
    tm, tn = tile
    return {"row_b": {}, "row_a": {}, "strip": {"oka": tm}, "word": {}, "row_digest": {"c": tn, "u": tn},
            "row_tally": {"flags": tn, "atoms": tn, "zeros": tn, "good": tn}, "tile": {"d": tm},
            "tally": {"atoms": tm, "zeros": tm, "elems": tm, "good": tm}}


def check_shape(shape) -> None:
    """Refuse a `pc8.Shape` with ``cap`` outside the split: the cap is in the hiding Programs, on a device with a salt-dead
    bit and a window of 4 atoms, at most 64 rows a tile, and its integer form fits 31 bits at the shape's tile and n."""
    if not shape.hidden:
        raise DomainError(f"Pearl-C's cap is in the hiding Programs: {shape} has cap and not hidden")
    if DEVICES[shape.dev].atom.name not in SALT_DEAD:
        raise DomainError(f"Pearl-C's cap circuit has no salt-dead bit for {shape.dev!r}'s atom")
    if shape.tile[0] > STRIP_BITS:
        raise DomainError(f"Pearl-C's cap circuit takes at most {STRIP_BITS} rows a tile: {shape}")
    _columns(shape.tile[1])
    _cap_constants(shape.k, *shape.tile, shape.n, _rho(shape.dev, None), shape.dev)
    if _row_rho(shape.row_rho) is not None:
        _row_constants(shape.k, shape.tile[1], shape.n, shape.row_rho, shape.dev)


def _bind_cap(v1: CompositeDefinition, v2: CompositeDefinition, shape) -> SpecializedDefinition:
    at = dict(M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0], TN=shape.tile[1], DEV=shape.dev)
    return bind(v1, **at) if shape.row_rho is None else bind(v2, **at, RHOR=shape.row_rho)


def call(shape) -> SpecializedDefinition:
    """`Pc8CallCap` at ``shape``: v1, or v2 with its per-row cap (``Shape.row_rho``)."""
    return _bind_cap(Pc8CallCapDef, Pc8CallCapDefV2, shape)


def check(shape) -> SpecializedDefinition:
    """`Pc8CheckCap` at ``shape``: v1, or v2 with its per-row cap (``Shape.row_rho``)."""
    return _bind_cap(Pc8CheckCapDef, Pc8CheckCapDefV2, shape)


def row_rho(scheme, k: int) -> int | None:
    """The per-row cap ``scheme``'s tile rule applies at depth ``k``, as RHOR (ρr = 1/RHOR), or None: under a row-capped
    hashing format (`HashFormat.row_cap`, -h3) `PearlC.row_cap_at(k)`, 4ρ at a census k below k_min and ρ elsewhere, as
    `clear_work.check_opened` applies it; under -h0, which caps the tile alone, none."""
    if not C.HASH_FORMATS[scheme.hashing].row_cap:
        return None
    at = Fraction(scheme.row_cap_at(k))
    if at.numerator != 1:
        raise DomainError(f"{scheme.name}: the cap circuit needs a row cap 1/RHOR at k = {k}, not {at}")
    return at.denominator


def served(scheme, shape: Shape) -> "pc8.Shape":
    """``scheme``'s unit ``shape`` as the hiding call with the cap: its device and tile, at its device's ρ, and with
    `row_rho`'s per-row cap.  Refuses a unit outside the scheme's domain (`PearlC.check`) and a scheme the cap circuit
    doesn't price (forming v0, or another device's prices)."""
    d = scheme.device
    if scheme.forming != "v1" or scheme.prices != d.prices or DEVICES.get(d.name) != d:
        raise DomainError(f"{scheme.name}: the cap circuit prices forming v1 at its device record's own prices")
    scheme.check(Matmul(0, shape, 0))
    out = pc8.Shape(shape.m, shape.k, shape.n, tile=scheme.tile, dev=d.name, hidden=True, cap=True,
                    row_rho=row_rho(scheme, shape.k))
    out.check()
    return out


__all__ = ["ADVICE", "CHUNK_WORDS", "FLAT_E", "PORTS", "Pc8CallCapDef", "Pc8CallCapDefV2", "Pc8CapRowDigestDef",
           "Pc8CapRowTallyDef", "Pc8CapStripDef", "Pc8CapTallyDef", "Pc8CapTallyDefV2", "Pc8CapTileDef", "Pc8CapWordDef",
           "Pc8CapWordUnitDef", "Pc8CheckCapDef", "Pc8CheckCapDefV2", "Pc8RowACapDef", "Pc8TileCapDef", "Pc8TileCapDefV2",
           "ROLES", "ROLE_OF", "SALT_DEAD", "STRIP_BITS", "STRIP_TAIL", "WINDOWS", "call", "certificates", "check",
           "check_shape", "flag_bits", "reads", "row_rho", "served", "templates", "unit_templates"]
