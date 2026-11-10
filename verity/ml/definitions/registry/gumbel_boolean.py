"""The Gumbel select on bits: the next version of each word Definition of `sampling`'s noise and select, the same function as
And/Xor/Not circuits.

    GumbelStreamKey_v2      (seed, pos: Array<64>) -> Array<32>                          word view: GumbelStreamKey_v1
    NvLogf_v3               (a: Array<32>) -> Array<32>                                  word view: NvLogf_v2
    GumbelSelectF32_v2{V}   (logits: Array<V, Array<32>>, seed, pos: Array<64>, temp: Array<32>) -> Array<32>
                                                                                         word view: GumbelSelectF32_v1{V}

(`Array<w>` is ``Array<w, Value<1>>``.) Each scalar piece is a builder over `forms`, traced (`trace.definition`): Philox4x32-10
(`philox_round`; a product by a round constant costs only its adders), Triton's ``tl.rand`` (`tl_rand`), libdevice's
``__nv_log1pf`` and ``__nv_logf`` with FTZ (`nv_log1pf`, `nv_logf`: `softmax`'s FTZ f32 operations step for step, the
toward-zero add `fp.f32_add`'s ``rz``; ``f32(e) * 2^-23`` and ``fma(f32(e), 2^-23, i0)`` written as the integers they equal
exactly), every NaN result 0x7FFFFFFF as `NvLogf_v2`'s: the noise never reaches one, so it is `GumbelNoiseLane_v1`'s.

A product by a constant is linear in its operand's bits, so through Philox's rounds every bit's form (`forms.L`) grows: one
trace of ten rounds holds gigabytes of them. The stream key and the lane are therefore Calls of pieces traced on their own,
cut where the next piece reads every output, so no gate one trace would prune is kept: the key's rounds 0-4 with the key
words rounds 5-9 read, and rounds 5-9 (`GumbelStreamKeyHead_v1`, `GumbelStreamKeyTail_v1`), and the lane's rounds 0-2, 3-7
(the counter words round 9's word 0 reads), and 8-9 with the uniform, the logs and the add (`GumbelLaneHead_v1`,
`GumbelLaneRounds_v1`, `GumbelLaneTail_v1`).

`GumbelSelectF32_v2{V}` is `GumbelSelectF32_v1{V}` (the first index of the maximum of ``logits[v] + g(key)[v]``, or of the
logits when ``temp`` is zero; NaN never taken, index 0 when lane 0 is a NaN) as a chain of Calls a partition query cuts at any
vocabulary size:

    GumbelStreamKey_v2             (seed, pos) -> the row's stream key
    GumbelSetup_v1{N}              (key, temp) -> the key's lane values (`lane_keys`, `round2_lows` for N = min(V, 64)
                                   offsets) and ``temp != 0``: every value the lanes share, so no two lanes compute one
    GumbelNoise_v1{V}              the noisy row: lanes in groups of G = 64 (`GumbelNoiseGroup_v1{G}`), each lane one member Call
                                   (`GumbelLane_v1`); every level's nodes read only its parameters (`verity.ml.circuits.cut.separable`)
    F32ArgmaxStage_v1{V,L}         the first maximum of the noisy row (`sampling_boolean`'s tree over f32 keys)

A served Program calls them at its root (`gumbel_select_calls`), as `token_select_calls` the greedy path's stages. Lane
``v = lo + 64 hi`` reads its index only through Philox round 0's ``v * PHILOX_ROUND_A``, which the lane adds up from two
constant operands (`lane_products`): ``lo * PHILOX_ROUND_A``, which the groups share, and ``hi * PHILOX_ROUND_A``, which the
lanes of a group share. Round 2's counter word 2 has low six bits that read only the key and ``lo``, so their part of its
product (`round2_lows`) is the setup's, one per offset. Every gate of a lane then reads its own lane, so the members of a level
compute no value twice (a product of the index itself would compute the partial sums of ``hi``'s bits once per lane of the
group).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, Explicit, concat
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, and_reduce, const_bits, multiply, mux, or_reduce, sub
from verity.ml.circuits.ops.boolean.scalar import gt_u, select
from verity.ml.circuits.ops.boolean.softmax import f32_add_ftz, f32_fma_ftz, f32_mul_ftz, ftz
from verity.ml.circuits.boolean.trace import definition
from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import topp_word_gates as TG
from verity.ml.definitions.registry.sampling_boolean import F32ArgmaxStage, G, GB, argmax_calls, traced_family

__all__ = ["LANE_ROUNDS", "GumbelLane", "GumbelLaneHead", "GumbelLaneRounds", "GumbelLaneTail", "GumbelNoise", "GumbelNoiseGroup", "GumbelSelectF32",
           "GumbelSetup", "GumbelStreamKey", "GumbelStreamKeyHead", "GumbelStreamKeyTail", "NvLogf", "gumbel_noise",
           "gumbel_select_calls", "lane_keys", "lane_products", "lane_rounds", "nv_log1pf", "nv_logf", "philox_round",
           "round2_lows", "stream_keys", "tl_rand"]

BIT = BL.BIT
U32 = Array(32, BIT)
F32 = U32
I32 = U32
I64 = Array(64, BIT)
#: Philox's counter (four words), and its words 0, 2 and 3
C4 = Array(4, U32)
C3 = Array(3, U32)
#: the rounds whose low key word a lane reads (`lane_keys`): round 0's is the key, which reaches only round 1's product, and
#: round 8's reaches only its word 0, which round 9's word 0 does not read
LANE_ROUNDS = (1, 2, 3, 4, 5, 6, 7, 9)
#: the lane values `lane_keys` returns: those rounds' key words and round 1's product (low word, high word)
KS = Array(len(LANE_ROUNDS), U32)
KA = Array(2, U32)
K2, K4, K5 = Array(2, U32), Array(4, U32), Array(5, U32)
#: a lane index is ``v = lo + 64 hi``; a lane reads ``v * PHILOX_ROUND_A`` as ``lo_a + 2^6 hi_a`` (`lane_products`), ``lo_a`` LOB
#: bits and ``hi_a`` (mod 2^58) HAB bits
LOB = 32 + GB
HAB = 64 - GB
LA = Array(LOB, BIT)
HA = Array(HAB, BIT)
#: a `round2_lows` entry: six counter bits times PHILOX_ROUND_B
LB = Array(GB + 32, BIT)

INF = 0x7F800000
NEG_INF = 0xFF800000
NEG_ZERO = 0x80000000
Z32 = [ZERO] * 32


def _c(value: int, width: int = 32):
    return const_bits(value & ((1 << width) - 1), width)


def _xor(a, b):
    return [x ^ y for x, y in zip(a, b)]


# --- Philox4x32-10 -----------------------------------------------------------------------------------------------------------

def _times(C, x, k: int):
    """The 64-bit product of the 32-bit word ``x`` and the constant ``k``: no partial-product ANDs."""
    return multiply(C, list(x), _c(k))


def philox_round(C, c, k0, k1, pa=None, pb=None):
    """One round of `sampling.philox4x32_10_word0` on the counter words ``c``; ``pa`` is ``c[0] * PHILOX_ROUND_A`` and ``pb``
    is ``c[2] * PHILOX_ROUND_B`` when the caller has them."""
    if pb is None:
        pb = _times(C, c[2], SM.PHILOX_ROUND_B)
    if pa is None:
        pa = _times(C, c[0], SM.PHILOX_ROUND_A)
    return [_xor(_xor(pb[32:64], c[1]), k0), pb[:32], _xor(_xor(pa[32:64], c[3]), k1), pa[:32]]


def stream_keys(C, seed):
    """The round keys of `sampling.stream_key`'s Philox (``tl.randint(seed, pos)``): ``(seed_lo + r PHILOX_KEY_A, seed_hi +
    r PHILOX_KEY_B)``, r = 0..9."""
    lo, hi = list(seed[:32]), list(seed[32:])
    return [(lo, hi)] + [(add(C, lo, _c(r * SM.PHILOX_KEY_A), width=32), add(C, hi, _c(r * SM.PHILOX_KEY_B), width=32))
                         for r in range(1, 10)]


def stream_key_head(C, seed, pos):
    """Rounds 0-4 on the counter (pos_lo, pos_hi, 0, 0), and the key words of rounds 5-9 that word 0 reads: the low ones of
    rounds 5, 6, 7 and 9 and the high ones of rounds 5-8."""
    keys = stream_keys(C, seed)
    c = [list(pos[:32]), list(pos[32:]), Z32, Z32]
    for k in keys[:5]:
        c = philox_round(C, c, *k)
    return c + [keys[r][0] for r in (5, 6, 7, 9)] + [keys[r][1] for r in (5, 6, 7, 8)]


def stream_key_tail(C, c0, c1, c2, c3, *k):
    """Rounds 5-9: word 0. Round 8's low key word reaches only its word 0, and round 9's high one only its word 2, neither of
    which round 9's word 0 reads, so zeros stand in for them."""
    c = [c0, c1, c2, c3]
    for k0, k1 in zip(k[:3] + (Z32,) + k[3:4], k[4:] + (Z32,)):
        c = philox_round(C, c, k0, k1)
    return c[0]


def lane_keys(C, key):
    """What every lane of the stream ``key`` computes from the key alone: the round keys ``key + r * PHILOX_KEY_A`` (r in
    `LANE_ROUNDS`) and ``key * PHILOX_ROUND_A``, round 1's product (its counter word 0 is the key)."""
    ks = [add(C, key, _c(r * SM.PHILOX_KEY_A), width=32) for r in LANE_ROUNDS]
    return ks, _times(C, key, SM.PHILOX_ROUND_A)


def round2_lows(C, ka, n: int = G):
    """For each offset ``lo`` < ``n`` in a group, ``c2[0:6] * PHILOX_ROUND_B`` (38 bits), where ``c2`` is round 2's counter
    word 2: ``ka_hi ^ (v * PHILOX_ROUND_A mod 2^32) ^ PHILOX_KEY_B``, whose low six bits read only ``ka`` and ``lo`` (``v = lo +
    64 hi``)."""
    out = []
    for lo in range(n):
        c = (lo * SM.PHILOX_ROUND_A) ^ SM.PHILOX_KEY_B
        out.append(_times(C, [NOT(ka[32 + i]) if c >> i & 1 else ka[32 + i] for i in range(GB)], SM.PHILOX_ROUND_B))
    return out


def lane_rounds(C, c, k0s, first: int, va=None, ka=None, lb=None):
    """Rounds ``first``.. of ``philox4x32_10_word0(v, 0, 0, 0, key, 0)`` (`sampling._rand_lane`) on the counter words ``c``,
    round r's key ``(k0s[r - first], r * PHILOX_KEY_B)``: round 0's product is ``va`` = ``v * PHILOX_ROUND_A`` (the lane index
    is read only through it), round 1's is ``ka`` (its counter word 0 is the key), and round 2's is ``lb + 2^6 (c2[6:32] *
    PHILOX_ROUND_B)``, ``lb`` the lane's `round2_lows` entry."""
    for i, k0 in enumerate(k0s):
        r = first + i
        pb = add(C, list(lb), [ZERO] * GB + _times(C, c[2][GB:], SM.PHILOX_ROUND_B), width=64) if r == 2 else None
        c = philox_round(C, c, k0, _c(r * SM.PHILOX_KEY_B), pa=va if r == 0 else ka if r == 1 else None, pb=pb)
    return c


# --- tl.rand and the libdevice logs ------------------------------------------------------------------------------------------

def tl_rand(C, w):
    """`sampling._rand_lane` from the Philox word ``w``: ``-x - 1`` (``~x``) for a negative i32, its f32 (RN), times
    ``TL_RAND_SCALE`` (FTZ), at least ``TL_RAND_MIN``."""
    xi = [b ^ w[31] for b in w[:31]]
    f = fp.round_f32(C, ZERO, _c(157, 9), 0, xi, ZERO, underflow=False)
    u = f32_mul_ftz(C, f, _c(SM.TL_RAND_SCALE_BITS))
    return select(C, gt_u(C, _c(SM.TL_RAND_MIN_BITS, 31), u[:31]), _c(SM.TL_RAND_MIN_BITS), u)


def _int_f32(C, k):
    """The f32 word of the two's-complement integer ``k`` (LSB first, at most 24 bits): exact, +0 for zero."""
    mag = fp.neg_if(C, k, k[-1])
    return fp.round_f32(C, k[-1], _c(126 + len(k), 9), 0, mag, ZERO, underflow=False)


def _canon_nan(C, r):
    nan = C.AND(and_reduce(C, r[23:31]), or_reduce(C, r[:23]))
    return select(C, nan, _c(SM.LOG_NAN), r)


def _poly(C, coeffs, f):
    """libdevice's Horner tail: ``((c0 f + c1) f + ... ) f - 0.5``, times f, plus f (FTZ FMAs)."""
    r = f32_fma_ftz(C, _c(coeffs[0]), f, _c(coeffs[1]))
    for k in coeffs[2:]:
        r = f32_fma_ftz(C, r, f, _c(k))
    r = f32_fma_ftz(C, r, f, _c(0xBF000000))
    r = f32_mul_ftz(C, r, f)
    return f32_fma_ftz(C, r, f, f)


def nv_logf(C, a):
    """`sampling.nv_logf_gpu` (`NvLogf_v2`): libdevice ``__nv_logf`` with FTZ, every NaN result 0x7FFFFFFF."""
    s, e = a[31], a[23:31]
    nan = C.AND(and_reduce(C, e), or_reduce(C, a[:23]))
    small = mux(C, s, [NOT(nan)], [NOT(or_reduce(C, e))])[0]      # a < FLT_MIN as values (every negative, +0, subnormals)
    a2 = select(C, small, f32_mul_ftz(C, a, _c(SM.TWO_P23_BITS)), a)
    e9 = sub(C, a2, _c(1059760811), 32)[0][23:]                   # (a2 - 0x3F2AAAAB) & 0xFF800000, as a 9-bit integer
    m = a2[:23] + sub(C, a2[23:], e9, 9)[0]
    k = sub(C, e9 + [e9[8]], [small, small, small, ZERO, small] + [ZERO] * 5, 10)[0]   # fma(f32(e), 2^-23, small ? -23 : 0)
    f = f32_add_ftz(C, m, _c(0xBF800000))
    r = f32_fma_ftz(C, _int_f32(C, k), _c(SM.LN2_BITS), _poly(C, SM.LOGF_POLY, f))
    s2, e2_max, m2_any = a2[31], and_reduce(C, a2[23:31]), or_reduce(C, a2[:23])
    big = OR(C, s2, e2_max)                                          # a2 >= 0x7F800000: -inf + inf (a NaN) unless a2 = +inf
    pinf = C.AND(C.AND(NOT(s2), e2_max), NOT(m2_any))
    r = select(C, big, select(C, pinf, _c(INF), _c(SM.LOG_NAN)), r)
    r = select(C, NOT(OR(C, or_reduce(C, a2[23:31]), m2_any)), _c(NEG_INF), r)
    return _canon_nan(C, r)


def nv_log1pf(C, a):
    """`sampling.nv_log1pf_gpu`: libdevice ``__nv_log1pf`` with FTZ (``__fadd_rz(a, 1)`` toward zero), every NaN result
    0x7FFFFFFF."""
    sw = ftz(C, fp.f32_add(C, ftz(C, a), _c(SM.ONE_BITS), rz=True))
    e9 = sub(C, sw, _c(1061158912), 32)[0][23:]                   # (s - 0x3F400000) & 0xFF800000, as a 9-bit integer
    m1 = a[:23] + sub(C, a[23:], e9, 9)[0]
    t = [ZERO] * 23 + sub(C, _c(1082130432 >> 23, 9), e9, 9)[0]
    u = f32_add_ftz(C, m1, f32_fma_ftz(C, _c(0x3E800000), t, _c(0xBF800000)))
    r = f32_fma_ftz(C, _int_f32(C, e9), _c(SM.LN2_BITS), _poly(C, SM.LOG1PF_POLY, u))   # f32(e) * 2^-23 = e9 exactly
    s, e_max = a[31], and_reduce(C, a[23:31])
    m_any = or_reduce(C, a[:23])
    _, over_one = sub(C, _c(0x3F800000, 31), a[:31], 31)             # |a| > 1
    special = OR(C, C.AND(NOT(s), e_max), C.AND(s, over_one))      # a >= 0x7F800000 and i32(a) > i32(0xBF800000)
    pinf = C.AND(C.AND(NOT(s), e_max), NOT(m_any))
    r = select(C, special, select(C, pinf, _c(INF), _c(SM.LOG_NAN)), r)
    r = select(C, C.AND(s, NOT(OR(C, or_reduce(C, a[23:31]), m_any))), _c(NEG_ZERO), r)
    return _canon_nan(C, r)


def gumbel_noise(C, u):
    """``-logf(-log1pf(-u))`` (`sampling.noise_lane` after the uniform)."""
    return fp.f32_neg(nv_logf(C, fp.f32_neg(nv_log1pf(C, fp.f32_neg(u)))))


NvLogf = definition("NvLogf", 3, nv_logf, (("a", F32),), F32, word=TG.NvLogfV2,
                    doc="libdevice __nv_logf (FTZ) on bits: NvLogf_v2 (every NaN result 0x7FFFFFFF)")

GumbelStreamKeyHead = definition("GumbelStreamKeyHead", 1, stream_key_head, (("seed", I64), ("pos", I64)), Tuple(C4, K4, K4),
                                 doc="Rounds 0-4 of the stream key's Philox4x32-10 (counter (pos, 0, 0), key seed): the counter, "
                                     "the low key words of rounds 5, 6, 7 and 9 and the high ones of rounds 5-8")
GumbelStreamKeyTail = definition("GumbelStreamKeyTail", 1, stream_key_tail, (("c", C4), ("k0", K4), ("k1", K4)), U32,
                                 doc="Rounds 5-9 of the stream key's Philox4x32-10 from round 5's counter and the key words "
                                     "word 0 reads: word 0")


def _stream_key(B, S, seed, pos):
    h = B.call(GumbelStreamKeyHead, seed, pos)
    return B.call(GumbelStreamKeyTail, h[0], h[1], h[2])


GumbelStreamKey = CompositeDefinition(
    "GumbelStreamKey", 2, (), lambda S: ((("seed", I64), ("pos", I64)), U32), _stream_key, conformance="exact-model-tested",
    doc="tl.randint(seed, pos) on bits: GumbelStreamKey_v1 (Philox4x32-10 word 0), its rounds 0-4 and 5-9 one Call each")
GumbelStreamKey.word = SM.GumbelStreamKey  # type: ignore[attr-defined]


# --- GumbelSelectF32_v2{V} as separable stages -------------------------------------------------------------------------------

def gumbel_setup(C, S, key, temp):
    """The values every lane of a row shares: the stream key's `lane_keys`, ``noisy`` = ``temp != 0`` (`F32Eq_v1`: +0 and -0
    are zero, a NaN is not), and the key's `round2_lows` for the first ``S.N`` offsets in a group."""
    ks, ka = lane_keys(C, key)
    return ks + [ka[:32], ka[32:], [or_reduce(C, temp[:31])]] + round2_lows(C, ka, S.N)


def _setup_sig(S):
    if not 1 <= S.N <= G:
        raise ValueError(f"a group has 1..{G} offsets, not {S.N}")
    return (("key", U32), ("temp", F32)), Tuple(KS, KA, BIT, Array(S.N, LB))


def lane_head(C, hi_a, lo_a, lb, k1, k2, ka_lo, ka_hi):
    """Rounds 0-2 of lane ``v = lo + 64 hi``, given as ``v * PHILOX_ROUND_A = lo_a + 2^6 hi_a`` (`lane_products`). Round 0's low
    key word, the key, reaches only round 1's product ``ka``, so zeros stand in for it."""
    va = add(C, list(lo_a) + [ZERO] * (64 - LOB), [ZERO] * GB + list(hi_a), width=64)
    return lane_rounds(C, [Z32] * 4, [Z32, k1, k2], 0, va=va, ka=ka_lo + ka_hi, lb=lb)


def lane_mid(C, c0, c1, c2, c3, *k0s):
    """Rounds 3-7: counter words 0, 2 and 3 (word 1 reaches only round 8's word 0, which round 9's word 0 does not read)."""
    c = lane_rounds(C, [c0, c1, c2, c3], k0s, 3)
    return [c[0], c[2], c[3]]


def lane_tail(C, c0, c2, c3, k9, x, noisy):
    """Rounds 8-9, the uniform and the noise ``g``, then one lane of `GumbelSelectStep_v1`: ``x + g`` (`F32AddFtz_v1`) when
    noisy, else ``x``. Round 8's word 1 and low key word reach only its word 0, so zeros stand in for them."""
    g = gumbel_noise(C, tl_rand(C, lane_rounds(C, [c0, Z32, c2, c3], [Z32, k9], 8)[0]))
    return select(C, noisy[0], f32_add_ftz(C, x, g), x)


GumbelSetup = traced_family("GumbelSetup", 1, ("N",), _setup_sig, gumbel_setup,
                            "The Gumbel select's shared values: the stream key's round keys a lane reads and its round-1 "
                            "product, temp != 0, and round 2's low partial products for the first N offsets in a group")
GumbelLaneHead = definition("GumbelLaneHead", 1, lane_head, (("hi_a", HA), ("lo_a", LA), ("lb", LB), ("k", K2), ("ka", KA)), C4,
                            doc="Rounds 0-2 of a lane's Philox4x32-10 (counter (lo + 64 hi, 0, 0, 0), key (key, 0)): the counter")
GumbelLaneRounds = definition("GumbelLaneRounds", 1, lane_mid, (("c", C4), ("k", K5)), C3,
                              doc="Rounds 3-7 of a lane's Philox4x32-10, round r's key (k[r - 3], r * PHILOX_KEY_B): counter "
                                  "words 0, 2 and 3")
GumbelLaneTail = definition("GumbelLaneTail", 1, lane_tail, (("c", C3), ("k", U32), ("x", F32), ("noisy", BIT)), F32,
                            doc="Rounds 8-9 of a lane's Philox4x32-10 from round 8's counter words 0, 2 and 3 and round 9's low "
                                "key word, tl.rand and the noise g = -logf(-log1pf(-u)), then x + g (add.ftz) when noisy, else x")


def _lane(B, S, x, hi_a, lo_a, lb, ks, ka, noisy):
    c = B.call(GumbelLaneRounds, B.call(GumbelLaneHead, hi_a, lo_a, lb, ks[0:2], ka), ks[2:7])
    return B.call(GumbelLaneTail, c, ks[7], x, noisy)


GumbelLane = CompositeDefinition(
    "GumbelLane", 1, (), lambda S: ((("x", F32), ("hi_a", HA), ("lo_a", LA), ("lb", LB), ("ks", KS), ("ka", KA), ("noisy", BIT)),
                                    F32), _lane, conformance="exact-model-tested",
    doc="One noisy lane of the Gumbel select: x + g(key)[lo + 64 hi] (add.ftz) when noisy, else x, the lane index through "
        "lo_a = lo * PHILOX_ROUND_A and hi_a = hi * PHILOX_ROUND_A mod 2^58, and lb, round 2's product of the counter's low six "
        "bits (offset lo's, from GumbelSetup_v1); its rounds 0-2, 3-7 and 8-9 with the noise one Call each")


def _group_sig(S):
    if not 1 <= S.G <= G:
        raise ValueError(f"a group has 1..{G} lanes, not {S.G}")
    return ((("x", Array(S.G, F32)), ("hi_a", HA), ("lo_a", Array(S.G, LA)), ("lb", Array(S.G, LB)), ("ks", KS), ("ka", KA),
             ("noisy", BIT)), Array(S.G, F32))


def _group(B, S, x, hi_a, lo_a, lb, ks, ka, noisy):
    return B.batch(GumbelLane, x, hi_a, lo_a, lb, ks, ka, noisy, axes=(0, None, 0, 0, None, None, None))


GumbelNoiseGroup = CompositeDefinition("GumbelNoiseGroup", 1, ("G",), _group_sig, _group, conformance="exact-model-tested",
                                       doc="G noisy lanes of one group (its number's product hi_a), each one GumbelLane_v1 Call")


def _noise_sig(S):
    if S.V < 1:
        raise ValueError(f"a logits row has V >= 1 entries, not {S.V}")
    return ((("x", Array(S.V, F32)), ("hi_a", Array(-(-S.V // G), HA)), ("lo_a", Array(G, LA)), ("lb", Array(min(S.V, G), LB)),
             ("ks", KS), ("ka", KA), ("noisy", BIT)), Array(S.V, F32))


def _noise(B, S, x, hi_a, lo_a, lb, ks, ka, noisy):
    full, rem = divmod(S.V, G)
    outs = []
    if full:
        outs.append(B.batch(bind(GumbelNoiseGroup, G=G), Coll(Array(full, Array(G, F32)), x[0:full * G].refs), hi_a[0:full], lo_a,
                            lb, ks, ka, noisy, axes=(0, 0, None, None, None, None, None)))
    if rem:
        outs.append(B.call(bind(GumbelNoiseGroup, G=rem), x[full * G:S.V], hi_a[full], lo_a[0:rem], lb[0:rem], ks, ka, noisy))
    return Coll(Array(S.V, F32), concat([o.refs for o in outs]))


GumbelNoise = CompositeDefinition("GumbelNoise", 1, ("V",), _noise_sig, _noise, conformance="exact-model-tested",
                                  doc="The noisy row of the Gumbel select: its lanes in groups of 64 (GumbelNoiseGroup_v1), the "
                                      "full groups as one batch")


def lane_products(call, V: int) -> tuple[Coll, Coll]:
    """`GumbelNoise_v1{V}`'s lane operands as constant bits: ``hi_a`` = ``hi * PHILOX_ROUND_A mod 2^58`` for each group number hi,
    ``lo_a`` = ``lo * PHILOX_ROUND_A`` for each offset lo in a group."""
    bit = [call(BL.constant(b)).refs[0] for b in (0, 1)]
    ng = -(-V // G)
    a = SM.PHILOX_ROUND_A
    hi = Explicit(tuple(bit[(g * a >> j) & 1] for g in range(ng) for j in range(HAB)))
    lo = Explicit(tuple(bit[(i * a >> j) & 1] for i in range(G) for j in range(LOB)))
    return Coll(Array(ng, HA), hi), Coll(Array(G, LA), lo)


def gumbel_select_calls(call, logits: Coll, seed: Coll, pos: Coll, temp: Coll, V: int) -> Coll:
    """``GumbelSelectF32_v2{V}`` as Calls of its stages (``call(fn, *args)``, as `sampling_boolean.token_select_calls`): the
    stream key, the setup, the noisy row, the argmax tree's levels; returns the i32 token's bits."""
    setup = call(bind(GumbelSetup, N=min(V, G)), call(GumbelStreamKey, seed, pos), temp)
    hi_a, lo_a = lane_products(call, V)
    y = call(bind(GumbelNoise, V=V), logits, hi_a, lo_a, setup[3], setup[0], setup[1], setup[2])
    return argmax_calls(call, y, V, F32ArgmaxStage)


def _gumbel_select(B, S, logits, seed, pos, temp):
    return gumbel_select_calls(B.call, logits, seed, pos, temp, S.V)


def _gumbel_select_word(b: dict):
    return bind(SM.GumbelSelectF32, V=b["V"])


GumbelSelectF32 = CompositeDefinition(
    "GumbelSelectF32", 2, ("V",), lambda S: ((("logits", Array(S.V, F32)), ("seed", I64), ("pos", I64), ("temp", F32)), I32),
    _gumbel_select, conformance="exact-model-tested",
    doc="The Gumbel select on bits: GumbelSelectF32_v1{V} (the first index of the maximum of logits + Gumbel noise, or of the "
        "logits at temp 0), as the chain of its separable stages.")
GumbelSelectF32.word = _gumbel_select_word  # type: ignore[attr-defined]
