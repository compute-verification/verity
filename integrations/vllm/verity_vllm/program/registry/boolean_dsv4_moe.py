"""`registry.dsv4_moe`'s Definitions as Boolean Definitions: the next version of each word Definition, the same function on bits
(F32 = I32 = Array<32, Value<1>>, BF16 = Array<16, Value<1>>).

    MufuLg2Mantissa_v2                  (x: F32) -> F32                           word view: MufuLg2Mantissa_v1
    Ex2Approx_v2                        (t: F32) -> F32                           word view: Ex2Approx_v1
    Dsv4HashWeight_v2                   (x: F32) -> F32                           word view: Dsv4HashWeight_v1
    MoeRouterHash_v2{E,SCALE}           (logits: Array<E, F32>, ids: Array<6, I32>) -> Array<6, F32>
                                                                                  word view: MoeRouterHash_v1{E,SCALE}
    Dsv4SqrtSoftplus_v2                 (x: F32) -> F32                           word view: Dsv4SqrtSoftplus_v1
    MoeRouterSqrtSoftplus_v2{E,SCALE}   (logits, bias: Array<E, F32>) -> (Array<6, F32>, Array<6, I32>)
                                                                                  word view: MoeRouterSqrtSoftplus_v1{E,SCALE}
    SiluMulClampBf16_v2{LIMIT}          (g, u: BF16) -> BF16                      word view: SiluMulClampBf16_v1{LIMIT}
    SiluMulClamp_v2{I,LIMIT}            (gu: Array<2I, BF16>) -> Array<I, BF16>   word view: SiluMulClamp_v1{I,LIMIT}

`MufuLg2Mantissa_v2` reads the interpolator the way `verity.ml.circuits.ops.boolean.mufu.table_word` reads a MUFU table: ``2v =
2 C0 + 2 C1 u + 2 C2 Q(u)`` is one carry-save sum over 39 columns, each product a row ``x XOR n_k`` per digit of ``2c + 1`` (no
AND), the digits' signs and the bits of C0 (the digits' constants folded in) functions of the six segment bits, each the XOR of
the monomials in its algebraic normal form; G is bits 3 .. 38 of the sum.  The word is then G's: its leading-zero count z
over 64 bits (G at the top, below it 28 zeros), the 23 bits under the leading one (G shifted left by z), the exponent field 126
- z, and +0 where m = 0.

The others are cut only where they read a MUFU, at Calls to core's `MufuEx2Ftz_v2`, `MufuSqrtFtz_v2` and `DivFullRcp_v2` and to
`MufuLg2Mantissa_v2`, and at libdevice's logarithm (`gumbel_boolean.NvLogf_v3`), so that each is one shared Call; every other
value is decoded in one traced piece:

    Ex2Approx_v2               `Ex2ApproxIn_v1` (whether t < -126, and the operand t * 0.5 or t), `MufuEx2Ftz_v2`,
                               `Ex2ApproxOut_v1` (the square where t < -126)
    Dsv4HashWeight_v2          `Dsv4HashWeightIn_v1` (|x| * -log2 e read as `Ex2ApproxIn_v1` reads t, and fmaxf(x, 0)),
                               `MufuEx2Ftz_v2`, `Dsv4HashWeightMid_v1` (the square, + 1, whether the sum is NaN or 2),
                               `MufuLg2Mantissa_v2`, `Dsv4HashWeightOut_v1` (MUFU.LG2's word at 2 and NaN, the FFMA with ln 2,
                               sqrt.rn as PoUW's `f32_sqrt_rn`, a NaN +0)
    MoeRouterHash_v2           `MoeRouterHashPick_v1{E}` (each id's logit: a multiplexer tree on its low bits, logits[0] when a
                               high bit is set, as `MoeRouterPick_v1`), six `Dsv4HashWeight_v2`, `MoeRouterHashNorm_v1{SCALE}`
                               (the warp's sum, the divisor, ``w * SCALE / divisor`` with F32Mul_v2's and F32Div_v2's NaN words)
    Dsv4SqrtSoftplus_v2        `Dsv4SqrtSoftplusIn_v1` (x * log2 e read as `Ex2ApproxIn_v1` reads t, and x > 20),
                               `MufuEx2Ftz_v2`, `Dsv4SqrtSoftplusMid_v1` (the square, 1 + it), `NvLogf_v3`,
                               `Dsv4SqrtSoftplusSelect_v1` (x where x > 20, else the logarithm), `MufuSqrtFtz_v2`
    MoeRouterSqrtSoftplus_v2   E `Dsv4SqrtSoftplus_v2`, `MoeRouterSqrtSoftplusTopK_v1{E}` (the currents, the six slots by their
                               order keys, each slot's score, the warp's sum and the divisor), `DivFullRcp_v2`,
                               `MoeRouterSqrtSoftplusNorm_v1{SCALE}` (div.full's scaled SCALE times the reciprocal, then each
                               slot's product with F32Mul_v2's NaN word)
    SiluMulClampBf16_v2        `SiluMulClampIn_v1{LIMIT}` (the clamps on the bf16 words: a NaN or a value above LIMIT is LIMIT,
                               an up below -LIMIT is -LIMIT), then `boolean_silu.SiluMulBf16_v4`
    SiluMulClamp_v2            I `SiluMulClampBf16_v2`

A comparison is of f32 order keys (`boolean_moe._key`, +0 and -0 at one place): the ordered ``t < -126`` and ``x > 20`` are
false at a NaN, and a slot's argmax is strict with the lower expert winning ties, `RouterArgmaxStep_v1`'s scan on currents that
are never NaN.  A slot's score is the XOR of the scores under the one-hot decode of its expert.  The sums are
`moe._lane0_sum`'s tree with F32Add_v1's NaN words; no NaN of them reaches an output, since a NaN sum makes the divisor 1.
"""
from __future__ import annotations

from functools import cache

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import (NOT, ONE, OR, ZERO, and_reduce, compress_columns, const_bits, lzc, mux, or_reduce,
                                              shl_window, sub)
from verity.ml.circuits.boolean.trace import definition
from verity.core.protocols.pouw.circuit.boolean import f32_sqrt_rn
from verity_vllm.program.registry import boolean_silu as BSL
from verity_vllm.program.registry import dsv4_moe as D
from verity_vllm.program.registry import moe as M
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.boolean_moe import _eq_word, _fmaxf, _key, _key_of, _less, _nan
from verity_vllm.program.registry.gumbel_boolean import NvLogf

__all__ = ["Dsv4HashWeight", "Dsv4HashWeightIn", "Dsv4HashWeightMid", "Dsv4HashWeightOut", "Dsv4SqrtSoftplus",
           "Dsv4SqrtSoftplusIn", "Dsv4SqrtSoftplusMid", "Dsv4SqrtSoftplusSelect", "Ex2Approx", "Ex2ApproxIn", "Ex2ApproxOut",
           "MoeRouterHash", "MoeRouterHashNorm", "MoeRouterHashPick", "MoeRouterSqrtSoftplus", "MoeRouterSqrtSoftplusNorm",
           "MoeRouterSqrtSoftplusTopK", "MufuLg2Mantissa", "SiluMulClamp", "SiluMulClampBf16", "SiluMulClampIn",
           "mufu_lg2_mantissa"]

BF16 = Array(16, BL.BIT)
F32 = I32 = Array(32, BL.BIT)
FLAG = Array(1, BL.BIT)
TOPK = D.TOPK
ONE_BITS, NEG_INF_BITS = 0x3F800000, 0xFF800000


def _w(bits: int) -> list:
    return const_bits(bits, 32)


# --- MUFU.LG2 on [1, 2) --------------------------------------------------------------------------------------------------------

#: the columns of 2v (2v < 2^39 on every mantissa), and the low bits of 2v below G
LG2_COLUMNS, LG2_DROP = 39, 1 + D.LG2_SHIFT
LG2_SEGMENT_BITS = 23 - D.LG2_OFFSET_BITS
LG2_SQUARE_BITS = 2 * D.LG2_OFFSET_BITS - D.LG2_SQUARE_FROM


@cache
def _lg2_plan() -> tuple[tuple[int, int], tuple[tuple[int, ...], ...]]:
    """The segment functions of `table_word`'s encoding for `dsv4_moe.LG2_COEFFS`: the digit signs n_k of C1 (K1 of them), of
    C2 (K2), then the 39 bits of 2 C0 with the digits' constants folded in, each as the monomials of the segment bits in its
    algebraic normal form; and (K1, K2)."""
    c0 = [2 * c for c, _, _ in D.LG2_COEFFS]
    rows, Ks = [], []
    for n, cs in ((D.LG2_OFFSET_BITS, [c for _, c, _ in D.LG2_COEFFS]), (LG2_SQUARE_BITS, [c for _, _, c in D.LG2_COEFFS])):
        K, B = BM._digits(cs)
        Ks.append(K)
        rows += [[1 - ((b >> k) & 1) for b in B] for k in range(K)]
        c0 = [c - ((1 << n) - 1) * ((((1 << K) - 1) ^ b) + 1) for c, b in zip(c0, B)]
    rows += [[(c >> j) & 1 for c in c0] for j in range(LG2_COLUMNS)]
    for r in rows:
        for i in range(LG2_SEGMENT_BITS):
            for g in range(len(r)):
                if g >> i & 1:
                    r[g] ^= r[g ^ (1 << i)]
    return (Ks[0], Ks[1]), tuple(tuple(g for g, a in enumerate(r) if a) for r in rows)


def mufu_lg2_mantissa(C, x):
    """`MufuLg2Mantissa_v1` on bits: the interpolant G as one carry-save sum, then G 2^-36 rounded toward zero, +0 at m = 0."""
    (K1, K2), anf = _lg2_plan()
    m = list(x[:23])
    u = m[:D.LG2_OFFSET_BITS]
    terms = BM._monomials(C, m[D.LG2_OFFSET_BITS:])
    fns = [BM._xor_all(terms[i] for i in r) for r in anf]
    cols: list[list] = [[] for _ in range(LG2_COLUMNS)]
    for v, signs in ((u, fns[:K1]), (BM.truncated_square(C, u, D.LG2_SQUARE_FROM), fns[K1:K1 + K2])):
        for k, n_k in enumerate(signs):
            for j, b in enumerate(v):
                if j + k < LG2_COLUMNS:
                    cols[j + k].append(b ^ n_k)
        for j, b in enumerate(v):
            cols[j].append(NOT(b))
    for j, b in enumerate(fns[K1 + K2:]):
        cols[j].append(b)
    G = compress_columns(C, cols)[LG2_DROP:LG2_COLUMNS]
    top = [ZERO] * (64 - len(G)) + list(G)
    _, z = lzc(C, top)
    mant = shl_window(C, top, z, 40, 63)
    exp, _ = sub(C, const_bits(len(G) - 1 + 127 + D.LG2_SCALE, 8), z, 8)
    nz = or_reduce(C, m)
    return [C.AND(nz, b) for b in mant + exp] + [ZERO]


MufuLg2Mantissa = definition("MufuLg2Mantissa", 2, mufu_lg2_mantissa, (("x", F32),), F32, word=D.MufuLg2Mantissa,
                             doc="MUFU.LG2 of 1.m on bits: MufuLg2Mantissa_v1's interpolator as one carry-save sum, the word "
                                 "rounded toward zero by a leading-zero count and a shift")


# --- ex2.approx.f32 ------------------------------------------------------------------------------------------------------------

def _ex2_in(C, t):
    """(the MUFU.EX2 operand, whether t < -126): ``t * 0.5`` (F32Mul_v1) below -126, else t; a NaN is not below."""
    small = C.AND(NOT(_nan(C, t)), _less(C, _key(C, t), const_bits(_key_of(D.MINUS_126), 32)))
    return mux(C, small, fp.f32_mul(C, t, _w(D.HALF), ir_nan=True), list(t)), small


def _ex2_out(C, y, small):
    """``y * y`` (F32Mul_v1) where t was below -126, else y."""
    return mux(C, small, fp.f32_mul(C, y, y, ir_nan=True), list(y))


def ex2_approx_in(C, t):
    op, small = _ex2_in(C, t)
    return [op, [small]]


def ex2_approx_out(C, y, small):
    return _ex2_out(C, y, small[0])


Ex2ApproxIn = definition("Ex2ApproxIn", 1, ex2_approx_in, (("t", F32),), Tuple(F32, FLAG),
                         doc="ex2.approx.f32 before its MUFU.EX2: the operand (t * 0.5 below -126) and whether t < -126")
Ex2ApproxOut = definition("Ex2ApproxOut", 1, ex2_approx_out, (("y", F32), ("small", FLAG)), F32,
                          doc="ex2.approx.f32 after its MUFU.EX2: y * y where t < -126")


def _ex2_approx(B, S, t):
    """`Ex2Approx_v1` on bits: `Ex2ApproxIn_v1`, `MufuEx2Ftz_v2`, `Ex2ApproxOut_v1`."""
    a = B.call(Ex2ApproxIn, t)
    return B.call(Ex2ApproxOut, B.call(BM.MufuEx2Ftz, a[0]), a[1])


Ex2Approx = EW.on_bits("Ex2Approx", 2, (), lambda S: ((("t", F32),), F32), _ex2_approx, D.Ex2Approx)


# --- the CUDA hash router (layers 0-2) -----------------------------------------------------------------------------------------

def hash_weight_in(C, x):
    """`Dsv4HashWeight_v1` before its MUFU.EX2: the operand of ``ex2.approx(|x| * -log2 e)`` and its flag, and fmaxf(x, 0)."""
    op, small = _ex2_in(C, fp.f32_mul(C, list(x[:31]) + [ZERO], _w(D.NEG_LOG2E), ir_nan=True))
    return [op, [small], _fmaxf(C, x, _w(0))]


def hash_weight_mid(C, y, small):
    """Between the MUFUs: ``y = ex2 + 1`` (F32Add_v1), and whether y is NaN or 2."""
    s = fp.f32_add(C, _ex2_out(C, y, small[0]), _w(ONE_BITS), ir_nan=True)
    return [s, [_nan(C, s), _eq_word(C, s, D.TWO)]]


def hash_weight_out(C, lg2, flags, f):
    """After MUFU.LG2: its word (1.0 at y = 2, 0x7FFFFFFF at a NaN), ``fma(lg2, ln 2, f)`` (F32Fma_v1), sqrt.rn, a NaN +0."""
    nan, two = flags
    l = mux(C, two, _w(ONE_BITS), mux(C, nan, _w(fp.F32_NAN_PTX), list(lg2)))
    r = f32_sqrt_rn(C, fp.f32_fma(C, l, _w(D.LN2), f))
    keep = NOT(_nan(C, r))
    return [C.AND(keep, b) for b in r]


Dsv4HashWeightIn = definition("Dsv4HashWeightIn", 1, hash_weight_in, (("x", F32),), Tuple(F32, FLAG, F32),
                              doc="the hash weight before its MUFU.EX2: the operand of ex2.approx(|x| * -log2 e), its flag, "
                                  "and fmaxf(x, 0)")
Dsv4HashWeightMid = definition("Dsv4HashWeightMid", 1, hash_weight_mid, (("y", F32), ("small", FLAG)),
                               Tuple(F32, Array(2, BL.BIT)),
                               doc="the hash weight between its MUFUs: ex2 + 1, and whether it is NaN or 2")
Dsv4HashWeightOut = definition("Dsv4HashWeightOut", 1, hash_weight_out,
                               (("lg2", F32), ("flags", Array(2, BL.BIT)), ("f", F32)), F32,
                               doc="the hash weight after its MUFU.LG2: the FFMA with ln 2 and fmaxf(x, 0), sqrt.rn, a NaN +0")


def _hash_weight(B, S, x):
    """`Dsv4HashWeight_v1` on bits: `Dsv4HashWeightIn_v1`, `MufuEx2Ftz_v2`, `Dsv4HashWeightMid_v1`, `MufuLg2Mantissa_v2`,
    `Dsv4HashWeightOut_v1`."""
    a = B.call(Dsv4HashWeightIn, x)
    y = B.call(Dsv4HashWeightMid, B.call(BM.MufuEx2Ftz, a[0]), a[1])
    return B.call(Dsv4HashWeightOut, B.call(MufuLg2Mantissa, y[0]), y[1], a[2])


Dsv4HashWeight = EW.on_bits("Dsv4HashWeight", 2, (), lambda S: ((("x", F32),), F32), _hash_weight, D.Dsv4HashWeight)


def _nb(E: int) -> int:
    return max(1, (E - 1).bit_length())


def router_hash_pick(S):
    """Each id's logit: the multiplexer tree on its low bits over the logits (logits[0] past E), and logits[0] when a bit above
    them is set (`MoeRouterPick_v1` on every id word)."""
    E, nb = S.E, _nb(S.E)

    def build(C, *li):
        logits, ids = li[:E], li[E:]
        padded = list(logits) + [logits[0]] * ((1 << nb) - E)
        out = []
        for e in ids:
            rows = padded
            for bit in e[:nb]:
                rows = [mux(C, bit, rows[i + 1], rows[i]) for i in range(0, len(rows), 2)]
            out.append(mux(C, or_reduce(C, e[nb:]), logits[0], rows[0]))
        return out

    return build


def _lane0(C, z: list) -> list:
    """`moe._lane0_sum` on bits: lane 0 of the butterfly over lanes holding z and +0, an add with a +0 lane dropped."""
    memo: dict[tuple[int, int], list | None] = {}

    def lane(j: int, t: int):
        if j == 0:
            return z[t] if t < len(z) else None
        if (j, t) not in memo:
            a, b = lane(j - 1, t), lane(j - 1, t ^ M.WARP_BUTTERFLY[j - 1])
            memo[j, t] = a if b is None else b if a is None else fp.f32_add(C, a, b, ir_nan=True)
        return memo[j, t]

    return lane(len(M.WARP_BUTTERFLY), 0)


def _divisor(C, total):
    """``total > 0 ? total : 1``."""
    pos = C.AND(C.AND(NOT(total[31]), or_reduce(C, total[:31])), NOT(_nan(C, total)))
    return mux(C, pos, list(total), _w(ONE_BITS))


def router_hash_norm(S):
    """``w[k] * SCALE / (sum > 0 ? sum : 1)``: the warp's sum, then F32Mul_v2 and F32Div_v2 (the GPU's NaN word)."""
    scale = _w(P.bits_of(S.SCALE))

    def build(C, *w):
        d = _divisor(C, _lane0(C, list(w)))
        return [fp.f32_div(C, fp.f32_mul(C, v, scale, nan_word=fp.F32_NAN_PTX), d, nan_word=fp.F32_NAN_PTX) for v in w]

    return build


MoeRouterHashPick = EW.specialized("MoeRouterHashPick", 1, ["E"], router_hash_pick,
                                   lambda S: ((("logits", Array(S.E, F32)), ("ids", Array(TOPK, I32))), Array(TOPK, F32)),
                                   doc="dsv4HashTopkSoftplusSqrt's reads of the six table ids' logits on bits (MoeRouterPick_v1)")
MoeRouterHashNorm = EW.specialized("MoeRouterHashNorm", 1, ["SCALE"], router_hash_norm,
                                   lambda S: ((("w", Array(TOPK, F32)),), Array(TOPK, F32)),
                                   doc="dsv4HashTopkSoftplusSqrt's warp sum and w * SCALE / (sum > 0 ? sum : 1) on bits")


def _router_hash(B, S, logits, ids):
    """`MoeRouterHash_v1` on bits: `MoeRouterHashPick_v1`, six `Dsv4HashWeight_v2`, `MoeRouterHashNorm_v1`."""
    x = B.call(bind(MoeRouterHashPick, E=S.E), logits, ids)
    return B.call(bind(MoeRouterHashNorm, SCALE=S.SCALE), B.batch(Dsv4HashWeight, x, axes=(0,)))


MoeRouterHash = EW.on_bits("MoeRouterHash", 2, ("E", "SCALE"),
                           lambda S: ((("logits", Array(S.E, F32)), ("ids", Array(TOPK, I32))), Array(TOPK, F32)),
                           _router_hash, D.MoeRouterHash)


# --- the Triton router (dsv4_topk) ---------------------------------------------------------------------------------------------

def sqrt_softplus_in(C, x):
    """`Dsv4SqrtSoftplus_v1` before its MUFU.EX2: the operand of ``ex2.approx(x * log2 e)`` and its flag, and x > 20."""
    op, small = _ex2_in(C, fp.f32_mul(C, x, _w(D.LOG2E), ir_nan=True))
    big = C.AND(NOT(_nan(C, x)), _less(C, const_bits(_key_of(D.TWENTY), 32), _key(C, x)))
    return [op, [small], [big]]


def sqrt_softplus_mid(C, y, small):
    """Between MUFU.EX2 and the logarithm: ``1 + ex2`` (F32Add_v1)."""
    return fp.f32_add(C, _w(ONE_BITS), _ex2_out(C, y, small[0]), ir_nan=True)


def sqrt_softplus_select(C, big, x, log):
    """x where x > 20, else the logarithm."""
    return mux(C, big[0], list(x), list(log))


Dsv4SqrtSoftplusIn = definition("Dsv4SqrtSoftplusIn", 1, sqrt_softplus_in, (("x", F32),), Tuple(F32, FLAG, FLAG),
                                doc="dsv4_topk's score before its MUFU.EX2: the operand of ex2.approx(x * log2 e), its flag, "
                                    "and x > 20")
Dsv4SqrtSoftplusMid = definition("Dsv4SqrtSoftplusMid", 1, sqrt_softplus_mid, (("y", F32), ("small", FLAG)), F32,
                                 doc="dsv4_topk's score between MUFU.EX2 and the logarithm: 1 + ex2")
Dsv4SqrtSoftplusSelect = definition("Dsv4SqrtSoftplusSelect", 1, sqrt_softplus_select,
                                    (("big", FLAG), ("x", F32), ("log", F32)), F32,
                                    doc="dsv4_topk's score before its MUFU.SQRT: x where x > 20, else the logarithm")


def _sqrt_softplus(B, S, x):
    """`Dsv4SqrtSoftplus_v1` on bits: `Dsv4SqrtSoftplusIn_v1`, `MufuEx2Ftz_v2`, `Dsv4SqrtSoftplusMid_v1`, `NvLogf_v3`,
    `Dsv4SqrtSoftplusSelect_v1`, `MufuSqrtFtz_v2`."""
    a = B.call(Dsv4SqrtSoftplusIn, x)
    log = B.call(NvLogf, B.call(Dsv4SqrtSoftplusMid, B.call(BM.MufuEx2Ftz, a[0]), a[1]))
    return B.call(BM.MufuSqrtFtz, B.call(Dsv4SqrtSoftplusSelect, a[2], x, log))


Dsv4SqrtSoftplus = EW.on_bits("Dsv4SqrtSoftplus", 2, (), lambda S: ((("x", F32),), F32), _sqrt_softplus, D.Dsv4SqrtSoftplus)


def _decode(C, bits: list) -> list:
    """The 2^n minterms of ``bits`` (LSB first): entry j is 1 iff the bits are j."""
    m = [ONE]
    for b in bits:
        m = [C.AND(x, NOT(b)) for x in m] + [C.AND(x, b) for x in m]
    return m


def router_topk(S):
    """`MoeRouterSqrtSoftplus_v1`'s selection on its scores w: ``current = w + bias`` (F32Add_v1), a NaN -1e30; six slots, each
    the strict argmax of the currents' order keys (the lower expert on ties), its one-hot decode reading the slot's score and
    setting its current to -inf's key; then the warp's sum of the six and the divisor ``sum > 0 ? sum : 1``."""
    E, nb = S.E, _nb(S.E)
    if E < TOPK:
        raise ValueError(f"E must be at least {TOPK}")
    masked = const_bits(_key_of(NEG_INF_BITS), 32)

    def build(C, *wb):
        w, bias = wb[:E], wb[E:]
        key = []
        for i in range(E):
            c = fp.f32_add(C, w[i], bias[i], ir_nan=True)
            key.append(_key(C, mux(C, _nan(C, c), _w(D.NAN_SCORE), c)))
        sel, ids = [], []
        for k in range(TOPK):
            best, idx = key[0], [ZERO] * nb
            for j in range(1, E):
                gt = _less(C, best, key[j])
                best, idx = mux(C, gt, key[j], best), mux(C, gt, const_bits(j, nb), idx)
            hit = _decode(C, idx)[:E]
            v = [C.AND(hit[0], b) for b in w[0]]
            for i in range(1, E):
                v = [a ^ C.AND(hit[i], b) for a, b in zip(v, w[i])]
            sel.append(v)
            ids.append(idx + [ZERO] * (32 - nb))
            if k < TOPK - 1:
                key = [mux(C, hit[i], masked, key[i]) for i in range(E)]
        return [*sel, *ids, _divisor(C, _lane0(C, sel))]

    return build


def router_sqrt_softplus_norm(S):
    """``w[k] * (DivFullScaleA(SCALE, d) * r)``, r = `DivFullRcp_v1`(d): F32Mul_v1, then F32Mul_v2 (the GPU's NaN word)."""
    scale = _w(P.bits_of(S.SCALE))

    def build(C, *swr):
        w, d, r = swr[:TOPK], swr[TOPK], swr[TOPK + 1]
        q = fp.f32_mul(C, BM.div_full_scale_a(C, scale, d), r, ir_nan=True)
        return [fp.f32_mul(C, v, q, nan_word=fp.F32_NAN_PTX) for v in w]

    return build


MoeRouterSqrtSoftplusTopK = EW.specialized(
    "MoeRouterSqrtSoftplusTopK", 1, ["E"], router_topk,
    lambda S: ((("w", Array(S.E, F32)), ("bias", Array(S.E, F32))), Tuple(Array(TOPK, F32), Array(TOPK, I32), F32)),
    doc="dsv4_topk's six slots on bits: the currents, the argmax rounds by order keys, each slot's score, the warp's sum and "
        "the divisor")
MoeRouterSqrtSoftplusNorm = EW.specialized(
    "MoeRouterSqrtSoftplusNorm", 1, ["SCALE"], router_sqrt_softplus_norm,
    lambda S: ((("w", Array(TOPK, F32)), ("d", F32), ("r", F32)), Array(TOPK, F32)),
    doc="dsv4_topk's renormalisation on bits: w * div.full(SCALE, d), the reciprocal r read from DivFullRcp_v2")


def _router_sqrt_softplus(B, S, logits, bias):
    """`MoeRouterSqrtSoftplus_v1` on bits: E `Dsv4SqrtSoftplus_v2`, `MoeRouterSqrtSoftplusTopK_v1`, `DivFullRcp_v2`,
    `MoeRouterSqrtSoftplusNorm_v1`."""
    w = B.batch(Dsv4SqrtSoftplus, logits, axes=(0,))
    s = B.call(bind(MoeRouterSqrtSoftplusTopK, E=S.E), w, bias)
    out = B.call(bind(MoeRouterSqrtSoftplusNorm, SCALE=S.SCALE), s[0], s[2], B.call(BM.DivFullRcp, s[2]))
    return tuple_of(out, s[1])


MoeRouterSqrtSoftplus = EW.on_bits(
    "MoeRouterSqrtSoftplus", 2, ("E", "SCALE"),
    lambda S: ((("logits", Array(S.E, F32)), ("bias", Array(S.E, F32))), Tuple(Array(TOPK, F32), Array(TOPK, I32))),
    _router_sqrt_softplus, D.MoeRouterSqrtSoftplus)


# --- the clamped SwiGLU --------------------------------------------------------------------------------------------------------

def silu_mul_clamp_in(S):
    """The clamps on bf16 words: ``g' = fminf(g, LIMIT)``, ``u' = fmaxf(fminf(u, LIMIT), -LIMIT)``; a NaN is LIMIT."""
    lim = D.limit_bits(S.LIMIT) >> 16
    mag = const_bits(lim, 15)

    def build(C, g, u):
        def flags(h):
            """(h is NaN, |h| > LIMIT): ``LIMIT - |h|`` borrows."""
            nan = C.AND(and_reduce(C, h[7:15]), or_reduce(C, h[:7]))
            return nan, sub(C, mag, h[:15], 15)[1]

        hi, lo = const_bits(lim, 16), const_bits(lim | 0x8000, 16)
        gn, gm = flags(g)
        un, um = flags(u)
        gc = mux(C, OR(C, gn, C.AND(NOT(g[15]), gm)), hi, list(g))
        uc = mux(C, OR(C, un, C.AND(NOT(u[15]), um)), hi, mux(C, C.AND(u[15], um), lo, list(u)))
        return [gc, uc]

    return build


SiluMulClampIn = EW.specialized("SiluMulClampIn", 1, ["LIMIT"], silu_mul_clamp_in,
                                lambda S: ((("g", BF16), ("u", BF16)), Tuple(BF16, BF16)),
                                doc="silu_and_mul_with_clamp's clamps on bits: the gate above LIMIT, the up to [-LIMIT, LIMIT], "
                                    "a NaN LIMIT")


def _silu_mul_clamp_bf16(B, S, g, u):
    """`SiluMulClampBf16_v1` on bits: `SiluMulClampIn_v1`, then `SiluMulBf16_v4`."""
    c = B.call(bind(SiluMulClampIn, LIMIT=S.LIMIT), g, u)
    return B.call(BSL.SiluMulBf16Expf, c[0], c[1])


SiluMulClampBf16 = EW.on_bits("SiluMulClampBf16", 2, ("LIMIT",), lambda S: ((("g", BF16), ("u", BF16)), BF16),
                              _silu_mul_clamp_bf16, D.SiluMulClampBf16)


def _silu_mul_clamp(B, S, gu):
    """`SiluMulClamp_v1` on bits: ``out[i] = SiluMulClampBf16_v2(gu[i], gu[I + i])``."""
    return B.batch(bind(SiluMulClampBf16, LIMIT=S.LIMIT), gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


SiluMulClamp = EW.on_bits("SiluMulClamp", 2, ("I", "LIMIT"), lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
                          _silu_mul_clamp, D.SiluMulClamp)
