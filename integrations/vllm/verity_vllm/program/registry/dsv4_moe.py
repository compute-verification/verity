"""DeepSeek-V4's routers and clamped SwiGLU (deepseek-ai/DeepSeek-V4-Pro on vLLM d9105ea80, sm_90) as registered Definitions.

    MufuLg2Mantissa_v1                  (x: F32) -> F32                                     MUFU.LG2 of 1.m, m = x's mantissa
    Ex2Approx_v1                        (t: F32) -> F32                                     ex2.approx.f32 as ptxas lowers it
    Dsv4HashWeight_v1                   (x: F32) -> F32                                     the hash kernel's sqrtsoftplus weight
    MoeRouterHash_v1{E,SCALE}           (logits: Array<E, F32>, ids: Array<6, I32>) -> Array<6, F32>
    Dsv4SqrtSoftplus_v1                 (x: F32) -> F32                                     dsv4_topk's sqrtsoftplus score
    MoeRouterSqrtSoftplus_v1{E,SCALE}   (logits, bias: Array<E, F32>) -> (Array<6, F32>, Array<6, I32>)
    SiluMulClampBf16_v1{LIMIT}          (g, u: BF16) -> BF16                                silu_and_mul_with_clamp's element
    SiluMulClamp_v1{I,LIMIT}            (gu: Array<2I, BF16>) -> Array<I, BF16>

The model's MoE layers route with sqrtsoftplus scores ``sqrt(softplus(x))`` of the f32 gate logits, six experts of E = 384,
renormalised and scaled by ``routed_scaling_factor`` (SCALE, 2.5).  Layers 0-2 take their experts from the checkpoint's table
``tid2eid[token]`` (`fused_topk_bias` -> `ops.topk_hash_softplus_sqrt`, the CUDA kernel `dsv4HashTopkSoftplusSqrt`); the others
take the top six of ``score + e_score_correction_bias`` (`dsv4_topk`, a Triton kernel).  Both are written here as the kernels
compute them on sm_90: nvcc's SASS for the CUDA kernel (no fast math) and Triton 3.7.1's PTX and SASS for `_dsv4_topk_kernel` at
E = 384 (BLOCK_N 512, one warp), read offline.

`MufuLg2Mantissa_v1` is MUFU.LG2 (``lg2.approx.f32``) of the number 1.m in [1, 2), m the 23 mantissa bits of x; the rest of x is
not read.  The unit's quadratic interpolator, with integer coefficients, reproduces all 2^23 words measured on an RTX PRO 6000
(cc 12.0; `LG2_MEASURED`): a segment g = m >> 17 and the offset u below it give ``G = (C0[g] + C1[g] u + C2[g] Q(u)) >> 2``,
Q(u) the truncated square MUFU.EX2's interpolator uses (columns >= 19 of u^2's folded partial products, core's
`verity.ml.circuits.ops.boolean.mufu`), and the word is G 2^-36 rounded toward zero; m = 0 is +0, where the interpolant is 7377.

`Ex2Approx_v1` is ``ex2.approx.f32`` (not .ftz): ptxas computes ``t < -126 ? MUFU.EX2(t * 0.5)^2 : MUFU.EX2(t)`` (FMUL,
subnormals kept), the form `registry.mla`'s `ex2_approx` builds on bits.

`Dsv4HashWeight_v1` is one weight of `dsv4HashTopkSoftplusSqrt` for a text token, ``sqrtf(fmaxf(x, 0.f) + __logf(1.f +
__expf(-fabsf(x))))`` with a NaN made +0.  Its SASS:

    FMUL  t = |x| * -0x3FB8AA3B; Ex2Approx; FADD y = e + 1      __expf(-|x|), then 1 + it
    MUFU.LG2 y                                                   __logf(y) = lg2(y) * ln 2: y in [1, 2] or NaN
    FFMA  v = lg2 * 0x3F317218 + fmaxf(x, 0)                     fmaxf: FMNMX, a NaN x gives +0
    sqrt.rn v                                                    MUFU.RSQ, Newton, the slow path: IEEE (`PouwF32SqrtRn_v1`)
    FSETP.GTU |r| > inf; FSEL                                    isnan(r) -> +0

so MUFU.LG2 reads only [1, 2] and NaN: lg2(2) = 1.0 (2^1 is exact on the unit, as every binade above 1 the probe swept) and a
NaN is 0x7FFFFFFF; on [1, 2) it is `MufuLg2Mantissa_v1`.  `MoeRouterHash_v1` is the kernel's output for one row: lane k < 6
holds the weight of expert ``ids[k]`` (an id in [0, E): the table's values), the butterfly xor 16 .. 1 adds the lanes (+0 in
lanes 6 .. 31; `moe._lane0_sum`'s tree, exact here since no weight is -0 or NaN), and ``out[k] = weight * SCALE / (sum > 0 ?
sum : 1)`` (FMUL, then ``div.rn``; the GPU's NaN word, F32Mul_v2 / F32Div_v2).  The ids are the table's, written unchanged.
Image tokens (``bias_vl`` with ``image_sentinel_lo``) and padding rows (``VLLM_MOE_SKIP_PADDING``) take other branches.

`Dsv4SqrtSoftplus_v1` is `_dsv4_topk_kernel`'s ``tl.sqrt(tl.where(x > 20, x, tl.log(1 + tl.exp(x))))``: ``mul.f32 x * log2 e``
and `Ex2Approx_v1`, ``add.f32 1 + e``, the inlined libdevice ``__nv_logf`` (`NvLogf_v2`: its argument is at least 1, inf or
NaN, so the subnormal prescale is never taken and the FTZ polynomial is libdevice's), the select, then ``sqrt.approx.ftz.f32``
(`MufuSqrtFtz_v1`).  `MoeRouterSqrtSoftplus_v1` is the kernel's row:

    current = score + bias (add.f32), a NaN made -1e30 (0xF149F2CA)
    six slots: the expert of the largest current, the lowest index among equals (``tl.max``, then ``redux.sync.min.s32`` of the
               indices that equal it: `RouterArgmaxStep_v1`'s strict scan, as no current is NaN or -0), its score (a sum of the
               score and zeros, which is the score), and its current set to -inf (so a row with fewer than six currents above
               -inf picks an expert again)
    sum = ((w0 + w4) + w2) + ((w1 + w5) + w3)                     ``tl.sum`` of the eight slots, slots 6 and 7 +0
    out[k] = w[k] * div.full(SCALE, sum > 0 ? sum : 1)            FMUL(MUFU.RCP(scaled divisor), scaled SCALE), then FMUL

A score is +0, at least 2^-12, +inf or 0x7FFFFFFF, so the divisor is 1, a normal number or +inf; at +inf the quotient is +0 and an
infinite score's product the GPU's NaN (F32Mul_v2).  The ids are int32 (vLLM's int64 / uint32 ids hold the same values).

`SiluMulClampBf16_v1{LIMIT}` is `act_and_mul_kernel<bf16, silu_kernel, act_first, HAS_CLAMP>` (`_C.silu_and_mul_with_clamp`,
alpha 1, beta 0), the routed experts' activation at TP8 (Marlin MXFP4 -> `apply_moe_activation` -> `silu_and_mul_with_clamp`
with no expert map) and the shared expert's (`SiluAndMulWithClamp`): ``g = bf16(fminf(f32(g), LIMIT))``, ``u = fmaxf(fminf(f32(u),
LIMIT), -LIMIT)`` (a NaN gate or up is LIMIT: fminf yields the other operand), then `act.SiluMulBf16_v2`'s element, the vec and
scalar paths alike.  LIMIT is a positive bf16-exact float (10.0), so the clamped up is a bf16 value and `SiluMulBf16_v2` reads it
exactly; ``alpha = 1`` makes ``-f * alpha`` the ``-f`` of `silu_and_mul`.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.scalar import F32_NAN_PTX, F32Eq, F32Fabs, F32Fmaxf, F32Fminf
from verity.core.protocols.pouw.circuit.words import PouwF32SqrtRn
from verity_vllm.program.registry import act, b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.moe import F32GtStrict, MoeRouterPick, RouterArgmaxStep, _array, _lane0_sum
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity_vllm.program.registry.topp_word_gates import NvLogfV2

__all__ = ["Dsv4HashWeight", "Dsv4SqrtSoftplus", "Ex2Approx", "LG2_COEFFS", "LG2_MEASURED", "MoeRouterHash", "MoeRouterSqrtSoftplus",
           "MufuLg2Mantissa", "SiluMulClamp", "SiluMulClampBf16", "TOPK", "limit_bits", "lg2_interpolant", "mufu_lg2_mantissa_bits",
           "truncated_square"]

#: the slots both kernels write (their constant 6)
TOPK = 6

LOG2E, NEG_LOG2E, LN2 = 0x3FB8AA3B, 0xBFB8AA3B, 0x3F317218
MINUS_126, HALF, TWO, TWENTY = 0xC2FC0000, 0x3F000000, 0x40000000, 0x41A00000
NAN_SCORE = 0xF149F2CA       # -1e30

# --- MUFU.LG2 on [1, 2) ---------------------------------------------------------------------------------------------------------

LG2_OFFSET_BITS, LG2_SQUARE_FROM, LG2_SHIFT, LG2_SCALE = 17, 19, 2, -36
#: (C0, C1, C2) per segment g = m >> 17
LG2_COEFFS = (
    (29509, 47272, -1454), (6148432709, 46545, -1410), (12203000645, 45840, -1370), (18166489925, 45156, -1330),
    (24041640773, 44492, -1292), (29831013189, 43847, -1254), (35537077061, 43221, -1220), (41162191685, 42612, -1184),
    (46708667205, 42020, -1152), (52178604869, 41444, -1118), (57574130501, 40884, -1088), (62897230661, 40339, -1060),
    (68149822277, 39809, -1036), (73333732165, 39292, -1008), (78450783045, 38788, -982), (83502646085, 38297, -958),
    (88490939205, 37818, -932), (93417288517, 37351, -910), (98283180869, 36896, -890), (103090078533, 36451, -866),
    (107839419205, 36018, -850), (112532534085, 35594, -828), (117170770757, 35180, -808), (121755394885, 34776, -792),
    (126287618885, 34380, -770), (130768622405, 33994, -754), (135199564613, 33616, -736), (139581543237, 33247, -722),
    (143915623237, 32886, -708), (148202869573, 32532, -692), (152444248901, 32186, -678), (156640731973, 31847, -662),
    (160793285445, 31516, -652), (164902798149, 31191, -638), (168970179397, 30872, -622), (172996260677, 30561, -614),
    (176981869381, 30255, -600), (180927816517, 29955, -586), (184834904901, 29662, -578), (188703847237, 29374, -566),
    (192535413573, 29091, -552), (196330328901, 28814, -542), (200089273157, 28542, -532), (203812897605, 28276, -524),
    (207501906757, 28014, -514), (211156931397, 27757, -506), (214778549061, 27505, -498), (218367394629, 27257, -488),
    (221924049733, 27013, -476), (225449096005, 26774, -468), (228943086405, 26540, -464), (232406541125, 26309, -454),
    (235840037701, 26082, -446), (239244047173, 25859, -438), (242619102021, 25640, -432), (245965656901, 25424, -422),
    (249284195141, 25213, -418), (252575208261, 25004, -408), (255839150917, 24799, -402), (259076424517, 24598, -398),
    (262287512389, 24399, -390), (265472783173, 24204, -384), (268632675141, 24012, -378), (271767585605, 23823, -372),
)
#: the measurement the coefficients reproduce: ``lg2.approx.ftz.f32`` of ``0x3F800000 | m`` for every m (`registry.mla_capture`'s
#: probe), the SHA-256 of the 2^23 little-endian u32 words
LG2_MEASURED = {"device": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "cc": "12.0", "run": "r20261004-060203-ba52",
                "run_record": "art:27ed13d53fac0cf67e3e93693d9ad2b6dd75824401ca5b9f9d48acda27602066", "words": 1 << 23,
                "sha256": "7fd9be9a924e40a0ee18ba20e7f5158135112947597c8dd86131d3d4d8f9dbfa"}


def truncated_square(u):
    """Q(u) for a 17-bit offset (an int, or an array of integers wide enough for 2^16): the folded partial products of u^2 (u_a
    at column 2a, u_a u_b at a + b + 1 for a < b) in columns >= 19, shifted right by 19."""
    nb, p = LG2_OFFSET_BITS, LG2_SQUARE_FROM
    bit = [(u >> a) & 1 for a in range(nb)]
    q = u * 0
    for a in range(nb):
        if 2 * a >= p:
            q = q + (bit[a] << (2 * a - p))
        for b in range(max(a + 1, p - a - 1), nb):
            q = q + ((bit[a] & bit[b]) << (a + b + 1 - p))
    return q


def lg2_interpolant(m: int) -> int:
    """G at the mantissa m: ``(C0[g] + C1[g] u + C2[g] Q(u)) >> 2``, g = m >> 17, u = m mod 2^17."""
    c0, c1, c2 = LG2_COEFFS[m >> LG2_OFFSET_BITS]
    u = m & ((1 << LG2_OFFSET_BITS) - 1)
    return (c0 + c1 * u + c2 * truncated_square(u)) >> LG2_SHIFT


def mufu_lg2_mantissa_bits(x: int) -> int:
    """MUFU.LG2 of 1.m, m = x & 0x7FFFFF: G 2^-36 rounded toward zero, and +0 at m = 0."""
    m = x & 0x7FFFFF
    if m == 0:
        return 0
    G = lg2_interpolant(m)
    e = G.bit_length() - 1
    frac = G << (23 - e) if e <= 23 else G >> (e - 23)
    return (e + 127 + LG2_SCALE) << 23 | frac & 0x7FFFFF


@primitive("MufuLg2Mantissa", 1, [("x", F32)], F32,
           conformance="exact-model-tested: lg2.approx.ftz.f32 on every 1.m of [1, 2), 8,388,608 words measured on the RTX PRO 6000 "
                       "(cc 12.0; run r20261004-060203-ba52) equal to the interpolant; cc 9.0 not yet captured "
                       "(tests/properties/dsv4_moe_capture_gpu.py)",
           # a measured MUFU table in substance, but not on `verity.ml.circuits.library`'s versioned list, so the guard lists it
           tables=(("lg2_coeffs", "registered"),))
def MufuLg2Mantissa(x: int) -> int:
    """MUFU.LG2 (``lg2.approx.f32``) of 1.m, m = x's 23 mantissa bits: the unit's quadratic interpolator rounded toward zero on
    the 2^-36 grid, +0 at m = 0."""
    return mufu_lg2_mantissa_bits(x)


# --- the CUDA hash router (layers 0-2) -------------------------------------------------------------------------------------------

C_MINUS_126, C_HALF, C_TWO, C_TWENTY = (b1.const(32, c) for c in (MINUS_126, HALF, TWO, TWENTY))
C_LOG2E, C_NEG_LOG2E, C_LN2, C_NAN_SCORE = (b1.const(32, c) for c in (LOG2E, NEG_LOG2E, LN2, NAN_SCORE))
ONE32 = b1.f32c(1.0)
PTX_NAN = b1.const(32, F32_NAN_PTX)

_UNCAPTURED = "not-yet-established: vLLM d9105ea80's Hopper lowering read offline; no GPU capture yet (tests/properties/dsv4_moe_capture_gpu.py)"


@composite("Ex2Approx", 1, [], lambda S: ((("t", F32),), F32), conformance=_UNCAPTURED)
def Ex2Approx(B, S, t):
    """``ex2.approx.f32`` (not .ftz): ``t < -126`` (ordered) halves t before MUFU.EX2 and squares the result (FMUL)."""
    small = B.call(F32GtStrict, B.call(C_MINUS_126), t)
    y = B.call(P.MufuEx2Ftz, B.call(P.SelectF32, small, B.call(P.F32Mul, t, B.call(C_HALF)), t))
    return B.call(P.SelectF32, small, B.call(P.F32Mul, y, y), y)


@composite("Dsv4HashWeight", 1, [], lambda S: ((("x", F32),), F32), conformance=_UNCAPTURED)
def Dsv4HashWeight(B, S, x):
    """``sqrtf(fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x))))``, a NaN +0, as nvcc compiles it for sm_90: ``y = Ex2Approx(|x| *
    -log2 e) + 1``, MUFU.LG2 of y (1.0 at y = 2, 0x7FFFFFFF at a NaN, `MufuLg2Mantissa_v1` on [1, 2)), ``fma(lg2, ln 2, fmaxf(x,
    0))``, sqrt.rn."""
    zero, one = B.call(b1.ZERO32), B.call(ONE32)
    y = B.call(P.F32Add, B.call(Ex2Approx, B.call(P.F32Mul, B.call(F32Fabs, x), B.call(C_NEG_LOG2E))), one)
    lg2 = B.call(P.SelectF32, B.call(F32Eq, y, y), B.call(MufuLg2Mantissa, y), B.call(PTX_NAN))
    lg2 = B.call(P.SelectF32, B.call(F32Eq, y, B.call(C_TWO)), one, lg2)
    r = B.call(PouwF32SqrtRn, B.call(P.F32Fma, lg2, B.call(C_LN2), B.call(F32Fmaxf, x, zero)))
    return B.call(P.SelectF32, B.call(F32Eq, r, r), r, zero)


def _denominator(B, total):
    """``total > 0 ? total : 1``."""
    return B.call(P.SelectF32, B.call(F32GtStrict, total, B.call(b1.ZERO32)), total, B.call(ONE32))


@composite("MoeRouterHash", 1, ["E", "SCALE"],
           lambda S: ((("logits", Array(S.E, F32)), ("ids", Array(TOPK, I32))), Array(TOPK, F32)), conformance=_UNCAPTURED)
def MoeRouterHash(B, S, logits, ids):
    """`dsv4HashTopkSoftplusSqrt`'s weights for one text token's row of f32 router logits and its six table ids (each in [0, E)):
    ``w[k] = Dsv4HashWeight_v1(logits[ids[k]])``, the warp's sum (`moe._lane0_sum`), ``w[k] * SCALE / (sum > 0 ? sum : 1)``."""
    w = [B.call(Dsv4HashWeight, B.call(bind(MoeRouterPick, E=S.E), logits, ids[k])) for k in range(TOPK)]
    denom = _denominator(B, _lane0_sum(B, w))
    scale = B.call(b1.f32c(S.SCALE))
    return _array([B.call(P.F32DivV2, B.call(P.F32MulV2, v, scale), denom) for v in w])


# --- the Triton router (dsv4_topk) -----------------------------------------------------------------------------------------------

@composite("Dsv4SqrtSoftplus", 1, [], lambda S: ((("x", F32),), F32), conformance=_UNCAPTURED)
def Dsv4SqrtSoftplus(B, S, x):
    """``tl.sqrt(tl.where(x > 20, x, tl.log(1 + tl.exp(x))))``: ``Ex2Approx_v1(x * log2 e) + 1``, `NvLogf_v2`, the select,
    `MufuSqrtFtz_v1`."""
    y = B.call(P.F32Add, B.call(ONE32), B.call(Ex2Approx, B.call(P.F32Mul, x, B.call(C_LOG2E))))
    big = B.call(F32GtStrict, x, B.call(C_TWENTY))
    return B.call(P.MufuSqrtFtz, B.call(P.SelectF32, big, x, B.call(NvLogfV2, y)))


def _first_max(B, s: list, c0, c1):
    """The lowest index of the largest of s (no NaN): `RouterArgmaxStep_v1`'s scan over s[1 .. E-2], then the last step's index."""
    if len(s) == 1:
        return c0
    carry = tuple_of(s[0], c0, c1)
    if len(s) > 2:
        carry = B.scan(RouterArgmaxStep, carry, xs=(_array(s[1:-1]),))[0]
    return B.call(P.SelectI32, B.call(F32GtStrict, s[-1], carry[0]), carry[2], carry[1])


@composite("MoeRouterSqrtSoftplus", 1, ["E", "SCALE"],
           lambda S: ((("logits", Array(S.E, F32)), ("bias", Array(S.E, F32))), Tuple(Array(TOPK, F32), Array(TOPK, I32))),
           conformance=_UNCAPTURED)
def MoeRouterSqrtSoftplus(B, S, logits, bias):
    """`dsv4_topk(logits, bias, rsf=SCALE)` for one token row of f32 router logits and the f32 `e_score_correction_bias`: per
    expert the score `Dsv4SqrtSoftplus_v1` and ``current = score + bias`` (a NaN -1e30); six slots of the first maximum of
    current, each slot's score, the winner's current set to -inf; then ``w[k] * div.full(SCALE, sum > 0 ? sum : 1)``.  Returns
    (topk_weights, topk_ids) as the kernel writes them."""
    if S.E < TOPK:
        raise ValueError(f"E must be at least {TOPK}")
    w = B.batch(Dsv4SqrtSoftplus, logits, axes=(0,))
    nan_score, neg_inf = B.call(C_NAN_SCORE), B.call(b1.NEG_INF32)
    cur = []
    for i in range(S.E):
        c = B.call(P.F32Add, w[i], bias[i])
        cur.append(B.call(P.SelectF32, B.call(F32Eq, c, c), c, nan_score))
    consts = [B.call(b1.const(32, i)) for i in range(S.E)]
    sel, ids = [], []
    for k in range(TOPK):
        e = _first_max(B, cur, consts[0], consts[1])
        hit = {i: B.call(P.I32Eq, consts[i], e) for i in range(1 if k == TOPK - 1 else 0, S.E)}
        v = w[0]
        for i in range(1, S.E):
            v = B.call(P.SelectF32, hit[i], w[i], v)
        sel.append(v)
        ids.append(e)
        if k < TOPK - 1:
            cur = [B.call(P.SelectF32, hit[i], neg_inf, cur[i]) for i in range(S.E)]
    denom = _denominator(B, _lane0_sum(B, sel))
    q = B.call(P.F32Mul, B.call(P.DivFullScaleA, B.call(b1.f32c(S.SCALE)), denom), B.call(P.DivFullRcp, denom))
    return tuple_of(_array([B.call(P.F32MulV2, v, q) for v in sel]), _array(ids))


# --- the clamped SwiGLU ----------------------------------------------------------------------------------------------------------

def limit_bits(limit: float) -> int:
    """The f32 word of a clamp limit, which must be a positive finite bf16 value."""
    bits = P.bits_of(limit)
    if float(P.f32_of(bits)) != float(limit) or bits & 0xFFFF or not 0 < bits < 0x7F800000:
        raise ValueError(f"LIMIT must be a positive finite bf16 value, got {limit!r}")
    return bits


@composite("SiluMulClampBf16", 1, ["LIMIT"], lambda S: ((("g", BF16), ("u", BF16)), BF16), conformance=_UNCAPTURED)
def SiluMulClampBf16(B, S, g, u):
    """``SiluMulBf16_v2(bf16(fminf(f32(g), LIMIT)), bf16(fmaxf(fminf(f32(u), LIMIT), -LIMIT)))``: both casts exact."""
    lim = limit_bits(S.LIMIT)
    hi, lo = B.call(b1.const(32, lim)), B.call(b1.const(32, lim | 0x80000000))
    gc = B.call(P.F2fpBf16, B.call(F32Fminf, B.call(P.Bf16ToF32, g), hi))
    uc = B.call(P.F2fpBf16, B.call(F32Fmaxf, B.call(F32Fminf, B.call(P.Bf16ToF32, u), hi), lo))
    return B.call(act.SiluMulBf16, gc, uc)


@composite("SiluMulClamp", 1, ["I", "LIMIT"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)), conformance=_UNCAPTURED)
def SiluMulClamp(B, S, gu):
    """`_C.silu_and_mul_with_clamp(out, gu, LIMIT, 1.0, 0.0)` for one row: ``out[i] = SiluMulClampBf16_v1(gu[i], gu[I + i])``."""
    return B.batch(bind(SiluMulClampBf16, LIMIT=S.LIMIT), gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))
