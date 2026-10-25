"""[DSV4] The head-512 compressor of DeepSeek-V4-Pro's sparse MLA as an H200 runs it (sm_90, TP8): vLLM d9105ea80's CuTe DSL
`compress_norm_rope_store_cutedsl` (vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py), which writes the
compressed rows `flashmla`'s decode and `flashmla_prefill`'s gather read on the ratio-4 and ratio-128 layers.  Line numbers are
at that commit ("cutedsl" that file, "compressor" `deepseek_v4/compressor.py`); "the PTX" and "the SASS" are the CuTe DSL's
sm_90a output for each kernel at the model's constants (head 512, rotary 64, quant block 64, `fp8_ds_mla`: token stride 576,
8 scale bytes).

The call (compressor:432-454; head 512 on CUDA always takes the CuTe DSL, compressor:405-414; the cache is the `fp8_ds_mla`
uint8 layout, so `store_full_kv` is False, compressor:398).  Before it, `save_partial_states` writes the token's state row:
kv a copy of `fused_wkv_wgate`'s f32 output and score `add.f32(score, ape[pos % ratio])` (`dsv4_indexer.Dsv4CompressorScore_v1`
at W = coff * 512: 1024 on ratio 4, 512 on ratio 128).  A token is compressed iff its slot is valid (>= 0), it has a request
and a cache slot (>= 0), and it ends a window, (pos + 1) % ratio == 0; on ratio 128 off a full CUDA graph the call returns
early when no token of the step ends one (compressor:366-373).  Its row goes to cache slot `kv_slot`: page p = slot // block,
offset o; values at p * stride0 + o * 576 (448 E4M3 NoPE bytes, then 64 bf16 RoPE elements), scales at p * stride0 +
block * 576 + o * 8 (7 UE8M0 bytes, one per NoPE group of 64, and a pad byte 0).

Ratio 4 (`SparseAttnCompressNormRopeStoreC4Kernel`, cutedsl:75-389, one CTA of 64 threads per token, thread t columns
8t..8t+7): the window is the 8 state rows at positions pos - 7 .. pos, row r reading kv[(r // 4) * 512 + j] and score
[1024 + (r // 4) * 512 + j] for column j; a row at a position below 0 is skipped (cutedsl:233, only at pos = 3: the last 4
rows), not read as score -inf.  Per column, from m = -inf, l = o = +0, each row in ascending order (cutedsl:259-275):

    m' = max.f32(m, s);  a = ex2.approx.ftz(mul(sub(m, m'), 0x3FB8AA3B));  b = ex2.approx.ftz(mul(sub(s, m'), 0x3FB8AA3B))
    l  = fma.rn(l, a, b);  o = fma.rn(o, a, mul(kv, b));  m = m'
    x  = div.rn(o, l)                                                                                   (cutedsl:280)

`Dsv4MlaC4Column_v1{ROWS}`.  The SASS's ex2 is MUFU.EX2 of the FMUL, no range reduction (`MufuEx2Ftz_v1`).

Ratio 128 (`SparseAttnCompressC128Block8Kernel`, cutedsl:931-1184, then `SparseAttnNormRopeStoreKernel` on its f32
`compressed_kv`, cutedsl:1245-1435): warp w of 8 holds the 16 rows 16w..16w+15 of the window pos - 127 .. pos (state blocks of
8 rows, cutedsl:1080-1117).  Per column, warp w's m_w = max.f32 left fold from -inf, then over its rows in ascending order
e = ex2.approx.ftz(mul(sub(s, m_w), 0x3FB8AA3B)), sum_w = add.f32(e, sum_w) and prod_w = fma.rn(kv, e, prod_w) from +0
(cutedsl:1119-1130); across the warps (cutedsl:1137-1184, xor butterflies over 8 lanes, `dsa.halving_tree`) g = the max tree
of m_w, c_w = ex2.approx.ftz(mul(sub(m_w, g), 0x3FB8AA3B)), S = the add tree of mul(sum_w, c_w), P = that of mul(prod_w, c_w),
x = div.rn(P, S): `Dsv4MlaC128Column_v1{W, R}`.

The row, both kernels (cutedsl:277-389 and 1350-1435): RMSNorm, then the last 64-group rotated and stored bf16, the other 7
quantised.  The sum of squares (fma.rn and add.f32):

    ratio 4    thread t: from +0, fma(x, x, .) over its 8 columns ascending; the warp's xor butterfly 16..1; then
               add(add(+0, warp 0), warp 1) (cutedsl:278-298)
    ratio 128  thread t (columns 2t, 2t+1): fma(x0, x0, mul(x1, x1)); the warp's butterfly 16..1; the 8 warps' partials
               a butterfly 4, 2, 1 (cutedsl:1354-1379)

then rrms = rsqrt.approx.ftz(fma.rn(sum, 2^-9, eps)) (the DSL's `sum / 512 + eps` at fastmath; `RsqrtApprox_v1`, whose
operand is at least eps, so never subnormal) and n = mul(mul(x, rrms), cvt.f32.bf16(w)) (`Dsv4MlaRmsNorm_v1`).  A NoPE group
of 64 (`Dsv4MlaQuantGroup_v1`): q = the bf16 rounding of n widened (F2FP, NaN 0x7FFF), amax = max.f32 over abs.f32(q) (ratio 4
per thread from +0 then 8 lanes, ratio 128 per thread then 32 lanes: max over sign-cleared words is order-free, and an all-NaN
group's +0 or NaN gives the same scale), s = max.f32(div.rn(amax, 448), 1e-4), the stored byte e = ((bits(s) + 0x7FFFFF) >> 23)
& 0xFF (`Ue8m0CeilByte_v1`), inv = (254 - e) << 23 as a word (`Ue8m0InvScale_v1`: e = 255 gives -inf), and each byte
cvt.rn.satfinite.e4m3x2(min(max.f32(mul(q, inv), -448), 448)) (the min a setp.ge / selp, never on a NaN; a NaN q gives -448).
The RoPE group (`Dsv4MlaRope_v1`): pair j of n's last 64, (x0, x1) = (n[448 + 2j], n[449 + 2j]), cos = cs[j], sin = cs[32 + j]
of the f32 `cos_sin_cache` row at the compressed position (pos - 3, or pos - 127), as the SASS contracts the DSL's
`x0 cos - x1 sin`, `x0 sin + x1 cos`: (fma(x0, cos, -mul(x1, sin)), fma(x1, cos, mul(x0, sin))), each F2FP to bf16.

Every add, sub, mul, fma and div here is the GPU's (`F32Add_v2` .. `F32Div_v2`: each NaN result 0x7FFFFFFF).

Registered functions (each `_v1`; `boolean_dsv4_compressor` has their Boolean `_v2`):
  Ue8m0CeilByte, Ue8m0InvScale            primitives: the scale's byte from its f32 word, the inverse scale from the byte
  Dsv4MlaC4Column{ROWS}                    (kv, score: ROWS f32) -> f32
  Dsv4MlaC128Column{W,R}                   (kv, score: W*R f32) -> f32
  Dsv4MlaRmsNorm{RATIO,D,QB,EPS}           (x: D f32, w: D bf16) -> D f32
  Dsv4MlaQuantGroup{QB}                    (x: QB f32) -> (q: QB e4m3, e: e8m0)
  Dsv4MlaRope{ROPE}                        (x: ROPE f32, cs: ROPE f32) -> ROPE bf16
  Dsv4MlaNormRopeQuant{RATIO,D,QB,EPS}     (x: D f32, w: D bf16, cs: QB f32) -> (nope: D-QB e4m3, rope: QB bf16, scale: D/QB e8m0)
  Dsv4MlaCompressC4{ROWS,D,QB,EPS}         (kv, score: ROWS x D f32, w, cs) -> the row
  Dsv4MlaCompressC128{W,R,D,QB,EPS}        (kv, score: W*R x D f32, w, cs) -> the row

A row is what `flashmla.FlashMlaKvRow_v1{DN = D - QB, DR = QB, QT = QB}` reads: the NoPE bytes, the RoPE elements and the
scale bytes, the last of them the pad 0.  `DSV4_COMPRESSOR` holds the model's statics.  D = 512 and QB = 64 are the kernels';
a smaller power-of-two D and QB is the same body for the checks (ratio 4: warps of min(32, D/8) threads; ratio 128: warps of
QB/2 threads, one per group).
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops.fp32 import F32DivV2
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.scalar import F32Neg
from verity_vllm.program.kernels import dsv4_compressor_twin
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import dsv4_indexer as I
from verity_vllm.program.registry.flashmla import E8M0
from verity_vllm.program.registry.prims import F32

M32 = 0xFFFFFFFF
LOG2E, F_NINF, F_448, F_NEG448, MIN_SCALE = I.LOG2E, 0xFF800000, D.F_448, 0xC3E00000, D.F_1EM4

_CONF = ("model-untested: read from the CuTe DSL kernels and their PTX and SASS at the commits the module docstring pins; "
         "tests/properties/dsv4_compressor_capture_gpu.py is its hardware check")

#: the model's statics: head 512, one scale per 64, the ratio-4 window of 8 state rows, ratio 128's 8 warps of 16 rows, and
#: the config's `rms_norm_eps`
DSV4_COMPRESSOR = {"D": 512, "QB": 64, "ROWS": 8, "W": 8, "R": 16, "EPS": 1e-6}

#: the element types of a signature: these words, and their bits in `boolean_dsv4_compressor`
WORDS = SimpleNamespace(**vars(I.WORDS), E8M0=E8M0)


@primitive("Ue8m0CeilByte", 1, [("s", F32)], E8M0, conformance=_CONF)
def Ue8m0CeilByte(s: int) -> int:
    """The UE8M0 byte of a positive scale word, its exponent rounded up: `((bits(s) + 0x7FFFFF) >> 23) & 0xFF` (add.s32,
    shr.u32, and.b32; cutedsl:369, 1415)."""
    return (((s & M32) + 0x7FFFFF) & M32) >> 23 & 0xFF


@primitive("Ue8m0InvScale", 1, [("e", E8M0)], F32, conformance=_CONF)
def Ue8m0InvScale(e: int) -> int:
    """The inverse of the scale 2^(e - 127) as a word: `(254 - e) << 23` modulo 2^32 (cutedsl:370, 1416; the PTX's
    `0x7F000000 - ((bits(s) + 0x7FFFFF) & 0x7F800000)`): 2^(127 - e) for e <= 254, -inf for 255."""
    return ((254 - (e & 0xFF)) << 23) & M32


DSV4_COMPRESSOR_PRIMITIVES = [Ue8m0CeilByte, Ue8m0InvScale]

WORD = {**I.WORD, "div": F32DivV2, "ue8m0": Ue8m0CeilByte, "inv": Ue8m0InvScale}


class Ops(D.Ops):
    """`dsa.Ops` with ``byte(bits)``, a constant byte."""

    def __init__(self, B, defs: dict, word, neg, byte):
        super().__init__(B, defs, word, neg)
        self.byte = byte


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x), lambda bits: B.call(const(8, bits)))


def on_words(B, S, signature, body, *colls) -> Coll:
    """`dsv4_indexer.on_words` over this module's steps."""
    params, ret = signature(S)
    parts = [[c[i] for i in range(t.n)] if isinstance(t, Array) else [c] for c, (_, t) in zip(colls, params)]
    return Coll(ret, concat([e.refs for e in body(word_ops(B), S, *parts)]))


def is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def check_row_statics(D_: int, QB: int) -> None:
    if not (is_pow2(D_) and is_pow2(QB) and 8 <= QB < D_):
        raise ValueError(f"Dsv4Mla: a row of power-of-two D with power-of-two groups 8 <= QB < D (D={D_}, QB={QB})")


# --- the columns -------------------------------------------------------------------------------------------------------------

def c4_column_sig(T):
    return lambda S: ((("kv", Array(S.ROWS, T.F32)), ("score", Array(S.ROWS, T.F32))), T.F32)


def c4_column(o, S, kv, score) -> list:
    """`Dsv4MlaC4Column_v1`'s body: the online softmax over the rows, then the division."""
    if S.ROWS < 1:
        raise ValueError(f"Dsv4MlaC4Column: ROWS={S.ROWS}")
    m, l, acc = o.word(F_NINF), o.word(0), o.word(0)
    for k, s in zip(kv, score):
        m2 = o.fmaxf(m, s)
        a = o.ex2(o.mul(o.sub(m, m2), o.word(LOG2E)))
        b = o.ex2(o.mul(o.sub(s, m2), o.word(LOG2E)))
        l = o.fma(l, a, b)
        acc = o.fma(acc, a, o.mul(k, b))
        m = m2
    return [o.div(acc, l)]


@composite("Dsv4MlaC4Column", 1, ["ROWS"], c4_column_sig(WORDS), conformance=_CONF)
def Dsv4MlaC4Column(B, S, kv, score):
    """One column of a ratio-4 compressed row (cutedsl:259-280): from m = -inf, l = o = +0, per row in ascending order m' =
    `max.f32(m, s)`, a = `ex2.approx.ftz((m - m') * log2 e)`, b = `ex2.approx.ftz((s - m') * log2 e)`, l = `fma(l, a, b)`, o =
    `fma(o, a, kv * b)`; then `div.rn(o, l)`.  ROWS = 8, or 4 at position 3 (its first four rows are skipped)."""
    return on_words(B, S, c4_column_sig(WORDS), c4_column, kv, score)


def check_c128_statics(W: int, R: int) -> None:
    if not (is_pow2(W) and R >= 1):
        raise ValueError(f"Dsv4MlaC128Column: a power-of-two count of warps W and R >= 1 rows each (W={W}, R={R})")


def c128_column_sig(T):
    return lambda S: ((("kv", Array(S.W * S.R, T.F32)), ("score", Array(S.W * S.R, T.F32))), T.F32)


def c128_column(o, S, kv, score) -> list:
    """`Dsv4MlaC128Column_v1`'s body: each warp's max, sum and product, then their combination."""
    check_c128_statics(S.W, S.R)
    ms, sums, prods = [], [], []
    for w in range(S.W):
        rows = range(w * S.R, (w + 1) * S.R)
        m = o.word(F_NINF)
        for r in rows:
            m = o.fmaxf(m, score[r])
        sm, pr = o.word(0), o.word(0)
        for r in rows:
            e = o.ex2(o.mul(o.sub(score[r], m), o.word(LOG2E)))
            sm = o.add(e, sm)
            pr = o.fma(kv[r], e, pr)
        ms.append(m)
        sums.append(sm)
        prods.append(pr)
    g = D.halving_tree(ms, o.fmaxf)
    c = [o.ex2(o.mul(o.sub(m, g), o.word(LOG2E))) for m in ms]
    total = D.halving_tree([o.mul(s, x) for s, x in zip(sums, c)], o.add)
    num = D.halving_tree([o.mul(p, x) for p, x in zip(prods, c)], o.add)
    return [o.div(num, total)]


@composite("Dsv4MlaC128Column", 1, ["W", "R"], c128_column_sig(WORDS), conformance=_CONF)
def Dsv4MlaC128Column(B, S, kv, score):
    """One column of a ratio-128 compressed row (cutedsl:1119-1184), over W warps of R rows: warp w's m_w = `max.f32` fold from
    -inf over its rows, then per row ascending e = `ex2.approx.ftz((s - m_w) * log2 e)`, sum_w = `e + sum_w`, prod_w =
    `fma(kv, e, prod_w)` from +0; g = the `max.f32` halving tree of the m_w, c_w = `ex2.approx.ftz((m_w - g) * log2 e)`, and
    `div.rn` of the add halving trees of `prod_w * c_w` and `sum_w * c_w`.  W = 8 and R = 16 are the kernel's."""
    return on_words(B, S, c128_column_sig(WORDS), c128_column, kv, score)


# --- the row: RMSNorm, quantisation, RoPE ----------------------------------------------------------------------------------

def check_norm_statics(RATIO: int, D_: int, QB: int, EPS: float) -> None:
    if RATIO not in (4, 128):
        raise ValueError(f"Dsv4MlaRmsNorm: RATIO={RATIO} names the kernel, 4 or 128")
    check_row_statics(D_, QB)
    D.check_eps(EPS)


def norm_sig(T):
    return lambda S: ((("x", Array(S.D, T.F32)), ("w", Array(S.D, T.BF16))), Array(S.D, T.F32))


def rms_norm(o, S, x, w) -> list:
    """`Dsv4MlaRmsNorm_v1`'s body: the kernel's sum of squares, rrms, then the scaled words."""
    check_norm_statics(S.RATIO, S.D, S.QB, S.EPS)
    if S.RATIO == 4:
        lanes = []
        for t in range(S.D // 8):
            acc = o.word(0)
            for v in x[8 * t:8 * t + 8]:
                acc = o.fma(v, v, acc)
            lanes.append(acc)
        width = min(32, len(lanes))
        total = o.word(0)
        for k in range(0, len(lanes), width):
            total = o.add(total, D.halving_tree(lanes[k:k + width], o.add))
    else:
        lanes = [o.fma(x[2 * t], x[2 * t], o.mul(x[2 * t + 1], x[2 * t + 1])) for t in range(S.D // 2)]
        width = S.QB // 2
        total = D.halving_tree([D.halving_tree(lanes[k:k + width], o.add) for k in range(0, len(lanes), width)], o.add)
    rrms = o.rsqrt(o.fma(total, o.word(D.f32_bits(1.0 / S.D)), o.word(D.f32_bits(S.EPS))))
    return [o.mul(o.mul(x[i], rrms), o.cvt(w[i])) for i in range(S.D)]


@composite("Dsv4MlaRmsNorm", 1, ["RATIO", "D", "QB", "EPS"], norm_sig(WORDS), conformance=_CONF)
def Dsv4MlaRmsNorm(B, S, x, w):
    """A compressed row's RMSNorm (cutedsl:277-303, 1350-1385): the sum of squares in kernel RATIO's order (ratio 4: per thread
    of 8 columns `fma(x, x, .)` from +0, the warp's add halving tree, then the warps' sums added to +0 in turn; ratio 128: per
    thread `fma(x0, x0, x1 * x1)`, the add halving tree of each warp of QB/2 threads, then of the warps), rrms =
    `rsqrt.approx.ftz(fma(sum, 1/D, eps))` (`RsqrtApprox_v1`: the operand is at least eps), and `mul(mul(x, rrms), f32(w))`."""
    return on_words(B, S, norm_sig(WORDS), rms_norm, x, w)


def quant_sig(T):
    return lambda S: ((("x", Array(S.QB, T.F32)),), Tuple(Array(S.QB, T.E4M3), T.E8M0))


def quant_group(o, S, x) -> list:
    """`Dsv4MlaQuantGroup_v1`'s body: the q bytes, then the scale byte."""
    if not is_pow2(S.QB):
        raise ValueError(f"Dsv4MlaQuantGroup: the amax butterfly takes a power-of-two group (QB={S.QB})")
    q = [o.cvt(o.bf16(v)) for v in x]
    s = o.fmaxf(o.div(D.halving_tree([o.fabs(v) for v in q], o.fmaxf), o.word(F_448)), o.word(MIN_SCALE))
    e = o.ue8m0(s)
    inv = o.inv(e)
    return [o.e4m3(o.fminf(o.fmaxf(o.mul(v, inv), o.word(F_NEG448)), o.word(F_448))) for v in q] + [e]


@composite("Dsv4MlaQuantGroup", 1, ["QB"], quant_sig(WORDS), conformance=_CONF)
def Dsv4MlaQuantGroup(B, S, x):
    """One NoPE group's UE8M0 quantisation (cutedsl:334-389, 1398-1435): q = `cvt.f32.bf16(cvt.rn.bf16x2.f32(x))`, s =
    `max.f32(div.rn(max |q|, 448), 1e-4)` (the max a halving tree over sign-cleared words: order-free), e = `Ue8m0CeilByte_v1(s)`,
    inv = `Ue8m0InvScale_v1(e)`, byte i = `cvt.rn.satfinite.e4m3x2(min(max.f32(q[i] * inv, -448), 448))`; the bytes, then e."""
    return on_words(B, S, quant_sig(WORDS), quant_group, x)


def rope_sig(T):
    return lambda S: ((("x", Array(S.ROPE, T.F32)), ("cs", Array(S.ROPE, T.F32))), Array(S.ROPE, T.BF16))


def rope(o, S, x, cs) -> list:
    """`Dsv4MlaRope_v1`'s body."""
    if not (S.ROPE > 0 and S.ROPE % 2 == 0):
        raise ValueError(f"Dsv4MlaRope: an even rotary width (ROPE={S.ROPE})")
    h, out = S.ROPE // 2, []
    for j in range(h):
        x0, x1, cos, sin = x[2 * j], x[2 * j + 1], cs[j], cs[h + j]
        out += [o.bf16(o.fma(x0, cos, o.neg(o.mul(x1, sin)))), o.bf16(o.fma(x1, cos, o.mul(x0, sin)))]
    return out


@composite("Dsv4MlaRope", 1, ["ROPE"], rope_sig(WORDS), conformance=_CONF)
def Dsv4MlaRope(B, S, x, cs):
    """The rotary group's GPT-J rotation and bf16 store (cutedsl:316-333, 1387-1397): pair j = (x[2j], x[2j+1]) becomes
    (`fma(x0, cos_j, -(x1 * sin_j))`, `fma(x1, cos_j, x0 * sin_j)`), each `cvt.rn.bf16x2.f32` (F2FP, NaN 0x7FFF); ``cs`` the f32
    cache row at the compressed position, [cos 0..ROPE/2-1 | sin 0..ROPE/2-1]."""
    return on_words(B, S, rope_sig(WORDS), rope, x, cs)


def row_type(S, T):
    return Tuple(Array(S.D - S.QB, T.E4M3), Array(S.QB, T.BF16), Array(S.D // S.QB, T.E8M0))


def nrq_sig(T):
    return lambda S: ((("x", Array(S.D, T.F32)), ("w", Array(S.D, T.BF16)), ("cs", Array(S.QB, T.F32))), row_type(S, T))


def norm_rope_quant(B, S, x, w, cs, defs, T) -> Coll:
    """`Dsv4MlaNormRopeQuant`'s body over the Definitions ``defs`` (norm, quant, rope, byte) and element types ``T``."""
    check_norm_statics(S.RATIO, S.D, S.QB, S.EPS)
    QB, G = S.QB, S.D // S.QB
    n = B.call(bind(defs.norm, RATIO=S.RATIO, D=S.D, QB=QB, EPS=S.EPS), x, w)
    quant = bind(defs.quant, QB=QB)
    parts = [B.call(quant, n[g * QB:(g + 1) * QB]) for g in range(G - 1)]
    rot = B.call(bind(defs.rope, ROPE=QB), n[(G - 1) * QB:S.D], cs)
    return tuple_of(Coll(Array(S.D - QB, T.E4M3), concat([p[0].refs for p in parts])), rot,
                    Coll(Array(G, T.E8M0), concat([p[1].refs for p in parts] + [defs.byte(B, 0).refs])))


@composite("Dsv4MlaNormRopeQuant", 1, ["RATIO", "D", "QB", "EPS"], nrq_sig(WORDS), conformance=_CONF)
def Dsv4MlaNormRopeQuant(B, S, x, w, cs):
    """A compressed row from its D f32 words as kernel RATIO stores it in `fp8_ds_mla`: `Dsv4MlaRmsNorm_v1`, then groups 0 ..
    D/QB - 2 `Dsv4MlaQuantGroup_v1{QB}` and the last `Dsv4MlaRope_v1{QB}`; the NoPE bytes, the RoPE elements, and the scale
    bytes with the pad byte 0 after them (group 0's thread writes it, cutedsl:384-389, 1429-1435)."""
    return norm_rope_quant(B, S, x, w, cs, WORD_DEFS, WORDS)


# --- the compressors -------------------------------------------------------------------------------------------------------

def compress_sig(T, rows):
    return lambda S: ((("kv", Array(rows(S), Array(S.D, T.F32))), ("score", Array(rows(S), Array(S.D, T.F32))),
                       ("w", Array(S.D, T.BF16)), ("cs", Array(S.QB, T.F32))), row_type(S, T))


def c4_rows(S) -> int:
    return S.ROWS


def c128_rows(S) -> int:
    return S.W * S.R


def compress(B, S, kv, score, w, cs, column, rows: int, ratio: int, defs, T) -> Coll:
    """A compressor's body: ``column`` (bound) over each of the D columns of the ``rows`` state rows, then the row."""
    def columns(a):
        return Coll(Array(S.D, Array(rows, T.F32)), concat([a[r][j].refs for j in range(S.D) for r in range(rows)]))
    c = B.batch(column, columns(kv), columns(score), axes=(0, 0))
    return B.call(bind(defs.nrq, RATIO=ratio, D=S.D, QB=S.QB, EPS=S.EPS), c, w, cs)


@composite("Dsv4MlaCompressC4", 1, ["ROWS", "D", "QB", "EPS"], compress_sig(WORDS, c4_rows), conformance=_CONF)
def Dsv4MlaCompressC4(B, S, kv, score, w, cs):
    """One ratio-4 compressed row (`SparseAttnCompressNormRopeStoreC4Kernel`, cutedsl:75-389) from its ROWS state rows at
    positions >= 0, row r's kv and score the D words it contributes (the module docstring's halves): column j
    `Dsv4MlaC4Column_v1{ROWS}`(kv[:, j], score[:, j]), then `Dsv4MlaNormRopeQuant_v1{4, D, QB, EPS}`."""
    return compress(B, S, kv, score, w, cs, bind(WORD_DEFS.c4, ROWS=S.ROWS), S.ROWS, 4, WORD_DEFS, WORDS)


@composite("Dsv4MlaCompressC128", 1, ["W", "R", "D", "QB", "EPS"], compress_sig(WORDS, c128_rows), conformance=_CONF)
def Dsv4MlaCompressC128(B, S, kv, score, w, cs):
    """One ratio-128 compressed row (`SparseAttnCompressC128Block8Kernel`, cutedsl:931-1184, then `SparseAttnNormRopeStoreKernel`,
    cutedsl:1245-1435) from its W*R state rows: column j `Dsv4MlaC128Column_v1{W, R}`, then
    `Dsv4MlaNormRopeQuant_v1{128, D, QB, EPS}` (the f32 `compressed_kv` between the kernels is the column's word)."""
    return compress(B, S, kv, score, w, cs, bind(WORD_DEFS.c128, W=S.W, R=S.R), S.W * S.R, 128, WORD_DEFS, WORDS)


WORD_DEFS = SimpleNamespace(c4=Dsv4MlaC4Column, c128=Dsv4MlaC128Column, norm=Dsv4MlaRmsNorm, quant=Dsv4MlaQuantGroup,
                            rope=Dsv4MlaRope, nrq=Dsv4MlaNormRopeQuant, byte=lambda B, v: B.call(const(8, v)))

DSV4_COMPRESSOR_COMPOSITES = [Dsv4MlaC4Column, Dsv4MlaC128Column, Dsv4MlaRmsNorm, Dsv4MlaQuantGroup, Dsv4MlaRope,
                              Dsv4MlaNormRopeQuant, Dsv4MlaCompressC4, Dsv4MlaCompressC128]

dsv4_compressor_twin.register_twin({d.id: d for d in DSV4_COMPRESSOR_COMPOSITES})
