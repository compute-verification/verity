"""[DSV4] The DeepSeek Sparse Attention indexer's own kernels as DeepSeek-V4-Pro runs them on an H200 (sm_90, TP8): vLLM
d9105ea80, Triton 3.7.1.  Line numbers are at that commit under `vllm/models/deepseek_v4/`; "the PTX" is Triton 3.7.1's sm_90
output for each kernel at the model's constexprs (index head 128, rotary 64, `compress_ratio` 4, `index_topk` 1024).

A ratio-4 layer's indexer (attention.py:977-1041) keeps its own compressor (compressor.py:309-370: `coff` 2, so a state of 8 rows
of 2 x 128 f32 words, compressor.py:232-233), DeepGEMM's logits over the compressed keys and a top-1024 selection.  The steps
other modules state: the logits are `dsa.DsaMqaLogit_v3{H=64, D=128}` (`fp8_fp4_paged_mqa_logits` with the FP8 cache, 128 e4m3
bytes and one f32 scale per entry, the same sm90 epilogue at 64 heads), a row of at most 1024 keys `dsa.DsaTopKShortRow_v1{1024}`.
The rest, each with its kernel of record:

1. `save_partial_states` (common/ops/save_partial_states.py:49-100): the state row's kv is a copy, its score
   `add.f32(score, ape[pos % 4])` (94-99; `fused_wkv_wgate`'s f32 outputs, `ape` the f32 parameter, compressor.py:255-261):
   `Dsv4CompressorScore_v1{W}`, W = 256.
2. The fused compress, norm, RoPE, quantisation and cache insert (`_fused_kv_compress_norm_rope_insert_indexer_attn`,
   common/ops/fused_compress_quant_cache.py:677-847; one warp, lane l column l + 32k): `Dsv4IndexerCompress_v1{D, ROPE, EPS}` =
   per column `Dsv4CompressColumn_v1{8}`, then `Dsv4IndexerNormRope_v1` and `Dsv4IndexerKQuant_v1{D}`.  Rows 0-3 read the
   state's kv[0:D], rows 4-7 kv[D:2D]; row r is position p - 7 + r, read as score -inf and kv 0 below 0 (only at p = 3): wiring.
3. `fused_inv_rope_fp8_quant` (common/ops/fused_inv_rope_fp8_quant.py:18-148, launcher 151-280, `deep_gemm_fp8_o_proj`'s input,
   nvidia/ops/o_proj.py:48): each head of 512 bf16 words, its last 64 inverse-rotated, quantised in 4 groups of 128:
   `Dsv4InvRopeQuantHead_v1{4, 128, 64}` = `Dsv4InvRope_v1{64}` and `Dsv4InvRopeQuant_v1{128}` per group.
4. The short-context fill (`_fill_short_context_topk_indices`, attention.py:71-88, launched at 1005-1012 when every request's
   compressed context fits the top-k): `Dsv4ShortContextTopK_v1{1024, 4}`.

The PTX of step 2, per column (fused_compress_quant_cache.py line numbers):

    m     = max.f32 left fold over rows 0..7                                        769
    e_r   = ex2.approx.f32(mul.f32(sub.f32(x_r, m), 0x3FB8AA3B))                   (ptxas: t < -126 ? MUFU.EX2(t/2)^2 : MUFU.EX2(t))
    den   = add.f32 left fold of e_r;  s_r = div.full.f32(e_r, den)
    c     = fma(s7, k7, ... fma(s2, k2, fma(s0, k0, mul(s1, k1))))                  777
    sq    = per lane: mul(c1, c1), then fma(c0, c0, .), fma(c2, c2, .), fma(c3, c3, .); the warp's xor butterfly of add.f32
    rrms  = rsqrt.approx.ftz(add.f32(eps, div.full.f32(sq, D)))                     781-782
    n     = mul(mul(c, rrms), cvt.f32.bf16(w))
    pair j (n[2j], n[2j+1]) = (fma(e, cos, -mul(o, sin)), fma(o, cos, mul(e, sin)))    820-822
          the first (D - ROPE)/2 pairs with cos = 1.0 and sin = 0.0, executed; the rest the f32 cache row at (p // 4) * 4
    v     = cvt.f32.bf16(cvt.rn.bf16.f32(pair))                                     831
    then `ue8m0_exponent` of the butterfly max of abs(v) with eps 1e-4 (832-835),
    inv   = ex2.approx.ftz(sub.f32(0, e)),  q = cvt.rn.satfinite.e4m3x2(min.xorsign.abs(mul(inv, v), 448)),
    scale = ex2.approx.ftz(e), stored f32                                           836-847

and of step 3: x = cvt.f32.bf16; offsets >= 448 rotated, pair j = (offset - 448) >> 1, even `fma(x, cos_j, mul(x', sin_j))`, odd
`fma(x, cos_j, -mul(x', sin_j))` (89-106, x' the partner, the f32 cache [cos 0..31 | sin 0..31]); per group of 128 the same exponent
with eps 1e-10 (110-112), scale = ex2.approx.ftz(e), q = e4m3(min.xorsign.abs(div.full.f32(x, scale), 448)) (121); scales stored f32
(sm_90 has no packed ue8m0 scales).

`ue8m0_exponent(amax, eps)` is `dsa`'s inlined `__nv_log2f` on s = mul.f32(max.f32(amax, eps), 0x3B124925 = f32(1/448)) with one
branch of the PTX kept: `setp.lt.u32 s, 0x7F800000` false gives y = fma.rn.ftz(s, inf, inf) = +inf (s = +inf; s is never NaN or
negative); the prescale (s < 2^-126) and s = 0 branches are unreachable.  e = cvt.rpi.ftz.f32.f32(y).  `min.xorsign.abs(x, 448)` is
min(|x|, 448) carrying x's sign: `F32Fminf` of `F32Fabs` plus the sign bit.

Every add, sub, mul and fma here is the GPU's (`F32Add_v2` .. `F32FmaFtz_v2`: each NaN result 0x7FFFFFFF), so a NaN operand gives
the hardware's word: a NaN output is 0x7FFFFFFF, a NaN quantises to +448 (0x7E), NormRope's NaN is F2FP's 0x7FFF widened.
`div.full.f32(a, b)` is `F32Mul_v2(DivFullScaleA_v1(a, b), DivFullRcp_v1(b))`.

Registered functions (each `_v1`; `boolean_dsv4_indexer` has their Boolean `_v2`):
  Dsv4CompressorScore{W}                (score, ape: W f32) -> W f32
  Dsv4CompressColumn{ROWS}              (kv, score: ROWS f32) -> f32
  Dsv4IndexerNormRope{D,ROPE,EPS}       (c: D f32, w: D bf16, cs: ROPE f32) -> D f32
  Dsv4IndexerKQuant{N}                  (x: N f32) -> (q: N e4m3, scale: f32)
  Dsv4IndexerCompress{D,ROPE,EPS}       (kv, score: 8 x D f32, w: D bf16, cs: ROPE f32) -> (q: D e4m3, scale: f32): one cache entry
  Dsv4InvRope{ROPE}                     (x: ROPE bf16, cs: ROPE f32) -> ROPE f32
  Dsv4InvRopeQuant{N}                   (x: N f32) -> (q: N e4m3, scale: f32)
  Dsv4InvRopeQuantHead{CHUNKS,QG,ROPE}  (x: CHUNKS*QG bf16, cs: ROPE f32) -> (q: CHUNKS*QG e4m3, scales: CHUNKS f32)
  Dsv4ShortContextTopK{TOPK,RATIO}      (pos: i64) -> TOPK i32
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.prims import F32ToE4m3Sat, const
from verity.ml.circuits.ops.scalar import (BitAnd, BitNot, BitOr, F32Fabs, F32Fmaxf, F32Fminf, F32Neg, I32Add, I32Le, SelectF32,
                                           SelectI32)
from verity.ml.definitions.registry import dsa as D
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.fp8 import E4M3
from verity.ml.definitions.registry.prims import BF16, F32, I32
from verity.ml.definitions.registry.ref_prims import F32GtStrict
from verity.ml.definitions.registry.sampling import bit_at
from verity.ml.definitions.registry.topp_word_gates import I32ToF32

_CONF = ("model-untested: read from the Triton kernels and Triton 3.7.1's PTX at the commits the module docstring pins; "
         "tests/properties/dsv4_indexer_capture_gpu.py is its hardware check")

LOG2E, HALF, MINUS_126 = 0x3FB8AA3B, 0x3F000000, 0xC2FC0000
ONE, INF, SIGN, INV_448 = 0x3F800000, 0x7F800000, 0x80000000, 0x3B124925
#: the kernels' amax floors: the compressor's 1e-4 (fused_compress_quant_cache.py:834), the inverse RoPE's 1e-10 (its `eps` constexpr)
K_EPS, INV_ROPE_EPS = D.F_1EM4, 0x2EDBE6FF
#: the compressor's state rows per column: `coff` (2) x the ratio (4)
ROWS = 8

#: the element types of a signature: these words, and their bits in `boolean_dsv4_indexer`
WORDS = SimpleNamespace(F32=F32, BF16=BF16, E4M3=E4M3, I32=I32, I64=Value(64))

WORD = dict(cvt=P.Bf16ToF32, bf16=P.F2fpBf16, add=P.F32AddV2, sub=P.F32SubV2, mul=P.F32MulV2, fma=P.F32FmaV2,
            fma_ftz=P.F32FmaFtzV2, rcp=P.DivFullRcp, scale_a=P.DivFullScaleA, rsqrt=P.RsqrtApprox, ex2=P.MufuEx2Ftz, fabs=F32Fabs,
            fmaxf=F32Fmaxf, fminf=F32Fminf, e4m3=F32ToE4m3Sat, iadd=I32Add, iand=D.U32And, isub=D.I32Sub, i2f=I32ToF32,
            ceil=D.F32CeilFtz, ile=I32Le, fsel=SelectF32, gt=F32GtStrict)


def word_ops(B) -> D.Ops:
    return D.Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x))


def on_words(B, S, signature, body, *colls) -> Coll:
    """``body(ops, S, *elements)`` over the word steps: each parameter as its list of scalar elements, the returned elements (in
    the return type's leaf order) packed as one value of ``signature(S)``'s return type."""
    params, ret = signature(S)
    parts = [[c[i] for i in range(t.n)] if isinstance(t, Array) else [c] for c, (_, t) in zip(colls, params)]
    return Coll(ret, concat([e.refs for e in body(word_ops(B), S, *parts)]))


# --- the shared steps ---------------------------------------------------------------------------------------------------------

def ex2_approx(o, t):
    """`ex2.approx.f32` (not .ftz) as ptxas lowers it: t < -126 (ordered) halves t before MUFU.EX2 and squares the result."""
    small = o.gt(o.word(MINUS_126), t)
    y = o.ex2(o.fsel(small, o.mul(t, o.word(HALF)), t))
    return o.fsel(small, o.mul(y, y), y)


def clamp_448(o, x):
    """`min.xorsign.abs.f32(x, 448)`: min(|x|, 448) with x's sign (a NaN x gives 448 with its sign)."""
    return o.iadd(o.fminf(o.fabs(x), o.word(D.F_448)), o.iand(x, o.word(SIGN)))


def ue8m0_exponent(o, amax, eps: int):
    """ceil(log2(max(amax, eps) / 448)) as the PTX computes it (the module docstring): `dsa.ue8m0_scale`'s `__nv_log2f` on
    s = mul(max(amax, eps), f32(1/448)), y = +inf when s >= 0x7F800000, then `cvt.rpi.ftz`."""
    s = o.mul(o.fmaxf(amax, o.word(eps)), o.word(INV_448))
    e = o.iand(o.iadd(s, o.word(D.LOG2F_OFFSET)), o.word(D.LOG2F_MASK))
    t = o.add(o.isub(s, e), o.word(D.F_NEG1))
    ef = o.fma_ftz(o.i2f(e), o.word(D.F_2M23), o.word(0))
    p = o.fma_ftz(o.word(D.LOG2F_POLY[0]), t, o.word(D.LOG2F_POLY[1]))
    for c in D.LOG2F_POLY[2:]:
        p = o.fma_ftz(p, t, o.word(c))
    y = o.add(ef, o.fma_ftz(t, o.word(D.LOG2F_LOG2E), o.mul(t, o.mul(t, p))))
    return o.ceil(o.fsel(o.ile(o.word(INF), s), o.word(INF), y))


def check_pow2(name: str, N: int) -> None:
    if not (N > 0 and N & (N - 1) == 0):
        raise ValueError(f"{name}: the amax butterfly takes a power-of-two row (N={N})")


# --- step 1: the state's score ------------------------------------------------------------------------------------------------

def score_sig(T):
    return lambda S: ((("score", Array(S.W, T.F32)), ("ape", Array(S.W, T.F32))), Array(S.W, T.F32))


def compressor_score(o, S, score, ape) -> list:
    return [o.add(s, a) for s, a in zip(score, ape)]


@composite("Dsv4CompressorScore", 1, ["W"], score_sig(WORDS), conformance=_CONF)
def Dsv4CompressorScore(B, S, score, ape):
    """`save_partial_states`' score row (save_partial_states.py:94-99): `add.f32(score[i], ape[i])`, score the `fused_wkv_wgate`
    output's f32 word and ape the row `ape[pos % ratio]` of the f32 parameter."""
    return on_words(B, S, score_sig(WORDS), compressor_score, score, ape)


# --- step 2: the compressor -------------------------------------------------------------------------------------------------

def column_sig(T):
    return lambda S: ((("kv", Array(S.ROWS, T.F32)), ("score", Array(S.ROWS, T.F32))), T.F32)


def compress_column(o, S, kv, score) -> list:
    """`Dsv4CompressColumn_v1`'s body: softmax of the scores, then the weighted sum of kv (the module docstring's PTX)."""
    m = score[0]
    for x in score[1:]:
        m = o.fmaxf(m, x)
    e = [ex2_approx(o, o.mul(o.sub(x, m), o.word(LOG2E))) for x in score]
    den = e[0]
    for x in e[1:]:
        den = o.add(den, x)
    rcp = o.rcp(den)
    s = [o.mul(o.scale_a(x, den), rcp) for x in e]
    if len(kv) == 1:
        return [o.mul(s[0], kv[0])]
    c = o.fma(s[0], kv[0], o.mul(s[1], kv[1]))
    for r in range(2, len(kv)):
        c = o.fma(s[r], kv[r], c)
    return [c]


@composite("Dsv4CompressColumn", 1, ["ROWS"], column_sig(WORDS), conformance=_CONF)
def Dsv4CompressColumn(B, S, kv, score):
    """One column of the compressed key (fused_compress_quant_cache.py:769-777): the max m of the ROWS scores (`max.f32`, left
    fold), e_r = `ex2.approx.f32((score_r - m) * log2 e)` (`ex2_approx`), their sum (`add.f32`, left fold), s_r = `div.full.f32(e_r,
    sum)`, and the kv rows' weighted sum from `mul(s1, kv1)`, then `fma(s0, kv0, .)` and `fma(s_r, kv_r, .)` for r >= 2."""
    return on_words(B, S, column_sig(WORDS), compress_column, kv, score)


def check_norm_statics(D_: int, ROPE: int, EPS: float) -> None:
    if not (D_ in (32, 128) and 0 < ROPE <= D_ and ROPE % 2 == 0):
        raise ValueError(f"the compressor's row is one warp of 1 or 4 columns per lane with an even rotary suffix (D={D_}, "
                         f"ROPE={ROPE})")
    D.check_eps(EPS)


def norm_rope_sig(T):
    return lambda S: ((("c", Array(S.D, T.F32)), ("w", Array(S.D, T.BF16)), ("cs", Array(S.ROPE, T.F32))), Array(S.D, T.F32))


def norm_rope(o, S, c, w, cs) -> list:
    """`Dsv4IndexerNormRope_v1`'s body: RMSNorm, the GPT-J rotation of every pair, the bf16 rounding."""
    check_norm_statics(S.D, S.ROPE, S.EPS)
    order = (1, 0, 2, 3)[:S.D // 32] if S.D > 32 else (0,)
    lanes = []
    for lane in range(32):
        cols = [c[lane + 32 * k] for k in order]
        acc = o.mul(cols[0], cols[0])
        for x in cols[1:]:
            acc = o.fma(x, x, acc)
        lanes.append(acc)
    n = o.word(D.f32_bits(S.D))
    var = o.mul(o.scale_a(D.halving_tree(lanes, o.add), n), o.rcp(n))
    rrms = o.rsqrt(o.add(o.word(D.f32_bits(S.EPS)), var))
    normed = [o.mul(o.mul(c[i], rrms), o.cvt(w[i])) for i in range(S.D)]
    nope, half = (S.D - S.ROPE) // 2, S.ROPE // 2
    out = []
    for j in range(S.D // 2):
        cos, sin = (o.word(ONE), o.word(0)) if j < nope else (cs[j - nope], cs[half + j - nope])
        ev, od = normed[2 * j], normed[2 * j + 1]
        out += [o.fma(ev, cos, o.neg(o.mul(od, sin))), o.fma(od, cos, o.mul(ev, sin))]
    return [o.cvt(o.bf16(x)) for x in out]


@composite("Dsv4IndexerNormRope", 1, ["D", "ROPE", "EPS"], norm_rope_sig(WORDS), conformance=_CONF)
def Dsv4IndexerNormRope(B, S, c, w, cs):
    """The compressed key's RMSNorm, RoPE and bf16 rounding (fused_compress_quant_cache.py:781-831): the squares' per-lane sum
    (lane l holds columns l + 32k; `mul(c1, c1)` then `fma` of c0, c2, c3), the warp butterfly of `add.f32` (`dsa.halving_tree`),
    rrms = `rsqrt.approx.ftz(eps + div.full.f32(sum, D))` (`RsqrtApprox_v1`: the operand is at least eps, so never subnormal),
    n = `mul(mul(c, rrms), f32(w))`; each pair (n[2j], n[2j+1]) rotated by `(fma(e, cos, -(o * sin)), fma(o, cos, e * sin))`, the
    first (D - ROPE)/2 pairs by cos 1.0 and sin 0.0 (executed: an inf partner gives NaN), the others by cs[j'] and cs[ROPE/2 + j'];
    each word rounded to bf16 by F2FP (NaN 0x7FFF) and widened.  D = 128 is the kernel's; 32 is the same body at one column per
    lane, for the checks."""
    return on_words(B, S, norm_rope_sig(WORDS), norm_rope, c, w, cs)


def quant_sig(T):
    return lambda S: ((("x", Array(S.N, T.F32)),), Tuple(Array(S.N, T.E4M3), T.F32))


def k_quant(o, S, x) -> list:
    """`Dsv4IndexerKQuant_v1`'s body: the q bytes, then the scale."""
    check_pow2("Dsv4IndexerKQuant", S.N)
    e = ue8m0_exponent(o, D.halving_tree([o.fabs(v) for v in x], o.fmaxf), K_EPS)
    inv = o.ex2(o.sub(o.word(0), e))
    return [o.e4m3(clamp_448(o, o.mul(inv, v))) for v in x] + [o.ex2(e)]


@composite("Dsv4IndexerKQuant", 1, ["N"], quant_sig(WORDS), conformance=_CONF)
def Dsv4IndexerKQuant(B, S, x):
    """The compressed key's ue8m0 quantisation (fused_compress_quant_cache.py:832-847): e = `ue8m0_exponent`(max |x|, 1e-4) (the
    max a butterfly over words with the sign cleared: order-free), inv = `ex2.approx.ftz(0 - e)`, q[i] =
    `cvt.rn.satfinite.e4m3x2(min.xorsign.abs(inv * x[i], 448))`, and the f32 scale `ex2.approx.ftz(e)`."""
    return on_words(B, S, quant_sig(WORDS), k_quant, x)


def compress_sig(T):
    return lambda S: ((("kv", Array(ROWS, Array(S.D, T.F32))), ("score", Array(ROWS, Array(S.D, T.F32))),
                       ("w", Array(S.D, T.BF16)), ("cs", Array(S.ROPE, T.F32))), Tuple(Array(S.D, T.E4M3), T.F32))


def compress(B, S, kv, score, w, cs, defs, T) -> Coll:
    """`Dsv4IndexerCompress`'s body over the Definitions ``defs`` (column, norm_rope, k_quant) and element types ``T``."""
    def columns(a):
        return Coll(Array(S.D, Array(ROWS, T.F32)), concat([a[r][j].refs for j in range(S.D) for r in range(ROWS)]))
    c = B.batch(bind(defs.column, ROWS=ROWS), columns(kv), columns(score), axes=(0, 0))
    v = B.call(bind(defs.norm_rope, D=S.D, ROPE=S.ROPE, EPS=S.EPS), c, w, cs)
    return B.call(bind(defs.k_quant, N=S.D), v)


@composite("Dsv4IndexerCompress", 1, ["D", "ROPE", "EPS"], compress_sig(WORDS), conformance=_CONF)
def Dsv4IndexerCompress(B, S, kv, score, w, cs):
    """One compressed index-key cache entry (`_fused_kv_compress_norm_rope_insert_indexer_attn`, fused_compress_quant_cache.py:
    677-847) from its 8 state rows as the kernel reads them: column j is `Dsv4CompressColumn_v1{8}`(kv[:, j], score[:, j]), then
    `Dsv4IndexerNormRope_v1{D, ROPE, EPS}` and `Dsv4IndexerKQuant_v1{D}`: the entry's D e4m3 bytes and f32 scale (132 bytes at
    D = 128).  ``cs`` is the f32 rotary cache row at the compressed position."""
    return compress(B, S, kv, score, w, cs, WORD_DEFS, WORDS)


# --- step 3: the inverse RoPE and its quantisation ----------------------------------------------------------------------------

def inv_rope_sig(T):
    return lambda S: ((("x", Array(S.ROPE, T.BF16)), ("cs", Array(S.ROPE, T.F32))), Array(S.ROPE, T.F32))


def inv_rope(o, S, x, cs) -> list:
    """`Dsv4InvRope_v1`'s body."""
    if not (S.ROPE > 0 and S.ROPE % 2 == 0):
        raise ValueError(f"Dsv4InvRope: an even rotary width (ROPE={S.ROPE})")
    xf = [o.cvt(v) for v in x]
    out = []
    for i in range(S.ROPE):
        t = o.mul(xf[i ^ 1], cs[S.ROPE // 2 + i // 2])
        out.append(o.fma(xf[i], cs[i // 2], t if i % 2 == 0 else o.neg(t)))
    return out


@composite("Dsv4InvRope", 1, ["ROPE"], inv_rope_sig(WORDS), conformance=_CONF)
def Dsv4InvRope(B, S, x, cs):
    """The inverse GPT-J rotation of a head's rotary suffix (fused_inv_rope_fp8_quant.py:89-106): x = f32(bf16), pair j =
    (x[2j], x[2j+1]) becomes (`fma(x[2j], cos_j, x[2j+1] * sin_j)`, `fma(x[2j+1], cos_j, -(x[2j] * sin_j))`), ``cs`` the position's
    f32 cache row [cos 0..ROPE/2-1 | sin 0..ROPE/2-1]."""
    return on_words(B, S, inv_rope_sig(WORDS), inv_rope, x, cs)


def inv_rope_quant(o, S, x) -> list:
    """`Dsv4InvRopeQuant_v1`'s body: the q bytes, then the scale."""
    check_pow2("Dsv4InvRopeQuant", S.N)
    scale = o.ex2(ue8m0_exponent(o, D.halving_tree([o.fabs(v) for v in x], o.fmaxf), INV_ROPE_EPS))
    rcp = o.rcp(scale)
    return [o.e4m3(clamp_448(o, o.mul(o.scale_a(v, scale), rcp))) for v in x] + [scale]


@composite("Dsv4InvRopeQuant", 1, ["N"], quant_sig(WORDS), conformance=_CONF)
def Dsv4InvRopeQuant(B, S, x):
    """One quantisation group of `fused_inv_rope_fp8_quant` (fused_inv_rope_fp8_quant.py:110-121): scale = `ex2.approx.ftz` of
    `ue8m0_exponent`(max |x|, 1e-10), q[i] = `cvt.rn.satfinite.e4m3x2(min.xorsign.abs(div.full.f32(x[i], scale), 448))`."""
    return on_words(B, S, quant_sig(WORDS), inv_rope_quant, x)


def head_sig(T):
    return lambda S: ((("x", Array(S.CHUNKS * S.QG, T.BF16)), ("cs", Array(S.ROPE, T.F32))),
                      Tuple(Array(S.CHUNKS * S.QG, T.E4M3), Array(S.CHUNKS, T.F32)))


def check_head_statics(QG: int, ROPE: int) -> None:
    check_pow2("Dsv4InvRopeQuant", QG)
    if not (0 < ROPE <= QG and ROPE % 2 == 0):
        raise ValueError(f"Dsv4InvRopeQuantHead: an even rotary suffix within the last group (ROPE={ROPE}, QG={QG})")


def inv_rope_head(B, S, x, cs, defs, T) -> Coll:
    """`Dsv4InvRopeQuantHead`'s body over the Definitions ``defs`` (cvt, inv_rope, quant) and element types ``T``."""
    hd = S.CHUNKS * S.QG
    check_head_statics(S.QG, S.ROPE)
    plain = B.batch(defs.cvt, x[0:hd - S.ROPE], axes=(0,))
    rot = B.call(bind(defs.inv_rope, ROPE=S.ROPE), x[hd - S.ROPE:hd], cs)
    words = [plain[i] for i in range(hd - S.ROPE)] + [rot[i] for i in range(S.ROPE)]
    quant = bind(defs.quant, N=S.QG)
    parts = [B.call(quant, Coll(Array(S.QG, T.F32), concat([v.refs for v in words[g * S.QG:(g + 1) * S.QG]])))
             for g in range(S.CHUNKS)]
    return tuple_of(Coll(Array(hd, T.E4M3), concat([p[0].refs for p in parts])),
                    Coll(Array(S.CHUNKS, T.F32), concat([p[1].refs for p in parts])))


@composite("Dsv4InvRopeQuantHead", 1, ["CHUNKS", "QG", "ROPE"], head_sig(WORDS), conformance=_CONF)
def Dsv4InvRopeQuantHead(B, S, x, cs):
    """One (token, head) of `fused_inv_rope_fp8_quant` (one warp per head, fused_inv_rope_fp8_quant.py:18-148): the first
    CHUNKS*QG - ROPE words widened (`Bf16ToF32_v1`), the last ROPE `Dsv4InvRope_v1{ROPE}`, and each group of QG
    `Dsv4InvRopeQuant_v1{QG}`: the head's e4m3 bytes and its CHUNKS f32 scales (4 x 128 and 64 rotary on the model)."""
    return inv_rope_head(B, S, x, cs, WORD_DEFS, WORDS)


# --- step 4: the short-context fill ---------------------------------------------------------------------------------------

def check_topk_statics(TOPK: int, RATIO: int) -> int:
    """The bit length of the largest threshold R(i+1) - 1, which must fit 31 bits."""
    if not (TOPK > 0 and RATIO > 0 and RATIO * TOPK <= 1 << 31):
        raise ValueError(f"Dsv4ShortContextTopK: TOPK, RATIO > 0 with RATIO * TOPK <= 2^31 (TOPK={TOPK}, RATIO={RATIO})")
    return (RATIO * TOPK - 1).bit_length()


def topk_sig(T):
    return lambda S: ((("pos", T.I64),), Array(S.TOPK, T.I32))


@composite("Dsv4ShortContextTopK", 1, ["TOPK", "RATIO"], topk_sig(WORDS), conformance=_CONF)
def Dsv4ShortContextTopK(B, S, pos):
    """`_fill_short_context_topk_indices` (attention.py:71-88) for one token: slot i is i when i < (pos + 1) // RATIO (int64,
    truncating: the PTX's `(q + ((q >> 63) >>u 62)) >> 2`), else -1.  That is slot i valid iff pos >= RATIO(i+1) - 1 and pos is not
    INT64_MAX (whose pos + 1 wraps negative): pos's bits b, its sign b63 clear and b0..b62 not all set, and either a bit from L
    up set or its low L bits (an i32) at least the threshold, L the threshold's bit length."""
    L = check_topk_statics(S.TOPK, S.RATIO)
    o = word_ops(B)
    at = bit_at(64)
    b = [B.call(at, pos, o.word(j)) for j in range(64)]

    def fold(op, bits):
        acc = bits[0]
        for x in bits[1:]:
            acc = B.call(op, acc, x)
        return acc

    ok = B.call(BitAnd, B.call(BitNot, b[63]), B.call(BitNot, fold(BitAnd, b[:63])))
    big = fold(BitOr, b[L:63])
    low = [B.call(SelectI32, b[j], o.word(1 << j), o.word(0)) for j in range(L)]
    lo = fold(I32Add, low) if low else o.word(0)
    out = []
    for i in range(S.TOPK):
        v = B.call(BitAnd, ok, B.call(BitOr, big, B.call(I32Le, o.word(S.RATIO * (i + 1) - 1), lo)))
        out.append(B.call(SelectI32, v, o.word(i), o.word(D.M32)))
    return Coll(Array(S.TOPK, I32), concat([w.refs for w in out]))


WORD_DEFS = SimpleNamespace(column=Dsv4CompressColumn, norm_rope=Dsv4IndexerNormRope, k_quant=Dsv4IndexerKQuant,
                            cvt=P.Bf16ToF32, inv_rope=Dsv4InvRope, quant=Dsv4InvRopeQuant)

DSV4_INDEXER = [Dsv4CompressorScore, Dsv4CompressColumn, Dsv4IndexerNormRope, Dsv4IndexerKQuant, Dsv4IndexerCompress, Dsv4InvRope,
                Dsv4InvRopeQuant, Dsv4InvRopeQuantHead, Dsv4ShortContextTopK]
