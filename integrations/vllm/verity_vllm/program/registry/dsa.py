"""[GLM-5.3] The DeepSeek Sparse Attention indexer as the H200 runs it: GLM-5.3-FP8 at TP8 on sm_90, eager, vLLM d9105ea80, Triton
3.7.1, DeepGEMM 8b1392b9.  Line numbers are at those commits; "the PTX" is Triton 3.7.1's sm_90 output for the two Triton kernels at
GLM-5.3's constexprs (`INDEX_K_DIM` 128, `INDEX_K_HALF_ROT_DIM` 32, `INDEX_ROPE_INTERLEAVE` True; `_fused_q_kernel` at `num_warps=1`).

The indexer's per-token work (`deepseek_v32/attention.py`, `model_executor/layers/sparse_attn_indexer.py`), each step's kernel of record:

1. `wq_b` (block-FP8 linear) and `wk_weights_proj` (bf16, unquantised, replicated: `disable_tp=True`) give index_q [32, 128] bf16,
   index_k [128] bf16 and the head weights [32] bf16 (deepseek_v2.py:700-742).
2. `_fused_norm_rope_kernel`, program 0 (common/kernels.py:315-415): index_k's LayerNorm (`k_norm`, eps 1e-6, f32 weight and bias:
   layernorm.py:342-347), the interleaved RoPE of its first 64 elements, the ue8m0 quantisation and the cache write
   (`_fp8_quant_and_cache_write`, kernels.py:63-87: 128 e4m3 bytes and one f32 scale per slot): `DsaIndexerKRow_v1`.
3. `_fused_q_kernel`, program 1 (kernels.py:761-826, launched by `fused_q`, kernels.py:935-982; the CuTe DSL branch at
   kernels.py:876-892 needs capability >= 100, `nvidia/ops/fused_q_cutedsl.py:28-52`, so not on sm_90): each head's interleaved RoPE
   and ue8m0 quantisation, and its weight times the q scale, `softmax_scale` = 128^-0.5 and `n_head^-0.5` = 32^-0.5
   (deepseek_v2.py:709, 742; f32 kernel arguments): `DsaIndexerQHead_v1`, a token's heads `DsaIndexerQ_v1`.
4. DeepGEMM `fp8_mqa_logits` (prefill, `sm90_fp8_mqa_logits.cuh:225-317`) or `fp8_paged_mqa_logits` (decode,
   `sm90_fp8_paged_mqa_logits.cuh:270-355`), `clean_logits=False` (sparse_attn_indexer.py:496-503, 600-611): one epilogue for both,
   per (query token, key): `DsaMqaLogit_v3`, a token's keys `DsaMqaLogits_v3`.  `DsaMqaLogit_v1` reads the source's order, which the
   H200 refutes: nvcc contracts each sum's first product into its second add.
5. The top-2048 selection: prefill `ops.top_k_per_row_prefill` (`topKPerRowPrefill<512, false>` on the first 12288 rows, `<512,
   true>` past them, sampler.cu:846-878), decode `cooperative_topk` on H200 (num_rows <= 64, sparse_attn_indexer.py:615-642; else
   `persistent_topk`, its FilteredTopK past 32 rows, topk.cu:39).  A row of at most 2048 keys takes each kernel's shortcut
   (sampler.cu:388-412, cooperative_topk.cuh:494-501): `DsaTopKShortRow_v1`.  A longer row is a radix selection whose output order,
   and choice among the words tied at its threshold, the kernels leave to their atomics (sampler.cu:269, 297;
   topk_histogram_4096.cuh:470, persistent_topk.cuh:291), so it is no function of the row: `DsaTopKSelection_v1` takes the indices
   the kernel wrote and states that they are a top-2048 set of the row under the kernels' order (below).

The top-k order.  Every kernel ranks a word by `convert_to_uint32_v2` (topk_histogram_4096.cuh:71-74, persistent_topk.cuh:50-53;
sampler.cu:48-51's `convert_to_uint32` is its complement): x | 2^31 for a clear sign, ~x for a set one, so -NaN < -inf < ... < -0 <
+0 < ... < +inf < +NaN; as a signed word that is `order_key`, x or 0x7FFFFFFF - x.  The coarse passes bin `__float2half_rn(x)` or the
key's top bits, monotone in the order for every word but a NaN.  So on a row without a NaN every kernel writes a top-k set, except
where a buffer overflows (then an index may be missing, stale or past the row, and `DsaTopKSelection_v1` is 0):

    prefill        none: a threshold bin of at most 2048 items is sorted (sampler.cu:341), the last radix step's ties are equal words
    cooperative    up to 16384 keys (`histogram_4096_topk<TOPK,12>`, cooperative_topk.cuh:504-511): more than TOPK words tied in the
                   threshold bin when it needs tie handling (topk_histogram_4096.cuh:447, 470, 485); longer (`large_topk`): more
                   than 1024 ties in one block's share (cooperative_topk.cuh:372, 394-396), or above + tied > TOPK with more than
                   max(TOPK, 1024) tied (445, 455, 462, 472)
    persistent     up to 8192 keys (`histogram_2048_topk`): more than 3708 words in the threshold bin (persistent_topk.cuh:171, 294),
                   or a threshold at half's -inf bin, where the 4-word load's padding counts as -inf (97-104, 266-268); up to
                   32768 (`histogram_256_topk`): more than 4096 (512, 582); longer: the exact radix (849-854)
    FilteredTopK   up to 32768 keys: `histogram_4096_topk<TOPK,12,8>` as above; longer: more than 16384 candidates (1190, 1267)

The ue8m0 quantisation (`_fp8_ue8m0_quantize`, kernels.py:50-60), in the PTX of both Triton kernels:

    amax  = max.f32 over abs.f32(x)                        a butterfly tree; max over words with the sign cleared is order-free
    s     = div.rn.f32(max.f32(amax, 0f38D1B717), 448.0)   1e-4 as f32
    y     = __nv_log2f(s)                                  inlined (below)
    scale = ex2.approx.ftz.f32(cvt.rpi.ftz.f32.f32(y))
    q[i]  = cvt.rn.satfinite.e4m3x2.f32(div.rn.f32(x[i], scale))

`__nv_log2f` as inlined (both PTX files, up to `__nv_log2f.exit`):

    e   = (bits(s) + 0xC0CAFB0D) & 0xFF800000              add.s32 -0x3F3504F3 (bits of sqrt(1/2)), and.b32
    m   = bits(s) - e                                      sub.s32: m in [sqrt(1/2), sqrt(2))
    ef  = fma.rn.ftz(cvt.rn.f32.s32(e), 2^-23, +0)
    t   = add.f32(m, -1.0)
    p   = fma.rn.ftz(0x3DC6B27F, t, 0xBE2C7F30); p = fma.rn.ftz(p, t, c) for c in LOG2F_POLY[2:]
    y   = add.f32(ef, fma.rn.ftz(t, 0x3FB8AA3B, mul.f32(t, mul.f32(t, p))))

and two branches that change no scale: the prescale (`setp.lt.f32 s, 2^-126`: s times 2^23 and ef's addend -23), which no operand
reaches (s >= 1e-4 / 448), and `bits(s) >= 0x7F800000` unsigned (`fma.rn.ftz(s, inf, inf)`), reached only by s = +inf, for which
the polynomial gives y = 128 (e = 128, m = 1) and so the same scale, ex2(128) = +inf.  So the scale is a power of two in
[2^-22, 2^127], or +inf when amax is.

Registered functions (`Name_v1` but where a version is written):
  U32And, I32Sub, F32CeilFtz                 primitives: and.b32, sub.s32, cvt.rpi.ftz.f32.f32
  DsaUe8m0Scale                             (amax: f32) -> f32
  DsaUe8m0Quant{N}                          (x: N f32) -> (q: N e4m3, scale: f32)
  DsaIndexerKRow{D,ROPE,EPS}                (k: D bf16, w, b: D f32, cs: ROPE bf16) -> (q: D e4m3, scale: f32): one cache slot
  DsaIndexerQHead{D,ROPE,NH}                (q: D bf16, cs: ROPE bf16, w: bf16) -> (q: D e4m3, w: f32): one (token, head)
  DsaIndexerQ{H,D,ROPE}                     (q: H x D bf16, cs: ROPE bf16, w: H bf16) -> (q: H x D e4m3, w: H f32): one token
  DsaMqaLogit_v3{H,D}                       (q: H x D e4m3, w: H f32, k: D e4m3, ks: f32) -> f32: one logit (v1: refuted)
  DsaMqaLogits_v3{N,H,D}                    (q, w, k: N x D e4m3, ks: N f32) -> N f32: one query token over N keys (v1: refuted)
  DsaTopKShortRow{TOPK}                     (len: i32) -> TOPK i32
  DsaTopKSelection{TOPK,N}                  (logits: N f32, len: i32, idx: TOPK i32) -> bit: idx a selection of the row
  DsaTopKSelection_v3{TOPK,N}               (logits, len, idx, thr: f32) -> bit: the same against a threshold witness

`cs` is the rotary cache row at the token's position, [cos 0..ROPE/2-1 | sin 0..ROPE/2-1] bf16 (`get_rope` builds it in the default
dtype, rotary_embedding/base.py:59-63).  `ks` is the key's f32 scale from its cache slot.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import F32ToE4m3Sat, HopperE4m3QgmmaDot32, const
from verity.ml.circuits.ops.scalar import BitAnd, BitNot, BitOr, F32Fabs, F32Fmaxf, F32Neg, I32Add, I32Eq, I32Le, SelectI32
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import E4M3
from verity_vllm.program.registry.prims import BF16, BIT, F32, I32, bits_of
from verity_vllm.program.registry.topp_word_gates import I32ToF32

M32 = 0xFFFFFFFF

_CONF = ("model-untested: read from the Triton kernels and their PTX at the commits the module docstring pins; "
         "tests/properties/glm53_dsa_capture_gpu.py is its hardware check")
_LOGIT_V1_CONF = ("refuted on an H200 (sm_90): property glm53_dsa_capture, run r20261012-222325-ce92 (record 626052b647671ba1), "
                  "828 of 19,513 prefill and 354 of 7,292 paged logits differ by 1 to 4 ulps; nvcc contracts each sum's first "
                  "product, DsaMqaLogit_v3")
_LOGIT_CONF = ("exact-model-tested; H200 (sm_90, 132 SMs; vLLM d9105ea80, DeepGEMM 8b1392b9): the operands and logits of property "
               "glm53_dsa_capture's run r20261012-222325-ce92, every one of the 19,513 prefill and 7,292 paged logits equal "
               "(tests/program/data/dsa_h200_logits.npz keeps eight that refute v1)")
_TOPK_CONF = ("model-untested: read from vLLM d9105ea80 sampler.cu:388-412 and cooperative_topk.cuh:494-501; "
              "tests/properties/glm53_dsa_capture_gpu.py is its hardware check")
_SELECTION_CONF = ("model-untested: read from vLLM d9105ea80's top-k kernels (sampler.cu, cooperative_topk.cuh, "
                   "topk_histogram_4096.cuh, persistent_topk.cuh; the module docstring, step 5); "
                   "tests/properties/glm53_dsa_capture_gpu.py is its hardware check")

#: `__nv_log2f`'s constants as the PTX has them: the exponent offset (-bits(sqrt(1/2)) mod 2^32), the exponent mask, 2^-23, the
#: polynomial's coefficients in Horner order (the first two are the first FMA's), and log2(e)
LOG2F_OFFSET = (-0x3F3504F3) & M32
LOG2F_MASK = 0xFF800000
LOG2F_POLY = (0x3DC6B27F, 0xBE2C7F30, 0x3E2FCF2A, 0xBE374E43, 0x3E520BF4, 0xBE763C8B, 0x3E93BF99, 0xBEB8AA49, 0x3EF6384A,
              0xBF38AA3B)
LOG2F_LOG2E = 0x3FB8AA3B
F_2M23, F_NEG1, F_1EM4, F_448 = 0x34000000, 0xBF800000, 0x38D1B717, 0x43E00000


@primitive("U32And", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def U32And(a: int, b: int) -> int:
    """`and.b32`: the bitwise AND of two 32-bit words."""
    return a & b & M32


@primitive("I32Sub", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Sub(a: int, b: int) -> int:
    """`sub.s32`: a - b modulo 2^32."""
    return (a - b) & M32


def f32_ceil_ftz(x: int) -> int:
    """`cvt.rpi.ftz.f32.f32` on a word: a subnormal is first flushed to the zero of its sign, then the value is rounded toward +inf to
    an integer (-1 < x < 0 gives -0); zeros and infinities pass, a NaN gives the GPU's 0x7FFFFFFF."""
    x &= M32
    e = (x >> 23) & 0xFF
    if e == 0xFF:
        return 0x7FFFFFFF if x & 0x7FFFFF else x
    if e == 0:
        return x & 0x80000000
    if e >= 150:
        return x
    neg = x >> 31
    if e < 127:
        return 0x80000000 if neg else 0x3F800000
    mask = (1 << (150 - e)) - 1
    if not x & mask:
        return x
    return x & ~mask & M32 if neg else ((x | mask) + 1) & M32


@primitive("F32CeilFtz", 1, [("x", F32)], F32, conformance="exact-model-tested")
def F32CeilFtz(x: int) -> int:
    """`cvt.rpi.ftz.f32.f32` (`tl.math.ceil` on sm_90): `f32_ceil_ftz`."""
    return f32_ceil_ftz(x)


DSA_PRIMITIVES = [U32And, I32Sub, F32CeilFtz]


class Ops:
    """The steps the indexer's bodies take, each ``ops.<name>(*colls)`` a Call of ``defs[name]``; ``word(bits)`` a constant 32-bit
    word and ``neg(x)`` the f32 negation (`F32Neg_v1`; on bits one NOT gate on the sign).  `word_ops` binds the word primitives,
    `boolean_dsa` their Boolean versions, so each body is written once."""

    def __init__(self, B, defs: dict, word, neg):
        self.B, self._defs, self._word, self.neg, self._words = B, defs, word, neg, {}

    def word(self, bits: int):
        if bits not in self._words:
            self._words[bits] = self._word(bits)
        return self._words[bits]

    def __getattr__(self, name: str):
        d = self._defs[name]
        return lambda *a: self.B.call(d, *a)


WORD = dict(cvt=P.Bf16ToF32, add=P.F32Add, mul=P.F32Mul, fma=P.F32Fma, fma_ftz=P.F32FmaFtz, div=P.F32Div, rcp=P.DivFullRcp,
            scale_a=P.DivFullScaleA, rsqrt=P.RsqrtApprox, ex2=P.MufuEx2Ftz, fabs=F32Fabs, fmaxf=F32Fmaxf, e4m3=F32ToE4m3Sat,
            iadd=I32Add, iand=U32And, isub=I32Sub, i2f=I32ToF32, ceil=F32CeilFtz, ile=I32Le, isel=SelectI32,
            dot=HopperE4m3QgmmaDot32, ieq=I32Eq, band=BitAnd, bnot=BitNot, bor=BitOr)


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x))


def f32_bits(v: float) -> int:
    return bits_of(np.float32(v))


def halving_tree(vals: list, op) -> object:
    """``op`` over ``vals`` (a power-of-two count) as a lane-0 xor-butterfly computes it: slot t < h becomes `op(v[t], v[t + h])`
    for h = n/2, n/4, ..., 1."""
    while len(vals) > 1:
        h = len(vals) // 2
        vals = [op(vals[t], vals[t + h]) for t in range(h)]
    return vals[0]


def ue8m0_scale(o: Ops, amax):
    """`DsaUe8m0Scale_v1`'s body (the module docstring's sequence)."""
    s = o.div(o.fmaxf(amax, o.word(F_1EM4)), o.word(F_448))
    e = o.iand(o.iadd(s, o.word(LOG2F_OFFSET)), o.word(LOG2F_MASK))
    t = o.add(o.isub(s, e), o.word(F_NEG1))
    ef = o.fma_ftz(o.i2f(e), o.word(F_2M23), o.word(0))
    p = o.fma_ftz(o.word(LOG2F_POLY[0]), t, o.word(LOG2F_POLY[1]))
    for c in LOG2F_POLY[2:]:
        p = o.fma_ftz(p, t, o.word(c))
    y = o.add(ef, o.fma_ftz(t, o.word(LOG2F_LOG2E), o.mul(t, o.mul(t, p))))
    return o.ex2(o.ceil(y))


@composite("DsaUe8m0Scale", 1, [], lambda S: ((("amax", F32),), F32), conformance=_CONF)
def DsaUe8m0Scale(B, S, amax):
    """`_fp8_ue8m0_quantize`'s scale from a row's amax (kernels.py:57-58; the module docstring has the PTX):
    `ex2.approx.ftz(cvt.rpi.ftz(__nv_log2f(div.rn(max.f32(amax, 1e-4), 448))))`, `__nv_log2f` its inlined polynomial without the
    branches no operand reaches (s >= 1e-4 / 448 is normal, and s = +inf yields 128 through the polynomial)."""
    return ue8m0_scale(word_ops(B), amax)


def check_quant_statics(N: int) -> None:
    if not (N > 0 and N & (N - 1) == 0):
        raise ValueError(f"DsaUe8m0Quant: the amax tree takes a power-of-two row (N={N})")


def ue8m0_quant(o: Ops, x: list, scale_def, e4m3=E4M3):
    """The ue8m0 quantisation of the N words ``x``: (q, scale), amax a halving tree of `fmaxf` over `fabs` (order-free), the scale a
    Call of ``scale_def``."""
    N = len(x)
    check_quant_statics(N)
    scale = o.B.call(scale_def, halving_tree([o.fabs(v) for v in x], o.fmaxf))
    q = [o.e4m3(o.div(v, scale)) for v in x]
    return tuple_of(Coll(Array(N, e4m3), concat([w.refs for w in q])), scale)


@composite("DsaUe8m0Quant", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Tuple(Array(S.N, E4M3), F32)), conformance=_CONF)
def DsaUe8m0Quant(B, S, x):
    """`_fp8_ue8m0_quantize` (kernels.py:50-60) on one row of N f32 words: scale = `DsaUe8m0Scale_v1`(max over |x|), q[i] =
    `cvt.rn.satfinite.e4m3x2.f32(div.rn(x[i], scale))`.  A NaN word is dropped by the max tree and quantises to 0x7F."""
    return ue8m0_quant(word_ops(B), [x[i] for i in range(S.N)], DsaUe8m0Scale)


def check_row_statics(D: int, ROPE: int) -> None:
    """GLM-5.3's rows are D = 128 (the PTX's); D = 32 or 64 is the same body at one or two warps, for the checks."""
    if not (D in (32, 64, 128) and 0 < ROPE <= D and ROPE % 2 == 0):
        raise ValueError(f"the indexer kernels take a row of 32 x a power-of-two warps (at most 4) with an even rotary prefix "
                         f"(D={D}, ROPE={ROPE})")


def rope_interleaved(o: Ops, x: list, cs, ROPE: int) -> list:
    """`INDEX_ROPE_INTERLEAVE`'s rotation of the first ROPE of the f32 words ``x`` (kernels.py:349-393, 770-803; PTX `neg.f32`,
    `selp`, `mul.f32`, `fma.rn.f32`): element i of pair j = i // 2 becomes `fma(x[i], cos[j], mul(s_i x[i ^ 1], sin[j]))`, s_i = -1
    for even i; the rest pass."""
    half = ROPE // 2
    cos = [o.cvt(cs[j]) for j in range(half)]
    sin = [o.cvt(cs[half + j]) for j in range(half)]
    out = list(x)
    for i in range(ROPE):
        partner = o.neg(x[i ^ 1]) if i % 2 == 0 else x[i ^ 1]
        out[i] = o.fma(x[i], cos[i // 2], o.mul(partner, sin[i // 2]))
    return out


def k_row(o: Ops, D: int, ROPE: int, k, w, b, cs, eps) -> list:
    """`DsaIndexerKRow_v1`'s f32 row before its quantisation."""
    check_row_statics(D, ROPE)
    x = [o.cvt(k[i]) for i in range(D)]
    warps = range(D // 32)
    n = o.word(f32_bits(D))
    rcp = o.rcp(n)
    mean = o.mul(o.scale_a(halving_tree([halving_tree(x[32 * g:32 * g + 32], o.add) for g in warps], o.add), n), rcp)
    nmean = o.neg(mean)
    d = [o.add(x[i], nmean) for i in range(D)]
    sq = [halving_tree([o.fma(d[32 * g + t], d[32 * g + t], o.mul(d[32 * g + t + 16], d[32 * g + t + 16])) for t in range(16)], o.add)
          for g in warps]
    var = o.mul(o.scale_a(halving_tree(sq, o.add), n), rcp)
    rstd = o.rsqrt(o.add(eps, var))
    normed = [o.fma(o.mul(d[i], rstd), w[i], b[i]) for i in range(D)]
    return rope_interleaved(o, normed, cs, ROPE)


def check_eps(EPS: float) -> None:
    if not float(np.float32(EPS)) >= 2.0 ** -126:
        raise ValueError(f"DsaIndexerKRow: EPS {EPS} below 2^-126 (rsqrt.approx.ftz and RsqrtApprox agree on operands from there)")


def _row(D: int, words: list, t=F32):
    return Coll(Array(D, t), concat([r.refs for r in words]))


def k_slot(o: Ops, S, k, w, b, cs, quant, t=F32):
    """`DsaIndexerKRow`'s body over ``o``'s steps, the quantisation a Call of ``quant`` (`DsaUe8m0Quant` at N = D)."""
    check_eps(S.EPS)
    return o.B.call(quant, _row(S.D, k_row(o, S.D, S.ROPE, k, w, b, cs, o.word(f32_bits(S.EPS))), t))


@composite("DsaIndexerKRow", 1, ["D", "ROPE", "EPS"],
           lambda S: ((("k", Array(S.D, BF16)), ("w", Array(S.D, F32)), ("b", Array(S.D, F32)), ("cs", Array(S.ROPE, BF16))),
                      Tuple(Array(S.D, E4M3), F32)),
           conformance=_CONF)
def DsaIndexerKRow(B, S, k, w, b, cs):
    """One index-K cache slot (`_fused_norm_rope_kernel` program 0, kernels.py:322-415; 128 threads, thread t element t): x = f32(k);
    the LayerNorm, its mean and variance `div.full.f32(sum, 128)` (`DivFullScaleA` times `DivFullRcp`), the sums each warp's
    butterfly then `(w0 + w2) + (w1 + w3)`, the squares' first level contracted (lane t < 16: `fma(d[t], d[t], mul(d[t+16],
    d[t+16]))`), `d = x - mean` (`sub.f32`, written as `add` of the negated mean: the same word but for NaN payloads, which no output
    reads), `rstd = rsqrt.approx.ftz(eps + var)` (`RsqrtApprox_v1`: the operand is at least eps, never subnormal), `n = fma(d * rstd,
    w, b)`; the interleaved RoPE (`rope_interleaved`; the kernel re-derives the partner's n from the same mean and rstd, the same
    word); then `DsaUe8m0Quant_v1{D}`: the slot's D e4m3 bytes and its f32 scale (kernels.py:76-87)."""
    return k_slot(word_ops(B), S, k, w, b, cs, bind(DsaUe8m0Quant, N=S.D))


def q_weight(o: Ops, D: int, NH: int, scale, w):
    """`index_weights *= q_scale; *= softmax_scale; *= head_scale` (kernels.py:816-826; PTX `mul.f32(scale, w)` then the two
    kernel arguments, f32(D^-0.5) and f32(NH^-0.5), each on the left)."""
    return o.mul(o.word(f32_bits(NH ** -0.5)), o.mul(o.word(f32_bits(D ** -0.5)), o.mul(scale, o.cvt(w))))


def q_head(o: Ops, S, q, cs, w, quant, t=F32):
    """`DsaIndexerQHead`'s body over ``o``'s steps, the quantisation a Call of ``quant``."""
    check_row_statics(S.D, S.ROPE)
    qs = o.B.call(quant, _row(S.D, rope_interleaved(o, [o.cvt(q[i]) for i in range(S.D)], cs, S.ROPE), t))
    return tuple_of(qs[0], q_weight(o, S.D, S.NH, qs[1], w))


@composite("DsaIndexerQHead", 1, ["D", "ROPE", "NH"],
           lambda S: ((("q", Array(S.D, BF16)), ("cs", Array(S.ROPE, BF16)), ("w", BF16)), Tuple(Array(S.D, E4M3), F32)),
           conformance=_CONF)
def DsaIndexerQHead(B, S, q, cs, w):
    """One (token, head) of `_fused_q_kernel` program 1 (kernels.py:761-826; one warp, lane l elements l, l+32, l+64, l+96): the
    interleaved RoPE of f32(q) (`rope_interleaved`), `DsaUe8m0Quant_v1{D}` (index_q_fp8's row), and the head's weight
    (`q_weight`, index_weights_out's word).  NH is the indexer's head count (32 on GLM-5.3, replicated at every TP rank)."""
    return q_head(word_ops(B), S, q, cs, w, bind(DsaUe8m0Quant, N=S.D))


def heads(B, head, H: int, D: int, q, cs, w, e4m3=E4M3, f32=F32):
    parts = [B.call(head, q[h], cs, w[h]) for h in range(H)]
    return tuple_of(Coll(Array(H, Array(D, e4m3)), concat([p[0].refs for p in parts])), Coll(Array(H, f32), concat([p[1].refs for p in parts])))


@composite("DsaIndexerQ", 1, ["H", "D", "ROPE"],
           lambda S: ((("q", Array(S.H, Array(S.D, BF16))), ("cs", Array(S.ROPE, BF16)), ("w", Array(S.H, BF16))),
                      Tuple(Array(S.H, Array(S.D, E4M3)), Array(S.H, F32))),
           conformance=_CONF)
def DsaIndexerQ(B, S, q, cs, w):
    """`fused_q`'s indexer outputs for one token (`index_q_fp8` [H, D], `index_weights_out` [H]): head h `DsaIndexerQHead_v1{D, ROPE,
    NH=H}` over q[h] and w[h]."""
    return heads(B, bind(DsaIndexerQHead, D=S.D, ROPE=S.ROPE, NH=S.H), S.H, S.D, q, cs, w)


def check_logit_statics(H: int, D: int) -> None:
    if not (H > 0 and H % 8 == 0 and D > 0 and D % 32 == 0):
        raise ValueError(f"DeepGEMM's MQA logits need H % 8 == 0 and D % 32 == 0 (H={H}, D={D})")


def mqa_logit(o: Ops, H: int, D: int, q, w, k, ks, contracted: bool = False):
    """`DsaMqaLogit_v1`'s body, or with ``contracted`` `DsaMqaLogit_v3`'s: a lane's sum over its heads hs starts from
    `mul(relu[hs0], w[hs0])` (v1) or from `fma(relu[hs0], w[hs0], mul(relu[hs1], w[hs1]))` (v3; the same at H = 8, one head)."""
    check_logit_statics(H, D)
    zero = o.word(0)
    relu = []
    for h in range(H):
        acc = zero
        for s in range(D // 32):
            acc = o.dot(acc, k[32 * s:32 * s + 32], q[h][32 * s:32 * s + 32])
        relu.append(o.fmaxf(acc, zero))
    v = []
    for c in range(4):
        sums = []
        for pair in (0, 1):
            hs = [2 * c + pair + 8 * j for j in range(H // 8)]
            if contracted and len(hs) > 1:
                acc, rest = o.fma(relu[hs[0]], w[hs[0]], o.mul(relu[hs[1]], w[hs[1]])), hs[2:]
            else:
                acc, rest = o.mul(relu[hs[0]], w[hs[0]]), hs[1:]
            for h in rest:
                acc = o.fma(relu[h], w[h], acc)
            sums.append(acc)
        v.append(o.mul(o.add(sums[0], sums[1]), ks))
    return o.add(o.add(v[0], v[1]), o.add(v[2], v[3]))


def _logit_sig(S):
    return (("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.D, E4M3)), ("ks", F32)), F32


def _logits_sig(S):
    return ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.N, Array(S.D, E4M3))),
             ("ks", Array(S.N, F32))), Array(S.N, F32))


@composite("DsaMqaLogit", 1, ["H", "D"], _logit_sig, conformance=_LOGIT_V1_CONF)
def DsaMqaLogit(B, S, q, w, k, ks):
    """One logit of DeepGEMM's sm90 MQA logits (prefill and paged decode share the epilogue): per head h, `acc[h]` = D/32
    `HopperE4m3QgmmaDot32` steps (wgmma m64nNk32 e4m3, the key row as A, the head's q as B) from `scale-d = 0`; the thread of quad lane
    c holds heads 2c + 8j and 2c + 1 + 8j (j < H/8: the m64nN accumulator layout) and sums each from `fmaxf(acc, 0) * w` by FFMAs in
    ascending j (`transform`, cuh:280-293), `v_c = (s0 + s1) * ks` (cuh:294-295), and the quad's xor-1 then xor-2 shuffle adds give
    `(v_0 + v_1) + (v_2 + v_3)` (cuh:298-302), stored as f32."""
    return mqa_logit(word_ops(B), S.H, S.D, q, w, k, ks)


@composite("DsaMqaLogits", 1, ["N", "H", "D"], _logits_sig, conformance=_LOGIT_V1_CONF)
def DsaMqaLogits(B, S, q, w, k, ks):
    """One query token's logits over N keys: logit[n] = `DsaMqaLogit_v1{H, D}`(q, w, k[n], ks[n])."""
    return B.batch(bind(DsaMqaLogit, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


@composite("DsaMqaLogit", 3, ["H", "D"], _logit_sig, conformance=_LOGIT_CONF)
def DsaMqaLogitV3(B, S, q, w, k, ks):
    """One logit of DeepGEMM's sm90 MQA logits as the SASS computes it: `DsaMqaLogit_v1` but for each lane's first two heads.  The
    source starts each sum from j = 0's product and adds the others' (`sum[4] = {transform(0), ..}`, `sum[k] += transform(j * 4 +
    k)`: sm90_fp8_mqa_logits.cuh:288-293, sm90_fp8_paged_mqa_logits.cuh:333-338), and nvcc contracts j = 0's product into j = 1's
    add: `fma(relu[h0], w[h0], relu[h1] * w[h1])`, then one FFMA per j >= 2 in ascending j.  The rest is v1's: the per-head dots,
    `v_c = (s0 + s1) * ks` and `(v_0 + v_1) + (v_2 + v_3)`."""
    return mqa_logit(word_ops(B), S.H, S.D, q, w, k, ks, contracted=True)


@composite("DsaMqaLogits", 3, ["N", "H", "D"], _logits_sig, conformance=_LOGIT_CONF)
def DsaMqaLogitsV3(B, S, q, w, k, ks):
    """One query token's logits over N keys: logit[n] = `DsaMqaLogit_v3{H, D}`(q, w, k[n], ks[n])."""
    return B.batch(bind(DsaMqaLogitV3, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


def topk_short(o: Ops, TOPK: int, n) -> list:
    """`DsaTopKShortRow_v1`'s words: `i < len ? i : -1` as `SelectI32(I32Le(len, i), -1, i)`."""
    m1 = o.word(M32)
    out = []
    for i in range(TOPK):
        ci = o.word(i)
        out.append(o.isel(o.ile(n, ci), m1, ci))
    return out


@composite("DsaTopKShortRow", 1, ["TOPK"], lambda S: ((("len", I32),), Array(S.TOPK, I32)), conformance=_TOPK_CONF)
def DsaTopKShortRow(B, S, n):
    """The selection of a row of at most TOPK keys: slot i holds i when i < len, else -1 (signed; a length of 0 or less is all -1).
    Prefill (`topKPerRowPrefill`, sampler.cu:388-412) has len = rowEnd - rowStart and writes indices from the row's start; decode
    (`cooperative_topk`, cooperative_topk.cuh:494-501) clamps a negative length at 0.  A row of more than TOPK keys is the radix
    selection, whose indices `DsaTopKSelection_v1` checks."""
    return Coll(Array(S.TOPK, I32), concat([w.refs for w in topk_short(word_ops(B), S.TOPK, n)]))


INT_MAX, INT_MIN = 0x7FFFFFFF, 0x80000000


def check_selection_statics(TOPK: int, N: int) -> None:
    if not 0 < TOPK < N:
        raise ValueError(f"DsaTopKSelection: a row of N > TOPK > 0 keys (TOPK={TOPK}, N={N})")


def pairwise(vals: list, op) -> object:
    """``op`` over ``vals`` as a balanced tree: neighbours meet level by level, an odd one out passing up."""
    while len(vals) > 1:
        vals = [op(vals[t], vals[t + 1]) for t in range(0, len(vals) - 1, 2)] + vals[len(vals) - len(vals) % 2:]
    return vals[0]


def order_key(o: Ops, x):
    """The top-k kernels' order on the f32 word x as a signed word (the module docstring): x when its sign is clear, else
    0x7FFFFFFF - x."""
    return o.isel(o.ile(o.word(0), x), x, o.isub(o.word(INT_MAX), x))


def selection_parts(o: Ops, TOPK: int, N: int, n, idx) -> tuple:
    """The steps `DsaTopKSelection_v1` and `_v3` share, in v1's order: ``le[j]`` = len <= j (j <= N), ``short`` (idx the
    shortcut's words), ``inside`` (every index in [0, len)), ``member[j]`` (some idx[i] == j) and ``count`` (the members)."""
    check_selection_statics(TOPK, N)
    zero, one = o.word(0), o.word(1)
    le = [o.ile(n, o.word(j)) for j in range(N + 1)]
    short = pairwise([o.ieq(idx[i], o.isel(le[i], o.word(M32), o.word(i))) for i in range(TOPK)], o.band)
    inside = pairwise([o.band(o.ile(zero, idx[i]), o.bnot(o.ile(n, idx[i]))) for i in range(TOPK)], o.band)
    member = [pairwise([o.ieq(idx[i], o.word(j)) for i in range(TOPK)], o.bor) for j in range(N)]
    count = pairwise([o.isel(m, one, zero) for m in member], o.iadd)
    return le, short, inside, member, count


def _selected(o: Ops, TOPK: int, N: int, le, short, inside, count, fits):
    """The bit from the shared steps and ``fits()``, the long row's key condition (emitted after the count's comparison)."""
    tight = o.band(inside, o.ieq(count, o.word(TOPK)))
    long_ok = o.band(tight, o.band(fits(), le[N]))
    return o.bor(o.band(le[TOPK], short), o.band(o.bnot(le[TOPK]), long_ok))


def topk_selection(o: Ops, TOPK: int, N: int, logits, n, idx):
    """`DsaTopKSelection_v1`'s bit."""
    le, short, inside, member, count = selection_parts(o, TOPK, N, n, idx)
    key = [order_key(o, logits[j]) for j in range(N)]
    least = pairwise([o.isel(member[j], key[j], o.word(INT_MAX)) for j in range(N)], lambda a, b: o.isel(o.ile(a, b), a, b))
    most = pairwise([o.isel(o.bor(member[j], le[j]), o.word(INT_MIN), key[j]) for j in range(N)],
                    lambda a, b: o.isel(o.ile(a, b), b, a))
    return _selected(o, TOPK, N, le, short, inside, count, lambda: o.ile(most, least))


def topk_threshold_selection(o: Ops, TOPK: int, N: int, logits, n, idx, thr):
    """`DsaTopKSelection_v3`'s bit: v1's, its key comparison made against the threshold key ``order_key(thr)``."""
    le, short, inside, member, count = selection_parts(o, TOPK, N, n, idx)
    tk = order_key(o, thr)
    fits = []
    for j in range(N):
        key = order_key(o, logits[j])
        fits.append(o.band(o.bor(o.bnot(member[j]), o.ile(tk, key)), o.bor(o.bor(member[j], le[j]), o.ile(key, tk))))
    return _selected(o, TOPK, N, le, short, inside, count, lambda: pairwise(fits, o.band))


@composite("DsaTopKSelection", 1, ["TOPK", "N"],
           lambda S: ((("logits", Array(S.N, F32)), ("len", I32), ("idx", Array(S.TOPK, I32))), BIT), conformance=_SELECTION_CONF)
def DsaTopKSelection(B, S, logits, n, idx):
    """Whether the TOPK indices ``idx`` are a selection the top-k kernels may write for a row of ``len`` keys, logits[:len] (the
    module docstring, step 5).  For len <= TOPK they are the shortcut's words, `DsaTopKShortRow_v1`'s.  For TOPK < len <= N they
    are TOPK distinct indices in [0, len), in any order, the least key among them (`order_key`, signed `I32Le`) at least the
    greatest key of the rest of the row: position j is a member when some idx[i] == j, the members number TOPK (with every index
    in range, so the indices are distinct), and the members' least key (INT_MAX when there is none) is at least the greatest key of
    the non-members below len (INT_MIN when there is none).  For len > N, never.  Which of the keys tied at the threshold the
    kernels take, and in what order they write the selection, follows their atomics, so the indices are the kernel's own and this
    function checks them."""
    return topk_selection(word_ops(B), S.TOPK, S.N, logits, n, idx)


@composite("DsaTopKSelection", 3, ["TOPK", "N"],
           lambda S: ((("logits", Array(S.N, F32)), ("len", I32), ("idx", Array(S.TOPK, I32)), ("thr", F32)), BIT),
           conformance=_SELECTION_CONF)
def DsaTopKSelectionV3(B, S, logits, n, idx, thr):
    """`DsaTopKSelection_v1` with a threshold: the same bit, except that for TOPK < len <= N the members' least key need not be
    computed.  Instead every member's key is at least the threshold key `order_key(thr)` and every non-member's below len at
    most it.  So v1 is 1 exactly when v3 is 1 for some ``thr``, and `selection_threshold` names one (a member's word of least
    key); a row of len <= TOPK or len > N reads no ``thr``.  ``thr`` is the prover's witness, not the kernel's: this form needs
    no least or greatest key over the row, so `boolean_dsa.DsaTopKSelection_v4` checks it chunk by chunk."""
    return topk_threshold_selection(word_ops(B), S.TOPK, S.N, logits, n, idx, thr)


def order_keys(words) -> np.ndarray:
    """`order_key` on an array of f32 words (float32, or their bits as integers), as int64."""
    w = np.asarray(words)
    w = (w.view(np.uint32) if w.dtype == np.float32 else w.astype(np.int64) & M32).astype(np.int64)
    return np.where(w <= INT_MAX, w, INT_MAX - w)


def _signed(v) -> np.ndarray:
    v = np.asarray(v).astype(np.int64) & M32
    return np.where(v > INT_MAX, v - (1 << 32), v)


def selection_ok(words, n: int, idx) -> bool:
    """`DsaTopKSelection_v1{TOPK = len(idx), N = len(words)}` in numpy: ``words`` the row's N f32 words, ``n`` its length, ``idx``
    the TOPK indices (int32, or their bits)."""
    idx, n = _signed(idx).reshape(-1), int(_signed(n))
    TOPK, N = idx.size, len(words)
    check_selection_statics(TOPK, N)
    j = np.arange(TOPK)
    if n <= TOPK:
        return bool(np.array_equal(idx, np.where(j < n, j, -1)))
    if n > N or not ((idx >= 0) & (idx < n)).all() or np.unique(idx).size != TOPK:
        return False
    key = order_keys(np.asarray(words)[:n])
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return bool(key[mask].min() >= key[~mask].max())


def selection_ok_thr(words, n: int, idx, thr) -> bool:
    """`DsaTopKSelection_v3{TOPK = len(idx), N = len(words)}` in numpy (``thr`` an f32 word, float32 or its bits)."""
    idx, n = _signed(idx).reshape(-1), int(_signed(n))
    TOPK, N = idx.size, len(words)
    check_selection_statics(TOPK, N)
    if n <= TOPK or n > N:
        return selection_ok(words, n, idx)
    if not ((idx >= 0) & (idx < n)).all() or np.unique(idx).size != TOPK:
        return False
    key, t = order_keys(np.asarray(words)[:n]), int(order_keys(np.asarray([thr]))[0])
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return bool((key[mask] >= t).all() and (key[~mask] <= t).all())


def selection_threshold(words, n: int, idx) -> int:
    """A ``thr`` for `DsaTopKSelection_v3`: the word of least key among the row's words that ``idx`` names in [0, min(len, N)),
    0 when it names none.  Whenever v1 accepts (words, len, idx), v3 accepts it with this word, so v3 at this word is v1."""
    w = np.asarray(words)
    w = (w.view(np.uint32) if w.dtype == np.float32 else w.astype(np.int64) & M32).astype(np.int64)
    idx, n = _signed(idx).reshape(-1), int(_signed(n))
    named = idx[(idx >= 0) & (idx < min(n, len(w)))]
    if not named.size:
        return 0
    return int(w[named[np.argmin(order_keys(w[named]))]])


DSA_COMPOSITES = [DsaUe8m0Scale, DsaUe8m0Quant, DsaIndexerKRow, DsaIndexerQHead, DsaIndexerQ, DsaMqaLogit, DsaMqaLogits,
                  DsaMqaLogitV3, DsaMqaLogitsV3, DsaTopKShortRow, DsaTopKSelection, DsaTopKSelectionV3]
