"""The kernels vLLM serves at its defaults under torch.compile (enforce_eager False, its own CUDA-graph mode, no batch invariance):
Inductor's fused Triton kernels and FA3's split decode, each with its launch config as statics, read from the kernels' PTX on sm_90.

Inductor's RMSNorm.  The four fused forms (`triton_red_fused_*rms_norm*`) differ only in how the row r is formed in f32: the row
itself (after the embedding gather), x + res, or x + (r1 + r2) with no bf16 rounding of r1 + r2 (`InductorAdd2RMSNorm_v1` also
stores bf16(r)).  The reduction is the launch's: a row has T = 32 NUM_WARPS / XBLOCK threads, a power of two, and thread t of
the row holds columns [VEC t, VEC t + VEC) of each R0_BLOCK chunk, VEC = R0_BLOCK / T of 2, 4 or 8 (one 4-, 8- or 16-byte
load); across chunks each column's accumulator is `fma(r, r, acc)` from +0; the thread adds its VEC in order, each warp
butterflies the row's lanes (xor min(T, 32)/2 .. 1), and a row of wpr = T / 32 > 1 warps butterflies its warp sums (xor
wpr/2 .. 1) through shared memory (`ld.shared` after the lanes' butterfly).  A row of fewer than 32 lanes (several rows a warp,
`threadsPerWarp [32/T, T]`) was read at cc 12.0: Qwen3's q/k head norm at XBLOCK 64, R0_BLOCK 64, 16 warps.  Then mean =
`div.full.f32(sum, N)` (`F32Mul_v2(DivFullScaleA_v2, DivFullRcp)`), `mean + EPS` (add.f32, not contracted),
`rsqrt.approx.ftz.f32` (`RsqrtApprox_v1`: its argument is at least EPS, a positive normal static, so the flush never applies),
and out = bf16((r * rstd) * w).  Any other layout is refused: its register order is unverified.

Inductor's per-head q/k norm fused with NeoX RoPE (Qwen3's attention, the combo kernel `triton_red_fused_*` over the qkv buffer,
the head norms' weights and the cos/sin cache row): each head's RMSNorm reduction as above at N = D, then the rotation of
y = (x * rstd) * w in f32, no bf16 store between them: `fma(y_lo, c, -(y_hi * s))` and `fma(y_hi, c, y_lo * s)`, `F2fpBf16`.

Inductor's SiLU-and-mul (`triton_poi_fused_mul_silu_slice_*`): n = 0 - g (sub.f32), libdevice's `expf` with ftz in its range
reduction (as Triton links it: `fma.rn.ftz`, `cvt.ftz.sat`, `fma.rm.ftz`, `ex2.approx.ftz`), whose final scaling by 2^j is
contracted with the `+ 1` into one `fma.rn.f32(e, 2^j, 1)`, then q = div.full.f32(g, d) and out = bf16(q * u).  `cvt.ftz.sat`
reads an `F32FmaFtz` result, never subnormal, so it is `F32Sat_v1`; `fma.rm.ftz` reads that (0 or a normal number) and two normal
constants and writes a number of at least 2^23, so it is `F32FmaRm_v1`.

Inductor's NeoX RoPE (`triton_poi_fused_*` over the qkv buffer and the cos/sin cache row): `fma(x, c, -(y * s))` and
`fma(y, c, x * s)` as `RopeOut_v1` / `RopeOutAdd_v1` compute them, stored by `cvt.rn.bf16x2.f32` (`F2fpBf16`: a NaN is 0x7FFF, where
the CUDA kernel `RopeOut_v1` models stores torch's 0x7FC0).

FA3's split decode (vllm-flash-attn 506341a1, `FlashAttnFwdSm90<..., Split>` and `FlashAttnFwdCombine`): a sequence of T keys in
NS splits (`sparse_mla_split.split_blocks`) runs each split's blocks last to first as `AttnBlock_v4`, the split's first visited
block FIRST and every block CHECK where it is first or at or above MASKED_FROM (the tile's `n_block_min_causal_local_mask`); each
split ends with `SparseMlaSplitEnd_v1` and the splits are merged by `SparseMlaCombine_v1` (the 16-lane, two-splits-a-lane combine
at head dim 64 sums lane j's `0 + scale_j` then `+ scale_(j+16)`, the 32-lane butterfly's first level).  NS = 1 is
`AttentionHead_v4`, as the epilogue writes a one-split sequence directly.

Registered functions (all `Name_v1`):
  InductorRMSNorm{N,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}           (x: N bf16, w: N bf16) -> N bf16
  InductorAddRMSNorm{N,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}        (x, res, w) -> N bf16
  InductorAdd2RMSNorm{N,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}       (x, r1, r2, w) -> (bf16(r), normed)
  InductorAdd2RMSNormFinal{N,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}  (x, r1, r2, w) -> normed
  InductorSiluMulElem                                        (g: bf16, u: bf16) -> bf16
  InductorSiluMul{I}                                         (gu: 2I bf16) -> I bf16
  InductorRoPEHead{D}, InductorRoPE{NHEADS,D}                (x: [NHEADS x] D bf16, cs: D bf16) -> [NHEADS x] D bf16
  InductorQKNormRoPEHead{D,EPS,XBLOCK,R0_BLOCK,NUM_WARPS}    (x: D bf16, w: D bf16, cs: D bf16) -> D bf16
  InductorQKNormRoPE{NHEADS,D,EPS,XBLOCK,R0_BLOCK,NUM_WARPS} (x: NHEADS x D bf16, w: D bf16, cs: D bf16) -> NHEADS x D bf16
  Fa3AttentionHeadSplit{T,NS,D,BN,DOT,INV,MASKED_FROM}       (q: D, k: T x D, v: T x D) -> D bf16
  Fa3AttentionSplit{T,NS,NH,KVH,D,BN,DOT,INV,MASKED_FROM}    (q: NH x D, kc: T x KVH x D, vc: T x KVH x D) -> NH x D bf16
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops.prims import ZERO32
from verity.ml.circuits.ops.scalar import F32BitsShl23, F32Neg, F32Sat
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import sparse_mla_split as SMS
from verity.ml.definitions.registry.b1 import ZERO16, f32c
from verity.ml.definitions.registry.fa3_check_inf import AttentionHeadV4, AttnBlockV4
from verity.ml.definitions.registry.moe import (C_252, C_HALF, C_LOG2E_HI, C_LOG2E_LO, C_LOG2E_OVER_252, C_MAGIC, C_NEG_MAGIC_BIAS,
                                              ONE32, F32FmaRm)
from verity.ml.definitions.registry.prims import BF16, F32

#: bf16 columns per thread per chunk the PTX was read at: one 4-, 8- or 16-byte vector load
VECS = (2, 4, 8)
WARP = 32
_CONF = ("model-untested: read from the generated Triton kernels' PTX at cc 9.0, a norm row of fewer than 32 lanes at cc 12.0 "
         "(the module docstring); "
         "tests/properties/served_compiled_kernels_gpu.py is its hardware check")


# --- RMSNorm ------------------------------------------------------------------------------------------------------------------

def norm_layout(N: int, EPS: float, XBLOCK: int, R0_BLOCK: int, NUM_WARPS: int) -> tuple[int, int, int]:
    """(threads per row, chunks per row, columns per thread per chunk) of a reduction launch; refuses a layout this module has
    not read from PTX."""
    tpr = WARP * NUM_WARPS // XBLOCK if XBLOCK >= 1 and NUM_WARPS >= 1 and WARP * NUM_WARPS % XBLOCK == 0 else 0
    vec = R0_BLOCK // tpr if tpr >= 1 and R0_BLOCK % tpr == 0 else 0
    if tpr < 1 or tpr & (tpr - 1) or vec not in VECS or N < 1 or N % vec:
        raise ValueError(f"Inductor norm: N={N} XBLOCK={XBLOCK} R0_BLOCK={R0_BLOCK} NUM_WARPS={NUM_WARPS} is not one chunk of "
                         f"{'/'.join(map(str, VECS))} columns per thread over a power-of-two count of threads per row")
    e = np.float32(EPS)
    if not (np.isfinite(e) and e >= np.finfo(np.float32).tiny):
        raise ValueError(f"Inductor norm: EPS={EPS} is not a positive normal f32 (rsqrt.approx.ftz reads mean + EPS)")
    return tpr, -(-N // R0_BLOCK), vec


def _lane0(B, vals):
    """Lane 0's value after an xor butterfly of adds over len(vals) lanes (xor len/2 .. 1), its adds alone: lane i adds lane
    i + o.  Every lane ends with the same word (each level adds equal pairs), so lane 0 is the row's."""
    while len(vals) > 1:
        o = len(vals) // 2
        vals = [B.call(P.F32AddV2, vals[i], vals[i + o]) for i in range(o)]
    return vals[0]


def _rstd(B, S, r, N):
    """rstd of the row r (N f32 values, a Python list) under the launch's reduction; a column slot no chunk reaches keeps the
    kernel's +0 accumulator and is added like any other."""
    tpr, chunks, vec = norm_layout(N, S.EPS, S.XBLOCK, S.R0_BLOCK, S.NUM_WARPS)
    zero = B.call(ZERO32)
    acc = []
    for c in range(S.R0_BLOCK):
        a = zero
        for k in range(chunks):
            j = k * S.R0_BLOCK + c
            if j < N:
                a = B.call(P.F32FmaV2, r[j], r[j], a)
        acc.append(a)
    threads = []
    for t in range(tpr):
        s = acc[vec * t]
        for x in acc[vec * t + 1:vec * t + vec]:
            s = B.call(P.F32AddV2, s, x)
        threads.append(s)
    lanes = min(tpr, WARP)
    total = _lane0(B, [_lane0(B, threads[i:i + lanes]) for i in range(0, tpr, lanes)])
    nf = B.call(f32c(float(N)))
    mean = B.call(P.F32MulV2, B.call(P.DivFullScaleAV2, total, nf), B.call(P.DivFullRcp, nf))
    return B.call(P.RsqrtApprox, B.call(P.F32AddV2, mean, B.call(f32c(S.EPS))))


def _scaled(B, r, rstd, w):
    """(r * rstd) * w in f32."""
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    return B.batch(P.F32MulV2, B.batch(P.F32MulV2, array_of(r), rstd, axes=(0, None)), wf, axes=(0, 0))


def _norm(B, S, r, w):
    """The normed row of r (N f32 values, a Python list): bf16((r * rstd) * w)."""
    return B.batch(P.F2fpBf16, _scaled(B, r, _rstd(B, S, r, S.N), w), axes=(0,))


def _rows(B, *xs):
    return [B.batch(P.Bf16ToF32, x, axes=(0,)) for x in xs]


def _add2(B, S, x, r1, r2):
    xf, af, bf = _rows(B, x, r1, r2)
    return [B.call(P.F32AddV2, xf[j], B.call(P.F32AddV2, af[j], bf[j])) for j in range(S.N)]


_NORM_STATICS = ["N", "EPS", "XBLOCK", "R0_BLOCK", "NUM_WARPS"]


def _norm_type(*names, out=None):
    def sig(S, bf16=BF16):
        row = Array(S.N, bf16)
        return tuple((n, row) for n in names), (out(row) if out else row)
    return sig


@composite("InductorRMSNorm", 1, _NORM_STATICS, _norm_type("x", "w"), conformance=_CONF)
def InductorRMSNorm(B, S, x, w):
    """Inductor's RMSNorm of the row x itself (the embedding form, after `Embedding_v1`): out = bf16((x * rstd) * w)."""
    xf, = _rows(B, x)
    return _norm(B, S, [xf[j] for j in range(S.N)], w)


@composite("InductorAddRMSNorm", 1, _NORM_STATICS, _norm_type("x", "res", "w"), conformance=_CONF)
def InductorAddRMSNorm(B, S, x, res, w):
    """Inductor's RMSNorm of r = x + res (add.f32): out = bf16((r * rstd) * w)."""
    xf, rf = _rows(B, x, res)
    return _norm(B, S, [B.call(P.F32AddV2, xf[j], rf[j]) for j in range(S.N)], w)


@composite("InductorAdd2RMSNorm", 1, _NORM_STATICS, _norm_type("x", "r1", "r2", "w", out=lambda row: Tuple(row, row)),
           conformance=_CONF)
def InductorAdd2RMSNorm(B, S, x, r1, r2, w):
    """Inductor's RMSNorm of r = x + (r1 + r2) (two add.f32, r1 + r2 not rounded to bf16): (bf16(r), bf16((r * rstd) * w))."""
    r = _add2(B, S, x, r1, r2)
    return tuple_of(B.batch(P.F2fpBf16, array_of(r), axes=(0,)), _norm(B, S, r, w))


@composite("InductorAdd2RMSNormFinal", 1, _NORM_STATICS, _norm_type("x", "r1", "r2", "w"), conformance=_CONF)
def InductorAdd2RMSNormFinal(B, S, x, r1, r2, w):
    """`InductorAdd2RMSNorm_v1`'s normed output alone (the model's last norm stores no residual)."""
    return _norm(B, S, _add2(B, S, x, r1, r2), w)


# --- SiLU-and-mul, RoPE -------------------------------------------------------------------------------------------------------

def silu_mul(B, g, u):
    """One element of `InductorSiluMul_v1` on bf16 words g, u."""
    gf, uf = B.call(P.Bf16ToF32, g), B.call(P.Bf16ToF32, u)
    n = B.call(P.F32SubV2, B.call(ZERO32), gf)
    s = B.call(F32Sat, B.call(P.F32FmaFtzV2, n, B.call(C_LOG2E_OVER_252), B.call(C_HALF)))
    jb = B.call(F32FmaRm, s, B.call(C_252), B.call(C_MAGIC))
    j = B.call(P.F32AddV2, jb, B.call(C_NEG_MAGIC_BIAS))
    f = B.call(P.F32FmaFtzV2, n, B.call(C_LOG2E_HI), B.call(F32Neg, j))
    f = B.call(P.F32FmaFtzV2, n, B.call(C_LOG2E_LO), f)
    d = B.call(P.F32FmaV2, B.call(P.MufuEx2Ftz, f), B.call(F32BitsShl23, jb), B.call(ONE32))
    q = B.call(P.F32MulV2, B.call(P.DivFullScaleAV2, gf, d), B.call(P.DivFullRcp, d))
    return B.call(P.F2fpBf16, B.call(P.F32MulV2, q, uf))


@composite("InductorSiluMulElem", 1, [], lambda S: ((("g", BF16), ("u", BF16)), BF16), conformance=_CONF)
def InductorSiluMulElem(B, S, g, u):
    """bf16(div.full(g, fma(ex2(f), 2^j, 1)) * u): libdevice `expf(0 - g)` with its ftz range reduction, the 2^j scaling
    contracted with the + 1."""
    return silu_mul(B, g, u)


@composite("InductorSiluMul", 1, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)), conformance=_CONF)
def InductorSiluMul(B, S, gu):
    """`InductorSiluMulElem_v1` over gate = gu[:I], up = gu[I:]."""
    return B.batch(InductorSiluMulElem, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


@composite("InductorRoPEHead", 1, ["D"], lambda S: ((("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
           conformance=_CONF)
def InductorRoPEHead(B, S, x, cs):
    """NeoX rotation of one head with the cache row cs = [cos | sin], R = D/2: out[i] = bf16(fma(x[i], cos[i], -(x[i+R] *
    sin[i]))), out[i+R] = bf16(fma(x[i+R], cos[i], x[i] * sin[i])), `F2fpBf16` stores."""
    R, D = S.D // 2, S.D
    lo, hi, cos, sin = (B.batch(P.Bf16ToF32, a, axes=(0,)) for a in (x[0:R], x[R:D], cs[0:R], cs[R:D]))
    return _rope(B, D, lo, hi, cos, sin)


def _rope(B, D, lo, hi, cos, sin):
    """The rotation of f32 halves lo, hi: [bf16(fma(lo, cos, -(hi * sin))) | bf16(fma(hi, cos, lo * sin))]."""
    mul, fma, cvt = (lambda a, b: B.batch(P.F32MulV2, a, b, axes=(0, 0))), P.F32FmaV2, P.F2fpBf16
    first = B.batch(fma, lo, cos, B.batch(F32Neg, mul(hi, sin), axes=(0,)), axes=(0, 0, 0))
    second = B.batch(fma, hi, cos, mul(lo, sin), axes=(0, 0, 0))
    return Coll(Array(D, BF16), concat([B.batch(cvt, first, axes=(0,)).refs, B.batch(cvt, second, axes=(0,)).refs]))


@composite("InductorRoPE", 1, ["NHEADS", "D"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("cs", Array(S.D, BF16))), Array(S.NHEADS, Array(S.D, BF16))),
           conformance=_CONF)
def InductorRoPE(B, S, x, cs):
    return B.batch(bind(InductorRoPEHead, D=S.D), x, cs, axes=(0, None))


_HEAD_NORM_STATICS = ["D", "EPS", "XBLOCK", "R0_BLOCK", "NUM_WARPS"]


@composite("InductorQKNormRoPEHead", 1, _HEAD_NORM_STATICS,
           lambda S: ((("x", Array(S.D, BF16)), ("w", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
           conformance=_CONF)
def InductorQKNormRoPEHead(B, S, x, w, cs):
    """One head's RMSNorm (the launch's reduction at N = D) and NeoX rotation of y = (x * rstd) * w, unrounded, by the cache row
    cs = [cos | sin]: out[i] = bf16(fma(y[i], cos[i], -(y[i+R] * sin[i]))), out[i+R] = bf16(fma(y[i+R], cos[i], y[i] * sin[i]))."""
    R, D = S.D // 2, S.D
    xf, = _rows(B, x)
    r = [xf[j] for j in range(D)]
    y = _scaled(B, r, _rstd(B, S, r, D), w)
    cos, sin = (B.batch(P.Bf16ToF32, a, axes=(0,)) for a in (cs[0:R], cs[R:D]))
    return _rope(B, D, y[0:R], y[R:D], cos, sin)


@composite("InductorQKNormRoPE", 1, ["NHEADS"] + _HEAD_NORM_STATICS,
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("w", Array(S.D, BF16)), ("cs", Array(S.D, BF16))),
                      Array(S.NHEADS, Array(S.D, BF16))),
           conformance=_CONF)
def InductorQKNormRoPE(B, S, x, w, cs):
    """`InductorQKNormRoPEHead_v1` on each of NHEADS heads, one norm weight and one cache row."""
    head = bind(InductorQKNormRoPEHead, D=S.D, EPS=S.EPS, XBLOCK=S.XBLOCK, R0_BLOCK=S.R0_BLOCK, NUM_WARPS=S.NUM_WARPS)
    return B.batch(head, x, w, cs, axes=(0, None, None))


# --- FA3 split decode ---------------------------------------------------------------------------------------------------------

def _head_type(S, bf16=BF16):
    return (("q", Array(S.D, bf16)), ("k", Array(S.T, Array(S.D, bf16))), ("v", Array(S.T, Array(S.D, bf16)))), Array(S.D, bf16)


@composite("Fa3AttentionHeadSplit", 1, ["T", "NS", "D", "BN", "DOT", "INV", "MASKED_FROM"], _head_type, conformance=_CONF)
def Fa3AttentionHeadSplit(B, S, q, k, v):
    """One query head over T keys in NS splits: `AttentionHead_v4` at NS = 1; else each split's `AttnBlock_v4`s last to first
    (FIRST at its first visited block, CHECK there and at blocks >= MASKED_FROM, the sequence's last block zero-padded), its
    `SparseMlaSplitEnd_v1{DV=D, SCALE=D^-1/2}` and `SparseMlaCombine_v1` over the non-empty splits."""
    T, NS, D, BN, MF = S.T, S.NS, S.D, S.BN, S.MASKED_FROM
    SMS.check_split_statics(T, NS)
    if NS == 1:
        return B.call(bind(AttentionHeadV4, T=T, D=D, BN=BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=MF), q, k, v)
    nb = -(-T // BN)
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"Fa3AttentionHeadSplit: MASKED_FROM {MF} outside blocks 0..{nb - 1}")
    end = bind(SMS.SparseMlaSplitEnd, DV=D, INV=S.INV, SCALE=D ** -0.5)
    zero_row = None
    lses, parts = [], []
    for b0, b1 in SMS.split_blocks(T, NS, BN):
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = k[lo:hi], v[lo:hi]
            if hi - lo < BN:
                if zero_row is None:
                    zero_row = Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
                pad = [zero_row.refs] * (BN - hi + lo)
                kb = Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
                vb = Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
            first = b == b1 - 1
            blk = bind(AttnBlockV4, D=D, NVIS=hi - lo, FIRST=first, BN=BN, DOT=S.DOT, CHECK=first or b >= MF)
            if first:
                z = B.call(ZERO32)
                st = B.call(blk, q, kb, vb, z, Coll(Array(4, F32), concat([z.refs] * 4)), Coll(Array(D, F32), concat([z.refs] * D)))
            else:
                st = B.call(blk, q, kb, vb, m, l, o)
            m, l, o = st[0], st[1], st[2]
        e = B.call(end, m, l, o)
        parts.append(e[0])
        lses.append(e[1])
    return B.call(bind(SMS.SparseMlaCombine, NSPLIT=len(parts), DV=D, INV=S.INV), array_of(lses), array_of(parts))


@composite("Fa3AttentionSplit", 1, ["T", "NS", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))), Array(S.NH, Array(S.D, BF16))),
           conformance=_CONF)
def Fa3AttentionSplit(B, S, q, kc, vc):
    """One query token's NH heads over T keys in NS splits (GQA: head h reads KV head h // (NH / KVH)), each
    `Fa3AttentionHeadSplit_v1`."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    head = bind(Fa3AttentionHeadSplit, T=T, NS=S.NS, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


INDUCTOR_COMPOSITES = [InductorRMSNorm, InductorAddRMSNorm, InductorAdd2RMSNorm, InductorAdd2RMSNormFinal, InductorSiluMulElem,
                       InductorSiluMul, InductorRoPEHead, InductorRoPE, InductorQKNormRoPEHead, InductorQKNormRoPE,
                       Fa3AttentionHeadSplit, Fa3AttentionSplit]
