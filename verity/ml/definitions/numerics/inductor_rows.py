"""Numpy twins of `registry.inductor`'s Definitions, vectorised over rows: the RMSNorm forms (`residual_rows`, `norm_rows`),
`InductorSiluMul_v1` (`silu_mul_rows`), `InductorRoPE_v1` (`derived_rows.rope_rows` with the `f2fp_bf16` store),
`InductorQKNormRoPE_v1` (`qk_norm_rope_rows`: `rstd_rows` and `derived_rows.rope_f32_rows`) and FA3's split
decode (`fa3_head_split_rows`, `fa3_attention_rows`, over `derived_rows.attn_block` and `sparse_mla_split_twins`' split end and
combine).  The Match's fast path, checked word for word against the Definitions in `tests/program/test_inductor.py`.  A
row whose dot reads a non-finite operand is flagged and never claimed exact (`attn_block`'s ``nonfinite``)."""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import sparse_mla_split_twins as SMT
from verity.ml.definitions.registry import inductor as I
from verity.ml.definitions.registry import sparse_mla_split as SMS
from verity.ml.definitions.registry.prims import bits_of

U32 = np.uint32
SIGN = U32(0x80000000)


def residual_rows(x, *rs) -> np.ndarray:
    """The f32 row r each RMSNorm form normalises: x itself, x + res, or x + (r1 + r2) (bf16 words in, f32 words out)."""
    xf, *rf = (DR.bf16_to_f32_words(a) for a in (x, *rs))
    if len(rf) == 2:
        rf = [DR.add_rn(rf[0], rf[1])]
    return DR.add_rn(xf, rf[0]) if rf else xf


def rstd_rows(r, *, N: int, EPS: float, XBLOCK: int, R0_BLOCK: int, NUM_WARPS: int) -> np.ndarray:
    """`registry.inductor._rstd` over rows: r (R, N) f32 words -> (R,) f32 words."""
    tpr, chunks, vec = I.norm_layout(N, EPS, XBLOCK, R0_BLOCK, NUM_WARPS)
    r = np.asarray(r, dtype=U32)
    R = r.shape[0]
    acc = np.zeros((R, R0_BLOCK), dtype=U32)
    for k in range(chunks):
        lo, hi = k * R0_BLOCK, min(N, (k + 1) * R0_BLOCK)
        acc[:, :hi - lo] = DR.fma_rn(r[:, lo:hi], r[:, lo:hi], acc[:, :hi - lo])
    cols = acc.reshape(R, tpr, vec)
    s = cols[:, :, 0]
    for i in range(1, vec):
        s = DR.add_rn(s, cols[:, :, i])
    lanes = min(tpr, I.WARP)
    warps = SMT._lane0(DR.add_rn, s.reshape(R * (tpr // lanes), lanes)).reshape(R, tpr // lanes)
    total = SMT._lane0(DR.add_rn, warps)
    nf = np.full(R, bits_of(np.float32(N)), dtype=U32)
    mean = DR.mul_rn(DR.div_full_scale_a(total, nf), DR.div_full_rcp(nf))
    return DR.rsqrt_approx(DR.add_rn(mean, np.full(R, bits_of(np.float32(EPS)), dtype=U32)))


def norm_rows(r, w, **S) -> np.ndarray:
    """`registry.inductor._norm` over rows: r (R, N) f32 words, w (N,) bf16 words -> (R, N) bf16 words."""
    r = np.asarray(r, dtype=U32)
    return DR.f2fp_bf16(DR.mul_rn(DR.mul_rn(r, rstd_rows(r, **S)[:, None]), DR.bf16_to_f32_words(w)))


def qk_norm_rope_rows(x, w, cs, *, NHEADS: int, D: int, **S) -> np.ndarray:
    """`InductorQKNormRoPE_v1` over rows: x (R, NHEADS, D), w (D,), cs (R, D) bf16 words -> (R, NHEADS, D) bf16 words."""
    x = np.asarray(x, dtype=np.uint16)
    xf = DR.bf16_to_f32_words(x.reshape(-1, D))
    y = DR.mul_rn(DR.mul_rn(xf, rstd_rows(xf, N=D, **S)[:, None]), DR.bf16_to_f32_words(w))
    return DR.rope_f32_rows(y.reshape(x.shape[0], NHEADS, D), cs, store=DR.f2fp_bf16)


def _sat(u) -> np.ndarray:
    """`F32Sat_v1`."""
    u = np.asarray(u, dtype=U32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)
    return np.where(nan | ((u >> 31) != 0), U32(0), np.minimum(u, U32(0x3F800000)))


def silu_mul_rows(gu) -> np.ndarray:
    """`InductorSiluMul_v1{I}` over rows: gu (..., 2I) bf16 words -> (..., I) bf16 words."""
    gu = np.asarray(gu, dtype=np.uint16)
    half = gu.shape[-1] // 2
    g, u = DR.bf16_to_f32_words(gu[..., :half]), DR.bf16_to_f32_words(gu[..., half:])
    n = DR._f32_binop(np.zeros_like(g), g, np.subtract)
    c = lambda w: np.full(g.shape, w, dtype=U32)
    s = _sat(DR.fma_ftz(n, c(0x3BBB989D), c(0x3F000000)))
    # fma.rm(s, 252, 12582913) with s in [0, 1]: the exact sum lies in [2^23, 2^24), whose f32 spacing is 1, and 12582913 is an
    # integer, so rounding toward -inf is 12582913 + floor(s * 252), s * 252 exact in f64
    jb = DR.as_u32((12582913.0 + np.floor(DR.as_f32(s).astype(np.float64) * 252.0)).astype(np.float32))
    j = DR.add_rn(jb, c(bits_of(np.float32(-12583039.0))))
    f = DR.fma_ftz(n, c(0x3FB8AA3B), j ^ SIGN)
    f = DR.fma_ftz(n, c(0x32A57060), f)
    d = DR.fma_rn(DR.mufu_ex2_ftz(f), jb << U32(23), c(0x3F800000))
    q = DR.mul_rn(DR.div_full_scale_a(g, d), DR.div_full_rcp(d))
    return DR.f2fp_bf16(DR.mul_rn(q, u))


def rope_rows(x, cs) -> np.ndarray:
    """`InductorRoPE_v1{NHEADS, D}` over rows: x (..., NHEADS, D), cs (..., D) bf16 words -> (..., NHEADS, D) bf16 words."""
    return DR.rope_rows(x, cs, store=DR.f2fp_bf16)


def fa3_head_split_rows(q, k, v, *, T: int, NS: int, BN: int, masked_from: int, dot16=DR.tc_dot16_hopper,
                        inv=DR.fa3_inv_sum) -> tuple[np.ndarray, np.ndarray]:
    """`Fa3AttentionHeadSplit_v1{T, NS, D, BN, DOT, INV, MASKED_FROM}` over rows: q (R, D), k, v (R, T, D) bf16 words -> (out (R, D)
    bf16 words, nonfinite (R,)).  NS = 1 is `AttentionHead_v4`."""
    SMS.check_split_statics(T, NS)
    q, k, v = (np.asarray(a, dtype=np.uint16) for a in (q, k, v))
    if NS == 1:
        st = DR.attention_head_states(q, k, v, BN, dot16=dot16, inv=inv, masked_from=masked_from)
        return st["out"], st["nonfinite"]
    R, D = q.shape
    nf = np.zeros(R, dtype=bool)
    ends = []
    for b0, b1 in SMS.split_blocks(T, NS, BN):
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = (np.zeros((R, BN, D), dtype=np.uint16) for _ in range(2))
            kb[:, :hi - lo], vb[:, :hi - lo] = k[:, lo:hi], v[:, lo:hi]
            first = b == b1 - 1
            m, l, o, f = DR.attn_block(q, kb, vb, m, l, o, D=D, NVIS=hi - lo, FIRST=first, dot16=dot16,
                                       check=first or b >= masked_from)
            nf |= f
        ends.append(SMT.split_end_rows(m, l, o, scale=D ** -0.5, inv=inv))
    out = SMT.combine_rows(np.stack([e[1] for e in ends], axis=1), np.stack([e[0] for e in ends], axis=1), inv=inv)
    return out, nf


def fa3_attention_rows(q, kc, vc, *, NS: int, BN: int, masked_from: int, dot_prim: str = "HopperBF16WgmmaDot16_v1",
                       inv_prim: str = "Fa3InvSum_v1") -> tuple[np.ndarray, np.ndarray]:
    """`Fa3AttentionSplit_v1` for one query token: q (NH, D), kc / vc (T, KVH, D) bf16 words -> (out (NH, D) bf16 words,
    nonfinite (NH,)); head h reads KV head h // (NH / KVH)."""
    q, kc, vc = (np.asarray(a, dtype=np.uint16) for a in (q, kc, vc))
    NH, KVH = q.shape[0], kc.shape[1]
    kvh = np.arange(NH) // (NH // KVH)
    k, v = (np.ascontiguousarray(np.transpose(a, (1, 0, 2))[kvh]) for a in (kc, vc))
    return fa3_head_split_rows(q, k, v, T=kc.shape[0], NS=NS, BN=BN, masked_from=masked_from,
                               dot16=DR.TC_DOT16_BY_PRIM[dot_prim], inv=DR.INV_SUM_BY_PRIM[inv_prim])
