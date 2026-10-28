"""RMSNorm on bf16 rows in circuits' form (the CPU replay's RmsSumSqPart_v1, F32SumPart_v1,
RmsNormRecip_v1 and RmsNormOut_v1): units_rmsnorm, one CTA per row, and its twin.

OPS, a row x of N bf16 words and the gain w (N), C = min(128, N), nc = N / C chunks, G = min(64, nc), ng = nc / G groups
(the frozen width 16384: 128 chunks of 128, 2 groups of 64).  f32(v) is the bf16 word shifted up 16; every f32 op is IEEE
RN with no flush, every NaN result 0x7FFFFFFF; bf16() is cvt.rn.bf16.f32 (RNE, NaN 0x7FFF):
  chunk_j = (((+0 + RN(x_jC^2)) + RN(x_jC+1^2)) + ... + RN(x_jC+C-1^2))     j < nc        RmsSumSqPart_v1{C}
  group_g = (((+0 + chunk_gG) + chunk_gG+1) + ... + chunk_gG+G-1)            g < ng        F32SumPart_v1{G}
  total   = (((+0 + group_0) + group_1) + ... + group_ng-1)
  rs      = RN(1 / sqrt.rn(RN(RN(total / f32(N)) + eps)))                    eps 0x3727C5AC (1e-5)   RmsNormRecip_v1{N, ng}
  out_i   = bf16(RN(f32(w_i) * f32(bf16(RN(f32(x_i) * rs)))))                                        RmsNormOut_v1
x^2 is RN(f32(x) * f32(x)) and the division and the square root are div.rn.f32 and sqrt.rn.f32.  On the GPU a CTA of 128
threads copies its row (N <= 16384) into shared memory once, coalesced; thread j sums chunk j from there, threads g < ng
sum the groups, thread 0 makes total and rs, and every thread writes outputs, coalesced; nothing is reassociated or
contracted (-fmad=false).
"""
from __future__ import annotations

import numpy as np

import words as W

CHUNK, GROUP = 128, 64
#: the words a CTA's shared memory holds (units.cu's RQ uint4s)
ROW_WORDS = 16384


def forms(N: int) -> tuple[int, int, int, int]:
    """(C, nc, G, ng) at width N."""
    C = min(CHUNK, N)
    nc = N // C
    G = min(GROUP, nc)
    if N < 1 or N % C or nc % G:
        raise ValueError(f"RMSNorm of {N}: chunks of {C}, groups of {G}")
    return C, nc, G, nc // G


def twin(x: np.ndarray, w: np.ndarray, eps: int = W.EPS) -> dict:
    """x (R, N) and w (N,) bf16 words -> out (R, N) bf16, chunks (R, nc), groups (R, ng) and rs (R,) f32 words."""
    x = np.asarray(x, np.uint16)
    R, N = x.shape
    C, nc, G, ng = forms(N)
    xf = W.widen(x)
    chunks = W.seq_sum(W.fmul(xf, xf).reshape(R, nc, C))
    groups = W.seq_sum(chunks.reshape(R, ng, G))
    total = W.seq_sum(groups)
    rs = W.fdiv(np.uint32(W.ONE), W.fsqrt(W.fadd(W.fdiv(total, np.uint32(W.f32word(N))), np.uint32(eps))))
    h = W.bf16(W.fmul(xf, rs[:, None]))
    out = W.bf16(W.fmul(W.widen(np.asarray(w, np.uint16))[None, :], W.widen(h)))
    return {"out": out, "chunks": chunks, "groups": groups, "rs": rs}


def gpu(x, w, *, eps: int = W.EPS, out=None, units: bool = False, parts: dict | None = None):
    """units_rmsnorm on x (R, N) and w (N,) bf16 CUDA tensors -> out (R, N) bf16; with ``units`` (or ``parts``, the
    caller's tensors for some of them), (out, {"chunks", "groups", "rs"}), int32 tensors of the f32 words."""
    import torch

    import lib as L
    R, N = x.shape
    C, nc, G, ng = forms(N)
    if C % 8 or N > ROW_WORDS:
        raise ValueError(f"units_rmsnorm takes chunks of a multiple of 8 words and rows of at most {ROW_WORDS}, not {nc} of "
                         f"{C}")
    x = L.words(x, torch.bfloat16, (R, N), "rmsnorm x")
    w = L.words(w, torch.bfloat16, (N,), "rmsnorm w")
    out = L.out_or_new(out, torch.bfloat16, (R, N), x, "rmsnorm")
    L.aligned("rmsnorm", x, w, out)
    units = units or parts is not None
    if units:
        parts = L.unit_outputs(parts, {"chunks": (torch.int32, (R, nc)), "groups": (torch.int32, (R, ng)),
                                       "rs": (torch.int32, (R,))}, x, "rmsnorm")
    if R:
        p = parts or {}
        L.lib().launch("units_rmsnorm", R, L.ROW_THREADS,
                       [L.ptr(x), L.ptr(w), L.ptr(out), L.ptr(p.get("chunks")), L.ptr(p.get("groups")), L.ptr(p.get("rs")),
                        L.i32(N), L.i32(C), L.i32(nc), L.i32(G), L.i32(ng), L.u(W.f32word(N)), L.u(eps), L.u(W.ONE)])
    return (out, parts) if units else out
