"""Softmax over one QK^T row in circuits' form (the CPU replay's SoftmaxMax_v1,
SoftmaxRecip_v1 and SoftmaxOut_v1): units_softmax, one thread per row of 32 words (a CTA's 128 rows come in and go out
as uint4s through shared memory), and its twin.

OPS, a row x of n bf16 scores (n = t + 1, 1..32; the 1/sqrt(64) scale is folded into wq, so there is none here):
  m    = the first greatest x_j by value (+0 = -0), or 0x7FFF if any x_j is NaN                      SoftmaxMax_v1{n}
  e_j  = ex2.approx.ftz.f32(RN(RN(f32(x_j) - f32(m)) * 0x3FB8AA3B))      MUFU.EX2 (`mufu.MufuEx2Ftz`): a subnormal
         operand reads as 0 and a subnormal result is +0
  s    = (((+0 + e_0) + e_1) + ... + e_n-1)
  r    = RN(1 / s)                                                                                    SoftmaxRecip_v1{n}
  p_j  = bf16(RN(e_j * r))                                                                            SoftmaxOut_v1
Every other f32 op is IEEE RN with no flush, every NaN result 0x7FFFFFFF, the division div.rn.f32, bf16() cvt.rn.bf16.f32.
The kernel uses only x_0..x_n-1 (it loads the whole row and never reads a word past n into any op) and writes p_j = 0x0000
for n <= j < L.
"""
from __future__ import annotations

import numpy as np

import words as W

LMAX = 32


def twin(x: np.ndarray, n) -> dict:
    """x (R, L) bf16 words and n (R,) lengths (1 <= n <= L) -> p (R, L) bf16 (0 past n), m (R,) bf16 and r (R,) f32 words."""
    x = np.asarray(x, np.uint16)
    R, L = x.shape
    n = np.broadcast_to(np.asarray(n, np.int64), (R,))
    if R and (n.min() < 1 or n.max() > L):
        raise ValueError(f"softmax lengths span [{n.min()}, {n.max()}], outside [1, {L}]")
    live = np.arange(L)[None, :] < n[:, None]
    o = np.where(live, W.order16(x), np.iinfo(np.int64).min)
    m = x[np.arange(R), np.argmax(o, axis=1)]
    m = np.where((W.isnan16(x) & live).any(axis=1), np.uint16(W.BF16_NAN), m).astype(np.uint16)
    e = W.ex2(W.fmul(W.fsub(W.widen(x), W.widen(m)[:, None]), np.uint32(W.LOG2E)))
    s = np.zeros(R, np.uint32)
    for j in range(L):
        s = np.where(live[:, j], W.fadd(s, e[:, j]), s)
    r = W.fdiv(np.uint32(W.ONE), s)
    p = np.where(live, W.bf16(W.fmul(e, r[:, None])), np.uint16(0)).astype(np.uint16)
    return {"p": p, "m": m, "r": r}


def gpu(x, lens, heads: int, *, out=None, units: bool = False, parts: dict | None = None, check_lens: bool = True):
    """units_softmax on x (R, 32) bf16, R = B heads, row b heads + h of sequence b at lens[b] (int32 (B,) on the device) ->
    p (R, 32) bf16; with ``units`` (or ``parts``, the caller's tensors for some of them) (p, {"m": int16 (R,), "r": int32
    (R,)}).  ``check_lens`` (default True) refuses lens outside [1, 32] at the price of a device sync."""
    import torch

    import lib as L
    R, Lw = x.shape
    if Lw != LMAX or heads < 1 or R % heads:
        raise ValueError(f"units_softmax takes rows of {LMAX} keys in whole sequences of {heads} heads, not {tuple(x.shape)}")
    B = R // heads
    x = L.words(x, torch.bfloat16, (R, Lw), "softmax x")
    lens = L.words(lens, torch.int32, (B,), "softmax lens")
    if check_lens and B and (int(lens.min()) < 1 or int(lens.max()) > Lw):
        raise ValueError(f"softmax lens span [{int(lens.min())}, {int(lens.max())}], outside [1, {Lw}]")
    out = L.out_or_new(out, torch.bfloat16, (R, Lw), x, "softmax")
    L.aligned("softmax", x, out)
    units = units or parts is not None
    if units:
        parts = L.unit_outputs(parts, {"m": (torch.int16, (R,)), "r": (torch.int32, (R,))}, x, "softmax")
    if B > 65535:
        raise ValueError(f"units_softmax takes at most 65535 sequences a launch, not {B}")
    if R:
        p = parts or {}
        L.lib().launch("units_softmax", (-(-heads // L.ROW_THREADS), B), L.ROW_THREADS,
                       [L.ptr(x), L.ptr(lens), L.ptr(out), L.ptr(p.get("m")), L.ptr(p.get("r")), L.i32(heads), L.u(W.ONE)])
    return (out, parts) if units else out
