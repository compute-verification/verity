"""Argmax over the head's logits (greedy decoding) in circuits' form (the CPU replay's ArgmaxChunkBest_v1,
ArgmaxChunkIdx_v1 and ArgmaxCombine_v1, whose token is `ops.rows.Argmax_v1{V}`'s scan: fixture.py holds every
pool token to `rows.argmax`): units_argmax, one thread per row, and its twin.

OPS, a row x of V bf16 logits, C = min(128, V), nch = ceil(V / C) chunks (the frozen vocabulary 256: 2 chunks of 128):
  chunk c: (best_c, idx_c) from (0xFF80, 0); for i ascending, x_cC+i is taken iff it is not NaN and above best_c by
           value (+0 = -0; strictly, so the first greatest stays)                ArgmaxChunkBest_v1{C}, ArgmaxChunkIdx_v1{C}
  combine: (run, k) from (x_0, 0); for c ascending, chunk c is taken, k = cC + idx_c, iff neither best_c nor run is NaN
           and best_c is above run by value                                       ArgmaxCombine_v1{V, C}
  token  = k
The tie rule: the first greatest wins, NaN is never taken, a row of NaNs and -infs gives 0, and a NaN x_0 gives 0.
Integers only.
"""
from __future__ import annotations

import numpy as np

import words as W

CHUNK = 128


def twin(x: np.ndarray) -> dict:
    """x (R, V) bf16 words -> token (R,), best (R, nch) and idx (R, nch) u16 words."""
    x = np.asarray(x, np.uint16)
    R, V = x.shape
    C = min(CHUNK, V)
    nch = -(-V // C)
    o = np.where(W.isnan16(x), np.iinfo(np.int64).min, W.order16(x))
    floor = int(W.order16(np.uint16(W.BF16_NEG_INF)))
    best, idx = np.empty((R, nch), np.uint16), np.empty((R, nch), np.uint16)
    for c in range(nch):
        oc = o[:, c * C:(c + 1) * C]
        j = np.argmax(oc, axis=1)
        some = oc[np.arange(R), j] > floor
        best[:, c] = np.where(some, x[np.arange(R), c * C + j], np.uint16(W.BF16_NEG_INF))
        idx[:, c] = np.where(some, j, 0)
    run, k = x[:, 0].copy(), np.zeros(R, np.int64)
    for c in range(nch):
        b = best[:, c]
        gt = ~W.isnan16(b) & ~W.isnan16(run) & (W.order16(b) > W.order16(run))
        k = np.where(gt, c * C + idx[:, c].astype(np.int64), k)
        run = np.where(gt, b, run)
    return {"token": (k & 0xFFFF).astype(np.uint16), "best": best, "idx": idx}


def gpu(x, *, out=None, units: bool = False, parts: dict | None = None):
    """units_argmax on x (R, V) bf16 -> token (R,) int16; with ``units`` (or ``parts``, the caller's tensors for some of
    them) (token, {"best", "idx"}), int16 (R, nch)."""
    import torch

    import lib as L
    R, V = x.shape
    if V < 1 or V > 0x10000:
        raise ValueError(f"units_argmax takes 1..65536 logits a row, not {V}")
    C = min(CHUNK, V)
    nch = -(-V // C)
    x = L.words(x, torch.bfloat16, (R, V), "argmax x")
    out = L.out_or_new(out, torch.int16, (R,), x, "argmax")
    units = units or parts is not None
    if units:
        parts = L.unit_outputs(parts, {"best": (torch.int16, (R, nch)), "idx": (torch.int16, (R, nch))}, x, "argmax")
    if R:
        p = parts or {}
        L.lib().launch("units_argmax", -(-R // 128), 128,
                       [L.ptr(x), L.ptr(p.get("best")), L.ptr(p.get("idx")), L.ptr(out), L.i32(R), L.i32(V), L.i32(C),
                        L.i32(nch)])
    return (out, parts) if units else out
