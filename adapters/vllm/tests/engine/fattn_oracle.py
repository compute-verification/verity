"""The int8 attention's oracle: compute accounting's LUT softmax and served output, from `benchmarks/pouw/edited_gemm`.

`attention(q, k, v, k_scales, scale)` is a numpy reimplementation, row by row in int64 and numpy float32 / Python float64 scalars,
of `fattn_ref.py`'s softmax (S, m, t = min((m - S) A >> 32, 255), e = LUT[t], P, L, s_q = 1/L, z_q = e(0)/L) and `apv_ref.py`'s f
with no dither (b = 0, so O + 64 S = P V over keys 1..pos and X(0) = V(0)), on the quantizers the plain arm puts before them (Q per
(head, token), K under the given per-KV-head scale, V per channel over the window) and fattn.cu's `fa_A` for A.

`gate_dir(...)` writes one query row's softmax in `fattn_ref.py`'s gate layout, so `fattn_ref.check` itself judges a softmax's
S, m, P, e(0), s_q and z_q (it reads key tiles of head dimension 128).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EDITED = Path(__file__).resolve().parents[4] / "benchmarks" / "pouw" / "edited_gemm"
if str(EDITED) not in sys.path:
    sys.path.insert(0, str(EDITED))

import fattn_ref  # noqa: E402

f32 = np.float32
#: fattn_ref.check's table
LUT = np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(np.int64)
LOG2E = 1.4426950408889634


def quant(x: np.ndarray, axis: int | None, scale: np.float32 | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(integers int64, float32 scales): amax over `axis` floored at 1e-30, / 127 (or the given scale), round half even, clamp."""
    s = scale if scale is not None else np.maximum(np.abs(x).max(axis=axis, keepdims=True), f32(1e-30)) / f32(127)
    return np.clip(np.rint(x / f32(s)), -127, 127).astype(np.int64), f32(s) if np.ndim(s) == 0 else s.astype(f32)


def fixed_point_a(s_q: np.float32, s_k: np.float32, scale: float) -> int:
    """fattn.cu's fa_A with sigma = s_q s_k scale: llrint(sigma log2(e) 16 2^32) in float64, left to right; saturated to its uint32_t."""
    return min(int(np.rint(float(s_q) * float(s_k) * float(scale) * LOG2E * 16.0 * 4294967296.0)), (1 << 32) - 1)


def softmax_row(S: np.ndarray, pos: int, A: int) -> dict:
    """fattn_ref.check's softmax of one query column's S over keys 0..pos."""
    m = int(S[: pos + 1].max())
    dd = (m - S[: pos + 1]).astype(np.uint64)
    t = np.minimum((dd * np.uint64(A)) >> np.uint64(32), 255).astype(np.int64)
    e = LUT[t]
    L = int(e.sum())
    P = np.zeros(S.shape[0], np.int64)
    P[1: pos + 1] = e[1:]
    return {"m": m, "e0": int(e[0]), "L": L, "P": P, "sp": f32(1) / f32(L), "sz": f32(e[0]) / f32(L)}


def served_row(P: np.ndarray, sp: np.float32, sz: np.float32, V: np.ndarray, sv: np.ndarray) -> np.ndarray:
    """apv_ref.check's want_f with b = 0: ((float32(O + 64 S) s_q) + (z_q float32(X(0)))) s_ch, each float32 op rounded."""
    body = (P @ V).astype(f32) * sp
    sink = sz * V[0].astype(f32)
    return ((body + sink) * sv).astype(f32)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_scales: torch.Tensor, scale: float) -> torch.Tensor:
    """out [n, H, d] bf16 of the window's last n rows q [n, H, d] over k, v [s, G, d] (bf16) and K scales [G] (float32)."""
    n, h, d = q.shape
    s, g_heads, _ = k.shape
    rep = h // g_heads
    out = np.zeros((n, h, d), f32)
    ks = k_scales.numpy().astype(f32)
    for g in range(g_heads):
        K, _ = quant(k[:, g].float().numpy(), None, ks[g])
        V, sv = quant(v[:, g].float().numpy(), 0)
        for j in range(rep):
            Q, sq = quant(q[:, g * rep + j].float().numpy(), 1)
            for i in range(n):
                row = softmax_row(K @ Q[i], s - n + i, fixed_point_a(sq[i, 0], ks[g], scale))
                out[i, g * rep + j] = served_row(row["P"], row["sp"], row["sz"], V, sv.reshape(-1))
    return torch.from_numpy(out).to(torch.bfloat16)


def gate_dir(d: Path, K: np.ndarray, Q: np.ndarray, pos: int, A: int, served: dict, S: np.ndarray) -> Path:
    """One query row (KV head 0, query column pos) in fattn_ref.py's gate layout: keys K [s, 128] and the query Q [128] (int8
    values), the softmax's served S [s] (int32), P (int8 over keys 0..pos, zero past), m, e(0), s_q, z_q under A."""
    s = K.shape[0]
    d.mkdir(parents=True, exist_ok=True)
    K.astype(np.int8).tofile(d / "X_0.bin")
    Q.astype(np.int8).tofile(d / f"Wh_0_{pos}.bin")
    S.astype(np.int32).tofile(d / f"S_0_{pos}.bin")
    P = np.zeros(max(s, (pos | 127) + 1), np.int8)
    P[: served["P"].shape[0]] = served["P"]
    P.tofile(d / f"P_0_{pos}.bin")
    meta = {"s": s, "H": 1, "rep": 1, "A": A, "lut": LUT.tolist(), "device_ok": True, "control_rejected": True,
            "cols": [[0, pos, float(served["sp"]), float(served["sz"]), served["m"], served["e0"]]]}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def check(d: Path) -> dict:
    return fattn_ref.check(d)
