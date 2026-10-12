"""numpy twins of the reference vocabulary (`registry.ref_prims`) -- a TESTING AID (fast evaluation over many rows),
validated against the gate-by-gate IR evaluation in `tests/program/test_ref_prims.py`; the registered bodies are the
definition, these are not."""

from __future__ import annotations

import numpy as np

from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.ref_prims import f32_erf_rn_bits, f32_exp_rn_bits, f32_tanh_rn_bits


def _np_u32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _np_f32(u: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(u, dtype=np.uint32).view(np.float32)


def np_widen(words: np.ndarray, DT: str) -> np.ndarray:
    """DT words (u16 for bf16 / u32 for f32) -> f32 values."""
    if DT == "bf16":
        return _np_f32(np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16)
    return _np_f32(np.asarray(words, dtype=np.uint32))


def np_narrow(x: np.ndarray, DT: str) -> np.ndarray:
    """f32 values -> DT words (RNE, torch NaN word)."""
    u = _np_u32(np.asarray(x, dtype=np.float32))
    if DT == "f32":
        return u
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = P.BF16_NAN_TORCH
    return r


def np_roundtrip(x: np.ndarray, DT: str) -> np.ndarray:
    """f32 values -> DT rounding -> f32 values (the `_narrow` then `_widen` pair)."""
    return np_widen(np_narrow(x, DT), DT)


def _np_fma(a, b, c):
    from verity_vllm.program.kernels.derived_rows import fma_rn
    return _np_f32(fma_rn(_np_u32(np.broadcast_to(a, np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(c)))),
                          _np_u32(np.broadcast_to(b, np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(c)))),
                          _np_u32(np.broadcast_to(c, np.broadcast_shapes(np.shape(a), np.shape(b), np.shape(c))))))


def _np_scalar_fn(fn_bits, x: np.ndarray) -> np.ndarray:
    u = _np_u32(np.asarray(x, dtype=np.float32)).ravel()
    out = np.fromiter((fn_bits(int(w)) for w in u), dtype=np.uint32, count=u.size)
    return _np_f32(out).reshape(np.shape(x))


def np_dot_ref(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`DotRef{K}`: x (..., K) f32, w (..., K) f32 -> (...) f32 — sequential fma ascending from +0."""
    x, w = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(w, np.float32))
    acc = np.zeros(x.shape[:-1], dtype=np.float32)
    for k in range(x.shape[-1]):
        acc = _np_fma(x[..., k], w[..., k], acc)
    return acc


def np_linear_ref(x_words, w_words, b_words, DT: str) -> np.ndarray:
    """`LinearRef{K,N,DT,BIAS}` over rows: x (R, K), w (N, K), b (N,) or None, DT words -> (R, N) DT words."""
    xf, wf = np_widen(x_words, DT), np_widen(w_words, DT)
    acc = np_dot_ref(xf[:, None, :], wf[None, :, :])
    if b_words is not None:
        acc = (acc + np_widen(b_words, DT)[None, :]).astype(np.float32)
    return np_narrow(acc, DT)


def _np_seq_sum(terms: np.ndarray) -> np.ndarray:
    """Sequential f32 sum over the last axis from +0."""
    acc = np.zeros(terms.shape[:-1], dtype=np.float32)
    for i in range(terms.shape[-1]):
        acc = (acc + terms[..., i]).astype(np.float32)
    return acc


def np_rmsnorm_ref(x_words, w_words, eps: float, DT: str) -> np.ndarray:
    xf, wf = np_widen(x_words, DT), np_widen(w_words, DT)
    N = xf.shape[-1]
    acc = _np_seq_sum((xf * xf).astype(np.float32))
    mean = (acc / np.float32(N)).astype(np.float32)
    rs = (np.float32(1.0) / np.sqrt((mean + np.float32(eps)).astype(np.float32))).astype(np.float32)
    h = np_roundtrip((xf * rs[..., None]).astype(np.float32), DT)
    return np_narrow((wf * h).astype(np.float32), DT)


def np_layernorm_ref(x_words, g_words, b_words, eps: float, DT: str) -> np.ndarray:
    xf, gf, bf = np_widen(x_words, DT), np_widen(g_words, DT), np_widen(b_words, DT)
    N = xf.shape[-1]
    mean = (_np_seq_sum(xf) / np.float32(N)).astype(np.float32)
    d = (xf - mean[..., None]).astype(np.float32)
    var = (_np_seq_sum((d * d).astype(np.float32)) / np.float32(N)).astype(np.float32)
    rstd = (np.float32(1.0) / np.sqrt((var + np.float32(eps)).astype(np.float32))).astype(np.float32)
    y = (((d * rstd[..., None]).astype(np.float32) * gf).astype(np.float32) + bf).astype(np.float32)
    return np_narrow(y, DT)


def np_gelu_ref(x_words, DT: str, approx: str) -> np.ndarray:
    xf = np_widen(x_words, DT)
    half_x = (xf * np.float32(0.5)).astype(np.float32)
    if approx == "none":
        f = _np_scalar_fn(f32_erf_rn_bits, (xf * np.float32(0.7071067811865476)).astype(np.float32))
    else:
        x3 = ((xf * xf).astype(np.float32) * xf).astype(np.float32)
        inner = (np.float32(0.7978845608028654) * (xf + (np.float32(0.044715) * x3).astype(np.float32)).astype(np.float32)).astype(np.float32)
        f = _np_scalar_fn(f32_tanh_rn_bits, inner)
    return np_narrow((half_x * (np.float32(1.0) + f).astype(np.float32)).astype(np.float32), DT)


def np_silu_ref(x_words, DT: str) -> np.ndarray:
    xf = np_widen(x_words, DT)
    e = _np_scalar_fn(f32_exp_rn_bits, (np.float32(0.0) - xf).astype(np.float32))
    return np_narrow((xf / (np.float32(1.0) + e).astype(np.float32)).astype(np.float32), DT)


def np_softmax_ref(x_words, DT: str) -> np.ndarray:
    """`SoftmaxRef{N,DT}` over rows (NaN-free rows; the NaN-propagating max is the registered semantics)."""
    xf = np_widen(x_words, DT)
    m = xf[..., 0]
    for i in range(1, xf.shape[-1]):
        m = np.where(xf[..., i] > m, xf[..., i], m)
    e = _np_scalar_fn(f32_exp_rn_bits, (xf - m[..., None]).astype(np.float32))
    s = _np_seq_sum(e)
    r = (np.float32(1.0) / s).astype(np.float32)
    return np_narrow((e * r[..., None]).astype(np.float32), DT)


def np_rope_ref(x_words, cos_words, sin_words, DT: str) -> np.ndarray:
    """`RoPERef{D,DT}` over rows: x (..., D), cos/sin (..., D) or (D,)."""
    xf, cf, sf = np_widen(x_words, DT), np_widen(cos_words, DT), np_widen(sin_words, DT)
    D = xf.shape[-1]
    R = D // 2
    a = np_roundtrip((xf * cf).astype(np.float32), DT)
    p = np_roundtrip((xf[..., R:] * sf[..., :R]).astype(np.float32), DT)
    q = np_roundtrip((xf[..., :R] * sf[..., R:]).astype(np.float32), DT)
    lo = (a[..., :R] - p).astype(np.float32)
    hi = (a[..., R:] + q).astype(np.float32)
    return np_narrow(np.concatenate([lo, hi], axis=-1), DT)


def np_rope_interleaved_ref(x_words, cos_words, sin_words, DT: str) -> np.ndarray:
    """`RoPEInterleavedRef{D,DT}` over rows: x (..., D), cos/sin (..., D/2) or (D/2,)."""
    xf, cf, sf = np_widen(x_words, DT), np_widen(cos_words, DT), np_widen(sin_words, DT)
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    a_e = np_roundtrip((xe * cf).astype(np.float32), DT)
    a_o = np_roundtrip((xo * cf).astype(np.float32), DT)
    p = np_roundtrip((xo * sf).astype(np.float32), DT)
    q = np_roundtrip((xe * sf).astype(np.float32), DT)
    out = np.empty(xf.shape, dtype=np.float32)
    out[..., 0::2] = (a_e - p).astype(np.float32)
    out[..., 1::2] = (a_o + q).astype(np.float32)
    return np_narrow(out, DT)


def np_relu_ref(x_words, DT: str) -> np.ndarray:
    xf = np_widen(x_words, DT)
    y = np.where(xf > 0, xf, np.float32(0.0)).astype(np.float32)
    y[xf == 0] = xf[xf == 0]  # ±0 keeps its sign (F32MaxRef(x, +0) returns x on equality)
    y[np.isnan(xf)] = np.float32(np.nan)
    return np_narrow(y, DT)


def np_eager_attention_head_ref(q_words, k_words, v_words, DT: str, scale_op: str) -> np.ndarray:
    """`EagerAttentionHeadRef{T,D,DT,SCALE_OP}` over rows: q (R, D), k/v (R, T, D) DT words -> (R, D) DT words."""
    qf, kf, vf = np_widen(q_words, DT), np_widen(k_words, DT), np_widen(v_words, DT)
    D = qf.shape[-1]
    s = np_roundtrip(np_dot_ref(qf[:, None, :], kf), DT)
    if scale_op == "mul":
        t = (s * np.float32(float(D) ** -0.5)).astype(np.float32)
    else:
        t = (s / np.float32(float(D) ** 0.5)).astype(np.float32)
    p = np_widen(np_softmax_ref(np_narrow(t, DT), DT), DT)
    o = np_dot_ref(p[:, None, :], np.swapaxes(vf, 1, 2))
    return np_narrow(o, DT)
