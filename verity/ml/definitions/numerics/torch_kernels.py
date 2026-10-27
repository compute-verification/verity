"""numpy models of the torch 2.13/2.14 CUDA kernels behind the training Definitions (`registry.torch_attention`), in each
kernel's own order, on rows of f32 (u32) and bf16 (u16) words. Every f32 result's NaN is the GPU's 0x7FFFFFFF and every
rounding to bf16 is F2FP's (`derived_rows.f2fp_bf16`, NaN 0x7FFF); nothing is flushed. ``expf`` is `NvExpf_v1`, libdevice's
sequence with the sm_89 MUFU.EX2 table, as the Definitions read it until the sm_120 table is pinned.

These are written from the kernels' source independently of the gates, so a test that compares them checks the gates. A GPU
difftest compares them with the CUDA kernels themselves.
"""
from __future__ import annotations

import numpy as np

from .derived_rows import as_f32, as_u32, bf16_to_f32_words, f2fp_bf16, fma_rn

U32 = np.uint32
GPU_NAN = 0x7FFFFFFF
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
BF16_MIN = 0xFF7F
VT0 = 4


def canon(u) -> np.ndarray:
    u = np.asarray(u, U32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    return np.where(nan, U32(GPU_NAN), u)


def _op(f, *xs) -> np.ndarray:
    with np.errstate(all="ignore"):
        return canon(as_u32(f(*(as_f32(np.asarray(x, U32)) for x in np.broadcast_arrays(*xs)))))


def add(a, b):
    return _op(np.add, a, b)


def mul(a, b):
    return _op(np.multiply, a, b)


def div(a, b):
    return _op(np.divide, a, b)


def fma(a, b, c):
    a, b, c = np.broadcast_arrays(np.asarray(a, U32), np.asarray(b, U32), np.asarray(c, U32))
    return canon(fma_rn(a, b, c))


def neg(a):
    return np.asarray(a, U32) ^ U32(0x80000000)


def gt(a, b) -> np.ndarray:
    with np.errstate(all="ignore"):
        return as_f32(np.asarray(a, U32)) > as_f32(np.asarray(b, U32))


def f32(h) -> np.ndarray:
    return bf16_to_f32_words(h)


def bf16(u) -> np.ndarray:
    return f2fp_bf16(np.asarray(u, U32))


def bf16_mul(a, b):
    return bf16(mul(f32(a), f32(b)))


def bf16_add(a, b):
    return bf16(add(f32(a), f32(b)))


def expf(x) -> np.ndarray:
    from verity.ml.circuits import evaluation as E
    from verity.ml.definitions.registry import moe as M

    x = np.asarray(x, U32)
    return canon(np.asarray(E.evaluate_batch(M.NvExpf, x.reshape(-1)), U32).reshape(x.shape))


def _lanes(N: int) -> tuple[int, int]:
    p = 1
    while p < N:
        p *= 2
    return min(p, 32), p // min(p, 32)


def _by_lane(x: np.ndarray, pad: int) -> np.ndarray:
    """(R, IT, WS): element it of lane l is x_{l + it*WS}, ``pad`` past N."""
    R, N = x.shape
    ws, it = _lanes(N)
    out = np.full((R, ws * it), pad, U32)
    out[:, :N] = x
    return out.reshape(R, it, ws)


def _warp_sum(v: np.ndarray) -> np.ndarray:
    """(R, IT, WS) -> (R,): each lane's sum from +0 in order, then lane 0's butterfly."""
    s = np.zeros(v.shape[::2], U32)
    for k in range(v.shape[1]):
        s = add(s, v[:, k, :])
    while s.shape[1] > 1:
        h = s.shape[1] // 2
        s = add(s[:, :h], s[:, h:])
    return s[:, 0]


def softmax_forward(x) -> np.ndarray:
    """`softmax_warp_forward<float, float, float>` on rows (R, N) of f32 words."""
    x = np.atleast_2d(np.asarray(x, U32))
    R, N = x.shape
    el = _by_lane(x, NEG_INF)
    ws = el.shape[2]
    m = el[:, 0, :].copy()
    for k in range(1, el.shape[1]):
        m = np.where(gt(m, el[:, k, :]), m, el[:, k, :])
    off = ws // 2
    while off:
        b = m[:, np.arange(ws) ^ off]
        m = np.where(gt(b, m), b, m)
        off //= 2
    e = expf(add(el, neg(m[:, None, :])))
    total = _warp_sum(e)
    q = div(e.reshape(R, -1)[:, :N], total[:, None])
    return np.where(((total & U32(0x7FFFFFFF)) == 0)[:, None], U32(QNAN), q)


def softmax_backward(g, y) -> np.ndarray:
    """``tmp = grad * output``, then `softmax_warp_backward<float, float, float>`, on rows (R, N) of f32 words."""
    g, y = np.atleast_2d(np.asarray(g, U32)), np.atleast_2d(np.asarray(y, U32))
    tmp = mul(g, y)
    total = _warp_sum(_by_lane(tmp, 0))
    return fma(neg(y), total[:, None], tmp)


def scale_bits(D: int) -> int:
    return int(np.array([D ** -0.5], np.float32).view(U32)[0])


def eager_softmax(s, n, D: int) -> tuple[np.ndarray, np.ndarray]:
    """`eager_attention_forward` between its bmm's on rows (R, N) of bf16 scores, row r seeing its first n[r] keys: (the
    saved f32 softmax, its bf16 copy)."""
    s = np.atleast_2d(np.asarray(s, np.uint16))
    a = bf16(mul(f32(s), U32(scale_bits(D))))
    keys = np.arange(s.shape[1])[None, :] < np.asarray(n).reshape(-1, 1)
    x = f32(bf16_add(a, np.where(keys, 0, BF16_MIN).astype(np.uint16)))
    y = softmax_forward(x)
    return y, bf16(y)


def eager_softmax_backward(dp, y, D: int) -> np.ndarray:
    gi = softmax_backward(f32(dp), y)
    return bf16(mul(f32(bf16(gi)), U32(scale_bits(D))))


def _neg16(h):
    return np.asarray(h, np.uint16) ^ np.uint16(0x8000)


def _z16(h):
    h = np.asarray(h, np.uint16)
    return np.where(h == 0x8000, np.uint16(0), h)


def rope_hf(x, cs) -> np.ndarray:
    """HF ``q * cos + rotate_half(q) * sin`` on rows (R, D) of bf16 words, cs (R, D) = [cos[0:R] | sin[0:R]]."""
    x, cs = np.atleast_2d(np.asarray(x, np.uint16)), np.atleast_2d(np.asarray(cs, np.uint16))
    h = x.shape[1] // 2
    cos = np.concatenate([cs[:, :h], cs[:, :h]], axis=1)
    sin = np.concatenate([cs[:, h:], cs[:, h:]], axis=1)
    rot = np.concatenate([_neg16(x[:, h:]), x[:, :h]], axis=1)
    return bf16_add(bf16_mul(x, cos), bf16_mul(rot, sin))


def rope_hf_backward(g, cs) -> np.ndarray:
    """The input's gradient of `rope_hf` from the output's g: A = g*cos, B = g*sin, then A + [B[R:] | -B[:R]], -0 as +0."""
    g, cs = np.atleast_2d(np.asarray(g, np.uint16)), np.atleast_2d(np.asarray(cs, np.uint16))
    h = g.shape[1] // 2
    cos = np.concatenate([cs[:, :h], cs[:, :h]], axis=1)
    sin = np.concatenate([cs[:, h:], cs[:, h:]], axis=1)
    A, B = bf16_mul(g, cos), bf16_mul(g, sin)
    return _z16(bf16_add(A, np.concatenate([B[:, h:], _neg16(B[:, :h])], axis=1)))


def repeat_kv_backward(g) -> np.ndarray:
    """Reduce.cuh's per-thread fold over axis 1 of (R, NREP, N) bf16 words: VT0 f32 accumulators from +0, input i into
    accumulator i mod VT0, then combined in order, then F2FP."""
    g = np.asarray(g, np.uint16)
    acc = np.zeros((VT0, g.shape[0], g.shape[2]), U32)
    idx, n = 0, g.shape[1]
    while idx + VT0 - 1 < n:
        for i in range(VT0):
            acc[i] = add(acc[i], f32(g[:, idx + i]))
        idx += VT0
    for i in range(VT0):
        if idx >= n:
            break
        acc[i] = add(acc[i], f32(g[:, idx]))
        idx += 1
    out = acc[0]
    for i in range(1, VT0):
        out = add(out, acc[i])
    return bf16(out)
