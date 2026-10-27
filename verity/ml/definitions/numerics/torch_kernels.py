"""numpy models of the torch 2.13/2.14 CUDA kernels behind the training Definitions (`registry.torch_attention`,
`registry.torch_loss`, `registry.torch_embedding`), in each kernel's own order, on rows of f32 (u32) and bf16 (u16) words.
Every f32 result's NaN is the GPU's 0x7FFFFFFF and every rounding to bf16 is F2FP's (`derived_rows.f2fp_bf16`, NaN 0x7FFF);
nothing is flushed.

These are written from the kernels' source independently of the gates, so a test that compares them checks the gates. The two
libdevice functions are the exception: ``expf`` evaluates `NvExpf_v2` and ``logf`` `NvLogf_v3`, the Boolean Definitions the
kernels' gates call, which their own modules' tests hold to their word Definitions. A GPU difftest compares the models with the
CUDA kernels themselves.
"""
from __future__ import annotations

import math

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


def _libdevice(defn, x) -> np.ndarray:
    from verity.ml.circuits.evaluation.bits import evaluate_words

    x = np.asarray(x, U32)
    return canon(np.asarray(evaluate_words(defn, x.reshape(-1)), U32).reshape(x.shape))


def expf(x) -> np.ndarray:
    from verity.ml.definitions.registry import boolean_moe

    return _libdevice(boolean_moe.NvExpf, x)


def logf(x) -> np.ndarray:
    from verity.ml.definitions.registry import gumbel_boolean

    return _libdevice(gumbel_boolean.NvLogf, x)


def _isnan(u) -> np.ndarray:
    u = np.asarray(u, U32)
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)


def fmaxf(a, b) -> np.ndarray:
    """``fmaxf`` (`F32Fmaxf_v1`): the larger, +0 over -0; a NaN operand yields the other, two NaNs 0x7FFFFFFF."""
    a, b = np.broadcast_arrays(np.asarray(a, U32), np.asarray(b, U32))
    na, nb = _isnan(a), _isnan(b)
    m = np.where(gt(a, b), a, b)
    m = np.where(((a | b) & U32(0x7FFFFFFF)) == 0, a & b, m)
    m = np.where(na, b, np.where(nb, a, m))
    return np.where(na & nb, U32(GPU_NAN), m)


def max_combine(a, b) -> np.ndarray:
    """`Max`'s ``a < b ? b : a``."""
    return np.where(gt(b, a), np.asarray(b, U32), np.asarray(a, U32))


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


LOWEST = 0xFF7FFFFF
SPAN, LANES = 4096, 32


def warp_reduce(v, op) -> np.ndarray:
    """``cuda_utils::WarpReduce``'s value at lane 0 over the last axis (32): ``s_l = op(s_l, s_{l+h})`` for h = 16 .. 1."""
    v = np.asarray(v, U32)
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = op(v[..., :h], v[..., h:])
    return v[..., 0]


def _threads(x) -> np.ndarray:
    """Rows (R, V) of bf16 logits as f32 (R, V / 4096, 1024, 4): thread t's load k holds columns 4(t + 1024k) + j."""
    x = np.atleast_2d(np.asarray(x, np.uint16))
    return f32(x).reshape(x.shape[0], x.shape[1] // SPAN, 1024, 4)


def log_softmax_maxes(x) -> np.ndarray:
    """The warps' maxes (R, 32) of `cunn_SoftMaxForward<4>` with 1024 threads on rows (R, V) of bf16 logits: each thread's
    fmaxf from -FLT_MAX over its loads in order, then lane 0's WarpReduce with Max."""
    xf = _threads(x)
    m = np.full(xf.shape[::2], LOWEST, U32)
    for k in range(xf.shape[1]):
        for j in range(4):
            m = fmaxf(m, xf[:, k, :, j])
    return warp_reduce(m.reshape(-1, LANES, LANES), max_combine)


def log_softmax_forward(x):
    """`cunn_SoftMaxForward<4>` with 1024 threads on rows (R, V) of bf16 logits, V % 4096 == 0: (out (R, V) f32, max (R,),
    logsum (R,), the warps' maxes (R, 32), the warps' sums (R, 32))."""
    x = np.atleast_2d(np.asarray(x, np.uint16))
    R = x.shape[0]
    xf = _threads(x)
    wm = log_softmax_maxes(x)
    mx = warp_reduce(wm, max_combine)
    e = expf(add(xf, neg(mx)[:, None, None, None]))
    s = np.zeros((R, 1024), U32)
    for k in range(xf.shape[1]):
        for j in range(4):
            s = add(s, e[:, k, :, j])
    ws = warp_reduce(s.reshape(R, LANES, LANES), add)
    ls = logf(warp_reduce(ws, add))
    out = add(add(f32(x), neg(mx)[:, None]), neg(ls)[:, None])
    return out, mx, ls, wm, ws


def nll_threads(M: int) -> int:
    return min(max(1 << math.floor(math.log2(M // 16) + 0.5), 32), 1024)


def nll_loss(lp, t, V: int, mean: bool) -> tuple[int, int]:
    """`nll_loss_forward_reduce_cuda_kernel_2d` over M rows' target log-probabilities lp (f32) and labels t (u32, valid below
    V): (output, total_weight), each thread's ``sh -= lp`` and ``acc += 1`` over rows t, t + T, ..., then the stride tree."""
    lp, t = np.asarray(lp, U32), np.asarray(t, U32)
    M = len(lp)
    T = nll_threads(M)
    sh, acc = np.zeros(T, U32), np.zeros(T, U32)
    for i0 in range(0, M, T):
        n = min(T, M - i0)
        v = t[i0:i0 + n] < V
        sh[:n] = np.where(v, add(sh[:n], neg(lp[i0:i0 + n])), sh[:n])
        acc[:n] = np.where(v, add(acc[:n], U32(0x3F800000)), acc[:n])
    stride = T // 2
    while stride:
        sh[:stride] = add(sh[:stride], sh[stride:2 * stride])
        acc[:stride] = add(acc[:stride], acc[stride:2 * stride])
        stride //= 2
    out = div(sh[:1], acc[:1]) if mean else sh[:1]
    return int(np.asarray(out).reshape(-1)[0]), int(acc[0])


def nll_loss_backward(go: int, tw: int, mean: bool) -> int:
    go_, tw_ = np.array([go], U32), np.array([tw], U32)
    return int(np.asarray(neg(div(go_, tw_) if mean else go_)).reshape(-1)[0])


def log_softmax_nll_backward(x, t, mx, ls, g, V: int) -> np.ndarray:
    """``nll_loss``'s backward then `cunn_SoftMaxBackward`'s and ``.float()``'s on rows (R, V) of bf16 logits with labels t,
    the forward's max and logsum, and the gradient g: the row's gradient gO (g at a valid label, +0 elsewhere), its sum s from
    +0 (one nonzero term, so in any order), bf16(fma(-expf(out), s, gO)) with out the forward's epilogue recomputed."""
    x = np.atleast_2d(np.asarray(x, np.uint16))
    t = np.asarray(t, U32).reshape(-1)
    g = np.broadcast_to(np.asarray(g, U32), t.shape)
    valid = t < V
    go = np.where((np.arange(x.shape[1])[None, :] == t[:, None]) & valid[:, None], g[:, None], U32(0))
    s = np.where(valid, add(U32(0), g), U32(0))
    out = add(add(f32(x), neg(np.asarray(mx, U32))[:, None]), neg(np.asarray(ls, U32))[:, None])
    return bf16(fma(neg(expf(out)), s[:, None], go))


def chunk(x, w: int) -> np.ndarray:
    """Warp w's chunk of rows (R, V): columns 4096k + 128w + c, in that order."""
    x = np.atleast_2d(np.asarray(x))
    R, V = x.shape
    return x.reshape(R, V // SPAN, LANES, 128)[:, :, w, :].reshape(R, -1)


NROWS_PER_THREAD = 10
EMBEDDING_FEATURE_MAX = 3072


def embedding_backward_sort(t, g, entries) -> np.ndarray:
    """``embedding_dense_backward_cuda``'s sort path, deterministic, for the given entries: rows of (M,) ids t and (M, H) bf16
    gradients g. ``radix_sort_pairs`` of (id, position) is stable, so an entry's segment is its positions in order;
    ``compute_grad_weight`` sums each NROWS_PER_THREAD of them from +0 and ``sum_and_scatter`` the partials from +0, stored as
    bf16. Returns (len(entries), H) bf16."""
    t, g = np.asarray(t, U32).reshape(-1), np.asarray(g, np.uint16)
    order = np.argsort(t, kind="stable")
    out = np.zeros((len(entries), g.shape[1]), np.uint16)
    for k, e in enumerate(entries):
        seg = order[t[order] == e]
        tot = np.zeros(g.shape[1], U32)
        for a in range(0, len(seg), NROWS_PER_THREAD):
            p = np.zeros(g.shape[1], U32)
            for i in seg[a:a + NROWS_PER_THREAD]:
                p = add(p, f32(g[i]))
            tot = add(tot, p)
        out[k] = bf16(tot)
    return out


def embedding_backward_feature(t, g, entries) -> np.ndarray:
    """``embedding_backward_feature_kernel`` for the given entries: per chunk of 32 consecutive positions, the first position
    holding the entry is the leader, whose f32 gradient adds the chunk's later ones in order; ``grad_weight += bf16(sum)`` is
    bf16(f32(grad_weight) + f32(bf16(sum))), from +0."""
    t, g = np.asarray(t, U32).reshape(-1), np.asarray(g, np.uint16)
    out = np.zeros((len(entries), g.shape[1]), np.uint16)
    for k, e in enumerate(entries):
        for a in range(0, len(t), 32):
            peers = a + np.flatnonzero(t[a:a + 32] == e)
            if len(peers):
                s = f32(g[peers[0]])
                for i in peers[1:]:
                    s = add(s, f32(g[i]))
                out[k] = bf16(add(f32(out[k]), f32(bf16(s))))
    return out


def embedding_dense_backward(t, g, entries) -> np.ndarray:
    """``embedding_dense_backward_cuda`` (``padding_idx`` -1, no ``scale_grad_by_freq``): the feature kernel up to 3072 ids,
    else the sort path."""
    f = embedding_backward_feature if np.size(t) <= EMBEDDING_FEATURE_MAX else embedding_backward_sort
    return f(t, g, entries)
