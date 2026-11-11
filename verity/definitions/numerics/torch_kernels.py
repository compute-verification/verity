"""numpy models of the torch 2.13/2.14 CUDA kernels behind the training Definitions (`registry.torch_attention`,
`registry.torch_loss`, `registry.torch_embedding`, `registry.torch_clip`), in each kernel's own order, on rows of f32 (u32) and
bf16 (u16) words.
Every f32 result's NaN is the GPU's 0x7FFFFFFF and every rounding to bf16 is F2FP's (`derived_rows.f2fp_bf16`, NaN 0x7FFF);
nothing is flushed.

These are written from the kernels' source independently of the gates, so a test that compares them checks the gates. The two
libdevice functions are the exception: ``expf`` evaluates `NvExpf_v2` and ``logf`` `NvLogf_v3`, the Boolean Definitions the
kernels' gates call, which their own modules' tests hold to their word Definitions. `tests/test_torch_training_gpu.py` holds
the models to the CUDA kernels themselves, on an sm_120 GPU's words from smollm2-135m training steps.
"""
from __future__ import annotations

import math

import numpy as np

from ..registry.torch_train import SM120_SMS, SM120_THREADS
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
    from verity.circuits.evaluation.bits import evaluate_words

    x = np.asarray(x, U32)
    return canon(np.asarray(evaluate_words(defn, x.reshape(-1)), U32).reshape(x.shape))


def expf(x) -> np.ndarray:
    from verity.definitions.registry import boolean_moe

    return _libdevice(boolean_moe.NvExpf, x)


def logf(x) -> np.ndarray:
    from verity.definitions.registry import gumbel_boolean

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


def _scalar_t(g) -> tuple:
    """The gradient g's words as (g, its f32 words, the store of an f32 word as scalar_t): bf16 words (uint16) widen and store
    by F2FP; f32 words (uint32) are their own."""
    g = np.asarray(g)
    if g.dtype == np.uint16:
        return g, f32, bf16
    if g.dtype == U32:
        return g, (lambda x: np.asarray(x, U32)), (lambda x: np.asarray(x, U32))
    raise TypeError(f"embedding gradient words are uint16 (bf16) or uint32 (f32), not {g.dtype}")


def embedding_backward_sort(t, g, entries) -> np.ndarray:
    """``embedding_dense_backward_cuda``'s sort path, deterministic, for the given entries: rows of (M,) ids t and (M, H)
    gradients g, bf16 words or f32 words (`_scalar_t`). ``radix_sort_pairs`` of (id, position) is stable, so an entry's segment
    is its positions in order; ``compute_grad_weight`` sums each NROWS_PER_THREAD of them from +0 in f32 and
    ``sum_and_scatter`` the partials from +0, stored as g's type. Returns (len(entries), H) words of g's type."""
    t, (g, wide, store) = np.asarray(t, U32).reshape(-1), _scalar_t(g)
    order = np.argsort(t, kind="stable")
    out = np.zeros((len(entries), g.shape[1]), g.dtype)
    for k, e in enumerate(entries):
        seg = order[t[order] == e]
        tot = np.zeros(g.shape[1], U32)
        for a in range(0, len(seg), NROWS_PER_THREAD):
            p = np.zeros(g.shape[1], U32)
            for i in seg[a:a + NROWS_PER_THREAD]:
                p = add(p, wide(g[i]))
            tot = add(tot, p)
        out[k] = store(tot)
    return out


def embedding_backward_feature(t, g, entries) -> np.ndarray:
    """``embedding_backward_feature_kernel`` for the given entries: per chunk of 32 consecutive positions, the first position
    holding the entry is the leader, whose f32 gradient adds the chunk's later ones in order, then ``grad_weight +=
    scalar_t(sum)`` from +0: for bf16 words bf16(f32(grad_weight) + f32(bf16(sum))), for f32 words grad_weight + sum."""
    t, (g, wide, store) = np.asarray(t, U32).reshape(-1), _scalar_t(g)
    out = np.zeros((len(entries), g.shape[1]), g.dtype)
    for k, e in enumerate(entries):
        for a in range(0, len(t), 32):
            peers = a + np.flatnonzero(t[a:a + 32] == e)
            if len(peers):
                s = wide(g[peers[0]])
                for i in peers[1:]:
                    s = add(s, wide(g[i]))
                out[k] = store(add(wide(out[k]), wide(store(s))))
    return out


def embedding_dense_backward(t, g, entries) -> np.ndarray:
    """``embedding_dense_backward_cuda`` (``padding_idx`` -1, no ``scale_grad_by_freq``) on bf16 or f32 gradient words: the
    feature kernel up to 3072 ids, else the sort path."""
    f = embedding_backward_feature if np.size(t) <= EMBEDDING_FEATURE_MAX else embedding_backward_sort
    return f(t, g, entries)


CHUNK, NORM_THREADS, ILP = 65536, 512, 4
NORM_LANES = NORM_THREADS * ILP
F32_ONE, BF16_ONE = 0x3F800000, 0x3F80
#: f32(1e-6), clip_grad's ``total_norm + 1e-6``
EPS_1E6 = 0x358637BD


def sqrt(a) -> np.ndarray:
    """sqrt.rn.f32."""
    return _op(np.sqrt, a)


def block_reduce_sum(v) -> np.ndarray:
    """``cuda_utils::BlockReduceSum``'s value at thread 0 over the last axis of 512 threads: each warp's WarpReduceSum, then
    warp 0's over the 16 warp sums and 16 zeros."""
    v = np.asarray(v, U32)
    w = warp_reduce(v.reshape(*v.shape[:-1], -1, LANES), add)
    return warp_reduce(np.concatenate([w, np.zeros_like(w)], axis=-1), add)


def lpnorm_chunks(x, aligned: bool | None = None) -> np.ndarray:
    """`LpNormFunctor<BFloat16, L2>`'s f32 partial of each 65536-element chunk of one tensor's bf16 words x. Chunk element
    2048j + c is lane c's j-th; each lane's ``acc = fma(x, x, acc)`` from +0 in j's order is accumulator c % 4 of thread c // 4
    on the 4-vector path, taken when numel % 4 == 0 and the pointer is 8-byte aligned (``aligned``, by default numel % 4 == 0:
    the caching allocator's pointers are aligned), else accumulator c // 512 of thread c % 512; each thread's
    ((((0 + a0) + a1) + a2) + a3), then `block_reduce_sum`.
    Elements past the end are +0: every partial is +0, positive, +inf or NaN, to which adding +0 changes nothing."""
    x = np.asarray(x, np.uint16).reshape(-1)
    n = x.size
    if aligned is None:
        aligned = n % ILP == 0
    C = -(-n // CHUNK)
    xp = np.zeros(C * CHUNK, np.uint16)
    xp[:n] = x
    lanes = f32(xp).reshape(C, CHUNK // NORM_LANES, NORM_LANES)
    s = np.zeros((C, NORM_LANES), U32)
    for j in range(lanes.shape[1]):
        s = fma(lanes[:, j], lanes[:, j], s)
    by = s.reshape(C, NORM_THREADS, ILP) if aligned else s.reshape(C, ILP, NORM_THREADS).transpose(0, 2, 1)
    val = np.zeros((C, NORM_THREADS), U32)
    for i in range(ILP):
        val = add(val, by[..., i])
    return block_reduce_sum(val)


def lpnorm_cleanup(partials) -> int:
    """`lpnorm_cleanup<BFloat16, L2>`'s f32 norm of one tensor from its chunks' partials: thread t's ``0 + o[t] + o[t + 512]
    + ...``, `block_reduce_sum`, sqrt.rn. The list's other tensors' slots are +0 and change nothing."""
    p = np.asarray(partials, U32).reshape(-1)
    o = np.zeros(max(1, -(-p.size // NORM_THREADS)) * NORM_THREADS, U32)
    o[:p.size] = p
    val = np.zeros(NORM_THREADS, U32)
    for k in range(o.size // NORM_THREADS):
        val = add(val, o[k * NORM_THREADS:(k + 1) * NORM_THREADS])
    return int(sqrt(block_reduce_sum(val)).reshape(-1)[0])


def foreach_norm(xs) -> tuple[np.ndarray, np.ndarray]:
    """``torch._foreach_norm(xs, 2)`` on a list of bf16 tensors on its fast route (one dtype and device, dense): (the f32
    norms, ``dtype=torch.float32``'s, and the bf16 norms, F2FP's)."""
    f = np.array([lpnorm_cleanup(lpnorm_chunks(x)) for x in xs], U32)
    return f, bf16(f)


def _last_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def reduce_config(L: int) -> tuple[bool, int, int]:
    """Reduce.cuh's ``setReduceConfig`` for one output of L >= 1 contiguous inputs on the RTX PRO 6000 (sm_120):
    (vectorize_input, block_width, CTAs). Vectorized when L >= 128, block_width the last power of two of L // 4 (vectorized) or
    L, at most 512; one row of threads. From 256 values a thread (``ceil(L / block_width)``) the input is split across CTAs:
    the device's grid of such CTAs (188 multiprocessors of 1536 threads: 564 of 512), clamped to ``ceil(values / 256)`` ..
    ``ceil(values / 16)``. Up to L = 4,620,288 (9,024 values a thread) that is ``ceil(values / 16)``, the same on any GPU whose
    grid is at least as large; past it the multiprocessor count sets it."""
    vec = L >= 128
    bw = min(_last_pow2(L // 4 if vec else L), 512)
    vpt = -(-L // bw)
    if vpt < 256:
        return vec, bw, 1
    grid = SM120_SMS * (SM120_THREADS // bw)
    return vec, bw, min(max(grid, -(-vpt // 256)), -(-vpt // 16))


def reduce_all(x, square: bool) -> int:
    """One output of `reduce_kernel` over the L contiguous f32 words x (a bf16 tensor's widened) at an aligned pointer, before
    its projection: ``NormTwoOps``' ``acc = fma(d, d, acc)`` (``square``) or sum's ``acc + d``, every accumulator from +0, at
    `reduce_config`'s launch. Thread g = c bw + t (t the lane of CTA c) has four accumulators: vectorized, its 4-vectors
    g + bw · CTAs · r while their last element is below L, element i into accumulator i, then CTA 0's tail element
    L - L % 4 + t into accumulator 0; else its elements t + bw m into accumulator m % 4. Each thread's ((a0 + a1) + a2) + a3,
    each CTA's tree over its bw threads (`block_x_reduce`, halves added). Past one CTA, the last CTA's lane t adds
    ``0 + p[t] + p[t + bw] + ...`` of the CTAs' partials p, then its tree."""
    x = np.asarray(x, U32).reshape(-1)
    L = x.size
    vec, bw, ctas = reduce_config(L)
    T = bw * ctas
    t = np.arange(T)
    acc = np.zeros((ILP, T), U32)

    def step(a, e):
        d = x[np.minimum(e, L - 1)]
        return np.where(e < L, fma(d, d, a) if square else add(a, d), a)

    if vec:
        for r in range(-(-(L // ILP) // T)):
            v = t + T * r
            for i in range(ILP):
                acc[i] = np.where(v < L // ILP, step(acc[i], ILP * v + i), acc[i])
        acc[0] = step(acc[0], L - L % ILP + t)
    else:
        for m in range(-(-L // bw)):
            acc[m % ILP] = step(acc[m % ILP], t + bw * m)
    s = add(add(add(acc[0], acc[1]), acc[2]), acc[3]).reshape(ctas, bw)
    while s.shape[1] > 1:
        s = add(s[:, :s.shape[1] // 2], s[:, s.shape[1] // 2:])
    p = s[:, 0]
    if ctas > 1:
        w = np.zeros(bw, U32)
        for j in range(0, ctas, bw):
            k = min(bw, ctas - j)
            w[:k] = add(w[:k], p[j:j + k])
        while w.size > 1:
            w = add(w[:w.size // 2], w[w.size // 2:])
        p = w
    return int(p[0])


def vector_norm(h) -> tuple[int, int]:
    """``torch.linalg.vector_norm(h, 2)`` of L contiguous bf16 words at an aligned pointer: `reduce_kernel` with
    ``NormTwoOps<BFloat16, float, BFloat16>`` (`reduce_all`, squares), sqrt.rn: (the f32, its bf16)."""
    f = sqrt(U32(reduce_all(f32(np.asarray(h, np.uint16).reshape(-1)), square=True))).reshape(-1)
    return int(f[0]), int(bf16(f)[0])


def clip_coef(t, max_norm: float = 1.0) -> np.ndarray:
    """clip_grad's ``torch.clamp(max_norm / (t + 1e-6), max=1.0)`` on bf16 words t: a = bf16(f32(t) + f32(1e-6)),
    r = bf16(1 / f32(a)) (``reciprocal``), m = bf16(f32(r) * f32(max_norm)) (the CPU double scalar's cast to float), then the
    clamp: 1.0 where f32(m) > 1, else m (a NaN kept)."""
    a = bf16(add(f32(t), U32(EPS_1E6)))
    r = bf16(div(U32(F32_ONE), f32(a)))
    with np.errstate(over="ignore"):
        k = np.array([max_norm], np.float64).astype(np.float32).view(U32)[0]
    m = bf16(mul(f32(r), k))
    return np.where(gt(f32(m), U32(F32_ONE)), np.uint16(BF16_ONE), m).astype(np.uint16)


def clip_grad_norm(grads, max_norm: float = 1.0) -> tuple[int, list[np.ndarray]]:
    """``torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)`` on bf16 grads on one device: `foreach_norm`,
    `vector_norm` of their stack, `clip_coef`, then ``_foreach_mul_`` by it, bf16(f32(g) * f32(coef)): (the total, the
    clipped grads)."""
    _, norms = foreach_norm(grads)
    _, total = vector_norm(norms)
    c = clip_coef(np.array([total], np.uint16), max_norm)
    return total, [bf16_mul(np.asarray(g, np.uint16), c) for g in grads]
