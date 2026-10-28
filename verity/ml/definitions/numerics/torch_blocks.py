"""numpy models of the training step's blocks, each composed of its Definitions' models (`torch_kernels`, `mla_check.bmm_rows`):
`registry.torch_attention`'s `EagerAttentionTorch_v1` and its backward on one sequence, `registry.torch_loss`'s
`LogSoftmaxRow_v1` and `LogSoftmaxNllBackwardRow_v1` on rows, `registry.torch_embedding`'s `EmbeddingBackwardTied_v1` (and
`_v2`, the same function) and `EmbeddingTorch_v1`, and `registry.torch_clip`'s `ClipGradNorm_v1` with its norms and coefficient.
Their tests hold each block to its model word for word. `MODELS` are the same models by family, as `program_walk` calls them:
the statics, then each argument's words with the call's members first (`program_walk.Model`)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verity.ml.definitions.registry import mla_check as MK

from . import torch_kernels as TK

__all__ = ["MODELS", "attention", "attention_backward", "clip", "embedding", "heads", "row", "row_backward", "tied", "tokens"]


def heads(x, H: int, D: int) -> np.ndarray:
    """(N, H·D) token-major rows as (H, N, D)."""
    return x.reshape(x.shape[0], H, D).transpose(1, 0, 2)


def tokens(x) -> np.ndarray:
    """(H, N, D) as (N, H·D)."""
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


def attention(q, k, v, cs, n, NH: int, KVH: int, D: int):
    """`EagerAttentionTorch_v1` on one sequence: (o, qe, ke, y, p) in its layouts."""
    N, R = q.shape[0], NH // KVH
    qe = heads(TK.rope_hf(q.reshape(N * NH, D), np.repeat(cs, NH, 0)).reshape(N, NH * D), NH, D)
    ke = heads(TK.rope_hf(k.reshape(N * KVH, D), np.repeat(cs, KVH, 0)).reshape(N, KVH * D), KVH, D)
    s = MK.bmm_rows(qe, np.repeat(ke, R, 0).transpose(0, 2, 1))
    y, p = (a.reshape(NH, N, N) for a in TK.eager_softmax(s.reshape(NH * N, N), np.tile(n, NH), D))
    o = MK.bmm_rows(p, np.repeat(heads(v, KVH, D), R, 0))
    return tokens(o), tokens(qe), tokens(ke), y.transpose(1, 0, 2), p.transpose(1, 0, 2)


def _unrepeat(g, KVH: int):
    """repeat_kv's backward on (NH, N, D): each KV head's R query heads summed, as (N, KVH·D)."""
    NH, N, D = g.shape
    R = NH // KVH
    if R == 1:
        return tokens(g)
    g = g.reshape(KVH, R, N, D).transpose(0, 2, 1, 3).reshape(KVH * N, R, D)
    return tokens(TK.repeat_kv_backward(g).reshape(KVH, N, D))


def attention_backward(do, qe, ke, v, y, p, cs, NH: int, KVH: int, D: int):
    """`EagerAttentionTorchBackward_v1` on one sequence: (dq, dk, dv) in projection space."""
    N, R = do.shape[0], NH // KVH
    dO, Q = heads(do, NH, D), heads(qe, NH, D)
    K, V = np.repeat(heads(ke, KVH, D), R, 0), np.repeat(heads(v, KVH, D), R, 0)
    Y, P = y.transpose(1, 0, 2), p.transpose(1, 0, 2)
    dP = MK.bmm_rows(dO, V.transpose(0, 2, 1))
    dS = TK.eager_softmax_backward(dP.reshape(-1, N), Y.reshape(-1, N), D).reshape(NH, N, N)
    dQ, dK, dV = MK.bmm_rows(dS, K), MK.bmm_rows(dS.transpose(0, 2, 1), Q), MK.bmm_rows(P.transpose(0, 2, 1), dO)
    dq = TK.rope_hf_backward(tokens(dQ).reshape(N * NH, D), np.repeat(cs, NH, 0)).reshape(N, NH * D)
    dk = TK.rope_hf_backward(_unrepeat(dK, KVH).reshape(N * KVH, D), np.repeat(cs, KVH, 0)).reshape(N, KVH * D)
    return dq, dk, _unrepeat(dV, KVH)


def row(x, t):
    """`LogSoftmaxRow_v1` on rows (R, V) of bf16 logits with labels t: (max, logsum, the target's log-probability), each (R,),
    the last from GatherBf16x's 0x7FFF for a label outside [0, V)."""
    x, t = np.atleast_2d(x), np.asarray(t, np.uint32).reshape(-1)
    V = x.shape[1]
    _, mx, ls, _, _ = TK.log_softmax_forward(x)
    xt = np.where(t < V, x[np.arange(len(x)), np.minimum(t, V - 1)], 0x7FFF)
    return mx, ls, TK.add(TK.add(TK.f32(xt), TK.neg(mx)), TK.neg(ls))


def row_backward(x, t, mx, ls, g) -> np.ndarray:
    """`LogSoftmaxNllBackwardRow_v1` on rows (R, V): the bf16 logits' gradient."""
    x = np.atleast_2d(x)
    return TK.log_softmax_nll_backward(x, t, mx, ls, g, x.shape[1])


def tied(t, g, w) -> np.ndarray:
    """`EmbeddingBackwardTied_v1` and `_v2`: the LM head's dW w (V, H) plus the embedding's gradient over the M ids t from g
    (M, H)."""
    return TK.bf16_add(w, TK.embedding_dense_backward(t, g, range(w.shape[0])))


def embedding(t, w) -> np.ndarray:
    """`EmbeddingTorch_v1`: the rows of w (V, H) at the M ids t, (M, H); 0x7FFF for an id outside [0, V)."""
    t, V = np.asarray(t, np.uint32), w.shape[0]
    return np.where((t < V)[:, None], w[np.minimum(t, V - 1)], np.uint64(0x7FFF)).astype(w.dtype)


def _f64(w) -> float:
    return float(np.array([w], np.uint64).view(np.float64)[0])


def clip(g, max_norm, sizes) -> tuple[np.ndarray, np.ndarray]:
    """`ClipGradNorm_v1`: (the total, the clipped grads' words in SIZES' order) of the grads' words g, concatenated in that
    order, at the f64 word max_norm."""
    parts = np.split(np.asarray(g, np.uint16).reshape(-1), np.cumsum(sizes)[:-1])
    total, out = TK.clip_grad_norm(parts, _f64(max_norm))
    return np.array([total], np.uint16), np.concatenate(out)


def _each(f: Callable[..., Any]) -> Callable[..., Any]:
    """A one-member model ``f(statics, *args)`` applied to each member, its outputs stacked member first."""
    def model(S: dict, *args: np.ndarray) -> Any:
        outs = [f(S, *(a[m] for a in args)) for m in range(args[0].shape[0])]
        return tuple(np.stack(o) for o in zip(*outs)) if isinstance(outs[0], tuple) else np.stack(outs)
    return model


MODELS: dict[str, Callable[..., Any]] = {
    "EagerAttentionTorch_v1": _each(lambda S, q, k, v, cs, n: attention(q, k, v, cs, n, S["NH"], S["KVH"], S["D"])),
    "EagerAttentionTorchBackward_v1": _each(lambda S, *a: attention_backward(*a, S["NH"], S["KVH"], S["D"])),
    "LogSoftmaxRow_v1": lambda S, x, t: row(x, t),
    "LogSoftmaxNllBackwardRow_v1": lambda S, x, t, mx, ls, g: row_backward(x, t, mx, ls, g),
    "EmbeddingBackwardTied_v1": _each(lambda S, t, g, w: tied(t, g, w)),
    "EmbeddingBackwardTied_v2": _each(lambda S, t, g, w: tied(t, g, w)),
    "EmbeddingTorch_v1": _each(lambda S, t, w: embedding(t, w)),
    "ClipGradNorm_v1": _each(lambda S, g, m: clip(g, m, S["SIZES"])),
    "ForeachNorm_v1": lambda S, x: TK.foreach_norm(list(x))[1],
    "LpNormChunk_v1": lambda S, x: np.array([TK.lpnorm_chunks(r, bool(S["ALIGNED"]))[0] for r in x], np.uint32),
    "LpNormCleanup_v1": lambda S, o: TK.bf16(np.array([TK.lpnorm_cleanup(r) for r in o], np.uint32)),
    "VectorNormBf16_v1": lambda S, h: np.array([TK.vector_norm(r)[1] for r in h], np.uint16),
    "ClipCoef_v1": lambda S, t, m: np.array([TK.clip_coef(np.array([a], np.uint16), _f64(b))[0] for a, b in zip(t, m)],
                                            np.uint16),
}
