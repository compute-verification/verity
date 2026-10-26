"""MQA attention at one position, on the CPU through the Definitions' own numpy kernels: the stand-in for kernels' KP2-ATTN
until it is handed over (a BF16 tensor-core product computed by torch is another function).

    scores   GemmCoordinate_v2{K=d, GEMM_HOPPER} (`HopperBF16WgmmaDot16`'s 16-wide steps in ascending k from +0, then
             `F2fpBf16`): query head h of slot b against the KV head's key at each position s <= t
    softmax  SoftmaxRow_v1{N = t + 1} (circuits' plan, section 1): m the first greatest score,
             e_j = MufuEx2Ftz(RN(RN(x_j - m) * RN32(log2 e))), s their sum from +0 in ascending j, r = RN(1 / s),
             p_j = bf16(RN(e_j * r))
    pv       GemmCoordinatePad_v1{T = t + 1, K = 32}: p and the value column padded with +0 to K = 32, then
             GemmCoordinate_v2{K=32, GEMM_HOPPER}

Words are numpy u16 (bf16) and u32 (f32); everything is evaluated by `evaluation.evaluate_batch` or numpy float32 ufuncs, each
one IEEE round to nearest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU

PAD_K = 32
LOG2E = np.float32(1.4426950408889634)


def qk_template(d: int):
    return bind(G.GemmCoordinate, K=d, **G.GEMM_HOPPER)


def pv_template():
    return bind(G.GemmCoordinate, K=PAD_K, **G.GEMM_HOPPER)


def f32_of_bf16(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_of_f32(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def scores(q: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """q (B, H, d) u16, keys (B, T, d) u16 -> (B, H, T) u16."""
    b, h, d = q.shape
    t = keys.shape[1]
    x = np.broadcast_to(q[:, :, None, :], (b, h, t, d)).reshape(-1, d)
    w = np.broadcast_to(keys[:, None, :, :], (b, h, t, d)).reshape(-1, d)
    return np.asarray(evaluate_batch(qk_template(d), x, w), dtype=np.uint16).reshape(b, h, t)


@dataclass
class Softmax:
    probs: np.ndarray    # (B, H, T) u16
    m: np.ndarray        # (B, H) u16: the first greatest score's word
    s: np.ndarray        # (B, H) u32: the sum of the exponentials
    r: np.ndarray        # (B, H) u32: RN(1 / s)


def softmax(x: np.ndarray) -> Softmax:
    """x (B, H, T) u16 scores -> the probabilities and the reductions' words."""
    xf = f32_of_bf16(x)
    first = np.argmax(xf, axis=-1)
    m = np.take_along_axis(np.asarray(x, dtype=np.uint16), first[..., None], -1)[..., 0]
    y = ((xf - f32_of_bf16(m)[..., None]) * LOG2E).astype(np.float32)
    e = MU.mufu_ex2_words(y.view(np.uint32), MU.table("ex2")).view(np.float32)
    s = np.zeros(x.shape[:-1], dtype=np.float32)
    for j in range(x.shape[-1]):
        s = (s + e[..., j]).astype(np.float32)
    r = (np.float32(1) / s).astype(np.float32)
    p = bf16_of_f32((e * r[..., None]).astype(np.float32))
    return Softmax(p, m, s.view(np.uint32), r.view(np.uint32))


def pad(a: np.ndarray, k: int = PAD_K) -> np.ndarray:
    out = np.zeros((*a.shape[:-1], k), dtype=np.uint16)
    out[..., :a.shape[-1]] = a
    return out


def pv(p: np.ndarray, values: np.ndarray) -> np.ndarray:
    """p (B, H, T) u16, values (B, T, d) u16 -> (B, H, d) u16."""
    b, h, t = p.shape
    d = values.shape[-1]
    x = np.broadcast_to(pad(p)[:, :, None, :], (b, h, d, PAD_K)).reshape(-1, PAD_K)
    w = np.broadcast_to(pad(np.swapaxes(values, 1, 2))[:, None, :, :], (b, h, d, PAD_K)).reshape(-1, PAD_K)
    return np.asarray(evaluate_batch(pv_template(), x, w), dtype=np.uint16).reshape(b, h, d)
