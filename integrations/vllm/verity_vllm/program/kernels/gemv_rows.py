"""The vectorised row of `GemvBiasF32_v1{K,N,V,T}` (`verity_vllm.program.registry.gemv_bias`), registered with its self-check bindings.

One FFMA of two widened bf16 words into an FP32 accumulator is computed as a float64 product plus the accumulator, rounded once to
float32: the product of two 8-bit significands is exact in float64, and when the sum needs more than 53 bits the smaller term lies below
half a float32 ulp of the larger, which is itself on the float32 grid, so the single rounding is the FFMA's.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.kernels.kernel_registry import Declined, register

KERNEL = "gemv"
DOMAIN = "instances whose operand and result words are all finite (the reference computes the others)"


def _finite16(a: np.ndarray) -> bool:
    return bool(((np.asarray(a, np.uint16) >> 7) & 0xFF != 0xFF).all())


def gemv_bias_f32(x: np.ndarray, w: np.ndarray, b: np.ndarray, *, K: int, N: int, V: int, T: int, **_s: Any) -> np.ndarray:
    """`GemvBiasF32_v1{K,N,V,T}(x, W, b)` on one token row: (N,) bf16 words."""
    K, N, V, T = int(K), int(N), int(V), int(T)
    if not (_finite16(x) and _finite16(w) and _finite16(b)):
        raise Declined("a non-finite operand word")
    xf = D.as_f32(D.bf16_to_f32_words(np.asarray(x, np.uint16).reshape(K))).astype(np.float64)
    wf = D.as_f32(D.bf16_to_f32_words(np.asarray(w, np.uint16).reshape(N, K))).astype(np.float64)
    bf = D.as_f32(D.bf16_to_f32_words(np.asarray(b, np.uint16).reshape(N)))
    parts = np.zeros((T, N), np.float32)
    for c in range(K // V):
        t = c % T
        for k in range(c * V, c * V + V):
            parts[t] = (parts[t].astype(np.float64) + xf[k] * wf[:, k]).astype(np.float32)
    off = T // 2
    while off:
        parts[:off] = parts[:off] + parts[off:2 * off]
        off //= 2
    out = D.f2fp_bf16(D.as_u32((parts[0] + bf).astype(np.float32))).astype(np.uint16)
    if not _finite16(out):
        raise Declined("a non-finite result word")
    return out


def _sample(rng: np.random.Generator, n: int, params, **statics: Any) -> list[np.ndarray]:
    """bf16 words at a random scale per instance (2^-20 .. 2^20), one in sixteen a signed zero or a subnormal."""
    from verity.evaluation.batch import type_shape
    out = []
    for pt in params:
        shape = (n, *type_shape(pt))
        v = (rng.standard_normal(shape) * 2.0 ** rng.integers(-20, 20, (n,) + (1,) * (len(shape) - 1))).astype(np.float32)
        u = (v.view(np.uint32) >> 16).astype(np.uint16)
        edge = rng.choice(np.array([0x0000, 0x8000, 0x0001, 0x807F], np.uint16), shape)
        out.append(np.where(rng.random(shape) < 1 / 16, edge, u))
    return out


def _targets(_rng) -> list:
    from verity.ir.defs import bind
    from verity_vllm.program.registry import gemv_bias
    return [bind(gemv_bias.GemvBiasF32, K=64, N=3, V=1, T=8), bind(gemv_bias.GemvBiasF32, K=64, N=3, V=8, T=8),
            bind(gemv_bias.GemvBiasF32, K=96, N=2, V=1, T=32)]


register("GemvBiasF32_v1", gemv_bias_f32, kernel=KERNEL, domain=DOMAIN, sample=_sample, check_targets=_targets)

__all__ = ["KERNEL", "gemv_bias_f32"]
