"""Vectorised numpy twins of `RMSNormFusedCuda`'s two variance-summation trees and its epilogue: f32 is IEEE RN-even in numpy,
and MUFU.RSQ goes through the registered primitive's table (`prims_twins.rms_twin`).

* `variance_v1` — `RMSNormFusedCuda_v1`: `fused_add_rms_norm_kernel<bf16, 8>` (N/8 threads x one 8-vector, pairs summed
  sequentially).
* `variance_v2` — `RMSNormFusedCuda_v2`: the generic `fused_add_rms_norm_kernel<c10::BFloat16, 0, true>` launched under
  `VLLM_BATCH_INVARIANT=1`: `block = min(N, 1024)` threads, then `cub::BlockReduce<float, 1024>`.
* `fused_rows` — the residual add and the shared epilogue `s = rsqrtf(var / N + eps)`, `out = bf16((z * s) * w)`.

The sweep of both trees over the B0 store's rows is `rmsnorm_fused_sweep`.
"""
from __future__ import annotations

import numpy as np

from verity.ml.definitions.registry.b1 import FUSED_RMS_BLOCK



def bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_f2fp(x: np.ndarray) -> np.ndarray:
    """`__float2bfloat16` / `cvt.rn.bf16.f32`: RNE, canonical NaN 0x7FFF."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FFF
    return r


def _warp_tree(P: np.ndarray) -> np.ndarray:
    """cub `WarpReduceShfl`: lane i adds lane i+off for off = 1,2,4,8,16 when that lane exists. P: [rows, <=32]."""
    lanes = {i: P[:, i] for i in range(P.shape[1])}
    for off in (1, 2, 4, 8, 16):
        lanes = {i: ((v + lanes[i + off]).astype(np.float32) if i + off in lanes else v)
                 for i, v in lanes.items() if i % (2 * off) == 0}
    return lanes[0]


def cub_block_reduce_sum(partials: np.ndarray) -> np.ndarray:
    """`BlockReduce<float,1024>.Reduce(v, plus, blockDim.x)` over `partials[:, 0:blockDim.x]`; warp aggregates
    added sequentially from warp 0 (thread 0's `ApplyWarpAggregates`)."""
    n = partials.shape[1]
    warps = [_warp_tree(partials[:, 32 * w:32 * w + 32]) for w in range((n + 31) // 32)]
    var = warps[0]
    for w in warps[1:]:
        var = (var + w).astype(np.float32)
    return var


def variance_v1(z: np.ndarray) -> np.ndarray:
    """`fused_add_rms_norm_kernel<bf16, 8>`: N/8 threads, `((z0²+z1²) + (z2²+z3²)) + ...` sequential over 4 pairs."""
    rows, N = z.shape
    v = z.reshape(rows, N // 8, 8)
    sq = (v * v).astype(np.float32)
    pair = (sq[:, :, 0::2] + sq[:, :, 1::2]).astype(np.float32)
    acc = pair[:, :, 0]
    for i in range(1, 4):
        acc = (acc + pair[:, :, i]).astype(np.float32)
    return cub_block_reduce_sum(acc)


def variance_v2(z: np.ndarray, block: int = FUSED_RMS_BLOCK) -> np.ndarray:
    """Generic `fused_add_rms_norm_kernel<c10::BFloat16, 0, true>`: `min(N, block)` threads, thread t owns
    `t, t+block, ...`, `variance += x*x` (fma; the bf16 square is exact in f32 so `RN(acc + x*x)` is the fma)."""
    rows, N = z.shape
    nthreads = min(N, block)
    acc = np.zeros((rows, nthreads), np.float32)
    for base in range(0, N, nthreads):
        cnt = min(nthreads, N - base)
        x = z[:, base:base + cnt]
        acc[:, :cnt] = (acc[:, :cnt] + (x * x).astype(np.float32)).astype(np.float32)
    return cub_block_reduce_sum(acc)


def rsqrt_approx(x: np.ndarray) -> np.ndarray:
    """`rsqrt.approx.f32` through the registered primitive's table (same model the circuit's gate evaluates)."""
    from verity.ml.definitions.numerics.prims_twins import rms_twin
    rm, t = rms_twin()
    return np.asarray(rm.rsqrt_f32(np.ascontiguousarray(x, dtype=np.float32), t), dtype=np.float32)


def fused_rows(x: np.ndarray, res: np.ndarray, w: np.ndarray, eps: float, variance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows of (normed, res') for bf16 word arrays x, res [rows, N] and w [rows, N] (per-row weight) under the
    given variance tree. Returns (normed words, res' words, var f32, s f32)."""
    rows, N = x.shape
    rp = f32_to_bf16_f2fp((bf16_to_f32(x) + bf16_to_f32(res)).astype(np.float32))
    z = bf16_to_f32(rp)
    var = variance(z)
    mean = (var / np.float32(N)).astype(np.float32)
    s = rsqrt_approx((mean + np.float32(eps)).astype(np.float32))
    y = (z * s[:, None]).astype(np.float32)
    y = (y * bf16_to_f32(w)).astype(np.float32)
    return f32_to_bf16_f2fp(y), rp, var, s


__all__ = ["bf16_to_f32", "cub_block_reduce_sum", "f32_to_bf16_f2fp", "fused_rows", "rsqrt_approx", "variance_v1", "variance_v2"]
