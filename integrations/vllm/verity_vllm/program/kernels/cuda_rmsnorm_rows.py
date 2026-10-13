"""Numpy twins of vLLM's served non-batch-invariant CUDA BF16 RMSNorm paths."""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.registry import cuda_rmsnorm as C
from verity_vllm.program.registry import prims as P

U16, U32 = np.uint16, np.uint32


def _bf16(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=U16)


def _square(x: np.ndarray) -> np.ndarray:
    return D.mul_rn(x, x)


def _warp_sum(p: np.ndarray) -> np.ndarray:
    lanes = {i: p[:, i] for i in range(p.shape[1])}
    for off in (1, 2, 4, 8, 16):
        lanes = {
            i: (D.add_rn(v, lanes[i + off]) if i + off in lanes else v)
            for i, v in lanes.items()
            if i % (2 * off) == 0
        }
    return lanes[0]


def cub_sum(partials: np.ndarray) -> np.ndarray:
    """CUB's warp-reductions tree and thread-0 warp-aggregate scan."""
    warps = [_warp_sum(partials[:, i : i + 32]) for i in range(0, partials.shape[1], 32)]
    total = warps[0]
    for warp in warps[1:]:
        total = D.add_rn(total, warp)
    return total


def _pure_partials(x: np.ndarray, block: int) -> np.ndarray:
    rows, N = x.shape
    v = x.reshape(rows, N // C.VECTOR, C.VECTOR)
    acc = np.empty((rows, block), dtype=U32)
    for base in range(0, v.shape[1], block):
        n = min(block, v.shape[1] - base)
        for j in range(C.VECTOR):
            z = v[:, base : base + n, j]
            if base == 0 and j == 0:
                acc[:, :n] = _square(z)
            else:
                acc[:, :n] = D.fma_rn(z, z, acc[:, :n])
    return acc


def _fused_partials(r: np.ndarray, block: int) -> np.ndarray:
    rows, N = r.shape
    v = r.reshape(rows, N // C.VECTOR, C.VECTOR)
    active = min(block, v.shape[1])
    acc = np.empty((rows, active), dtype=U32)
    for base in range(0, v.shape[1], block):
        n = min(active, v.shape[1] - base)
        z = v[:, base : base + n]
        pairs = [D.add_rn(_square(z[:, :, j]), _square(z[:, :, j + 1])) for j in range(0, C.VECTOR, 2)]
        part = pairs[0]
        for pair in pairs[1:]:
            part = D.add_rn(part, pair)
        acc[:, :n] = part if base == 0 else D.add_rn(acc[:, :n], part)
    return acc


def _finish(x: np.ndarray, w: np.ndarray, partials: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    N = x.shape[1]
    var = cub_sum(partials)
    n = np.full(len(x), P.bits_of(float(N)), dtype=U32)
    e = np.full(len(x), P.bits_of(float(eps)), dtype=U32)
    scale = D.rsqrt_approx(D.add_rn(D.div_rn(var, n), e))
    xf, wf = D.bf16_to_f32_words(x), D.bf16_to_f32_words(w)
    out = D.f2fp_bf16(D.mul_rn(D.mul_rn(xf, scale[:, None]), wf))
    return out, scale


def rms_norm_rows(x, w, eps: float, block: int) -> dict[str, np.ndarray]:
    """Every word of ``RMSNormCudaVector8_v1{N, EPS, BLOCK}`` over rows."""
    x, w = _bf16(x), _bf16(w)
    if x.ndim != 2 or w.shape != (x.shape[1],):
        raise ValueError(f"x {x.shape}, w {w.shape}")
    out, scale = _finish(x, w, _pure_partials(D.bf16_to_f32_words(x), block), eps)
    return {"out": out, "scale": scale}


def fused_add_rms_norm_rows(x, res, w, eps: float, block: int) -> dict[str, np.ndarray]:
    """Every word of ``RMSNormFusedCudaVector8_v1{N, EPS, BLOCK}`` over rows."""
    x, res, w = _bf16(x), _bf16(res), _bf16(w)
    if x.ndim != 2 or res.shape != x.shape or w.shape != (x.shape[1],):
        raise ValueError(f"x {x.shape}, res {res.shape}, w {w.shape}")
    r = D.f2fp_bf16(D.add_rn(D.bf16_to_f32_words(x), D.bf16_to_f32_words(res)))
    out, scale = _finish(r, w, _fused_partials(D.bf16_to_f32_words(r), block), eps)
    return {"out": out, "residual": r, "scale": scale}

