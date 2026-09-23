"""Round-1 contractions of the base-field matrix M~_1 = f viewed as (C, R) (column c = f[c R : (c+1) R]) with
F_{p^6} vectors, as cupy kernels (one pass over f each, memory-bound):

* ``col_contract(f, zrow)``: u[c] = sum_row f[c, row] zrow[row]      (zrow (6, R) planes)  -> (C, 6)
* ``row_contract(f, rbar)``:  y[row] = sum_c f[c, row] rbar[c]        (rbar (C, 6))          -> (6, R) planes

Accumulation in u64 with a reduction mod p every 4 products (4 (p-1)^2 < 2^64).  Torch fallback without cupy.
"""
from __future__ import annotations

import numpy as np
import torch

from .ext import D, P

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None

_SRC = r"""
#define P 2013265921ull
typedef unsigned long long u64;

/* grid (nchunk, C), block 256: partial (C, nchunk, 6) u32 */
extern "C" __global__ void col_contract(const unsigned* __restrict__ f, const unsigned* __restrict__ z,
                                        unsigned* __restrict__ partial, int R, int chunk)
{
    const int c = blockIdx.y, ch = blockIdx.x, t = threadIdx.x;
    const unsigned* fc = f + (size_t)c * R;
    const int r0 = ch * chunk, r1 = min(R, r0 + chunk);
    u64 acc[6] = {0, 0, 0, 0, 0, 0};
    int cnt = 0;
    for (int r = r0 + t; r < r1; r += 256) {
        const u64 v = fc[r];
        #pragma unroll
        for (int d = 0; d < 6; ++d) acc[d] += v * z[(size_t)d * R + r];
        if (++cnt == 4) { cnt = 0;
            #pragma unroll
            for (int d = 0; d < 6; ++d) acc[d] %= P; }
    }
    __shared__ u64 red[6][256];
    #pragma unroll
    for (int d = 0; d < 6; ++d) red[d][t] = acc[d] % P;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            #pragma unroll
            for (int d = 0; d < 6; ++d) red[d][t] += red[d][t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        #pragma unroll
        for (int d = 0; d < 6; ++d) partial[((size_t)c * gridDim.x + ch) * 6 + d] = (unsigned)(red[d][0] % P);
    }
}

/* grid R/256, block 256: y (6, R) */
extern "C" __global__ void row_contract(const unsigned* __restrict__ f, const unsigned* __restrict__ rb,
                                        unsigned* __restrict__ y, int R, int C)
{
    extern __shared__ unsigned rbs[];          /* C x 6 */
    const int t = threadIdx.x;
    for (int i = t; i < C * 6; i += 256) rbs[i] = rb[i];
    __syncthreads();
    const int r = blockIdx.x * 256 + t;
    if (r >= R) return;
    u64 acc[6] = {0, 0, 0, 0, 0, 0};
    int cnt = 0;
    for (int c = 0; c < C; ++c) {
        const u64 v = f[(size_t)c * R + r];
        #pragma unroll
        for (int d = 0; d < 6; ++d) acc[d] += v * rbs[c * 6 + d];
        if (++cnt == 4) { cnt = 0;
            #pragma unroll
            for (int d = 0; d < 6; ++d) acc[d] %= P; }
    }
    #pragma unroll
    for (int d = 0; d < 6; ++d) y[(size_t)d * R + r] = (unsigned)(acc[d] % P);
}
"""

_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=("-std=c++17",))
    return _MOD


def _launch(name, grid, block, args, smem=0):
    import warnings
    ts = torch.cuda.current_stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cs = cp.cuda.ExternalStream(ts.cuda_stream)
    k = _mod().get_function(name)
    with cs:
        k(grid, block, args, shared_mem=smem)


def gpu_ok(t: torch.Tensor) -> bool:
    return cp is not None and t.is_cuda


def col_contract(f: torch.Tensor, zrow: torch.Tensor) -> np.ndarray:
    """f (C, R) int32 device; zrow (6, R) int32/int64 device -> (C, 6) numpy."""
    C, R = f.shape
    if not gpu_ok(f):
        f64 = f.to(torch.int64)
        z64 = zrow.to(torch.int64)
        out = torch.zeros((C, D), dtype=torch.int64)
        step = 1 << 12
        for r0 in range(0, R, step):
            blk = f64[:, r0:r0 + step]
            for d in range(D):
                out[:, d] = (out[:, d] + ((blk * z64[d, r0:r0 + step]) % P).sum(1)) % P
        return out.numpy()
    z32 = zrow.to(torch.int32).contiguous()
    f = f.contiguous()
    chunk = max(1024, min(R, 1 << 16))
    chunk = ((chunk + 255) // 256) * 256
    nchunk = (R + chunk - 1) // chunk
    partial = torch.empty((C, nchunk, D), dtype=torch.int32, device=f.device)
    _launch("col_contract", (nchunk, C), (256,),
            (np.uint64(f.data_ptr()), np.uint64(z32.data_ptr()), np.uint64(partial.data_ptr()), np.int32(R), np.int32(chunk)))
    return (partial.to(torch.int64).sum(1) % P).cpu().numpy()


def row_contract(f: torch.Tensor, rbar: np.ndarray) -> torch.Tensor:
    """f (C, R) int32 device; rbar (C, 6) numpy -> y (6, R) int32 device."""
    C, R = f.shape
    rb = torch.as_tensor(np.ascontiguousarray(rbar.astype(np.int64) % P), device=f.device)
    if not gpu_ok(f):
        f64 = f.to(torch.int64)
        y = torch.zeros((D, R), dtype=torch.int64)
        for c in range(C):
            for d in range(D):
                y[d] = (y[d] + (f64[c] * int(rb[c, d])) % P) % P
        return y.to(torch.int32)
    rb32 = rb.to(torch.int32).contiguous()
    f = f.contiguous()
    y = torch.empty((D, R), dtype=torch.int32, device=f.device)
    _launch("row_contract", ((R + 255) // 256,), (256,),
            (np.uint64(f.data_ptr()), np.uint64(rb32.data_ptr()), np.uint64(y.data_ptr()), np.int32(R), np.int32(C)),
            smem=4 * C * D)
    return y
