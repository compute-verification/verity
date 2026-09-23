"""Fused CUDA rounds for the zero-check sumcheck (``sumcheck.py``): two cupy RawKernels per round.

``sc_msg``: for every pair ``(x', x' + half)`` of the three tables, the products ``A0 B0 - C0``, ``A1 B1 - C1``,
``(A1 - A0)(B1 - B0)``, the eq weight ``w = w_rows[k] * w_cols[c]`` and the three weighted accumulations (block-reduced,
one atomic per block and coordinate) -> ``q(0), q(1), q(inf)``.  ``sc_fold``: ``T0 + r (T1 - T0)`` for the three tables,
IN PLACE (the output occupies the first half of every plane).  The message is computed before the challenge is drawn,
as the interactive protocol requires.

Extension elements are 6 int32 planes ``(6, R, Cc)`` with an explicit plane stride; round 1 reads base tables ``(R, Cc)``
int32 and its fold writes fresh ext planes.  Plain ``% p`` on uint64 (the compiler's constant-modulus sequence); no
Montgomery form yet -- the obvious next 2-3x.
"""
from __future__ import annotations

import torch

from backends.direct.ligero.compile import P
from backends.direct.ligero.v2.fc import D

_SRC = r"""
#define P 2013265921ULL
#define NR 31ULL
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u32 mulmod(u32 a, u32 b) { return (u32)(((u64)a * (u64)b) % P); }
__device__ __forceinline__ u32 addmod(u32 a, u32 b) { u32 s = a + b; return s >= (u32)P ? s - (u32)P : s; }
__device__ __forceinline__ u32 submod(u32 a, u32 b) { return a >= b ? a - b : a + (u32)P - b; }

__device__ __forceinline__ void ext_mul(const u32* a, const u32* b, u32* out) {
    u64 t[11];
    #pragma unroll
    for (int i = 0; i < 11; i++) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < 6; j++) t[i + j] += ((u64)a[i] * (u64)b[j]) % P;
    }
    #pragma unroll
    for (int c = 0; c < 6; c++) {
        u64 v = t[c];
        if (c + 6 < 11) v += (t[c + 6] % P) * NR;
        out[c] = (u32)(v % P);
    }
}

__device__ __forceinline__ void load6(const int* __restrict__ T, long long ps, long long o, u32* v) {
#if BASE_IN
    v[0] = (u32)T[o];
    #pragma unroll
    for (int p = 1; p < 6; p++) v[p] = 0;
#else
    #pragma unroll
    for (int p = 0; p < 6; p++) v[p] = (u32)T[p * ps + o];
#endif
}

extern "C" __global__ void sc_msg(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    const u32* __restrict__ wr, const u32* __restrict__ wc,
    u64* acc, long long hR, long long Cc, long long ps)
{
    __shared__ u64 sh[18][256];
    const long long n = hR * Cc;
    u64 s0[6], s1[6], si[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) { s0[p] = 0; s1[p] = 0; si[p] = 0; }
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o0 = k * Cc + c, o1 = (k + hR) * Cc + c;
        u32 a0[6], a1[6], b0[6], b1[6], c0[6], c1[6];
        load6(A, ps, o0, a0); load6(A, ps, o1, a1);
        load6(B, ps, o0, b0); load6(B, ps, o1, b1);
        load6(C, ps, o0, c0); load6(C, ps, o1, c1);
        u32 wrow[6], wcol[6], w[6];
        #pragma unroll
        for (int p = 0; p < 6; p++) { wrow[p] = wr[p * hR + k]; wcol[p] = wc[p * Cc + c]; }
        ext_mul(wrow, wcol, w);
        u32 t0[6], t1[6], ti[6];
#if BASE_IN
        const u32 p0 = submod(mulmod(a0[0], b0[0]), c0[0]);
        const u32 p1 = submod(mulmod(a1[0], b1[0]), c1[0]);
        const u32 pi = mulmod(submod(a1[0], a0[0]), submod(b1[0], b0[0]));
        #pragma unroll
        for (int p = 0; p < 6; p++) { t0[p] = mulmod(w[p], p0); t1[p] = mulmod(w[p], p1); ti[p] = mulmod(w[p], pi); }
#else
        u32 p0[6], p1[6], pi[6], da[6], db[6];
        ext_mul(a0, b0, p0); ext_mul(a1, b1, p1);
        #pragma unroll
        for (int p = 0; p < 6; p++) { p0[p] = submod(p0[p], c0[p]); p1[p] = submod(p1[p], c1[p]); da[p] = submod(a1[p], a0[p]); db[p] = submod(b1[p], b0[p]); }
        ext_mul(da, db, pi);
        ext_mul(w, p0, t0); ext_mul(w, p1, t1); ext_mul(w, pi, ti);
#endif
        #pragma unroll
        for (int p = 0; p < 6; p++) { s0[p] += t0[p]; s1[p] += t1[p]; si[p] += ti[p]; }
    }
    #pragma unroll
    for (int p = 0; p < 6; p++) { sh[p][threadIdx.x] = s0[p] % P; sh[6 + p][threadIdx.x] = s1[p] % P; sh[12 + p][threadIdx.x] = si[p] % P; }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int q = 0; q < 18; q++) sh[q][threadIdx.x] += sh[q][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 18) atomicAdd(&acc[threadIdx.x], sh[threadIdx.x][0] % P);
}

extern "C" __global__ void sc_fold(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ C,
    int* Ao, int* Bo, int* Co, const u32* __restrict__ r6,
    long long hR, long long Cc, long long ps_in, long long ps_out)
{
    const long long n = hR * Cc;
    u32 r[6];
    #pragma unroll
    for (int p = 0; p < 6; p++) r[p] = r6[p];
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
        const long long k = idx / Cc, c = idx - k * Cc;
        const long long o0 = k * Cc + c, o1 = (k + hR) * Cc + c;
        u32 a0[6], a1[6], b0[6], b1[6], c0[6], c1[6], d[6], fa[6], fb[6], fc[6];
        load6(A, ps_in, o0, a0); load6(A, ps_in, o1, a1);
        load6(B, ps_in, o0, b0); load6(B, ps_in, o1, b1);
        load6(C, ps_in, o0, c0); load6(C, ps_in, o1, c1);
#if BASE_IN
        const u32 da = submod(a1[0], a0[0]), db = submod(b1[0], b0[0]), dc = submod(c1[0], c0[0]);
        #pragma unroll
        for (int p = 0; p < 6; p++) { fa[p] = mulmod(r[p], da); fb[p] = mulmod(r[p], db); fc[p] = mulmod(r[p], dc); }
#else
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(a1[p], a0[p]);
        ext_mul(r, d, fa);
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(b1[p], b0[p]);
        ext_mul(r, d, fb);
        #pragma unroll
        for (int p = 0; p < 6; p++) d[p] = submod(c1[p], c0[p]);
        ext_mul(r, d, fc);
#endif
        #pragma unroll
        for (int p = 0; p < 6; p++) {
            Ao[p * ps_out + o0] = (int)addmod(fa[p], a0[p]);
            Bo[p * ps_out + o0] = (int)addmod(fb[p], b0[p]);
            Co[p * ps_out + o0] = (int)addmod(fc[p], c0[p]);
        }
    }
}
"""

_KERNELS: dict = {}


def _kernel(name: str, base_in: bool):
    import cupy as cp
    key = (name, bool(base_in))
    if key not in _KERNELS:
        src = f"#define BASE_IN {1 if base_in else 0}\n" + _SRC
        _KERNELS[key] = cp.RawKernel(src, name, options=("-std=c++17",))
    return _KERNELS[key]


def _cp(t: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(t.contiguous())


def planes_of_eq(tab: torch.Tensor) -> torch.Tensor:
    """``(n, 6)`` int64 ext table -> ``(6, n)`` int32 planes (contiguous)."""
    return (tab % P).t().contiguous().to(torch.int32)


class CudaState:
    """Sumcheck-1 state on the device: three tables as int32 base ``(R, Cc)`` (round 1) or ext planes; the same
    ``round()`` / ``fold()`` / ``final()`` protocol as ``sumcheck._RefState``.  Rows phase: the top row bit is folded
    while ``R > 1``; then the ``(6, 1, Cc)`` planes are read as ``(6, Cc, 1)`` and the column bits are folded the same way."""

    THREADS = 256
    MAX_BLOCKS = 8192

    def __init__(self, tabs: dict, tau_rows: torch.Tensor, tau_cols: torch.Tensor):
        from .sumcheck import eq_table
        dev = tabs["a"].device
        self.dev = dev
        self.R, self.Cc = tabs["a"].shape
        self.base = True
        self.T = {s: tabs[s].to(torch.int32).contiguous() for s in "abc"}      # (R, Cc) base
        self.ps = self.R * self.Cc                                              # plane stride once ext
        self.tau_rows, self.tau_cols = tau_rows, tau_cols
        self.phase = "rows"
        self._eq_table = eq_table
        self.wc_full = planes_of_eq(eq_table(tau_cols))                        # (6, Cc)
        self.bytes_peak = 0

    def _weights(self):
        if self.phase == "rows":
            wr = planes_of_eq(self._eq_table(self.tau_rows[:-1]))               # (6, R/2)
            wc = self.wc_full
        else:
            wr = planes_of_eq(self._eq_table(self.tau_cols[:-1]))               # (6, Cc/2)
            wc = torch.zeros((D, 1), dtype=torch.int32, device=self.dev)
            wc[0, 0] = 1
        return wr, wc

    def _launch(self, n: int):
        blocks = int(min(self.MAX_BLOCKS, max(1, (n + self.THREADS - 1) // self.THREADS)))
        return (blocks,), (self.THREADS,)

    def round(self) -> torch.Tensor:
        import cupy as cp
        if self.phase == "rows" and self.R == 1:
            self.phase = "cols"
            self.R, self.Cc = self.Cc, 1
        wr, wc = self._weights()
        hR = self.R // 2
        acc = torch.zeros((18,), dtype=torch.int64, device=self.dev)
        kern = _kernel("sc_msg", self.base)
        n = hR * self.Cc
        grid, block = self._launch(n)
        kern(grid, block, tuple(_cp(self.T[s]) for s in "abc") + (_cp(wr), _cp(wc), _cp(acc), cp.int64(hR), cp.int64(self.Cc), cp.int64(self.ps)))
        return acc.reshape(3, D) % P

    def fold(self, r: torch.Tensor) -> None:
        import cupy as cp
        hR = self.R // 2
        r6 = (r % P).to(torch.int32).contiguous()
        kern = _kernel("sc_fold", self.base)
        if self.base:
            out = {s: torch.empty((D, hR, self.Cc), dtype=torch.int32, device=self.dev) for s in "abc"}
            ps_out = hR * self.Cc
            args = tuple(_cp(self.T[s]) for s in "abc") + tuple(_cp(out[s]) for s in "abc")
        else:
            out, ps_out = self.T, self.ps
            args = tuple(_cp(self.T[s]) for s in "abc") * 2
        n = hR * self.Cc
        grid, block = self._launch(n)
        kern(grid, block, args + (_cp(r6), cp.int64(hR), cp.int64(self.Cc), cp.int64(self.ps), cp.int64(ps_out)))
        if self.base:
            self.T, self.ps, self.base = out, ps_out, False
        self.R = hR
        if self.phase == "rows":
            self.tau_rows = self.tau_rows[:-1]
        else:
            self.tau_cols = self.tau_cols[:-1]

    def final(self) -> torch.Tensor:
        out = []
        for s in "abc":
            flat = self.T[s].reshape(-1)
            out.append(torch.stack([flat[p * self.ps].to(torch.int64) for p in range(D)]))
        return torch.stack(out)
