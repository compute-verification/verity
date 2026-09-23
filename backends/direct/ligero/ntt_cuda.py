"""Four-step CUDA NTT for the prover's batched transforms (lane hp2-host): ``n = 256 * N2`` (``32 <= N2 <= 256``, i.e.
``n`` in 8192 ... 65536), int64 rows in, canonical int64 rows out, the SAME map as ``field.ntt`` (natural order in and
out, forward ``X[k] = sum_i x[i] w^(ik)``, inverse divides by ``n``).

The DFT over F_p is unique and the output is canonical, so any correct evaluation is bit-identical to the torch
radix-2 chain; ``ntt_cuda_test.py`` checks ``torch.equal`` against ``field.ntt`` for both directions on random rows
of every supported size.  ``field.ntt`` dispatches here on CUDA when ``field.CUDA_NTT`` is set.

Why: ``field.ntt`` is 16 torch.compile'd radix-2 stages (a ``cat`` + mul + mod kernel each over 3-19 MB), launch- and
pass-bound: measured on the H100 per bf16-hopper sub-batch, ``intt`` (6, 65536) 0.71 + 1.33 ms and ``eval_on_coset``
(36, 16640 -> 65536) 0.44 ms inside the prover's graphs.  Here: two kernels per call.  Index ``i = i1 + 256 i2``,
``k = N2 k1 + k2``: ``X[N2 k1 + k2] = sum_i1 w_256^(i1 k1) w_n^(i1 k2) sum_i2 x[i1 + 256 i2] w_N2^(i2 k2)``.
Kernel 1 does the length-``N2`` column transforms (32 columns per block, in shared memory, bit-reversed load + DIT)
and the ``w_n^(i1 k2)`` twiddle in place; kernel 2 the length-256 row transforms (32 rows per block) and writes the
result transposed (``out[N2 k1 + k2]``), scaled by ``n^-1`` for the inverse.  Plain ``(a * b) % p`` on 64-bit
products (the work is ~1 M butterflies per row; the torch chain's cost was passes and launches, not arithmetic).
"""
from __future__ import annotations

import numpy as np
import torch

from .field import P, modinv, root_of_unity

_SRC = r"""
#define P 2013265921u
#define N1 256
#define LOGN1 8
#define N2 %(n2)d
#define LOGN2 %(logn2)d
#define TCOLS 32
#define TROWS 32
typedef unsigned long long u64;

__device__ __forceinline__ unsigned mulp(unsigned a, unsigned b) { return (unsigned)(((u64)a * b) %% P); }
__device__ __forceinline__ unsigned addp(unsigned a, unsigned b) { unsigned s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ unsigned subp(unsigned a, unsigned b) { return a >= b ? a - b : a + P - b; }

/* step 1: batch row blockIdx.y, columns [32 blockIdx.x, +32) of the (N2 x N1) matrix M[i2][i1] = x[i1 + N1 i2]:
   length-N2 DIT NTT down each column (bit-reversed load), then * big[k2][i1] = w_n^(i1 k2); y in M's layout. */
extern "C" __global__ void __launch_bounds__(256)
ntt_step1(const long long* __restrict__ x, unsigned* __restrict__ y,
          const unsigned* __restrict__ tw,      /* [N2]: tw[m + t] = w_(2m)^t */
          const unsigned* __restrict__ big,     /* [N2][N1] */
          long long n)
{
    __shared__ unsigned s[N2][TCOLS];
    const long long base = (long long)blockIdx.y * n;
    const int c0 = blockIdx.x * TCOLS;
    for (int e = threadIdx.x; e < N2 * TCOLS; e += 256) {
        const int i2 = e / TCOLS, c = e - i2 * TCOLS;
        long long v = x[base + c0 + c + (long long)N1 * i2] %% (long long)P;   /* any int64 in: reduce, canonical */
        if (v < 0) v += P;
        s[__brev(i2) >> (32 - LOGN2)][c] = (unsigned)v;
    }
    __syncthreads();
    #pragma unroll 1
    for (int lm = 0; lm < LOGN2; ++lm) {
        const int m = 1 << lm;
        for (int e = threadIdx.x; e < (N2 / 2) * TCOLS; e += 256) {
            const int b = e / TCOLS, c = e - b * TCOLS;
            const int t = b & (m - 1);
            const int i = ((b >> lm) << (lm + 1)) | t;
            const unsigned a = s[i][c];
            const unsigned d = mulp(s[i + m][c], __ldg(&tw[m + t]));
            s[i][c] = addp(a, d);
            s[i + m][c] = subp(a, d);
        }
        __syncthreads();
    }
    for (int e = threadIdx.x; e < N2 * TCOLS; e += 256) {
        const int k2 = e / TCOLS, c = e - k2 * TCOLS;
        y[base + c0 + c + (long long)N1 * k2] = mulp(s[k2][c], __ldg(&big[k2 * N1 + c0 + c]));
    }
}

/* step 2: rows [32 blockIdx.x, +32) (k2) of Y[k2][i1]: length-N1 DIT NTT along each row (bit-reversed load),
   * ninv, written transposed: out[N2 k1 + k2] = Z[k2][k1]. */
extern "C" __global__ void __launch_bounds__(256)
ntt_step2(const unsigned* __restrict__ y, long long* __restrict__ out,
          const unsigned* __restrict__ tw,      /* [N1] */
          unsigned ninv, long long n)
{
    __shared__ unsigned s[TROWS][N1 + 1];         /* +1: the transposed reads below are bank-conflict-free */
    const long long base = (long long)blockIdx.y * n;
    const int r0 = blockIdx.x * TROWS;
    for (int e = threadIdx.x; e < TROWS * N1; e += 256) {
        const int r = e / N1, i1 = e - r * N1;
        s[r][__brev(i1) >> 24] = y[base + (long long)N1 * (r0 + r) + i1];
    }
    __syncthreads();
    #pragma unroll 1
    for (int lm = 0; lm < LOGN1; ++lm) {
        const int m = 1 << lm;
        for (int e = threadIdx.x; e < TROWS * (N1 / 2); e += 256) {
            const int r = e / (N1 / 2), b = e - r * (N1 / 2);
            const int t = b & (m - 1);
            const int i = ((b >> lm) << (lm + 1)) | t;
            const unsigned a = s[r][i];
            const unsigned d = mulp(s[r][i + m], __ldg(&tw[m + t]));
            s[r][i] = addp(a, d);
            s[r][i + m] = subp(a, d);
        }
        __syncthreads();
    }
    for (int e = threadIdx.x; e < TROWS * N1; e += 256) {
        const int k1 = e / TROWS, r = e - k1 * TROWS;
        out[base + (long long)N2 * k1 + r0 + r] = (long long)mulp(s[r][k1], ninv);
    }
}
"""


def _stage_twiddles(S: int, w: int) -> np.ndarray:
    """``tw[m + t] = w_S^((S / 2m) t)`` for m = 1, 2, ..., S/2 and t < m (``w`` = a primitive S-th root)."""
    tw = np.zeros(S, dtype=np.uint32)
    m = 1
    while m < S:
        wm = pow(w, S // (2 * m), P)          # a primitive 2m-th root
        cur = 1
        for t in range(m):
            tw[m + t] = cur
            cur = cur * wm % P
        m *= 2
    return tw


class _Plan:
    def __init__(self, n: int, device):
        import cupy as cp
        assert n % 256 == 0 and 32 <= n // 256 <= 256 and (n & (n - 1)) == 0
        self.n, self.device = n, device
        N2 = n // 256
        self.N2 = N2
        logn2 = N2.bit_length() - 1
        src = _SRC % dict(n2=N2, logn2=logn2)
        mod = cp.RawModule(code=src, options=("-std=c++17",))
        self.k1, self.k2 = mod.get_function("ntt_step1"), mod.get_function("ntt_step2")
        self.tabs = {}
        for inverse in (False, True):
            wn = root_of_unity(n)
            if inverse:
                wn = modinv(wn)
            w1, w2 = pow(wn, N2, P), pow(wn, 256, P)                 # primitive 256-th and N2-th roots
            pw = np.empty(n, dtype=np.uint32)                        # w_n^j
            cur = 1
            for j in range(n):
                pw[j] = cur
                cur = cur * wn % P
            k2 = np.arange(N2, dtype=np.int64)[:, None]
            i1 = np.arange(256, dtype=np.int64)[None, :]
            big = pw[(i1 * k2) % n].reshape(-1)                      # [N2][N1]
            t = lambda a: torch.as_tensor(a.astype(np.int32), device=device)  # noqa: E731
            self.tabs[inverse] = (t(_stage_twiddles(N2, w2)), t(_stage_twiddles(256, w1)), t(big),
                                  np.uint32(modinv(n) if inverse else 1))

    def run(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        import cupy as cp
        B, n = x.shape
        tw2, tw1, big, ninv = self.tabs[inverse]
        y = torch.empty((B, n), dtype=torch.int32, device=x.device)
        out = torch.empty((B, n), dtype=torch.int64, device=x.device)
        with cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
            a = lambda t: cp.asarray(t)  # noqa: E731 -- zero-copy views
            self.k1((256 // 32, B), (256,), (a(x), a(y), a(tw2), a(big), np.int64(n)))
            self.k2((self.N2 // 32, B), (256,), (a(y), a(out), a(tw1), ninv, np.int64(n)))
        return out


_PLANS: dict = {}


def supported(x: torch.Tensor) -> bool:
    n = x.shape[-1]
    return (x.is_cuda and x.dtype == torch.int64 and n % 256 == 0 and (n & (n - 1)) == 0
            and 32 <= n // 256 <= 256 and x.numel() > 0)


def ntt(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """``field.ntt`` on the last dimension for supported CUDA int64 inputs (``supported``)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    key = (n, str(x.device))
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _Plan(n, x.device)
    xb = x.contiguous().reshape(-1, n)
    return plan.run(xb, inverse).reshape(lead + (n,))
