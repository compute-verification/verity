"""Batched BabyBear Reed-Solomon encoder on SIMT cores: one CUDA block per row, everything in shared memory.

Per row (``k`` values on the message domain ``H``, ``n = 4k`` outputs; ``babybear`` module docstring for the domains):

1. load the ``k`` values into shared memory (16 KiB at ``k = 4096``);
2. Gentleman-Sande (DIF) radix-2 INTT over ``H`` -- natural input, bit-reversed unscaled coefficients;
3. for each of the three non-trivial cosets ``c = 1, 2, 3``: multiply coefficient ``i`` (found at bit-reversed
   position) by ``k^-1 omega_n^(c i)`` (one fused table per coset, Montgomery form), then a Cooley-Tukey (DIT) radix-2
   NTT over ``H`` -- bit-reversed input, natural output = ``f(omega^c omega_k^j)``, written coalesced to output block
   ``c``;
4. output block 0 is the input row (systematic code): copied by default (``write_systematic=True``, the full 20.4 GB
   codeword materialised) or left to the caller.

Arithmetic: 31-bit Montgomery multiplication with ``R = 2^32`` (twiddles stored as ``w R mod p``, data in plain form,
so ``mont(a, wR) = a w``), add/sub with one conditional correction.  Butterflies per row: ``4 x (k/2) log2 k``
(1 INTT + 3 NTTs of size ``k``) = 98,304 at ``k = 4096``, plus ``3k`` coset twiddle multiplies -- 30% fewer than the
textbook INTT_k + NTT_n (139,264), and the ``k`` systematic outputs are never recomputed.  Stage twiddles
(``omega_(2m)^t``, one 16 KiB table per direction) are read through the read-only cache (``__ldg``), the coset tables
(48 KiB) likewise.  Shared-memory bank conflicts are 2-way in the five stages with ``m < 32``, none above.

Bit-exact against ``babybear.encode_rows`` (``tests/``).  The kernel is compiled by cupy's NVRTC on first use.
"""
from __future__ import annotations

import numpy as np

from . import babybear as bb

_SRC = r"""
#define P 2013265921u
#define NPINV %(npinv)du      /* -p^{-1} mod 2^32 */
#define K %(k)d
#define LOGK %(logk)d
#define THREADS %(threads)d
#define PER_THREAD (K / 2 / THREADS)

__device__ __forceinline__ unsigned add_p(unsigned a, unsigned b) {
    unsigned s = a + b;
    return s >= P ? s - P : s;
}
__device__ __forceinline__ unsigned sub_p(unsigned a, unsigned b) {
    return a >= b ? a - b : a + P - b;
}
/* a in [0,p), w = w' R mod p (Montgomery form): returns a w' mod p. */
__device__ __forceinline__ unsigned mont_mul(unsigned a, unsigned w) {
    unsigned long long x = (unsigned long long)a * w;
    unsigned m = (unsigned)x * NPINV;
    unsigned long long t = (x + (unsigned long long)m * P) >> 32;
    return t >= P ? (unsigned)(t - P) : (unsigned)t;
}

/* DIF (Gentleman-Sande): natural in, bit-reversed out. tw[m + t] = omega_{2m}^{-t} (Montgomery). */
__device__ __forceinline__ void intt_dif(unsigned* x, const unsigned* __restrict__ tw) {
    #pragma unroll 1
    for (int lm = LOGK - 1; lm >= 0; --lm) {
        const int m = 1 << lm;
        #pragma unroll
        for (int s = 0; s < PER_THREAD; ++s) {
            const int b = threadIdx.x + s * THREADS;
            const int t = b & (m - 1);
            const int i = ((b >> lm) << (lm + 1)) | t;
            const unsigned a = x[i], c = x[i + m];
            x[i] = add_p(a, c);
            x[i + m] = mont_mul(sub_p(a, c), __ldg(&tw[m + t]));
        }
        __syncthreads();
    }
}

/* DIT (Cooley-Tukey): bit-reversed in, natural out. tw[m + t] = omega_{2m}^{t} (Montgomery). */
__device__ __forceinline__ void ntt_dit(unsigned* x, const unsigned* __restrict__ tw) {
    #pragma unroll 1
    for (int lm = 0; lm < LOGK; ++lm) {
        const int m = 1 << lm;
        #pragma unroll
        for (int s = 0; s < PER_THREAD; ++s) {
            const int b = threadIdx.x + s * THREADS;
            const int t = b & (m - 1);
            const int i = ((b >> lm) << (lm + 1)) | t;
            const unsigned a = x[i];
            const unsigned c = mont_mul(x[i + m], __ldg(&tw[m + t]));
            x[i] = add_p(a, c);
            x[i + m] = sub_p(a, c);
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(THREADS)
rs_encode(const unsigned* __restrict__ in, unsigned* __restrict__ out,
          const unsigned* __restrict__ tw_inv, const unsigned* __restrict__ tw_fwd,
          const unsigned* __restrict__ coset_tw,   /* 3 x K: k^{-1} omega_n^{c i} at bit-reversed position, Montgomery */
          int rows, int write_systematic)
{
    __shared__ unsigned coef[K];
    __shared__ unsigned x[K];
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const unsigned* src = in + (size_t)row * K;
        unsigned* dst = out + (size_t)row * (4 * K);
        for (int i = threadIdx.x; i < K; i += THREADS) {
            const unsigned v = __ldg(&src[i]);
            coef[i] = v;
            if (write_systematic) dst[i] = v;
        }
        __syncthreads();
        intt_dif(coef, tw_inv);                    /* coef: bit-reversed unscaled coefficients */
        #pragma unroll 1
        for (int c = 1; c < 4; ++c) {
            const unsigned* ct = coset_tw + (size_t)(c - 1) * K;
            for (int i = threadIdx.x; i < K; i += THREADS) x[i] = mont_mul(coef[i], __ldg(&ct[i]));
            __syncthreads();
            ntt_dit(x, tw_fwd);
            unsigned* d = dst + (size_t)c * K;
            for (int i = threadIdx.x; i < K; i += THREADS) d[i] = x[i];
            /* x is rewritten in the next iteration only after the sync inside ntt_dit's first stage; the reads
               above complete before the next writes because every thread writes only the indices it read. */
        }
        __syncthreads();
    }
}
"""


_SRC_V2 = r"""
#define P 2013265921u
#define PINV %(pinv)du        /* p^{-1} mod 2^32 */
#define K %(k)d
#define LOGK %(logk)d
#define THREADS %(threads)d
#define LOGT %(logt)d
#define PER (K / THREADS)            /* elements per thread: i = t + THREADS * s */

__device__ __forceinline__ unsigned add_p(unsigned a, unsigned b) { unsigned s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ unsigned sub_p(unsigned a, unsigned b) { return a >= b ? a - b : a + P - b; }
/* Signed-style Montgomery reduction (Plonky3 monty_reduce): x = a w < p 2^32; t = x p^{-1} mod 2^32; the low words of
   x and t p cancel, so (x - t p) / 2^32 = hi(x) - hi(t p) in (-p, p). */
__device__ __forceinline__ unsigned mont_mul(unsigned a, unsigned w) {
    unsigned long long x = (unsigned long long)a * w;
    unsigned lo = (unsigned)x, hi = (unsigned)(x >> 32);
    unsigned t = lo * PINV;
    unsigned u = __umulhi(t, P);
    unsigned r = hi - u;
    return hi < u ? r + P : r;
}

/* Register-resident stages: thread t owns i = t + THREADS*s (s < PER), which contains the butterfly pairs of every
   stage with m >= THREADS (partner s + m/THREADS).  Shared-memory stages (m < THREADS) use the coalesced butterfly
   mapping b = t + THREADS*s.  Global loads/stores are the strided ownership: perfectly coalesced. */

extern "C" __global__ void __launch_bounds__(THREADS)
rs_encode(const unsigned* __restrict__ in, unsigned* __restrict__ out,
          const unsigned* __restrict__ tw_inv, const unsigned* __restrict__ tw_fwd,
          const unsigned* __restrict__ coset_tw, int rows, int write_systematic)
{
    __shared__ unsigned x[K];
    unsigned r[PER], coef[PER];
    const int t = threadIdx.x;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const unsigned* src = in + (size_t)row * K;
        unsigned* dst = out + (size_t)row * (4 * K);
        #pragma unroll
        for (int s = 0; s < PER; ++s) {
            r[s] = __ldg(&src[t + THREADS * s]);
            if (write_systematic) dst[t + THREADS * s] = r[s];
        }
        /* INTT, DIF: register stages m = K/2 .. THREADS */
        #pragma unroll
        for (int lm = LOGK - 1; lm >= LOGT; --lm) {
            const int m = 1 << lm, ms = m / THREADS;
            #pragma unroll
            for (int s = 0; s < PER; ++s) {
                if (!(s & ms)) {
                    const unsigned a = r[s], c = r[s + ms];
                    r[s] = add_p(a, c);
                    r[s + ms] = mont_mul(sub_p(a, c), __ldg(&tw_inv[m + t + THREADS * (s & (ms - 1))]));
                }
            }
        }
        #pragma unroll
        for (int s = 0; s < PER; ++s) x[t + THREADS * s] = r[s];
        __syncthreads();
        /* INTT, DIF: shared-memory stages m = THREADS/2 .. 1 */
        #pragma unroll
        for (int lm = LOGT - 1; lm >= 0; --lm) {
            const int m = 1 << lm;
            #pragma unroll
            for (int s = 0; s < PER / 2; ++s) {
                const int b = t + s * THREADS;
                const int tt = b & (m - 1);
                const int i = ((b >> lm) << (lm + 1)) | tt;
                const unsigned a = x[i], c = x[i + m];
                x[i] = add_p(a, c);
                x[i + m] = mont_mul(sub_p(a, c), __ldg(&tw_inv[m + tt]));
            }
            __syncthreads();
        }
        #pragma unroll
        for (int s = 0; s < PER; ++s) coef[s] = x[t + THREADS * s];      /* bit-reversed unscaled coefficients */
        #pragma unroll 1
        for (int c = 1; c < 4; ++c) {
            const unsigned* ct = coset_tw + (size_t)(c - 1) * K;
            #pragma unroll
            for (int s = 0; s < PER; ++s) x[t + THREADS * s] = mont_mul(coef[s], __ldg(&ct[t + THREADS * s]));
            __syncthreads();
            /* NTT, DIT: shared-memory stages m = 1 .. THREADS/2 */
            #pragma unroll
            for (int lm = 0; lm < LOGT; ++lm) {
                const int m = 1 << lm;
                #pragma unroll
                for (int s = 0; s < PER / 2; ++s) {
                    const int b = t + s * THREADS;
                    const int tt = b & (m - 1);
                    const int i = ((b >> lm) << (lm + 1)) | tt;
                    const unsigned a = x[i];
                    const unsigned cc = mont_mul(x[i + m], __ldg(&tw_fwd[m + tt]));
                    x[i] = add_p(a, cc);
                    x[i + m] = sub_p(a, cc);
                }
                __syncthreads();
            }
            #pragma unroll
            for (int s = 0; s < PER; ++s) r[s] = x[t + THREADS * s];
            /* NTT, DIT: register stages m = THREADS .. K/2 */
            #pragma unroll
            for (int lm = LOGT; lm < LOGK; ++lm) {
                const int m = 1 << lm, ms = m / THREADS;
                #pragma unroll
                for (int s = 0; s < PER; ++s) {
                    if (!(s & ms)) {
                        const unsigned a = r[s];
                        const unsigned cc = mont_mul(r[s + ms], __ldg(&tw_fwd[m + t + THREADS * (s & (ms - 1))]));
                        r[s] = add_p(a, cc);
                        r[s + ms] = sub_p(a, cc);
                    }
                }
            }
            unsigned* d = dst + (size_t)c * K;
            #pragma unroll
            for (int s = 0; s < PER; ++s) d[t + THREADS * s] = r[s];
        }
    }
}
"""


def twiddle_tables(k: int, n: int) -> dict[str, np.ndarray]:
    """Host-side constant tables (uint32, Montgomery form): DIF inverse-stage twiddles, DIT forward-stage twiddles and
    the three fused coset tables ``k^-1 omega_n^(c i)`` stored at bit-reversed coefficient position."""
    wn = bb.root_of_unity(n)
    wk = pow(wn, n // k, bb.P)
    wk_inv = pow(wk, -1, bb.P)
    tw_fwd = np.zeros(k, dtype=np.uint64)
    tw_inv = np.zeros(k, dtype=np.uint64)
    m = 1
    while m < k:
        tw_fwd[m:2 * m] = bb.powers(pow(wk, k // (2 * m), bb.P), m)
        tw_inv[m:2 * m] = bb.powers(pow(wk_inv, k // (2 * m), bb.P), m)
        m *= 2
    rev = bb.bit_reverse_indices(k)
    kinv = pow(k, -1, bb.P)
    cos = []
    for c in range(1, n // k):
        t = (bb.powers(pow(wn, c, bb.P), k) * np.uint64(kinv)) % np.uint64(bb.P)   # indexed by coefficient i
        cos.append(t[rev])                                                          # position -> coefficient rev(pos)
    mont = lambda a: ((a % np.uint64(bb.P)) * np.uint64(bb.R_MOD_P) % np.uint64(bb.P)).astype(np.uint32)
    return {"tw_fwd": mont(tw_fwd), "tw_inv": mont(tw_inv), "coset": mont(np.concatenate(cos))}


class RSEncoderSIMT:
    """cupy-backed encoder; ``__call__(rows_u32 (R, k) on device, out (R, n) or None) -> out``."""

    def __init__(self, k: int = bb.K_CODE, n: int = bb.N_CODE, threads: int = 512, blocks: int | None = None, variant: str = "v2"):
        import cupy as cp

        if n != 4 * k or k & (k - 1) or k < 2 * threads or threads & (threads - 1):
            raise ValueError("this kernel is specialised to n = 4k, k and threads powers of two, k >= 2 x threads")
        self.k, self.n, self.threads, self.variant = k, n, threads, variant
        self.cp = cp
        params = {"npinv": bb.NEG_P_INV_MOD_R, "pinv": bb.P_INV_MOD_R, "k": k, "logk": k.bit_length() - 1, "threads": threads, "logt": threads.bit_length() - 1}
        src = {"v1": _SRC, "v2": _SRC_V2}[variant] % params
        self.kernel = cp.RawKernel(src, "rs_encode", options=("-std=c++17",))
        tabs = twiddle_tables(k, n)
        self.tw_fwd = cp.asarray(tabs["tw_fwd"])
        self.tw_inv = cp.asarray(tabs["tw_inv"])
        self.coset = cp.asarray(tabs["coset"])
        sms = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        self.blocks = blocks or sms * 16

    def __call__(self, rows, out=None, write_systematic: bool = True):
        cp = self.cp
        rows = cp.ascontiguousarray(rows).view(cp.uint32)
        R = rows.shape[0]
        if rows.shape != (R, self.k):
            raise ValueError(rows.shape)
        if out is None:
            out = cp.empty((R, self.n), dtype=cp.uint32)
        self.kernel((min(self.blocks, R),), (self.threads,),
                    (rows, out, self.tw_inv, self.tw_fwd, self.coset, np.int32(R), np.int32(1 if write_systematic else 0)))
        return out

    @staticmethod
    def butterflies_per_row(k: int) -> int:
        return 4 * (k // 2) * (k.bit_length() - 1)
