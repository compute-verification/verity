"""Fused device kernels for the prover's tests (the 52% bucket after the SIMT encoder): the D-fold linear combination
of a row matrix and the batched quadratic test, each in ONE pass over the matrix instead of D (or 3 + D) torch passes.

  lincomb(C, X)      (D, cols) = C (D, R) @ X (R, cols) mod p                              -- protocol._combine
  lincomb(C, X, sq)  (D, cols) = sum_j C[d, j] (X_j^2 - X_j) mod p                         -- witness._bool_row x D
  quad_sparse(...)   (D, cols) = sum_q rho[d, q] (A_q(X) B_q(X) - C_q(X))  for the general (non-booleanity) constraints
                     with A_q = const + sum coef X_j read straight from the CSR tables    -- witness.apply_sparse x 3

One thread per column, a grid split over the rows (S partial sums, reduced mod p afterwards) so the L40S has
> 500K threads in flight; every X read is coalesced (a row across the warp) and X is read once.  Products
c * x < 2^62 are accumulated three at a time in 64 bits before one reduction.  Results are exact mod p: bit-exact
against the torch paths (tests_fused_test.py, and the gates run through them).  X may be int32 (the SIMT codeword)
or int64 (coefficient matrices); C and the tables are uint32 < p.
"""

from __future__ import annotations

import numpy as np
import torch

from .field import P

_SRC = r"""
#define P 2013265921u
typedef unsigned long long u64;

/* partial[s][d][col] = sum over rows [s*chunk, (s+1)*chunk) of C[d][j] * f(X[rows ? rows[j] : j][col]) */
template <typename T, int D, int SQ>
__device__ __forceinline__ void lincomb_body(const unsigned* __restrict__ C, const T* __restrict__ X,
                                             unsigned* __restrict__ partial, int R, int cols, int chunk,
                                             const int* __restrict__ rows = nullptr)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (col >= cols) return;
    const int j0 = s * chunk, j1 = min(R, j0 + chunk);
    u64 acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0;
    int cnt = 0;
    for (int j = j0; j < j1; ++j) {
        u64 x = (u64)(unsigned)X[(size_t)(rows ? rows[j] : j) * cols + col];
        if (SQ) x = (x * x + (u64)P - x) % P;              /* x^2 - x mod p (x < p) */
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] += (u64)C[(size_t)d * R + j] * x;
        if (++cnt == 3) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[d] %= P;
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) partial[((size_t)s * D + d) * cols + col] = (unsigned)(acc[d] % P);
}

/* out[d][col] = sum_r Renc[d][r][col] * U[rows[r]][col] mod p (the chain quotient on the codeword domain) */
template <typename T, int D>
__device__ __forceinline__ void chainq_body(const T* __restrict__ Renc, const unsigned* __restrict__ U,
                                            const int* __restrict__ rows, u64* __restrict__ out, int R, int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    u64 acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0;
    int cnt = 0;
    for (int r = 0; r < R; ++r) {
        const u64 x = (u64)U[(size_t)rows[r] * cols + col];
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] += (u64)(unsigned)Renc[((size_t)d * R + r) * cols + col] * x;
        if (++cnt == 3) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[d] %= P;
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) out[(size_t)d * cols + col] = acc[d] % P;
}

extern "C" __global__ void chainq_u32_D@D@(const unsigned* Renc, const unsigned* U, const int* rows, u64* out, int R, int cols)
{ chainq_body<unsigned, @D@>(Renc, U, rows, out, R, cols); }
extern "C" __global__ void chainq_u64_D@D@(const u64* Renc, const unsigned* U, const int* rows, u64* out, int R, int cols)
{ chainq_body<u64, @D@>(Renc, U, rows, out, R, cols); }

/* coef[d][lane_pos[16 o + i]][steps v + s] += sum_x u[d][o][x][v] pw[x][s][i] mod p (the fingerprint constraints' lane
   coefficients, chain.chain_coefs); grid (cols / THREADS, n_op 16, D) */
extern "C" __global__ void fplane_D@D@(const unsigned* __restrict__ u, const unsigned* __restrict__ pw,
                                       const int* __restrict__ lane_pos, u64* __restrict__ coef,
                                       int R, int l, int n_op, int C, int n_slots, int steps)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int oi = blockIdx.y, d = blockIdx.z;
    if (col >= n_slots * steps) return;
    const int v = col / steps, s = col - v * steps, o = oi >> 4, i = oi & 15;
    const unsigned* uu = u + (size_t)(d * n_op + o) * C * n_slots + v;
    const unsigned* pp = pw + (size_t)s * 16 + i;
    u64 acc = 0;
    for (int x = 0; x < C; ++x) {
        acc += (u64)uu[(size_t)x * n_slots] * pp[(size_t)x * steps * 16];
        if (x % 3 == 2) acc %= P;
    }
    u64* dst = coef + ((size_t)d * R + lane_pos[oi]) * l + col;
    *dst = (*dst + acc) % P;
}

extern "C" __global__ void lincomb_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<unsigned, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void lincomb_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<u64, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<unsigned, @D@, 1>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<u64, @D@, 1>(C, X, out, R, cols, chunk); }
/* the booleanity rows read in place through a row-index table (no index_select copy of the rows; lane hostphase) */
extern "C" __global__ void boolcomb_rows_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk, const int* rows)
{ lincomb_body<unsigned, @D@, 1>(C, X, out, R, cols, chunk, rows); }
extern "C" __global__ void boolcomb_rows_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk, const int* rows)
{ lincomb_body<u64, @D@, 1>(C, X, out, R, cols, chunk, rows); }

/* VEC = 4 adjacent columns per thread through 16-byte loads, the block's slice of C staged in shared memory (lane
   hp2-host): the same sums as lincomb_body -- one C read per 4 products instead of per product, 4 D independent
   accumulator chains per thread, and a reduction every 4 rows (c, x < p: 4 (p-1)^2 < 2^64).  Requires cols % 4 == 0 and
   16-byte aligned rows (the Python side checks and falls back to the scalar kernels). */
template <typename T> __device__ __forceinline__ void load4(const T* __restrict__ p, u64* x);
template <> __device__ __forceinline__ void load4<unsigned>(const unsigned* __restrict__ p, u64* x)
{
    const uint4 q = *reinterpret_cast<const uint4*>(p);
    x[0] = q.x; x[1] = q.y; x[2] = q.z; x[3] = q.w;
}
template <> __device__ __forceinline__ void load4<u64>(const u64* __restrict__ p, u64* x)
{
    const ulonglong2 a = *reinterpret_cast<const ulonglong2*>(p), b = *reinterpret_cast<const ulonglong2*>(p + 2);
    x[0] = (u64)(unsigned)a.x; x[1] = (u64)(unsigned)a.y; x[2] = (u64)(unsigned)b.x; x[3] = (u64)(unsigned)b.y;
}

/* Montgomery reduction t -> t 2^-32 mod p as a value < 2^32 + p (exact: t + m p is a multiple of 2^32; the low words
   sum to 0 or 2^32 -- the carry is (t_lo != 0)).  PPRIME = -p^-1 mod 2^32.  ~5 instructions against ~25 for a 64-bit
   modulo by p.  The v4 kernels take C pre-scaled by 2^32 (lincomb) or 2^64 (booleanity: the row value is itself
   reduced once) so that one reduction per lazily accumulated product sum yields the plain sum mod p. */
#define PPRIME 0x77ffffffu
__device__ __forceinline__ u64 mont_reduce(u64 t)
{
    const unsigned t_lo = (unsigned)t;
    const unsigned m = t_lo * PPRIME;
    const u64 mp = (u64)m * (u64)P;
    return (t >> 32) + (mp >> 32) + (u64)(t_lo != 0u);
}

template <typename T, int D, int SQ>
__device__ __forceinline__ void lincomb_v4_body(const unsigned* __restrict__ C, const T* __restrict__ X,
                                                unsigned* __restrict__ partial, int R, int cols, int chunk,
                                                const int* __restrict__ rows = nullptr)
{
    extern __shared__ unsigned Cs[];                                   /* [D][chunk], C pre-scaled (see mont_reduce) */
    int* Rs = reinterpret_cast<int*>(Cs + D * chunk);                  /* [chunk] row indices (rows variant) */
    const int s = blockIdx.y;
    const int j0 = s * chunk, j1 = min(R, j0 + chunk), nj = j1 - j0;
    for (int i = threadIdx.x; i < D * nj; i += blockDim.x) {
        const int d = i / nj, jj = i - d * nj;
        Cs[d * chunk + jj] = C[(size_t)d * R + j0 + jj];
    }
    if (rows) {   /* staged once per block: the loop's row address then comes from shared memory, not a dependent
                     global load per row (measured: the rows variant ran 0.84-0.96 ms vs 0.72 for the gathered one) */
        for (int i = threadIdx.x; i < nj; i += blockDim.x) Rs[i] = rows[j0 + i];
    }
    __syncthreads();
    const int col0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (col0 >= cols) return;
    u64 lazy[D][4];     /* sums of up to 4 products c w < (p-1)^2 each: < 2^64 */
    u64 red[D][4];      /* sums of the reductions of the lazy sums: each < 2^32 + p, at most chunk / 4 <= 256 of them */
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) { lazy[d][v] = 0; red[d][v] = 0; }
    }
    int cnt = 0;
    #pragma unroll 4
    for (int jj = 0; jj < nj; ++jj) {
        const int j = j0 + jj;
        u64 x[4];
        load4<T>(X + (size_t)(rows ? Rs[jj] : j) * cols + col0, x);
        if (SQ) {
            /* w = (x^2 - x) 2^-32 mod p, canonical: t = x^2 + p - x < p^2 + p, so the reduction is < 2^30 + p + 1 < 2p.
               The (unsigned) truncation is exact (w < p < 2^31) and tells the compiler the high word is zero, so the
               products below are one 32x32->64 IMAD each instead of a 64x64 multiply (x's provenance is a 64-bit
               conditional subtract, which it cannot range-analyse; the non-SQ x comes zero-extended from load4). */
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const u64 w = mont_reduce(x[v] * x[v] + (u64)P - x[v]);
                x[v] = (u64)(unsigned)((w >= (u64)P) ? w - (u64)P : w);
            }
        }
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const u64 c = Cs[d * chunk + jj];
            #pragma unroll
            for (int v = 0; v < 4; ++v) lazy[d][v] += c * x[v];
        }
        if (++cnt == 4) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) { red[d][v] += mont_reduce(lazy[d][v]); lazy[d][v] = 0; }
            }
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        uint4 o;
        o.x = (unsigned)((red[d][0] + mont_reduce(lazy[d][0])) % P);
        o.y = (unsigned)((red[d][1] + mont_reduce(lazy[d][1])) % P);
        o.z = (unsigned)((red[d][2] + mont_reduce(lazy[d][2])) % P);
        o.w = (unsigned)((red[d][3] + mont_reduce(lazy[d][3])) % P);
        *reinterpret_cast<uint4*>(partial + ((size_t)s * D + d) * cols + col0) = o;
    }
}

extern "C" __global__ void lincomb_v4_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_v4_body<unsigned, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void lincomb_v4_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_v4_body<u64, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_v4_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_v4_body<unsigned, @D@, 1>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_v4_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_v4_body<u64, @D@, 1>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_rows_v4_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk, const int* rows)
{ lincomb_v4_body<unsigned, @D@, 1>(C, X, out, R, cols, chunk, rows); }
extern "C" __global__ void boolcomb_rows_v4_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk, const int* rows)
{ lincomb_v4_body<u64, @D@, 1>(C, X, out, R, cols, chunk, rows); }

/* general quadratic constraints from CSR tables: for side s in {a,b,c}: ptr_s[q]..ptr_s[q+1] -> (j, coef), const_s[q] */
template <typename T>
__device__ __forceinline__ unsigned sparse_eval(const T* __restrict__ X, int cols, int col,
                                                const int* __restrict__ ptr, const int* __restrict__ jj,
                                                const unsigned* __restrict__ cc, unsigned cst, int q)
{
    u64 acc = cst;
    int cnt = 0;
    for (int e = ptr[q]; e < ptr[q + 1]; ++e) {
        acc += (u64)cc[e] * (u64)(unsigned)X[(size_t)jj[e] * cols + col];
        if (++cnt == 3) { cnt = 0; acc %= P; }
    }
    return (unsigned)(acc % P);
}

template <typename T, int D>
__device__ __forceinline__ void quad_body(const unsigned* __restrict__ rho, const T* __restrict__ X,
                                          const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
                                          const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
                                          const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc,
                                          unsigned* __restrict__ partial, int Q, int cols, int chunk)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (col >= cols) return;
    const int q0 = s * chunk, q1 = min(Q, q0 + chunk);
    u64 acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0;
    int cnt = 0;
    for (int q = q0; q < q1; ++q) {
        u64 a = sparse_eval<T>(X, cols, col, pa, ja, ca, ka[q], q);
        u64 b = sparse_eval<T>(X, cols, col, pb, jb, cb, kb[q], q);
        u64 c = sparse_eval<T>(X, cols, col, pc, jc, ccn, kc[q], q);
        const unsigned t = (unsigned)((a * b + (u64)P - c) % P);      /* < p: 32-bit multiplicand (one IMAD below) */
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] += (u64)rho[(size_t)d * Q + q] * (u64)t;
        if (++cnt == 3) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[d] %= P;
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) partial[((size_t)s * D + d) * cols + col] = (unsigned)(acc[d] % P);
}

/* The general constraints, VEC = 4 adjacent columns per thread (16-byte row loads; one table read per 4 columns) in
   Montgomery arithmetic, every intermediate canonical (< p) and 32-bit: a side's terms are summed lazily 4 products at
   a time (4 (p-1)^2 < 2^64) and each such sum is Montgomery-reduced (< 2^31.9 + p < 3p) and brought below p by two
   conditional subtractions.  The side-a coefficients and constant come scaled by 2^64 / 2^32 (A arrives as A 2^32), so
   mont(A 2^32 * B) = A B (< 2p: one subtraction); rho comes scaled by 2^32 (see lincomb_v4_body).  X canonical.
   The same residues as quad_body. */
__device__ __forceinline__ unsigned canon3p(u64 r)            /* r < 3p -> r mod p */
{
    unsigned v = (unsigned)(r >= 2ull * P ? r - 2ull * P : r);    /* < 2p < 2^32 */
    return v >= P ? v - P : v;
}

template <typename T>
__device__ __forceinline__ void side_eval4(const T* __restrict__ X, int cols, int col0, const int* __restrict__ ptr,
                                           const int* __restrict__ jj, const unsigned* __restrict__ cc, unsigned cst,
                                           int q, unsigned* out)
{
    unsigned acc[4] = {cst, cst, cst, cst};
    const int e0 = __ldg(ptr + q), e1 = __ldg(ptr + q + 1);
    for (int e = e0; e < e1; e += 4) {
        u64 lazy[4] = {0, 0, 0, 0};
        const int ne = e1 - e;                  /* the same for the whole warp: the guards below do not diverge */
        uint4 xr[4];
        unsigned cr[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {          /* up to 4 independent row loads in flight before the products */
            if (u < ne) {
                cr[u] = __ldg(cc + e + u);
                xr[u] = *reinterpret_cast<const uint4*>(X + (size_t)__ldg(jj + e + u) * cols + col0);
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (u < ne) {
                const u64 c = cr[u];
                lazy[0] += c * xr[u].x; lazy[1] += c * xr[u].y; lazy[2] += c * xr[u].z; lazy[3] += c * xr[u].w;
            }
        }
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const unsigned s = acc[v] + canon3p(mont_reduce(lazy[v]));   /* < 2p < 2^32 */
            acc[v] = s >= P ? s - P : s;
        }
    }
    #pragma unroll
    for (int v = 0; v < 4; ++v) out[v] = acc[v];
}

template <typename T, int D>
__device__ __forceinline__ void quad_v4_body(const unsigned* __restrict__ rho, const T* __restrict__ X,
                                             const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
                                             const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
                                             const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc,
                                             unsigned* __restrict__ partial, int Q, int cols, int chunk, int qfast)
{
    extern __shared__ unsigned Rs[];                                   /* [D][chunk] rho 2^32 */
    /* qfast: the q-chunks of one column slice are consecutive blocks (co-resident: their row reads meet in L2) */
    const int s = qfast ? blockIdx.x : blockIdx.y, cblk = qfast ? blockIdx.y : blockIdx.x;
    const int q0 = s * chunk, q1 = min(Q, q0 + chunk), nq = q1 - q0;
    for (int i = threadIdx.x; i < D * nq; i += blockDim.x) {
        const int d = i / nq, qq = i - d * nq;
        Rs[d * chunk + qq] = rho[(size_t)d * Q + q0 + qq];
    }
    __syncthreads();
    const int col0 = (cblk * blockDim.x + threadIdx.x) * 4;
    if (col0 >= cols) return;
    u64 lazy[D][4];
    unsigned red[D][4];     /* canonical running sums */
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) { lazy[d][v] = 0; red[d][v] = 0; }
    }
    int cnt = 0;
    for (int qq = 0; qq < nq; ++qq) {
        const int q = q0 + qq;
        unsigned a[4], b[4], c[4];
        side_eval4<T>(X, cols, col0, pa, ja, ca, __ldg(ka + q), q, a);
        side_eval4<T>(X, cols, col0, pb, jb, cb, __ldg(kb + q), q, b);
        side_eval4<T>(X, cols, col0, pc, jc, ccn, __ldg(kc + q), q, c);
        u64 t[4];
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            u64 ab = mont_reduce((u64)a[v] * (u64)b[v]);                  /* < 2p */
            unsigned abu = (unsigned)(ab >= (u64)P ? ab - (u64)P : ab);
            unsigned tv = abu + (P - c[v]);                               /* < 2p */
            t[v] = (u64)(tv >= P ? tv - P : tv);
        }
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const u64 r = Rs[d * chunk + qq];
            #pragma unroll
            for (int v = 0; v < 4; ++v) lazy[d][v] += r * t[v];
        }
        if (++cnt == 4) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const unsigned sm = red[d][v] + canon3p(mont_reduce(lazy[d][v]));
                    red[d][v] = sm >= P ? sm - P : sm;
                    lazy[d][v] = 0;
                }
            }
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        unsigned o4[4];
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const unsigned sm = red[d][v] + canon3p(mont_reduce(lazy[d][v]));
            o4[v] = sm >= P ? sm - P : sm;
        }
        *reinterpret_cast<uint4*>(partial + ((size_t)s * D + d) * cols + col0) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    }
}

extern "C" __global__ void quad_v4_u32_D@D@(const unsigned* rho, const unsigned* X,
    const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
    const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
    const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc, unsigned* out, int Q, int cols, int chunk, int qfast)
{ quad_v4_body<unsigned, @D@>(rho, X, pa, ja, ca, ka, pb, jb, cb, kb, pc, jc, ccn, kc, out, Q, cols, chunk, qfast); }

/* out[i] = sum_s partial[s][i] mod p as int64 (S <= 2^9 terms < 2^31): the (S, D, cols) partial sums in one pass */
extern "C" __global__ void reduce_partial(const unsigned* __restrict__ partial, u64* __restrict__ out, int S, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 acc = 0;
    for (int s = 0; s < S; ++s) acc += partial[(size_t)s * n + i];
    out[i] = acc % P;
}

extern "C" __global__ void quad_u32_D@D@(const unsigned* rho, const unsigned* X,
    const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
    const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
    const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc, unsigned* out, int Q, int cols, int chunk)
{ quad_body<unsigned, @D@>(rho, X, pa, ja, ca, ka, pb, jb, cb, kb, pc, jc, ccn, kc, out, Q, cols, chunk); }
extern "C" __global__ void quad_u64_D@D@(const unsigned* rho, const u64* X,
    const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
    const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
    const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc, unsigned* out, int Q, int cols, int chunk)
{ quad_body<u64, @D@>(rho, X, pa, ja, ca, ka, pb, jb, cb, kb, pc, jc, ccn, kc, out, Q, cols, chunk); }
"""

THREADS = 256
MIN_THREADS_IN_FLIGHT = 1 << 19     # split the rows until this many threads are resident (L40S: 142 SMs x 2048)
VEC_COLS = 4                        # columns per thread in the linear combinations (4: the v4 kernels; 1: the scalar kernels, A/B)
_R32 = (1 << 32) % P                # 2^32 mod p (Montgomery scale of C for the v4 kernels)
_R64 = _R32 * _R32 % P              # 2^64 mod p
assert (P * 0x77FFFFFF) % (1 << 32) == (1 << 32) - 1, "PPRIME = -p^-1 mod 2^32"
MIN_THREADS_V4 = 1 << 17            # the v4 kernels: 4x the work per thread, so a quarter of the threads in flight
MAX_CHUNK_V4 = 1024                 # rows per block slice (shared C: D x chunk x 4 bytes <= 24 KB)
QUAD_V4 = True                      # the general quadratic constraints through quad_v4 (int32 X; False: the scalar quad kernel, A/B)
QUAD_SPLITS = 128                   # quad_v4: constraint chunks (grid dimension) -- the partial sums are (QUAD_SPLITS, D, cols)
QUAD_QFAST = True                   # quad_v4: the chunks of one column slice as consecutive blocks (L2 reuse of their rows)
QUAD_THREADS = 128                  # 4090, fp8-ada-v3x4 l=4096: 0.785 ms vs 0.87 at (32 chunks, column-major, 256 threads)
REDUCE_KERNEL = True                # the (S, D, cols) partial sums reduced by one kernel (False: torch int64 upcast + sum + mod)


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class FusedTests:
    def __init__(self, D: int, device):
        import cupy as cp
        self.cp = cp
        self.D = D
        self.device = torch.device(device)
        self.mod = cp.RawModule(code=_SRC.replace("@D@", str(D)), options=("-std=c++17",))
        self.k = {name: self.mod.get_function(f"{name}_D{D}")
                  for name in ("lincomb_u32", "lincomb_u64", "boolcomb_u32", "boolcomb_u64", "quad_u32", "quad_u64",
                               "boolcomb_rows_u32", "boolcomb_rows_u64",
                               "lincomb_v4_u32", "lincomb_v4_u64", "boolcomb_v4_u32", "boolcomb_v4_u64",
                               "boolcomb_rows_v4_u32", "boolcomb_rows_v4_u64", "chainq_u32", "chainq_u64", "fplane",
                               "quad_v4_u32")}
        self.k["reduce_partial"] = self.mod.get_function("reduce_partial")

    def _cp(self, t: torch.Tensor):
        return self.cp.asarray(t)

    def _stream(self):
        import warnings
        ts = torch.cuda.current_stream(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self.cp.cuda.ExternalStream(ts.cuda_stream)

    @staticmethod
    def _grid(cols: int, R: int):
        splits = max(1, min(R, -(-MIN_THREADS_IN_FLIGHT // max(cols, 1))))
        chunk = -(-R // splits)
        splits = -(-R // chunk)
        return (-(-cols // THREADS), splits), chunk

    @staticmethod
    def _grid_v4(cols: int, R: int):
        threads = cols // 4
        splits = max(1, min(R, -(-MIN_THREADS_V4 // max(threads, 1))))
        splits = max(splits, -(-R // MAX_CHUNK_V4))
        chunk = -(-R // splits)
        splits = -(-R // chunk)
        return (-(-threads // THREADS), splits), chunk

    @staticmethod
    def _v4_ok(X: torch.Tensor) -> bool:
        return VEC_COLS == 4 and X.shape[1] % 4 == 0 and X.data_ptr() % 16 == 0

    def _reduce(self, partial: torch.Tensor) -> torch.Tensor:
        # (S, D, cols) uint32 partial sums -> (D, cols) int64 mod p; S <= 512 terms < 2^31 fit in int64
        if not REDUCE_KERNEL:
            return partial.to(torch.int64).sum(0) % P
        S = partial.shape[0]
        out = torch.empty(partial.shape[1:], dtype=torch.int64, device=self.device)
        nel = out.numel()
        with self._stream():
            self.k["reduce_partial"]((-(-nel // THREADS),), (THREADS,),
                                     (self._cp(partial), self._cp(out), np.int32(S), np.int64(nel)))
        return out

    def lincomb(self, C: torch.Tensor, X: torch.Tensor, square_minus: bool = False,
                rows: torch.Tensor | None = None) -> torch.Tensor:
        """(D, cols) int64 = C (D, R) @ f(X) (R, cols) mod p; f = identity, or x^2 - x when ``square_minus``.
        ``rows`` (R,) int: the rows of X the R combination columns refer to (read in place; ``square_minus`` only) --
        the same result as ``lincomb(C, X.index_select(0, rows), True)`` without the copy."""
        D, R = C.shape
        assert D == self.D and (X.shape[0] == R if rows is None else rows.shape[0] == R)
        cols = X.shape[1]
        X = X.contiguous()
        if X.dtype not in (torch.int32, torch.int64):
            raise TypeError(X.dtype)
        v4 = self._v4_ok(X)
        if v4:   # Montgomery arithmetic in the kernel: C scaled by 2^64 (the row value is reduced once too) or 2^32
            C32 = ((C % P) * (_R64 if square_minus else _R32) % P).to(torch.int32).contiguous()
        else:
            C32 = (C % P).to(torch.int32).contiguous()
        (gx, S), chunk = self._grid_v4(cols, R) if v4 else self._grid(cols, R)
        partial = torch.empty((S, D, cols), dtype=torch.int32, device=self.device)
        suffix = ("_v4" if v4 else "") + ("_u32" if X.dtype == torch.int32 else "_u64")
        args = [self._cp(C32), self._cp(X), self._cp(partial), np.int32(R), np.int32(cols), np.int32(chunk)]
        if rows is not None:
            if not square_minus:
                raise ValueError("rows is for the booleanity combination")
            name = "boolcomb_rows" + suffix
            args.append(self._cp(rows.to(torch.int32).contiguous()))
        else:
            name = ("boolcomb" if square_minus else "lincomb") + suffix
        smem = ((D + (1 if rows is not None else 0)) * chunk * 4) if v4 else 0     # C slice (+ the row indices)
        with self._stream():
            self.k[name]((gx, S), (THREADS,), tuple(args), shared_mem=smem)
        return self._reduce(partial)

    def chain_quotient(self, Renc: torch.Tensor, U: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """(D, n) int64 = sum_r Renc[:, r] * U[rows[r]] mod p in one pass (``Renc`` (D, R, n) int32 codewords or int64
        residues, ``U`` (M, n) int32 residues) -- the chain test's ``((Renc * U[rows]) % p).sum(1) % p`` without the
        int64 upcast, the gathered rows and the (D, R, n) product."""
        D, R, cols = Renc.shape
        assert D == self.D and U.dtype == torch.int32 and U.shape[1] == cols and rows.shape[0] == R
        Renc, U = Renc.contiguous(), U.contiguous()
        out = torch.empty((D, cols), dtype=torch.int64, device=self.device)
        name = "chainq_u32" if Renc.dtype == torch.int32 else "chainq_u64"
        with self._stream():
            self.k[name]((-(-cols // THREADS),), (THREADS,),
                         (self._cp(Renc), self._cp(U), self._cp(rows.to(torch.int32).contiguous()), self._cp(out),
                          np.int32(R), np.int32(cols)))
        return out

    def fp_lanes(self, u: torch.Tensor, pw: torch.Tensor, lane_pos: torch.Tensor, coef: torch.Tensor, steps: int) -> None:
        """``coef[:, lane_pos[16 o + i], steps v + s] += sum_x u[:, o, x, v] pw[x, s, i] mod p`` in place (``u`` (D, n_op,
        C, n_slots) and ``pw`` (C, steps, 16) residues, ``coef`` (D, R, l) int64 residues, contiguous)."""
        D, n_op, C, n_slots = u.shape
        assert D == self.D and coef.dtype == torch.int64 and coef.is_contiguous() and tuple(pw.shape) == (C, steps, 16)
        cols = n_slots * steps
        with self._stream():
            self.k["fplane"]((-(-cols // THREADS), n_op * 16, D), (THREADS,),
                             (self._cp(u.to(torch.int32).contiguous()), self._cp(pw.to(torch.int32).contiguous()),
                              self._cp(lane_pos.to(torch.int32).contiguous()), self._cp(coef),
                              np.int32(coef.shape[1]), np.int32(coef.shape[2]), np.int32(n_op), np.int32(C),
                              np.int32(n_slots), np.int32(steps)))

    def quad_general(self, rho: torch.Tensor, X: torch.Tensor, csr: dict) -> torch.Tensor:
        """(D, cols) = sum_q rho[d, q] (A_q B_q - C_q)(X) over the general constraints in ``csr`` (see ``csr_tables``)."""
        D, Q = rho.shape
        assert D == self.D and Q == csr["Q"]
        cols = X.shape[1]
        X = X.contiguous()
        if QUAD_V4 and X.dtype == torch.int32 and self._v4_ok(X):
            return self._quad_v4(rho, X, csr)
        rho32 = (rho % P).to(torch.int32).contiguous()
        (gx, S), chunk = self._grid(cols, Q)
        partial = torch.empty((S, D, cols), dtype=torch.int32, device=self.device)
        name = "quad_u32" if X.dtype == torch.int32 else "quad_u64"
        args = [self._cp(rho32), self._cp(X)]
        for side in ("a", "b", "c"):
            args += [self._cp(csr[side][k]) for k in ("ptr", "j", "coef", "const")]
        args += [self._cp(partial), np.int32(Q), np.int32(cols), np.int32(chunk)]
        with self._stream():
            self.k[name]((gx, S), (THREADS,), tuple(args))
        return self._reduce(partial)

    def _quad_v4(self, rho: torch.Tensor, X: torch.Tensor, csr: dict) -> torch.Tensor:
        """``quad_general`` through quad_v4 (int32 canonical X, 4 columns per thread): the tables Montgomery-scaled once
        per csr (side a by 2^64 / 2^32, sides b and c by 2^32 / 1), rho by 2^32."""
        D, Q = rho.shape
        cols = X.shape[1]
        mt = csr.get("v4")
        if mt is None:
            mt = {}
            for side in ("a", "b", "c"):
                cs, ks = (_R64, _R32) if side == "a" else (_R32, 1)
                t = csr[side]
                mt[side] = (t["ptr"], t["j"],
                            (t["coef"].to(torch.int64) % P * cs % P).to(torch.int32).contiguous(),
                            (t["const"].to(torch.int64) % P * ks % P).to(torch.int32).contiguous())
            csr["v4"] = mt
        rho32 = ((rho % P) * _R32 % P).to(torch.int32).contiguous()
        gx = -(-(cols // 4) // QUAD_THREADS)
        chunk = -(-Q // max(1, min(Q, QUAD_SPLITS)))
        S = -(-Q // chunk)
        partial = torch.empty((S, D, cols), dtype=torch.int32, device=self.device)
        args = [self._cp(rho32), self._cp(X)]
        for side in ("a", "b", "c"):
            args += [self._cp(x) for x in mt[side]]
        args += [self._cp(partial), np.int32(Q), np.int32(cols), np.int32(chunk), np.int32(1 if QUAD_QFAST else 0)]
        grid = (S, gx) if QUAD_QFAST else (gx, S)
        with self._stream():
            self.k["quad_v4_u32"](grid, (QUAD_THREADS,), tuple(args), shared_mem=D * chunk * 4)
        return self._reduce(partial)


def csr_tables(gen: dict, Q: int, device) -> dict:
    """The (qq, jj, cc, consts) triples of witness._triples for the general constraints, as CSR (rows sorted by q)."""
    out = {"Q": Q}
    for side in ("a", "b", "c"):
        qq, jj, cc, consts = gen[side]
        order = torch.argsort(qq, stable=True)
        qq_s = qq[order]
        counts = torch.bincount(qq_s, minlength=Q)
        ptr = torch.zeros(Q + 1, dtype=torch.int64, device=device)
        ptr[1:] = torch.cumsum(counts, 0)
        out[side] = {"ptr": ptr.to(torch.int32).contiguous(), "j": jj[order].to(torch.int32).contiguous(),
                     "coef": (cc[order] % P).to(torch.int32).contiguous(), "const": (consts % P).to(torch.int32).contiguous()}
    return out


_FT: dict = {}


def fused_for(D: int, device) -> FusedTests | None:
    """The kernels for this D on this CUDA device, or None (CPU / no cupy)."""
    if not str(device).startswith("cuda") or not available():
        return None
    key = (D, str(device))
    f = _FT.get(key)
    if f is None:
        try:
            f = FusedTests(D, device)
        except Exception:  # noqa: BLE001 -- no NVRTC: the torch path
            return None
        _FT[key] = f
    return f
