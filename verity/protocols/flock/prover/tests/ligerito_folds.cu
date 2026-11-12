// cuda/ligerito_folds.cuh against upstream's level-0 folds and messages (cuda-ghash/ligerito_f256.cuh's lf_fold_base_pair,
// lf_fold_ext_pair, lf_msg_partial and lf_msg_combine, after introduce_glue.cuh's glue over the whole eq table), for
// test_ligerito_folds.py. lf_check draws A and B (2^log_n base words each), split eq tables at log_n / 2, beta and a coin a
// fold from seed, runs `folds` folds both ways, the first with the glue when `glue` is set, and counts the words that differ:
// bad[0] of the folds' outputs, bad[1] of their messages.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "ligerito_f256.cuh"
#include "../cuda/ligerito_folds.cuh"

__host__ __device__ inline uint64_t lc_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__global__ void lc_fill(uint64_t* p, long long n, uint64_t seed) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        p[i] = lc_mix(seed ^ lc_mix((uint64_t)i));
}

__global__ void lc_eq_full(const F128* lo, const F128* hi, int k, long long n, F128* eq) {
    for (long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x; x < n; x += (long long)gridDim.x * blockDim.x)
        eq[x] = ghash_mul_karatsuba(lo[x & ((1LL << k) - 1)], hi[x >> k]);
}

__global__ void lc_diff(const uint64_t* a, const uint64_t* b, long long n, unsigned long long* bad) {
    unsigned long long k = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        k += a[i] != b[i];
    atomicAdd(bad, k);
}

static int lc_blocks(long long items) {
    long long b = (items + LF_TPB - 1) / LF_TPB;
    return (int)(b < 1 ? 1 : b > LF_MAX_BLOCKS ? LF_MAX_BLOCKS : b);
}

extern "C" int lf_check(int log_n, int folds, int glue, unsigned long long seed, unsigned long long* bad) {
    const long long n = 1LL << log_n;
    const int k = log_n / 2;
    if (log_n < 3 || folds < 1 || folds > log_n - 2) return 1;
    F128 *A = nullptr, *B = nullptr, *Bref = nullptr, *lo = nullptr, *hi = nullptr, *eq = nullptr;
    F256Ext *out[2][4] = {}, *p0 = nullptr, *p2 = nullptr, *msg = nullptr;
    unsigned long long* d_bad = nullptr;
    cudaError_t e = cudaMalloc(&A, n * sizeof(F128));
    if (e == cudaSuccess) e = cudaMalloc(&B, n * sizeof(F128));
    if (e == cudaSuccess) e = cudaMalloc(&Bref, n * sizeof(F128));
    if (e == cudaSuccess) e = cudaMalloc(&lo, ((size_t)1 << k) * sizeof(F128));
    if (e == cudaSuccess) e = cudaMalloc(&hi, ((size_t)1 << (log_n - k)) * sizeof(F128));
    if (e == cudaSuccess) e = cudaMalloc(&eq, n * sizeof(F128));
    for (int w = 0; w < 2; w++)
        for (int i = 0; i < 4 && e == cudaSuccess; i++) e = cudaMalloc(&out[w][i], (n / 2) * sizeof(F256Ext));
    if (e == cudaSuccess) e = cudaMalloc(&p0, LF_MAX_BLOCKS * sizeof(F256Ext));
    if (e == cudaSuccess) e = cudaMalloc(&p2, LF_MAX_BLOCKS * sizeof(F256Ext));
    if (e == cudaSuccess) e = cudaMalloc(&msg, 4 * sizeof(F256Ext));
    if (e == cudaSuccess) e = cudaMalloc(&d_bad, 2 * sizeof(unsigned long long));
    if (e == cudaSuccess) e = cudaMemset(d_bad, 0, 2 * sizeof(unsigned long long));
    if (e == cudaSuccess) {
        lc_fill<<<256, 256>>>((uint64_t*)A, 2 * n, seed);
        lc_fill<<<256, 256>>>((uint64_t*)B, 2 * n, seed + 1);
        lc_fill<<<64, 256>>>((uint64_t*)lo, 2LL << k, seed + 2);
        lc_fill<<<64, 256>>>((uint64_t*)hi, 2LL << (log_n - k), seed + 3);
        lc_eq_full<<<256, 256>>>(lo, hi, k, n, eq);
        e = cudaMemcpy(Bref, B, n * sizeof(F128), cudaMemcpyDeviceToDevice);
    }
    const F128 beta{lc_mix(seed + 4), lc_mix(seed + 5)};
    long long len = n;
    for (int j = 0; j < folds && e == cudaSuccess; j++) {
        const F256Ext r{{lc_mix(seed + 6 + 4 * j), lc_mix(seed + 7 + 4 * j)}, {lc_mix(seed + 8 + 4 * j), lc_mix(seed + 9 + 4 * j)}};
        const long long half = len / 2, pairs = half / 2;
        const int fo = 2 * (j & 1), fi = 2 - fo, blocks = lc_blocks(pairs);
        F256Ext *ra = out[0][fo], *rb = out[0][fo + 1], *fa = out[1][fo], *fb = out[1][fo + 1];
        if (j == 0) {
            if (glue) launch_glue(Bref, eq, beta, n);
            lf_fold_base_pair<<<(unsigned)((half + LF_TPB - 1) / LF_TPB), LF_TPB>>>(A, Bref, ra, rb, half, r);
        } else {
            lf_fold_ext_pair<<<(unsigned)((half + LF_TPB - 1) / LF_TPB), LF_TPB>>>(out[0][fi], out[0][fi + 1], ra, rb, half, r);
        }
        lf_msg_partial<<<blocks, LF_TPB>>>(ra, rb, pairs, p0, p2);
        lf_msg_combine<<<1, LF_TPB>>>(p0, p2, blocks, msg);
        if (j == 0) {
            LfGlue g;
            if (glue) g = LfGlue{lo, hi, k, beta};
            lf_fold_base_pair_msg<<<blocks, LF_TPB>>>(A, B, g, fa, fb, pairs, r, p0, p2);
        } else {
            lf_fold_ext_pair_msg<<<blocks, LF_TPB>>>(out[1][fi], out[1][fi + 1], fa, fb, pairs, r, p0, p2);
        }
        lf_msg_combine<<<1, LF_TPB>>>(p0, p2, blocks, msg + 2);
        lc_diff<<<256, 256>>>((const uint64_t*)ra, (const uint64_t*)fa, 4 * half, d_bad);
        lc_diff<<<256, 256>>>((const uint64_t*)rb, (const uint64_t*)fb, 4 * half, d_bad);
        lc_diff<<<1, 32>>>((const uint64_t*)msg, (const uint64_t*)(msg + 2), 8, d_bad + 1);
        e = cudaGetLastError();
        len = half;
    }
    if (e == cudaSuccess) e = cudaMemcpy(bad, d_bad, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for (F128* p : {A, B, Bref, lo, hi, eq}) cudaFree(p);
    for (int w = 0; w < 2; w++)
        for (int i = 0; i < 4; i++) cudaFree(out[w][i]);
    cudaFree(p0); cudaFree(p2); cudaFree(msg); cudaFree(d_bad);
    return e == cudaSuccess ? 0 : 2;
}
