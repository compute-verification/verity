// Ligerito level 0's folds, each with the round message that follows it (ligerito_f256.cuh's lf_fold_base_pair or
// lf_fold_ext_pair, then msg256's lf_msg_partial over the fold's outputs), for verity's cuda_circuit_patch.py: a thread folds
// the inputs of two adjacent outputs, writes both, and adds their pair's message terms, so the outputs are not read back. The
// first fold can also add level 0's last OOD glue, beta * eq(z) over the split eq tables, to B's inputs as it reads them
// (combine_basis_polynomials_split's), which then never writes B. Block partials grouped any way sum to the same element
// (characteristic 2), so lf_msg_combine's u_0 and u_2 over them are msg256's. Included by ligerito_f256.cuh after LF_TPB.
#pragma once

// a glue deferred into the first fold (none while lo is null): B[j] += beta * lo[j mod 2^k] * hi[j >> k]
struct LfGlue {
    F128* lo = nullptr;
    F128* hi = nullptr;
    int k = 0;
    F128 beta{};
};

// eq(z) at x from its split tables (cuda_circuit_patch.py's igl_eq_split)
__device__ __forceinline__ F128 lf_eq_split(const F128* __restrict__ lo, const F128* __restrict__ hi, int k, long long x) {
    return ghash_mul_karatsuba(lo[x & ((1LL << k) - 1)], hi[x >> k]);
}

// lf_fold_base_pair's output from base inputs a0 = in[2j], a1 = in[2j + 1]
__device__ __forceinline__ F256Ext lf_fold_base_one(F128 a0, F128 a1, F256Ext r) {
    const F128 x = f128_add(a0, a1);
    return F256Ext{f128_add(a0, ghash_mul_karatsuba(r.c0, x)), ghash_mul_karatsuba(r.c1, x)};
}

// lf_fold_ext_pair's output
__device__ __forceinline__ F256Ext lf_fold_ext_one(F256Ext a0, F256Ext a1, F256Ext r) {
    return f256x_add(a0, f256x_mul(r, f256x_add(a0, a1)));
}

// lf_msg_partial's terms of the output pair (a0, a1), (b0, b1)
__device__ __forceinline__ void lf_msg_terms(F256Ext a0, F256Ext a1, F256Ext b0, F256Ext b1, F256Ext& e0, F256Ext& e2) {
    e0 = f256x_add(e0, f256x_mul(a0, b0));
    e2 = f256x_add(e2, f256x_mul(f256x_add(a0, a1), f256x_add(b0, b1)));
}

// the block's sums of e0 and e2 into p0, p2 at its index (lf_msg_partial's reduction); every thread of the block calls it
__device__ __forceinline__ void lf_msg_block(F256Ext e0, F256Ext e2, F256Ext* p0, F256Ext* p2) {
    __shared__ F256Ext s0[LF_TPB];
    __shared__ F256Ext s2[LF_TPB];
    const int x = threadIdx.x;
    s0[x] = e0;
    s2[x] = e2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (x < s) { s0[x] = f256x_add(s0[x], s0[x + s]); s2[x] = f256x_add(s2[x], s2[x + s]); }
        __syncthreads();
    }
    if (x == 0) { p0[blockIdx.x] = s0[0]; p2[blockIdx.x] = s2[0]; }
}

// The first fold of A and B (4 pairs words each, B's with g's glue) into Ao and Bo (2 pairs each), and its message's block
// partials; LF_TPB threads a block.
__global__ void lf_fold_base_pair_msg(const F128* __restrict__ A, const F128* __restrict__ B, LfGlue g, F256Ext* __restrict__ Ao,
                                      F256Ext* __restrict__ Bo, long long pairs, F256Ext r, F256Ext* p0, F256Ext* p2) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    F256Ext e0 = f256x_zero(), e2 = f256x_zero();
    for (long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x; m < pairs; m += stride) {
        F128 b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            b[i] = B[4 * m + i];
            if (g.lo) b[i] = f128_add(b[i], ghash_mul_karatsuba(g.beta, lf_eq_split(g.lo, g.hi, g.k, 4 * m + i)));
        }
        const F256Ext a0 = lf_fold_base_one(A[4 * m], A[4 * m + 1], r), a1 = lf_fold_base_one(A[4 * m + 2], A[4 * m + 3], r);
        const F256Ext b0 = lf_fold_base_one(b[0], b[1], r), b1 = lf_fold_base_one(b[2], b[3], r);
        Ao[2 * m] = a0;
        Ao[2 * m + 1] = a1;
        Bo[2 * m] = b0;
        Bo[2 * m + 1] = b1;
        lf_msg_terms(a0, a1, b0, b1, e0, e2);
    }
    lf_msg_block(e0, e2, p0, p2);
}

// A later fold of A and B (4 pairs elements each) into Ao and Bo (2 pairs each), and its message's block partials; LF_TPB
// threads a block.
__global__ void lf_fold_ext_pair_msg(const F256Ext* __restrict__ A, const F256Ext* __restrict__ B, F256Ext* __restrict__ Ao,
                                     F256Ext* __restrict__ Bo, long long pairs, F256Ext r, F256Ext* p0, F256Ext* p2) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    F256Ext e0 = f256x_zero(), e2 = f256x_zero();
    for (long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x; m < pairs; m += stride) {
        const F256Ext a0 = lf_fold_ext_one(A[4 * m], A[4 * m + 1], r), a1 = lf_fold_ext_one(A[4 * m + 2], A[4 * m + 3], r);
        const F256Ext b0 = lf_fold_ext_one(B[4 * m], B[4 * m + 1], r), b1 = lf_fold_ext_one(B[4 * m + 2], B[4 * m + 3], r);
        Ao[2 * m] = a0;
        Ao[2 * m + 1] = a1;
        Bo[2 * m] = b0;
        Bo[2 * m + 1] = b1;
        lf_msg_terms(a0, a1, b0, b1, e0, e2);
    }
    lf_msg_block(e0, e2, p0, p2);
}
