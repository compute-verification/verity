// attn_ref: attention's two BF16 GEMMs on sm_90a (H100, H200), slow and exact, the fallback beside attn_bf16.  Every
// output word is GemmCoordinate_v2{K, GEMM_HOPPER} (verity.ml.circuits.ops.gemm): K/16 steps of
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (sm90.mma.m16n8k16.bf16, pinned to HOPPER_BF16_M16N8K16) in
// ascending k from a +0 accumulator, then one `cvt.rn.bf16.f32`.
//
//   attn_ref_qk   S[b,r,j] = GemmCoordinate{K=dim}(Q[b,r,:], K[b,j,:])                     j < len(b); fill for len(b) <= j < lcap
//   attn_ref_pv   O[b,r,d] = GemmCoordinate{K=pad16(len)}(P[b,r,:len] ++ 0, V[b,:len,d] ++ 0)   len = len(b)
//
// b is a (sequence, KV head) pair and r one of the query heads that read that KV head (attn_ref.py flattens MQA and GQA to
// this); len(b) = lens[b] clamped to [0, lcap].  Every buffer is contiguous: Q [nb][rows][dim], K and V [nb][lcap][dim],
// P and S [nb][rows][lcap], O [nb][rows][dim]; dim is a multiple of 16.  Positions past len(b) in K, V and P are never
// multiplied: they enter the mma as zero codes, as the Definition pads both operands.
//
// One warp owns one 16 x 8 tile of output words (16 query heads by 8 keys, or by 8 d) and carries all its accumulators
// through the whole k range in registers: no split-K, no atomics, no shared memory.  The grid is (nb, row tiles / warps
// a block, column tiles); warp w of block (x, y, z) owns row tile y * warps + w, so the launch shape decides only which
// warp runs a tile, never a word's steps.  A word's products depend only on its row of A and its column of B, so the
// zeroed rows and columns of a partial tile change no live word.
//
// Operands are gathered one 16-bit word at a time and packed, lower k in the low half (PTX's m16n8k16 fragment
// layout); nothing does floating-point arithmetic but the mma and the cvt (sass_gate.py refuses any other instruction).

#include <stdint.h>

namespace {

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint16_t cvt_bf16(float x) {
    uint16_t r;
    asm volatile("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ uint32_t word(const uint16_t* p, bool ok) { return ok ? (uint32_t)*p : 0u; }

__device__ __forceinline__ uint32_t pack(uint32_t lo, uint32_t hi) { return lo | hi << 16; }

struct Tile {
    int lane, g, t, r0, c0;
    int64_t b;
};

__device__ __forceinline__ Tile tile() {
    Tile w;
    w.lane = threadIdx.x & 31;
    w.g = w.lane >> 2;
    w.t = w.lane & 3;
    w.r0 = 16 * ((int)blockIdx.y * (int)(blockDim.x >> 5) + (int)(threadIdx.x >> 5));
    w.c0 = 8 * (int)blockIdx.z;
    w.b = blockIdx.x;
    return w;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) attn_ref_qk(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                                                              uint16_t* __restrict__ s, const int32_t* __restrict__ lens,
                                                              int32_t rows, int32_t dim, int32_t lcap, uint32_t fill) {
    const Tile w = tile();
    if (w.r0 >= rows) return;
    const int len = min(max(lens[w.b], 0), lcap);
    const uint16_t* Q = q + w.b * rows * dim;
    const uint16_t* K = k + w.b * lcap * dim;
    uint16_t* S = s + w.b * rows * lcap;
    const int ra = w.r0 + w.g, rb = ra + 8, key = w.c0 + w.g;
    const bool aok = ra < rows, bok = rb < rows, kok = key < len;
    const uint16_t* qa = Q + (int64_t)ra * dim;
    const uint16_t* qb = Q + (int64_t)rb * dim;
    const uint16_t* kr = K + (int64_t)key * dim;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < dim; k0 += 16) {
        const int lo = k0 + 2 * w.t, hi = lo + 8;
        const uint32_t A[4] = {pack(word(qa + lo, aok), word(qa + lo + 1, aok)), pack(word(qb + lo, bok), word(qb + lo + 1, bok)),
                               pack(word(qa + hi, aok), word(qa + hi + 1, aok)), pack(word(qb + hi, bok), word(qb + hi + 1, bok))};
        const uint32_t B[2] = {pack(word(kr + lo, kok), word(kr + lo + 1, kok)), pack(word(kr + hi, kok), word(kr + hi + 1, kok))};
        mma_bf16(acc, A, B);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int r = e < 2 ? ra : rb, j = w.c0 + 2 * w.t + (e & 1);
        if (r < rows && j < lcap) S[(int64_t)r * lcap + j] = j < len ? cvt_bf16(acc[e]) : (uint16_t)fill;
    }
}

extern "C" __global__ void __launch_bounds__(256) attn_ref_pv(const uint16_t* __restrict__ p, const uint16_t* __restrict__ v,
                                                              uint16_t* __restrict__ o, const int32_t* __restrict__ lens,
                                                              int32_t rows, int32_t dim, int32_t lcap) {
    const Tile w = tile();
    if (w.r0 >= rows) return;
    const int len = min(max(lens[w.b], 0), lcap);
    const uint16_t* P = p + w.b * rows * lcap;
    const uint16_t* V = v + w.b * lcap * dim;
    uint16_t* O = o + w.b * rows * dim;
    const int ra = w.r0 + w.g, rb = ra + 8, d = w.c0 + w.g;
    const bool aok = ra < rows, bok = rb < rows;
    const uint16_t* pa = P + (int64_t)ra * lcap;
    const uint16_t* pb = P + (int64_t)rb * lcap;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < len; k0 += 16) {
        const int lo = k0 + 2 * w.t, hi = lo + 8;
        const bool l0 = lo < len, l1 = lo + 1 < len, h0 = hi < len, h1 = hi + 1 < len;
        const uint32_t A[4] = {pack(word(pa + lo, aok && l0), word(pa + lo + 1, aok && l1)),
                               pack(word(pb + lo, bok && l0), word(pb + lo + 1, bok && l1)),
                               pack(word(pa + hi, aok && h0), word(pa + hi + 1, aok && h1)),
                               pack(word(pb + hi, bok && h0), word(pb + hi + 1, bok && h1))};
        const uint32_t B[2] = {pack(word(V + (int64_t)lo * dim + d, l0), word(V + (int64_t)(lo + 1) * dim + d, l1)),
                               pack(word(V + (int64_t)hi * dim + d, h0), word(V + (int64_t)(hi + 1) * dim + d, h1))};
        mma_bf16(acc, A, B);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int r = e < 2 ? ra : rb, c = w.c0 + 2 * w.t + (e & 1);
        if (r < rows) O[(int64_t)r * dim + c] = cvt_bf16(acc[e]);
    }
}
