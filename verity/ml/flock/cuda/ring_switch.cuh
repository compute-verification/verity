// The ring switch's row kernels over eq halves (eq_split), included by prove_chunk.cuh; tests/ring_switch.cu builds them alone
// against a Flock b684b12 checkout's cuda-ghash.
#pragma once
#include <cstdint>
#include "f128.cuh"

// A 2^d eq table's entry x from its halves: eq[x] = lo[x mod 2^k] * hi[x >> k] (build_eq_device over the first k challenges and
// the rest), the same field element as the doubling's
__device__ __forceinline__ F128 chunk_eq_split(const F128* __restrict__ lo, const F128* __restrict__ hi, int k, long long x) {
    return ghash_mul_karatsuba(lo[x & ((1LL << k) - 1)], hi[x >> k]);
}

// chunk_ring_fold_rows with its suffix as eq halves (no 2^d table): a warp takes 32 rows at a time, each lane one row's witness
// word and eq entry, and folds all 32 through shuffles
__global__ void chunk_ring_fold_rows_split(const F128* __restrict__ witness, const F128* __restrict__ lo, const F128* __restrict__ hi,
                                           int k, long long len, int chunks, F128* __restrict__ partials) {
    __shared__ F128 red[8][128];   // launched at 256 threads
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5, sh = 4 * (lane & 15);
    const long long per = (len + chunks - 1) / chunks, r0 = (long long)blockIdx.x * per, r1 = min(len, r0 + per);
    F128 acc[4] = {};
    for (long long base = r0 + 32LL * wid; base < r1; base += 32LL * nw) {
        const long long i = base + lane;
        F128 w{0, 0}, s{0, 0};
        if (i < r1) { w = witness[i]; s = chunk_eq_split(lo, hi, k, i); }
#pragma unroll 4
        for (int j = 0; j < 32; j++) {
            const u64 wl = __shfl_sync(0xffffffffu, w.lo, j), wh = __shfl_sync(0xffffffffu, w.hi, j);
            const u64 sl = __shfl_sync(0xffffffffu, s.lo, j), shi = __shfl_sync(0xffffffffu, s.hi, j);
            const unsigned nib = (unsigned)(((lane < 16 ? wl : wh) >> sh) & 15);
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const u64 m = 0ull - (u64)((nib >> b) & 1);
                acc[b].lo ^= sl & m;
                acc[b].hi ^= shi & m;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 4; b++) red[wid][4 * lane + b] = acc[b];
    __syncthreads();
    for (int t = threadIdx.x; t < 128; t += blockDim.x) {
        F128 x{0, 0};
        for (int q = 0; q < nw; q++) { x.lo ^= red[q][t].lo; x.hi ^= red[q][t].hi; }
        partials[(long long)t * chunks + blockIdx.x] = x;
    }
}

// Both claims' 128 weights as 32 tables each of their nibbles' sums: tab[c][p][v] = XOR of w_c[4p + b] over v's set bits b
__device__ __forceinline__ void ring_weight_tables(const F128* __restrict__ weights_ab, const F128* __restrict__ weights_c,
                                                   F128 (*tab)[32][16]) {
    for (int e = threadIdx.x; e < 2 * 32 * 16; e += blockDim.x) {
        const int c = e >> 9, p = (e >> 4) & 31, v = e & 15;
        const F128* w = c ? weights_c : weights_ab;
        F128 x{0, 0};
        for (int b = 0; b < 4; b++)
            if ((v >> b) & 1) { x.lo ^= w[4 * p + b].lo; x.hi ^= w[4 * p + b].hi; }
        tab[c][p][v] = x;
    }
}

// Σ_j v_j·w_c[j] by the tables: 32 lookups
__device__ __forceinline__ F128 ring_weigh(const F128 (*tab)[32][16], int c, F128 v) {
    F128 acc{0, 0};
#pragma unroll
    for (int p = 0; p < 32; p++) {
        const F128 x = tab[c][p][((p < 16 ? v.lo : v.hi) >> (4 * (p & 15))) & 15];
        acc.lo ^= x.lo;
        acc.hi ^= x.hi;
    }
    return acc;
}

// ring_switch_combine_basis's sums, the same XORs, with both suffixes as eq halves: each claim's 128 weights as 32 tables of their
// nibbles' sums in shared memory, so a row is 64 lookups rather than a walk over its set bits
__global__ void chunk_ring_combine_basis_split(const F128* __restrict__ ab_lo, const F128* __restrict__ ab_hi,
                                               const F128* __restrict__ c_lo, const F128* __restrict__ c_hi, int k,
                                               const F128* __restrict__ weights_ab, const F128* __restrict__ weights_c, long long len,
                                               F128* __restrict__ combined) {
    __shared__ F128 tab[2][32][16];
    ring_weight_tables(weights_ab, weights_c, tab);
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += (long long)gridDim.x * blockDim.x) {
        const F128 a = ring_weigh(tab, 0, chunk_eq_split(ab_lo, ab_hi, k, i)), c = ring_weigh(tab, 1, chunk_eq_split(c_lo, c_hi, k, i));
        combined[i] = F128{a.lo ^ c.lo, a.hi ^ c.hi};
    }
}

// m16n8k256 b1 AND-popcount: d[r][n] = popc(row r of a & column n of b), with the PTX ISA's fragment layout (lane = 4g + q:
// a0/a2 row g, a1/a3 row g + 8, a0/a1 bits 32q.. of k, a2/a3 bits 128 + 32q..; b0/b1 column g, the same bits; d0/d1 row g,
// columns 2q and 2q + 1, d2/d3 row g + 8)
__device__ __forceinline__ void ring_bmma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
#else
    __trap();
#endif
}

// A warp's 32x32 bit transpose: lane l returns bit l of every lane's x, lane r's in bit r
__device__ __forceinline__ uint32_t ring_transpose32(uint32_t x, int lane) {
#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int s = 16 >> i;
        const uint32_t m = i == 0 ? 0x0000FFFFu : i == 1 ? 0x00FF00FFu : i == 2 ? 0x0F0F0F0Fu : i == 3 ? 0x33333333u : 0x55555555u;
        const bool hi = lane & s;
        const uint32_t got = __shfl_xor_sync(0xffffffffu, hi ? (x & m) << s : (x >> s) & m, s);
        x = (x & (hi ? ~m : m)) | got;
    }
    return x;
}

// chunk_ring_combine_basis_split's rows, the same field elements, as a GF(2) product on the tensor cores. Row i = (b, a), with
// a = i mod 2^k, is M_ab(lo_ab[a]·hi_ab[b]) + M_c(lo_c[a]·hi_c[b]) for M_c(v) = Σ_j v_j·w_c[j], and both are linear in the lo
// entry: Z_b·(lo_ab[a] ‖ lo_c[a]) over GF(2), where Z_b's column j < 128 is M_ab(x^j·hi_ab[b]) and column 128 + j is
// M_c(x^j·hi_c[b]). A block builds Z_b's 128 rows of 256 bits in shared memory, one column a thread; each warp then takes 16 rows
// a at a time as an MMA's A fragment, and sixteen MMAs, eight output bits each, give every output bit's popcount, whose parity is
// the bit. B's column g of tile t is output bit 32(g >> 1) + 2t + (g & 1), so lane 4g + q's sixteen tiles hold word q of rows g
// and g + 8 in bit order, shifted in one bit at a time. Needs k >= 4 (an MMA's 16 rows share b), nb = len >> k, and sm_80 on.
__global__ void __launch_bounds__(256) chunk_ring_combine_basis_bmma(
    const F128* __restrict__ ab_lo, const F128* __restrict__ ab_hi, const F128* __restrict__ c_lo, const F128* __restrict__ c_hi,
    int k, const F128* __restrict__ weights_ab, const F128* __restrict__ weights_c, long long nb, F128* __restrict__ combined) {
    __shared__ F128 tab[2][32][16];
    __shared__ uint32_t zt[128][8];   // Z_b's row n: bit l of word w is column 32w + l
    ring_weight_tables(weights_ab, weights_c, tab);
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    const int c = threadIdx.x >> 7, j = threadIdx.x & 127;   // this thread's column 32·wid + lane: claim c's x^j
    const F128 xj = j < 64 ? F128{1ull << j, 0} : F128{0, 1ull << (j - 64)};
    const uint32_t* la = reinterpret_cast<const uint32_t*>(ab_lo);
    const uint32_t* lc = reinterpret_cast<const uint32_t*>(c_lo);
    const int rows = 1 << k;
    for (long long b = blockIdx.x; b < nb; b += gridDim.x) {
        const F128 col = ring_weigh(tab, c, ghash_mul_karatsuba(xj, c ? c_hi[b] : ab_hi[b]));
        const uint32_t cw[4] = {(uint32_t)col.lo, (uint32_t)(col.lo >> 32), (uint32_t)col.hi, (uint32_t)(col.hi >> 32)};
        uint32_t mine[4];   // row 32h + lane's word wid
#pragma unroll
        for (int h = 0; h < 4; h++) mine[h] = ring_transpose32(cw[h], lane);
        __syncthreads();   // every warp has read the previous b's rows
#pragma unroll
        for (int h = 0; h < 4; h++) zt[32 * h + lane][wid] = mine[h];
        __syncthreads();
        uint32_t bf[16][2];
#pragma unroll
        for (int t = 0; t < 16; t++) {
            const int n = 32 * (g >> 1) + 2 * t + (g & 1);
            bf[t][0] = zt[n][q];
            bf[t][1] = zt[n][4 + q];
        }
        uint32_t* out = reinterpret_cast<uint32_t*>(combined + (b << k));
        const int step = 16 * (blockDim.x >> 5);
        int r = 16 * wid + g;
        uint32_t a[4] = {};
        if (r < rows) { a[0] = la[4 * r + q]; a[1] = la[4 * (r + 8) + q]; a[2] = lc[4 * r + q]; a[3] = lc[4 * (r + 8) + q]; }
        for (; r < rows; r += step) {
            const int rn = r + step;
            uint32_t an[4] = {};
            if (rn < rows) { an[0] = la[4 * rn + q]; an[1] = la[4 * (rn + 8) + q]; an[2] = lc[4 * rn + q]; an[3] = lc[4 * (rn + 8) + q]; }
            uint32_t w0 = 0, w1 = 0;
#pragma unroll
            for (int t0 = 0; t0 < 16; t0 += 4) {
                uint32_t d[4][4];
#pragma unroll
                for (int u = 0; u < 4; u++) ring_bmma(a[0], a[1], a[2], a[3], bf[t0 + u][0], bf[t0 + u][1], d[u]);
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    w0 = __funnelshift_r(w0, d[u][0], 1);
                    w0 = __funnelshift_r(w0, d[u][1], 1);
                    w1 = __funnelshift_r(w1, d[u][2], 1);
                    w1 = __funnelshift_r(w1, d[u][3], 1);
                }
            }
            out[4 * r + q] = w0;
            out[4 * (r + 8) + q] = w1;
#pragma unroll
            for (int h = 0; h < 4; h++) a[h] = an[h];
        }
    }
}
