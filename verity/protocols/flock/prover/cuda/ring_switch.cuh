// The ring switch's row kernels over eq halves (eq_split), included by prove_chunk.cuh; tests/ring_switch.cu builds them alone
// against a Flock b684b12 checkout's cuda-ghash.
#pragma once
#include <algorithm>
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

// ring_bmma accumulating into d
__device__ __forceinline__ void ring_bmma_acc(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// v's four 32-bit words, low first: word h holds bits 32h..32h+31
__device__ __forceinline__ void ring_words(F128 v, uint32_t* w) {
    w[0] = (uint32_t)v.lo; w[1] = (uint32_t)(v.lo >> 32); w[2] = (uint32_t)v.hi; w[3] = (uint32_t)(v.hi >> 32);
}

// A warp's 32x32 bit transpose: lane l returns bit l of every lane's x, lane r's in bit r
__device__ __forceinline__ uint32_t ring_transpose32(uint32_t x, int lane) {
    // stages 16 and 8 move whole bytes, so each is one permute of the lane's word and its partner's: the low lane keeps its
    // m-bytes with the partner's m-bytes above them, the high lane its other bytes with the partner's below them
    x = __byte_perm(x, __shfl_xor_sync(0xffffffffu, x, 16), lane & 16 ? 0x3276 : 0x5410);
    x = __byte_perm(x, __shfl_xor_sync(0xffffffffu, x, 8), lane & 8 ? 0x3715 : 0x6240);
#pragma unroll
    for (int i = 2; i < 5; i++) {
        const int s = 16 >> i;
        const uint32_t m = i == 2 ? 0x0F0F0F0Fu : i == 3 ? 0x33333333u : 0x55555555u;
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
    const int rows = 1 << k, step = 16 * (blockDim.x >> 5);
    for (long long b = blockIdx.x; b < nb; b += gridDim.x) {
        const F128 col = ring_weigh(tab, c, ghash_mul_karatsuba(xj, c ? c_hi[b] : ab_hi[b]));
        uint32_t cw[4];
        ring_words(col, cw);
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

// chunk_ring_fold_rows_split's sums, the same field elements, as GF(2) products on the tensor cores. Σ_i w_i[t]·lo[a]·hi[b] over
// rows i = (b, a) is Σ_b hi[b]·Y_b[t] with Y_b[t] = Σ_a w_(b,a)[t]·lo[a], and Y_b as a 128x128 bit matrix is W_b^T·LO over GF(2),
// summed over the rows a. A block takes RING_FOLD_ROWS rows a (all of them when there are fewer) and a share of the b's. It
// transposes its lo entries into shared memory once (row u's word w holds bit u of 32 entries, words swizzled so a warp's B
// fragments hit distinct banks); then per b and 256 rows a it transposes the witness rows the same way, and warp w accumulates
// Y_b's rows t in [16w, 16w + 16) as one m16n8k256 A fragment against sixteen B fragments of LO, with column g of tile n being
// eq bit 32(g >> 1) + 2n + (g & 1) so lane 4g + q's parities make word q of rows g and g + 8. Each row's four words meet in one
// lane of the quad, which adds the row times hi[b]. partials[t·gridDim.x + block] as chunk_ring_fold_rows_split's, launched at
// ring_fold_bmma_blocks blocks; needs k >= 8, nb = len >> k and sm_80 on.
constexpr int RING_FOLD_ROWS = 2048;

inline int ring_fold_bmma_blocks(int k, long long nb, int max_blocks) {
    const int n_ar = (1 << k) / std::min(1 << k, RING_FOLD_ROWS);
    return n_ar * (int)std::min<long long>(nb, std::max(1, max_blocks / n_ar));
}

__device__ __forceinline__ int ring_lot_at(int u, int w) { return u * (RING_FOLD_ROWS / 32) + (w ^ ((u & 1) << 2 | ((u >> 5) & 3) << 3)); }

__global__ void __launch_bounds__(256) chunk_ring_fold_rows_bmma(const F128* __restrict__ witness, const F128* __restrict__ lo,
                                                                 const F128* __restrict__ hi, int k, long long nb,
                                                                 F128* __restrict__ partials) {
    __shared__ uint32_t lot[128 * (RING_FOLD_ROWS / 32)];
    __shared__ uint32_t wt[2][128][9];   // a step's witness rows transposed: row t's word G holds bit t of rows 32G..32G+31
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    const int rows = min(1 << k, RING_FOLD_ROWS), n_ar = (1 << k) / rows, steps = rows / 256;
    const int a0 = (blockIdx.x % n_ar) * rows, n_bg = gridDim.x / n_ar;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        uint32_t v[4];
        ring_words(lo[a0 + r], v);
#pragma unroll
        for (int h = 0; h < 4; h++) lot[ring_lot_at(32 * h + lane, r >> 5)] = ring_transpose32(v[h], lane);
    }
    __syncthreads();
    uint32_t acc[16][4] = {};
    F128 y_acc{0, 0};   // lanes q < 2: Σ_b hi[b]·Y_b[16·wid + g + 8q]
    long long b = blockIdx.x / n_ar;
    int s = 0, n_step = 0;
    F128 v = b < nb ? witness[(b << k) + a0 + threadIdx.x] : F128{0, 0};
    while (b < nb) {
        const long long bn = s + 1 < steps ? b : b + n_bg;
        const int sn = s + 1 < steps ? s + 1 : 0;
        const F128 vn = bn < nb ? witness[(bn << k) + a0 + 256 * sn + threadIdx.x] : F128{0, 0};
        uint32_t vw[4], tw[4];
        ring_words(v, vw);
#pragma unroll
        for (int h = 0; h < 4; h++) tw[h] = ring_transpose32(vw[h], lane);
        // alternating buffers: every read of the one written here came before the last step's barrier
        uint32_t (*cur)[9] = wt[n_step & 1];
#pragma unroll
        for (int h = 0; h < 4; h++) cur[32 * h + lane][wid] = tw[h];
        __syncthreads();
        const int t0 = 16 * wid + g;
        const uint32_t fa0 = cur[t0][q], fa1 = cur[t0 + 8][q], fa2 = cur[t0][4 + q], fa3 = cur[t0 + 8][4 + q];
#pragma unroll
        for (int n = 0; n < 16; n++) {
            const int u = 32 * (g >> 1) + 2 * n + (g & 1);
            ring_bmma_acc(fa0, fa1, fa2, fa3, lot[ring_lot_at(u, 8 * s + q)], lot[ring_lot_at(u, 8 * s + 4 + q)], acc[n]);
        }
        if (s == steps - 1) {
            uint32_t w0 = 0, w1 = 0;
#pragma unroll
            for (int n = 0; n < 16; n++) {
                w0 = __funnelshift_r(w0, acc[n][0], 1);
                w0 = __funnelshift_r(w0, acc[n][1], 1);
                w1 = __funnelshift_r(w1, acc[n][2], 1);
                w1 = __funnelshift_r(w1, acc[n][3], 1);
#pragma unroll
                for (int e = 0; e < 4; e++) acc[n][e] = 0;
            }
            uint32_t y[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t x0 = __shfl_sync(0xffffffffu, w0, 4 * g + j), x1 = __shfl_sync(0xffffffffu, w1, 4 * g + j);
                y[j] = q == 0 ? x0 : x1;
            }
            if (q < 2) {
                const F128 p = ghash_mul_karatsuba(F128{(u64)y[0] | (u64)y[1] << 32, (u64)y[2] | (u64)y[3] << 32}, hi[b]);
                y_acc.lo ^= p.lo;
                y_acc.hi ^= p.hi;
            }
        }
        b = bn;
        s = sn;
        v = vn;
        n_step++;
    }
    if (q < 2) partials[(long long)(16 * wid + g + 8 * q) * gridDim.x + blockIdx.x] = y_acc;
}
