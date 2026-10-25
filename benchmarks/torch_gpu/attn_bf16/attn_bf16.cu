// Attention's two BF16 GEMMs on sm_90a (H100, H200) and sm_120a (RTX PRO 6000), each output word exactly
// GemmCoordinate_v2{K, GEMM_HOPPER}: K/16 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` steps in ascending k from
// a +0 accumulator, then one `cvt.rn.bf16.f32`.  Both cards' mma.sync BF16 step is Hopper's (sm90.mma.m16n8k16.bf16, pinned;
// the PRO 6000's capture of sm120.mma.m16n8k16.bf16 agrees with it).  Softmax is its own unit: QK^T's words come out, PV's
// P goes in.
//
//   attn_qk_d{64,128}[_h8]   S[b,h,q,j] = GemmCoordinate{K=D}(Q[b,h,q,:], K[b,kv(h),j,:])                   for j < len(b,q)
//   attn_pv_d{64,128}[_h8]   O[b,h,q,d] = GemmCoordinate{K=kp}(P[b,h,q,:len] ++ 0, V[b,kv(h),:len,d] ++ 0)    len = len(b,q)
//
// with kp = max(kpv, pad16(len)): kpv = 0 pads each query's keys to a multiple of 16, and kpv = 32 is the H200 model's PV
// (every KV length 1..32 padded with +0 to K = 32).
// kv(h) = h >> gshift: 2^gshift query heads share a KV head (MHA: 0; MQA: log2 H).  Sequence b holds lens[b] keys and its
// last nq[b] positions are the queries (decode: nq = 1; prefill: nq = lens; a chunk: anything between), so query q < nq[b]
// sees len(b,q) = lens[b] - nq[b] + q + 1 keys (causal).  Nothing else is written: S past len(b,q), and every row
// q >= nq[b], keep what the buffer held.
//
// A warp computes a 16 x 8 tile of output words at a time and runs each word's whole k chain in registers: no split-K, no
// atomics, no shared reduction.  In the q8 kernels a tile's 8 columns are 8 queries of one head (grid: query tiles of 8, H,
// B), and a block's warps stride over its key tiles (QK^T) or d tiles (PV).  In the _h8 kernels they are one query of 8
// heads that share a KV head (8 divides 2^gshift; grid: Mq, any Y, B): block y runs the head groups y, y + Y, ..., its
// warps take every warps-th of them, and a warp runs every tile of its group (QK^T: each key tile, with the group's Q loaded
// once; PV: each d tile, with P loaded once a key block), so a decode step reads each key and value once for 8 heads.  The
// launch decides which warp runs a tile, never a word's steps.
// QK^T's tile is 16 keys (A, K's rows) x 8 columns (B, Q's rows) over k = D.  PV's is 16 d (A, V^T) x 8 columns (B, P^T)
// over k = keys.  A word's products depend only on its row of A and column of B, so keys past a column's length are zero
// codes in both operands, as the Definition pads them: P is loaded as zero there, and V too.  V is shared by the tile's
// columns, and in the query-tiled kernels they can stop at different keys, so in a 16-key block each distinct stopping
// point gets its own mma with V zeroed past it, and each column takes its word from the mma of its own; a column whose keys
// ended before the block keeps its accumulator unless the block is below kpv, so its chain is its own kp/16 steps.  The _h8
// kernels' columns share one length, so each block is one mma.
//
// The kernels do no floating-point arithmetic: operands and accumulators move as bits, and the only floating-point
// instructions are the mma and the cvt (sass_gate.py refuses any other).

#include <stdint.h>

struct AttnArgs {
    const uint16_t* x;   // QK: Q[b][h][q][D]   PV: P[b][h][q][Lcap]
    const uint16_t* kv;  // QK: K[b][kv][j][D]  PV: V[b][kv][j][D]
    uint16_t* out;       // QK: S[b][h][q][Lcap]   PV: O[b][h][q][D]
    const int32_t* lens; // [B] keys of each sequence
    const int32_t* nq;   // [B] query rows of each sequence, its last nq positions
    int64_t xs[3], kvs[3], os[3];  // element strides of batch, head and row; the last dimension is contiguous
    int32_t B, H, Mq, Lcap;        // Mq: the query rows each (b, h) has room for; Lcap: S's and the cache's key rows
    int32_t gshift;                // log2 of the query heads a KV head serves
    int32_t kpv;                   // PV's least K, a multiple of 16 (0: pad16(len) alone)
};

namespace {

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ uint16_t cvt_bf16(float x) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ uint32_t pack(uint32_t lo, uint32_t hi) { return lo | hi << 16; }

__device__ __forceinline__ uint32_t ld_pair(const uint16_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// QK^T's A operand at step s: K rows ja and jb (zero codes where a row is past the cache's keys), k = 16s + 2t ...
__device__ __forceinline__ void k_rows(uint32_t (&A)[4], const uint16_t* karow, const uint16_t* kbrow, bool jaok, bool jbok,
                                       int c) {
    A[0] = jaok ? ld_pair(karow + c) : 0u;
    A[1] = jbok ? ld_pair(kbrow + c) : 0u;
    A[2] = jaok ? ld_pair(karow + c + 8) : 0u;
    A[3] = jbok ? ld_pair(kbrow + c + 8) : 0u;
}

// A q8 block's tile, from the grid alone (no division: the SASS gate refuses the floating-point reciprocal an integer one
// becomes): columns c are queries q0 + c of head h.
struct Tile {
    int b, h, q0, warp, warps, L, M;
    __device__ __forceinline__ int q(int c) const { return q0 + c; }
    __device__ __forceinline__ bool ok(int c) const { return q0 + c < M; }
    __device__ __forceinline__ int seen(int c) const { return ok(c) ? L - M + q0 + c + 1 : 0; }
    __device__ __forceinline__ int last() const { return min(8, M - q0) - 1; }  // the column seeing most keys
};

__device__ __forceinline__ Tile tile_of(const AttnArgs& a) {
    const int b = blockIdx.z;
    return {b, (int)blockIdx.y, 8 * (int)blockIdx.x, (int)(threadIdx.x >> 5), (int)(blockDim.x >> 5), a.lens[b], a.nq[b]};
}

template <int D>
__device__ __forceinline__ void qk(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Tile w = tile_of(a);
    if (!w.ok(0)) return;
    const int b = w.b, L = w.L, seen_max = w.seen(w.last());
    const uint16_t* K = a.kv + b * a.kvs[0] + (w.h >> a.gshift) * a.kvs[1];
    const bool qok = w.ok(g);
    const uint16_t* qrow = a.x + b * a.xs[0] + w.h * a.xs[1] + (qok ? w.q(g) : 0) * a.xs[2];
    uint16_t* S = a.out + b * a.os[0] + w.h * a.os[1];

    for (int j0 = 16 * w.warp; j0 < seen_max; j0 += 16 * w.warps) {
        const int ja = j0 + g, jb = j0 + g + 8;
        const bool jaok = ja < L, jbok = jb < L;
        const uint16_t* karow = K + (jaok ? ja : 0) * a.kvs[2];
        const uint16_t* kbrow = K + (jbok ? jb : 0) * a.kvs[2];
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {
            const int c = 16 * s + 2 * t;
            uint32_t A[4];
            k_rows(A, karow, kbrow, jaok, jbok, c);
            const uint32_t Bf[2] = {qok ? ld_pair(qrow + c) : 0u, qok ? ld_pair(qrow + c + 8) : 0u};
            mma_bf16(acc, A, Bf, acc);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int col = 2 * t + (e & 1), j = e < 2 ? ja : jb;
            if (w.ok(col) && j < w.seen(col)) S[w.q(col) * a.os[2] + j] = cvt_bf16(acc[e]);
        }
    }
}

template <int D>
__device__ __forceinline__ void pv(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Tile w = tile_of(a);
    if (!w.ok(0)) return;
    const int b = w.b, L = w.L;
    const int steps = max(a.kpv >> 4, (w.seen(w.last()) + 15) >> 4);
    const uint16_t* P = a.x + b * a.xs[0] + w.h * a.xs[1] + (w.ok(g) ? w.q(g) : 0) * a.xs[2];
    const int64_t vr = a.kvs[2];
    const int Lg = w.seen(g), La = w.seen(2 * t), Lb = w.seen(2 * t + 1);

    for (int d0 = 16 * w.warp; d0 < D; d0 += 16 * w.warps) {
        const uint16_t* V = a.kv + b * a.kvs[0] + (w.h >> a.gshift) * a.kvs[1] + d0 + g;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int s = 0; s < steps; ++s) {
            const int k0 = 16 * s;
            // this lane's keys: k0 + 2t + {0, 1} (A's a0/a1 and B's b0) and k0 + 2t + 8 + {0, 1} (a2/a3 and b1)
            const int kk[4] = {k0 + 2 * t, k0 + 2 * t + 1, k0 + 2 * t + 8, k0 + 2 * t + 9};
            uint32_t p[4], v0[4], v8[4];  // P at kk; V at kk for d0 + g and d0 + g + 8; zero codes past a length
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const bool live = kk[i] < L;
                p[i] = kk[i] < Lg ? P[kk[i]] : 0u;
                v0[i] = live ? V[kk[i] * vr] : 0u;
                v8[i] = live ? V[kk[i] * vr + 8] : 0u;
            }
            const uint32_t Bf[2] = {pack(p[0], p[1]), pack(p[2], p[3])};
            const int ma = min(max(La - k0, 0), 16), mb = min(max(Lb - k0, 0), 16);
            float next[4] = {acc[0], acc[1], acc[2], acc[3]};
            for (int j = 0; j < 8; ++j) {  // warp-uniform: one mma per distinct count of the block's keys a column sees
                const int m = min(max(w.seen(j) - k0, 0), 16);
                bool again = m == 0 && k0 >= a.kpv;
                for (int i = 0; i < j; ++i) again |= min(max(w.seen(i) - k0, 0), 16) == m;
                if (again) continue;
                uint32_t w0[4], w8[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    w0[i] = kk[i] - k0 < m ? v0[i] : 0u;
                    w8[i] = kk[i] - k0 < m ? v8[i] : 0u;
                }
                const uint32_t A[4] = {pack(w0[0], w0[1]), pack(w8[0], w8[1]), pack(w0[2], w0[3]), pack(w8[2], w8[3])};
                float d[4];
                mma_bf16(d, A, Bf, acc);
                if (ma == m) next[0] = d[0], next[2] = d[2];
                if (mb == m) next[1] = d[1], next[3] = d[3];
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[e] = next[e];
        }
        uint16_t* O = a.out + b * a.os[0] + d0 + g;
        const int ca = 2 * t, cb = 2 * t + 1;
        if (w.ok(ca)) {
            O[w.h * a.os[1] + w.q(ca) * a.os[2]] = cvt_bf16(acc[0]);
            O[w.h * a.os[1] + w.q(ca) * a.os[2] + 8] = cvt_bf16(acc[2]);
        }
        if (w.ok(cb)) {
            O[w.h * a.os[1] + w.q(cb) * a.os[2]] = cvt_bf16(acc[1]);
            O[w.h * a.os[1] + w.q(cb) * a.os[2] + 8] = cvt_bf16(acc[3]);
        }
    }
}

// The _h8 kernels' head groups: query q = blockIdx.x of sequence b = blockIdx.z, and this warp's first group's head h0
// and the stride between its groups' heads.
struct Groups {
    int b, q, L, seen, h0, hstep;
};

__device__ __forceinline__ Groups groups_of(const AttnArgs& a) {
    const int b = blockIdx.z, q = blockIdx.x, L = a.lens[b], M = a.nq[b];
    const int Y = gridDim.y, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    return {b, q, L, q < M ? L - M + q + 1 : 0, 8 * ((int)blockIdx.y + Y * warp), 8 * Y * warps};
}

template <int D>
__device__ __forceinline__ void qk8(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of(a);
    if (!w.seen) return;
    uint16_t* S = a.out + w.b * a.os[0] + w.q * a.os[2];
#pragma unroll 1
    for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
        const uint16_t* K = a.kv + w.b * a.kvs[0] + (h0 >> a.gshift) * a.kvs[1];
        const uint16_t* qrow = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
        uint32_t Bq[D / 16][2];
#pragma unroll
        for (int s = 0; s < D / 16; ++s) Bq[s][0] = ld_pair(qrow + 16 * s + 2 * t), Bq[s][1] = ld_pair(qrow + 16 * s + 2 * t + 8);
#pragma unroll 1
        for (int j0 = 0; j0 < w.seen; j0 += 16) {
            const int ja = j0 + g, jb = j0 + g + 8;
            const bool jaok = ja < w.L, jbok = jb < w.L;
            const uint16_t* karow = K + (jaok ? ja : 0) * a.kvs[2];
            const uint16_t* kbrow = K + (jbok ? jb : 0) * a.kvs[2];
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int s = 0; s < D / 16; ++s) {
                uint32_t A[4];
                k_rows(A, karow, kbrow, jaok, jbok, 16 * s + 2 * t);
                mma_bf16(acc, A, Bq[s], acc);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = e < 2 ? ja : jb;
                if (j < w.seen) S[(h0 + 2 * t + (e & 1)) * a.os[1] + j] = cvt_bf16(acc[e]);
            }
        }
    }
}

template <int D>
__device__ __forceinline__ void pv8(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of(a);
    if (!w.seen) return;
    const int steps = max(a.kpv >> 4, (w.seen + 15) >> 4);
    const int64_t vr = a.kvs[2];
#pragma unroll 1
    for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
        const uint16_t* P = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
        const uint16_t* V = a.kv + w.b * a.kvs[0] + (h0 >> a.gshift) * a.kvs[1] + g;
        float acc[D / 16][4];
#pragma unroll
        for (int dt = 0; dt < D / 16; ++dt) acc[dt][0] = acc[dt][1] = acc[dt][2] = acc[dt][3] = 0.0f;
        for (int s = 0; s < steps; ++s) {
            const int k0 = 16 * s;
            const int kk[4] = {k0 + 2 * t, k0 + 2 * t + 1, k0 + 2 * t + 8, k0 + 2 * t + 9};
            bool live[4];
            uint32_t p[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                live[i] = kk[i] < w.seen;
                p[i] = live[i] ? P[kk[i]] : 0u;
            }
            const uint32_t Bf[2] = {pack(p[0], p[1]), pack(p[2], p[3])};
#pragma unroll
            for (int dt = 0; dt < D / 16; ++dt) {
                uint32_t v0[4], v8[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    v0[i] = live[i] ? V[kk[i] * vr + 16 * dt] : 0u;
                    v8[i] = live[i] ? V[kk[i] * vr + 16 * dt + 8] : 0u;
                }
                const uint32_t A[4] = {pack(v0[0], v0[1]), pack(v8[0], v8[1]), pack(v0[2], v0[3]), pack(v8[2], v8[3])};
                mma_bf16(acc[dt], A, Bf, acc[dt]);
            }
        }
        uint16_t* Oa = a.out + w.b * a.os[0] + (h0 + 2 * t) * a.os[1] + w.q * a.os[2] + g;
        uint16_t* Ob = Oa + a.os[1];
#pragma unroll
        for (int dt = 0; dt < D / 16; ++dt) {
            Oa[16 * dt] = cvt_bf16(acc[dt][0]);
            Oa[16 * dt + 8] = cvt_bf16(acc[dt][2]);
            Ob[16 * dt] = cvt_bf16(acc[dt][1]);
            Ob[16 * dt + 8] = cvt_bf16(acc[dt][3]);
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) attn_qk_d64(const AttnArgs a) { qk<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d128(const AttnArgs a) { qk<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d64(const AttnArgs a) { pv<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d128(const AttnArgs a) { pv<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d64_h8(const AttnArgs a) { qk8<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d128_h8(const AttnArgs a) { qk8<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d64_h8(const AttnArgs a) { pv8<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d128_h8(const AttnArgs a) { pv8<128>(a); }
