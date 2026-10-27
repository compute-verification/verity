// Attention's two BF16 GEMMs on sm_90a (H100, H200) and sm_120a (RTX PRO 6000), each output word exactly
// GemmCoordinate_v2{K, GEMM_HOPPER}: K/16 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` steps in ascending k from
// a +0 accumulator, then one `cvt.rn.bf16.f32`.  Both cards' mma.sync BF16 step is Hopper's (sm90.mma.m16n8k16.bf16, pinned;
// the PRO 6000's capture of sm120.mma.m16n8k16.bf16 agrees with it).  Softmax is its own Definition: QK^T's words come out, PV's
// P goes in.
//
//   attn_qk_d{64,128}[_h8|_h16|_s16|_ref]   S[b,h,q,j] = GemmCoordinate{K=D}(Q[b,h,q,:], K[b,kv(h),j,:])     for j < len(b,q)
//   attn_pv_d{64,128}[_h8|_h16|_s16|_ref]   O[b,h,q,d] = GemmCoordinate{K=kp}(P[b,h,q,:len] ++ 0, V[b,kv(h),:len,d] ++ 0)
//                                                                                                        len = len(b,q)
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
// B), and a CTA's warps stride over its key tiles (QK^T) or d tiles (PV).  In the _h8 kernels they are one query of 8
// heads that share a KV head (8 divides 2^gshift; grid: Mq, any Y, B): CTA y runs the head groups y, y + Y, ..., its
// warps take every warps-th of them, and a warp runs every tile of its group (QK^T: each key tile, with the group's Q loaded
// once; PV: each d tile, with P loaded once a key segment), so a decode step reads each key and value once for 8 heads.  The
// _h16 kernels run head groups of 16 (16 divides 2^gshift) the same way, with the heads as the tile's 16 rows instead.  The
// _s16 kernels are _h16's tiles where every query sees at most S16_KEYS = 32 keys and PV's kp is at most 32 (the H200
// model's decode step): a warp loads the shared operand once for all of its groups of one KV head (K's 4 key tiles, V's 2
// key steps), and a query past those bounds is left unwritten.  The launch decides which warp runs a tile, never a word's
// steps.
// QK^T's tile is 16 keys (A, K's rows) x 8 columns (B, Q's rows) over k = D, and in _h16 16 heads (A, Q's rows) x 8 keys
// (B, K^T).  PV's is 16 d (A, V^T) x 8 columns (B, P^T) over k = keys, and in _h16 16 heads (A, P's rows) x 8 d (B, V).  The
// step's products are exact and its total semantics symmetric in its operands (a NaN product is the one canonical NaN), so
// which operand holds the heads doesn't change a word.  _h16 loads A's rows and stores S's and O's word pairs 32 bits at a
// time where the buffer's strides and base allow, and its PV O's rows 16 bytes a lane where they allow that (store_o).
// A word's products depend only on its row of A and column of B, so keys past a column's length are zero
// codes in both operands, as the Definition pads them: P is loaded as zero there, and V too.  V is shared by the tile's
// columns, and in the query-tiled kernels they can stop at different keys, so in a 16-key segment each distinct stopping
// point gets its own mma with V zeroed past it, and each column takes its word from the mma of its own; a column whose keys
// ended before the segment keeps its accumulator unless the segment is below kpv, so its chain is its own kp/16 steps.  The _h8
// kernels' columns share one length, so each segment is one mma.
// The _ref kernels are the slow reference path, tuned for nothing: a CTA takes 8
// columns (QK^T: keys; PV: d) of one KV head and sequence (grid: column tiles, Hkv, B), its warps take turns over the
// (16 query heads of that KV head, query row) pairs, and every operand word is gathered on its own, so a tile's rows share
// one query and its chain is that query's.
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

constexpr int S16_KEYS = 32;  // the _s16 kernels' most keys a query sees, and PV's largest kp

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

// A q8 CTA's tile, from the grid alone (no division: the SASS gate refuses the floating-point reciprocal an integer one
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
            for (int j = 0; j < 8; ++j) {  // warp-uniform: one mma per distinct count of the segment's keys a column sees
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

// The _h8, _h16 and _s16 kernels' head groups of N heads: query q = blockIdx.x of sequence b = blockIdx.z, and this warp's first
// group's head h0 and the stride between its groups' heads.
struct Groups {
    int b, q, L, seen, h0, hstep;
};

template <int N>
__device__ __forceinline__ Groups groups_of(const AttnArgs& a) {
    const int b = blockIdx.z, q = blockIdx.x, L = a.lens[b], M = a.nq[b];
    const int Y = gridDim.y, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    return {b, q, L, q < M ? L - M + q + 1 : 0, N * ((int)blockIdx.y + Y * warp), N * Y * warps};
}

// Whether a [B][H][rows][cols] buffer's word pairs at even columns are 32-bit aligned: its base and every stride even.
__device__ __forceinline__ bool pairs_ok(const void* base, const int64_t (&st)[3]) {
    return ((st[0] | st[1] | st[2]) & 1) == 0 && (reinterpret_cast<uintptr_t>(base) & 3) == 0;
}

// Words k and k + 1 of a row as A's 32-bit pair (k even), the zero code for each at or past n.
__device__ __forceinline__ uint32_t ld2(const uint16_t* p, int k, int n, bool pairs) {
    if (k + 1 < n) return pairs ? ld_pair(p) : pack(p[0], p[1]);
    return k < n ? (uint32_t)p[0] : 0u;
}

// cvt.rn.bf16.f32 of x (low half) and y (high half) as one packed cvt: ptxas lowers a lone cvt.rn.bf16.f32 to F2F.BF16.F32
// on sm_120a next to packed ones, which the SASS gate refuses.
__device__ __forceinline__ uint32_t cvt2_bf16(float x, float y) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(y), "f"(x));
    return r;
}

// cvt of x and y to words k and k + 1 (k even), each only if below n.
__device__ __forceinline__ void st2(uint16_t* p, float x, float y, int k, int n, bool pairs) {
    const uint32_t w = cvt2_bf16(x, y);
    if (k + 1 < n && pairs) {
        *reinterpret_cast<uint32_t*>(p) = w;
        return;
    }
    if (k < n) p[0] = (uint16_t)w;
    if (k + 1 < n) p[1] = (uint16_t)(w >> 16);
}

// Whether a [B][H][rows][cols] buffer's 8-word runs at columns 8i are 16-byte aligned: its base and every stride.
__device__ __forceinline__ bool quads_ok(const void* base, const int64_t (&st)[3]) {
    return ((st[0] | st[1] | st[2]) & 7) == 0 && (reinterpret_cast<uintptr_t>(base) & 15) == 0;
}

// The quad's 4 x 4 transpose of 32-bit words: lane t's w[j] becomes lane j's w[t] (t = lane & 3), by exchanges with lanes
// t ^ 1 and t ^ 2 and integer selects.  It moves bits only, and every lane of the warp must run it.
__device__ __forceinline__ void quad_transpose(uint32_t (&w)[4], int t) {
    const bool t0 = t & 1, t1 = t & 2;
#pragma unroll
    for (int j1 = 0; j1 < 2; ++j1) {  // swap the lane's bit 0 with the index's: send the words whose index bit 0 isn't t0
        const uint32_t got = __shfl_xor_sync(0xffffffffu, t0 ? w[2 * j1] : w[2 * j1 + 1], 1);
        if (t0) w[2 * j1] = got; else w[2 * j1 + 1] = got;
    }
#pragma unroll
    for (int j0 = 0; j0 < 2; ++j0) {  // then bit 1
        const uint32_t got = __shfl_xor_sync(0xffffffffu, t1 ? w[j0] : w[2 + j0], 2);
        if (t1) w[j0] = got; else w[2 + j0] = got;
    }
}

// PV's words of rows Oa (heads h0 + g) and Ob (h0 + g + 8) for N d tiles from Oa's and Ob's column.  Where O's 8-word runs
// are 16-byte aligned, each 4 tiles' words are transposed within the quad, so lane t stores columns 32c + 8t .. + 7 of
// each row as one 16-byte word and a row's 4 lanes 64 contiguous bytes; else each lane stores its word pairs as they come.
template <int N>
__device__ __forceinline__ void store_o(uint16_t* Oa, uint16_t* Ob, const float (&acc)[N][4], int t, bool quads, bool pairs) {
    static_assert(N % 4 == 0, "store_o transposes 4 d tiles at a time");
    if (quads) {
#pragma unroll
        for (int c = 0; c < N / 4; ++c) {
            uint32_t wa[4], wb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wa[i] = cvt2_bf16(acc[4 * c + i][0], acc[4 * c + i][1]);
                wb[i] = cvt2_bf16(acc[4 * c + i][2], acc[4 * c + i][3]);
            }
            quad_transpose(wa, t);
            quad_transpose(wb, t);
            *reinterpret_cast<uint4*>(Oa + 32 * c + 8 * t) = make_uint4(wa[0], wa[1], wa[2], wa[3]);
            *reinterpret_cast<uint4*>(Ob + 32 * c + 8 * t) = make_uint4(wb[0], wb[1], wb[2], wb[3]);
        }
        return;
    }
#pragma unroll
    for (int dt = 0; dt < N; ++dt) {  // d = 8 dt + 2t, + 1 of heads h0 + g (acc 0, 1) and h0 + g + 8 (2, 3)
        st2(Oa + 8 * dt + 2 * t, acc[dt][0], acc[dt][1], 0, 2, pairs);
        st2(Ob + 8 * dt + 2 * t, acc[dt][2], acc[dt][3], 0, 2, pairs);
    }
}

// The _h16 and _s16 PV's d a pass (8 d tiles), which bounds their registers at head dim 128.
constexpr int PV_DC = 64;

// B of PV's step with A's keys ka, ka + 1, ka + 8, ka + 9 (ka = 16s + 2t) for d tiles dt < N: V's rows of those keys (the
// zero code at or past n) at column 8 dt of V, which points at this lane's d (d0 + g).
template <int N>
__device__ __forceinline__ void v_frags(uint32_t (&Bv)[N][2], const uint16_t* V, int64_t vr, int ka, int n) {
    const int kb = ka + 8;
    const bool l0 = ka < n, l1 = ka + 1 < n, l2 = kb < n, l3 = kb + 1 < n;
    const uint16_t* r0 = V + (l0 ? ka : 0) * vr;
    const uint16_t* r1 = V + (l1 ? ka + 1 : 0) * vr;
    const uint16_t* r2 = V + (l2 ? kb : 0) * vr;
    const uint16_t* r3 = V + (l3 ? kb + 1 : 0) * vr;
#pragma unroll
    for (int dt = 0; dt < N; ++dt) {
        Bv[dt][0] = pack(l0 ? r0[8 * dt] : 0u, l1 ? r1[8 * dt] : 0u);
        Bv[dt][1] = pack(l2 ? r2[8 * dt] : 0u, l3 ? r3[8 * dt] : 0u);
    }
}

template <int D>
__device__ __forceinline__ void qk8(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of<8>(a);
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
    const Groups w = groups_of<8>(a);
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

template <int D>
__device__ __forceinline__ void qk16(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of<16>(a);
    if (!w.seen) return;
    const bool pairs = pairs_ok(a.out, a.os);
#pragma unroll 1
    for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
        const uint16_t* K = a.kv + w.b * a.kvs[0] + (h0 >> a.gshift) * a.kvs[1];
        const uint16_t* qa = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
        const uint16_t* qb = qa + 8 * a.xs[1];
        uint32_t Aq[D / 16][4];  // heads h0 + g and h0 + g + 8, k = 16s + 2t ...
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {
            const int c = 16 * s + 2 * t;
            Aq[s][0] = ld_pair(qa + c), Aq[s][1] = ld_pair(qb + c), Aq[s][2] = ld_pair(qa + c + 8), Aq[s][3] = ld_pair(qb + c + 8);
        }
        uint16_t* Sa = a.out + w.b * a.os[0] + (h0 + g) * a.os[1] + w.q * a.os[2];
        uint16_t* Sb = Sa + 8 * a.os[1];
#pragma unroll 1
        for (int j0 = 0; j0 < w.seen; j0 += 8) {
            const int j = j0 + g;  // B's column: key j
            const bool jok = j < w.L;
            const uint16_t* krow = K + (jok ? j : 0) * a.kvs[2];
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int s = 0; s < D / 16; ++s) {
                const int c = 16 * s + 2 * t;
                const uint32_t Bk[2] = {jok ? ld_pair(krow + c) : 0u, jok ? ld_pair(krow + c + 8) : 0u};
                mma_bf16(acc, Aq[s], Bk, acc);
            }
            const int jc = j0 + 2 * t;  // this lane's words: keys jc, jc + 1 of heads h0 + g (acc 0, 1) and h0 + g + 8 (2, 3)
            st2(Sa + jc, acc[0], acc[1], jc, w.seen, pairs);
            st2(Sb + jc, acc[2], acc[3], jc, w.seen, pairs);
        }
    }
}

template <int D>
__device__ __forceinline__ void pv16(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of<16>(a);
    if (!w.seen) return;
    const int steps = max(a.kpv >> 4, (w.seen + 15) >> 4);
    const bool pin = pairs_ok(a.x, a.xs), pout = pairs_ok(a.out, a.os), qout = quads_ok(a.out, a.os);
    const int64_t vr = a.kvs[2];
#pragma unroll 1
    for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
        const uint16_t* pa = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
        const uint16_t* pb = pa + 8 * a.xs[1];
#pragma unroll 1
        for (int d0 = 0; d0 < D; d0 += PV_DC) {
            const uint16_t* V = a.kv + w.b * a.kvs[0] + (h0 >> a.gshift) * a.kvs[1] + d0 + g;  // B's column: d0 + 8 dt + g
            float acc[PV_DC / 8][4];
#pragma unroll
            for (int dt = 0; dt < PV_DC / 8; ++dt) acc[dt][0] = acc[dt][1] = acc[dt][2] = acc[dt][3] = 0.0f;
#pragma unroll 1
            for (int s = 0; s < steps; ++s) {
                const int ka = 16 * s + 2 * t, kb = ka + 8;  // A's keys ka, ka + 1 and kb, kb + 1; B's rows the same
                const uint32_t A[4] = {ld2(pa + ka, ka, w.seen, pin), ld2(pb + ka, ka, w.seen, pin),
                                       ld2(pa + kb, kb, w.seen, pin), ld2(pb + kb, kb, w.seen, pin)};
                uint32_t Bv[PV_DC / 8][2];
                v_frags(Bv, V, vr, ka, w.seen);
#pragma unroll
                for (int dt = 0; dt < PV_DC / 8; ++dt) mma_bf16(acc[dt], A, Bv[dt], acc[dt]);
            }
            uint16_t* Oa = a.out + w.b * a.os[0] + (h0 + g) * a.os[1] + w.q * a.os[2] + d0;
            store_o(Oa, Oa + 8 * a.os[1], acc, t, qout, pout);
        }
    }
}

template <int D>
__device__ __forceinline__ void qk16s(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of<16>(a);
    if (!w.seen || w.seen > S16_KEYS) return;
    const bool pairs = pairs_ok(a.out, a.os);
    uint32_t Bk[S16_KEYS / 8][D / 16][2];  // key tile n's B at step s, loaded once for every group of KV head kvh
    int kvh = -1;
#pragma unroll 1
    for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
        if ((h0 >> a.gshift) != kvh) {
            kvh = h0 >> a.gshift;
            const uint16_t* K = a.kv + w.b * a.kvs[0] + kvh * a.kvs[1];
#pragma unroll
            for (int n = 0; n < S16_KEYS / 8; ++n) {
                const int j = 8 * n + g;  // B's column: key j
                const bool jok = j < w.seen;
                const uint16_t* krow = K + (jok ? j : 0) * a.kvs[2];
#pragma unroll
                for (int s = 0; s < D / 16; ++s) {
                    const int c = 16 * s + 2 * t;
                    Bk[n][s][0] = jok ? ld_pair(krow + c) : 0u;
                    Bk[n][s][1] = jok ? ld_pair(krow + c + 8) : 0u;
                }
            }
        }
        const uint16_t* qa = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
        const uint16_t* qb = qa + 8 * a.xs[1];
        float acc[S16_KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < S16_KEYS / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {  // each word's chain: steps s = 0 .. D/16 - 1 in order, from +0
            const int c = 16 * s + 2 * t;
            const uint32_t Aq[4] = {ld_pair(qa + c), ld_pair(qb + c), ld_pair(qa + c + 8), ld_pair(qb + c + 8)};
#pragma unroll
            for (int n = 0; n < S16_KEYS / 8; ++n)
                if (8 * n < w.seen) mma_bf16(acc[n], Aq, Bk[n][s], acc[n]);
        }
        uint16_t* Sa = a.out + w.b * a.os[0] + (h0 + g) * a.os[1] + w.q * a.os[2];
        uint16_t* Sb = Sa + 8 * a.os[1];
#pragma unroll
        for (int n = 0; n < S16_KEYS / 8; ++n) {
            const int jc = 8 * n + 2 * t;
            st2(Sa + jc, acc[n][0], acc[n][1], jc, w.seen, pairs);
            st2(Sb + jc, acc[n][2], acc[n][3], jc, w.seen, pairs);
        }
    }
}

template <int D>
__device__ __forceinline__ void pv16s(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const Groups w = groups_of<16>(a);
    if (!w.seen) return;
    const int steps = max(a.kpv >> 4, (w.seen + 15) >> 4);
    if (steps > S16_KEYS / 16) return;
    const bool pin = pairs_ok(a.x, a.xs), pout = pairs_ok(a.out, a.os), qout = quads_ok(a.out, a.os);
    const int64_t vr = a.kvs[2];
#pragma unroll 1
    for (int d0 = 0; d0 < D; d0 += PV_DC) {
        uint32_t Bv[S16_KEYS / 16][PV_DC / 8][2];  // step s's B for d tile dt, loaded once for every group of KV head kvh
        int kvh = -1;
#pragma unroll 1
        for (int h0 = w.h0; h0 < a.H; h0 += w.hstep) {
            if ((h0 >> a.gshift) != kvh) {
                kvh = h0 >> a.gshift;
                const uint16_t* V = a.kv + w.b * a.kvs[0] + kvh * a.kvs[1] + d0 + g;  // B's column: d = d0 + 8 dt + g
#pragma unroll
                for (int s = 0; s < S16_KEYS / 16; ++s) v_frags(Bv[s], V, vr, 16 * s + 2 * t, w.seen);
            }
            const uint16_t* pa = a.x + w.b * a.xs[0] + (h0 + g) * a.xs[1] + w.q * a.xs[2];
            const uint16_t* pb = pa + 8 * a.xs[1];
            float acc[PV_DC / 8][4];
#pragma unroll
            for (int dt = 0; dt < PV_DC / 8; ++dt) acc[dt][0] = acc[dt][1] = acc[dt][2] = acc[dt][3] = 0.0f;
#pragma unroll
            for (int s = 0; s < S16_KEYS / 16; ++s) {  // each word's chain: its kp/16 steps in order, from +0
                if (s < steps) {
                    const int ka = 16 * s + 2 * t, kb = ka + 8;
                    const uint32_t A[4] = {ld2(pa + ka, ka, w.seen, pin), ld2(pb + ka, ka, w.seen, pin),
                                           ld2(pa + kb, kb, w.seen, pin), ld2(pb + kb, kb, w.seen, pin)};
#pragma unroll
                    for (int dt = 0; dt < PV_DC / 8; ++dt) mma_bf16(acc[dt], A, Bv[s][dt], acc[dt]);
                }
            }
            uint16_t* Oa = a.out + w.b * a.os[0] + (h0 + g) * a.os[1] + w.q * a.os[2] + d0;
            store_o(Oa, Oa + 8 * a.os[1], acc, t, qout, pout);
        }
    }
}

// Word i of p, the zero code unless ok.
__device__ __forceinline__ uint32_t word(const uint16_t* p, int64_t i, bool ok) { return ok ? (uint32_t)p[i] : 0u; }

// The _ref kernels.  Item i of a CTA is row tile i & (2^rs - 1) of query i >> rs: rows ra and rb are heads
// (kvh << gshift) + ra and + rb, the zero code past the KV head's G heads, and B's column is key or d c0 + g.
template <int D, bool QK>
__device__ __forceinline__ void ref(const AttnArgs& a) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int b = blockIdx.z, kvh = blockIdx.y, c0 = 8 * (int)blockIdx.x, L = a.lens[b], M = a.nq[b];
    const int G = 1 << a.gshift, rs = max(a.gshift - 4, 0), h0 = kvh << a.gshift;
    const uint16_t* KV = a.kv + b * a.kvs[0] + kvh * a.kvs[1];
    const int64_t kr = a.kvs[2];
#pragma unroll 1
    for (int i = threadIdx.x >> 5; i < (M << rs); i += blockDim.x >> 5) {
        const int q = i >> rs, ra = 16 * (i & ((1 << rs) - 1)) + g, rb = ra + 8, n = L - M + q + 1;
        if (n <= 0 || (QK && c0 >= n)) continue;
        const bool aok = ra < G, bok = rb < G;
        const uint16_t* xa = a.x + b * a.xs[0] + (h0 + (aok ? ra : 0)) * a.xs[1] + q * a.xs[2];
        const uint16_t* xb = a.x + b * a.xs[0] + (h0 + (bok ? rb : 0)) * a.xs[1] + q * a.xs[2];
        const int col = c0 + g;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (QK) {
            const bool kok = col < n;
            const uint16_t* kc = KV + (kok ? col : 0) * kr;
#pragma unroll 1
            for (int k0 = 0; k0 < D; k0 += 16) {
                const int lo = k0 + 2 * t, hi = lo + 8;
                const uint32_t A[4] = {pack(word(xa, lo, aok), word(xa, lo + 1, aok)), pack(word(xb, lo, bok), word(xb, lo + 1, bok)),
                                       pack(word(xa, hi, aok), word(xa, hi + 1, aok)), pack(word(xb, hi, bok), word(xb, hi + 1, bok))};
                const uint32_t Bf[2] = {pack(word(kc, lo, kok), word(kc, lo + 1, kok)), pack(word(kc, hi, kok), word(kc, hi + 1, kok))};
                mma_bf16(acc, A, Bf, acc);
            }
        } else {
            const int steps = max(a.kpv >> 4, (n + 15) >> 4);
            const uint16_t* vc = KV + col;
#pragma unroll 1
            for (int s = 0; s < steps; ++s) {
                const int lo = 16 * s + 2 * t, hi = lo + 8;
                const bool l0 = lo < n, l1 = lo + 1 < n, h0k = hi < n, h1k = hi + 1 < n;
                const uint32_t A[4] = {pack(word(xa, lo, aok && l0), word(xa, lo + 1, aok && l1)),
                                       pack(word(xb, lo, bok && l0), word(xb, lo + 1, bok && l1)),
                                       pack(word(xa, hi, aok && h0k), word(xa, hi + 1, aok && h1k)),
                                       pack(word(xb, hi, bok && h0k), word(xb, hi + 1, bok && h1k))};
                const uint32_t Bf[2] = {pack(word(vc, lo * kr, l0), word(vc, (lo + 1) * kr, l1)),
                                        pack(word(vc, hi * kr, h0k), word(vc, (hi + 1) * kr, h1k))};
                mma_bf16(acc, A, Bf, acc);
            }
        }
        // this lane's words: columns c0 + 2t and c0 + 2t + 1 of rows ra (acc 0, 1) and rb (2, 3), 16 bits a store
        const int cc = c0 + 2 * t;
        uint16_t* out = a.out + b * a.os[0] + q * a.os[2] + cc;
        if (aok) st2(out + (h0 + ra) * a.os[1], acc[0], acc[1], cc, QK ? n : D, false);
        if (bok) st2(out + (h0 + rb) * a.os[1], acc[2], acc[3], cc, QK ? n : D, false);
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
extern "C" __global__ void __launch_bounds__(256) attn_qk_d64_h16(const AttnArgs a) { qk16<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d128_h16(const AttnArgs a) { qk16<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d64_h16(const AttnArgs a) { pv16<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d128_h16(const AttnArgs a) { pv16<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d64_s16(const AttnArgs a) { qk16s<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d128_s16(const AttnArgs a) { qk16s<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d64_s16(const AttnArgs a) { pv16s<64>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d128_s16(const AttnArgs a) { pv16s<128>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d64_ref(const AttnArgs a) { ref<64, true>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_qk_d128_ref(const AttnArgs a) { ref<128, true>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d64_ref(const AttnArgs a) { ref<64, false>(a); }
extern "C" __global__ void __launch_bounds__(256) attn_pv_d128_ref(const AttnArgs a) { ref<128, false>(a); }
