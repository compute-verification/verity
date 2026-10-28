// The ZK statement's row commitment on the serving GPU, for served_zk's cost measurement (rows_bench.py): a call's rows as
// hm96-sha512 row leaves of a frame-v3-sha512 tree (verity.ml.commitments: rowleaf's sha512/row-seg/v1 and
// sha512/row/v2, hm96's SHA512 instance, merkle's frame-v3 leaf and node), the hashing being C-Flock's (verity/ml/flock/cuda/
// sha512.cuh: s512_compress) and the frame messages, the mask, the salt digest and the hm96 leaf's value
// verity/ml/kernels/row_commit.cuh's over it.  Every launch takes rows_bench.py's Args (16 pointers, 16 ints).
//   sz_mid       the SHA-512 state after each of `count` 128-byte first blocks (the segment prefixes S(j, s, m_j), Q, P,
//                hm96's salt and leaf prefixes): constants of a shape and a key, computed once
//   sz_segments  d_j = SHA-512(S(j, s, m_j) || segment j) for every segment of every row, a lane a segment, a warp's 32 segments'
//                128-byte chunks staged through shared memory with coalesced loads (sha512_staged_merkle_leaves's scheme);
//                each segment a whole number of 128-byte chunks
//   sz_finish    a thread a row: x (row-seg: SHA-512(Q || d_0 .. d_(s-1)); v2, s = 0: SHA-512(P || row) of a row of at most
//                111 bytes), then by mode: 0 x alone (the service's unsalted digest), 1 b || c and the hm96 leaf
//                H(leaf_prefix || b || c) under row i's salt (the host's), 2 also the frame-v3 leaf
//                H(leaf_prefix_i || hm96 leaf) into the tree's level 0, and with one CTA the tree's levels above it; a
//                grid-stride loop over the rows
//   sz_level     one level of the tree (a thread a node), for trees wider than one CTA
#include "../../../verity/ml/flock/cuda/sha512.cuh"
#include "../../../verity/ml/kernels/row_commit.cuh"

struct Args { unsigned long long p[16]; int i[16]; };

#define SLOT 256   // bytes a leaf or node message head takes in its buffer; its length is the slot's first u32

// C-Flock's compression as row_commit.cuh's policy; compress_kw takes the block back from its schedule's first 16 words
struct SzC {
    static __device__ __forceinline__ void compress(uint64_t* st, uint64_t* w) { s512_compress(st, w); }
    static __device__ __forceinline__ void compress_kw(uint64_t* st, const uint64_t* kw) {
        uint64_t w[16];
#pragma unroll
        for (int j = 0; j < 16; j++) w[j] = kw[j] - SHA512_K[j];
        s512_compress(st, w);
    }
};

// SHA-512 of head || a || [u64be 64 || b] (b optional): a frame leaf (head || value) or node (head || left || len || right),
// head being merkle's message up to its value or its left child (rows_bench.py's slots)
__device__ __noinline__ void sz_frame(const uint8_t* head_slot, const uint8_t* a, const uint8_t* b, uint8_t* out) {
    uint8_t m[512];
    int n = *(const uint32_t*)head_slot;
    for (int t = 0; t < n; t++) m[t] = head_slot[4 + t];
    for (int t = 0; t < 64; t++) m[n + t] = a[t];
    n += 64;
    if (b) n = rc::frame_part(m, n, b, 64);
    for (int t = n, total = (n + 17 + 127) / 128 * 128; t < total; t++) m[t] = 0;
    uint64_t h[8];
    rc::digest_padded<SzC>(m, n, h);
    rc::st_be(out, h, 8);
}

// p: blocks, mids   i: count
extern "C" __global__ void sz_mid(Args a) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= a.i[0]) return;
    rc::midstate<SzC>((const uint8_t*)a.p[0] + 128 * t, (uint64_t*)a.p[1] + 8 * t);
}

// p: rows (row r at r * stride bytes), seg_mids (s states), digests out (rows * s * 64)   i: rows, stride, s, last segment bytes
extern "C" __global__ void sz_segments(Args a) {
    extern __shared__ uint32_t sz_sw[];
    const uint8_t* data = (const uint8_t*)a.p[0];
    const uint64_t* mids = (const uint64_t*)a.p[1];
    uint8_t* out = (uint8_t*)a.p[2];
    long long rows = a.i[0], stride = a.i[1];
    int s = a.i[2], last = a.i[3];
    long long total = rows * s;
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t* sh = sz_sw + wid * (32 * 33);
    long long warp0 = ((long long)blockIdx.x * (blockDim.x >> 5) + wid) * 32;
    if (warp0 >= total) return;
    long long g = warp0 + lane;
    int j = (int)(g % s);
    int bytes = j == s - 1 ? last : 1024;
    uint64_t st[8];
    if (g < total)
        for (int q = 0; q < 8; q++) st[q] = mids[8 * j + q];
    for (int c = 0; c < 8; c++) {
#pragma unroll
        for (int it = 0; it < 8; it++) {
            int idx = it * 32 + lane, piece = idx >> 3, off = idx & 7;
            long long pg = warp0 + piece;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (pg < total) {
                int pj = (int)(pg % s);
                int pb = pj == s - 1 ? last : 1024;
                if (c * 128 < pb) v = *(const uint4*)(data + (pg / s) * stride + (long long)pj * 1024 + c * 128 + off * 16);
            }
            uint32_t* d = sh + piece * 33 + off * 4;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        __syncwarp();
        if (g < total && c * 128 < bytes) {
            uint64_t w[16];
#pragma unroll
            for (int q = 0; q < 16; q++) w[q] = s512_be(sh[lane * 33 + 2 * q], sh[lane * 33 + 2 * q + 1]);
            s512_compress(st, w);
        }
        __syncwarp();
    }
    if (g >= total) return;
    s512_pad_block(st, (uint64_t)(128 + bytes));
    s512_out(st, out + g * 64);
}

__device__ __noinline__ void sz_tree(uint8_t* tree, const uint8_t* node_heads, int width) {
    int off = 0, lvl_nodes = 0;
    for (int w = width; w > 1; w >>= 1) {
        __syncthreads();
        for (int j = threadIdx.x; j < w / 2; j += blockDim.x)
            sz_frame(node_heads + (long long)(lvl_nodes + j) * SLOT, tree + (long long)(off + 2 * j) * 64,
                     tree + (long long)(off + 2 * j + 1) * 64, tree + (long long)(off + w + j) * 64);
        off += w;
        lvl_nodes += w / 2;
    }
}

// p: digests (row-seg: rows * s * 64) or rows (v2), q_mid (Q's or P's state), tables, salts (rows * 192), hm96 mids (salt
//    prefix, leaf prefix), leaf heads (rows slots), node heads (width - 1 slots), x out, b||c out, tree (level 0 first)
// i: rows, s (0: v2 rows), row stride (v2), mode, unused, unused, width, row bytes (v2)
__device__ void sz_row(const Args& a, long long i) {
    int s = a.i[1], stride = a.i[2], mode = a.i[3];
    const uint64_t* q = (const uint64_t*)a.p[1];
    {
        uint64_t st[8], w[16];
        for (int t = 0; t < 8; t++) st[t] = q[t];
        if (s == 0) {
            int len = a.i[7];
            const uint8_t* r = (const uint8_t*)a.p[0] + i * stride;
            uint8_t blk[128];
            for (int t = 0; t < 128; t++) blk[t] = t < len ? r[t] : (t == len ? 0x80 : 0);
            uint64_t bits = (uint64_t)(128 + len) * 8;
            for (int t = 0; t < 8; t++) blk[127 - t] = (uint8_t)(bits >> (8 * t));
            for (int j = 0; j < 16; j++) {
                uint64_t x = 0;
                for (int t = 0; t < 8; t++) x = (x << 8) | blk[8 * j + t];
                w[j] = x;
            }
            s512_compress(st, w);
        } else {
            const uint4* d = (const uint4*)((const uint8_t*)a.p[0] + i * (long long)s * 64);
            int j = 0;
            for (; j + 1 < s; j += 2) {
                for (int t = 0; t < 8; t++) {
                    uint4 v = d[4 * j + t];
                    w[2 * t] = s512_be(v.x, v.y);
                    w[2 * t + 1] = s512_be(v.z, v.w);
                }
                s512_compress(st, w);
            }
            uint64_t bits = (uint64_t)(128 + 64 * s) * 8;
            if (j < s) {
                for (int t = 0; t < 4; t++) {
                    uint4 v = d[4 * j + t];
                    w[2 * t] = s512_be(v.x, v.y);
                    w[2 * t + 1] = s512_be(v.z, v.w);
                }
                w[8] = 0x8000000000000000ull;
                for (int t = 9; t < 15; t++) w[t] = 0;
                w[15] = bits;
                s512_compress(st, w);
            } else {
                s512_pad_block(st, 128 + 64 * (uint64_t)s);
            }
        }
        uint8_t* xo = (uint8_t*)a.p[7] + i * 64;
        if (mode == 0) {
            rc::st_be(xo, st, 8);
        } else {
            uint32_t yw[48], mk[16];
            const uint32_t* y = (const uint32_t*)a.p[3] + 48 * i;
            for (int t = 0; t < 48; t++) yw[t] = y[t];
            const uint64_t* mid = (const uint64_t*)a.p[4];
            uint64_t B[16], h[8];
            rc::mask_positions(rc::PosTab<true>{(const uint8_t*)a.p[2]}, yw, mk);
            rc::mask_words(mk, B);
            for (int k = 0; k < 8; k++) B[k] ^= st[k];
            rc::salt_digest<SzC>(mid, rc::SaltU32{yw, 1}, B + 8);
            rc::st_be((uint8_t*)a.p[8] + i * 128, B, 16);
            rc::hm96_value<SzC>(mid + 8, sx::kw_pad256(), B, h);
            alignas(16) uint8_t hl[64];
            rc::st_be(hl, h, 8);
            for (int t = 0; t < 64; t++) xo[t] = hl[t];
            if (mode == 2) sz_frame((const uint8_t*)a.p[5] + i * SLOT, hl, nullptr, (uint8_t*)a.p[9] + i * 64);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) sz_finish(Args a) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < a.i[0]; i += (long long)gridDim.x * blockDim.x)
        sz_row(a, i);
    if (a.i[3] == 2 && gridDim.x == 1) sz_tree((uint8_t*)a.p[9], (const uint8_t*)a.p[6], a.i[6]);
}

// p: tree, node heads of this level   i: read offset (nodes), parents
extern "C" __global__ void sz_level(Args a) {
    long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= a.i[1]) return;
    uint8_t* tree = (uint8_t*)a.p[0];
    long long off = a.i[0];
    sz_frame((const uint8_t*)a.p[1] + j * SLOT, tree + (off + 2 * j) * 64, tree + (off + 2 * j + 1) * 64,
             tree + (off + 2 * a.i[1] + j) * 64);
}
