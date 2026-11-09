// SHA-512 Merkle trees for verity/flock-circuit (FIPS 180-4; installed by verity/ml/flock/cuda_sha512_patch.py, with the
// GPU's one SHA-512, verity/ml/kernels/sha512.cuh, beside it).
//   leaf i = SHA-512(bytes [i*leaf_size, (i+1)*leaf_size)),  node = SHA-512(left || right),  64-byte digests in standard
//   (big-endian) byte order; the flat layout of merkle.cuh: leaves, then each level above, root last (2n-1 nodes).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "../../kernels/sha512.cuh"

// the big-endian 64-bit word of 8 message bytes held as two little-endian u32 loads
__device__ __forceinline__ uint64_t s512_be(uint32_t a, uint32_t b) { return sx::be64(a, b); }

__device__ __forceinline__ uint64_t s512_bswap(uint64_t x) { return sx::bswap64(x); }

__device__ __forceinline__ void s512_iv(uint64_t st[8]) { sx::iv(st); }

__device__ __forceinline__ void s512_compress(uint64_t st[8], uint64_t w[16]) { sx::compress<0, 1>(st, w); }

// the digest's 64 bytes, big-endian words (out 8-byte aligned: tree nodes are at 64-byte strides)
__device__ __forceinline__ void s512_out(const uint64_t st[8], uint8_t* out) {
    uint64_t* o = (uint64_t*)out;
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = s512_bswap(st[j]);
}

// the padding block of a message of `len` bytes, len % 128 == 0
__device__ __forceinline__ void s512_pad_block(uint64_t st[8], uint64_t len) {
    uint64_t w[16];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = len * 8;
    s512_compress(st, w);
}

// the last block of a message of `len` bytes whose 0x80 left no room for the length (len % 128 > 111)
__device__ __forceinline__ void s512_len_block(uint64_t st[8], uint64_t len) {
    uint64_t w[16];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = 0;
    w[15] = len * 8;
    s512_compress(st, w);
}

// leaf_size % 16 == 0: a warp owns 32 consecutive leaves and, per 128-byte chunk (one SHA-512 block), stages the chunk of
// each of its leaves into shared memory with coalesced 16-byte loads (33-word stride: conflict-free), then lane l
// compresses its own leaf's block. A tail of r = leaf_size % 128 bytes is staged the same way, zero past r, and its block
// takes the padding: 0x80 opens word r/8, and the bit length is word 15 (r <= 111) or a last block of its own. With no
// tail the padding block is a constant. Blocks of at most S512_LEAVES_TPB threads, two an SM.
static constexpr int S512_LEAVES_TPB = 256;
__global__ void __launch_bounds__(S512_LEAVES_TPB, 2) sha512_staged_merkle_leaves(const uint8_t* __restrict__ data, uint8_t* tree,
                                                                                  long long num_leaves, int leaf_size) {
    extern __shared__ uint32_t sw512[];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t* s = sw512 + wid * (32 * 33);
    long long warp0 = ((long long)blockIdx.x * (blockDim.x >> 5) + wid) * 32;
    if (warp0 >= num_leaves) return;
    long long myleaf = warp0 + lane;
    uint64_t st[8];
    s512_iv(st);
    int nchunks = leaf_size >> 7, r = leaf_size & 127;
    for (int c = 0; c < nchunks + (r != 0); c++) {
        const int bytes = c < nchunks ? 128 : r, pad = c < nchunks ? 16 : r >> 3;
#pragma unroll
        for (int it = 0; it < 8; it++) {
            int idx = it * 32 + lane;
            int piece = idx >> 3, off = idx & 7;
            long long leaf = warp0 + piece;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (leaf < num_leaves && off * 16 < bytes)
                v = *(const uint4*)(data + leaf * (long long)leaf_size + (long long)c * 128 + off * 16);
            uint32_t* d = s + piece * 33 + off * 4;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        __syncwarp();
        if (myleaf < num_leaves) {
            uint64_t w[16];
#pragma unroll
            for (int j = 0; j < 16; j++)
                w[j] = j == pad ? 0x8000000000000000ull : s512_be(s[lane * 33 + 2 * j], s[lane * 33 + 2 * j + 1]);
            if (c == nchunks && r <= 111) w[15] = (uint64_t)leaf_size * 8;
            s512_compress(st, w);
        }
        __syncwarp();
    }
    if (myleaf >= num_leaves) return;
    if (r == 0) s512_pad_block(st, (uint64_t)leaf_size);
    if (r > 111) s512_len_block(st, (uint64_t)leaf_size);
    s512_out(st, tree + myleaf * 64);
}

// any leaf size: one thread per leaf, byte loads, the padding built in place
__global__ void sha512_merkle_leaves_bytes(const uint8_t* __restrict__ data, uint8_t* tree, long long num_leaves, int leaf_size) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= num_leaves) return;
    const uint8_t* m = data + i * (long long)leaf_size;
    long long len = leaf_size;
    long long total = ((len + 17 + 127) / 128) * 128;
    uint64_t st[8];
    s512_iv(st);
    for (long long b = 0; b < total; b += 128) {
        uint64_t w[16];
        for (int j = 0; j < 16; j++) {
            uint64_t x = 0;
            for (int k = 0; k < 8; k++) {
                long long t = b + 8 * j + k;
                uint8_t byte = t < len ? m[t] : (t == len ? 0x80 : 0);
                x = (x << 8) | byte;
            }
            w[j] = x;
        }
        if (b + 128 == total) w[15] = (uint64_t)len * 8;
        s512_compress(st, w);
    }
    s512_out(st, tree + i * 64);
}

// one thread per parent: SHA-512 of the 128 contiguous bytes of its two children, the padding chunk over its precomputed
// schedule
__global__ void sha512_merkle_level(uint8_t* tree, long long read_start, long long write_start, long long num_parents) {
    long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= num_parents) return;
    const uint4* ch = (const uint4*)(tree + (read_start + 2 * j) * 64);
    uint64_t st[8], w[16];
    s512_iv(st);
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint4 v = ch[q];
        w[2 * q] = s512_be(v.x, v.y);
        w[2 * q + 1] = s512_be(v.z, v.w);
    }
    s512_compress(st, w);
    sx::compress_kw<0, 1>(st, sx::kw_pad128());
    s512_out(st, tree + (write_start + j) * 64);
}

// ---- hm96-sha512/v1 leaves (verity.ml.commitments.hm96, PROTOCOL.md section 7) --------------------------------------------
//   x = the SHA-512 leaf above; y = the leaf's 192-byte salt; c = SHA-512(salt_prefix || y); b = x XOR M(key) y;
//   leaf = SHA-512(leaf_prefix || b || c). Salts are the host's (zk_hooks::LeafSalts, read from the OS): the tree callback hands
//   tree k's salts (192 bytes per leaf, in host memory until the proof ends); a salted tree's leaves are hashed and finished
//   HM96_CHUNK at a time, each piece's salts copied on a stream of their own into the next of HM96_RING device buffers beside
//   the previous piece's leaves, unless the tree's salts were copied ahead of its build (hm96_copy_ahead); an opening's are
//   copied from the host's.

typedef const uint8_t* (*Hm96TreeCb)(void* ctx, int tree, long long leaves);

static const long long HM96_CHUNK = 1 << 18;   // leaves per piece of a salted tree (48 MiB of salts on the device)
static const int HM96_RING = 2;                 // device buffers of a piece's salts: one copied while the other's piece runs
// hm96_mask_leaves' B fragments: (mask word, n-tile, k-tile, lane) of 8 x 8 x 6 x 32, two words each
static constexpr int HM96_FRAGS = 8 * 8 * 6 * 32;

// a tree's salts copied to the device before its build (hm96_copy_ahead): the host's, the device's copy, and its copy's end
struct Hm96Ahead {
    const uint8_t* h;
    const uint8_t* d;
    long long leaves;
    cudaEvent_t copied;
};

struct Hm96Ctx {
    Hm96TreeCb tree_cb = nullptr;
    void* ctx = nullptr;
    int trees = 0;
    long long level0_leaves = 0;
    uint32_t* d_frag = nullptr;     // hm96_fragments of the key's tables
    uint64_t* d_mid = nullptr;      // SHA-512 states after the salt prefix block, then after the leaf prefix block
    uint8_t* d_salts = nullptr;     // HM96_RING buffers of HM96_CHUNK leaves' salts, zeroed when the proof ends
    cudaStream_t copy = nullptr;    // the salts' copies (non-blocking: beside the kernels on the default stream)
    cudaEvent_t copied[HM96_RING] = {};   // buffer r's piece is on the device
    cudaEvent_t done[HM96_RING] = {};     // buffer r's piece is finished, so the buffer can take the next
    std::map<const uint8_t*, std::pair<const uint8_t*, long long>> salts;   // device tree -> its host salts and leaves
    std::map<int, Hm96Ahead> ahead;   // tree k -> its salts copied ahead, until its build takes them
};
static Hm96Ctx g_hm96;

__global__ void hm96_midstates(const uint8_t* prefixes, uint64_t* mid) {
    for (int k = 0; k < 2; k++) {
        uint64_t st[8], w[16];
        s512_iv(st);
        const uint32_t* p = (const uint32_t*)(prefixes + 128 * k);
        for (int j = 0; j < 16; j++) w[j] = s512_be(p[2 * j], p[2 * j + 1]);
        s512_compress(st, w);
        for (int j = 0; j < 8; j++) mid[8 * k + j] = st[j];
    }
}

// hm96_mask_leaves' B fragments from the key's tables ([192][256][8] u64: per salt byte and value, the XOR of its set bits'
// columns, little-endian words), on the host. M's column c, salt bit c (bit c % 8 of byte c / 8, bit c % 32 of the salt's
// little-endian word c / 32), is the entry of byte c / 8 at the value with that bit alone; the entries are linear in the value,
// so the columns give every entry. Fragment f = ((w * 8 + t) * 6 + kt) * 32 + 4 g + q holds lane 4g + q's (b0, b1) of k-tile kt
// of n-tile t of mask word w: that n-tile's column g is the mask's bit o = 64 w + 16 (g >> 1) + 2 t + (g & 1), and b0 / b1 are
// M's row o over salt words 12 q + 2 kt and 12 q + 2 kt + 1, the k bits hm96_mask_leaves puts in a0 / a2.
inline void hm96_fragments(const uint64_t* tables, uint32_t* frag) {
    for (int f = 0; f < HM96_FRAGS; f++) {
        const int lane = f & 31, kt = (f >> 5) % 6, t = (f / (32 * 6)) & 7, w = f / (32 * 6 * 8);
        const int g = lane >> 2, q = lane & 3, o = 64 * w + 16 * (g >> 1) + 2 * t + (g & 1);
        for (int s = 0; s < 2; s++) {
            uint32_t r = 0;
            for (int i = 0; i < 32; i++) {
                const int c = 32 * (12 * q + 2 * kt + s) + i;
                r |= (uint32_t)(tables[((size_t)(c >> 3) * 256 + (1u << (c & 7))) * 8 + (o >> 6)] >> (o & 63) & 1) << i;
            }
            frag[2 * f + s] = r;
        }
    }
}

// m16n8k256 b1 AND-popcount accumulating into d, with the PTX ISA's fragment layout (lane = 4g + q: a0/a2 row g, a1/a3 row
// g + 8, a0/a1 bits 32q.. of k, a2/a3 bits 128 + 32q..; b0/b1 column g, the same bits; d0/d1 row g, columns 2q and 2q + 1,
// d2/d3 row g + 8)
__device__ __forceinline__ void hm96_bmma_acc(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// Leaves [base, n) of `tree`: M(key) y of leaf i's salt y (at salts[48 (i - base)]) XORed into its digest's little-endian
// words, b = x XOR M(key) y in place. M y over GF(2) is the parity of AND-popcounts: warp w of a 256-thread block owns mask word
// w and holds its B fragments (hm96_fragments) in registers, and takes the block's tiles of 16 leaves in turn, A being their
// salts (lane 4g + q: leaves g and g + 8, salt words 12 q .. 12 q + 11, two a k-tile). Six k-tiles accumulate an n-tile, whose
// counts' low bits lane q keeps as bits 16 q + 2 t, 16 q + 2 t + 1 of the word; a quad's lanes meet by shuffles, and lanes 0
// and 1 of the quad XOR leaf g's and leaf g + 8's word into the tree.
static constexpr int HM96_MASK_TPB = 256;
__global__ void __launch_bounds__(HM96_MASK_TPB, 1) hm96_mask_leaves(uint8_t* tree, const uint32_t* __restrict__ salts, long long base,
                                                                    long long n, const uint32_t* __restrict__ frag) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
    uint32_t bf[8][6][2];
#pragma unroll
    for (int t = 0; t < 8; t++)
#pragma unroll
        for (int kt = 0; kt < 6; kt++) {
            const uint2 v = __ldg((const uint2*)frag + ((w * 8 + t) * 6 + kt) * 32 + lane);
            bf[t][kt][0] = v.x;
            bf[t][kt][1] = v.y;
        }
    const long long tiles = (n - base + 15) / 16;
    // tile mt's salt words of leaves g and g + 8 (zero past n)
    auto load = [&](long long mt, uint32_t* a, uint32_t* b) {
        const long long l0 = base + 16 * mt + g, l1 = l0 + 8;
#pragma unroll
        for (int h = 0; h < 3; h++) {
            const uint4 u = l0 < n ? __ldg((const uint4*)(salts + 48 * (l0 - base) + 12 * q) + h) : make_uint4(0, 0, 0, 0);
            const uint4 v = l1 < n ? __ldg((const uint4*)(salts + 48 * (l1 - base) + 12 * q) + h) : make_uint4(0, 0, 0, 0);
            a[4 * h] = u.x; a[4 * h + 1] = u.y; a[4 * h + 2] = u.z; a[4 * h + 3] = u.w;
            b[4 * h] = v.x; b[4 * h + 1] = v.y; b[4 * h + 2] = v.z; b[4 * h + 3] = v.w;
        }
    };
    uint32_t s0[12] = {}, s1[12] = {};
    if (blockIdx.x < tiles) load(blockIdx.x, s0, s1);
    for (long long mt = blockIdx.x; mt < tiles; mt += gridDim.x) {
        const long long l0 = base + 16 * mt + g, l1 = l0 + 8;
        // the next tile's salts are loaded while this tile's products run (one block an SM hides little latency otherwise)
        uint32_t n0[12] = {}, n1[12] = {};
        if (mt + gridDim.x < tiles) load(mt + gridDim.x, n0, n1);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int t = 0; t < 8; t++) {
            uint32_t d[4] = {0, 0, 0, 0};
#pragma unroll
            for (int kt = 0; kt < 6; kt++)
                hm96_bmma_acc(s0[2 * kt], s1[2 * kt], s0[2 * kt + 1], s1[2 * kt + 1], bf[t][kt][0], bf[t][kt][1], d);
            lo |= (d[0] & 1) << (2 * t) | (d[1] & 1) << (2 * t + 1);
            hi |= (d[2] & 1) << (2 * t) | (d[3] & 1) << (2 * t + 1);
        }
        // lanes q = 0, 1 hold the word's bits 0..31 in halves, q = 2, 3 its bits 32..63
        if (q & 1) { lo <<= 16; hi <<= 16; }
        lo |= __shfl_xor_sync(0xffffffffu, lo, 1);
        hi |= __shfl_xor_sync(0xffffffffu, hi, 1);
        const uint32_t loh = __shfl_xor_sync(0xffffffffu, lo, 2), hih = __shfl_xor_sync(0xffffffffu, hi, 2);
        if (q == 0 && l0 < n) ((uint64_t*)(tree + l0 * 64))[w] ^= (uint64_t)loh << 32 | lo;
        if (q == 1 && l1 < n) ((uint64_t*)(tree + l1 * 64))[w] ^= (uint64_t)hih << 32 | hi;
#pragma unroll
        for (int j = 0; j < 12; j++) { s0[j] = n0[j]; s1[j] = n1[j]; }
    }
}

// the most threads a block of hm96_finish_leaves has
static constexpr int HM96_TPB = 256;
// the block sizes hm96_finish_leaves takes
inline bool hm96_tpb_fits(int tpb) { return tpb >= 32 && tpb <= HM96_TPB && tpb % 32 == 0; }

// Leaves [base, n) of `tree` hashed from b (the digest with its mask in, hm96_mask_leaves) and the salt, leaf i's at
// salts[48 (i - base)] (little-endian words, the host's bytes): c from the salt prefix's midstate, then the leaf from the leaf
// prefix's over b || c, its padding chunk over the precomputed schedule. One thread a leaf, blocks of at most HM96_TPB threads,
// a multiple of 32.
__global__ void __launch_bounds__(HM96_TPB, 3) hm96_finish_leaves(uint8_t* tree, const uint32_t* __restrict__ salts,
                                                                 long long base, long long n, const uint64_t* __restrict__ mid) {
    const long long i = base + blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    // the salt's words are read again just before each compression and b only for the last, so few stay live across the hash
    const uint4* sy = (const uint4*)(salts + 48 * (i - base));
    uint64_t st[8], w[16];
#pragma unroll
    for (int j = 0; j < 8; j++) st[j] = mid[j];
#pragma unroll
    for (int q = 0; q < 8; q++) {
        const uint4 v = __ldg(sy + q);
        w[2 * q] = s512_be(v.x, v.y);
        w[2 * q + 1] = s512_be(v.z, v.w);
    }
    s512_compress(st, w);
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const uint4 v = __ldg(sy + 8 + q);
        w[2 * q] = s512_be(v.x, v.y);
        w[2 * q + 1] = s512_be(v.z, v.w);
    }
    w[8] = 0x8000000000000000ull;
#pragma unroll
    for (int j = 9; j < 15; j++) w[j] = 0;
    w[15] = (128 + 192) * 8;
    s512_compress(st, w);                                   // st = c's words
    const ulonglong2* x = (const ulonglong2*)(tree + i * 64);
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const ulonglong2 a = x[q];
        w[2 * q] = s512_bswap(a.x);
        w[2 * q + 1] = s512_bswap(a.y);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) { w[8 + j] = st[j]; st[j] = mid[8 + j]; }
    s512_compress(st, w);
    sx::compress_kw<0, 1>(st, sx::kw_pad256());
    s512_out(st, tree + i * 64);
}

// the SMs' count, for hm96_mask_leaves' grid (one block an SM)
inline int hm96_sms() {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms < 1) sms = 1;
    }
    return sms;
}

// Leaves [base, n) of `tree` finished on `stream`: their masks (hm96_mask_leaves, over the key's fragments `frag` on the device),
// then their hashes at tpb threads a block (hm96_tpb_fits).
inline void launch_hm96_finish(uint8_t* tree, const uint32_t* salts, long long base, long long n, int tpb, const uint32_t* frag,
                               const uint64_t* mid, cudaStream_t stream = 0) {
    if (n <= base) return;
    const long long tiles = (n - base + 15) / 16, grid = tiles < hm96_sms() ? tiles : hm96_sms();
    hm96_mask_leaves<<<(unsigned)grid, HM96_MASK_TPB, 0, stream>>>(tree, salts, base, n, frag);
    hm96_finish_leaves<<<(unsigned)((n - base + tpb - 1) / tpb), tpb, 0, stream>>>(tree, salts, base, n, mid);
}

// The salts of `d_tree`'s leaves at `q` (192 bytes each, in order), copied from the host's; empty when unsalted.
inline std::vector<uint8_t> hm96_opened_salts(const uint8_t* d_tree, const std::vector<size_t>& q) {
    std::vector<uint8_t> out;
    if (!g_hm96.tree_cb) return out;
    auto it = g_hm96.salts.find(d_tree);
    if (it == g_hm96.salts.end()) { fprintf(stderr, "hm96: an opening of a tree built without salts\n"); abort(); }
    out.resize(q.size() * 192);
    for (size_t k = 0; k < q.size(); k++) {
        if ((long long)q[k] >= it->second.second) { fprintf(stderr, "hm96: an opening past the tree's leaves\n"); abort(); }
        memcpy(out.data() + 192 * k, it->second.first + 192 * q[k], 192);
    }
    return out;
}

// Tree k's salts from the host (the level-0 tree's for k = 0: only a tree of its size takes them).
inline const uint8_t* hm96_tree_salts(int k, long long num_leaves) {
    if (k == 0 && num_leaves != g_hm96.level0_leaves) {
        fprintf(stderr, "hm96: tree 0 takes level 0's salts (%lld leaves), refused for a tree of %lld\n", g_hm96.level0_leaves, num_leaves);
        abort();
    }
    const uint8_t* h = g_hm96.tree_cb(g_hm96.ctx, k, num_leaves);
    if (!h) { fprintf(stderr, "hm96: the host gave no salts for tree %d\n", k); abort(); }
    return h;
}

// the salts' copy stream and its events, destroyed: each piece's salts are then copied on the default stream
inline void hm96_drop_copy_stream() {
    for (int r = 0; r < HM96_RING; r++) {
        if (g_hm96.copied[r]) cudaEventDestroy(g_hm96.copied[r]);
        if (g_hm96.done[r]) cudaEventDestroy(g_hm96.done[r]);
        g_hm96.copied[r] = g_hm96.done[r] = nullptr;
    }
    if (g_hm96.copy) cudaStreamDestroy(g_hm96.copy);
    g_hm96.copy = nullptr;
}

// The next trees' salts, tree g_hm96.trees + j's leaves[j] x 192 bytes, taken from the host now in tree order and copied on the
// copy stream into d[j] once the default stream's work so far is done (d[j] may be memory that work releases), so each of
// those trees' builds waits only for its own copy. They must then be built in that order at those sizes, each d[j] kept until
// its tree is built and zeroed by the caller. False, nothing taken, without a salt context, its copy stream, or level 0 built.
inline bool hm96_copy_ahead(const std::vector<long long>& leaves, const std::vector<uint8_t*>& d) {
    if (!g_hm96.tree_cb || !g_hm96.copy || g_hm96.trees < 1 || leaves.size() != d.size()) return false;
    std::vector<cudaEvent_t> ev(leaves.size() + 1, nullptr);
    for (size_t j = 0; j < ev.size(); j++) {
        if (cudaEventCreateWithFlags(&ev[j], cudaEventDisableTiming) == cudaSuccess) continue;
        cudaGetLastError();
        for (size_t i = 0; i < j; i++) cudaEventDestroy(ev[i]);
        return false;
    }
    cudaEventRecord(ev[0], 0);
    cudaStreamWaitEvent(g_hm96.copy, ev[0], 0);
    cudaEventDestroy(ev[0]);
    for (size_t j = 0; j < leaves.size(); j++) {
        const int k = g_hm96.trees + (int)j;
        const Hm96Ahead a{hm96_tree_salts(k, leaves[j]), d[j], leaves[j], ev[j + 1]};
        cudaMemcpyAsync(d[j], a.h, (size_t)leaves[j] * 192, cudaMemcpyHostToDevice, g_hm96.copy);
        cudaEventRecord(a.copied, g_hm96.copy);
        g_hm96.ahead[k] = a;
    }
    return true;
}

// One proof's hm96 context (a null tree callback leaves the build unsalted); the device's copy of the salts is zeroed when it
// ends. The context is process-global: a second install while one is live is refused (two proofs would share a tree counter and
// level 0).
struct Hm96Guard {
    bool mine = false;
    Hm96Guard(Hm96TreeCb tree_cb, void* ctx, const uint8_t* prefixes, const uint64_t* tables, long long level0_leaves) {
        if (!tree_cb) return;
        if (g_hm96.tree_cb) { fprintf(stderr, "hm96: a salt context is already installed (one proof at a time per process)\n"); abort(); }
        mine = true;
        g_hm96.tree_cb = tree_cb; g_hm96.ctx = ctx; g_hm96.trees = 0; g_hm96.salts.clear(); g_hm96.ahead.clear();
        g_hm96.level0_leaves = level0_leaves;
        cudaMalloc(&g_hm96.d_salts, (size_t)HM96_RING * HM96_CHUNK * 192);
        bool ok = cudaStreamCreateWithFlags(&g_hm96.copy, cudaStreamNonBlocking) == cudaSuccess;
        for (int r = 0; r < HM96_RING; r++)
            ok = ok && cudaEventCreateWithFlags(&g_hm96.copied[r], cudaEventDisableTiming) == cudaSuccess
                    && cudaEventCreateWithFlags(&g_hm96.done[r], cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            hm96_drop_copy_stream();
        }
        std::vector<uint32_t> frag(2 * HM96_FRAGS);
        hm96_fragments(tables, frag.data());
        cudaMalloc(&g_hm96.d_frag, frag.size() * sizeof(uint32_t));
        cudaMemcpy(g_hm96.d_frag, frag.data(), frag.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        uint8_t* d_pre = nullptr;
        cudaMalloc(&d_pre, 256);
        cudaMemcpy(d_pre, prefixes, 256, cudaMemcpyHostToDevice);
        cudaMalloc(&g_hm96.d_mid, 16 * sizeof(uint64_t));
        hm96_midstates<<<1, 1>>>(d_pre, g_hm96.d_mid);
        cudaDeviceSynchronize();
        cudaFree(d_pre);
    }
    ~Hm96Guard() {
        if (!mine) return;
        if (g_hm96.d_frag) cudaFree(g_hm96.d_frag);
        if (g_hm96.d_mid) cudaFree(g_hm96.d_mid);
        if (g_hm96.copy) cudaStreamSynchronize(g_hm96.copy);
        for (auto& a : g_hm96.ahead) cudaEventDestroy(a.second.copied);
        g_hm96.ahead.clear();
        hm96_drop_copy_stream();
        if (g_hm96.d_salts) { cudaMemset(g_hm96.d_salts, 0, (size_t)HM96_RING * HM96_CHUNK * 192); cudaFree(g_hm96.d_salts); }
        g_hm96.tree_cb = nullptr; g_hm96.ctx = nullptr; g_hm96.d_frag = nullptr; g_hm96.d_mid = nullptr; g_hm96.d_salts = nullptr;
        g_hm96.trees = 0; g_hm96.salts.clear(); g_hm96.level0_leaves = 0;
    }
};

inline void launch_merkle512(const uint8_t* d_data, uint8_t* d_tree, long long num_leaves, int leaf_size, int tpb = 256) {
    // leaves [at, at + n): each a function of its own bytes, so a tree hashed in pieces has the same leaves
    auto leaves = [&](long long at, long long n) {
        const uint8_t* data = d_data + at * (long long)leaf_size;
        if (leaf_size % 16 == 0 && ((uintptr_t)d_data & 15) == 0) {
            int lt = tpb < S512_LEAVES_TPB ? tpb : S512_LEAVES_TPB, wpb = lt >> 5;
            long long warps = (n + 31) / 32;
            long long blocks = (warps + wpb - 1) / wpb;
            size_t smem = (size_t)wpb * 32 * 33 * sizeof(uint32_t);
            sha512_staged_merkle_leaves<<<(unsigned)blocks, lt, smem>>>(data, d_tree + at * 64, n, leaf_size);
        } else {
            long long blocks = (n + tpb - 1) / tpb;
            sha512_merkle_leaves_bytes<<<(unsigned)blocks, tpb>>>(data, d_tree + at * 64, n, leaf_size);
        }
    };
    if (!g_hm96.tree_cb) {
        leaves(0, num_leaves);
    } else {
        if (!hm96_tpb_fits(tpb)) {
            fprintf(stderr, "hm96: launch_merkle512 at %d threads a block, which hm96_finish_leaves can't take: pass a multiple of 32 "
                            "up to HM96_TPB (%d), such as the default 256\n", tpb, HM96_TPB);
            abort();
        }
        const int k = g_hm96.trees++;
        const auto pre = g_hm96.ahead.find(k);
        const Hm96Ahead a = pre == g_hm96.ahead.end() ? Hm96Ahead{nullptr, nullptr, 0, nullptr} : pre->second;
        if (a.h) {
            g_hm96.ahead.erase(pre);
            if (a.leaves != num_leaves) {
                fprintf(stderr, "hm96: tree %d's salts were copied ahead for %lld leaves, refused for a tree of %lld\n", k, a.leaves,
                        num_leaves);
                abort();
            }
            leaves(0, num_leaves);
            cudaStreamWaitEvent(0, a.copied, 0);
            launch_hm96_finish(d_tree, (const uint32_t*)a.d, 0, num_leaves, tpb, g_hm96.d_frag, g_hm96.d_mid);
            cudaEventDestroy(a.copied);
        }
        const uint8_t* h = a.h ? a.h : hm96_tree_salts(k, num_leaves);
        // piece j's salts go into buffer j % HM96_RING once that buffer's last piece is finished, and are copied beside piece
        // j - 1's leaves and finishing and piece j's leaves; the default stream finishes piece j once they are on the device
        for (long long at = 0, j = 0; !a.h && at < num_leaves; at += HM96_CHUNK, j++) {
            const long long n = num_leaves - at < HM96_CHUNK ? num_leaves - at : HM96_CHUNK;
            const int r = (int)(j % HM96_RING);
            uint8_t* d_s = g_hm96.d_salts + (size_t)r * HM96_CHUNK * 192;
            if (g_hm96.copy) {
                cudaStreamWaitEvent(g_hm96.copy, g_hm96.done[r], 0);
                cudaMemcpyAsync(d_s, h + 192 * at, (size_t)n * 192, cudaMemcpyHostToDevice, g_hm96.copy);
                cudaEventRecord(g_hm96.copied[r], g_hm96.copy);
            } else {
                cudaMemcpy(d_s, h + 192 * at, (size_t)n * 192, cudaMemcpyHostToDevice);
            }
            leaves(at, n);
            if (g_hm96.copy) cudaStreamWaitEvent(0, g_hm96.copied[r], 0);
            launch_hm96_finish(d_tree, (const uint32_t*)d_s, at, at + n, tpb, g_hm96.d_frag, g_hm96.d_mid);
            if (g_hm96.copy) cudaEventRecord(g_hm96.done[r], 0);
        }
        g_hm96.salts[d_tree] = {h, num_leaves};
    }
    long long read_start = 0, read_len = num_leaves;
    while (read_len > 1) {
        long long next = read_len >> 1;
        long long write_start = read_start + read_len;
        long long b = (next + tpb - 1) / tpb;
        sha512_merkle_level<<<(unsigned)b, tpb>>>(d_tree, read_start, write_start, next);
        read_start += read_len;
        read_len = next;
    }
}
