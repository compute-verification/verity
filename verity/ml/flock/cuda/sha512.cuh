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

// leaf_size % 128 == 0: a warp owns 32 consecutive leaves and, per 128-byte chunk (one SHA-512 block), stages the chunk of
// each of its leaves into shared memory with coalesced 16-byte loads (33-word stride: conflict-free), then lane l
// compresses its own leaf's block. The padding block is a constant.
__global__ void sha512_staged_merkle_leaves(const uint8_t* __restrict__ data, uint8_t* tree, long long num_leaves, int leaf_size) {
    extern __shared__ uint32_t sw512[];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t* s = sw512 + wid * (32 * 33);
    long long warp0 = ((long long)blockIdx.x * (blockDim.x >> 5) + wid) * 32;
    if (warp0 >= num_leaves) return;
    long long myleaf = warp0 + lane;
    uint64_t st[8];
    s512_iv(st);
    int nchunks = leaf_size >> 7;
    for (int c = 0; c < nchunks; c++) {
#pragma unroll
        for (int it = 0; it < 8; it++) {
            int idx = it * 32 + lane;
            int piece = idx >> 3, off = idx & 7;
            long long leaf = warp0 + piece;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (leaf < num_leaves) v = *(const uint4*)(data + leaf * (long long)leaf_size + (long long)c * 128 + off * 16);
            uint32_t* d = s + piece * 33 + off * 4;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        __syncwarp();
        if (myleaf < num_leaves) {
            uint64_t w[16];
#pragma unroll
            for (int j = 0; j < 16; j++) w[j] = s512_be(s[lane * 33 + 2 * j], s[lane * 33 + 2 * j + 1]);
            s512_compress(st, w);
        }
        __syncwarp();
    }
    if (myleaf >= num_leaves) return;
    s512_pad_block(st, (uint64_t)leaf_size);
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

// one thread per parent: SHA-512 of the 128 contiguous bytes of its two children
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
    s512_pad_block(st, 128);
    s512_out(st, tree + (write_start + j) * 64);
}

// ---- hm96-sha512/v1 leaves (verity.ml.commitments.hm96, PROTOCOL.md section 7) --------------------------------------------
//   x = the SHA-512 leaf above; y = the leaf's 192-byte salt; c = SHA-512(salt_prefix || y); b = x XOR M(key) y;
//   leaf = SHA-512(leaf_prefix || b || c). Salts are the host's (zk_hooks::LeafSalts, read from the OS): the tree callback hands
//   tree k's salts (192 bytes per leaf, in host memory until the proof ends); the device reads them HM96_CHUNK leaves at a time
//   where it finishes the leaves, and an opening's are copied from the host's.

typedef const uint8_t* (*Hm96TreeCb)(void* ctx, int tree, long long leaves);

static const long long HM96_CHUNK = 1 << 18;   // leaves per upload of a tree's salts (48 MiB on the device)

struct Hm96Ctx {
    Hm96TreeCb tree_cb = nullptr;
    void* ctx = nullptr;
    int trees = 0;
    long long level0_leaves = 0;
    uint64_t* d_tables = nullptr;   // [192][256][8]: per salt byte and value, the XOR of its set bits' columns
    uint64_t* d_mid = nullptr;      // SHA-512 states after the salt prefix block, then after the leaf prefix block
    uint8_t* d_salts = nullptr;     // HM96_CHUNK leaves' salts, zeroed when the proof ends
    std::map<const uint8_t*, std::pair<const uint8_t*, long long>> salts;   // device tree -> its host salts and leaves
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

// the most threads a block of hm96_finish_leaves has
static constexpr int HM96_TPB = 256;

// Leaves [base, n) of `tree`, leaf i's salt at salts[48 (i - base)] (little-endian words, the host's bytes); blocks of at most
// HM96_TPB threads, a multiple of 4, block b finishing the blockDim.x leaves from base + b blockDim.x.
// The mask reads one 64-byte table entry per salt byte, at a random value: four lanes share a leaf and each reads one 16-byte
// quarter of every entry, so a warp's load touches 8 entries' lines rather than 32. Each lane XORs its quarter (the order of
// the XORs changes no bit), the block meets in shared memory, and each thread then hashes its own leaf.
__global__ void __launch_bounds__(HM96_TPB) hm96_finish_leaves(uint8_t* tree, const uint32_t* __restrict__ salts, long long base,
                                                              long long n, const uint64_t* __restrict__ tables,
                                                              const uint64_t* __restrict__ mid) {
    __shared__ ulonglong2 sm[HM96_TPB * 4];
    const long long first = base + blockIdx.x * (long long)blockDim.x;
    const int k = threadIdx.x & 3;
    for (int l = threadIdx.x >> 2; l < (int)blockDim.x; l += blockDim.x >> 2) {
        ulonglong2 acc = make_ulonglong2(0, 0);
        if (first + l < n) {
            const uint4* sy = (const uint4*)(salts + 48 * (first + l - base));
            // salt byte p is byte p % 4 of word p / 4 (little-endian); unrolled, so the table rows' loads are independent
#pragma unroll
            for (int q = 0; q < 12; q++) {
                const uint4 v = __ldg(sy + q);
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const int p = 16 * q + j;
                    const ulonglong2 e =
                        __ldg((const ulonglong2*)(tables + ((size_t)p * 256 + ((w[j >> 2] >> (8 * (j & 3))) & 0xff)) * 8) + k);
                    acc.x ^= e.x;
                    acc.y ^= e.y;
                }
            }
        }
        sm[4 * l + k] = acc;
    }
    __syncthreads();
    const long long i = first + threadIdx.x;
    if (i >= n) return;
    uint32_t yw[48];
    const uint4* sy = (const uint4*)(salts + 48 * (i - base));
#pragma unroll
    for (int q = 0; q < 12; q++) {
        const uint4 v = __ldg(sy + q);
        yw[4 * q] = v.x; yw[4 * q + 1] = v.y; yw[4 * q + 2] = v.z; yw[4 * q + 3] = v.w;
    }
    const uint64_t* x = (const uint64_t*)(tree + i * 64);   // the digest's bytes as little-endian words, as the mask's limbs
    uint64_t b[8];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const ulonglong2 v = sm[4 * threadIdx.x + q];
        b[2 * q] = x[2 * q] ^ v.x;
        b[2 * q + 1] = x[2 * q + 1] ^ v.y;
    }
    uint64_t st[8], w[16];
#pragma unroll
    for (int j = 0; j < 8; j++) st[j] = mid[j];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = s512_be(yw[2 * j], yw[2 * j + 1]);
    s512_compress(st, w);
#pragma unroll
    for (int j = 0; j < 8; j++) w[j] = s512_be(yw[32 + 2 * j], yw[32 + 2 * j + 1]);
    w[8] = 0x8000000000000000ull;
#pragma unroll
    for (int j = 9; j < 15; j++) w[j] = 0;
    w[15] = (128 + 192) * 8;
    s512_compress(st, w);                                   // st = c's words
    uint64_t lt[8];
#pragma unroll
    for (int j = 0; j < 8; j++) lt[j] = mid[8 + j];
#pragma unroll
    for (int k = 0; k < 8; k++) { w[k] = s512_bswap(b[k]); w[8 + k] = st[k]; }
    s512_compress(lt, w);
    s512_pad_block(lt, 256);
    s512_out(lt, tree + i * 64);
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

// One proof's hm96 context (a null tree callback leaves the build unsalted); the device's copy of the salts is zeroed when it
// ends. The context is process-global: a second install while one is live is refused (two proofs would share a tree counter and
// level 0).
struct Hm96Guard {
    bool mine = false;
    Hm96Guard(Hm96TreeCb tree_cb, void* ctx, const uint8_t* prefixes, const uint64_t* tables, long long level0_leaves) {
        if (!tree_cb) return;
        if (g_hm96.tree_cb) { fprintf(stderr, "hm96: a salt context is already installed (one proof at a time per process)\n"); abort(); }
        mine = true;
        g_hm96.tree_cb = tree_cb; g_hm96.ctx = ctx; g_hm96.trees = 0; g_hm96.salts.clear(); g_hm96.level0_leaves = level0_leaves;
        cudaMalloc(&g_hm96.d_salts, (size_t)HM96_CHUNK * 192);
        size_t tb = (size_t)192 * 256 * 8 * sizeof(uint64_t);
        cudaMalloc(&g_hm96.d_tables, tb);
        cudaMemcpy(g_hm96.d_tables, tables, tb, cudaMemcpyHostToDevice);
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
        if (g_hm96.d_tables) cudaFree(g_hm96.d_tables);
        if (g_hm96.d_mid) cudaFree(g_hm96.d_mid);
        if (g_hm96.d_salts) { cudaMemset(g_hm96.d_salts, 0, (size_t)HM96_CHUNK * 192); cudaFree(g_hm96.d_salts); }
        g_hm96.tree_cb = nullptr; g_hm96.ctx = nullptr; g_hm96.d_tables = nullptr; g_hm96.d_mid = nullptr; g_hm96.d_salts = nullptr;
        g_hm96.trees = 0; g_hm96.salts.clear(); g_hm96.level0_leaves = 0;
    }
};

inline void launch_merkle512(const uint8_t* d_data, uint8_t* d_tree, long long num_leaves, int leaf_size, int tpb = 256) {
    if (leaf_size % 128 == 0 && ((uintptr_t)d_data & 15) == 0) {
        int wpb = tpb >> 5;
        long long warps = (num_leaves + 31) / 32;
        long long blocks = (warps + wpb - 1) / wpb;
        size_t smem = (size_t)wpb * 32 * 33 * sizeof(uint32_t);
        sha512_staged_merkle_leaves<<<(unsigned)blocks, tpb, smem>>>(d_data, d_tree, num_leaves, leaf_size);
    } else {
        long long blocks = (num_leaves + tpb - 1) / tpb;
        sha512_merkle_leaves_bytes<<<(unsigned)blocks, tpb>>>(d_data, d_tree, num_leaves, leaf_size);
    }
    if (g_hm96.tree_cb) {
        const uint8_t* h = hm96_tree_salts(g_hm96.trees++, num_leaves);
        // each piece's copy waits for the previous piece's kernel (one stream), so one device buffer serves the tree
        for (long long at = 0; at < num_leaves; at += HM96_CHUNK) {
            const long long n = num_leaves - at < HM96_CHUNK ? num_leaves - at : HM96_CHUNK;
            cudaMemcpy(g_hm96.d_salts, h + 192 * at, (size_t)n * 192, cudaMemcpyHostToDevice);
            hm96_finish_leaves<<<(unsigned)((n + tpb - 1) / tpb), tpb>>>(d_tree, (const uint32_t*)g_hm96.d_salts, at, at + n,
                                                                         g_hm96.d_tables, g_hm96.d_mid);
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
