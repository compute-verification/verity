// SHA-512 Merkle trees for verity/flock-circuit (FIPS 180-4; installed by backends/flock/cuda_sha512_patch.py).
//   leaf i = SHA-512(bytes [i*leaf_size, (i+1)*leaf_size)),  node = SHA-512(left || right),  64-byte digests in standard
//   (big-endian) byte order; the flat layout of merkle.cuh: leaves, then each level above, root last (2n-1 nodes).
#pragma once
#include <cstdint>

__constant__ uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
    0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
    0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
    0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
    0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
    0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
    0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
    0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
    0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
    0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
    0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
    0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
    0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};

__device__ __forceinline__ uint64_t s512_rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

// the big-endian 64-bit word of 8 message bytes held as two little-endian u32 loads
__device__ __forceinline__ uint64_t s512_be(uint32_t a, uint32_t b) {
    return ((uint64_t)__byte_perm(a, 0, 0x0123) << 32) | __byte_perm(b, 0, 0x0123);
}

__device__ __forceinline__ uint64_t s512_bswap(uint64_t x) {
    return ((uint64_t)__byte_perm((uint32_t)x, 0, 0x0123) << 32) | __byte_perm((uint32_t)(x >> 32), 0, 0x0123);
}

__device__ __forceinline__ void s512_iv(uint64_t st[8]) {
    st[0] = 0x6a09e667f3bcc908ull; st[1] = 0xbb67ae8584caa73bull; st[2] = 0x3c6ef372fe94f82bull; st[3] = 0xa54ff53a5f1d36f1ull;
    st[4] = 0x510e527fade682d1ull; st[5] = 0x9b05688c2b3e6c1full; st[6] = 0x1f83d9abfb41bd6bull; st[7] = 0x5be0cd19137e2179ull;
}

__device__ __forceinline__ void s512_compress(uint64_t st[8], uint64_t w[16]) {
    uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint64_t wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint64_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            wi = (s512_rotr(w2, 19) ^ s512_rotr(w2, 61) ^ (w2 >> 6)) + w[(i - 7) & 15]
               + (s512_rotr(w15, 1) ^ s512_rotr(w15, 8) ^ (w15 >> 7)) + w[i & 15];
            w[i & 15] = wi;
        }
        uint64_t t1 = h + (s512_rotr(e, 14) ^ s512_rotr(e, 18) ^ s512_rotr(e, 41)) + ((e & f) ^ (~e & g)) + SHA512_K[i] + wi;
        uint64_t t2 = (s512_rotr(a, 28) ^ s512_rotr(a, 34) ^ s512_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

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
