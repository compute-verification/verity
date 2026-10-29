// The native worker step's commitment kernels: a copy of hidden_gpu_src/hidden_gpu_tree.cu's kernels with extern "C" launchers instead
// of the torch wrappers + PYBIND11_MODULE, so native_collect.cpp can run the whole per-step commitment with the GIL released.  Same
// header/prefix rules, same leaf and node bytes -> same roots; the parent's hash is tree_parent, which tree_levels_kernel shares to build
// up to ten levels in one launch, a block an aligned slice.
// GPU-resident commitment over the M1 hidden-value stream / the H1 thread digests.
//   chunk_leaves : one thread per 64-word chunk -> SHA-256(16-word header(chunk index, tag, geometry) || 64 stream words)
//                  (bit-identical to hidden_stream.commit_stream's leaves)
//   tree_level   : one thread per parent -> production node rule  h(NODE_PREFIX || left || u64be(32) || right)
//                  and lift rule h(LIFT_PREFIX || child) for the odd trailing node (verity_vllm.commit.scheme)
//   tree_levels  : several levels in one launch, each block folding its own aligned slice, the same parent rule
// Plain SHA-256 (FIPS 180-4), one message per thread, register-resident blocks; correctness is checked against hashlib
// on the host (hidden_gpu.py --check) before any timing is reported.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__constant__ uint32_t K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ uint32_t bswap(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

struct Sha {
    uint32_t h[8];
    __device__ __forceinline__ void init() {
        h[0] = 0x6a09e667u; h[1] = 0xbb67ae85u; h[2] = 0x3c6ef372u; h[3] = 0xa54ff53au;
        h[4] = 0x510e527fu; h[5] = 0x9b05688cu; h[6] = 0x1f83d9abu; h[7] = 0x5be0cd19u;
    }
    // m[16] big-endian words, consumed (message schedule in place)
    __device__ __forceinline__ void compress(uint32_t (&m)[16]) {
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint32_t w;
            if (i < 16) {
                w = m[i];
            } else {
                const uint32_t w15 = m[(i + 1) & 15], w2 = m[(i + 14) & 15];
                const uint32_t s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >> 3);
                const uint32_t s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >> 10);
                w = m[i & 15] + s0 + m[(i + 9) & 15] + s1;
                m[i & 15] = w;
            }
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K256[i] + w;
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
};

// byte-addressed block builder (constant indices after unrolling -> register ops)
struct Blk {
    uint32_t w[16];
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = 0u;
    }
    __device__ __forceinline__ void byte(int i, uint32_t b) { w[i >> 2] |= (b & 0xFFu) << (24 - 8 * (i & 3)); }
};

__constant__ uint8_t NODE_PREFIX[31] = {0, 0, 0, 0x13, 'v', 'e', 'r', 'i', 't', 'y', '-', 'v', 'l', 'l', 'm', '/', 'n', 'o', 'd', 'e', '/', 'v', '1', 0, 0, 0, 0, 0, 0, 0, 0x20};
__constant__ uint8_t LIFT_PREFIX[31] = {0, 0, 0, 0x13, 'v', 'e', 'r', 'i', 't', 'y', '-', 'v', 'l', 'l', 'm', '/', 'l', 'i', 'f', 't', '/', 'v', '1', 0, 0, 0, 0, 0, 0, 0, 0x20};

// digest bytes (32) of a little-endian-u32-stored digest: byte j of the digest = (d[j/4] >> (8*(j%4))) & 0xFF
__device__ __forceinline__ uint32_t dbyte(const uint32_t *d, int j) { return (d[j >> 2] >> (8 * (j & 3))) & 0xFFu; }

// leaf = SHA-256(header[16 words] || chunk[CW words]), both little-endian u32 memory words; CW*4 + 64 bytes total
template <int CW>
__global__ void chunk_leaves_kernel(const uint32_t *__restrict__ stream, uint32_t *__restrict__ out, long n_chunks,
                                    uint32_t tag, uint32_t launch_tag, uint32_t src_mask, uint32_t HB, uint32_t M, uint32_t NB, long ci_base) {
    const long ci = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_chunks) return;
    Sha s; s.init();
    uint32_t m[16];
    // header block: TAG, (7<<16)|(1<<8)|src, launch_tag, chunk_index, chunk_words, HB, M, NB, 0...
    m[0] = bswap(tag); m[1] = bswap((7u << 16) | (1u << 8) | (src_mask & 0xFFu)); m[2] = bswap(launch_tag); m[3] = bswap((uint32_t)(ci + ci_base));   // P0: ci_base = first chunk index of a window (0 for a whole step)
    m[4] = bswap((uint32_t)CW); m[5] = bswap(HB); m[6] = bswap(M); m[7] = bswap(NB);
    #pragma unroll
    for (int i = 8; i < 16; ++i) m[i] = 0u;
    s.compress(m);
    const uint4 *src4 = reinterpret_cast<const uint4 *>(stream + ci * CW);
    #pragma unroll
    for (int blk = 0; blk < CW / 16; ++blk) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 v = src4[blk * 4 + q];
            m[4 * q + 0] = bswap(v.x); m[4 * q + 1] = bswap(v.y); m[4 * q + 2] = bswap(v.z); m[4 * q + 3] = bswap(v.w);
        }
        s.compress(m);
    }
    // padding block: 0x80, zeros, bit length
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0u;
    m[0] = 0x80000000u;
    m[15] = (uint32_t)((64 + CW * 4) * 8);
    s.compress(m);
    uint4 *o4 = reinterpret_cast<uint4 *>(out + ci * 8);
    o4[0] = make_uint4(bswap(s.h[0]), bswap(s.h[1]), bswap(s.h[2]), bswap(s.h[3]));
    o4[1] = make_uint4(bswap(s.h[4]), bswap(s.h[5]), bswap(s.h[6]), bswap(s.h[7]));
}

// parent pi of a level of n_in nodes = node(child 2pi, child 2pi+1) or lift(child 2pi) when 2pi is the odd trailing node
__device__ __forceinline__ void tree_parent(const uint32_t *in, uint32_t *out, long n_in, long pi) {
    const uint32_t *L = in + 2 * pi * 8;
    const bool lift = (2 * pi + 1 >= n_in);
    Sha s; s.init();
    Blk b; b.clear();
    if (!lift) {
        const uint32_t *R = L + 8;
        // block 0: prefix[0..30] | left[31..62] | u64be(32) byte 0 at 63
        #pragma unroll
        for (int i = 0; i < 31; ++i) b.byte(i, NODE_PREFIX[i]);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(31 + j, dbyte(L, j));
        // byte 63 = 0 (first byte of u64be(32))
        s.compress(b.w);
        b.clear();
        // block 1: u64be(32) bytes 1..7 -> [0..6] (= 0,0,0,0,0,0,0x20), right at [7..38], 0x80 at 39, len 103*8 = 824
        b.byte(6, 0x20u);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(7 + j, dbyte(R, j));
        b.byte(39, 0x80u);
        b.w[15] = 824u;
        s.compress(b.w);
    } else {
        // lift: prefix[0..30] | child[31..62] | 0x80 at 63 ; block 1: len 63*8 = 504
        #pragma unroll
        for (int i = 0; i < 31; ++i) b.byte(i, LIFT_PREFIX[i]);
        #pragma unroll
        for (int j = 0; j < 32; ++j) b.byte(31 + j, dbyte(L, j));
        b.byte(63, 0x80u);
        s.compress(b.w);
        b.clear();
        b.w[15] = 504u;
        s.compress(b.w);
    }
    uint4 *o4 = reinterpret_cast<uint4 *>(out + pi * 8);
    o4[0] = make_uint4(bswap(s.h[0]), bswap(s.h[1]), bswap(s.h[2]), bswap(s.h[3]));
    o4[1] = make_uint4(bswap(s.h[4]), bswap(s.h[5]), bswap(s.h[6]), bswap(s.h[7]));
}

// one level, a thread a parent
__global__ void tree_level_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out, long n_in) {
    const long pi = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pi >= (n_in + 1) / 2) return;
    tree_parent(in, out, n_in, pi);
}

// n_levels successive levels above `in` (n_in nodes), each level's nodes right after the previous level's in `out`.  Block b folds its
// own slice, nodes [b 2^n_levels, (b+1) 2^n_levels) (the last may be shorter): pairs never cross an aligned slice, and a level's odd
// trailing node is in the last slice, so the lift falls where the whole level's does.  A level reads the one this block wrote before the
// barrier, so no pointer here is __restrict__ (no non-coherent loads).
__global__ void tree_levels_kernel(const uint32_t *in, uint32_t *out, long n_in, int n_levels) {
    const long first = (long)blockIdx.x << n_levels;
    long len = n_in - first < (1L << n_levels) ? n_in - first : (1L << n_levels);
    in += first * 8;
    for (int k = 0; k < n_levels; ++k) {
        const long n_out = (n_in + 1) / 2, len_out = (len + 1) / 2;
        uint32_t *dst = out + (first >> (k + 1)) * 8;
        for (long pi = threadIdx.x; pi < len_out; pi += blockDim.x) tree_parent(in, dst, len, pi);
        __syncthreads();
        in = dst;
        out += n_out * 8;
        n_in = n_out;
        len = len_out;
    }
}

}  // namespace

extern "C" int verity_chunk_leaves_launch_base(const uint32_t* stream, uint32_t* out, long n_chunks, int chunk_words, uint32_t launch_tag,
                                               uint32_t src_mask, uint32_t HB, uint32_t M, uint32_t NB, long ci_base, cudaStream_t st) {
  const int threads = 128;
  const long blocks = (n_chunks + threads - 1) / threads;
  if (n_chunks <= 0) return 0;
  if (chunk_words == 64) chunk_leaves_kernel<64><<<blocks, threads, 0, st>>>(stream, out, n_chunks, 0x68326166u, launch_tag, src_mask, HB, M, NB, ci_base);
  else if (chunk_words == 128) chunk_leaves_kernel<128><<<blocks, threads, 0, st>>>(stream, out, n_chunks, 0x68326166u, launch_tag, src_mask, HB, M, NB, ci_base);
  else if (chunk_words == 32) chunk_leaves_kernel<32><<<blocks, threads, 0, st>>>(stream, out, n_chunks, 0x68326166u, launch_tag, src_mask, HB, M, NB, ci_base);
  else return -1;
  return static_cast<int>(cudaGetLastError());
}

extern "C" int verity_chunk_leaves_launch(const uint32_t* stream, uint32_t* out, long n_chunks, int chunk_words, uint32_t launch_tag,
                                          uint32_t src_mask, uint32_t HB, uint32_t M, uint32_t NB, cudaStream_t st) {
  return verity_chunk_leaves_launch_base(stream, out, n_chunks, chunk_words, launch_tag, src_mask, HB, M, NB, 0, st);
}

extern "C" int verity_tree_level_launch(const uint32_t* in, uint32_t* out, long n_in, cudaStream_t st) {
  const long n_out = (n_in + 1) / 2;
  const int threads = 128;
  const long blocks = (n_out + threads - 1) / threads;
  tree_level_kernel<<<blocks, threads, 0, st>>>(in, out, n_in);
  return static_cast<int>(cudaGetLastError());
}

// n_levels (at most 10: a block's slice is 2^n_levels nodes) levels above n_in nodes in one launch (tree_levels_kernel); out holds all of
// them, level after level
extern "C" int verity_tree_levels_launch(const uint32_t* in, uint32_t* out, long n_in, int n_levels, cudaStream_t st) {
  if (n_levels <= 0 || n_in <= 0) return 0;
  if (n_levels > 10) return static_cast<int>(cudaErrorInvalidValue);
  const long blocks = (n_in + (1L << n_levels) - 1) >> n_levels;
  tree_levels_kernel<<<blocks, 256, 0, st>>>(in, out, n_in, n_levels);
  return static_cast<int>(cudaGetLastError());
}
