// The bandwidth audit's GPU tagger (verity/core/protocols/bandwidth/PROTOCOL.md §3-4): every block's tag over the store E
// in device memory, the tags' leaves and the RFC 6962 root, for the three tag constructions.
//
//   ro-blake3/v1  two lanes per block, one per 1,024-byte chunk of the message u64be j ‖ E_j (16 compressions each under
//                 the window's key K), the warp staging each 64-byte message block of its 32 chunks through shared memory
//                 while the next one is loaded into registers, with a 256-byte L2 fetch per miss; then the third chunk (one
//                 compression) beside the parent, then the root.  A second kernel hashes the leaves of 16 tags
//                 (`b3_leaves`, five SHA-512 compressions each).
//   gf31-poly/v1  one warp per 32 blocks: lane l derives g for block base + l (one SHA-512 compression from the tag
//                 prefix's midstate), then for each block the warp reads the row as 257 uint2 (u_q = w_2q·g + w_2q+1),
//                 lane l runs Horner in H = g^64 over q = 32k + l (lane 0 also takes q = 256), and the lanes combine with
//                 h^(32 − l), h = g², from a warp scan of powers.  Mersenne folding, no division.  A second kernel
//                 hashes the leaves of 16 tags (`gf31_leaves`).
//   ro-sha512/v1  one thread per block, the warp staging each 128-byte chunk of its 32 rows through shared memory
//                 (coalesced), 17 compressions after the prefix's midstate, and the block's leaf fused (one leaf per tag).
//   tree          bottom-up, an odd last node promoted, which is RFC 6962's shape: one launch per level while a level
//                 has more than 2^15 nodes, then nine levels per launch, each block reducing 512 nodes to one
//                 (`node_span`); every level stays in the workspace for openings.
//
// Built two ways from this file: `bwtag` (BW_MAIN: validate against the Python reference, then time) and `libbwtag.so`
// (the C ABI at the end, for a harness that holds E in its own device buffer).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                                  \
  do {                                                                                                         \
    cudaError_t e_ = (x);                                                                                      \
    if (e_ != cudaSuccess) {                                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));         \
      return (int)e_;                                                                                          \
    }                                                                                                          \
  } while (0)

namespace bw {

constexpr uint32_t P31 = 0x7fffffffu;
constexpr uint64_t BLOCK_BYTES = 2056;
constexpr int ROW_U64 = 257;  // a block is 257 eight-byte words
constexpr int TAG_GF31 = 0, TAG_RO = 1, TAG_B3 = 2, TAGS = 3;
constexpr unsigned FULL = 0xffffffffu;

static const char* const TAG_NAMES[TAGS] = {"gf31-poly/v1", "ro-sha512/v1", "ro-blake3/v1"};
static const char* const TAG_SHORT[TAGS] = {"gf31", "ro", "b3"};
static const char TAG_DOMAIN[] = "verity/bandwidth/tag/v1";        // + its NUL: 24 bytes
static const char LEAF_DOMAIN[] = "verity/bandwidth/tag-leaf/v1";  // + its NUL: 29 bytes

#define SHA512_K                                                                                                 \
  {0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, \
   0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, \
   0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, \
   0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, \
   0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL, \
   0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, \
   0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, \
   0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL, \
   0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, \
   0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, \
   0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, \
   0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, \
   0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL, \
   0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, \
   0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, \
   0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL}
#define SHA512_IV                                                                                               \
  {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, \
   0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL}

__constant__ uint64_t K_DEV[80] = SHA512_K;
__constant__ uint64_t IV_DEV[8] = SHA512_IV;
static const uint64_t K_HOST[80] = SHA512_K;
static const uint64_t IV_HOST[8] = SHA512_IV;

struct Mid {
  uint64_t h[8];
};

__host__ __device__ __forceinline__ uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

// One SHA-512 compression of the big-endian words `w` (clobbered) into `s`: five passes of 16 rounds, the schedule
// updated in place at the top of each pass after the first, so a compiler cannot hoist all 80 words at once.
__host__ __device__ __forceinline__ void compress(uint64_t s[8], uint64_t w[16]) {
#ifdef __CUDA_ARCH__
  const uint64_t* K = K_DEV;
#else
  const uint64_t* K = K_HOST;
#endif
  uint64_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const uint64_t w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (rotr(w15, 1) ^ rotr(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] + (rotr(w2, 19) ^ rotr(w2, 61) ^ (w2 >> 6));
      }
      const uint64_t t1 = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + ((e & f) ^ (~e & g)) + K[r + t] + w[t];
      const uint64_t t2 = (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

__device__ __forceinline__ uint64_t bswap64(uint64_t v) {
  uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
  return ((uint64_t)__byte_perm(lo, 0, 0x0123) << 32) | __byte_perm(hi, 0, 0x0123);
}

// x mod p for any x < 2^64: two folds (2^31 ≡ 1) and one conditional subtract.
__device__ __forceinline__ uint32_t red64(uint64_t x) {
  uint64_t s = (x & P31) + (x >> 31);
  uint32_t t = (uint32_t)((s & P31) + (s >> 31));
  return t >= P31 ? t - P31 : t;
}

__device__ __forceinline__ uint32_t mulmod(uint32_t a, uint32_t b) { return red64((uint64_t)a * b); }

// BLAKE3 (the specification's reference implementation, as `verity.ml.crypto.blake3`).
#define B3_IV0 0x6A09E667u
#define B3_IV1 0xBB67AE85u
#define B3_IV2 0x3C6EF372u
#define B3_IV3 0xA54FF53Au
constexpr uint32_t B3_CHUNK_START = 1, B3_CHUNK_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16;

struct Key3 {
  uint32_t k[8];
};

__host__ __device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << (32 - n));
#endif
}

#define B3_G(a, b, c, d, x, y) \
  a += b + (x);                \
  d = rotr32(d ^ a, 16);       \
  c += d;                      \
  b = rotr32(b ^ c, 12);       \
  a += b + (y);                \
  d = rotr32(d ^ a, 8);        \
  c += d;                      \
  b = rotr32(b ^ c, 7)

// One compression of the message words `m` (permuted in place) from chaining value `cv`; `out` gets the first 8 words of
// the output, the next chaining value (or, under ROOT, the hash).  Counters here are below 2^32.
__host__ __device__ __forceinline__ void b3_compress(const uint32_t cv[8], uint32_t m[16], uint32_t counter, uint32_t len,
                                                     uint32_t flags, uint32_t out[8]) {
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = B3_IV0, v9 = B3_IV1, v10 = B3_IV2, v11 = B3_IV3, v12 = counter, v13 = 0, v14 = len, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    B3_G(v0, v4, v8, v12, m[0], m[1]);
    B3_G(v1, v5, v9, v13, m[2], m[3]);
    B3_G(v2, v6, v10, v14, m[4], m[5]);
    B3_G(v3, v7, v11, v15, m[6], m[7]);
    B3_G(v0, v5, v10, v15, m[8], m[9]);
    B3_G(v1, v6, v11, v12, m[10], m[11]);
    B3_G(v2, v7, v8, v13, m[12], m[13]);
    B3_G(v3, v4, v9, v14, m[14], m[15]);
    if (r < 6) {
      const uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
  out[0] = v0 ^ v8;
  out[1] = v1 ^ v9;
  out[2] = v2 ^ v10;
  out[3] = v3 ^ v11;
  out[4] = v4 ^ v12;
  out[5] = v5 ^ v13;
  out[6] = v6 ^ v14;
  out[7] = v7 ^ v15;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host: SHA-512 of short messages, and the prefixes' midstates.

static void sha512_host(const uint8_t* msg, size_t len, uint64_t out[8]) {
  uint64_t s[8];
  memcpy(s, IV_HOST, sizeof s);
  std::vector<uint8_t> m(msg, msg + len);
  m.push_back(0x80);
  while (m.size() % 128 != 112) m.push_back(0);
  for (int i = 0; i < 8; ++i) m.push_back(0);
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 7; i >= 0; --i) m.push_back((uint8_t)(bits >> (8 * i)));
  for (size_t off = 0; off < m.size(); off += 128) {
    uint64_t w[16];
    for (int i = 0; i < 16; ++i) {
      uint64_t v = 0;
      for (int k = 0; k < 8; ++k) v = (v << 8) | m[off + 8 * i + k];
      w[i] = v;
    }
    compress(s, w);
  }
  memcpy(out, s, sizeof s);
}

static void append_frame(std::vector<uint8_t>& out, char tag, const uint8_t* body, size_t n) {
  out.push_back((uint8_t)tag);
  for (int i = 7; i >= 0; --i) out.push_back((uint8_t)((uint64_t)n >> (8 * i)));
  out.insert(out.end(), body, body + n);
}

// Z(domain ‖ NUL ‖ frame(name) ‖ frame(seed)), one 128-byte block.
static std::vector<uint8_t> prefix_block(const char* domain, const char* name, const uint8_t seed[32]) {
  std::vector<uint8_t> p(domain, domain + strlen(domain) + 1);
  append_frame(p, 's', (const uint8_t*)name, strlen(name));
  append_frame(p, 'b', seed, 32);
  p.resize(128, 0);
  return p;
}

// ro-blake3/v1's key K: the first 32 bytes of SHA512(prefix), as BLAKE3's little-endian key words.
static Key3 b3_key(const uint8_t seed[32]) {
  std::vector<uint8_t> p = prefix_block(TAG_DOMAIN, TAG_NAMES[TAG_B3], seed);
  uint64_t d[8];
  sha512_host(p.data(), p.size(), d);
  Key3 k;
  for (int i = 0; i < 8; ++i) {
    const uint64_t w = d[i / 2];
    const uint32_t be = (uint32_t)(i & 1 ? w : w >> 32);
    k.k[i] = (be >> 24) | ((be >> 8) & 0xff00u) | ((be << 8) & 0xff0000u) | (be << 24);
  }
  return k;
}

// The midstate after the prefix block.
static Mid prefix_midstate(const char* domain, const char* name, const uint8_t seed[32]) {
  std::vector<uint8_t> p = prefix_block(domain, name, seed);
  Mid m;
  memcpy(m.h, IV_HOST, sizeof m.h);
  uint64_t w[16];
  for (int i = 0; i < 16; ++i) {
    uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v = (v << 8) | p[8 * i + k];
    w[i] = v;
  }
  compress(m.h, w);
  return m;
}

static void digest_bytes(const uint64_t s[8], uint8_t out[64]) {
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 8; ++k) out[8 * i + k] = (uint8_t)(s[i] >> (56 - 8 * k));
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernels.

// The test store of `bandwidth_vectors.py`: block j's bytes are SHA512("verity/bandwidth/test-block/v1" ‖ NUL ‖ u64be j ‖
// u32be c) for c = 0..32, truncated to 2,056.  E is written as little-endian u64 words, so the bytes land in order.
__global__ void fill_store(uint64_t* E8, uint64_t blocks) {
  uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (id >= blocks * 33) return;
  uint64_t j = id / 33;
  uint32_t c = (uint32_t)(id % 33);
  uint8_t m[128];
  for (int i = 0; i < 128; ++i) m[i] = 0;
  const char dom[] = "verity/bandwidth/test-block/v1";
  for (int i = 0; i < 30; ++i) m[i] = (uint8_t)dom[i];
  for (int i = 0; i < 8; ++i) m[31 + i] = (uint8_t)(j >> (56 - 8 * i));
  for (int i = 0; i < 4; ++i) m[39 + i] = (uint8_t)(c >> (24 - 8 * i));
  m[43] = 0x80;
  m[127] = (uint8_t)(43 * 8);
  m[126] = (uint8_t)((43 * 8) >> 8);
  uint64_t w[16];
  for (int i = 0; i < 16; ++i) {
    uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v = (v << 8) | m[8 * i + k];
    w[i] = v;
  }
  uint64_t s[8];
  for (int i = 0; i < 8; ++i) s[i] = IV_DEV[i];
  compress(s, w);
  int n = c == 32 ? 1 : 8;
  for (int i = 0; i < n; ++i) E8[j * ROW_U64 + 8 * c + i] = bswap64(s[i]);
}

__device__ uint32_t gf31_point(const Mid& mid, uint64_t j) {
  for (uint32_t c = 0;; ++c) {
    uint64_t s[8], w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = mid.h[i];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = 0;
    w[0] = j;
    w[1] = ((uint64_t)c << 32) | 0x80000000ULL;
    w[15] = (128 + 12) * 8;
    compress(s, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t x = (uint32_t)(s[i] >> 33);
      if (x < P31 - 2) return 2 + x;
      x = (uint32_t)s[i] >> 1;
      if (x < P31 - 2) return 2 + x;
    }
  }
}

// gf31-poly/v1's tags of blocks [0, count): one warp per 32 consecutive blocks, each lane deriving its block's point
// first, then each block's row read once as uint2.
__global__ void __launch_bounds__(256) gf31_kernel(const uint2* __restrict__ E, uint64_t count, Mid tagmid,
                                                   uint32_t* __restrict__ tags) {
  const int lane = threadIdx.x & 31;
  const uint64_t base = ((blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) >> 5) * 32;
  if (base >= count) return;
  const uint64_t jl = base + lane;
  const uint32_t g_lane = jl < count ? gf31_point(tagmid, jl) : 2;
  const int nb = (int)(count - base < 32 ? count - base : 32);
  uint32_t my_y = 0;
  for (int b = 0; b < nb; ++b) {
    const uint2* row = E + (base + b) * ROW_U64;
    uint2 x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = __ldcs(row + 32 * k + lane);
    uint2 x8 = make_uint2(0, 0);
    if (lane == 0) x8 = __ldcs(row + 256);
    const uint32_t g = __shfl_sync(FULL, g_lane, b);
    uint32_t v = mulmod(g, g);
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      uint32_t t = __shfl_up_sync(FULL, v, d);
      if (lane >= d) v = mulmod(v, t);
    }
    const uint32_t H = __shfl_sync(FULL, v, 31);
    uint32_t c = __shfl_sync(FULL, v, 31 - lane);
    if (lane == 0) c = 1;
    uint32_t a = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) a = red64((uint64_t)a * H + red64((uint64_t)x[k].x * g + x[k].y));
    if (lane == 0) a = red64((uint64_t)a * H + red64((uint64_t)x8.x * g + x8.y));
    uint32_t t = mulmod(a, c);
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      t += __shfl_xor_sync(FULL, t, d);
      t = t >= P31 ? t - P31 : t;
    }
    if (lane == b) my_y = t;
  }
  if (jl < count) tags[jl] = my_y;
}

// gf31-poly/v1's leaves over the tags of blocks [0, count) of B: 16 tags each, the last of B's leaves short.
__global__ void gf31_leaves(const uint32_t* __restrict__ tags, uint64_t B, uint64_t count, Mid leafmid,
                            uint64_t* __restrict__ leaves) {
  const uint64_t leaf = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, first = leaf * 16;
  if (first >= count) return;
  const int n = (int)(B - first < 16 ? B - first : 16);
  uint64_t w[16], s[8];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = leaf;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (i < n) w[1 + i / 2] |= (uint64_t)tags[first + i] << ((i & 1) ? 0 : 32);
#pragma unroll
  for (int i = 0; i <= 16; ++i)
    if (i == n) w[1 + i / 2] |= (i & 1) ? (0x80ULL << 24) : (0x80ULL << 56);
  w[15] = (uint64_t)(128 + 8 + 4 * n) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = leafmid.h[i];
  compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

constexpr int RO_WARPS = 4;
constexpr int RS = 17;  // shared-memory row stride, in u64

// ro-sha512/v1 over blocks [0, count), each block's leaf fused.
__global__ void __launch_bounds__(RO_WARPS * 32, 4) ro_kernel(const uint64_t* __restrict__ E8, uint64_t count, Mid tagmid,
                                                           Mid leafmid, uint64_t* __restrict__ leaves) {
  __shared__ uint64_t sm[RO_WARPS][32 * RS];
  const int lane = threadIdx.x & 31;
  const uint64_t base = ((blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) >> 5) * 32;
  if (base >= count) return;
  uint64_t* my = sm[threadIdx.x >> 5];
  const uint64_t j = base + lane;
  const bool live = j < count;
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = tagmid.h[i];
#pragma unroll 1
  for (int t = 0; t < 16; ++t) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int r = 2 * i + (lane >> 4), wd = lane & 15, e = 16 * t - 1 + wd;
      uint64_t v = 0;
      if (base + r < count && e >= 0) v = __ldcs(E8 + (base + r) * ROW_U64 + e);
      my[r * RS + wd] = v;
    }
    __syncwarp();
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = bswap64(my[lane * RS + i]);
    if (t == 0) w[0] = j;
    compress(s, w);
    __syncwarp();
  }
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  if (live) {
    w[0] = bswap64(__ldcs(E8 + j * ROW_U64 + 255));
    w[1] = bswap64(__ldcs(E8 + j * ROW_U64 + 256));
  }
  w[2] = 0x80ULL << 56;
  w[15] = (128 + 8 + BLOCK_BYTES) * 8;
  compress(s, w);
  uint64_t ls[8];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = j;
#pragma unroll
  for (int i = 0; i < 8; ++i) w[1 + i] = s[i];
  w[9] = 0x80ULL << 56;
  w[15] = (128 + 72) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) ls[i] = leafmid.h[i];
  compress(ls, w);
  if (live)
#pragma unroll
    for (int i = 0; i < 8; ++i) leaves[j * 8 + i] = ls[i];
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

// A streaming load whose L2 miss fetches the whole 256-byte block: a warp reads 32 rows 64 bytes at a time, and without
// the hint each step reopens a DRAM page per row.
__device__ __forceinline__ uint64_t ld_l2_256(const uint64_t* p) {
  uint64_t v;
  asm("ld.global.nc.L1::no_allocate.L2::256B.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}

constexpr int B3_WARPS = 4;
constexpr int B3S = 9;  // shared-memory stream stride, in u64

// ro-blake3/v1 over blocks [0, count), 16 blocks a warp: lanes 2r and 2r + 1 hash block base + r's first and second
// chunk, whose message block g (0..15) is E's u64 words 128c + 8g − 1 .. 128c + 8g + 6 (word −1 is u64be j), staged
// through shared memory 4 streams × 64 bytes per load instruction, one message block ahead.  Then the odd lane hashes the
// third chunk (E's words 255 and 256) while the even lane forms the parent, and the even lane forms the root.  Each tag
// as 8 little-endian words.
__global__ void __launch_bounds__(B3_WARPS * 32, 5) b3_kernel(const uint64_t* __restrict__ E8, uint64_t count, Key3 key,
                                                           uint32_t* __restrict__ tags) {
  __shared__ uint64_t sm[B3_WARPS][32 * B3S];
  const int lane = threadIdx.x & 31, sub = lane >> 3, wd = lane & 7;
  const uint32_t c = lane & 1;
  const uint64_t base = ((blockIdx.x * (uint64_t)blockDim.x + threadIdx.x) >> 5) * 16;
  if (base >= count) return;
  uint64_t* my = sm[threadIdx.x >> 5];
  const uint64_t j = base + (lane >> 1);
  uint64_t pre[8];
  auto load = [&](int g) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int s = 4 * i + sub, e = 128 * (s & 1) + 8 * g - 1 + wd;
      const uint64_t r = base + (s >> 1);
      pre[i] = r < count && e >= 0 ? ld_l2_256(E8 + r * ROW_U64 + e) : 0;
    }
  };
  load(0);
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll 1
  for (int g = 0; g < 16; ++g) {
#pragma unroll
    for (int i = 0; i < 8; ++i) my[(4 * i + sub) * B3S + wd] = pre[i];
    __syncwarp();
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint64_t x = my[lane * B3S + i];
      m[2 * i] = (uint32_t)x;
      m[2 * i + 1] = (uint32_t)(x >> 32);
    }
    __syncwarp();
    if (g < 15) load(g + 1);
    if (g == 0 && !c) {
      m[0] = bswap32((uint32_t)(j >> 32));
      m[1] = bswap32((uint32_t)j);
    }
    b3_compress(cv, m, c, 64, B3_KEYED | (g == 0 ? B3_CHUNK_START : 0) | (g == 15 ? B3_CHUNK_END : 0), cv);
  }
  uint32_t m[16], out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t other = __shfl_xor_sync(FULL, cv[i], 1);
    m[i] = c ? 0 : cv[i];
    m[8 + i] = c ? 0 : other;
  }
  if (c && j < count) {
    const uint64_t e255 = ld_l2_256(E8 + j * ROW_U64 + 255), e256 = ld_l2_256(E8 + j * ROW_U64 + 256);
    m[0] = (uint32_t)e255;
    m[1] = (uint32_t)(e255 >> 32);
    m[2] = (uint32_t)e256;
    m[3] = (uint32_t)(e256 >> 32);
  }
  b3_compress(key.k, m, c ? 2 : 0, c ? 16 : 64, B3_KEYED | (c ? B3_CHUNK_START | B3_CHUNK_END : B3_PARENT), out);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m[i] = out[i];
    m[8 + i] = __shfl_xor_sync(FULL, out[i], 1);
  }
  if (!c) {
    uint32_t y[8];
    b3_compress(key.k, m, 0, 64, B3_KEYED | B3_PARENT | B3_ROOT, y);
    if (j < count) {
      uint4* o = (uint4*)(tags + j * 8);
      o[0] = make_uint4(y[0], y[1], y[2], y[3]);
      o[1] = make_uint4(y[4], y[5], y[6], y[7]);
    }
  }
}

// ro-blake3/v1's leaves over the tags of blocks [0, count) of B: 16 tags (u64be leaf ‖ 512 bytes, five compressions after
// the leaf prefix's midstate) each, the last of B's leaves short.
__global__ void b3_leaves(const uint32_t* __restrict__ tags, uint64_t B, uint64_t count, Mid leafmid,
                          uint64_t* __restrict__ leaves) {
  const uint64_t leaf = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, first = leaf * 16;
  if (first >= count) return;
  const int n = (int)(B - first < 16 ? B - first : 16);
  const int data = 1 + 4 * n, nb = (8 * data + 17 + 127) / 128;
  const uint32_t* T = tags + first * 8;
  uint64_t s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = leafmid.h[i];
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      uint64_t v = 0;
      if (q == 0)
        v = leaf;
      else if (q < data)
        v = ((uint64_t)bswap32(T[2 * (q - 1)]) << 32) | bswap32(T[2 * (q - 1) + 1]);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nb - 1) v = (uint64_t)(128 + 8 + 32 * n) * 8;
      w[i] = v;
    }
    compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

// SHA512(0x01 ‖ L ‖ R), 129 bytes from the IV: two compressions.  The message is L ‖ R shifted right by one byte.
__device__ __forceinline__ void node_hash(const uint64_t* L, const uint64_t* R, uint64_t* out) {
  uint64_t w[16], s[8], prev = 1;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const uint64_t cur = k < 8 ? L[k] : R[k - 8];
    w[k] = (prev << 56) | (cur >> 8);
    prev = cur;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = IV_DEV[i];
  compress(s, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = (prev << 56) | (0x80ULL << 48);
  w[15] = 129 * 8;
  compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

__device__ __forceinline__ void level_step(const uint64_t* in, uint64_t n, uint64_t* out, uint64_t i) {
  if (2 * i + 1 < n)
    node_hash(in + 16 * i, in + 16 * i + 8, out + 8 * i);
  else
#pragma unroll
    for (int k = 0; k < 8; ++k) out[8 * i + k] = in[16 * i + k];
}

// One level, a thread per output node: for the wide levels, where throughput rather than latency counts.
constexpr uint64_t WIDE = 1 << 15;

__global__ void __launch_bounds__(128, 4) node_level(const uint64_t* __restrict__ in, uint64_t n, uint64_t* __restrict__ out) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i < (n + 1) / 2) level_step(in, n, out, i);
}

// Up to SPAN levels in one launch: block b of SPAN_THREADS threads takes nodes [b·2^SPAN, (b + 1)·2^SPAN) of the first
// level and every node above them, one level per barrier.  Ranges stay aligned, so a block's inputs at each level are
// its own outputs of the level below, and the odd last node of a level falls in the last block.
constexpr int SPAN = 9, SPAN_THREADS = 1 << (SPAN - 1);

struct Span {
  uint64_t off[SPAN + 1], n[SPAN + 1];  // in nodes; off[t], n[t] for t = 0..levels
  int levels;
};

__global__ void __launch_bounds__(SPAN_THREADS) node_span(uint64_t* nodes, Span s) {
  for (int t = 0; t < s.levels; ++t) {
    const uint64_t i = blockIdx.x * (uint64_t)(SPAN_THREADS >> t) + threadIdx.x;
    if (threadIdx.x < (SPAN_THREADS >> t) && i < s.n[t + 1]) level_step(nodes + s.off[t] * 8, s.n[t], nodes + s.off[t + 1] * 8, i);
    __syncthreads();
  }
}

__global__ void read_kernel(const uint4* __restrict__ p, uint64_t n16, uint32_t* out) {
  uint32_t acc = 0;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n16; i += (uint64_t)gridDim.x * blockDim.x) {
    uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) acc ^= __shfl_xor_sync(FULL, acc, d);
  if ((threadIdx.x & 31) == 0) atomicXor(out, acc);
}

__global__ void sha512_selftest(const uint64_t* w_in, uint64_t* out) {
  uint64_t w[16], s[8];
  for (int i = 0; i < 16; ++i) w[i] = w_in[i];
  for (int i = 0; i < 8; ++i) s[i] = IV_DEV[i];
  compress(s, w);
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

// BLAKE3 keyed over the empty input: one compression, CHUNK_START | CHUNK_END | ROOT.
__global__ void b3_selftest(Key3 key, uint32_t* out) {
  uint32_t m[16] = {}, y[8];
  b3_compress(key.k, m, 0, 0, B3_KEYED | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, y);
  for (int i = 0; i < 8; ++i) out[i] = y[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// One store's tag pass: the workspace and the launches.

struct Levels {
  std::vector<uint64_t> off, n;  // in nodes; level 0 is the leaves
  uint64_t total = 0;
};

static Levels levels_for(uint64_t leaves) {
  Levels L;
  uint64_t off = 0, n = leaves;
  for (;;) {
    L.off.push_back(off);
    L.n.push_back(n);
    off += n;
    if (n == 1) break;
    n = (n + 1) / 2;
  }
  L.total = off;
  return L;
}

static uint64_t leaf_count(int tag, uint64_t blocks) { return tag == TAG_RO ? blocks : (blocks + 15) / 16; }

// The bytes of one tag the workspace keeps (gf31 as a native u32, b3 as its 32 bytes); ro keeps none, only its leaves.
static size_t tag_bytes(int tag) { return tag == TAG_GF31 ? 4 : tag == TAG_B3 ? 32 : 0; }

struct Pass {
  int tag = TAG_GF31;
  uint64_t blocks = 0;
  Levels lv;
  uint32_t* tags = nullptr;   // gf31 and b3
  uint64_t* nodes = nullptr;  // every level, contiguous
};

static size_t tag_area(int tag, uint64_t blocks) { return ((blocks * tag_bytes(tag) + 255) / 256) * 256; }

static size_t workspace_bytes(int tag, uint64_t blocks) {
  Levels lv = levels_for(leaf_count(tag, blocks));
  return tag_area(tag, blocks) + lv.total * 64;
}

static Pass make_pass(int tag, uint64_t blocks, void* workspace) {
  Pass p;
  p.tag = tag;
  p.blocks = blocks;
  p.lv = levels_for(leaf_count(tag, blocks));
  char* ws = (char*)workspace;
  if (tag_bytes(tag)) {
    p.tags = (uint32_t*)ws;
    ws += tag_area(tag, blocks);
  }
  p.nodes = (uint64_t*)ws;
  return p;
}

static void launch_tags(const Pass& p, const void* E, const uint8_t seed[32], uint64_t count, cudaStream_t st) {
  Mid lm = prefix_midstate(LEAF_DOMAIN, TAG_NAMES[p.tag], seed);
  uint64_t warps = (count + 31) / 32;
  const unsigned leaf_grid = (unsigned)(((count + 15) / 16 + 127) / 128);
  if (p.tag == TAG_B3) {
    unsigned grid = (unsigned)(((count + 15) / 16 + B3_WARPS - 1) / B3_WARPS);
    b3_kernel<<<grid, B3_WARPS * 32, 0, st>>>((const uint64_t*)E, count, b3_key(seed), p.tags);
    b3_leaves<<<leaf_grid, 128, 0, st>>>(p.tags, p.blocks, count, lm, p.nodes);
    return;
  }
  Mid tm = prefix_midstate(TAG_DOMAIN, TAG_NAMES[p.tag], seed);
  if (p.tag == TAG_GF31) {
    gf31_kernel<<<(unsigned)((warps * 32 + 255) / 256), 256, 0, st>>>((const uint2*)E, count, tm, p.tags);
    gf31_leaves<<<leaf_grid, 128, 0, st>>>(p.tags, p.blocks, count, lm, p.nodes);
  } else {
    unsigned grid = (unsigned)((warps + RO_WARPS - 1) / RO_WARPS);
    ro_kernel<<<grid, RO_WARPS * 32, 0, st>>>((const uint64_t*)E, count, tm, lm, p.nodes);
  }
}

static void launch_tree(const Pass& p, cudaStream_t st) {
  const size_t top = p.lv.n.size() - 1;
  size_t k = 0;
  for (; k < top && p.lv.n[k] > WIDE; ++k) {
    const uint64_t nout = p.lv.n[k + 1];
    node_level<<<(unsigned)((nout + 127) / 128), 128, 0, st>>>(p.nodes + p.lv.off[k] * 8, p.lv.n[k],
                                                               p.nodes + p.lv.off[k + 1] * 8);
  }
  while (k < top) {
    Span s;
    s.levels = (int)std::min<size_t>(SPAN, top - k);
    for (int t = 0; t <= s.levels; ++t) {
      s.off[t] = p.lv.off[k + t];
      s.n[t] = p.lv.n[k + t];
    }
    node_span<<<(unsigned)((p.lv.n[k] + (2 * SPAN_THREADS) - 1) / (2 * SPAN_THREADS)), SPAN_THREADS, 0, st>>>(p.nodes, s);
    k += s.levels;
  }
}

static int read_root(const Pass& p, uint8_t root[64], cudaStream_t st) {
  uint64_t s[8];
  CK(cudaMemcpyAsync(s, p.nodes + p.lv.off.back() * 8, 64, cudaMemcpyDeviceToHost, st));
  CK(cudaStreamSynchronize(st));
  digest_bytes(s, root);
  return 0;
}

static cudaEvent_t pass_events[3];

static int tag_pass(const Pass& p, const void* E, const uint8_t seed[32], uint8_t root[64], cudaStream_t st,
                    uint64_t tagged) {
  if (!pass_events[0])
    for (cudaEvent_t& e : pass_events) CK(cudaEventCreate(&e));
  CK(cudaEventRecord(pass_events[0], st));
  launch_tags(p, E, seed, tagged, st);
  CK(cudaEventRecord(pass_events[1], st));
  launch_tree(p, st);
  CK(cudaEventRecord(pass_events[2], st));
  CK(cudaGetLastError());
  return read_root(p, root, st);
}

}  // namespace bw

// ---------------------------------------------------------------------------------------------------------------------
// The C ABI: `tag` is 0 for gf31-poly/v1, 1 for ro-sha512/v1 and 2 for ro-blake3/v1; `store` is E in device memory, 2,056
// bytes per block, 8-byte aligned; `workspace` is `bw_workspace_bytes` of device memory, which keeps the tags (gf31, b3)
// and every tree level after the pass, for openings.

extern "C" {

size_t bw_workspace_bytes(int tag, uint64_t blocks) { return bw::workspace_bytes(tag, blocks); }

int bw_tag_pass(int tag, const void* store, uint64_t blocks, const uint8_t* seed, void* workspace, uint8_t* root,
                void* stream) {
  bw::Pass p = bw::make_pass(tag, blocks, workspace);
  return bw::tag_pass(p, store, seed, root, (cudaStream_t)stream, blocks);
}

// A dishonest pass for negative controls: only blocks [0, tagged) are tagged, and the tree is built over whatever the
// workspace holds for the rest.
int bw_tag_pass_prefix(int tag, const void* store, uint64_t blocks, uint64_t tagged, const uint8_t* seed, void* workspace,
                       uint8_t* root, void* stream) {
  if (tagged > blocks) return -1;
  bw::Pass p = bw::make_pass(tag, blocks, workspace);
  return bw::tag_pass(p, store, seed, root, (cudaStream_t)stream, tagged);
}

// The `count` tags from block `first` on, after a pass, as the workspace keeps them (gf31: native u32 values; b3: 32
// bytes each); -1 for ro, whose tags are not kept.
int bw_read_tags(int tag, uint64_t blocks, const void* workspace, uint64_t first, uint64_t count, void* out) {
  if (!bw::tag_bytes(tag) || first + count > blocks) return -1;
  bw::Pass p = bw::make_pass(tag, blocks, (void*)workspace);
  CK(cudaMemcpy(out, (const char*)p.tags + first * bw::tag_bytes(tag), count * bw::tag_bytes(tag), cudaMemcpyDeviceToHost));
  return 0;
}

// Node `index` of tree level `level` (0 = leaves) as 64 digest bytes, after a pass; -1 past the tree.
int bw_read_node(int tag, uint64_t blocks, const void* workspace, uint32_t level, uint64_t index, uint8_t* out) {
  bw::Pass p = bw::make_pass(tag, blocks, (void*)workspace);
  if (level >= p.lv.n.size() || index >= p.lv.n[level]) return -1;
  uint64_t s[8];
  CK(cudaMemcpy(s, p.nodes + (p.lv.off[level] + index) * 8, 64, cudaMemcpyDeviceToHost));
  bw::digest_bytes(s, out);
  return 0;
}

uint32_t bw_levels(int tag, uint64_t blocks, uint64_t* sizes) {
  bw::Levels lv = bw::levels_for(bw::leaf_count(tag, blocks));
  for (size_t k = 0; k < lv.n.size(); ++k) sizes[k] = lv.n[k];
  return (uint32_t)lv.n.size();
}

int bw_fill_test_store(void* store, uint64_t blocks) {
  bw::fill_store<<<(unsigned)((blocks * 33 + 255) / 256), 256>>>((uint64_t*)store, blocks);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  return 0;
}

// The last pass's device time, from its first tag kernel to its root: the tags alone and the whole pass, in ms.
int bw_last_pass_ms(float* tags_ms, float* pass_ms) {
  if (!bw::pass_events[0]) return -1;
  CK(cudaEventElapsedTime(tags_ms, bw::pass_events[0], bw::pass_events[1]));
  CK(cudaEventElapsedTime(pass_ms, bw::pass_events[0], bw::pass_events[2]));
  return 0;
}

// Block j's 2,056 bytes.
int bw_read_block(const void* store, uint64_t j, uint8_t* out) {
  CK(cudaMemcpy(out, (const uint8_t*)store + j * bw::BLOCK_BYTES, bw::BLOCK_BYTES, cudaMemcpyDeviceToHost));
  return 0;
}

// Device memory for a harness without a CUDA binding of its own: nullptr on failure.
void* bw_alloc(size_t bytes) {
  void* p = nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}

int bw_free(void* p) {
  CK(cudaFree(p));
  return 0;
}

int bw_memset(void* p, int value, size_t bytes) {
  CK(cudaMemset(p, value, bytes));
  return 0;
}

// Reads `bytes` (a multiple of 16) of `buf` once and waits: an honest prover's step before a window, which leaves L2
// holding clean lines of `buf` (no part of E, nothing dirty).
int bw_read_scrub(const void* buf, size_t bytes) {
  static uint32_t* sink = nullptr;
  if (!sink) CK(cudaMalloc(&sink, 4));
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  bw::read_kernel<<<sms * 8, 512>>>((const uint4*)buf, bytes / 16, sink);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  return 0;
}

}  // extern "C"

#ifdef BW_MAIN
// ---------------------------------------------------------------------------------------------------------------------
// bwtag: validate against the Python reference's roots, then time.  Nothing is timed unless every check passed.

namespace {

using bw::Pass;

constexpr int IDLE_MS = 250;
constexpr int BUSY_READS = 16;  // about 6 ms at the raw read's rate

std::string hex(const uint8_t* b, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[b[i] >> 4];
    s += d[b[i] & 15];
  }
  return s;
}

bool unhex(const std::string& s, uint8_t* out, size_t n) {
  if (s.size() != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) out[i] = (uint8_t)strtoul(s.substr(2 * i, 2).c_str(), nullptr, 16);
  return true;
}

std::string arg(int argc, char** argv, const char* key, const char* dflt = nullptr) {
  std::string k = std::string("--") + key;
  for (int i = 1; i + 1 < argc; ++i)
    if (k == argv[i]) return argv[i + 1];
  if (!dflt) {
    fprintf(stderr, "missing %s\n", k.c_str());
    exit(64);
  }
  return dflt;
}

struct Stats {
  double min, p50, p90, p99, max, mean;
  size_t n;
};

Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stats s{};
  s.n = v.size();
  auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t)(f * (v.size() - 1) + 0.5))]; };
  s.min = v.front();
  s.max = v.back();
  s.p50 = q(0.5);
  s.p90 = q(0.9);
  s.p99 = q(0.99);
  double t = 0;
  for (double x : v) t += x;
  s.mean = t / v.size();
  return s;
}

std::string js(const Stats& s) {
  char b[256];
  snprintf(b, sizeof b, "{\"n\": %zu, \"min\": %.3f, \"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f, \"max\": %.3f, \"mean\": %.3f}",
           s.n, s.min, s.p50, s.p90, s.p99, s.max, s.mean);
  return b;
}

FILE* open_out(const std::string& dir, const char* name) {
  std::string path = dir + "/" + name;
  FILE* f = fopen(path.c_str(), "w");
  if (!f) {
    perror(path.c_str());
    exit(66);
  }
  return f;
}

int selftest() {
  // SHA-512("abc"), FIPS 180-2 appendix C.1.
  const uint64_t want[8] = {0xddaf35a193617abaULL, 0xcc417349ae204131ULL, 0x12e6fa4e89a97ea2ULL, 0x0a9eeee64b55d39aULL,
                            0x2192992a274fc1a8ULL, 0x36ba3c23a3feebbdULL, 0x454d4423643ce80eULL, 0x2a9ac94fa54ca49fULL};
  uint64_t host[8];
  bw::sha512_host((const uint8_t*)"abc", 3, host);
  uint64_t w[16] = {0x6162638000000000ULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24};
  uint64_t *dw, *ds, dev[8];
  CK(cudaMalloc(&dw, 128));
  CK(cudaMalloc(&ds, 64));
  CK(cudaMemcpy(dw, w, 128, cudaMemcpyHostToDevice));
  bw::sha512_selftest<<<1, 1>>>(dw, ds);
  CK(cudaMemcpy(dev, ds, 64, cudaMemcpyDeviceToHost));
  cudaFree(dw);
  cudaFree(ds);
  return memcmp(host, want, 64) == 0 && memcmp(dev, want, 64) == 0 ? 0 : 1;
}

int b3_selftest() {
  // BLAKE3's test vectors: keyed_hash of the empty input under "whats the Elvish word for friend".
  const bw::Key3 key = {{0x74616877u, 0x68742073u, 0x6c452065u, 0x68736976u, 0x726f7720u, 0x6f662064u, 0x72662072u,
                         0x646e6569u}};
  const uint32_t want[8] = {0x56b7b292u, 0x763ced04u, 0x626f9d1fu, 0x928a2c39u,
                            0xa30ead27u, 0xe77395f0u, 0x8a49f183u, 0x260dd64eu};
  uint32_t m[16] = {}, host[8], dev[8];
  bw::b3_compress(key.k, m, 0, 0, bw::B3_KEYED | bw::B3_CHUNK_START | bw::B3_CHUNK_END | bw::B3_ROOT, host);
  uint32_t* d;
  CK(cudaMalloc(&d, 32));
  bw::b3_selftest<<<1, 1>>>(key, d);
  CK(cudaMemcpy(dev, d, 32, cudaMemcpyDeviceToHost));
  cudaFree(d);
  return memcmp(host, want, 32) == 0 && memcmp(dev, want, 32) == 0 ? 0 : 1;
}

struct Validation {
  std::string name;
  std::string got, want;
  bool ok;
};

// One store's tag passes against the expected roots, with the output buffers poisoned first, and a control that skips
// the tag kernel (the tree over poison) and must not match.
int validate(uint64_t blocks, const uint8_t seed[32], const std::string want[bw::TAGS], std::vector<Validation>& out) {
  void* E;
  CK(cudaMalloc(&E, blocks * bw::BLOCK_BYTES));
  if (bw_fill_test_store(E, blocks)) return 1;
  for (int tag = 0; tag < bw::TAGS; ++tag) {
    size_t wsb = bw::workspace_bytes(tag, blocks);
    void* ws;
    CK(cudaMalloc(&ws, wsb));
    Pass p = bw::make_pass(tag, blocks, ws);
    uint8_t root[64];
    CK(cudaMemset(ws, 0xA5, wsb));
    if (bw::tag_pass(p, E, seed, root, 0, blocks)) return 1;
    std::string got = hex(root, 64);
    out.push_back({std::string(bw::TAG_NAMES[tag]) + " B=" + std::to_string(blocks), got, want[tag], got == want[tag]});
    CK(cudaMemset(ws, 0xA5, wsb));
    bw::launch_tree(p, 0);
    if (bw::read_root(p, root, 0)) return 1;
    std::string ctl = hex(root, 64);
    out.push_back({std::string(bw::TAG_NAMES[tag]) + " B=" + std::to_string(blocks) + " control (no tag kernel)", ctl,
                   "anything but " + want[tag], ctl != want[tag]});
    cudaFree(ws);
  }
  cudaFree(E);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = arg(argc, argv, "out");
  const uint64_t blocks = strtoull(arg(argc, argv, "blocks", "524288").c_str(), nullptr, 10);
  const int reps = atoi(arg(argc, argv, "reps", "100").c_str());
  uint8_t seed[32], small_seed[32];
  if (!unhex(arg(argc, argv, "seed"), seed, 32) || !unhex(arg(argc, argv, "small-seed"), small_seed, 32)) return 64;
  std::string want[bw::TAGS], small_want[bw::TAGS];
  for (int tag = 0; tag < bw::TAGS; ++tag) {
    want[tag] = arg(argc, argv, (std::string("expect-") + bw::TAG_SHORT[tag]).c_str());
    small_want[tag] = arg(argc, argv, (std::string("small-expect-") + bw::TAG_SHORT[tag]).c_str());
  }

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int bus_width = 0;
  cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);
  {
    FILE* f = open_out(dir, "device.json");
    char uuid[64];
    snprintf(uuid, sizeof uuid, "GPU-%s", hex((const uint8_t*)prop.uuid.bytes, 16).c_str());
    const double sm_bytes = (double)prop.regsPerMultiprocessor * 4 + 128.0 * 1024;
    fprintf(f,
            "{\"name\": \"%s\", \"uuid_hex\": \"%s\", \"cc\": \"%d.%d\", \"total_global_bytes\": %zu, \"l2_bytes\": %d, "
            "\"persisting_l2_max_bytes\": %d, \"sms\": %d, \"regs_per_sm\": %d, \"smem_per_sm_bytes\": %zu, "
            "\"smem_per_block_optin_bytes\": %zu, \"l1_smem_per_sm_bytes_assumed\": 131072, \"bus_width_bits\": %d, "
            "\"pci_bus_id\": %d, \"s_chip_bytes\": %.0f, \"s_chip_formula\": \"l2 + sms * (regs_per_sm * 4 + 128 KiB)\"}\n",
            prop.name, uuid, prop.major, prop.minor, prop.totalGlobalMem, prop.l2CacheSize, prop.persistingL2CacheMaxSize,
            prop.multiProcessorCount, prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor,
            prop.sharedMemPerBlockOptin, bus_width, prop.pciBusID,
            (double)prop.l2CacheSize + prop.multiProcessorCount * sm_bytes);
    fclose(f);
  }

  // Gates.
  std::vector<Validation> checks;
  const int st = selftest();
  checks.push_back({"sha512('abc') host and device", st ? "mismatch" : "match", "match", st == 0});
  const int b3 = b3_selftest();
  checks.push_back({"blake3 keyed('') host and device", b3 ? "mismatch" : "match", "match", b3 == 0});
  if (validate(37, small_seed, small_want, checks)) return 1;
  if (validate(blocks, seed, want, checks)) return 1;
  bool all = true;
  {
    FILE* f = open_out(dir, "validation.json");
    fprintf(f, "{\"checks\": [\n");
    for (size_t i = 0; i < checks.size(); ++i) {
      all = all && checks[i].ok;
      fprintf(f, "  {\"check\": \"%s\", \"ok\": %s, \"got\": \"%s\", \"want\": \"%s\"}%s\n", checks[i].name.c_str(),
              checks[i].ok ? "true" : "false", checks[i].got.c_str(), checks[i].want.c_str(),
              i + 1 < checks.size() ? "," : "");
    }
    fprintf(f, "], \"all_ok\": %s}\n", all ? "true" : "false");
    fclose(f);
  }
  if (!all) {
    fprintf(stderr, "validation failed: nothing timed (see validation.json)\n");
    return 3;
  }

  // Timing.  E is the full store; a tag's timed reps run under three L2 conditions (cold, warm, clean).
  void* E;
  CK(cudaMalloc(&E, blocks * bw::BLOCK_BYTES));
  if (bw_fill_test_store(E, blocks)) return 1;
  const size_t scrub_bytes = 512ull << 20;
  void* scrub;
  CK(cudaMalloc(&scrub, scrub_bytes));
  CK(cudaMemset(scrub, 0, scrub_bytes));
  uint32_t* sink;
  CK(cudaMalloc(&sink, 4));
  cudaEvent_t e0, e1, e2;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&e2));
  auto ms = [](cudaEvent_t a, cudaEvent_t b) {
    float t;
    cudaEventElapsedTime(&t, a, b);
    return (double)t;
  };
  const double ebytes = (double)blocks * bw::BLOCK_BYTES;

  FILE* f = open_out(dir, "timing.json");
  fprintf(f, "{\"blocks\": %llu, \"store_bytes\": %.0f, \"reps\": %d,\n", (unsigned long long)blocks, ebytes, reps);

  // Raw device read of E.
  {
    std::vector<double> t, gbs;
    int sms = prop.multiProcessorCount;
    for (int r = 0; r < reps; ++r) {
      CK(cudaMemsetAsync(scrub, r & 0xff, scrub_bytes));
      CK(cudaEventRecord(e0));
      bw::read_kernel<<<sms * 8, 512>>>((const uint4*)E, (uint64_t)(ebytes / 16), sink);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      t.push_back(ms(e0, e1));
      gbs.push_back(ebytes / (t.back() * 1e-3) / 1e9);
    }
    fprintf(f, " \"raw_read_ms\": %s,\n \"raw_read_gbs\": %s,\n", js(stats(t)).c_str(), js(stats(gbs)).c_str());
  }

  for (int tag = 0; tag < bw::TAGS; ++tag) {
    const char* nm = bw::TAG_SHORT[tag];
    void* ws;
    CK(cudaMalloc(&ws, bw::workspace_bytes(tag, blocks)));
    Pass p = bw::make_pass(tag, blocks, ws);
    // cold: L2 full of dirty lines (a 512 MiB write); warm: whatever the last pass left; clean: L2 holding clean lines
    // of another buffer (a 512 MiB read), which an honest prover can arrange before every window.  idle: clean after
    // the GPU sat idle for IDLE_MS (as between the harness's windows, while the host opens and checks); busy: idle, then
    // BUSY_READS reads of that buffer before the pass (a GPU kept busy until the seed, the last read the clean one).
    // idle and busy run at most 20 reps.
    const char* const conds[5] = {"cold", "warm", "clean", "idle", "busy"};
    for (int cond = 0; cond < 5; ++cond) {
      std::vector<double> dev, tags_ms, tree_ms, host;
      for (int r = 0; r < (cond >= 3 ? std::min(reps, 20) : reps); ++r) {
        uint8_t s[32], root[64];
        memcpy(s, seed, 32);
        s[0] ^= (uint8_t)r;
        s[1] ^= (uint8_t)(r >> 8);
        s[2] ^= (uint8_t)(cond + 1);
        if (cond == 0) CK(cudaMemsetAsync(scrub, r & 0xff, scrub_bytes));
        if (cond >= 3) std::this_thread::sleep_for(std::chrono::milliseconds(IDLE_MS));
        for (int k = 0; k < (cond == 4 ? BUSY_READS : cond >= 2 ? 1 : 0); ++k)
          bw::read_kernel<<<prop.multiProcessorCount * 8, 512>>>((const uint4*)scrub, scrub_bytes / 16, sink);
        CK(cudaDeviceSynchronize());
        auto h0 = std::chrono::steady_clock::now();
        CK(cudaEventRecord(e0));
        bw::launch_tags(p, E, s, blocks, 0);
        CK(cudaEventRecord(e1));
        bw::launch_tree(p, 0);
        CK(cudaEventRecord(e2));
        if (bw::read_root(p, root, 0)) return 1;
        auto h1 = std::chrono::steady_clock::now();
        dev.push_back(ms(e0, e2));
        tags_ms.push_back(ms(e0, e1));
        tree_ms.push_back(ms(e1, e2));
        host.push_back(std::chrono::duration<double, std::milli>(h1 - h0).count());
      }
      fprintf(f, " \"%s_%s\": {\"device_ms\": %s, \"tags_ms\": %s, \"tree_ms\": %s, \"host_e2e_ms\": %s},\n", nm,
              conds[cond], js(stats(dev)).c_str(), js(stats(tags_ms)).c_str(), js(stats(tree_ms)).c_str(),
              js(stats(host)).c_str());
    }
    // A prover that skips a fraction f of the blocks (and guesses their tags): the tag kernel over the rest.
    fprintf(f, " \"%s_skip\": [", nm);
    const double fracs[] = {0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50};
    for (size_t fi = 0; fi < sizeof fracs / sizeof *fracs; ++fi) {
      const uint64_t count = (uint64_t)((1.0 - fracs[fi]) * blocks);
      std::vector<double> dev;
      for (int r = 0; r < reps; ++r) {
        CK(cudaMemsetAsync(scrub, r & 0xff, scrub_bytes));
        CK(cudaEventRecord(e0));
        bw::launch_tags(p, E, seed, count, 0);
        bw::launch_tree(p, 0);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        dev.push_back(ms(e0, e1));
      }
      fprintf(f, "%s{\"f\": %.2f, \"blocks_tagged\": %llu, \"device_ms\": %s}", fi ? ", " : "", fracs[fi],
              (unsigned long long)count, js(stats(dev)).c_str());
    }
    fprintf(f, "],\n");
    cudaFree(ws);
  }

  // PCIe: pinned copies both ways, and a kernel reading pinned host memory in place (zero-copy), 1 GiB each.
  {
    const size_t n = 1ull << 30;
    void *hp, *dp, *mapped;
    CK(cudaHostAlloc(&hp, n, cudaHostAllocMapped));
    memset(hp, 1, n);
    CK(cudaHostGetDevicePointer(&mapped, hp, 0));
    dp = scrub;  // 512 MiB: copy halves
    std::vector<double> h2d, d2h, zc;
    for (int r = 0; r < 5; ++r) {
      CK(cudaEventRecord(e0));
      CK(cudaMemcpyAsync(dp, hp, scrub_bytes, cudaMemcpyHostToDevice));
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      h2d.push_back(scrub_bytes / (ms(e0, e1) * 1e-3) / 1e9);
      CK(cudaEventRecord(e0));
      CK(cudaMemcpyAsync(hp, dp, scrub_bytes, cudaMemcpyDeviceToHost));
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      d2h.push_back(scrub_bytes / (ms(e0, e1) * 1e-3) / 1e9);
      CK(cudaEventRecord(e0));
      bw::read_kernel<<<prop.multiProcessorCount * 8, 512>>>((const uint4*)mapped, n / 16, sink);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      zc.push_back(n / (ms(e0, e1) * 1e-3) / 1e9);
    }
    fprintf(f, " \"pcie_h2d_gbs\": %s,\n \"pcie_d2h_gbs\": %s,\n \"pcie_zero_copy_read_gbs\": %s\n}\n",
            js(stats(h2d)).c_str(), js(stats(d2h)).c_str(), js(stats(zc)).c_str());
    cudaFreeHost(hp);
  }
  fclose(f);
  cudaFree(E);
  cudaFree(scrub);
  printf("ok: %s/{device,validation,timing}.json\n", dir.c_str());
  return 0;
}
#endif
