// P2's per-block keys with the v4 framing on the GPU (`reference.keys`, the spec's section 4): row P (K or t) is the first
// W/8 bytes of SHA-512(prefix || u64be(segment) || u64be(block) || P || u32be(i)) for i = 0, 1, ..., written as big-endian
// rows like `gpu.Kernels.keys`.
//
// The v4 prefix (`reference.expand_prefix`) is whole SHA-512 blocks (256 bytes), so the host compresses it once (the
// midstate) and every 64-byte chunk is one compression of a last block whose words are fixed but for W0 = segment,
// W1 = block and W2 = P << 56 | i << 24 | 0x80 << 16 (W15 is the message's length in bits). A row is 2,056 bytes,
// 33 compressions, the last one's H0 alone.
//
//   P2K4_MODE 1  a thread a (purpose, row), its 33 chunks in turn (the default)
//   P2K4_MODE 0  a thread a (purpose, row, chunk), consecutive threads consecutive chunks
//   P2K4_STAGE   (mode 0) each warp's 32 chunks go out through shared memory as one contiguous run of 8-byte words
//   P2K4_ROLL    rounds 32 to 79 as a loop of 16 unrolled rounds (less code), K from constant memory
#pragma once
#include <cstdint>

namespace p2keys4 {

#define P2K4_K_LIST                                                                                                        \
  0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL,         \
      0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,     \
      0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,     \
      0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,     \
      0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL,     \
      0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,     \
      0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL,     \
      0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,     \
      0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,     \
      0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,     \
      0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL,     \
      0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,     \
      0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL,     \
      0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,     \
      0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,     \
      0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
__constant__ uint64_t K_DEV[80] = {P2K4_K_LIST};
static const uint64_t K_HOST[80] = {P2K4_K_LIST};
#undef P2K4_K_LIST

__host__ __device__ __forceinline__ uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
__host__ __device__ __forceinline__ uint64_t S0(uint64_t x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
__host__ __device__ __forceinline__ uint64_t S1(uint64_t x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }
__host__ __device__ __forceinline__ uint64_t s0(uint64_t x) { return rotr(x, 1) ^ rotr(x, 8) ^ (x >> 7); }
__host__ __device__ __forceinline__ uint64_t s1(uint64_t x) { return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6); }
__host__ __device__ __forceinline__ uint64_t ch(uint64_t e, uint64_t f, uint64_t g) { return g ^ (e & (f ^ g)); }
__host__ __device__ __forceinline__ uint64_t maj(uint64_t a, uint64_t b, uint64_t c) { return (a & b) | (c & (a | b)); }

__host__ __device__ __forceinline__ const uint64_t *kconst() {
#ifdef __CUDA_ARCH__
  return K_DEV;
#else
  return K_HOST;
#endif
}

// a round, and one whose schedule word is zero
#define P2K4_ROUNDX(a, b, c, d, e, f, g, h, kw)                                                                          \
  do {                                                                                                                     \
    const uint64_t t1_ = (h) + S1(e) + ch(e, f, g) + (kw);                                                               \
    (d) += t1_;                                                                                                            \
    (h) = t1_ + S0(a) + maj(a, b, c);                                                                                     \
  } while (0)
#define P2K4_ROUND(a, b, c, d, e, f, g, h, k, w) P2K4_ROUNDX(a, b, c, d, e, f, g, h, (k) + (w))
#define P2K4_ROUND0(a, b, c, d, e, f, g, h, k) P2K4_ROUNDX(a, b, c, d, e, f, g, h, (k))

// The state after the prefix's whole blocks, and the last block's W15 (the message's length in bits).
struct Mid {
  uint64_t h[8];
  uint64_t w15;
};

constexpr uint64_t IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                            0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

inline void compress_host(uint64_t hs[8], const uint64_t in[16]) {
  uint64_t w[80], v[8];
  for (int i = 0; i < 16; i++) w[i] = in[i];
  for (int i = 16; i < 80; i++) w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];
  for (int i = 0; i < 8; i++) v[i] = hs[i];
  for (int t = 0; t < 80; t++) {
    const uint64_t t1 = v[7] + S1(v[4]) + ch(v[4], v[5], v[6]) + K_HOST[t] + w[t], t2 = S0(v[0]) + maj(v[0], v[1], v[2]);
    for (int i = 7; i > 0; i--) v[i] = v[i - 1];
    v[4] += t1;
    v[0] = t1 + t2;
  }
  for (int i = 0; i < 8; i++) hs[i] += v[i];
}

inline uint64_t be64(const uint8_t *p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x = x << 8 | p[i];
  return x;
}

// Mid for this prefix, or false when it is not whole SHA-512 blocks.
inline bool midstate(const uint8_t *prefix, int plen, Mid &m) {
  if (plen <= 0 || plen % 128 != 0) return false;
  for (int i = 0; i < 8; i++) m.h[i] = IV[i];
  for (int b = 0; b < plen / 128; b++) {
    uint64_t w[16];
    for (int i = 0; i < 16; i++) w[i] = be64(prefix + 128 * b + 8 * i);
    compress_host(m.h, w);
  }
  m.w15 = (uint64_t)(plen + 16 + 1 + 4) * 8;
  return true;
}

// The digest of chunk (seg, blk, w4): the last block's 80 rounds from the midstate.
__host__ __device__ __forceinline__ void chunk(const Mid &m, uint64_t seg, uint64_t blk, uint64_t w4, uint64_t o[8]) {
  const uint64_t *K = kconst();
  // round t's a .. h are the names rotated right by t, so after round 79 they are a .. h again
  uint64_t a = m.h[0], b = m.h[1], c = m.h[2], d = m.h[3], e = m.h[4], f = m.h[5], g = m.h[6], h = m.h[7];
  uint64_t W[16] = {seg, blk, w4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m.w15};
  P2K4_ROUND(a, b, c, d, e, f, g, h, K[0], W[0]);
  P2K4_ROUND(h, a, b, c, d, e, f, g, K[1], W[1]);
  P2K4_ROUND(g, h, a, b, c, d, e, f, K[2], W[2]);
  P2K4_ROUND0(f, g, h, a, b, c, d, e, K[3]);
  P2K4_ROUND0(e, f, g, h, a, b, c, d, K[4]);
  P2K4_ROUND0(d, e, f, g, h, a, b, c, K[5]);
  P2K4_ROUND0(c, d, e, f, g, h, a, b, K[6]);
  P2K4_ROUND0(b, c, d, e, f, g, h, a, K[7]);
  P2K4_ROUND0(a, b, c, d, e, f, g, h, K[8]);
  P2K4_ROUND0(h, a, b, c, d, e, f, g, K[9]);
  P2K4_ROUND0(g, h, a, b, c, d, e, f, K[10]);
  P2K4_ROUND0(f, g, h, a, b, c, d, e, K[11]);
  P2K4_ROUND0(e, f, g, h, a, b, c, d, K[12]);
  P2K4_ROUND0(d, e, f, g, h, a, b, c, K[13]);
  P2K4_ROUND0(c, d, e, f, g, h, a, b, K[14]);
  P2K4_ROUND(b, c, d, e, f, g, h, a, K[15], W[15]);
#define P2K4_W(j) (W[(j) & 15] += s1(W[((j) + 14) & 15]) + W[((j) + 9) & 15] + s0(W[((j) + 1) & 15]))
// rounds tk .. tk + 7 on schedule words tw .. tw + 7 (tw = tk mod 16, a constant even where tk is not)
#define P2K4_EIGHT(tk, tw, wa)                                                                                             \
  P2K4_ROUND(a, b, c, d, e, f, g, h, K[(tk) + 0], wa);                                                                   \
  P2K4_ROUND(h, a, b, c, d, e, f, g, K[(tk) + 1], P2K4_W((tw) + 1));                                                     \
  P2K4_ROUND(g, h, a, b, c, d, e, f, K[(tk) + 2], P2K4_W((tw) + 2));                                                     \
  P2K4_ROUND(f, g, h, a, b, c, d, e, K[(tk) + 3], P2K4_W((tw) + 3));                                                     \
  P2K4_ROUND(e, f, g, h, a, b, c, d, K[(tk) + 4], P2K4_W((tw) + 4));                                                     \
  P2K4_ROUND(d, e, f, g, h, a, b, c, K[(tk) + 5], P2K4_W((tw) + 5));                                                     \
  P2K4_ROUND(c, d, e, f, g, h, a, b, K[(tk) + 6], P2K4_W((tw) + 6));                                                     \
  P2K4_ROUND(b, c, d, e, f, g, h, a, K[(tk) + 7], P2K4_W((tw) + 7))
  P2K4_EIGHT(16, 0, P2K4_W(0));
  P2K4_EIGHT(24, 8, P2K4_W(8));
#ifdef P2K4_ROLL
#pragma unroll 1
  for (int t = 32; t < 80; t += 16) {
    P2K4_EIGHT(t, 0, P2K4_W(0));
    P2K4_EIGHT(t + 8, 8, P2K4_W(8));
  }
#else
  P2K4_EIGHT(32, 0, P2K4_W(0));
  P2K4_EIGHT(40, 8, P2K4_W(8));
  P2K4_EIGHT(48, 0, P2K4_W(0));
  P2K4_EIGHT(56, 8, P2K4_W(8));
  P2K4_EIGHT(64, 0, P2K4_W(0));
  P2K4_EIGHT(72, 8, P2K4_W(8));
#endif
#undef P2K4_EIGHT
#undef P2K4_W
  o[0] = m.h[0] + a, o[1] = m.h[1] + b, o[2] = m.h[2] + c, o[3] = m.h[3] + d;
  o[4] = m.h[4] + e, o[5] = m.h[5] + f, o[6] = m.h[6] + g, o[7] = m.h[7] + h;
}
#undef P2K4_ROUND
#undef P2K4_ROUND0
#undef P2K4_ROUNDX

__host__ __device__ __forceinline__ uint64_t w4(int purpose, uint32_t i) {
  return (uint64_t)(purpose ? 't' : 'K') << 56 | (uint64_t)i << 24 | 0x80ULL << 16;
}

constexpr int ROW_WORDS = P2_BITS / 64;   // W/8 bytes
constexpr int CHUNKS = (ROW_WORDS + 7) / 8;
static_assert(P2_BITS % 64 == 0, "rows of whole 64-bit words");

__device__ __forceinline__ uint64_t bswap(uint64_t x) {
  return (uint64_t)__byte_perm((uint32_t)x, 0, 0x0123) << 32 | __byte_perm((uint32_t)(x >> 32), 0, 0x0123);
}

#ifndef P2K4_MODE
#define P2K4_MODE 1
#endif
#ifndef P2K4_TPB
#define P2K4_TPB 128
#endif
#ifndef P2K4_MINB
#define P2K4_MINB 1
#endif

// rows k and t (each [count, W/64]) of blocks first .. first + count - 1; `per` is count * CHUNKS rounded up to whole warps
// (mode 0) or count (mode 1), the threads of one purpose
__global__ void __launch_bounds__(P2K4_TPB, P2K4_MINB) keys_kernel(const __grid_constant__ Mid mid, uint64_t first, uint32_t count,
                                                                  uint64_t per_segment, uint64_t per, uint64_t *__restrict__ kout,
                                                                  uint64_t *__restrict__ tout) {
  const uint64_t gid = (uint64_t)blockIdx.x * P2K4_TPB + threadIdx.x;
  if (gid >= 2 * per) return;
  const int p = gid >= per;
  const uint64_t q = gid - (p ? per : 0);
  uint64_t *out = p ? tout : kout;
#if P2K4_MODE == 0
  const uint64_t r = q / CHUNKS;
  const uint32_t c = (uint32_t)(q % CHUNKS);
  const bool live = r < count;
  uint64_t o[8];
  if (live) {
    const uint64_t idx = first + r;
    chunk(mid, idx / per_segment, idx % per_segment, w4(p, c), o);
  }
  const uint64_t at = r * ROW_WORDS + 8 * c;   // this chunk's first word
#ifdef P2K4_STAGE
  // the warp's chunks are one purpose's consecutive chunks, so their words are one run from its first lane's
  __shared__ uint64_t stage[P2K4_TPB / 32][32 * 8];
  uint64_t *st = stage[threadIdx.x / 32];
  const uint32_t lane = threadIdx.x % 32;
  const uint64_t base = __shfl_sync(0xffffffffu, at, 0);
  const uint32_t n = 8 * c + 8 <= ROW_WORDS ? 8 : ROW_WORDS - 8 * c;
  if (live) {
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (j < n) st[at - base + j] = bswap(o[j]);
  }
  __syncwarp();
  const uint32_t words = __reduce_max_sync(0xffffffffu, live ? (uint32_t)(at - base) + n : 0);
  for (uint32_t e = lane; e < words; e += 32) out[base + e] = st[e];
#else
  if (live) {
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (8 * c + j < ROW_WORDS) out[at + j] = bswap(o[j]);
  }
#endif
#else
  const uint64_t idx = first + q, seg = idx / per_segment, blk = idx % per_segment;
  uint64_t *row = out + q * ROW_WORDS;
#pragma unroll 1
  for (uint32_t c = 0; c < CHUNKS; c++) {
    uint64_t o[8];
    chunk(mid, seg, blk, w4(p, c), o);
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (8 * c + j < ROW_WORDS) row[8 * c + j] = bswap(o[j]);
  }
#endif
}

inline uint64_t threads_per_purpose(uint32_t count) {
#if P2K4_MODE == 0
  return ((uint64_t)count * CHUNKS + 31) / 32 * 32;
#else
  return count;
#endif
}

}  // namespace p2keys4
