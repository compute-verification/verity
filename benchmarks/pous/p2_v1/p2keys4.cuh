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

#include "../../../verity/ml/kernels/sha512.cuh"

namespace p2keys4 {

// a round, and one whose schedule word is zero
#define P2K4_ROUNDX(a, b, c, d, e, f, g, h, kw)                                                                          \
  do {                                                                                                                     \
    const uint64_t t1_ = (h) + sx::S1<0>(e) + sx::ch(e, f, g) + (kw);                                                    \
    (d) += t1_;                                                                                                            \
    (h) = t1_ + sx::S0<0>(a) + sx::maj(a, b, c);                                                                          \
  } while (0)
#define P2K4_ROUND(a, b, c, d, e, f, g, h, k, w) P2K4_ROUNDX(a, b, c, d, e, f, g, h, (k) + (w))
#define P2K4_ROUND0(a, b, c, d, e, f, g, h, k) P2K4_ROUNDX(a, b, c, d, e, f, g, h, (k))

// The state after the prefix's whole blocks, and the last block's W15 (the message's length in bits).
struct Mid {
  uint64_t h[8];
  uint64_t w15;
};

// Mid for this prefix, or false when it is not whole SHA-512 blocks.
inline bool midstate(const uint8_t *prefix, int plen, Mid &m) {
  if (plen <= 0 || plen % 128 != 0) return false;
  sx::iv(m.h);
  for (int b = 0; b < plen / 128; b++) {
    uint64_t w[16];
    for (int i = 0; i < 16; i++) w[i] = sha::be64(prefix + 128 * b + 8 * i);
    sx::compress<0, 0>(m.h, w);
  }
  m.w15 = (uint64_t)(plen + 16 + 1 + 4) * 8;
  return true;
}

// The digest of chunk (seg, blk, w4): the last block's 80 rounds from the midstate.
__host__ __device__ __forceinline__ void chunk(const Mid &m, uint64_t seg, uint64_t blk, uint64_t w4, uint64_t o[8]) {
  const uint64_t *K = sx::kconst();
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
#define P2K4_W(j) (W[(j) & 15] += sx::s1<0>(W[((j) + 14) & 15]) + W[((j) + 9) & 15] + sx::s0<0>(W[((j) + 1) & 15]))
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
