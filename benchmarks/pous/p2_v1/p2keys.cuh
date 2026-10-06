// P2 v1's per-block keys on the GPU: K = SHAKE256(prefix || u64be(segment) || u64be(block) || "K")[0:W/8] and t from the same
// with "t" (its first candidate), one thread per block, both written as big-endian rows like `gpu.Kernels.keys`.
//
// Every block's message shares the prefix, so the host absorbs the prefix's whole rate blocks once (the midstate) and bakes the
// rest of it, each purpose byte and the padding into the last block's words; a thread xors in its segment and block, then
// permutes once to absorb and again for each further rate block of the W/8 bytes it squeezes (16 permutations at 2,056 bytes). Squeezed words go through
// shared memory so that each warp writes 136-byte runs of its 32 rows.
#pragma once
#include <cstdint>

namespace p2keys {

constexpr int RATE = 136, RW = RATE / 8;

#define P2K_RC_LIST                                                                                                        \
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL, 0x000000000000808bULL,         \
      0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL, 0x0000000000000088ULL,     \
      0x0000000080008009ULL, 0x000000008000000aULL, 0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,     \
      0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,     \
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
__constant__ uint64_t RC_DEV[24] = {P2K_RC_LIST};
static const uint64_t RC_HOST[24] = {P2K_RC_LIST};
#undef P2K_RC_LIST

__host__ __device__ __forceinline__ uint64_t rotl(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

__host__ __device__ __forceinline__ void keccak_f(uint64_t st[25]) {
  constexpr int PI_DEST[25] = {0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4};
  constexpr int RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
#ifdef __CUDA_ARCH__
  const uint64_t *rc = RC_DEV;
#else
  const uint64_t *rc = RC_HOST;
#endif
#pragma unroll 1
  for (int r = 0; r < 24; ++r) {
    uint64_t bc[5], b[25];
#pragma unroll
    for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint64_t t = bc[(i + 4) % 5] ^ rotl(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
#pragma unroll
    for (int s = 0; s < 25; ++s) b[PI_DEST[s]] = RHO[s] ? rotl(st[s], RHO[s]) : st[s];
#pragma unroll
    for (int j = 0; j < 25; j += 5)
#pragma unroll
      for (int i = 0; i < 5; ++i) st[j + i] = b[j + i] ^ ((~b[j + (i + 1) % 5]) & b[j + (i + 2) % 5]);
    st[0] ^= rc[r];
  }
}

// The state after the prefix's whole rate blocks; the last block's words per purpose (the prefix's tail, the purpose byte,
// SHAKE's padding); where segment || block go in it.
struct Mid {
  uint64_t st[25];
  uint64_t last[2][RW];
  int at;
};

// Mid for this prefix, or false when its tail and the 17 bytes per block don't fit one rate block.
inline bool midstate(const uint8_t *prefix, int plen, Mid &m) {
  const int full = plen / RATE, tail = plen % RATE;
  if (tail + 17 + 1 > RATE) return false;
  for (int i = 0; i < 25; i++) m.st[i] = 0;
  for (int b = 0; b < full; b++) {
    for (int w = 0; w < RW; w++) {
      uint64_t x = 0;
      for (int i = 0; i < 8; i++) x |= (uint64_t)prefix[b * RATE + 8 * w + i] << (8 * i);
      m.st[w] ^= x;
    }
    keccak_f(m.st);
  }
  const uint8_t purpose[2] = {'K', 't'};
  for (int p = 0; p < 2; p++) {
    uint8_t blk[RATE] = {0};
    for (int i = 0; i < tail; i++) blk[i] = prefix[full * RATE + i];
    blk[tail + 16] = purpose[p];
    blk[tail + 17] ^= 0x1F;
    blk[RATE - 1] ^= 0x80;
    for (int w = 0; w < RW; w++) {
      uint64_t x = 0;
      for (int i = 0; i < 8; i++) x |= (uint64_t)blk[8 * w + i] << (8 * i);
      m.last[p][w] = x;
    }
  }
  m.at = tail;
  return true;
}

constexpr int WARPS = 4;
#ifndef P2K_MINB
#define P2K_MINB 4
#endif
constexpr int OUT_WORDS = P2_BITS / 64;   // W/8 bytes
static_assert(P2_BITS % 64 == 0, "rows of whole 64-bit words");

// rows k and t (each W/64 64-bit words, the SHAKE output bytes in order) of blocks idx0 + r, r < rows <= 32, lane r
// expanding row r; st_ is the warp's 32 * RW words of staging
__device__ __forceinline__ void keys_rows(const Mid &mid, uint64_t idx0, uint32_t rows, uint64_t per_segment,
                                          uint64_t *__restrict__ kout, uint64_t *__restrict__ tout, uint64_t *st_,
                                          uint32_t lane) {
  const uint64_t idx = idx0 + lane;
  const uint64_t seg = idx / per_segment, blkno = idx % per_segment;
  uint64_t sb[RW];
#pragma unroll
  for (int w = 0; w < RW; w++) {
    uint64_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int o = 8 * w + i - mid.at;   // byte o of u64be(seg) || u64be(block)
      if (o >= 0 && o < 16) x |= ((o < 8 ? seg >> (56 - 8 * o) : blkno >> (56 - 8 * (o - 8))) & 0xFF) << (8 * i);
    }
    sb[w] = x;
  }
#pragma unroll 1
  for (int p = 0; p < 2; p++) {
    uint64_t *out = p ? tout : kout;
    uint64_t st[25];
#pragma unroll
    for (int i = 0; i < 25; i++) st[i] = mid.st[i] ^ (i < RW ? mid.last[p][i] ^ sb[i] : 0);
#pragma unroll 1
    for (int b = 0; b < (OUT_WORDS + RW - 1) / RW; b++) {
      keccak_f(st);
      __syncwarp();
#pragma unroll
      for (int w = 0; w < RW; w++) st_[lane * RW + w] = st[w];
      __syncwarp();
      const int words = min(RW, OUT_WORDS - RW * b);
      for (uint32_t e = lane; e < rows * (uint32_t)words; e += 32) {
        const uint32_t r = e / words, w = e % words;
        out[(size_t)r * OUT_WORDS + RW * b + w] = st_[r * RW + w];
      }
    }
  }
}

// rows k and t (each [count, W/64]) for blocks first .. first + count - 1, 32 rows a warp, the grid striding over them
__global__ void __launch_bounds__(32 * WARPS, P2K_MINB) keys_kernel(const __grid_constant__ Mid mid, uint64_t first, uint32_t count,
                                                          uint64_t per_segment, uint64_t *__restrict__ kout,
                                                          uint64_t *__restrict__ tout) {
  __shared__ uint64_t stage[WARPS][32 * RW];
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  for (uint64_t row0 = ((uint64_t)blockIdx.x * WARPS + wid) * 32; row0 < count; row0 += (uint64_t)gridDim.x * WARPS * 32) {
    const uint32_t rows = count - row0 < 32 ? (uint32_t)(count - row0) : 32;
    keys_rows(mid, first + row0, rows, per_segment, kout + row0 * OUT_WORDS, tout + row0 * OUT_WORDS, stage[wid], lane);
  }
}

}  // namespace p2keys
