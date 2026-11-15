// lfill: the register-only lane filler, for the SMs a served step holds. No
// shared memory beyond a block's 32 words of reduction, no MMA, a few registers a thread, so it co-resides with every
// decode kernel at the lowest stream priority and takes only the issue slots and scalar pipes the step leaves. Its work
// is verified lane chains under FILL-OPT, a lane step (lf_step: LOP3 IMAD SHF IMAD) on words fresh from the round's
// coins:
//
// A tile is one block of T threads. K = dseed(c0, c1, tile); the tile's keys (a0, a1) are the first two u32 halves of
// smix(K ^ LF_KEYS ^ i), i = 0, 1, ..., that pass pf_key_ok (odd, far from 0 mod 2^24, not ±1 mod 32); thread u's words
// are (x, y) = the halves of smix(K ^ (LF_POS << 32 | u)); its chain q < CH starts at pf_y0(LF_POS | (CH u + q)) and takes
// `steps` lane steps on (x, y); the tile's receipt is the XOR over its threads' chains of pf_fold(K, LF_POS | (CH u + q),
// v). A thread's CH chains are independent, so CH sets its issue rate when it has the SMSP to itself (latency-bound at
// small CH) and the blocks an SM set its share beside the step. lf_ref replays a tile on the host.
#pragma once

#ifndef LF_CH
#define LF_CH 8
#endif
static_assert(LF_CH == 1 || LF_CH == 2 || LF_CH == 4 || LF_CH == 8 || LF_CH == 16, "LF_CH is 1, 2, 4, 8 or 16");
constexpr uint32_t LF_POS = 0x20000000u;
constexpr uint64_t LF_KEYS = 0x4C46494C4C4B4559ull;  // "LFILLKEY"

__host__ __device__ __forceinline__ uint32_t lf_step(uint32_t v, uint32_t x, uint32_t y, uint32_t a0, uint32_t a1) {
  v = (v ^ x) * a0 + y;
  return pf_rotl(v, 13) * a1 + x;
}

__host__ __device__ inline void lf_keys(uint64_t K, uint32_t* a0, uint32_t* a1) {
  int n = 0;
  for (uint64_t i = 0; n < 2; i++) {
    const uint64_t r = smix(K ^ LF_KEYS ^ i);
    for (int h = 0; h < 2 && n < 2; h++) {
      const uint32_t a = h ? (uint32_t)(r >> 32) : (uint32_t)r;
      if (pf_key_ok(a)) {
        if (n++ == 0) *a0 = a;
        else *a1 = a;
      }
    }
  }
}
__host__ __device__ __forceinline__ uint64_t lf_thread(uint64_t K, uint32_t a0, uint32_t a1, uint32_t u, int steps) {
  const uint64_t r = smix(K ^ ((uint64_t)LF_POS << 32 | u));
  const uint32_t x = (uint32_t)r, y = (uint32_t)(r >> 32);
  uint32_t v[LF_CH];
#pragma unroll
  for (int q = 0; q < LF_CH; q++) v[q] = pf_y0(LF_POS | (LF_CH * u + q));
#pragma unroll 1
  for (int s = 0; s < steps; s++) {
#pragma unroll
    for (int q = 0; q < LF_CH; q++) v[q] = lf_step(v[q], x, y, a0, a1);
  }
  uint64_t h = 0;
#pragma unroll
  for (int q = 0; q < LF_CH; q++) h ^= pf_fold(K, LF_POS | (LF_CH * u + q), v[q]);
  return h;
}
// the warp instructions of a tile's chains: T / 32 warps x CH chains x steps x 4
__host__ __device__ constexpr double lf_tile_instr(int threads, int steps) { return threads / 32.0 * LF_CH * steps * 4.0; }

#ifdef __CUDACC__
__global__ void __launch_bounds__(1024) lfill_k(uint64_t c0, uint64_t c1, unsigned long long tile0, int tiles, int steps,
                                                unsigned long long* __restrict__ rec) {
  __shared__ unsigned long long red[32];
  for (int i = 0; i < tiles; i++) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * tiles + i;
    const uint64_t K = dseed(c0, c1, tile0 + idx);
    uint32_t a0, a1;
    lf_keys(K, &a0, &a1);
    uint64_t h = lf_thread(K, a0, a1, threadIdx.x, steps);
#pragma unroll
    for (int o = 16; o; o >>= 1) h ^= __shfl_xor_sync(~0u, h, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t x = 0;
      for (int v = 0; v < (int)(blockDim.x >> 5); v++) x ^= red[v];
      rec[idx] = x;
    }
    __syncthreads();
  }
}
#endif

static uint64_t lf_ref(uint64_t c0, uint64_t c1, unsigned long long tile, int threads, int steps) {
  const uint64_t K = dseed(c0, c1, tile);
  uint32_t a0 = 0, a1 = 0;
  lf_keys(K, &a0, &a1);
  uint64_t h = 0;
  for (int u = 0; u < threads; u++) h ^= lf_thread(K, a0, a1, (uint32_t)u, steps);
  return h;
}
