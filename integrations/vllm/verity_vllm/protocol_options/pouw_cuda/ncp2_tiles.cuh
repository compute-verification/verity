// A tile's route-U accumulator hashed as it is computed: the function (`tile_fused`, and the low-byte variant, timed only) that
// `ncp2_host.cpp`'s twin runs on the CPU against `pouw_native`, and on the device `kf_tiles` and its launch (`launch_tiles`),
// which `ncp2_fused.cu` calls. Everything about how a tile's chain is computed and scheduled is here: the running sums' storage,
// the SHA-256 compression the word leaf uses, the block shape and its shared memory. The leaves and their order are fixed by
// `pouw_native`; `pouw_device.first_difference` holds every build of this file to it.
#pragma once
#include "ncp2_fast.cuh"

namespace ncp2 {

// ---- a tile, hashed as it is computed -------------------------------------------------------------------------------------

// Σ over 4 bytes of u8(a)·s8(b), plus c.
HD int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
#ifdef __CUDA_ARCH__
  int32_t d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#else
  for (int i = 0; i < 4; ++i) c += (int32_t)((a >> (8 * i)) & 0xFFu) * (int32_t)(int8_t)(b >> (8 * i));
  return c;
#endif
}

// One tile: X strip rows ``xs`` and Y strip columns ``ys`` (16 × k3 bytes each), ``run(j)`` the running sum of output
// j = 16·r + c, starting at c₀. Every 16-deep step adds X·Y's step to each running sum and hands row r's 16 words to the
// leaf as one block, step-major then row-major; at the end ``run`` holds Z. With HASH false, the same arithmetic unhashed.
template <bool HASH, class Run>
HD void tile_fused(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  WordLeaf wl;
  if (HASH) wl_init(wl);
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    for (int r = 0; r < 16; ++r) {
      uint32_t xv[4], w[16];
      load16(xs + (size_t)r * k3 + 16 * t, xv);
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        int32_t acc = dot4(xv[0], yv[c][0], 0);
        acc = dot4(xv[1], yv[c][1], acc);
        acc = dot4(xv[2], yv[c][2], acc);
        acc = dot4(xv[3], yv[c][3], acc);
        uint32_t v = run(16 * r + c) + (uint32_t)acc;
        run(16 * r + c) = v;
        w[c] = v;
      }
      if (HASH) wl_block(wl, w);
    }
  }
  if (HASH) wl_final(wl, leaf);
}

// The low-byte leaf variant (cheap-binding's NCP-INT row 4; its γ is unproved: timed only, not committed, never a result): TurboSHAKE128 with domain byte 0x0B over
// the low byte of every checkpoint in the word leaf's order, 8 rows (128 bytes, 16 lanes) per permutation, each block's other
// 5 lanes of the rate zero.
constexpr uint8_t LOWBYTE_DS = 0x0B;

template <class Run>
HD void tile_fused_lowbyte(const uint8_t* xs, const uint8_t* ys, int k3, int steps, Run run, uint8_t leaf[32]) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  for (int t = 0; t < steps; ++t) {
    uint32_t yv[16][4];
#pragma unroll
    for (int c = 0; c < 16; ++c) load16(ys + (size_t)c * k3 + 16 * t, yv[c]);
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int r = 8 * half + rr;
        uint32_t xv[4];
        load16(xs + (size_t)r * k3 + 16 * t, xv);
        uint64_t lo = 0, hi = 0;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          int32_t acc = dot4(xv[0], yv[c][0], 0);
          acc = dot4(xv[1], yv[c][1], acc);
          acc = dot4(xv[2], yv[c][2], acc);
          acc = dot4(xv[3], yv[c][3], acc);
          uint32_t v = run(16 * r + c) + (uint32_t)acc;
          run(16 * r + c) = v;
          if (c < 8) lo |= (uint64_t)(v & 0xFFu) << (8 * c);
          else hi |= (uint64_t)(v & 0xFFu) << (8 * (c - 8));
        }
        s[2 * rr] ^= lo;
        s[2 * rr + 1] ^= hi;
      }
      if (t == steps - 1 && half == 1) {
        s[16] ^= LOWBYTE_DS;
        s[20] ^= 0x80ull << 56;
      }
      keccak_p<12>(s);
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) leaf[i] = (uint8_t)(s[i >> 3] >> (8 * (i & 7)));
}

}  // namespace ncp2

#ifdef __CUDACC__

namespace ncp2 {

struct SmemRun {
  uint32_t* base;
  __device__ __forceinline__ uint32_t& operator()(int j) const { return base[j * 32]; }
};

// LEAF 0: no leaf (the no-hash control); 1: the word leaf (committed); 2: the low-byte variant (timed only; γ unproved).
template <int LEAF>
__global__ void __launch_bounds__(32) kf_tiles(int ns, size_t tiles, int steps, int k3, int n, const uint8_t* x,
                                               const uint8_t* y, const uint32_t* c0, uint8_t* tile_leaves, uint32_t* z) {
  extern __shared__ uint32_t sm[];
  const int lane = threadIdx.x;
  const size_t tile = (size_t)blockIdx.x * 32 + lane;
  if (tile >= tiles) return;
  const int rs = (int)(tile / ns), cs = (int)(tile % ns);
  SmemRun run{sm + lane};
  for (int j = 0; j < D * D; ++j) run(j) = c0[cs * D + (j & 15)];
  uint8_t leaf[32];
  if (LEAF == 2) tile_fused_lowbyte(x + (size_t)rs * D * k3, y + (size_t)cs * D * k3, k3, steps, run, leaf);
  else tile_fused<LEAF == 1>(x + (size_t)rs * D * k3, y + (size_t)cs * D * k3, k3, steps, run, leaf);
  if (LEAF)
    for (int i = 0; i < 32; ++i) tile_leaves[32 * tile + i] = leaf[i];
  for (int j = 0; j < D * D; ++j) z[(size_t)(rs * D + (j >> 4)) * n + cs * D + (j & 15)] = run(j);
}

// The tile kernel's attributes, once per process.
static cudaError_t tiles_setup() {
  cudaError_t e = cudaFuncSetAttribute(kf_tiles<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(kf_tiles<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(kf_tiles<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * D * D * 4);
  return e;
}

// Every tile of a call on ``stream``: its leaf into ``tile_leaves`` (``leaf`` 1 the word leaf, 2 the low-byte variant, 0 none)
// and its Z into ``z`` (m × n, row-major).
static void launch_tiles(int leaf, int ns, size_t tiles, int steps, int k3, int n, const uint8_t* x, const uint8_t* y,
                         const uint32_t* c0, uint8_t* tile_leaves, uint32_t* z, cudaStream_t stream) {
  const size_t smem = 32 * D * D * 4;
  const unsigned grid = (unsigned)((tiles + 31) / 32);
  if (leaf == 2) kf_tiles<2><<<grid, 32, smem, stream>>>(ns, tiles, steps, k3, n, x, y, c0, tile_leaves, z);
  else if (leaf == 1) kf_tiles<1><<<grid, 32, smem, stream>>>(ns, tiles, steps, k3, n, x, y, c0, tile_leaves, z);
  else kf_tiles<0><<<grid, 32, smem, stream>>>(ns, tiles, steps, k3, n, x, y, c0, nullptr, z);
}

}  // namespace ncp2

#endif
