// The honest step the probes time beside other work (side.cu, power.cu): 8 independent int8 IMMA (m16n8k32) on drawn
// operands, then each accumulator's 4 words folded by duo (receipt_folds.cuh's rf_duo_step, 3 scalar instructions a word
// as ptxas lowers it) into that accumulator's own (y, z), so the fold binds every output word of every MMA.
#pragma once
#include <stdint.h>

#include "../../receipts/receipt_folds.cuh"

__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// m16n8k16: half of mma_imma's K, the same 4 words a lane (the honest MMA at L = 16 on sm_89)
__device__ __forceinline__ void mma_imma16(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

struct Honest {
  uint32_t x[4], y[2], ys[8], zs[8], a0, a1, k0, k1;
  int32_t d[8][4];
  // zero: all-zero operands, the lightest tensor load (every accumulator stays 0)
  __device__ explicit Honest(int mode, bool zero = false) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) x[i] = zero ? 0u : rf_mix32(t * 8u + i + 1u);
    y[0] = zero ? 0u : rf_mix32(t * 8u + 5u);
    y[1] = zero ? 0u : rf_mix32(t * 8u + 6u);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      ys[c] = rf_served_y0(0x5eedu, t * 8u + c);
      zs[c] = rf_served_z0(ys[c]);
#pragma unroll
      for (int w = 0; w < 4; w++) d[c][w] = 0;
    }
    a0 = rf_served_ka(mode);
    a1 = rf_served_ka(mode + 1);
    k0 = rf_served_kk(mode);
    k1 = rf_served_kk(mode + 1);
  }
  // NM of the 8 MMAs (8: honest), and the fold of all 8 accumulators' words or none; K16: m16n8k16 MMAs
  template <int NM = 8, bool FOLD = true, bool K16 = false>
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int c = 0; c < NM; c++) {
      if constexpr (K16) mma_imma16(d[c], x, y);
      else mma_imma(d[c], x, y);
    }
#pragma unroll
    for (int c = 0; c < (FOLD ? 8 : 0); c++)
      rf_duo_step(ys[c], zs[c], (uint32_t)d[c][0], (uint32_t)d[c][1], (uint32_t)d[c][2], (uint32_t)d[c][3], a0, a1,
                  k0, k1);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= rf_duo_final(ys[c], zs[c]);
    return s;
  }
};
