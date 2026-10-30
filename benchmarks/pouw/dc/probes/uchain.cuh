// The warp-uniform chains the uniform-datapath probes carry: eight pairs of registers folded in place, so ptxas keeps
// them on the uniform datapath (UIMAD/ULOP3/… ) wherever it keeps a chain uniform, and never hoists or folds them to a
// constant. uniform.cuh carries them beside the synchronous honest step (mma.sync); async_uni (sm90/sm100) carries the
// same chains beside a saturated asynchronous tensor op (wgmma, tcgen05.mma), the facet_break the model has not measured.
#pragma once
#include <cstdint>

enum { S_I, S_X, S_A, S_F, S_P, S_S };

template <int K, int S>
struct UChains {
  uint32_t u[8];
  __device__ explicit UChains(uint32_t seed) {
#pragma unroll
    for (int c = 0; c < 8; c++) u[c] = seed * (2u * c + 3u) + (uint32_t)c;
  }
  __device__ __forceinline__ void step() {
#pragma unroll
    for (int i = 0; i < K / 8; i++)
#pragma unroll
      for (int p = 0; p < 8; p += 2) {
        const uint32_t k = 0x7F4A7C15u + (uint32_t)(i * 8 + p) * 0x9E3779B9u;
        if constexpr (S == S_I) {
          u[p] = u[p] * 0x9E3779B1u + u[p + 1];
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        }
        if constexpr (S == S_X) {
          u[p] = (u[p] & u[p + 1]) ^ k;
          u[p + 1] = (u[p + 1] | u[p]) ^ (k >> 3);
        }
        if constexpr (S == S_A) {
          u[p] = u[p] + u[p + 1] + k;
          u[p + 1] = u[p + 1] ^ u[p] ^ (k >> 3);
        }
        if constexpr (S == S_F) {
          u[p] = __funnelshift_l(u[p], u[p + 1], 7 + (i + p) % 19);
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        }
        if constexpr (S == S_P) {
          u[p] = __byte_perm(u[p], u[p + 1], 0x6152u ^ (uint32_t)(((i + p) & 3) << 4));
          u[p + 1] = u[p + 1] ^ u[p] ^ k;
        }
        if constexpr (S == S_S) {
          u[p] = (u[p] >> 3) + u[p + 1];
          u[p + 1] = u[p + 1] ^ (u[p] << 5) ^ k;
        }
      }
  }
  __device__ uint32_t fold() const { return u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]; }
};
