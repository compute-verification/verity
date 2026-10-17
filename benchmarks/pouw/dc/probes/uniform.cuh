// EPS1 DC r4: the uniform datapath on sm_90a and sm_100a, built as opcode_probe's uniform.cu and uniform_up.cu are
// (FACET-BREAK, note:20261017T0524Z-draft-eps1-facet-break): 16 warps (4 a SMSP, one block a SM), K uniform instructions
// an iteration on eight warp-uniform chains seeded from a parameter and carried around the loop, and every vector setting
// inside the honest step (8 IMMA m16n8k32, duo on each accumulator), the one place ptxas 13.0 keeps a chain on the uniform
// datapath. sm90_uniform.cu and sm100_uniform.cu pin each loop's counts exactly, so a chain that lands on lanes refuses.
//   u<F>_<K><S>  F, the fixed stream: z none (alone: the honest MMAs unread, which ptxas deletes after placing the chain);
//                h the honest step (beside duo's fold); g h + 64 FFMA reading the accumulators (issue binds); l h + 32
//                LOP3 reading them (the ALU pipe binds harder); m 32 IMMA and one fold (mma.sync binds the tensor pipe
//                that tcgen05 and wgmma share; without a fold reading the accumulators ptxas puts the chains on lanes);
//                b h + 256 FFMA reading the accumulators (issue saturated: what a uniform instruction costs issue)
//                S, the uniform step on each chain pair: i UIMAD + ULOP3; x ULOP3; a UIADD3 + ULOP3; f USHF + ULOP3;
//                p UPRMT + ULOP3; s USHF/ULEA + ULOP3
#pragma once
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"

constexpr int UTPB = 512, UITERS = 2048;

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

enum { F_Z, F_H, F_G, F_L, F_M, F_B };

template <int NF, int NL>
struct UFeed {
  float f[8], g;
  uint32_t v[8], k;
  __device__ explicit UFeed(int mode) {
    g = __uint_as_float(0x3F800001u ^ (uint32_t)mode);
    k = 0x3C013C01u ^ (uint32_t)mode;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      f[c] = 0.f;
      v[c] = (uint32_t)c;
    }
  }
  __device__ __forceinline__ void step(const Honest& h) {
#pragma unroll
    for (int q = 0; q < NF; q++) f[q & 7] = __fmaf_rn(__int_as_float(h.d[q & 7][(q >> 3) & 3]), g, f[q & 7]);
#pragma unroll
    for (int q = 0; q < NL; q++) v[q & 7] = v[q & 7] ^ (uint32_t)h.d[q & 7][(q >> 3) & 3] ^ (k + (uint32_t)q);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= __float_as_uint(f[c]) ^ v[c];
    return s;
  }
};

template <int F, int S, int K>
__device__ __forceinline__ void uni(const OpArgs& a) {
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Honest h(a.mode);
  UFeed<F == F_G ? 64 : F == F_B ? 256 : 0, F == F_L ? 32 : 0> e(a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    if constexpr (F == F_Z) h.step<8, false>();
    if constexpr (F == F_M) {
      h.step<8, false>();
      h.step<8, false>();
      h.step<8, false>();
    }
    if constexpr (F != F_Z) h.step();
    e.step(h);
    u.step();
  }
  OP_END
  OP_SINK(u.fold() ^ h.fold() ^ e.fold())
}

#define UK(nm, F, S, K) \
  OP_KERNEL(nm, UTPB) { uni<F, S, K>(a); }
#define UFORMS(F, N, K)        \
  UK(u##N##_##K##i, F, S_I, K) \
  UK(u##N##_##K##x, F, S_X, K) \
  UK(u##N##_##K##a, F, S_A, K) \
  UK(u##N##_##K##f, F, S_F, K) \
  UK(u##N##_##K##p, F, S_P, K)
UFORMS(F_Z, z, 64)
UFORMS(F_Z, z, 128)
UK(uz_64s, F_Z, S_S, 64)
UK(uh_0, F_H, S_I, 0)
UFORMS(F_H, h, 32)
UFORMS(F_H, h, 64)
UFORMS(F_H, h, 128)
UK(uh_64s, F_H, S_S, 64)
UK(ug_0, F_G, S_I, 0)
UFORMS(F_G, g, 64)
UFORMS(F_G, g, 128)
UK(ul_0, F_L, S_I, 0)
UFORMS(F_L, l, 64)
UFORMS(F_L, l, 128)
UK(um_0, F_M, S_I, 0)
UFORMS(F_M, m, 32)
UFORMS(F_M, m, 64)
UK(ub_0, F_B, S_I, 0)
UFORMS(F_B, b, 64)
UK(ub_128i, F_B, S_I, 128)
UK(ub_128x, F_B, S_X, 128)

#define P_U(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = UTPB, .expect = EX, .counted = CNT, .iters = UITERS}
