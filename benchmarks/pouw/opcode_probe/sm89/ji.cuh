// facet_break's joint-issue schedule on sm_89: facet_break/ji.cuh
// jmix at placement 0, over the L40S's two int8 forms. Every warp runs 8U MMAs an iteration on NA accumulators (MMA c
// on c mod NA) and Q lane ops a group of 8 MMAs (Q a multiple of 8), Q/8 after each MMA, each on one of 8 chains
// (facet_break/mix.cuh's Lanes, the same PTX). On sm_120 FMUL by an immediate, 112 a 8 MMAs, on 2-3 accumulators at 2
// warps a SMSP beat every joint-issue row; `ji.py` generates the grid around it and pins every loop opcode.
#pragma once
#include "../probe.cuh"
#include "../probes/honest.cuh"

enum { L_FMULI, L_FADDI, L_FFMAI, L_FFMA, L_IADD3, L_AXI, L_LOP3, L_IMAD, L_IMADI };

struct JLanes {
  uint32_t v[8], k1, k2;
  float f[8], kf1, kf2;
  __device__ explicit JLanes(const OpArgs& a) {
    k1 = 0x3C013C01u ^ (uint32_t)a.mode;
    k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
    kf1 = 1.0000001f + (float)a.mode;
    kf2 = 1e-7f * (float)(a.mode + 1);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      v[i] = threadIdx.x * 8u + i + 1u;
      f[i] = 1.0f + 1e-3f * (float)(threadIdx.x * 8u + i);
    }
  }
  template <int L>
  __device__ __forceinline__ void op(int q) {
    const int s = q & 7;
    if constexpr (L == L_FMULI) asm volatile("mul.rn.f32 %0, %0, 0f3F800010;" : "+f"(f[s]));
    if constexpr (L == L_FADDI) asm volatile("add.rn.f32 %0, %0, 0f33800000;" : "+f"(f[s]));
    if constexpr (L == L_FFMAI) asm volatile("fma.rn.f32 %0, %0, 0f3F800010, %1;" : "+f"(f[s]) : "f"(kf2));
    if constexpr (L == L_FFMA) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[s]) : "f"(kf1), "f"(kf2));
    // a chain's own operand twice in a row would merge into one 3-input IADD3: each add reads the next chain
    if constexpr (L == L_IADD3) asm volatile("add.u32 %0, %0, %1;" : "+r"(v[s]) : "r"(v[(s + 1) & 7]));
    if constexpr (L == L_AXI) {
      if ((q >> 3) & 1) asm volatile("xor.b32 %0, %0, 0x5A5A5A5A;" : "+r"(v[s]));
      else asm volatile("add.u32 %0, %0, 0x9E3779B9;" : "+r"(v[s]));
    }
    if constexpr (L == L_LOP3) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_IMAD) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v[s]) : "r"(k1), "r"(k2));
    if constexpr (L == L_IMADI) asm volatile("mad.lo.u32 %0, %0, 0x10035, %1;" : "+r"(v[s]) : "r"(k2));
  }
  template <int L>
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (L <= L_FFMA) s ^= __float_as_uint(f[i]);
      else s ^= v[i];
    }
    return s;
  }
};

template <int K, int NA>
struct JMma {
  uint32_t a[4], b[2];
  int32_t d[NA][4];
  __device__ explicit JMma(uint32_t lane) {
    const uint32_t w = 0x01FF7F80u;
    a[0] = w ^ (lane << 8); a[1] = w; a[2] = w ^ 0x00010000u; a[3] = w;
    b[0] = w; b[1] = w ^ (lane << 16);
#pragma unroll
    for (int c = 0; c < NA; c++)
#pragma unroll
      for (int j = 0; j < 4; j++) d[c][j] = 0;
  }
  __device__ __forceinline__ void go(int c) {
    if constexpr (K == 16) mma_imma16(d[c % NA], a, b);
    else mma_imma(d[c % NA], a, b);
  }
  __device__ uint32_t fold() const {
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < NA; c++)
#pragma unroll
      for (int j = 0; j < 4; j++) s ^= (uint32_t)d[c][j];
    return s;
  }
};

template <int K, int NA, int U, int L, int Q>
__device__ __forceinline__ void jm(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  JMma<K, NA> m(lane);
  JLanes o(a);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int g = 0; g < U; g++)
#pragma unroll
      for (int c = 0; c < 8; c++) {
        m.go(g * 8 + c);
        // a constant trip count, so the innermost loop unrolls before the ones around it
#pragma unroll
        for (int j = 0; j < Q / 8; j++) o.template op<L>(g * Q + c * (Q / 8) + j);
      }
  }
  OP_END
  OP_SINK((lane ^ m.fold() ^ o.template fold<L>()))
}

#define JM(name, K, NA, U, L, Q, TPB) \
  OP_KERNEL(name, TPB) { jm<K, NA, U, L, Q>(a); }
