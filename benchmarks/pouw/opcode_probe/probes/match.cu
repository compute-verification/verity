// MATCH's rates on sm_120 and sm_90, for the coverage table's MATCH rows (machine_model/data/cover/rows.toml,
// CMP-166). Joint points: rows.jsonl only, no census line. Each warp runs 8 independent chains v = match(v); the mask a
// lane gets is the same for every lane of its class and differs between classes, so the lanes' partition into classes
// holds every trip.
//   match_any_d32, _q32, _c2, _u32   MATCH.ANY on 32-bit words, the lanes in 32 classes (all distinct), 4, 2, or 1
//   match_any_d64, _u64              MATCH.ANY.U64, 32 classes or 1
//   mix_u32_lop3_Q                   every warp 8 MATCH.ANY on uniform values, its fastest case, and Q LOP3 on chains of
//                                    their own, spread after the MATCHes: other warps' LOP3 issue while one waits
//   mix_u32_lop3_Q_ffma_F            the same with F FFMA too, on the FMA pipe beside LOP3's
//   mix_c2_lop3_32_ffma_32           2 classes, a mask that says something
//   mix_u64_lop3_Q[_ffma_F]          MATCH.ANY.U64
// MATCH.ALL isn't here: its row is dominated at its issue slot whatever its rate, and the gate reads its leading
// predicate operand as its destination. L simulates MATCH.ANY in 119 units a warp instruction: the lanes in 11 groups of
// at most 3, one unit for each of the 55 pairs of groups writes the 15 equalities of its 6 words, and each lane's 2 units
// assemble its mask from its 10 (238 for .U64, the high and low words apart). LOP3 and FFMA are 32 units, so the row is
// dominated if (119 r_MATCH + 32 (r_LOP3 + r_FFMA)) / 128 ≤ 1 at every point (238 for .U64).
#include "../probe.cuh"

constexpr int TPB = 1024;

template <int CLASSES>
__device__ __forceinline__ unsigned start(const OpArgs& a, int u) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned c = CLASSES == 32 ? lane : CLASSES == 4 ? lane >> 3 : CLASSES == 2 ? lane >> 4 : 0;
  return (unsigned)a.iters * 977u + c * 8u + (unsigned)u;
}

template <int CLASSES>
__device__ __forceinline__ void any32(const OpArgs& a) {
  unsigned v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = start<CLASSES>(a, u);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) v[u] = __match_any_sync(~0u, v[u]);
  }
  OP_END
  OP_SINK(v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7])
}

template <int CLASSES>
__device__ __forceinline__ void any64(const OpArgs& a) {
  unsigned long long v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = ((unsigned long long)start<CLASSES>(a, u) << 32) | start<CLASSES>(a, u + 8);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      const unsigned m = __match_any_sync(~0u, v[u]);
      v[u] = ((unsigned long long)m << 32) | m;
    }
  }
  OP_END
  OP_SINK((unsigned)(v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7]))
}

template <bool WIDE, int CLASSES, int Q, int F>
__device__ __forceinline__ void mix(const OpArgs& a) {
  unsigned long long v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = start<CLASSES>(a, u);
  uint32_t w[8];
  float f[8];
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
  const float c1 = 1.0000001f + (float)a.mode, c2 = 0.5f + (float)a.mode;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    w[i] = threadIdx.x * 8u + i + 1u;
    f[i] = (float)(threadIdx.x + i);
  }
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      if constexpr (WIDE) {
        const unsigned m = __match_any_sync(~0u, v[u]);
        v[u] = ((unsigned long long)m << 32) | m;
      } else {
        v[u] = __match_any_sync(~0u, (unsigned)v[u]);
      }
#pragma unroll
      for (int j = 0; j < Q / 8; j++)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[(u * Q / 8 + j) & 7]) : "r"(k1), "r"(k2));
#pragma unroll
      for (int j = 0; j < F / 8; j++)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[(u * F / 8 + j) & 7]) : "f"(c1), "f"(c2));
    }
  }
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= (uint32_t)v[u] ^ w[u] ^ __float_as_uint(f[u]);
  OP_SINK(s)
}

OP_KERNEL(match_any_d32, TPB) { any32<32>(a); }
OP_KERNEL(match_any_q32, TPB) { any32<4>(a); }
OP_KERNEL(match_any_c2, TPB) { any32<2>(a); }
OP_KERNEL(match_any_u32, TPB) { any32<1>(a); }
OP_KERNEL(match_any_d64, TPB) { any64<32>(a); }
OP_KERNEL(match_any_u64, TPB) { any64<1>(a); }
#define MIX(n, W, C, Q) OP_KERNEL(mix_##n##_lop3_##Q, TPB) { mix<W, C, Q, 0>(a); }
#define MIXF(n, W, C, Q, F) OP_KERNEL(mix_##n##_lop3_##Q##_ffma_##F, TPB) { mix<W, C, Q, F>(a); }
MIX(u32, false, 1, 8) MIX(u32, false, 1, 16) MIX(u32, false, 1, 32) MIX(u32, false, 1, 64) MIX(u32, false, 1, 128)
MIXF(u32, false, 1, 16, 16) MIXF(u32, false, 1, 32, 32) MIXF(u32, false, 1, 48, 48) MIXF(u32, false, 1, 64, 64)
MIXF(c2, false, 2, 32, 32)
MIX(u64, true, 1, 16) MIX(u64, true, 1, 64) MIX(u64, true, 1, 128) MIXF(u64, true, 1, 32, 32)
#define P_MIX(n, OP, Q) \
  {.name = "mix_" #n "_lop3_" #Q, .kernel = (const void*)mix_##n##_lop3_##Q, .tpb = TPB, .expect = OP "=8,LOP3.LUT=" #Q}
#define P_MIXF(n, OP, Q, F)                                                                                 \
  {.name = "mix_" #n "_lop3_" #Q "_ffma_" #F, .kernel = (const void*)mix_##n##_lop3_##Q##_ffma_##F, .tpb = TPB, \
   .expect = OP "=8,LOP3.LUT=" #Q ",FFMA=" #F}

static const OpProbe PROBES[] = {
    {.name = "match_any_d32", .kernel = (const void*)match_any_d32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_q32", .kernel = (const void*)match_any_q32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_c2", .kernel = (const void*)match_any_c2, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_u32", .kernel = (const void*)match_any_u32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_d64", .kernel = (const void*)match_any_d64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
    {.name = "match_any_u64", .kernel = (const void*)match_any_u64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
    P_MIX(u32, "MATCH.ANY", 8), P_MIX(u32, "MATCH.ANY", 16), P_MIX(u32, "MATCH.ANY", 32),
    P_MIX(u32, "MATCH.ANY", 64), P_MIX(u32, "MATCH.ANY", 128),
    P_MIXF(u32, "MATCH.ANY", 16, 16), P_MIXF(u32, "MATCH.ANY", 32, 32), P_MIXF(u32, "MATCH.ANY", 48, 48),
    P_MIXF(u32, "MATCH.ANY", 64, 64), P_MIXF(c2, "MATCH.ANY", 32, 32),
    P_MIX(u64, "MATCH.ANY.U64", 16), P_MIX(u64, "MATCH.ANY.U64", 64), P_MIX(u64, "MATCH.ANY.U64", 128),
    P_MIXF(u64, "MATCH.ANY.U64", 32, 32),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
