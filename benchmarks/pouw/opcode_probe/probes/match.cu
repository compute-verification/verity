// MATCH's rates on sm_120, for the coverage table's MATCH rows (machine_model/data/cover/rows.toml, CMP-166). Joint
// points: rows.jsonl only, no census line. Each warp runs 8 independent chains v = match(v); the mask a lane gets is the
// same for every lane of its class and differs between classes, so the lanes' partition into classes holds every trip.
//   match_any_d32, _q32, _u32  MATCH.ANY on 32-bit words, the lanes in 32 classes (all distinct), 4, or 1 (uniform)
//   match_any_d64, _u64        MATCH.ANY.U64, 32 classes or 1
//   mix_u32_lop3_Q, mix_u64_lop3_Q   every warp 8 MATCH.ANY (.U64) on uniform values, its fastest case, and Q LOP3 on
//                              chains of their own, spread after the MATCHes: other warps' LOP3 issue while one waits
// MATCH.ALL isn't here: its row is dominated at its issue slot whatever its rate, and the gate reads its leading
// predicate operand as its destination. L simulates a lane's MATCH.ANY result in 8 units (v and 5 words, then v, the
// mask and 4 words at a time over the other 31; .U64 twice that), 256 a warp instruction (512), and LOP3 in 32, so the
// row's hypothesis is that (256 r_MATCH + 32 r_LOP3) / 128 ≤ 1 at every point (512 for .U64).
#include "../probe.cuh"

constexpr int TPB = 1024;

template <int CLASSES>
__device__ __forceinline__ unsigned start(const OpArgs& a, int u) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned c = CLASSES == 32 ? lane : CLASSES == 4 ? lane >> 3 : 0;
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

template <bool WIDE, int Q>
__device__ __forceinline__ void mix(const OpArgs& a) {
  unsigned long long v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = start<1>(a, u);
  uint32_t w[8];
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
#pragma unroll
  for (int i = 0; i < 8; i++) w[i] = threadIdx.x * 8u + i + 1u;
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
      for (int q = u * Q / 8; q < (u + 1) * Q / 8; q++)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[q & 7]) : "r"(k1), "r"(k2));
    }
  }
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= (uint32_t)v[u] ^ w[u];
  OP_SINK(s)
}

OP_KERNEL(match_any_d32, TPB) { any32<32>(a); }
OP_KERNEL(match_any_q32, TPB) { any32<4>(a); }
OP_KERNEL(match_any_u32, TPB) { any32<1>(a); }
OP_KERNEL(match_any_d64, TPB) { any64<32>(a); }
OP_KERNEL(match_any_u64, TPB) { any64<1>(a); }
#define MIX(n, W, Q) OP_KERNEL(mix_##n##_lop3_##Q, TPB) { mix<W, Q>(a); }
MIX(u32, false, 8) MIX(u32, false, 16) MIX(u32, false, 32) MIX(u32, false, 64) MIX(u32, false, 128)
MIX(u64, true, 16) MIX(u64, true, 64) MIX(u64, true, 128)
#define P_MIX(n, OP, Q) \
  {.name = "mix_" #n "_lop3_" #Q, .kernel = (const void*)mix_##n##_lop3_##Q, .tpb = TPB, .expect = OP "=8,LOP3.LUT=" #Q}

static const OpProbe PROBES[] = {
    {.name = "match_any_d32", .kernel = (const void*)match_any_d32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_q32", .kernel = (const void*)match_any_q32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_u32", .kernel = (const void*)match_any_u32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_d64", .kernel = (const void*)match_any_d64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
    {.name = "match_any_u64", .kernel = (const void*)match_any_u64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
    P_MIX(u32, "MATCH.ANY", 8), P_MIX(u32, "MATCH.ANY", 16), P_MIX(u32, "MATCH.ANY", 32),
    P_MIX(u32, "MATCH.ANY", 64), P_MIX(u32, "MATCH.ANY", 128),
    P_MIX(u64, "MATCH.ANY.U64", 16), P_MIX(u64, "MATCH.ANY.U64", 64), P_MIX(u64, "MATCH.ANY.U64", 128),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
