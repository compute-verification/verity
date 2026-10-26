// MATCH's rates on sm_120, for the coverage table's MATCH rows (machine_model/data/cover/rows.toml, CMP-166). Joint
// points: rows.jsonl only, no census line. Each warp runs 8 independent chains v = match(v); the mask a lane gets is the
// same for every lane of its class and differs between classes, so the lanes' partition into classes holds every trip.
//   match_any_d32, _q32, _u32  MATCH.ANY on 32-bit words, the lanes in 32 classes (all distinct), 4, or 1 (uniform)
//   match_any_d64, _u64        MATCH.ANY.U64, 32 classes or 1
// MATCH.ALL isn't here: its row is dominated at its issue slot whatever its rate, and the gate reads its leading
// predicate operand as its destination. The row's hypothesis: MATCH.ANY issues at most 128/224 = 0.57 warp instructions a SM-clock (.U64 0.29), which an L
// tree of 7 units a lane dominates; faster than that at any partition, and the row stays a hole at its measured rate.
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

OP_KERNEL(match_any_d32, TPB) { any32<32>(a); }
OP_KERNEL(match_any_q32, TPB) { any32<4>(a); }
OP_KERNEL(match_any_u32, TPB) { any32<1>(a); }
OP_KERNEL(match_any_d64, TPB) { any64<32>(a); }
OP_KERNEL(match_any_u64, TPB) { any64<1>(a); }

static const OpProbe PROBES[] = {
    {.name = "match_any_d32", .kernel = (const void*)match_any_d32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_q32", .kernel = (const void*)match_any_q32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_u32", .kernel = (const void*)match_any_u32, .tpb = TPB, .expect = "MATCH.ANY=8"},
    {.name = "match_any_d64", .kernel = (const void*)match_any_d64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
    {.name = "match_any_u64", .kernel = (const void*)match_any_u64, .tpb = TPB, .expect = "MATCH.ANY.U64=8"},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
