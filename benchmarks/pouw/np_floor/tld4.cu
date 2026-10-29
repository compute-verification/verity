// The texture fetch as a word writer on sm_90a: TLD of a 16-byte texel (tex1Dfetch<uint4>, 4 registers a lane) beside
// the 4-byte one (1 register), each over a 32 KB window the texture cache holds (`_c`) and over 64 MB (`_g`), on
// probe.cuh. 32 warps an SM, 8 fetches an iteration, each with its own texel run (lane l reads base + l), so ptxas can't
// merge two and the unit keeps 256 fetches in flight. Joint points: rows.jsonl only.
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024, U = 8;

// texel index of slot u, lane, warp and iteration: a 32 KB window (2,048 uint4 or 8,192 u32 texels) or a 64 MB walk
template <bool WIDE, bool CACHED>
__device__ __forceinline__ unsigned tld_idx(int u, unsigned lane, unsigned w, unsigned it) {
  if constexpr (CACHED) return ((w * U + u) * 32u + lane + (it & 7u) * 1024u) & (WIDE ? 2047u : 8191u);
  const unsigned gt = (blockIdx.x * (TPB / 32) + w) * 32u + lane, total = gridDim.x * TPB;
  return (gt + (it * U + u) * total) & ((64u << 20) / (WIDE ? 16u : 4u) - 1u);
}

template <bool WIDE, bool CACHED>
__device__ __forceinline__ void tld_probe(const OpArgs& a) {
  const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = (int)tld_idx<WIDE, CACHED>(u, lane, w, (unsigned)it);
      if constexpr (WIDE) {
        const uint4 v = tex1Dfetch<uint4>(a.tex4, i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      } else {
        acc ^= tex1Dfetch<unsigned>(a.tex1, i);
      }
    }
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(tld4_c, TPB) { tld_probe<true, true>(a); }
OP_KERNEL(tld4_g, TPB) { tld_probe<true, false>(a); }
OP_KERNEL(tld1_c, TPB) { tld_probe<false, true>(a); }
OP_KERNEL(tld1_g, TPB) { tld_probe<false, false>(a); }

static const OpProbe PROBES[] = {
    {.name = "tld4_c", .kernel = (const void*)tld4_c, .tpb = TPB, .expect = "TLD*=8", .words = 128},
    {.name = "tld4_g", .kernel = (const void*)tld4_g, .tpb = TPB, .expect = "TLD*=8", .words = 128, .iters = 512},
    {.name = "tld1_c", .kernel = (const void*)tld1_c, .tpb = TPB, .expect = "TLD*=8", .words = 32},
    {.name = "tld1_g", .kernel = (const void*)tld1_g, .tpb = TPB, .expect = "TLD*=8", .words = 32, .iters = 512},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
