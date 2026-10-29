// The honest wgmma against the cheapest word programs MH100w allows, in one binary, so probe.py times them interleaved
// rep by rep under one gate. The free values (the data's partial dots D', which MH100w makes free) stream from shared
// memory by LDS.128 (a move), 4 words a lane a load:
//   wa_s8_n256  honest: wgmma m64n256k32 s8, 2 warpgroups, 4 a commit group (CORE-H32's wg_alone), 4,096 words a warp
//               instruction
//   adv_free    free_d: word = D' + g + P_t, g = G[p_i, j] (public) and P_t the row-step's coin prefix (one broadcast
//               LDS a step, a move), one IADD3 a word
//   adv_chain   chain: word_t = word_{t-1} + d_t + a_t, a_t the row-step's coin bit (one broadcast LDS a step), one IADD3
//               a word
// Each adversary word is folded into the sink (two LOP3 a 4 words, the stand-in for the hash's read). 32 warps a block,
// 8 loads an iteration, each with its own 16-B rows (lane l: row base + l + 32 slot, the base moving 256 rows a warp and
// an iteration round a ring of 2,048), so ptxas can't merge two and no access conflicts.
#include "../dc/probes/wg_common.cuh"

OP_KERNEL(wa_s8_n256, 256) { wg_alone<F_s8_n256, 1>(a); }

constexpr int ATPB = 1024, U = 8, RING = 2048, STEPS = 64;

__host__ __device__ inline unsigned race_off(int slot, int lane, int warp, int it) {
  return (((unsigned)lane + 32u * ((unsigned)slot + U * (unsigned)(warp + it))) & (RING - 1)) * 16u;
}
// 8 LDS.128, then 8 loads of the row-step's coin bit or prefix (one word every lane reads: a broadcast) past the ring
__host__ __device__ inline unsigned step_off(int slot, int lane, int warp, int it) {
  return slot < U ? race_off(slot, lane, warp, it) : RING * 16u + 4u * ((unsigned)(it * U + slot - U) & (STEPS - 1));
}

template <bool CHAIN>
__device__ __forceinline__ void race_adv(const OpArgs& a) {
  __shared__ __align__(128) unsigned char smem[RING * 16];
  __shared__ unsigned step[STEPS];  // chain: the row-step's coin bit a_t; free: its prefix P_t
  for (int i = threadIdx.x; i < RING * 4; i += blockDim.x)
    reinterpret_cast<unsigned*>(smem)[i] = i * 2654435761u + blockIdx.x + 1;
  if (threadIdx.x == 0)
    for (unsigned t = 0, s = (unsigned)a.mode; t < STEPS; t++) {
      const unsigned at = dc_mix32(t + 77u * blockIdx.x) & 1u;
      s += at;
      step[t] = CHAIN ? at : s;
    }
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const unsigned g0 = dc_mix32(4u * threadIdx.x) & 63u, g1 = dc_mix32(4u * threadIdx.x + 1u) & 63u,
                 g2 = dc_mix32(4u * threadIdx.x + 2u) & 63u, g3 = dc_mix32(4u * threadIdx.x + 3u) & 63u;
  unsigned w0 = 0, w1 = 0, w2 = 0, w3 = 0, acc = threadIdx.x + 1;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned d0, d1, d2, d3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                   : "r"(sbase + race_off(u, lane, w, it)));
      const unsigned s = *(volatile unsigned*)&step[(it * U + u) & (STEPS - 1)];
      if constexpr (CHAIN) {
        w0 += d0 + s, w1 += d1 + s, w2 += d2 + s, w3 += d3 + s;
        acc ^= w0 ^ w1 ^ w2 ^ w3;
      } else {
        acc ^= (d0 + g0 + s) ^ (d1 + g1 + s) ^ (d2 + g2 + s) ^ (d3 + g3 + s);
      }
    }
  }
  OP_END
  OP_SINK(acc)
}

OP_KERNEL(adv_free, ATPB) { race_adv<false>(a); }
OP_KERNEL(adv_chain, ATPB) { race_adv<true>(a); }

static const OpProbe PROBES[] = {
    {.name = "wa_s8_n256", .kernel = (const void*)wa_s8_n256, .tpb = 256, .expect = "IGMMA.64x256x32.S8.S8*=4",
     .iters = 4096, .note = "words a warp instruction: 4096"},
    {.name = "adv_free", .kernel = (const void*)adv_free, .tpb = ATPB, .expect = "LDS.128=8,IADD3=32",
     .smem_ops = "LDS.128*8,LDS*8", .smem = step_off, .note = "words a warp instruction of LDS.128: 128"},
    {.name = "adv_chain", .kernel = (const void*)adv_chain, .tpb = ATPB, .expect = "LDS.128=8,IADD3=32",
     .smem_ops = "LDS.128*8,LDS*8", .smem = step_off, .note = "words a warp instruction of LDS.128: 128"},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
