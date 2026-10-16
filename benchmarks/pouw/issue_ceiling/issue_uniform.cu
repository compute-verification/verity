// Uniform-datapath probe (compute-accounting, note:20261016T0439Z-draft-issue-uniform).
//
// VF2 (RS-MODEL §13) prices a uniform-datapath instruction at 0.594 SMSP-cycles, assuming it takes the SM's dispatch
// slot. If a uniform instruction co-issues on a separate port (takes NO vector slot), 0.594 exceeds real cost and VF2 is
// unsound. Round 3 saw %clock reads route half to CS2UR (uniform) and the mix beat JI2's single-slot budget, which is
// consistent with co-issue but was never a saturated test.
//
// Here we saturate the dispatch slot with one vector kind at ~1 instr a SMSP-cycle (FFMA, FP32 full-rate; IADD3 as a
// secondary), then add a live chain of uniform UIADD3 at several densities and ask whether the vector rate falls:
//   - uniform takes a full slot  => cycles grow by 1 a uniform, vector rate ~ 1/(1+density), slots a uniform ~ 1.
//   - uniform takes no slot       => cycles unchanged, vector rate stays ~1, slots a uniform ~ 0  => VF2 unsound.
//
// The uniform accumulators u0,u1 are seeded from blockIdx (warp-uniform) and updated only with uniform values, so ptxas
// keeps them in uniform registers and emits U-ops; the mutual recurrence is non-foldable and kept live into the sink.
// issue_uniform.py confirms the FFMA/IADD3 (vector) and UIADD3 (uniform) counts from the SASS of the one timed loop.
#include "common.cuh"

#define NACC 16  // independent vector accumulators: ILP to cover FFMA/IADD3 latency and hold the dispatch at 1/cyc

enum { VK_FFMA, VK_IADD3 };
static const char* VK_NAME[] = {"ffma", "iadd3"};
static const char* VK_SASS[] = {"FFMA", "IADD3"};

// One timed loop: NACC vector ops (saturator) + D uniform UIADD3 (the chain under test), per outer iteration.
template <int VK, int D>
__global__ void __launch_bounds__(512) k_vec(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t tid = threadIdx.x;
  float facc[NACC];
  uint32_t iacc[NACC];
#pragma unroll
  for (int c = 0; c < NACC; c++) {
    facc[c] = (float)((tid * 7u + c * 13u) & 255u) * 0.00390625f + 1.0f;   // lane-dependent => vector registers
    iacc[c] = tid * 0x9e3779b9u + c * 0x85ebca6bu + 1u;                    // lane-dependent => vector registers
  }
  // uniform accumulators: blockIdx-seeded, updated only with uniform values => uniform datapath (U-ops)
  uint32_t u0 = blockIdx.x * 2654435761u + 12345u;
  uint32_t u1 = blockIdx.x * 40503u + 6789u;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < NACC; c++) {
      if (VK == VK_FFMA) facc[c] = fmaf(facc[c], 1.0000001f, 0.5f);       // FFMA, independent across c
      else               iacc[c] = iacc[c] + iacc[c ^ 1] + 0x1357u;       // IADD3 (3-input), coupled pairs, non-foldable
    }
#pragma unroll
    for (int d = 0; d < D; d++) {
      // mutual recurrence of two uniform accumulators: each step is one UIADD3 (u = u + u' + const), non-foldable
      if (d & 1) u1 = u1 + u0 + (uint32_t)(0x9e37u + d);
      else       u0 = u0 + u1 + (uint32_t)(0x85ebu + d);
    }
  }
  CLK_END(rec)
  float fs = 0;
  uint32_t is = 0;
#pragma unroll
  for (int c = 0; c < NACC; c++) {
    fs += facc[c];
    is ^= iacc[c];
  }
  sink[blockIdx.x * blockDim.x + tid] = (VK == VK_FFMA ? __float_as_uint(fs) : is) ^ u0 ^ u1;
}

// ---- host ----------------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 16;  // warps a SM (4 a SMSP): saturate the dispatch slot

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int VK, int D>
static void vec_case() {
  auto kern = k_vec<VK, D>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  // per-SMSP window counts: 4 warps a SMSP
  double warps_smsp = WSM / 4.0;
  double n_vec = warps_smsp * NACC * iters;   // vector instrs a SMSP over the window
  double n_uni = warps_smsp * D * iters;      // uniform instrs a SMSP over the window
  printf("{\"probe\":\"uniform\",\"vk\":\"%s\",\"density\":%d,\"nacc\":%d,\"warps\":%d,\"iters\":%d,"
         "\"cyc\":%.0f,\"mhz\":%.1f,\"vec_per_smsp_per_cyc\":%.4f,\"uni_per_smsp_per_cyc\":%.4f,"
         "\"n_vec_smsp\":%.0f,\"n_uni_smsp\":%.0f}\n",
         VK_NAME[VK], D, NACC, WSM, iters, t.cyc_med, t.mhz,
         n_vec / t.cyc_med, n_uni / t.cyc_med, n_vec, n_uni);
  fflush(stdout);
}

template <int VK>
static void sweep() {
  vec_case<VK, 0>();
  vec_case<VK, 4>();
  vec_case<VK, 8>();
  vec_case<VK, 16>();
  vec_case<VK, 32>();
  vec_case<VK, 64>();
}

int main(int, char**) {
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 512 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  sweep<VK_FFMA>();
  sweep<VK_IADD3>();
  printf("{\"probe\":\"uniform\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
