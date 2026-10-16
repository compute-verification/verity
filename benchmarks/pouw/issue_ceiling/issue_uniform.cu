// Uniform-datapath probe (compute-accounting, note:20261016T0439Z-draft-issue-uniform).
//
// VF2 (RS-MODEL §13) prices a uniform-datapath instruction at 0.594 SMSP-cycles, assuming it takes the SM's dispatch
// slot. If a uniform instruction co-issues on a separate port (takes NO vector slot), 0.594 exceeds real cost and VF2 is
// unsound. Round 3 saw %clock reads route half to CS2UR (uniform) and beat JI2's single-slot budget — consistent with
// co-issue, but never a saturated test.
//
// Saturate the dispatch with FFMA (FP32 full-rate) at ~1 instr a SMSP-cycle, then add a live chain of UIADD3 at
// densities D (uniform a SMSP-iter). The discriminating regime is LOW density (D < NACC, the uniform pipe not yet
// saturated, the vector pipe is the bottleneck):
//   - uniform shares the dispatch slot  => the FFMA rate falls at once, slots a uniform ~ 1 even at low D.
//   - uniform co-issues (own port)      => the FFMA rate is unchanged at low D (slots ~ 0); only once D exceeds NACC does
//                                          the uniform pipe saturate and cycles grow — so HIGH density can't tell the two
//                                          apart, LOW density can.
//
// To put the chain on the uniform datapath, u0,u1 are seeded from kernel PARAMETERS (s0,s1): a parameter loads via LDCU
// into a uniform register, so a recurrence over it stays uniform and ptxas emits UIADD3 — exactly as the loop counter
// `it` (from the `iters` parameter) already does. (blockIdx is read by S2R into a vector register, so seeding from it
// keeps the chain vector — the trap the first version fell into.) The chain is updated only with uniform values and
// consumed by one lane-0 store, so it stays live and uniform.
// issue_uniform.py confirms the FFMA (vector) and UIADD3 (uniform) counts from the SASS and flags if promotion failed.
#include "common.cuh"

#define NACC 32  // independent vector accumulators: ILP to hold the FFMA dispatch at ~1/cyc

enum { VK_FFMA, VK_IADD3 };
static const char* VK_NAME[] = {"ffma", "iadd3"};
static const char* VK_SASS[] = {"FFMA", "IADD3"};

template <int VK, int D>
__global__ void __launch_bounds__(1024) k_vec(int iters, uint32_t s0, uint32_t s1, uint32_t* vsink, uint32_t* usink, BlockClk* rec) {
  const uint32_t tid = threadIdx.x;
  uint32_t acc[NACC];  // one array; FFMA path reinterprets, so 1024 threads fit without spilling
#pragma unroll
  for (int c = 0; c < NACC; c++) acc[c] = tid * 0x9e3779b9u + c * 0x85ebca6bu + 1u;  // lane-dependent => vector regs
  // uniform accumulators: parameter-seeded (LDCU => uniform reg), uniform-only updates, uniform consumer => UIADD3
  uint32_t u0 = s0, u1 = s1;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < NACC; c++) {
      if (VK == VK_FFMA) {
        float f = __uint_as_float(acc[c]);
        f = fmaf(f, 0.99902344f, 0.5f);           // real FFMA (m != 1, converges, no FADD rewrite, no inf)
        acc[c] = __float_as_uint(f);
      } else {
        acc[c] = acc[c] + acc[c ^ 1] + 0x1357u;   // IADD3 (3-input), coupled pairs, non-foldable
      }
    }
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (d & 1) u1 = u1 + u0 + (uint32_t)(0x9e37u + d);  // one UIADD3 (uniform, non-foldable mutual recurrence)
      else       u0 = u0 + u1 + (uint32_t)(0x85ebu + d);
    }
  }
  CLK_END(rec)
  uint32_t vs = 0;
#pragma unroll
  for (int c = 0; c < NACC; c++) vs ^= acc[c];
  vsink[blockIdx.x * blockDim.x + tid] = vs;
  // u0,u1's ONLY live-out is this warp-uniform predicate (UISETP). With no vector consumer, ptxas keeps the whole
  // recurrence in uniform registers (UIADD3) instead of UMOV-ing it out to feed a store — the trap v2 fell into.
  if (u0 + u1 == 0x1234567u) usink[blockIdx.x] = tid;
}

// ---- host ----------------------------------------------------------------------------------------------------------
static uint32_t* g_vsink;
static uint32_t* g_usink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 32;  // warps a SM (8 a SMSP): saturate the dispatch slot

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int VK, int D>
static void vec_case() {
  auto kern = k_vec<VK, D>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  // s0,s1 are runtime args (ptxas sees them as parameters => LDCU into uniform regs, non-foldable)
  const uint32_t s0 = 0x1234u + (uint32_t)g_sms, s1 = 0x89abu;
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, s0, s1, g_vsink, g_usink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double warps_smsp = WSM / 4.0;
  double n_vec = warps_smsp * NACC * iters;  // vector instrs a SMSP over the window
  double n_uni = warps_smsp * D * iters;     // uniform instrs a SMSP over the window
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
  CK(cudaMalloc(&g_vsink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_usink, (size_t)g_sms * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  sweep<VK_FFMA>();
  sweep<VK_IADD3>();
  printf("{\"probe\":\"uniform\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
