// Uniform-datapath probe (compute-accounting, note:20261016T0439Z-draft-issue-uniform).
//
// VF2 (RS-MODEL §13) prices a uniform-datapath instruction at 0.594 SMSP-cycles, assuming it takes the SM's dispatch
// slot. If a uniform instruction co-issues on a separate port (takes NO vector slot), 0.594 exceeds real cost and VF2 is
// unsound. Round 3 saw %clock reads route half to CS2UR (uniform) and beat JI2's single-slot budget — consistent with
// co-issue, but never a saturated test.
//
// Saturate the dispatch with FFMA (FP32 full-rate) at ~1 instr a SMSP-cycle, then add uniform-datapath work and watch
// the FFMA rate. The discriminating regime is LOW density (the uniform pipe not yet saturated, the vector pipe the
// bottleneck): if uniform shares the dispatch slot the FFMA rate falls at once; if it co-issues on its own port the FFMA
// rate is unchanged until the uniform pipe itself saturates.
//
// How to get uniform instructions ptxas won't demote to the vector datapath: a data recurrence over uniform values is
// NOT enough — ptxas only puts the uniform datapath to work for loop control (an induction variable `it++` is UIADD3,
// `it<n` is UISETP) and for param/special reads (LDCU, CS2UR). So the uniform work here is a runtime-trip INNER LOOP:
// its trip count `ucount` is a kernel parameter (ptxas can't unroll or close-form it), and its body is a uniform
// recurrence anchored to the induction variable with an xorshift so it can't be eliminated. The whole inner loop lands
// on the uniform datapath; issue_uniform.py reads the actual U-ops a inner-iteration out of the SASS.
#include "common.cuh"

#define NACC 32  // independent vector accumulators: ILP to hold the FFMA dispatch at ~1/cyc

enum { VK_FFMA, VK_IADD3 };
static const char* VK_NAME[] = {"ffma", "iadd3"};
static const char* VK_SASS[] = {"FFMA", "IADD3"};

template <int VK>
__global__ void __launch_bounds__(1024) k_vec(int iters, uint32_t ucount, uint32_t s0, uint32_t* vsink, uint32_t* usink, BlockClk* rec) {
  const uint32_t tid = threadIdx.x;
  uint32_t acc[NACC];  // one array; FFMA path reinterprets, so 1024 threads fit without spilling
#pragma unroll
  for (int c = 0; c < NACC; c++) acc[c] = tid * 0x9e3779b9u + c * 0x85ebca6bu + 1u;  // lane-dependent => vector regs
  // four independent keep-alive accumulators: ILP=4 (× warps) => the inner loop is dispatch-bound, not latency-bound
  uint32_t v0 = s0, v1 = s0 ^ 0x5a5a5a5au, v2 = s0 + 0x9e3779b9u, v3 = s0 ^ 0xa5a5a5a5u;
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
    // uniform-datapath work: a runtime-trip inner loop (ucount a parameter => a real loop ptxas can't unroll or
    // close-form). Its induction `j++` and bound compare land on the uniform datapath (UIADD3 + UISETP) — the only
    // uniform instructions ptxas reliably emits; a data recurrence over uniform values gets demoted to vector. v0,v1
    // are two independent geometric recurrences (v=2v+j, non-foldable => the loop survives) that give the inner loop
    // ILP so it is dispatch-bound, not latency-bound: then cycles a inner-iteration == instructions a inner-iteration,
    // and we can see whether the two uniform ops are in that count (took a slot) or not (co-issued for free).
#pragma unroll 1
    for (uint32_t j = 0; j < ucount; j++) {
      v0 = v0 + v0 + j;
      v1 = v1 + v1 + j;
      v2 = v2 + v2 + j;
      v3 = v3 + v3 + j;
    }
  }
  CLK_END(rec)
  uint32_t vs = 0;
#pragma unroll
  for (int c = 0; c < NACC; c++) vs ^= acc[c];
  vsink[blockIdx.x * blockDim.x + tid] = vs;
  if ((v0 ^ v1 ^ v2 ^ v3) == 0x1234567u) usink[blockIdx.x] = tid;
}

// ---- host ----------------------------------------------------------------------------------------------------------
static uint32_t* g_vsink;
static uint32_t* g_usink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 32;  // warps a SM (8 a SMSP): saturate the dispatch slot

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int VK>
static void vec_case(uint32_t ucount) {
  auto kern = k_vec<VK>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  const uint32_t s0 = 0x1234u + (uint32_t)g_sms;  // runtime arg => LDCU into a uniform reg, non-foldable
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, ucount, s0, g_vsink, g_usink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double warps_smsp = WSM / 4.0;
  double n_vec = warps_smsp * NACC * iters;       // vector instrs a SMSP over the window
  double n_inner = warps_smsp * ucount * iters;   // inner-loop iterations a SMSP (× U-ops-a-iter from SASS = uniform instrs)
  printf("{\"probe\":\"uniform\",\"vk\":\"%s\",\"density\":%u,\"nacc\":%d,\"warps\":%d,\"iters\":%d,"
         "\"cyc\":%.0f,\"mhz\":%.1f,\"vec_per_smsp_per_cyc\":%.4f,\"inner_per_smsp_per_cyc\":%.4f,"
         "\"n_vec_smsp\":%.0f,\"n_inner_smsp\":%.0f}\n",
         VK_NAME[VK], ucount, NACC, WSM, iters, t.cyc_med, t.mhz,
         n_vec / t.cyc_med, n_inner / t.cyc_med, n_vec, n_inner);
  fflush(stdout);
}

template <int VK>
static void sweep() {
  for (uint32_t u : {0u, 4u, 8u, 16u, 32u, 64u}) vec_case<VK>(u);
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
