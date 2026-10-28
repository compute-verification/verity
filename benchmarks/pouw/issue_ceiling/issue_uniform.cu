// Uniform-datapath probe.
//
// VF2 prices a uniform-datapath instruction at 0.594 SMSP-cycles, assuming it takes the SM's dispatch
// slot. If a uniform instruction co-issues on a separate port (takes NO vector slot), 0.594 exceeds real cost and VF2 is
// unsound. The question is whether a uniform instruction and a vector instruction can ISSUE in the same cycle.
//
// Warp-specialized test on one SMSP: some warps run a pure vector saturator (FFMA/IADD3), other warps run pure
// uniform-datapath work (a runtime-trip inner loop whose induction is UIADD3/UISETP and each clock() read is CS2UR).
// Run each alone and then both together on the same SMSPs:
//   - one shared dispatch port  => both together take ~ the sum of the issue the two need (the vector rate FALLS when
//                                  uniform warps are added), so a uniform instruction consumes a dispatch slot.
//   - a separate uniform port   => both together take ~ max(vector time, uniform time) (the vector rate is UNCHANGED),
//                                  so a uniform instruction takes no vector slot and VF2's 0.594/op over-prices.
// A warp's role is its slot within the SMSP (warp = tid/32; smsp = warp&3; slot = warp>>2): slot < nf is a vector warp,
// the next nu are uniform warps, the rest idle. issue_uniform.py reads the uniform ops a inner-iteration from the SASS.
#include "common.cuh"

#define NACC 32  // independent vector accumulators: ILP to hold the vector dispatch at ~1/cyc

enum { VK_FFMA, VK_IADD3 };
static const char* VK_NAME[] = {"ffma", "iadd3"};
static const char* VK_SASS[] = {"FFMA", "IADD3"};

template <int VK>
__global__ void __launch_bounds__(1024) k_spec(int iters, uint32_t ucount, uint32_t s0, int nf, int nu,
                                               uint32_t* vsink, uint32_t* usink, BlockClk* rec) {
  const uint32_t tid = threadIdx.x;
  const int slot = (int)(tid >> 5) >> 2;        // warp-within-SMSP index 0..7 (4 SMSPs a SM)
  const bool do_vec = slot < nf;
  const bool do_uni = slot >= nf && slot < nf + nu;
  uint32_t acc[NACC];
#pragma unroll
  for (int c = 0; c < NACC; c++) acc[c] = tid * 0x9e3779b9u + c * 0x85ebca6bu + 1u;
  uint32_t uacc = s0;  // uniform accumulator; each inner step reads %clock (CS2UR, a uniform-datapath op ptxas can't drop)
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
    if (do_vec) {
#pragma unroll
      for (int c = 0; c < NACC; c++) {
        if (VK == VK_FFMA) {
          float f = __uint_as_float(acc[c]);
          f = fmaf(f, 0.99902344f, 0.5f);         // real FFMA (m != 1, converges, no FADD rewrite)
          acc[c] = __float_as_uint(f);
        } else {
          acc[c] = acc[c] + acc[c ^ 1] + 0x1357u; // IADD3
        }
      }
    }
    if (do_uni) {
      // runtime-trip inner loop; induction (j++/j<ucount) is UIADD3/UISETP and each clock() read is CS2UR (uniform
      // datapath, non-eliminable). This is the purest stream of uniform-datapath ops ptxas will emit.
#pragma unroll 1
      for (uint32_t j = 0; j < ucount; j++) {
        uacc = uacc + (uint32_t)clock() + j;
      }
    }
  }
  CLK_END(rec)
  if (do_vec) {
    uint32_t vs = 0;
#pragma unroll
    for (int c = 0; c < NACC; c++) vs ^= acc[c];
    vsink[blockIdx.x * blockDim.x + tid] = vs;
  }
  if (do_uni && uacc == 0x1234567u) usink[blockIdx.x] = tid;
}

// ---- host ----------------------------------------------------------------------------------------------------------
static uint32_t* g_vsink;
static uint32_t* g_usink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 32;  // warps a SM (8 a SMSP)

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int VK>
static void spec_case(uint32_t ucount, int nf, int nu) {
  auto kern = k_spec<VK>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  const uint32_t s0 = 0x1234u + (uint32_t)g_sms;  // runtime arg => LDCU into a uniform reg, non-foldable
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, ucount, s0, nf, nu, g_vsink, g_usink, g_rec); }, g_sms, g_rec);
  };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double n_vec = (double)nf * NACC * iters;      // vector instrs a SMSP (nf vector warps a SMSP)
  double n_inner = (double)nu * ucount * iters;  // inner-loop iterations a SMSP (× U-ops-a-iter from SASS = uniform instrs)
  printf("{\"probe\":\"spec\",\"vk\":\"%s\",\"ucount\":%u,\"nf\":%d,\"nu\":%d,\"nacc\":%d,\"iters\":%d,"
         "\"cyc\":%.0f,\"mhz\":%.1f,\"vec_per_smsp_per_cyc\":%.4f,\"inner_per_smsp_per_cyc\":%.4f,"
         "\"n_vec_smsp\":%.0f,\"n_inner_smsp\":%.0f}\n",
         VK_NAME[VK], ucount, nf, nu, NACC, iters, t.cyc_med, t.mhz,
         n_vec / t.cyc_med, n_inner / t.cyc_med, n_vec, n_inner);
  fflush(stdout);
}

template <int VK>
static void sweep() {
  // fixed uniform load (ucount), compare vector-alone / uniform-alone / both on the same SMSPs
  for (uint32_t uc : {16u, 32u}) {
    spec_case<VK>(uc, 4, 0);  // vector alone (4 warps a SMSP)
    spec_case<VK>(uc, 0, 4);  // uniform alone (4 warps a SMSP)
    spec_case<VK>(uc, 4, 4);  // both together on the same SMSPs
  }
}

int main(int, char**) {
  g_sms = num_sms();
  CK(cudaMalloc(&g_vsink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_usink, (size_t)g_sms * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  sweep<VK_FFMA>();
  sweep<VK_IADD3>();
  printf("{\"probe\":\"spec\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
