// Round-5 64-bit-destination probe (compute-accounting, note:20261016T0640Z-draft-issue-wide).
//
// VF3c (RS-MODEL §15; gpu-model/rs-price-final-vf3c.py) charges every 64-bit-destination instruction TWO words of floor
// credit, on the claim that it might do two 32-bit word-ops of attack work. IADD.64 sets the price; if it is really one
// dispatch slot and cannot do two independent word-ops, the two-word credit inflates every vector kind's price. This
// probe measures, for each kind, (1) how ptxas lowers it (one 64-bit instruction or a pair IADD3 + IADD3.X), (2) its
// standalone dispatch throughput per SMSP-cycle, (3) the IMMA-s8 MAC displacement it causes at 1/4/8/16 ops a MMA
// (unrolled k_mixu and warp-specialized k_spec), and (4) bit-exactly on the GPU whether one 64-bit add performs two
// independent 32-bit adds (carry coupling; guard-bit SWAR for two 31-bit lanes). issue_wide.py reads the SASS.
#include "common.cuh"

// ---- IMMA s8 (round-1/2's) --------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static const double IMMA_MACS = 4096;  // 16x8x32 MACs a MMA

// ---- the kinds --------------------------------------------------------------------------------------------------
// 64-bit destination kinds: iadd_64, mov_64, cs2r_64 (clock64 pair read), imad_wide (32x32+64->64).
// 32-bit contrasts: iadd3_32, imad_32. Every op is asm volatile so ptxas cannot drop it; inputs chain through the
// rotating accumulators so nothing hoists out of the loop.
enum { X_IADD3, X_IMAD, X_IADD64, X_MOV64, X_CS2R64, X_IMADWIDE, X_N };
static const char* X_NAME[] = {"iadd3_32", "imad_32", "iadd_64", "mov_64", "cs2r_64", "imad_wide"};

// Each kind is a self-recurrence on accumulator i with a fixed register addend (c32/c64): 8 independent chains a warp x 8
// warps a SMSP = 64-wide ILP, so standalone the SMSP is dispatch-bound (not latency-bound). asm volatile keeps every op
// and stops ptxas folding the repeated adds. mov_64 keeps a cross-copy (it is elidable either way).
template <int X>
__device__ __forceinline__ void oone(uint32_t* r, uint64_t* w, int i, uint32_t c32, uint64_t c64) {
  if (X == X_IADD3)        asm volatile("add.u32 %0,%0,%1;" : "+r"(r[i]) : "r"(c32));
  else if (X == X_IMAD)    asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[i]) : "r"(c32), "r"(c32));
  else if (X == X_IADD64)  asm volatile("add.u64 %0,%0,%1;" : "+l"(w[i]) : "l"(c64));
  else if (X == X_MOV64)   asm volatile("mov.b64 %0,%1;" : "=l"(w[i]) : "l"(w[(i + 1) & 7]));
  else if (X == X_CS2R64)  asm volatile("mov.u64 %0,%%clock64;" : "=l"(w[i]));
  else {  // IMAD.WIDE: low word of the changing accumulator, so the 32x32->64 multiply can't be hoisted
    uint32_t a = (uint32_t)w[i];
    asm volatile("mad.wide.u32 %0,%1,%2,%0;" : "+l"(w[i]) : "r"(a), "r"(c32));
  }
}

__device__ __forceinline__ void init_r(uint32_t* r, uint64_t* w, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
    w[i] = ((uint64_t)r[i] << 32) ^ (0x1234567u + i) | 1u;
  }
}
__device__ __forceinline__ void init_mma(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x08080808u ^ lane; a[1] = 0x01010101u; a[2] = 0x08010801u; a[3] = 0x01080108u;
  b[0] = 0x08080808u; b[1] = 0x01010101u ^ lane;
}
__device__ __forceinline__ uint32_t consume(const uint32_t* r, const uint64_t* w) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x ^= r[i] ^ (uint32_t)w[i] ^ (uint32_t)(w[i] >> 32);
  return x;
}

// ---- (2) standalone throughput: 8 warps a SMSP, 64 ops an iter (ILP 8), no MMA ---------------------------------
template <int X>
__global__ void __launch_bounds__(1024) k_alone(int iters, uint32_t* sink, BlockClk* rec) {
  uint32_t r[8]; uint64_t w[8]; init_r(r, w, threadIdx.x);
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < 64; q++) oone<X>(r, w, q & 7, c32, c64);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = consume(r, w);
}

// ---- (3) MMA-alone baseline (2 MMA warps a SMSP, 8 accumulators) ------------------------------------------------
static const int WSM = 16;  // 4 warps a SMSP

__global__ void __launch_bounds__(512) k_specmma(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(lane, a, b);
  int32_t d[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) mma_imma(d[c], a, b);
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (uint32_t)d[c][0];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- warp-specialized: 2 MMA warps + 2 lane warps a SMSP; lane warps do R ops an iter (ratio R/8) ---------------
template <int X, int R>
__global__ void __launch_bounds__(512) k_spec(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);  // first half of warps
  uint32_t a[4], b[2]; init_mma(lane, a, b);
  int32_t d[8][4] = {};
  uint32_t r[8]; uint64_t w[8]; init_r(r, w, threadIdx.x);
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 8; c++) mma_imma(d[c], a, b);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) oone<X>(r, w, q & 7, c32, c64);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) {
#pragma unroll
    for (int c = 0; c < 8; c++) x ^= (uint32_t)d[c][0];
  } else x ^= consume(r, w);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- unrolled single-warp interleave: per MMA, OPM ops --------------------------------------------------------
template <int X, int OPM>
__global__ void __launch_bounds__(512) k_mixu(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(lane, a, b);
  int32_t d[8][4] = {};
  uint32_t r[8]; uint64_t w[8]; init_r(r, w, threadIdx.x);
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma_imma(d[c], a, b);
#pragma unroll
      for (int q = 0; q < OPM; q++) oone<X>(r, w, (c + q) & 7, c32, c64);
    }
  }
  CLK_END(rec)
  uint32_t x = lane ^ consume(r, w);
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (uint32_t)d[c][0];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- (4) bit-exact two-word test: can one 64-bit add do two independent 32-bit adds? ---------------------------
__global__ void k_twoword(const uint64_t* in, uint64_t* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint64_t s;
  // carry coupling: 0x00000000FFFFFFFF + 1 -> 0x0000000100000000 iff the low word's carry crosses into the high word
  asm volatile("add.u64 %0,%1,%2;" : "=l"(s) : "l"(in[0]), "l"(in[1])); out[0] = s;
  // naive packed 2x32 add, low lane engineered to overflow: high lane is corrupted by the crossing carry
  asm volatile("add.u64 %0,%1,%2;" : "=l"(s) : "l"(in[2]), "l"(in[3])); out[1] = s;
  // guard-bit SWAR: two 31-bit lanes (bit 31 and bit 63 held 0), one add -> two independent 31-bit sums
  asm volatile("add.u64 %0,%1,%2;" : "=l"(s) : "l"(in[4]), "l"(in[5])); out[2] = s;
}

// ---- host -------------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM_ALONE = 32;  // 8 warps a SMSP for standalone saturation

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int X>
static void alone_case() {
  auto kern = k_alone<X>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM_ALONE * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double ops = (double)WSM_ALONE * 64 * iters;  // ops a SM
  printf("{\"probe\":\"wide\",\"mode\":\"alone\",\"op\":\"%s\",\"warps\":%d,\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"ops_per_smsp_per_cyc\":%.4f}\n",
         X_NAME[X], WSM_ALONE, iters, t.cyc_med, t.mhz, ops / t.cyc_med / 4.0);
  fflush(stdout);
}

static double specmma_alone() {
  auto kern = k_specmma;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  int warps = WSM / 2;  // 2 MMA warps a SMSP
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double mmas = (double)warps * iters * 8;
  double macs = mmas * IMMA_MACS / t.cyc_med;
  printf("{\"probe\":\"wide\",\"mode\":\"specmma_alone\",\"mma_warps\":%d,\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"mma_macs_per_sm_per_clk\":%.2f}\n", warps, iters, t.cyc_med, t.mhz, macs);
  fflush(stdout);
  return macs;
}

template <int X, int R>
static void spec_case(double macs_alone) {
  auto kern = k_spec<X, R>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  int mma_warps = WSM / 2, alu_warps = WSM / 2;
  double mmas = (double)mma_warps * iters * 8;
  double ops = (double)alu_warps * iters * R;
  double macs = mmas * IMMA_MACS / t.cyc_med;
  printf("{\"probe\":\"wide\",\"mode\":\"spec\",\"op\":\"%s\",\"ops_per_mma\":%.3f,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"op_calls_per_sm_per_clk\":%.4f,\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         X_NAME[X], (double)R / 8.0, iters, t.cyc_med, t.mhz, ops / t.cyc_med, macs, macs / macs_alone);
  fflush(stdout);
}

template <int X, int OPM>
static void mixu_case(double macs_alone) {
  auto kern = k_mixu<X, OPM>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double mmas = (double)WSM * iters * 8;
  double ops = (double)WSM * iters * 8 * OPM;
  double macs = mmas * IMMA_MACS / t.cyc_med;
  printf("{\"probe\":\"wide\",\"mode\":\"mixu\",\"op\":\"%s\",\"ops_per_mma\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"op_calls_per_sm_per_clk\":%.4f,\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         X_NAME[X], OPM, iters, t.cyc_med, t.mhz, ops / t.cyc_med, macs, macs / macs_alone);
  fflush(stdout);
}

template <int X>
static void sweep(double a) {
  spec_case<X, 8>(a); spec_case<X, 32>(a); spec_case<X, 64>(a); spec_case<X, 128>(a);  // 1/4/8/16 a MMA
  mixu_case<X, 1>(a); mixu_case<X, 4>(a); mixu_case<X, 8>(a); mixu_case<X, 16>(a);
}

static void twoword() {
  uint64_t *d_in, *d_out, h_in[6], h_out[3] = {};
  // carry test
  h_in[0] = 0x00000000FFFFFFFFull; h_in[1] = 0x0000000000000001ull;
  // naive packed add, low lane overflows: (hi=7:lo=0xFFFFFFFF)+(hi=2:lo=5)
  h_in[2] = (7ull << 32) | 0xFFFFFFFFull; h_in[3] = (2ull << 32) | 5ull;
  // guard-bit SWAR: lanes are 31-bit (bits 31 and 63 are 0). lane0 0x7FFFFFFF+1 stays in 31 bits; lane1 0x10+0x20
  h_in[4] = ((uint64_t)0x00000010u << 32) | 0x7FFFFFFFu; h_in[5] = ((uint64_t)0x00000020u << 32) | 0x00000001u;
  CK(cudaMalloc(&d_in, sizeof h_in)); CK(cudaMalloc(&d_out, sizeof h_out));
  CK(cudaMemcpy(d_in, h_in, sizeof h_in, cudaMemcpyHostToDevice));
  k_twoword<<<1, 32>>>(d_in, d_out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));
  // verdicts
  bool carry_couples = (h_out[0] == 0x0000000100000000ull);
  uint32_t lo1 = (uint32_t)h_out[1], hi1 = (uint32_t)(h_out[1] >> 32);
  bool naive_corrupts = (hi1 != (7u + 2u));                    // high lane != 9 => low carry leaked across
  uint32_t lo2 = (uint32_t)h_out[2], hi2 = (uint32_t)(h_out[2] >> 32);
  bool swar_independent = (lo2 == 0x80000000u) && (hi2 == 0x00000030u);  // 0x7FFFFFFF+1, 0x10+0x20, no cross-carry
  printf("{\"probe\":\"wide\",\"mode\":\"twoword\",\"add64_carry\":\"0x%016llx\",\"carry_couples\":%s,"
         "\"packed_naive\":\"0x%016llx\",\"naive_corrupts\":%s,"
         "\"packed_swar\":\"0x%016llx\",\"swar_two_independent_31bit\":%s}\n",
         (unsigned long long)h_out[0], carry_couples ? "true" : "false",
         (unsigned long long)h_out[1], naive_corrupts ? "true" : "false",
         (unsigned long long)h_out[2], swar_independent ? "true" : "false");
  fflush(stdout);
  cudaFree(d_in); cudaFree(d_out);
}

int main(int, char**) {
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));

  alone_case<X_IADD3>(); alone_case<X_IMAD>(); alone_case<X_IADD64>();
  alone_case<X_MOV64>(); alone_case<X_CS2R64>(); alone_case<X_IMADWIDE>();

  double a = specmma_alone();
  sweep<X_IADD3>(a); sweep<X_IMAD>(a); sweep<X_IADD64>(a);
  sweep<X_MOV64>(a); sweep<X_CS2R64>(a); sweep<X_IMADWIDE>(a);

  twoword();
  printf("{\"probe\":\"wide\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
