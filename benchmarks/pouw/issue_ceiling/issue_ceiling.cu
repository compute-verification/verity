// Mixed-stream issue ceiling (compute-accounting Q1, note lanes/compute-accounting *-draft-issue-ceiling.md).
//
// RS-MODEL §7's clean priced floor pays every single-word opcode through one "issue row" worth ≈53% of the weight,
// and that row rests on a hardware fact: one warp instruction per SM sub-partition per cycle. The census measured it
// only on single-pipe streams (note *-draft-census-sm120 §3, FFMA/LOP3/IMAD/… beside QMMA). The referee
// (note *-draft-theorem-e-referee II.2) flags it as the model's largest soundness dependency: if a stream that mixes
// a tensor MMA with a CUDA-core opcode ever issued more than one warp instruction per SMSP-cycle, every such opcode
// would fall below 16 u a lane-word and the floor would stop being clean.
//
// This probe interleaves a tensor MMA (IMMA s8 m16n8k32, or OMMA nvf4 e2m1 m16n8k64) with each single-pipe opcode the
// model prices through issue — IADD3, LOP3, PRMT, SHFL, IMAD, IDP.4A (dp4a), HFMA2, LDS.128 and a warp-uniform BRA —
// at several ratios, and reports, per SM clock: total warp instructions issued, the MMA count, the opcode count, and
// the MMA's MAC rate while the issue slot is shared. One block a SM, W warps, each warp carries 8 independent MMA
// accumulators and 8 independent register chains (latency hidden); every block records its own clock64 span.
//
//   k_mma_<imma|omma>   8 MMAs an iteration, alone        (the MMA-only rate, and the SASS the MMA compiles to)
//   k_op_<op>           8 ops an iteration, alone         (the opcode-only rate, and the SASS the op compiles to)
//   k_mix<M,X,R>        8 MMAs + R ops an iteration        (both; R/8 ops a MMA)
//
// census.py-style: issue_ceiling.py reads each loop's SASS (cuobjdump) to confirm the opcode that issued and the
// per-call instruction count behind every "warp instr/SM/clk" figure.
#include "common.cuh"

#include <cstring>

// ---- the two tensor MMAs ----------------------------------------------------------------------------------------
// IMMA: s8 m16n8k32, s32 accumulate (IMMA.16832). a = 4×b32 (16 int8 a lane), b = 2×b32, d = 4×s32. 4096 MACs.
__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// OMMA: nvf4 e2m1 m16n8k64, block-scaled (OMMA.SF.16864), exactly the census probe's. 8192 MACs.
__device__ __forceinline__ void mma_omma(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x38383838u), "r"(0x38383838u));
}

enum { M_IMMA, M_OMMA };
static const char* M_NAME[] = {"imma_s8_16832", "omma_nvf4_16864"};
static const double M_MACS[] = {4096, 8192};

// ---- the single-pipe opcodes ------------------------------------------------------------------------------------
// Each takes three register operands (x updated in place) so an 8-wide rotation keeps the pipe fed without a
// cross-op dependency; __shared__ base and a warp-uniform runtime flag feed LDS.128 and the BRA.
#define F(x) __uint_as_float(x)

__device__ __forceinline__ void x_iadd3(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("iadd3 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_lop3(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_prmt(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_shfl(uint32_t& x, uint32_t y, uint32_t, uint32_t*, uint32_t) {
  asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 0x1f, 0xffffffff;" : "+r"(x) : "r"(y));
}
__device__ __forceinline__ void x_imad(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_idp4a(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_hfma2(uint32_t& x, uint32_t y, uint32_t z, uint32_t*, uint32_t) {
  asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
// LDS.128: load four 32-bit words from shared memory; x folds them so the load isn't dead.
__device__ __forceinline__ void x_lds128(uint32_t& x, uint32_t y, uint32_t, uint32_t* smem, uint32_t) {
  uint32_t a, b, c, d;
  uint32_t off = (x ^ y) & 0x1FC;  // 0..508 words, 16B-aligned index into smem
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem + off);
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(addr));
  x = a ^ b ^ c ^ d;
}
// BRA: a warp-uniform conditional branch. `uni` is a runtime flag (passed non-zero) the compiler can't fold, and it
// is loop-invariant and warp-uniform, so ptxas hoists the setp and leaves a uniform @P BRA over one add an iteration.
__device__ __forceinline__ void x_bra(uint32_t& x, uint32_t, uint32_t, uint32_t*, uint32_t uni) {
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "setp.eq.s32 %%p, %1, 0;\n\t"
      "@%%p bra DONE%=;\n\t"
      "add.s32 %0, %0, 7;\n\t"
      "DONE%=:\n\t"
      "}\n\t"
      : "+r"(x)
      : "r"(uni));
}

enum { X_IADD3, X_LOP3, X_PRMT, X_SHFL, X_IMAD, X_IDP4A, X_HFMA2, X_LDS128, X_BRA, X_N };
static const char* X_NAME[] = {"iadd3", "lop3", "prmt", "shfl", "imad", "idp4a", "hfma2", "lds128", "bra"};

template <int X>
__device__ __forceinline__ void xop(uint32_t& x, uint32_t y, uint32_t z, uint32_t* smem, uint32_t uni) {
  if (X == X_IADD3) x_iadd3(x, y, z, smem, uni);
  else if (X == X_LOP3) x_lop3(x, y, z, smem, uni);
  else if (X == X_PRMT) x_prmt(x, y, z, smem, uni);
  else if (X == X_SHFL) x_shfl(x, y, z, smem, uni);
  else if (X == X_IMAD) x_imad(x, y, z, smem, uni);
  else if (X == X_IDP4A) x_idp4a(x, y, z, smem, uni);
  else if (X == X_HFMA2) x_hfma2(x, y, z, smem, uni);
  else if (X == X_LDS128) x_lds128(x, y, z, smem, uni);
  else x_bra(x, y, z, smem, uni);
}

// ---- operand set-up shared by every kernel ----------------------------------------------------------------------
__device__ __forceinline__ void init_mma(int M, uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x08080808u ^ lane; a[1] = 0x01010101u; a[2] = 0x08010801u; a[3] = 0x01080108u;
  b[0] = 0x08080808u; b[1] = 0x01010101u ^ lane;
}
__device__ __forceinline__ void init_r(uint32_t* r, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
}
__device__ __forceinline__ uint32_t* smem_base() {
  extern __shared__ uint32_t pad[];
  return pad;
}

// ---- MMA alone --------------------------------------------------------------------------------------------------
template <int M>
__global__ void __launch_bounds__(512) k_mma(int iters, uint32_t* sink, BlockClk* rec) {
  uint32_t* s = smem_base();
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
  init_mma(M, lane, a, b);
  int32_t di[8][4] = {};
  float df[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      if (M == M_IMMA) mma_imma(di[c], a, b);
      else mma_omma(df[c], a, b);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
  if (x == 0x1234u) sink[blockIdx.x * blockDim.x + threadIdx.x] = x + s[0];
}

// ---- one opcode alone -------------------------------------------------------------------------------------------
template <int X>
__global__ void __launch_bounds__(512) k_op(int iters, uint32_t uni, uint32_t* sink, BlockClk* rec) {
  uint32_t* s = smem_base();
  uint32_t r[8];
  init_r(r, threadIdx.x);
  if (threadIdx.x < 512) s[threadIdx.x] = r[0] + threadIdx.x;  // fill the smem LDS.128 reads
  __syncthreads();
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < 8; q++) xop<X>(r[q], r[(q + 1) & 7], r[(q + 2) & 7], s, uni);
  }
  CLK_END(rec)
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x += r[i];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- MMA interleaved with an opcode -----------------------------------------------------------------------------
// Each iteration issues 8 MMAs; R ops are spread R/8 a MMA across the 8-register rotation.
template <int M, int X, int R>
__global__ void __launch_bounds__(512) k_mix(int iters, uint32_t uni, uint32_t* sink, BlockClk* rec) {
  uint32_t* s = smem_base();
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
  init_mma(M, lane, a, b);
  int32_t di[8][4] = {};
  float df[8][4] = {};
  uint32_t r[8];
  init_r(r, threadIdx.x);
  if (threadIdx.x < 512) s[threadIdx.x] = r[0] + threadIdx.x;
  __syncthreads();
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      if (M == M_IMMA) mma_imma(di[c], a, b);
      else mma_omma(df[c], a, b);
#pragma unroll
      for (int q = 0; q < R / 8; q++) {
        const int i = (c * (R / 8) + q) & 7;
        xop<X>(r[i], r[(i + 1) & 7], r[(i + 2) & 7], s, uni);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
#pragma unroll
  for (int i = 0; i < 8; i++) x += r[i];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- host driver ------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static int g_warps = 16;  // 4 warps a SMSP: maximal issue pressure
static const uint32_t UNI = 1;

template <int M>
static double mma_alone(double* macs_per_clk) {
  auto kern = k_mma<M>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, g_warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = std::max(64, (int)(2e7 / (w.cyc_med / 64)));
  Timing t = launch(iters);
  double mmas = (double)g_warps * iters * 8;
  *macs_per_clk = mmas * M_MACS[M] / t.cyc_med;
  double mma_warp_instr_per_clk = mmas / t.cyc_med;
  printf("{\"probe\":\"issue\",\"mode\":\"mma_alone\",\"mma\":\"%s\",\"warps\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"mma_calls_per_sm_per_clk\":%.4f,\"mma_macs_per_sm_per_clk\":%.2f}\n",
         M_NAME[M], g_warps, iters, t.cyc_med, t.mhz, mma_warp_instr_per_clk, *macs_per_clk);
  fflush(stdout);
  return t.cyc_med;
}

template <int X>
static void op_alone() {
  auto kern = k_op<X>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, g_warps * 32, ONE_BLOCK_SMEM>>>(it, UNI, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = std::max(64, (int)(2e7 / (w.cyc_med / 64)));
  Timing t = launch(iters);
  double ops = (double)g_warps * iters * 8;
  printf("{\"probe\":\"issue\",\"mode\":\"op_alone\",\"op\":\"%s\",\"warps\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"op_calls_per_sm_per_clk\":%.4f}\n",
         X_NAME[X], g_warps, iters, t.cyc_med, t.mhz, ops / t.cyc_med);
  fflush(stdout);
}

template <int M, int X, int R>
static void mix_case(double mma_macs_alone) {
  auto kern = k_mix<M, X, R>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, g_warps * 32, ONE_BLOCK_SMEM>>>(it, UNI, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = std::max(64, (int)(2e7 / (w.cyc_med / 64)));
  Timing t = launch(iters);
  double mmas = (double)g_warps * iters * 8, ops = (double)g_warps * iters * R;
  double mma_calls = mmas / t.cyc_med, op_calls = ops / t.cyc_med;
  // "warp instr/SM/clk" counting one SASS a mma.sync and one a op: a lower bound on the true issue, corrected from
  // the SASS by issue_ceiling.py. The ceiling is 4 a SM a clock (1 a SMSP).
  double warp_instr_min = mma_calls + op_calls;
  printf("{\"probe\":\"issue\",\"mode\":\"mix\",\"mma\":\"%s\",\"op\":\"%s\",\"ops_per_mma\":%.3f,\"warps\":%d,"
         "\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"mma_calls_per_sm_per_clk\":%.4f,\"op_calls_per_sm_per_clk\":%.4f,"
         "\"warp_instr_min_per_sm_per_clk\":%.4f,\"warp_instr_min_per_smsp_per_clk\":%.4f,"
         "\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         M_NAME[M], X_NAME[X], R / 8.0, g_warps, iters, t.cyc_med, t.mhz, mma_calls, op_calls,
         warp_instr_min, warp_instr_min / 4.0, mmas * M_MACS[M] / t.cyc_med,
         (mmas * M_MACS[M] / t.cyc_med) / mma_macs_alone);
  fflush(stdout);
}

template <int M, int X>
static void mix_sweep(double mma_macs_alone) {
  mix_case<M, X, 8>(mma_macs_alone);
  mix_case<M, X, 16>(mma_macs_alone);
  mix_case<M, X, 32>(mma_macs_alone);
  mix_case<M, X, 64>(mma_macs_alone);
  mix_case<M, X, 128>(mma_macs_alone);
}

template <int M>
static void all_ops(double mma_macs_alone) {
  mix_sweep<M, X_IADD3>(mma_macs_alone);
  mix_sweep<M, X_LOP3>(mma_macs_alone);
  mix_sweep<M, X_PRMT>(mma_macs_alone);
  mix_sweep<M, X_SHFL>(mma_macs_alone);
  mix_sweep<M, X_IMAD>(mma_macs_alone);
  mix_sweep<M, X_IDP4A>(mma_macs_alone);
  mix_sweep<M, X_HFMA2>(mma_macs_alone);
  mix_sweep<M, X_LDS128>(mma_macs_alone);
  mix_sweep<M, X_BRA>(mma_macs_alone);
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  (void)only;
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 512 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));

  op_alone<X_IADD3>();
  op_alone<X_LOP3>();
  op_alone<X_PRMT>();
  op_alone<X_SHFL>();
  op_alone<X_IMAD>();
  op_alone<X_IDP4A>();
  op_alone<X_HFMA2>();
  op_alone<X_LDS128>();
  op_alone<X_BRA>();

  double macs_imma = 0, macs_omma = 0;
  mma_alone<M_IMMA>(&macs_imma);
  mma_alone<M_OMMA>(&macs_omma);

  all_ops<M_IMMA>(macs_imma);
  all_ops<M_OMMA>(macs_omma);

  printf("{\"probe\":\"issue\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
