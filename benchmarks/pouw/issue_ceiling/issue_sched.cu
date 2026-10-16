// Round-2 scheduling probe (compute-accounting, note:20261016T0300Z-draft-issue-ceiling-r2).
//
// Round 1 (note:20261016T0150Z-draft-issue-ceiling) interleaved a tensor MMA with a lane op inside ONE warp, which
// serialises issue: at 8 ops a MMA the IMMA kept only 0.756 of its alone MAC rate. RS-MODEL §9 fitted joint capacity
// rows to that point and lifted τ 0.205 -> 0.239 — but only if no schedule does better. The s8 tensor pipe is
// throughput-bound (~1 IMMA per 16 SMSP-cycles), so a co-resident lane warp could in principle fill the ~15 free issue
// slots a MMA and keep the MMA near its alone rate (perfect pipe overlap). This probe measures the schedules that could:
//
//   k_specmma<M>      MMA-alone at the spec geometry (2 MMA warps a SMSP, 8 independent accumulators)
//   k_spec<M,X,R>     warp-specialized: 2 MMA warps + 2 lane warps a SMSP; MMA warps 8 MMAs an iter, lane warps R ops.
//                     Effective ratio R/8 ops a MMA. One clock64 span over the block, so the MMA rate is read over the
//                     window both classes run.
//   k_mixu<M,X,R>     round-1's single-warp interleave, body unrolled x8 (loop overhead ~1/8).
//
// Lane ops X: IADD3 (= LOP3/PRMT, which §9 found identical), IDP.4A, HFMA2, MOV, SEL, FFMA, LDG, and MIX3
// (IADD3+IDP.4A+HFMA2 round-robin). Warp->SMSP is warp_id % 4 in practice, so warps {0,4} / {8,12} (etc.) put 2 MMA and
// 2 lane warps on each SMSP. issue_sched.py reads every loop's SASS (cuobjdump) to confirm the opcode that issued.
#include "common.cuh"

// ---- the two tensor MMAs (round-1's) ----------------------------------------------------------------------------
__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
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

// ---- lane ops ---------------------------------------------------------------------------------------------------
// one fused IADD3 (two dependent PTX adds; ptxas fuses them, as the census does)
__device__ __forceinline__ void o_iadd3(uint32_t& x, uint32_t y, uint32_t z, const uint32_t*) {
  asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void o_idp4a(uint32_t& x, uint32_t y, uint32_t z, const uint32_t*) {
  asm volatile("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void o_hfma2(uint32_t& x, uint32_t y, uint32_t z, const uint32_t*) {
  asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
// MOV kept live by asm volatile; x := y, so it stays in a rotating chain and isn't dead
__device__ __forceinline__ void o_mov(uint32_t& x, uint32_t y, uint32_t, const uint32_t*) {
  asm volatile("mov.b32 %0, %1;" : "=r"(x) : "r"(y));
}
// SEL: a data-dependent predicate then a select (emits ISETP + SEL)
__device__ __forceinline__ void o_sel(uint32_t& x, uint32_t y, uint32_t z, const uint32_t*) {
  asm volatile("{\n\t.reg .pred %%p;\n\tsetp.ne.s32 %%p, %2, 0;\n\tselp.b32 %0, %0, %1, %%p;\n\t}"
               : "+r"(x) : "r"(y), "r"(z));
}
// FFMA (FP32)
__device__ __forceinline__ void o_ffma(uint32_t& x, uint32_t y, uint32_t z, const uint32_t*) {
  float xf = __uint_as_float(x);
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(xf) : "f"(__uint_as_float(y)), "f"(__uint_as_float(z)));
  x = __float_as_uint(xf);
}
// LDG: a global load at a bounded index (emits LDG.E)
#define GMASK 0xFFFFFu  // 1<<20 words
__device__ __forceinline__ void o_ldg(uint32_t& x, uint32_t y, uint32_t, const uint32_t* g) {
  uint32_t v;
  const uint32_t* p = g + ((x ^ y) & GMASK);
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(p));
  x ^= v;
}

enum { X_IADD3, X_IDP4A, X_HFMA2, X_MOV, X_SEL, X_FFMA, X_LDG, X_MIX3, X_N };
static const char* X_NAME[] = {"iadd3", "idp4a", "hfma2", "mov", "sel", "ffma", "ldg", "mix3"};

// apply op #j (for MIX3, cycle three kinds); i selects the register, g the global buffer
template <int X>
__device__ __forceinline__ void oone(int j, uint32_t* r, int i, const uint32_t* g) {
  uint32_t& x = r[i];
  uint32_t y = r[(i + 1) & 7], z = r[(i + 2) & 7];
  if (X == X_IADD3) o_iadd3(x, y, z, g);
  else if (X == X_IDP4A) o_idp4a(x, y, z, g);
  else if (X == X_HFMA2) o_hfma2(x, y, z, g);
  else if (X == X_MOV) o_mov(x, y, z, g);
  else if (X == X_SEL) o_sel(x, y, z, g);
  else if (X == X_FFMA) o_ffma(x, y, z, g);
  else if (X == X_LDG) o_ldg(x, y, z, g);
  else {  // MIX3: IADD3, IDP.4A, HFMA2 in turn
    int m = j % 3;
    if (m == 0) o_iadd3(x, y, z, g);
    else if (m == 1) o_idp4a(x, y, z, g);
    else o_hfma2(x, y, z, g);
  }
}

// ---- operand set-up ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void init_mma(int M, uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x08080808u ^ lane; a[1] = 0x01010101u; a[2] = 0x08010801u; a[3] = 0x01080108u;
  b[0] = 0x08080808u; b[1] = 0x01010101u ^ lane;
}
__device__ __forceinline__ void init_r(uint32_t* r, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
}
__device__ __forceinline__ uint32_t* smem_base() { extern __shared__ uint32_t pad[]; return pad; }

// ---- spec geometry's MMA-alone baseline (2 MMA warps a SMSP) ----------------------------------------------------
template <int M>
__global__ void __launch_bounds__(512) k_specmma(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(M, lane, a, b);
  int32_t di[8][4] = {}; float df[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) { if (M == M_IMMA) mma_imma(di[c], a, b); else mma_omma(df[c], a, b); }
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- warp-specialized: MMA warps + lane warps co-resident -------------------------------------------------------
// First half of warps issue only MMAs, second half only op X. With 16 warps a SM, each SMSP (warp_id%4) hosts warps
// {r,r+4} (MMA) and {r+8,r+12} (lane). Both classes loop `iters`; ratio = R/8 ops a MMA.
template <int M, int X, int R>
__global__ void __launch_bounds__(512) k_spec(int iters, const uint32_t* g, uint32_t* sink, BlockClk* rec) {
  uint32_t* s = smem_base();
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);  // first half of warps
  uint32_t a[4], b[2]; init_mma(M, lane, a, b);
  int32_t di[8][4] = {}; float df[8][4] = {};
  uint32_t r[8]; init_r(r, threadIdx.x);
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 8; c++) { if (M == M_IMMA) mma_imma(di[c], a, b); else mma_omma(df[c], a, b); }
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) oone<X>(q, r, q & 7, g);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) {
#pragma unroll
    for (int c = 0; c < 8; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i];
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- fully-unrolled single-warp interleave (round-1's k_mix, body unrolled x8) -----------------------------------
template <int M, int X, int R>
__global__ void __launch_bounds__(512) k_mixu(int iters, const uint32_t* g, uint32_t* sink, BlockClk* rec) {
  uint32_t* s = smem_base();
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(M, lane, a, b);
  int32_t di[8][4] = {}; float df[8][4] = {};
  uint32_t r[8]; init_r(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {       // 8 straight-line sub-bodies: overhead is ~1/8 of round-1's
#pragma unroll
      for (int c = 0; c < 8; c++) {
        if (M == M_IMMA) mma_imma(di[c], a, b); else mma_omma(df[c], a, b);
#pragma unroll
        for (int q = 0; q < R / 8; q++) { const int i = (c * (R / 8) + q) & 7; oone<X>(u * 8 + c + q, r, i, g); }
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

// ---- high-occupancy warp-specialized (32 warps a SM, 8 a SMSP, 4 MMA + 4 lane) ----------------------------------
// Tests whether more warps let the lane ops hide behind IMMA. 4 MMA accumulators a warp (register-safe at 1024
// threads); 4 MMA warps a SMSP x 4 accumulators = 16 in-flight IMMAs a SMSP, enough to saturate the tensor pipe.
template <int M>
__global__ void __launch_bounds__(1024) k_specmma_hi(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(M, lane, a, b);
  int32_t di[4][4] = {}; float df[4][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 4; c++) { if (M == M_IMMA) mma_imma(di[c], a, b); else mma_omma(df[c], a, b); }
  }
  CLK_END(rec)
  uint32_t x = lane;
#pragma unroll
  for (int c = 0; c < 4; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

template <int M, int X, int R>
__global__ void __launch_bounds__(1024) k_spec_hi(int iters, const uint32_t* g, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);  // first half of warps
  uint32_t a[4], b[2]; init_mma(M, lane, a, b);
  int32_t di[4][4] = {}; float df[4][4] = {};
  uint32_t r[8]; init_r(r, threadIdx.x);
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 4; c++) { if (M == M_IMMA) mma_imma(di[c], a, b); else mma_omma(df[c], a, b); }
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) oone<X>(q, r, q & 7, g);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) {
#pragma unroll
    for (int c = 0; c < 4; c++) x ^= (M == M_IMMA) ? (uint32_t)di[c][0] : __float_as_uint(df[c][0]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i];
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- host driver ------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static uint32_t* g_ldg;
static int g_sms;
static const int WSM = 16;     // warps a SM (4 a SMSP), as round 1
static const int WSM_HI = 32;  // high-occupancy variant (8 a SMSP, 4 MMA + 4 lane)

static int size_iters(double cyc_warm, int warm) {
  return std::max(64, (int)(2e7 / (cyc_warm / warm)));
}

template <int M>
static double specmma_alone(double* macs_per_clk) {
  auto kern = k_specmma<M>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  int warps = WSM / 2;  // same MMA-warp count/geometry as k_spec (2 a SMSP)
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double mmas = (double)warps * iters * 8;
  *macs_per_clk = mmas * M_MACS[M] / t.cyc_med;
  printf("{\"probe\":\"sched\",\"mode\":\"specmma_alone\",\"mma\":\"%s\",\"mma_warps\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"mma_calls_per_sm_per_clk\":%.4f,\"mma_macs_per_sm_per_clk\":%.2f}\n",
         M_NAME[M], warps, iters, t.cyc_med, t.mhz, mmas / t.cyc_med, *macs_per_clk);
  fflush(stdout);
  return t.cyc_med;
}

template <int M, int X, int R>
static void spec_case(double mma_macs_alone) {
  auto kern = k_spec<M, X, R>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_ldg, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  int mma_warps = WSM / 2, alu_warps = WSM / 2;
  double mmas = (double)mma_warps * iters * 8;        // MMAs a SM
  double ops = (double)alu_warps * iters * R;         // lane ops a SM
  double macs = mmas * M_MACS[M] / t.cyc_med;
  printf("{\"probe\":\"sched\",\"mode\":\"spec\",\"mma\":\"%s\",\"op\":\"%s\",\"ops_per_mma\":%.3f,"
         "\"mma_warps\":%d,\"alu_warps\":%d,\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"mma_calls_per_sm_per_clk\":%.4f,\"op_calls_per_sm_per_clk\":%.4f,"
         "\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         M_NAME[M], X_NAME[X], (double)R / 8.0, mma_warps, alu_warps, iters, t.cyc_med, t.mhz,
         mmas / t.cyc_med, ops / t.cyc_med, macs, macs / mma_macs_alone);
  fflush(stdout);
}

template <int M, int X, int R>
static void mixu_case(double mma_macs_alone) {
  auto kern = k_mixu<M, X, R>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_ldg, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double mmas = (double)WSM * iters * 64;             // 8 sub-bodies x 8 MMAs a SM-warp
  double ops = (double)WSM * iters * 8 * R;           // 8 sub-bodies x R ops
  double macs = mmas * M_MACS[M] / t.cyc_med;
  printf("{\"probe\":\"sched\",\"mode\":\"mixu\",\"mma\":\"%s\",\"op\":\"%s\",\"ops_per_mma\":%.3f,"
         "\"warps\":%d,\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"mma_calls_per_sm_per_clk\":%.4f,\"op_calls_per_sm_per_clk\":%.4f,"
         "\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         M_NAME[M], X_NAME[X], (double)R / 8.0, WSM, iters, t.cyc_med, t.mhz,
         mmas / t.cyc_med, ops / t.cyc_med, macs, macs / mma_macs_alone);
  fflush(stdout);
}

template <int M, int X>
static void sweep(double a) { spec_case<M, X, 32>(a); spec_case<M, X, 64>(a); spec_case<M, X, 128>(a);
                              mixu_case<M, X, 32>(a); mixu_case<M, X, 64>(a); mixu_case<M, X, 128>(a); }

template <int M>
static double specmma_hi_alone(double* macs_per_clk) {
  auto kern = k_specmma_hi<M>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  int warps = WSM_HI / 2;  // 16 MMA warps a SM (4 a SMSP)
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double mmas = (double)warps * iters * 4;
  *macs_per_clk = mmas * M_MACS[M] / t.cyc_med;
  printf("{\"probe\":\"sched\",\"mode\":\"specmma_hi_alone\",\"mma\":\"%s\",\"mma_warps\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"mma_calls_per_sm_per_clk\":%.4f,\"mma_macs_per_sm_per_clk\":%.2f}\n",
         M_NAME[M], warps, iters, t.cyc_med, t.mhz, mmas / t.cyc_med, *macs_per_clk);
  fflush(stdout);
  return t.cyc_med;
}

template <int M, int X, int R>
static void spec_hi_case(double mma_macs_alone) {
  auto kern = k_spec_hi<M, X, R>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) { return run_timed([&] { kern<<<g_sms, WSM_HI * 32, ONE_BLOCK_SMEM>>>(it, g_ldg, g_sink, g_rec); }, g_sms, g_rec); };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  int mma_warps = WSM_HI / 2, alu_warps = WSM_HI / 2;
  double mmas = (double)mma_warps * iters * 4;   // 4 MMAs a warp-iter
  double ops = (double)alu_warps * iters * R;
  double macs = mmas * M_MACS[M] / t.cyc_med;
  printf("{\"probe\":\"sched\",\"mode\":\"spec_hi\",\"mma\":\"%s\",\"op\":\"%s\",\"ops_per_mma\":%.3f,"
         "\"mma_warps\":%d,\"alu_warps\":%d,\"iters\":%d,\"cyc\":%.0f,\"mhz\":%.1f,"
         "\"mma_calls_per_sm_per_clk\":%.4f,\"op_calls_per_sm_per_clk\":%.4f,"
         "\"mma_macs_per_sm_per_clk\":%.2f,\"mma_macs_frac_of_alone\":%.4f}\n",
         M_NAME[M], X_NAME[X], (double)R / 4.0, mma_warps, alu_warps, iters, t.cyc_med, t.mhz,
         mmas / t.cyc_med, ops / t.cyc_med, macs, macs / mma_macs_alone);
  fflush(stdout);
}

template <int M, int X>
static void sweep_hi(double a) { spec_hi_case<M, X, 32>(a); spec_hi_case<M, X, 64>(a); }  // ratio R/4 = 8, 16

int main(int, char**) {
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 512 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  CK(cudaMalloc(&g_ldg, (size_t)(GMASK + 1) * 4));
  CK(cudaMemset(g_ldg, 0x5a, (size_t)(GMASK + 1) * 4));

  double a_imma = 0, a_omma = 0;
  specmma_alone<M_IMMA>(&a_imma);
  specmma_alone<M_OMMA>(&a_omma);

  sweep<M_IMMA, X_IADD3>(a_imma);
  sweep<M_IMMA, X_IDP4A>(a_imma);
  sweep<M_IMMA, X_HFMA2>(a_imma);
  sweep<M_IMMA, X_MOV>(a_imma);
  sweep<M_IMMA, X_SEL>(a_imma);
  sweep<M_IMMA, X_FFMA>(a_imma);
  sweep<M_IMMA, X_LDG>(a_imma);
  sweep<M_IMMA, X_MIX3>(a_imma);
  // OMMA where cheap: the ALU representative only
  sweep<M_OMMA, X_IADD3>(a_omma);

  // high-occupancy confirmation (32 warps a SM): does more occupancy let lane ops hide behind IMMA?
  double a_imma_hi = 0;
  specmma_hi_alone<M_IMMA>(&a_imma_hi);
  sweep_hi<M_IMMA, X_IADD3>(a_imma_hi);
  sweep_hi<M_IMMA, X_IDP4A>(a_imma_hi);
  sweep_hi<M_IMMA, X_HFMA2>(a_imma_hi);
  sweep_hi<M_IMMA, X_MIX3>(a_imma_hi);

  printf("{\"probe\":\"sched\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
