// Round-6 joint-row probe.
//
// RS-W' r3 (VF3d) has τ = 0.1208 set by the two-word credits of IADD.64
// and DADD/DFMA, each priced at ONE dispatch slot on the issue row + JI3. A joint row MMA·a_M + op·a_X <= cycles can
// recover τ if a_X is measured. This probe measures, for DADD, DFMA, RED.64 and the 16-byte vector RED (and, to confirm
// r5's rates are dispatch-bound not latency-bound, IADD.64 and IMAD.WIDE with the 32-bit IADD3/IMAD as saturation
// witnesses), two numbers in JI3 units:
//   (1) displacement of IMMA per op = a_X/a_M = slope of (1 - IMMA-MAC-frac-of-alone) vs ops-a-MMA at small OPM
//       (k_mixu unrolled, k_spec warp-specialized), a fraction of one IMMA;
//   (2) own sustained rate = ops / SMSP-cycle standalone = 1/a_X, a fraction of the 1-op-a-cycle JI3 reference
//       (k_alone at 8-way and 16-way ILP; the 16-way run shows the rate has plateaued, i.e. dispatch-bound).
// issue_wide.py reads the SASS to confirm the lowering (add.f64->DADD, fma.rn.f64->DFMA, red.global.add.u64->REDG.E.ADD.64,
// red.global.add.v4.f32->REDG.E.ADD.F32x4; add.u64->IADD.64, mad.wide.u32->IMAD.WIDE.U32).
#include "common.cuh"

// ---- IMMA s8 ----------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_imma(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static const double IMMA_MACS = 4096;  // 16x8x32 MACs a MMA

// ---- the kinds --------------------------------------------------------------------------------------------------
// INT32 contrasts (u32 accumulator): iadd3_32, imad_32. INT64 targets (u64 accumulator): iadd_64, imad_wide.
// FP64 targets (double accumulator): dadd, dfma. RED targets (global reduction): red64, redv4_f32.
enum { X_IADD3, X_IMAD, X_IADD64, X_IMADWIDE, X_DADD, X_DFMA, X_RED64, X_REDV4, X_N };
static const char* X_NAME[] = {"iadd3_32", "imad_32", "iadd_64", "imad_wide", "dadd", "dfma", "red64", "redv4_f32"};

template <int X> __device__ constexpr bool is_i32() { return X == X_IADD3 || X == X_IMAD; }
template <int X> __device__ constexpr bool is_i64() { return X == X_IADD64 || X == X_IMADWIDE; }
template <int X> __device__ constexpr bool is_f64() { return X == X_DADD || X == X_DFMA; }
template <int X> __device__ constexpr bool is_red() { return X == X_RED64 || X == X_REDV4; }

// Each op is a self-recurrence on accumulator `slot` with a fixed register addend (constant across the loop so nothing
// hoists, but the chain is only `slot` long, giving NACC-wide ILP a warp x warps a SMSP). asm volatile keeps every op.
// RED streams through a big global pool (POOL power of 2): op `seq` of thread `gtid` hits pool[(gtid + seq*total) & mask].
// Consecutive lanes hit consecutive elements (coalesced); each seq step jumps `total` elements (a new line), so L2 line
// contention is spread over the whole pool, isolating the issue-side cost from the L2-word-add throughput (VF3d's own row).
template <int X>
__device__ __forceinline__ void oone(uint32_t* r, uint64_t* w, double* f, uint64_t* pool64, float4* poolv4,
                                     int slot, uint32_t gtid, uint32_t total, uint32_t mask, uint32_t seq,
                                     uint32_t c32, uint64_t c64, double cd, float fv) {
  if constexpr (is_i32<X>()) {
    if (X == X_IADD3) asm volatile("add.u32 %0,%0,%1;" : "+r"(r[slot]) : "r"(c32));
    else              asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[slot]) : "r"(c32), "r"(c32));
  } else if constexpr (is_i64<X>()) {
    if (X == X_IADD64) asm volatile("add.u64 %0,%0,%1;" : "+l"(w[slot]) : "l"(c64));
    else {  // IMAD.WIDE: multiply the low word of the changing accumulator so the 32x32->64 can't be hoisted
      uint32_t a = (uint32_t)w[slot];
      asm volatile("mad.wide.u32 %0,%1,%2,%0;" : "+l"(w[slot]) : "r"(a), "r"(c32));
    }
  } else if constexpr (is_f64<X>()) {
    if (X == X_DADD) asm volatile("add.f64 %0,%0,%1;" : "+d"(f[slot]) : "d"(cd));
    else             asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(f[slot]) : "d"(cd), "d"(cd));
  } else {  // RED: streaming global reduction, side-effecting so ptxas keeps every one
    uint32_t idx = (uint32_t)(((uint64_t)gtid + (uint64_t)seq * total) & mask);
    if (X == X_RED64) {
      uint64_t* p = pool64 + idx;
      asm volatile("red.global.add.u64 [%0], %1;" ::"l"(p), "l"(c64) : "memory");
    } else {
      float4* p = poolv4 + idx;
      asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(fv), "f"(fv), "f"(fv), "f"(fv) : "memory");
    }
  }
}

template <int X, int NACC>
__device__ __forceinline__ void init_acc(uint32_t* r, uint64_t* w, double* f, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    if constexpr (is_i32<X>()) r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
    else if constexpr (is_i64<X>()) w[i] = ((((uint64_t)(tid * 0x9e3779b9u + i)) << 32) ^ (0x1234567u + i)) | 1u;
    else if constexpr (is_f64<X>()) f[i] = (double)((tid & 0xff) + i + 1);  // I2F, no DADD/DFMA in init
  }
}
template <int X, int NACC>
__device__ __forceinline__ uint32_t consume_acc(const uint32_t* r, const uint64_t* w, const double* f,
                                                uint64_t* g64, float4* gv4) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    if constexpr (is_i32<X>()) x ^= r[i];
    else if constexpr (is_i64<X>()) x ^= (uint32_t)w[i] ^ (uint32_t)(w[i] >> 32);
    else if constexpr (is_f64<X>()) { long long b = __double_as_longlong(f[i]); x ^= (uint32_t)b ^ (uint32_t)(b >> 32); }
  }
  if constexpr (is_red<X>()) x ^= (uint32_t)(uintptr_t)g64 ^ (uint32_t)(uintptr_t)gv4;
  return x;
}

__device__ __forceinline__ void init_mma(uint32_t lane, uint32_t* a, uint32_t* b) {
  a[0] = 0x08080808u ^ lane; a[1] = 0x01010101u; a[2] = 0x08010801u; a[3] = 0x01080108u;
  b[0] = 0x08080808u; b[1] = 0x01010101u ^ lane;
}

// ---- (2) standalone throughput: 8 warps a SMSP, 64 ops an iter, NACC-wide ILP, no MMA -------------------------
template <int X, int NACC>
__global__ void __launch_bounds__(1024) k_alone(int iters, uint32_t* sink, BlockClk* rec,
                                                uint64_t* pool64, float4* poolv4, uint32_t total, uint32_t mask) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  const double cd = 1.0000001192092896;
  const float fv = (float)(c32 & 0xffff) + 1.0f;
  uint32_t r[NACC]; uint64_t w[NACC]; double f[NACC];
  init_acc<X, NACC>(r, w, f, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int q = 0; q < 64; q++)
      oone<X>(r, w, f, pool64, poolv4, q & (NACC - 1), gtid, total, mask, (uint32_t)it * 64u + (uint32_t)q,
              c32, c64, cd, fv);
  }
  CLK_END(rec)
  sink[gtid] = consume_acc<X, NACC>(r, w, f, pool64, poolv4);
}

// ---- (3) MMA-alone baseline (2 MMA warps a SMSP, 8 accumulators) ----------------------------------------------
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

// ---- warp-specialized: 2 MMA warps + 2 lane warps a SMSP; lane warps do R ops an iter (ratio R/8) -------------
template <int X, int R>
__global__ void __launch_bounds__(512) k_spec(int iters, uint32_t* sink, BlockClk* rec,
                                              uint64_t* pool64, float4* poolv4, uint32_t total, uint32_t mask) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);  // first half of warps
  uint32_t a[4], b[2]; init_mma(lane, a, b);
  int32_t d[8][4] = {};
  uint32_t r[8]; uint64_t w[8]; double f[8];
  init_acc<X, 8>(r, w, f, threadIdx.x);
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  const double cd = 1.0000001192092896;
  const float fv = (float)(c32 & 0xffff) + 1.0f;
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
      for (int q = 0; q < R; q++)
        oone<X>(r, w, f, pool64, poolv4, q & 7, gtid, total, mask, (uint32_t)it * R + (uint32_t)q, c32, c64, cd, fv);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) {
#pragma unroll
    for (int c = 0; c < 8; c++) x ^= (uint32_t)d[c][0];
  } else x ^= consume_acc<X, 8>(r, w, f, pool64, poolv4);
  sink[gtid] = x;
}

// ---- unrolled single-warp interleave: per MMA, OPM ops -------------------------------------------------------
template <int X, int OPM>
__global__ void __launch_bounds__(512) k_mixu(int iters, uint32_t* sink, BlockClk* rec,
                                              uint64_t* pool64, float4* poolv4, uint32_t total, uint32_t mask) {
  const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; init_mma(lane, a, b);
  int32_t d[8][4] = {};
  uint32_t r[8]; uint64_t w[8]; double f[8];
  init_acc<X, 8>(r, w, f, threadIdx.x);
  const uint32_t c32 = threadIdx.x | 1u;
  const uint64_t c64 = ((uint64_t)c32 << 32) | c32;
  const double cd = 1.0000001192092896;
  const float fv = (float)(c32 & 0xffff) + 1.0f;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma_imma(d[c], a, b);
#pragma unroll
      for (int q = 0; q < OPM; q++)
        oone<X>(r, w, f, pool64, poolv4, (c + q) & 7, gtid, total, mask,
                ((uint32_t)it * 8u + (uint32_t)c) * OPM + (uint32_t)q, c32, c64, cd, fv);
    }
  }
  CLK_END(rec)
  uint32_t x = lane ^ consume_acc<X, 8>(r, w, f, pool64, poolv4);
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= (uint32_t)d[c][0];
  sink[gtid] = x;
}

// ---- host -----------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static uint64_t* g_pool64;
static float4* g_poolv4;
static int g_sms;
static const int WSM_ALONE = 32;          // 8 warps a SMSP for standalone saturation
static const uint32_t POOL = 1u << 25;    // RED streaming pool elements (power of 2): 256 MiB u64, 512 MiB float4
static const uint32_t POOL_MASK = POOL - 1;

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <int X, int NACC>
static void alone_case() {
  auto kern = k_alone<X, NACC>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  uint32_t total = (uint32_t)g_sms * WSM_ALONE * 32;  // grid threads: coalescing stride for the RED pool
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, WSM_ALONE * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec, g_pool64, g_poolv4, total, POOL_MASK); },
                     g_sms, g_rec);
  };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  Timing t = launch(iters);
  double ops = (double)WSM_ALONE * 64 * iters;  // ops a SM
  printf("{\"probe\":\"wide\",\"mode\":\"alone\",\"op\":\"%s\",\"ilp\":%d,\"warps\":%d,\"iters\":%d,\"cyc\":%.0f,"
         "\"mhz\":%.1f,\"ops_per_smsp_per_cyc\":%.4f}\n",
         X_NAME[X], NACC, WSM_ALONE, iters, t.cyc_med, t.mhz, ops / t.cyc_med / 4.0);
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
  uint32_t total = (uint32_t)g_sms * WSM * 32;
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec, g_pool64, g_poolv4, total, POOL_MASK); },
                     g_sms, g_rec);
  };
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
  uint32_t total = (uint32_t)g_sms * WSM * 32;
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, WSM * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec, g_pool64, g_poolv4, total, POOL_MASK); },
                     g_sms, g_rec);
  };
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

int main(int, char**) {
  g_sms = num_sms();
  size_t total_max = (size_t)g_sms * 1024;
  CK(cudaMalloc(&g_sink, total_max * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  CK(cudaMalloc(&g_pool64, (size_t)POOL * sizeof(uint64_t)));
  CK(cudaMalloc(&g_poolv4, (size_t)POOL * sizeof(float4)));
  CK(cudaMemset(g_pool64, 0, (size_t)POOL * sizeof(uint64_t)));
  CK(cudaMemset(g_poolv4, 0, (size_t)POOL * sizeof(float4)));

  // standalone at 8-way and 16-way ILP (the 16-way run shows each rate has plateaued -> dispatch-bound not latency-bound)
  alone_case<X_IADD3, 8>();  alone_case<X_IADD3, 16>();
  alone_case<X_IMAD, 8>();   alone_case<X_IMAD, 16>();
  alone_case<X_IADD64, 8>(); alone_case<X_IADD64, 16>();
  alone_case<X_IMADWIDE, 8>(); alone_case<X_IMADWIDE, 16>();
  alone_case<X_DADD, 8>();   alone_case<X_DADD, 16>();
  alone_case<X_DFMA, 8>();   alone_case<X_DFMA, 16>();
  alone_case<X_RED64, 8>();  alone_case<X_RED64, 16>();
  alone_case<X_REDV4, 8>();  alone_case<X_REDV4, 16>();

  double a = specmma_alone();
  sweep<X_IADD3>(a); sweep<X_IMAD>(a); sweep<X_IADD64>(a); sweep<X_IMADWIDE>(a);
  sweep<X_DADD>(a); sweep<X_DFMA>(a); sweep<X_RED64>(a); sweep<X_REDV4>(a);

  printf("{\"probe\":\"wide\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
