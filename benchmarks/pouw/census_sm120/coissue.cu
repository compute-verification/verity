// Shared-issue census (item 3): how much tensor work and CUDA-core work slow each other.
//   same kernel: each loop iteration issues 8 independent MMAs (FP8 QMMA, NVFP4 OMMA or BF16 HMMA) and R CUDA ops of
//                one kind (FFMA, FFMA with immediate operands, LOP3, F2FP E4M3 pack, IMAD) on registers the MMAs don't touch. Each (MMA, op, R) is
//                timed with both, with the MMAs only and with the CUDA ops only, one block a SM of W warps.
//   two kernels: a QMMA kernel and a CUDA-op kernel launched on two streams, each one 4-warp block a SM (40 KB of
//                shared memory each, so one of each fits a SM), timed alone and together; %smid and %globaltimer
//                confirm they shared the SMs and overlapped.
#include "common.cuh"

#include <cstring>

#define F(x) __uint_as_float(x)
__device__ __forceinline__ void x_ffma(uint32_t& x, uint32_t y, uint32_t z) {
  float t = F(x);
  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(t) : "f"(F(y)), "f"(F(z)));
  x = __float_as_uint(t);
}
__device__ __forceinline__ void x_lop3(uint32_t& x, uint32_t y, uint32_t z) {
  asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_imad(uint32_t& x, uint32_t y, uint32_t z) {
  asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void x_f2fp(uint32_t& x, uint32_t y, uint32_t z) {
  asm volatile("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %2; cvt.u32.u16 %0, t;}" : "=r"(x) : "f"(F(y)), "f"(F(z)));
}
__device__ __forceinline__ void x_ffma_c(uint32_t& x, uint32_t, uint32_t) {
  float t = F(x);
  asm volatile("fma.rn.f32 %0, %0, 0f3F7FFFFE, 0f3F000000;" : "+f"(t));
  x = __float_as_uint(t);
}
enum { X_FFMA, X_LOP3, X_IMAD, X_F2FP, X_FFMA_C };
static const char* X_NAME[] = {"ffma", "lop3", "imad", "f2fp_e4m3x2", "ffma_c"};
template <int X>
__device__ __forceinline__ void xop(uint32_t& x, uint32_t y, uint32_t z) {
  if (X == X_FFMA) x_ffma(x, y, z);
  else if (X == X_LOP3) x_lop3(x, y, z);
  else if (X == X_IMAD) x_imad(x, y, z);
  else if (X == X_FFMA_C) x_ffma_c(x, y, z);
  else x_f2fp(x, y, z);
}

enum { M_QMMA, M_OMMA, M_HMMA };
static const char* M_NAME[] = {"e4m3_f32_qmma", "nvfp4_omma", "bf16_f32_hmma"};
static const double M_MACS[] = {4096, 8192, 2048};
template <int M>
__device__ __forceinline__ void mma(float* d, const uint32_t* a, const uint32_t* b) {
  if (M == M_QMMA)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else if (M == M_OMMA)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0x38383838u), "r"(0x38383838u));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int M, int X, int R, bool DO_MMA>
__global__ void __launch_bounds__(512) k_co(int iters, uint32_t* sink, BlockClk* rec) {
  extern __shared__ uint32_t pad[];
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4] = {0x38383838u ^ lane, 0x30303030u, 0x38303830u, 0x30383038u}, b[2] = {0x38383838u, 0x30303030u ^ lane};
  if (M == M_HMMA) { a[0] = 0x3f803f80u ^ (lane & 0x7f); a[1] = a[2] = a[3] = 0x3f803f80u; b[0] = b[1] = 0x3f803f80u; }
  float d[8][4] = {};
  uint32_t r[8];
#pragma unroll
  for (int i = 0; i < 8; i++) r[i] = (threadIdx.x * 0x9e3779b9u + i * 0x85ebca6bu) & 0x3FFFFFFFu;
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      if (DO_MMA) mma<M>(d[c], a, b);
#pragma unroll
      for (int q = 0; q < R / 8; q++) {
        const int i = (c * (R / 8) + q) & 7;
        xop<X>(r[i], r[(i + 1) & 7], r[(i + 2) & 7]);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x += r[i];
#pragma unroll
  for (int c = 0; c < 8; c++) x ^= __float_as_uint(d[c][0] + d[c][1] + d[c][2] + d[c][3]);
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

static uint32_t* g_sink;
static BlockClk *g_rec, *g_rec2;
static int g_sms;

template <int M, int X, int R, bool DO_MMA>
static Timing co_time(int warps, int iters) {
  auto kern = k_co<M, X, R, DO_MMA>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(iters, g_sink, g_rec); }, g_sms, g_rec);
}

template <int M, int X, int R>
static void co_case(int warps, const char* only) {
  char label[96];
  snprintf(label, sizeof label, "co:%s:%s:%d", M_NAME[M], X_NAME[X], R);
  if (only && !strstr(label, only)) return;
  Timing tm = co_time<M, X, 0, true>(warps, 64);
  int iters = std::max(64, (int)(2e7 / (tm.cyc_med / 64)));
  Timing both = co_time<M, X, R, true>(warps, iters);
  Timing mm = co_time<M, X, 0, true>(warps, iters);
  Timing cu = co_time<M, X, R, false>(warps, iters);
  double mmas = (double)warps * iters * 8, xops = (double)warps * iters * R;
  printf("{\"probe\":\"coissue\",\"mode\":\"same_kernel\",\"mma\":\"%s\",\"xop\":\"%s\",\"xops_per_mma\":%.3f,\"warps\":%d,"
         "\"iters\":%d,\"cyc_both\":%.0f,\"cyc_mma_only\":%.0f,\"cyc_x_only\":%.0f,\"mhz_both\":%.1f,"
         "\"mma_macs_per_sm_per_clk_alone\":%.2f,\"mma_macs_per_sm_per_clk_both\":%.2f,"
         "\"x_warp_ops_per_sm_per_clk_alone\":%.4f,\"x_warp_ops_per_sm_per_clk_both\":%.4f,"
         "\"mma_slowdown\":%.4f,\"x_slowdown\":%.4f,\"both_over_sum\":%.4f}\n",
         M_NAME[M], X_NAME[X], R / 8.0, warps, iters, both.cyc_med, mm.cyc_med, cu.cyc_med, both.mhz,
         mmas * M_MACS[M] / mm.cyc_med, mmas * M_MACS[M] / both.cyc_med, xops / cu.cyc_med, xops / both.cyc_med,
         both.cyc_med / mm.cyc_med, both.cyc_med / cu.cyc_med, both.cyc_med / (mm.cyc_med + cu.cyc_med));
  fflush(stdout);
}

template <int M, int X>
static void co_sweep(int warps, const char* only) {
  co_case<M, X, 8>(warps, only);
  co_case<M, X, 16>(warps, only);
  co_case<M, X, 32>(warps, only);
  co_case<M, X, 64>(warps, only);
  co_case<M, X, 128>(warps, only);
}

// ---- two concurrent kernels -------------------------------------------------------------------------------------
constexpr int TWO_SMEM = 40 * 1024;

template <int X>
static void two_kernels(const char* only) {
  char label[64];
  snprintf(label, sizeof label, "two:qmma+%s", X_NAME[X]);
  if (only && !strstr(label, only)) return;
  auto kt = k_co<M_QMMA, X, 0, true>;
  auto kc = k_co<M_QMMA, X, 64, false>;
  CK(cudaFuncSetAttribute(kt, cudaFuncAttributeMaxDynamicSharedMemorySize, TWO_SMEM));
  CK(cudaFuncSetAttribute(kc, cudaFuncAttributeMaxDynamicSharedMemorySize, TWO_SMEM));
  const int warps = 4;
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  int it_t = 1, it_c = 1;
  {  // calibrate each alone to about 2e7 cycles
    Timing a = run_timed([&] { kt<<<g_sms, warps * 32, TWO_SMEM, s1>>>(256, g_sink, g_rec); }, g_sms, g_rec, 1);
    Timing b = run_timed([&] { kc<<<g_sms, warps * 32, TWO_SMEM, s2>>>(256, g_sink, g_rec2); }, g_sms, g_rec2, 1);
    it_t = std::max(256, (int)(2e7 / (a.cyc_med / 256)));
    it_c = std::max(256, (int)(2e7 / (b.cyc_med / 256)));
  }
  Timing ta = run_timed([&] { kt<<<g_sms, warps * 32, TWO_SMEM, s1>>>(it_t, g_sink, g_rec); }, g_sms, g_rec);
  Timing tc = run_timed([&] { kc<<<g_sms, warps * 32, TWO_SMEM, s2>>>(it_c, g_sink, g_rec2); }, g_sms, g_rec2);
  // together: both launched back to back on two streams
  std::vector<BlockClk> h1(g_sms), h2(g_sms);
  double best_t = 1e300, best_c = 1e300, overlap_frac = 0;
  int shared_sms = 0;
  for (int rep = 0; rep < 3; rep++) {
    kt<<<g_sms, warps * 32, TWO_SMEM, s1>>>(it_t, g_sink, g_rec);
    kc<<<g_sms, warps * 32, TWO_SMEM, s2>>>(it_c, g_sink + g_sms * 512, g_rec2);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h1.data(), g_rec, g_sms * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2.data(), g_rec2, g_sms * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    Timing a = summarize(h1, 0), b = summarize(h2, 0);
    long long a0 = LLONG_MAX, a1 = 0, b0 = LLONG_MAX, b1 = 0;
    std::vector<int> sa(1024, 0);
    for (auto& r : h1) { a0 = std::min(a0, r.t0_ns); a1 = std::max(a1, r.t1_ns); sa[r.smid & 1023] = 1; }
    int sh = 0;
    for (auto& r : h2) { b0 = std::min(b0, r.t0_ns); b1 = std::max(b1, r.t1_ns); sh += sa[r.smid & 1023]; }
    double ov = (double)(std::min(a1, b1) - std::max(a0, b0)) / (double)std::max(a1 - a0, b1 - b0);
    if (a.cyc_med < best_t) { best_t = a.cyc_med; best_c = b.cyc_med; overlap_frac = ov; shared_sms = sh; }
  }
  double mmas = (double)warps * it_t * 8, xops = (double)warps * it_c * 64;
  printf("{\"probe\":\"coissue\",\"mode\":\"two_kernels\",\"mma\":\"%s\",\"xop\":\"%s\",\"warps_each\":%d,"
         "\"cyc_mma_alone\":%.0f,\"cyc_x_alone\":%.0f,\"cyc_mma_together\":%.0f,\"cyc_x_together\":%.0f,"
         "\"mma_macs_per_sm_per_clk_alone\":%.2f,\"mma_macs_per_sm_per_clk_together\":%.2f,"
         "\"x_warp_ops_per_sm_per_clk_alone\":%.4f,\"x_warp_ops_per_sm_per_clk_together\":%.4f,"
         "\"overlap_fraction\":%.3f,\"x_blocks_on_mma_sms\":%d,\"mhz_mma_alone\":%.1f}\n",
         M_NAME[M_QMMA], X_NAME[X], warps, ta.cyc_med, tc.cyc_med, best_t, best_c, mmas * 4096 / ta.cyc_med,
         mmas * 4096 / best_t, xops / tc.cyc_med, xops / best_c, overlap_frac, shared_sms, ta.mhz);
  fflush(stdout);
  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  g_sms = num_sms();
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 1024 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  CK(cudaMalloc(&g_rec2, (size_t)g_sms * sizeof(BlockClk)));
  for (int warps : {8, 16}) {
    co_sweep<M_QMMA, X_FFMA>(warps, only);
    co_sweep<M_QMMA, X_FFMA_C>(warps, only);
    co_sweep<M_QMMA, X_LOP3>(warps, only);
    co_sweep<M_QMMA, X_IMAD>(warps, only);
    co_sweep<M_QMMA, X_F2FP>(warps, only);
    co_sweep<M_OMMA, X_FFMA>(warps, only);
    co_sweep<M_HMMA, X_FFMA>(warps, only);
  }
  two_kernels<X_FFMA>(only);
  two_kernels<X_FFMA_C>(only);
  two_kernels<X_LOP3>(only);
  two_kernels<X_F2FP>(only);
  two_kernels<X_IMAD>(only);
  printf("{\"probe\":\"coissue\",\"done\":true}\n");
  return 0;
}
