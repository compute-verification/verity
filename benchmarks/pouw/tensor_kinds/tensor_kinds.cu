// M1 of EPS1 TENSOR (note:20261017T0321Z-draft-eps1-tensor §5): the joint issue cost of every native sm_120 MMA form.
//
// JI3e's MMA coefficient (8.115 SMSP-cycles a MMA beside other vector-pipe instructions) was fitted beside IMMA s8
// m16n8k32 only. The issue form of the tensor theorem is tight only if no other form writes its output slots for less
// issue. Each form here is timed alone and beside IADD3 at 4, 8, 12 and 16 ops a MMA, in the two schedules of ISSUE's
// round 2 (note:20261016T0300Z-draft-issue-ceiling-r2): warp-specialized (k_spec: 2 MMA warps + 2 IADD3 warps a SMSP)
// and interleaved (k_mixu: one warp issues both, body unrolled x8). tensor_kinds.py reads the loops' SASS and fits the
// coefficient a in a·#MMA + #other <= cycles for each form.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                \
  do {                                                                                       \
    cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                 \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(2);                                                                               \
    }                                                                                        \
  } while (0)

struct BlockClk {
  long long cyc;
  int smid;
  int pad;
};

#define CLK_BEGIN \
  __syncthreads(); \
  long long _t0 = clock64();
#define CLK_END(rec)                                   \
  __syncthreads();                                     \
  if (threadIdx.x == 0) {                              \
    rec[blockIdx.x].cyc = clock64() - _t0;             \
    int s_;                                            \
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s_));    \
    rec[blockIdx.x].smid = s_;                         \
  }

struct Timing {
  double ms = 0, cyc_med = 0, cyc_min = 0, cyc_max = 0, mhz = 0;
  int blocks = 0, distinct_sms = 0;
};

// One warm-up launch, then `reps` launches; the rep with the smallest median block cycles.
template <class F>
Timing run_timed(F launch, int nblocks, BlockClk* d_rec, int reps = 5) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  launch();
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Timing best;
  best.cyc_med = 1e300;
  std::vector<BlockClk> h(nblocks);
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaMemcpy(h.data(), d_rec, nblocks * sizeof(BlockClk), cudaMemcpyDeviceToHost));
    std::vector<double> c;
    std::vector<int> s;
    for (auto& b : h) { c.push_back((double)b.cyc); s.push_back(b.smid); }
    std::sort(c.begin(), c.end());
    std::sort(s.begin(), s.end());
    Timing t;
    t.ms = ms;
    t.blocks = nblocks;
    t.distinct_sms = (int)(std::unique(s.begin(), s.end()) - s.begin());
    t.cyc_med = c[c.size() / 2];
    t.cyc_min = c.front();
    t.cyc_max = c.back();
    t.mhz = ms > 0 ? t.cyc_max / (ms * 1e3) : 0;
    if (t.cyc_med < best.cyc_med) best = t;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best;
}

// Dynamic shared memory that keeps a second block off an SM (sm_120: 100 KB a SM).
constexpr int ONE_BLOCK_SMEM = 60 * 1024;

// ---- the forms ----------------------------------------------------------------------------------------------------
// Each form: its accumulator type and register count, its MACs and output slots a warp instruction, and the asm.
// Operand registers a[0..3], b[0..3], e (sparse metadata) are loop-invariant and set by init_ops.
#define A4 "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
#define A2 "r"(a[0]), "r"(a[1])
#define B2 "r"(b[0]), "r"(b[1])
#define B4 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3])

struct ImmaS8k32 {  // IMMA.16832.S8.S8: the honest MMA and JI3e's reference
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_s8_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaU8S8k32 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "imma_u8s8_m16n8k32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B2);
  }
};
struct ImmaS8k16 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "imma_s8_m16n8k16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A2, "r"(b[0]));
  }
};
struct ImmaS8m8k16 {
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 8 * 8 * 16, SLOTS = 64;
  static constexpr const char* NAME = "imma_s8_m8n8k16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(b[0]));
  }
};
struct ImmaSpS8k64 {  // 2:4 sparse, 32 useful MACs an output: a second exact depth-32 form
  static constexpr uint32_t W = 0x01020304u;
  using Acc = int32_t;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;  // useful (nonzero) MACs
  static constexpr const char* NAME = "imma_sp_s8_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t e) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : A4, B4, "r"(e));
  }
};
struct HmmaF16k16F32 {
  static constexpr uint32_t W = 0x14001400u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 128;
  static constexpr const char* NAME = "hmma_f16_m16n8k16_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct HmmaF16k16F16 {  // f16 accumulate: 128 outputs in 64 slots
  static constexpr uint32_t W = 0x14001400u;
  using Acc = uint32_t;
  static constexpr int NC = 2;
  static constexpr double MACS = 16 * 8 * 16, SLOTS = 64;
  static constexpr const char* NAME = "hmma_f16_m16n8k16_f16";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1]) : A4, B2);
  }
};
struct HmmaTf32k8 {
  static constexpr uint32_t W = 0x3a800000u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 8, SLOTS = 128;
  static constexpr const char* NAME = "hmma_tf32_m16n8k8";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct QmmaE4m3k32 {
  static constexpr uint32_t W = 0x20202020u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 32, SLOTS = 128;
  static constexpr const char* NAME = "qmma_e4m3_m16n8k32_f32";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2);
  }
};
struct OmmaNvf4k64 {
  static constexpr uint32_t W = 0x22222222u;
  using Acc = float;
  static constexpr int NC = 4;
  static constexpr double MACS = 16 * 8 * 64, SLOTS = 128;
  static constexpr const char* NAME = "omma_nvf4_m16n8k64";
  __device__ static void mma(Acc* d, const uint32_t* a, const uint32_t* b, uint32_t) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : A4, B2, "r"(0x38383838u), "r"(0x38383838u));
  }
};

// Small operands of each form's type, so no accumulator chain of millions of MMAs reaches inf or NaN; the integer
// forms wrap. The sparse metadata 0x4444 4444 selects positions (0, 1) of every group of four (ordered, valid).
template <class F>
__device__ __forceinline__ void init_ops(uint32_t lane, uint32_t* a, uint32_t* b, uint32_t& e) {
  const uint32_t w = F::W;
#pragma unroll
  for (int i = 0; i < 4; i++) { a[i] = w ^ (lane & 1u); b[i] = w; }
  e = 0x44444444u;
}

// ---- lane op: one fused IADD3 (two dependent PTX adds; ptxas fuses them) --------------------------------------------
__device__ __forceinline__ void iadd3(uint32_t& x, uint32_t y, uint32_t z) {
  asm volatile("add.u32 %0, %0, %1;\n\tadd.u32 %0, %0, %2;" : "+r"(x) : "r"(y), "r"(z));
}
__device__ __forceinline__ void op(uint32_t* r, int i) { iadd3(r[i & 7], r[(i + 1) & 7], r[(i + 2) & 7]); }
__device__ __forceinline__ void init_r(uint32_t* r, uint32_t tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
}

template <class F>
__device__ __forceinline__ uint32_t fold_acc(typename F::Acc (*d)[4]) {
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int j = 0; j < F::NC; j++) {
      typename F::Acc v = d[c][j];
      x ^= *reinterpret_cast<uint32_t*>(&v);
    }
  return x;
}

// MMA alone: 2 MMA warps a SMSP, 8 independent accumulators, the geometry k_spec's MMA warps use.
template <class F>
__global__ void __launch_bounds__(512) k_alone(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
  }
  CLK_END(rec)
  sink[blockIdx.x * blockDim.x + threadIdx.x] = fold_acc<F>(d) ^ lane;
}

// Warp-specialized: with 16 warps a SM, SMSP r hosts warps {r, r+4} (MMA) and {r+8, r+12} (IADD3); R/8 ops a MMA.
template <class F, int R>
__global__ void __launch_bounds__(512) k_spec(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool mma_role = warp < (blockDim.x >> 6);
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  uint32_t r[8];
  init_r(r, threadIdx.x);
  CLK_BEGIN
  if (mma_role) {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int c = 0; c < 8; c++) F::mma(d[c], a, b, e);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
      for (int q = 0; q < R; q++) op(r, q);
    }
  }
  CLK_END(rec)
  uint32_t x = lane;
  if (mma_role) x ^= fold_acc<F>(d);
  else
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// Interleaved: every warp issues a MMA then R/8 IADD3s, 8 accumulators, body unrolled x8 (64 MMAs an iteration).
template <class F, int R>
__global__ void __launch_bounds__(512) k_mixu(int iters, uint32_t* sink, BlockClk* rec) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[4], e;
  init_ops<F>(lane, a, b, e);
  typename F::Acc d[8][4] = {};
  uint32_t r[8];
  init_r(r, threadIdx.x);
  CLK_BEGIN
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        F::mma(d[c], a, b, e);
#pragma unroll
        for (int q = 0; q < R / 8; q++) op(r, c * (R / 8) + q);
      }
    }
  }
  CLK_END(rec)
  uint32_t x = lane ^ fold_acc<F>(d);
#pragma unroll
  for (int i = 0; i < 8; i++) x += r[i];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- host ---------------------------------------------------------------------------------------------------------
static uint32_t* g_sink;
static BlockClk* g_rec;
static int g_sms;
static const int WSM = 16;  // warps a SM, 4 a SMSP

static int size_iters(double cyc_warm, int warm) { return std::max(64, (int)(2e7 / (cyc_warm / warm))); }

template <class K>
static Timing timed(K kern, int warps, int* iters_out) {
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, ONE_BLOCK_SMEM));
  auto launch = [&](int it) {
    return run_timed([&] { kern<<<g_sms, warps * 32, ONE_BLOCK_SMEM>>>(it, g_sink, g_rec); }, g_sms, g_rec);
  };
  Timing w = launch(64);
  int iters = size_iters(w.cyc_med, 64);
  *iters_out = iters;
  return launch(iters);
}

static void emit(const char* mode, const char* form, double macs, double slots, int R, int mma_warps, int op_warps,
                 int iters, double mmas_sm, double ops_sm, const Timing& t) {
  printf("{\"probe\":\"tensor_kinds\",\"mode\":\"%s\",\"form\":\"%s\",\"macs\":%.0f,\"slots\":%.0f,\"ops_per_mma\":%.3f,"
         "\"mma_warps\":%d,\"op_warps\":%d,\"iters\":%d,\"cyc_med\":%.0f,\"cyc_min\":%.0f,\"cyc_max\":%.0f,"
         "\"distinct_sms\":%d,\"mhz\":%.1f,\"mmas_per_sm\":%.0f,\"ops_per_sm\":%.0f,"
         "\"smsp_cycles_per_mma\":%.4f}\n",
         mode, form, macs, slots, mmas_sm > 0 ? ops_sm / mmas_sm : 0.0, mma_warps, op_warps, iters, t.cyc_med, t.cyc_min,
         t.cyc_max, t.distinct_sms, t.mhz, mmas_sm, ops_sm, mmas_sm > 0 ? 4.0 * t.cyc_med / mmas_sm : 0.0);
  fflush(stdout);
}

template <class F>
static void alone() {
  int iters;
  const int warps = WSM / 2;
  Timing t = timed(k_alone<F>, warps, &iters);
  emit("alone", F::NAME, F::MACS, F::SLOTS, 0, warps, 0, iters, (double)warps * iters * 8, 0, t);
}
template <class F, int R>
static void spec() {
  int iters;
  Timing t = timed(k_spec<F, R>, WSM, &iters);
  emit("spec", F::NAME, F::MACS, F::SLOTS, R, WSM / 2, WSM / 2, iters, (double)(WSM / 2) * iters * 8,
       (double)(WSM / 2) * iters * R, t);
}
template <class F, int R>
static void mixu() {
  int iters;
  Timing t = timed(k_mixu<F, R>, WSM, &iters);
  emit("mixu", F::NAME, F::MACS, F::SLOTS, R, WSM, 0, iters, (double)WSM * iters * 64, (double)WSM * iters * 8 * R, t);
}
template <class F>
static void form() {
  alone<F>();
  spec<F, 32>(); spec<F, 64>(); spec<F, 96>(); spec<F, 128>();
  mixu<F, 32>(); mixu<F, 64>(); mixu<F, 96>(); mixu<F, 128>();
}

int main() {
  int d = 0;
  CK(cudaGetDevice(&d));
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, d));
  CK(cudaMalloc(&g_sink, (size_t)g_sms * 512 * 4));
  CK(cudaMalloc(&g_rec, (size_t)g_sms * sizeof(BlockClk)));
  // Reference first and last, so drift across the run shows as a difference between its two passes.
  form<ImmaS8k32>();
  form<ImmaU8S8k32>();
  form<ImmaS8k16>();
  form<ImmaS8m8k16>();
  form<ImmaSpS8k64>();
  form<HmmaF16k16F32>();
  form<HmmaF16k16F16>();
  form<HmmaTf32k8>();
  form<QmmaE4m3k32>();
  form<OmmaNvf4k64>();
  form<ImmaS8k32>();
  printf("{\"probe\":\"tensor_kinds\",\"done\":true,\"num_sms\":%d}\n", g_sms);
  return 0;
}
