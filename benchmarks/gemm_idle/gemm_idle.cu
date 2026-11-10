// A real-operand GEMM's tensor rate on one sm_120 card, and the tensor rate a co-resident undeclared kernel gets beside it.
//
//   gemm_idle info                                    the card (SMs, L2, shared memory, registers), one JSON line
//   gemm_idle peak KIND CLASS WARPS NACC SECS [SLEEP] mma.sync m16n8k32 on every SM from registers: WARPS warps an SM,
//                                                     NACC independent accumulators a warp, operands CLASS (uniform | zero),
//                                                     SLEEP ns of __nanosleep after each batch
//   gemm_idle gemm LIB KIND M N K SECS WCLASS W[:SLEEP]...
//                                                     the GEMM D[M x N] = X[M x K] . W[N x K]^T (M tokens, N out features,
//                                                     K in features) by LIB (lt: cuBLASLt's fastest of its heuristic's
//                                                     candidates; c2x: CUTLASS 2.x int8; c3x: CUTLASS 3.x sm_120 FP8, both
//                                                     the fastest of the PoUW harness's table), alone, then beside the
//                                                     co-runner (uniform operands, 8 accumulators) at W warps an SM and
//                                                     SLEEP ns a batch, with the fastest candidate it can sit beside, then
//                                                     alone again (alone_after: the drift a keep is read against)
//
// KIND is s8 (int8 in, int32 out) or e4m3 (FP8 E4M3 in, f32 accumulate, bf16 out). X is uniform random bytes (e4m3: NaN codes
// moved to the largest finite), W is WCLASS (uniform, or gauss: about N(0, 32) in int8, N(0, 100) in e4m3, as a
// per-channel or per-tensor quantized weight is). The operands stream from HBM: a ring of distinct (W, X, D) sets past
// three times the L2, replayed as one CUDA graph. Every pass's window is the GPU's globaltimer between two marker kernels
// in the GEMM's stream; the co-runner counts its MMAs between the markers (it reads them from global memory), so its rate
// is over the same window. NVML is sampled every 10 ms through each pass (SM clock, power, clock event reasons), and the
// pass's energy is NVML's counter difference. One JSON line a pass on stdout.
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(WITH_C2X) || defined(WITH_C3X)
#include "cutlass_api.h"
#endif

typedef uint32_t u32;
typedef uint64_t u64;

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));             \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)
#define LTC(x)                                                                                       \
  do {                                                                                               \
    cublasStatus_t s_ = (x);                                                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                               \
      fprintf(stderr, "%s:%d %s: cublas status %d\n", __FILE__, __LINE__, #x, int(s_));              \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

static int g_sms = 0;
static size_t g_l2 = 0;
static constexpr double MMA_MACS = 16.0 * 8 * 32;

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__host__ __device__ __forceinline__ u64 mix64(u64 x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

// e4m3's NaN codes (S.1111.111) to the largest finite magnitude (S.1111.110)
__host__ __device__ __forceinline__ u32 no_nan(u32 w) {
  for (int b = 0; b < 4; ++b)
    if (((w >> (8 * b)) & 0x7F) == 0x7F) w ^= 1u << (8 * b);
  return w;
}

enum { CLS_UNIFORM = 0, CLS_GAUSS = 1, CLS_ZERO = 2 };

__global__ void fill_k(u32* p, size_t n, u64 seed, int cls, int e4m3) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const u64 h = mix64(seed ^ (i * 0x9E3779B97F4A7C15ULL));
    u32 w = 0;
    if (cls == CLS_UNIFORM) {
      w = e4m3 ? no_nan((u32)h) : (u32)h;
    } else if (cls == CLS_GAUSS) {
      const u64 h2 = mix64(h + 0x632BE59BD9B4E019ULL);
      for (int b = 0; b < 4; ++b) {
        const u32 q = (u32)((b < 2 ? h : h2) >> (32 * (b & 1)));
        // Irwin-Hall of four bytes: mean 510, sd 147.8
        const float s = float(int(q & 255) + int((q >> 8) & 255) + int((q >> 16) & 255) + int(q >> 24) - 510);
        u32 byte;
        if (e4m3) {
          byte = __nv_cvt_float_to_fp8(s * (100.f / 147.8f), __NV_SATFINITE, __NV_E4M3);
        } else {
          const int v = max(-127, min(127, __float2int_rn(s * (32.f / 147.8f))));
          byte = (u32)(v & 255);
        }
        w |= byte << (8 * b);
      }
    }
    p[i] = w;
  }
}

static void fill(void* p, size_t bytes, u64 seed, int cls, int e4m3) {
  fill_k<<<4 * g_sms, 256>>>(reinterpret_cast<u32*>(p), bytes / 4, seed, cls, e4m3);
  CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------- tensor kernel
struct Co {
  unsigned long long mma, mma_in, t0_min, t1_max, clk, ns, saw_s, saw_e, warps, sink;
  u32 sm[8];
};

__device__ __forceinline__ void mma(int (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma(float (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Each warp runs NACC independent MMA chains over two A and two B fragments (consecutive MMAs differ in their operands)
// until the deadline, or, with `mark`, until it sees the GEMM's end marker; it counts its MMAs between the two markers. The
// markers are loaded at the top of a batch and read at its end, so the load's latency is under the batch's MMAs. With
// `sleep_ns` each warp sleeps that long after each batch of 32 x NACC MMAs (a paced co-runner).
template <class T, int NACC>
__global__ void __launch_bounds__(1024, 1)
    tensor_k(int uniform, u64 seed, u64 deadline, const volatile u64* mark, Co* co, u32 sleep_ns) {
  constexpr bool E4M3 = std::is_same<T, float>::value;
  u32 a[2][4], b[2][2];
  const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      const u32 v = uniform ? (u32)mix64(seed ^ (tid * 16 + i * 4 + j)) : 0u;
      a[i][j] = E4M3 ? no_nan(v) : v;
    }
    for (int j = 0; j < 2; ++j) {
      const u32 v = uniform ? (u32)mix64(~seed ^ (tid * 16 + 8 + i * 2 + j)) : 0u;
      b[i][j] = E4M3 ? no_nan(v) : v;
    }
  }
  T d[NACC][4];
  for (int i = 0; i < NACC; ++i)
    for (int j = 0; j < 4; ++j) d[i][j] = T(0);
  u64 n = 0, ns = ~0ull, ne = ~0ull;
  const u64 t0 = gtimer();
  const long long c0 = clock64();
  for (;;) {
    const u64 m0 = mark ? mark[0] : 0, m1 = mark ? mark[1] : 0;
#pragma unroll 1
    for (int r = 0; r < 32; ++r) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) mma(d[i], a[i & 1], b[(i >> 1) & 1]);
    }
    n += 32 * NACC;
    bool stop = false;
    if (mark) {
      // the counts at the batch's start: the markers were read there
      const bool s = __any_sync(0xffffffffu, m0 != 0), e = __any_sync(0xffffffffu, m1 != 0);
      if (ns == ~0ull && s) ns = n - 32 * NACC;
      if (ne == ~0ull && e) {
        ne = n - 32 * NACC;
        stop = true;
      }
    }
    if (__any_sync(0xffffffffu, gtimer() >= deadline)) stop = true;
    if (stop) break;
    if (sleep_ns) __nanosleep(sleep_ns);
  }
  const u64 t1 = gtimer();
  const long long c1 = clock64();
  T s = T(0);
  for (int i = 0; i < NACC; ++i)
    for (int j = 0; j < 4; ++j) s += d[i][j];
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&co->mma, (unsigned long long)n);
    atomicAdd(&co->warps, 1ull);
    if (ns != ~0ull) atomicAdd(&co->saw_s, 1ull);
    if (ns != ~0ull && ne != ~0ull) {
      atomicAdd(&co->saw_e, 1ull);
      atomicAdd(&co->mma_in, (unsigned long long)(ne - ns));
    }
    atomicMin(&co->t0_min, (unsigned long long)t0);
    atomicMax(&co->t1_max, (unsigned long long)t1);
    atomicAdd(&co->clk, (unsigned long long)(c1 - c0));
    atomicAdd(&co->ns, (unsigned long long)(t1 - t0));
    u32 smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    atomicOr(&co->sm[(smid / 32) % 8], 1u << (smid % 32));
  }
  if (s == T(77)) atomicAdd(&co->sink, 1ull);
}

__global__ void mark_k(u64* m, int i) { m[i] = gtimer(); }
__global__ void now_k(u64* m) { *m = gtimer(); }

using tensor_fn = void (*)(int, u64, u64, const volatile u64*, Co*, u32);
static tensor_fn tensor_kernel(bool e4m3, int nacc) {
  if (e4m3) return nacc == 4 ? tensor_k<float, 4> : tensor_k<float, 8>;
  return nacc == 4 ? tensor_k<int, 4> : tensor_k<int, 8>;
}

// ------------------------------------------------------------------------------------------------------------------ NVML
struct Sampler {
  nvmlDevice_t dev{};
  bool ok = false;
  std::atomic<bool> on{false}, quit{false};
  std::mutex mu;
  std::vector<u32> mhz, mw;
  unsigned long long reasons = 0;
  std::thread th;
  unsigned long long e0 = 0;
  std::chrono::steady_clock::time_point w0;

  void init(const cudaDeviceProp& p) {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    char uuid[96];
    const unsigned char* u = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
    snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1],
             u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    ok = nvmlDeviceGetHandleByUUID(uuid, &dev) == NVML_SUCCESS;
    if (!ok) return;
    th = std::thread([this] {
      while (!quit.load()) {
        if (on.load()) {
          unsigned int c = 0, w = 0;
          unsigned long long r = 0;
          nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c);
          nvmlDeviceGetPowerUsage(dev, &w);
          nvmlDeviceGetCurrentClocksEventReasons(dev, &r);
          std::lock_guard<std::mutex> g(mu);
          mhz.push_back(c);
          mw.push_back(w);
          reasons |= r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }
  void begin() {
    if (!ok) return;
    {
      std::lock_guard<std::mutex> g(mu);
      mhz.clear();
      mw.clear();
      reasons = 0;
    }
    nvmlDeviceGetTotalEnergyConsumption(dev, &e0);
    w0 = std::chrono::steady_clock::now();
    on = true;
  }
  // the pass's sensors as JSON members
  std::string end() {
    if (!ok) return "\"nvml\": null";
    on = false;
    unsigned long long e1 = 0;
    nvmlDeviceGetTotalEnergyConsumption(dev, &e1);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    unsigned int t = 0;
    nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t);
    std::lock_guard<std::mutex> g(mu);
    std::vector<u32> m = mhz;
    std::sort(m.begin(), m.end());
    double mean = 0, pw = 0;
    for (u32 x : mhz) mean += x;
    for (u32 x : mw) pw += x;
    const size_t k = m.size();
    char buf[512];
    snprintf(buf, sizeof buf,
             "\"nvml\": {\"samples\": %zu, \"sm_mhz_mean\": %.1f, \"sm_mhz_min\": %u, \"sm_mhz_median\": %u, "
             "\"sm_mhz_max\": %u, \"power_w_mean\": %.1f, \"energy_w\": %.1f, \"temp_c\": %u, \"reasons\": \"0x%llx\"}",
             k, k ? mean / k : 0.0, k ? m[0] : 0, k ? m[k / 2] : 0, k ? m[k - 1] : 0, k ? pw / k / 1000 : 0.0,
             secs > 0 ? (e1 - e0) / 1000.0 / secs : 0.0, t, reasons);
    return buf;
  }
  ~Sampler() {
    quit = true;
    if (th.joinable()) th.join();
  }
};
static Sampler g_nv;

// ----------------------------------------------------------------------------------------------------------- peak mode
struct CoRun {
  Co* co;
  u64* mark;
  u64* now;
  cudaStream_t s;
  void reset() {
    Co z{};
    z.t0_min = ~0ull;
    CK(cudaMemcpyAsync(co, &z, sizeof z, cudaMemcpyHostToDevice, s));
    CK(cudaMemsetAsync(mark, 0, 2 * sizeof(u64), s));
    CK(cudaStreamSynchronize(s));
  }
  u64 gpu_now() {
    now_k<<<1, 1, 0, s>>>(now);
    u64 t;
    CK(cudaMemcpyAsync(&t, now, sizeof t, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    return t;
  }
  Co read() {
    Co h;
    CK(cudaMemcpy(&h, co, sizeof h, cudaMemcpyDeviceToHost));
    return h;
  }
};

static int popcount_sm(const Co& c) {
  int n = 0;
  for (u32 w : c.sm) n += __builtin_popcount(w);
  return n;
}

static void set_carveout() {
  for (bool e : {false, true})
    for (int n : {4, 8}) {
      CK(cudaFuncSetAttribute(tensor_kernel(e, n), cudaFuncAttributePreferredSharedMemoryCarveout,
                              cudaSharedmemCarveoutMaxShared));
    }
}

static int run_peak(bool e4m3, int uniform, int warps, int nacc, double secs, u32 sleep_ns) {
  CoRun cr{};
  CK(cudaMalloc(&cr.co, sizeof(Co)));
  CK(cudaMalloc(&cr.mark, 2 * sizeof(u64)));
  CK(cudaMalloc(&cr.now, sizeof(u64)));
  CK(cudaStreamCreateWithFlags(&cr.s, cudaStreamNonBlocking));
  tensor_fn k = tensor_kernel(e4m3, nacc);
  // a short warm launch, then the pass
  for (int pass = 0; pass < 2; ++pass) {
    cr.reset();
    const double d = pass ? secs : 0.2;
    const u64 deadline = cr.gpu_now() + (u64)(d * 1e9) + 2000000;
    if (pass) g_nv.begin();
    k<<<g_sms, 32 * warps, 0, cr.s>>>(uniform, 0x5EED0000ull + pass, deadline, nullptr, cr.co, sleep_ns);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(cr.s));
    if (!pass) continue;
    const std::string nv = g_nv.end();
    const Co h = cr.read();
    const double win = (h.t1_max - h.t0_min) * 1e-9;
    const double macs = h.mma * MMA_MACS / win;
    const double mhz = h.ns ? double(h.clk) / double(h.ns) * 1e3 : 0;
    printf("{\"pass\": \"peak\", \"kind\": \"%s\", \"class\": \"%s\", \"warps\": %d, \"nacc\": %d, \"sleep_ns\": %u, "
           "\"secs\": %.3f, \"macs_per_s\": %.6e, \"sm_mhz_clock64\": %.1f, \"macs_per_sm_clk\": %.2f, \"sms\": %d, "
           "\"blocks_sms\": %d, %s}\n",
           e4m3 ? "e4m3" : "s8", uniform ? "uniform" : "zero", warps, nacc, sleep_ns, win, macs, mhz,
           mhz > 0 ? macs / (g_sms * mhz * 1e6) : 0.0, g_sms, popcount_sm(h), nv.c_str());
    fflush(stdout);
  }
  return 0;
}

// ----------------------------------------------------------------------------------------------------------- GEMM mode
struct Set {
  void *W, *X, *D;
};
using call_fn = std::function<bool(int set, cudaStream_t s)>;

// A graph of `calls` calls cycling over the sets, captured on s.
static cudaGraphExec_t capture(const call_fn& f, int nsets, int calls, cudaStream_t s) {
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
  bool ok = true;
  for (int i = 0; i < calls && ok; ++i) ok = f(i % nsets, s);
  CK(cudaStreamEndCapture(s, &g));
  if (!ok) {
    cudaGraphDestroy(g);
    return nullptr;
  }
  cudaGraphExec_t e;
  if (cudaGraphInstantiate(&e, g, 0) != cudaSuccess) {
    cudaGraphDestroy(g);
    cudaGetLastError();
    return nullptr;
  }
  cudaGraphDestroy(g);
  return e;
}

// seconds a call: the graph launched until `target` seconds have passed (event timing)
static double time_graph(cudaGraphExec_t e, int calls, cudaStream_t s, double target) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaGraphLaunch(e, s));  // warm
  CK(cudaStreamSynchronize(s));
  CK(cudaEventRecord(a, s));
  CK(cudaGraphLaunch(e, s));
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  const int reps = std::max(1, std::min(1000, int(std::ceil(target / std::max(1e-6, ms * 1e-3)))));
  CK(cudaEventRecord(a, s));
  for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(e, s));
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  CK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms * 1e-3 / (double(reps) * calls);
}

struct Candidate {
  std::string name;
  call_fn f;
  double t = 0;  // seconds a call
};

// cuBLASLt: our D (M x N row-major) is its column-major N x M = op(A) . op(B) with A = W (K x N, op T), B = X (K x M).
struct LtPlan {
  cublasLtHandle_t h{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulPreference_t pref{};
  void* ws = nullptr;
  size_t wsb = 64 << 20;
  float* scale = nullptr;
};

static void lt_candidates(bool e4m3, int M, int N, int K, std::vector<Set>& sets, std::vector<Candidate>& out, LtPlan& p) {
  LTC(cublasLtCreate(&p.h));
  const cudaDataType_t in = e4m3 ? CUDA_R_8F_E4M3 : CUDA_R_8I, ot = e4m3 ? CUDA_R_16BF : CUDA_R_32I;
  LTC(cublasLtMatmulDescCreate(&p.op, e4m3 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32I, e4m3 ? CUDA_R_32F : CUDA_R_32I));
  const cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
  LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn));
  if (e4m3) {
    CK(cudaMalloc(&p.scale, sizeof(float)));
    const float one = 1.f;
    CK(cudaMemcpy(p.scale, &one, sizeof one, cudaMemcpyHostToDevice));
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p.scale, sizeof p.scale));
    LTC(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p.scale, sizeof p.scale));
  }
  LTC(cublasLtMatrixLayoutCreate(&p.a, in, K, N, K));
  LTC(cublasLtMatrixLayoutCreate(&p.b, in, K, M, K));
  LTC(cublasLtMatrixLayoutCreate(&p.c, ot, N, M, N));
  LTC(cublasLtMatmulPreferenceCreate(&p.pref));
  LTC(cublasLtMatmulPreferenceSetAttribute(p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &p.wsb, sizeof p.wsb));
  CK(cudaMalloc(&p.ws, p.wsb));
  cublasLtMatmulHeuristicResult_t res[16];
  int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(p.h, p.op, p.a, p.b, p.c, p.c, p.pref, 16, res, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  for (int i = 0; i < got; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    const cublasLtMatmulAlgo_t algo = res[i].algo;
    LtPlan* pp = &p;
    Set* ss = sets.data();
    call_fn f = [pp, ss, algo, e4m3](int i, cudaStream_t s) {
      static const int32_t one_i = 1, zero_i = 0;
      static const float one_f = 1.f, zero_f = 0.f;
      const void* al = e4m3 ? (const void*)&one_f : (const void*)&one_i;
      const void* be = e4m3 ? (const void*)&zero_f : (const void*)&zero_i;
      return cublasLtMatmul(pp->h, pp->op, al, ss[i].W, pp->a, ss[i].X, pp->b, be, ss[i].D, pp->c, ss[i].D, pp->c, &algo,
                            pp->ws, pp->wsb, s) == CUBLAS_STATUS_SUCCESS;
    };
    out.push_back({"lt#" + std::to_string(i), f});
  }
}

#if defined(WITH_C2X) || defined(WITH_C3X)
struct CutlassHold {
  std::vector<std::vector<phc_prepared*>> prepared;
  void* ws = nullptr;
};

static void cutlass_candidates(const phc_entry* tab, int count, int kind, int M, int N, int K, std::vector<Set>& sets,
                               std::vector<Candidate>& out, CutlassHold& hold, cudaStream_t s) {
  for (int e = 0; e < count; ++e) {
    const phc_entry& en = tab[e];
    if (en.kind != kind || !en.query) continue;
    phc_args q{};
    q.m = M;
    q.n = N;
    q.k = K;
    size_t ws = 0, sfa = 0, sfb = 0;
    if (en.query(&q, &ws, &sfa, &sfb) != 0) {
      fprintf(stderr, "%s: not implementable at %d %d %d\n", en.name, M, N, K);
      continue;
    }
    void* w = nullptr;
    if (ws) CK(cudaMalloc(&w, ws));
    std::vector<phc_prepared*> per;
    bool ok = true;
    for (auto& st : sets) {
      phc_args a = q;
      a.A = st.X;
      a.B = st.W;
      a.D = st.D;
      phc_prepared* pr = nullptr;
      // a stream-K kernel's workspace is its own each call; one workspace serves every set (calls are serialized)
      if (en.prepare(&a, w, ws, s, &pr) != 0) {
        ok = false;
        break;
      }
      per.push_back(pr);
    }
    if (!ok) {
      fprintf(stderr, "%s: prepare failed\n", en.name);
      continue;
    }
    hold.prepared.push_back(per);
    const size_t idx = hold.prepared.size() - 1;
    CutlassHold* hp = &hold;
    out.push_back({en.name, [hp, idx](int i, cudaStream_t s) { return hp->prepared[idx][i]->launch(s) == 0; }});
  }
}
#endif

// host e4m3 decode
static double e4m3_val(uint8_t c) {
  const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
  return s ? -v : v;
}
static double bf16_val(uint16_t h) {
  u32 w = u32(h) << 16;
  float f;
  memcpy(&f, &w, 4);
  return f;
}

// sampled entries of set 0's D against the host's
static std::string check(bool e4m3, int M, int N, int K, const Set& s0) {
  std::vector<uint8_t> hW((size_t)N * K), hX((size_t)M * K);
  CK(cudaMemcpy(hW.data(), s0.W, hW.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hX.data(), s0.X, hX.size(), cudaMemcpyDeviceToHost));
  std::mt19937 rng(11);
  int bad = 0;
  double worst = 0;
  for (int t = 0; t < 24; ++t) {
    const int r = rng() % M, c = rng() % N;
    if (e4m3) {
      double want = 0, mag = 0;
      for (int k = 0; k < K; ++k) {
        const double p = e4m3_val(hX[(size_t)r * K + k]) * e4m3_val(hW[(size_t)c * K + k]);
        want += p;
        mag += std::fabs(p);
      }
      uint16_t g;
      CK(cudaMemcpy(&g, reinterpret_cast<uint16_t*>(s0.D) + (size_t)r * N + c, 2, cudaMemcpyDeviceToHost));
      const double err = std::fabs(bf16_val(g) - want), tol = std::fabs(want) / 128 + mag * 1e-5 + 1e-6;
      worst = std::max(worst, err / tol);
      bad += err > tol;
    } else {
      int64_t want = 0;
      for (int k = 0; k < K; ++k) want += int64_t(int8_t(hX[(size_t)r * K + k])) * int8_t(hW[(size_t)c * K + k]);
      int32_t g;
      CK(cudaMemcpy(&g, reinterpret_cast<int32_t*>(s0.D) + (size_t)r * N + c, 4, cudaMemcpyDeviceToHost));
      bad += int64_t(g) != want;
    }
  }
  char buf[128];
  snprintf(buf, sizeof buf, "\"check\": {\"entries\": 24, \"bad\": %d, \"worst_over_tol\": %.3f}", bad, worst);
  return buf;
}

static int run_gemm(const std::string& lib, bool e4m3, int M, int N, int K, double secs, int wcls,
                    const std::vector<std::pair<int, u32>>& co) {
  const size_t wb = (size_t)N * K, xb = (size_t)M * K, db = (size_t)M * N * (e4m3 ? 2 : 4);
  const size_t ring_bytes = 3 * g_l2;
  int R = int(std::max<size_t>(2, (ring_bytes + wb + xb - 1) / (wb + xb)));
  R = std::min(R, 256);
  std::vector<Set> sets(R);
  for (int i = 0; i < R; ++i) {
    CK(cudaMalloc(&sets[i].W, wb));
    CK(cudaMalloc(&sets[i].X, std::max<size_t>(xb, 16)));
    CK(cudaMalloc(&sets[i].D, db));
    fill(sets[i].W, wb, 0xA11CE000ull + i, wcls, e4m3);
    fill(sets[i].X, xb, 0xB0B00000ull + i, CLS_UNIFORM, e4m3);
    CK(cudaMemset(sets[i].D, 0, db));
  }
  CK(cudaDeviceSynchronize());
  int prio_lo, prio_hi;
  CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, prio_hi));
  CK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, prio_lo));

  std::vector<Candidate> cands;
  LtPlan lt;
#if defined(WITH_C2X) || defined(WITH_C3X)
  CutlassHold hold;
#endif
  if (lib == "lt") {
    lt_candidates(e4m3, M, N, K, sets, cands, lt);
#ifdef WITH_C2X
  } else if (lib == "c2x" && !e4m3) {
    int n;
    const phc_entry* t = phc_table_2x(&n);
    cutlass_candidates(t, n, 7, M, N, K, sets, cands, hold, s1);
#endif
#ifdef WITH_C3X
  } else if (lib == "c3x" && e4m3) {
    int n;
    const phc_entry* t = phc_table_3x_0(&n);
    cutlass_candidates(t, n, 1, M, N, K, sets, cands, hold, s1);
#endif
  }
  const char* kn = e4m3 ? "e4m3" : "s8";
  const char* wn = wcls == CLS_GAUSS ? "gauss" : wcls == CLS_ZERO ? "zero" : "uniform";
  const double mnk = double(M) * N * K;
  char head[384];
  snprintf(head, sizeof head,
           "\"lib\": \"%s\", \"kind\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"wclass\": \"%s\", \"ring\": %d, "
           "\"ring_mb\": %.1f",
           lib.c_str(), kn, M, N, K, wn, R, R * (wb + xb + db) / 1048576.0);
  if (cands.empty()) {
    printf("{\"pass\": \"tune\", %s, \"error\": \"no candidate\"}\n", head);
    fflush(stdout);
    return 0;
  }
  // a graph is whole ring cycles: about 4 ms of GEMMs, at most 64 calls but at least one cycle, from the first candidate
  // that runs, called directly
  double t0 = 0;
  for (auto& c : cands) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    bool ok = true;
    for (int i = 0; i < R && ok; ++i) ok = c.f(i, s1);
    CK(cudaEventRecord(a, s1));
    for (int i = 0; i < R && ok; ++i) ok = c.f(i, s1);
    CK(cudaEventRecord(b, s1));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGetLastError();
    if (ok) {
      t0 = ms * 1e-3 / R;
      break;
    }
  }
  const int cycles = t0 > 0 ? std::max(1, std::min((64 + R - 1) / R, int(std::ceil(0.004 / (R * t0))))) : 1;
  const int calls = R * cycles;
  std::string tl;
  int best = -1;
  std::vector<cudaGraphExec_t> graphs(cands.size(), nullptr);
  for (size_t i = 0; i < cands.size(); ++i) {
    graphs[i] = capture(cands[i].f, R, calls, s1);
    if (!graphs[i]) {
      cudaGetLastError();
      continue;
    }
    cands[i].t = time_graph(graphs[i], calls, s1, 0.06);
    char b[160];
    snprintf(b, sizeof b, "%s{\"name\": \"%s\", \"macs_per_s\": %.6e}", tl.empty() ? "" : ", ", cands[i].name.c_str(),
             mnk / cands[i].t);
    tl += b;
    if (best < 0 || cands[i].t < cands[best].t) best = int(i);
  }
  if (best < 0) {
    printf("{\"pass\": \"tune\", %s, \"error\": \"no candidate ran\"}\n", head);
    fflush(stdout);
    return 0;
  }
  printf("{\"pass\": \"tune\", %s, \"best\": \"%s\", \"candidates\": [%s]}\n", head, cands[best].name.c_str(), tl.c_str());
  fflush(stdout);
  cudaGraphExec_t g = graphs[best];
  CK(cudaGraphLaunch(g, s1));
  CK(cudaStreamSynchronize(s1));
  const std::string chk = check(e4m3, M, N, K, sets[0]);

  CoRun cr{};
  CK(cudaMalloc(&cr.co, sizeof(Co)));
  CK(cudaMalloc(&cr.mark, 2 * sizeof(u64)));
  CK(cudaMalloc(&cr.now, sizeof(u64)));
  cr.s = s1;
  std::vector<int> order;
  for (size_t i = 0; i < cands.size(); ++i)
    if (graphs[i]) order.push_back(int(i));
  std::sort(order.begin(), order.end(), [&](int a, int b) { return cands[a].t < cands[b].t; });
  struct Win {
    double secs, rate;
    Co co;
    bool beside_all;
  };
  // one window of `reps` launches of candidate c's graph, beside the co-runner (w warps an SM, sleeping sl ns a batch,
  // until co_secs from its launch at the latest) when w > 0
  auto window = [&](int c, int reps, int w, u32 sl, double co_secs) {
    cr.reset();
    if (w) {
      const u64 deadline = cr.gpu_now() + (u64)(co_secs * 1e9);
      tensor_kernel(e4m3, 8)<<<g_sms, 32 * w, 0, s2>>>(1, 0xC0C0ull + w, deadline, cr.mark, cr.co, sl);
      CK(cudaGetLastError());
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    mark_k<<<1, 1, 0, s1>>>(cr.mark, 0);
    for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(graphs[c], s1));
    mark_k<<<1, 1, 0, s1>>>(cr.mark, 1);
    CK(cudaStreamSynchronize(s1));
    CK(cudaStreamSynchronize(s2));
    u64 mk[2];
    CK(cudaMemcpy(mk, cr.mark, sizeof mk, cudaMemcpyDeviceToHost));
    Win r;
    r.secs = (mk[1] - mk[0]) * 1e-9;
    r.rate = double(reps) * calls * mnk / r.secs;
    r.co = w ? cr.read() : Co{};
    r.beside_all = w && r.co.warps > 0 && r.co.saw_e == r.co.warps;
    return r;
  };
  std::vector<std::pair<int, u32>> passes = {{0, 0u}};
  for (auto& e : co)
    if (e.first > 0) passes.push_back(e);
  if (passes.size() > 1) passes.push_back({0, 0u});
  double alone = 0;
  for (auto [w, sl] : passes) {
    const bool after = !w && alone > 0;
    int c = best, tried = 0;
    if (w) {
      // the fastest candidate the co-runner can sit beside (a kernel whose CTA takes all of an SM's shared memory or
      // registers can't): one graph launch beside it, which finishes before the co-runner's deadline only if both fit
      c = -1;
      for (int i : order) {
        if (tried == 4) break;
        ++tried;
        if (window(i, 1, w, 0, 0.15 + 8 * cands[i].t * calls).beside_all) {
          c = i;
          break;
        }
      }
      if (c < 0) {
        printf("{\"pass\": \"beside\", %s, \"co_warps\": %d, \"co_sleep_ns\": %u, \"co_resident\": false, \"tried\": %d}\n",
               head, w, sl, tried);
        fflush(stdout);
        continue;
      }
    }
    const int reps = std::max(1, int(std::ceil(secs / (cands[c].t * calls))));
    g_nv.begin();
    // the co-runner quits at the GEMM's end marker, so a long deadline costs time only when the GEMM slows past it
    const Win r = window(c, reps, w, sl, w ? 6.0 * reps * calls * cands[c].t + 0.3 : 0);
    const std::string nv = g_nv.end();
    if (!w && !after) alone = r.rate;
    std::string cos = "\"co_warps\": 0";
    if (w) {
      const Co& h = r.co;
      const double mhz = h.ns ? double(h.clk) / double(h.ns) * 1e3 : 0;
      char b[1024];
      snprintf(b, sizeof b,
               "\"co_warps\": %d, \"co_sleep_ns\": %u, \"co_resident\": true, \"co_outlasted_gemm\": %s, \"tried\": %d, "
               "\"co_macs_per_s\": %.6e, \"co_warps_ran\": %llu, \"co_saw_start\": %llu, \"co_saw_end\": %llu, "
               "\"co_sms\": %d, \"co_sm_mhz_clock64\": %.1f, \"co_secs\": %.3f, \"tuned_macs_per_s\": %.6e, "
               "\"gemm_vs_alone\": %.4f",
               w, sl, r.beside_all ? "true" : "false", tried, h.mma_in * MMA_MACS / r.secs, h.warps, h.saw_s, h.saw_e,
               popcount_sm(h), mhz, (h.t1_max - h.t0_min) * 1e-9, mnk / cands[c].t, alone > 0 ? r.rate / alone : 0.0);
      cos = b;
    }
    printf("{\"pass\": \"%s\", %s, \"algo\": \"%s\", \"calls\": %lld, \"secs\": %.4f, \"macs_per_s\": %.6e, %s, %s, %s}\n",
           w ? "beside" : after ? "alone_after" : "alone", head, cands[c].name.c_str(), (long long)reps * calls, r.secs,
           r.rate, cos.c_str(), chk.c_str(), nv.c_str());
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: gemm_idle info | peak KIND CLASS WARPS NACC SECS [SLEEP_NS] | "
                    "gemm LIB KIND M N K SECS WCLASS W[:SLEEP_NS]...\n");
    return 2;
  }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  g_l2 = p.l2CacheSize;
  const std::string mode = argv[1];
  if (mode == "info") {
    int smem_sm = 0;
    CK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    printf("{\"name\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"l2_bytes\": %zu, \"smem_per_block_optin\": %zu, "
           "\"smem_per_sm\": %d, \"regs_per_sm\": %d, \"max_threads_per_sm\": %d, \"global_mem\": %zu, "
           "\"cublaslt\": %zu}\n",
           p.name, p.major, p.minor, g_sms, g_l2, p.sharedMemPerBlockOptin, smem_sm, p.regsPerMultiprocessor,
           p.maxThreadsPerMultiProcessor, p.totalGlobalMem, cublasLtGetVersion());
    return 0;
  }
  g_nv.init(p);
  set_carveout();
  if (mode == "peak" && (argc == 7 || argc == 8)) {
    return run_peak(std::string(argv[2]) == "e4m3", std::string(argv[3]) == "uniform", atoi(argv[4]), atoi(argv[5]),
                    atof(argv[6]), argc == 8 ? u32(atoi(argv[7])) : 0u);
  }
  if (mode == "gemm" && argc >= 9) {
    const std::string wc = argv[8];
    std::vector<std::pair<int, u32>> co;  // W[:SLEEP_NS]
    for (int i = 9; i < argc; ++i) {
      const char* colon = strchr(argv[i], ':');
      co.push_back({atoi(argv[i]), colon ? u32(atoi(colon + 1)) : 0u});
    }
    return run_gemm(argv[2], std::string(argv[3]) == "e4m3", atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atof(argv[7]),
                    wc == "gauss" ? CLS_GAUSS : wc == "zero" ? CLS_ZERO : CLS_UNIFORM, co);
  }
  fprintf(stderr, "bad arguments\n");
  return 2;
}
