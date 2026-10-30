// The H200's dense tensor MAC rate over the host's wall clock by tensor duty cycle: whether any mix of tensor work with
// idle or non-tensor work moves more MACs a second than 132 SMs x 4096 MACs a clock at 1830 MHz (9.894e14), whatever
// the SM clock is locked to. Timings only.
//
// One CTA an SM. Each period (SM-cycles of clock64, counted from the CTA's start) issues a burst of whole batches of
// tensor operations, BURST cycles of work at the rate the arm sustains continuously (its calibration, MACs a SM-cycle),
// drains them, and fills the rest of the period, whose length makes the burst DUTY percent of it; DUTY 100 issues back
// to back with no drain or fill. Periods run until SECS seconds of %globaltimer have passed, checked about every
// 65,536 cycles.
// Arms:    wg_s8    wgmma m64n256k32 s8.s8 -> s32, one warpgroup with A and B resident in shared memory (KS k-slices,
//                   cycled), as int8_class_rate's wgmma kernel;
//          wg_e4m3  the same, e4m3.e4m3 -> f32, each batch drained;
//          sync_s8  mma.sync m16n8k32 s8.s8 -> s32, SW warps with SA accumulators each, operands in registers (KS sets).
// Classes: zero; const (every byte 0x7F, e4m3 0x38 = 1.0); uniform (bytes uniform: s8 on [-128, 127], or every finite
//          e4m3 encoding); s2, u2, u4 (Verity.OperandClass: uniform on [-2, 1], [0, 3], [0, 15], as s8). Each byte is
//          drawn from a hash of its position, so consecutive operations read different operands.
// Fills:   idle (__nanosleep until the period ends) or alu (four FFMA chains a thread until it ends).
// Readings, per point: MACs over the host's wall clock (cudaEvents around the one launch); clock64 over %globaltimer
// for the whole kernel and for its tensor phases alone (warp 0's issue to its drain); MACs a SM-cycle.
//
// Build: python3 gen_wgmma.py && nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -I. -I../int8_class_rate \
//          -o duty duty.cu
// Run:   duty ARM CLASS FILL BURST SECS DUTY...    (BURST in SM-cycles, DUTY in percent)
// One JSON line for the calibration, then one a point, on stdout.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "wgmma_e4m3.cuh"
#include "wgmma_i8.cuh"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

using u64 = unsigned long long;
using u32 = uint32_t;

// 132 SMs x 4096 dense MACs a clock at 1830 MHz
constexpr double BOUND = 132.0 * 4096 * 1830e6;
constexpr u64 CHECK_CYCLES = 65536;

struct Stamp {
  u64 t0, t1, c0, c1, ops, tns, tcyc;
  u32 sm, pad;
};

enum Cls { ZERO = 0, CONST = 1, UNIFORM = 2, S2 = 3, U2 = 4, U4 = 5 };
static const char* CLS_NAMES[] = {"zero", "const", "uniform", "s2", "u2", "u4"};

__host__ __device__ inline u32 mix(u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// four operand bytes of class `cls` at word i, e4m3 encodings where `fp8`
__device__ inline u32 word(u32 i, int cls, bool fp8) {
  const u32 x = mix(i * 2654435761u + 0x9e3779b9u);
  u32 w = 0;
  for (int j = 0; j < 4; ++j) {
    const u32 b = (x >> (8 * j)) & 0xffu;
    u32 v;
    if (cls == ZERO) v = 0;
    else if (cls == CONST) v = fp8 ? 0x38u : 0x7fu;
    else if (fp8) v = (b & 0x7fu) == 0x7fu ? b - 1 : b;  // 0x7F and 0xFF are e4m3's NaNs
    else if (cls == UNIFORM) v = b;
    else if (cls == S2) v = (u32)((int)(b & 3u) - 2) & 0xffu;
    else if (cls == U2) v = b & 3u;
    else v = b & 15u;
    w |= v << (8 * j);
  }
  return w;
}

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ u32 smid() {
  u32 s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ u64 clk() { return (u64)clock64(); }

enum Fill { IDLE = 0, ALU = 1 };

template <int FILL>
__device__ __forceinline__ void fill_until(u64 end, float* f) {
  if constexpr (FILL == IDLE) {
    while (clk() < end) __nanosleep(64);
  } else {
    while (clk() < end) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) f[j] = fmaf(f[j], 0.9999f, 1e-4f);
      }
    }
  }
}

__device__ __forceinline__ void stop_check(int* s_stop, u64 tend, bool& stop) {
  __syncthreads();
  if (threadIdx.x == 0) *s_stop = gtimer() >= tend;
  __syncthreads();
  stop = *s_stop;
}

__device__ __forceinline__ uint64_t desc(const void* p, u32 lbo, u32 sbo) {
  const u32 a = static_cast<u32>(__cvta_generic_to_shared(p));
  return (uint64_t)((a & 0x3FFFF) >> 4) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 | (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
}

constexpr int KS = 4;
constexpr int N = 256;
constexpr double WG_MACS = 64.0 * N * 32;
// e4m3 drains every batch (wait_group 0): under wait_group 1 ptxas serializes its f32-accumulator wgmmas (C7511), so
// its batch is FP8_REP passes over the KS k-slices, which amortizes the drain
constexpr int FP8_REP = 8;
template <bool FP8>
constexpr int WB = FP8 ? FP8_REP * KS : KS;

template <typename T>
__device__ __forceinline__ void pass(T* d) {
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    if constexpr (std::is_same_v<T, float>) asm volatile("" : "+f"(d[i])::"memory");
    else asm volatile("" : "+r"(d[i])::"memory");
  }
}

// nb batches of WB wgmmas a period (one warpgroup); cont: back to back, no drain or fill. Every batch is whole: a
// wgmma issued under a condition gets its own warpgroup.arrive, which serializes the pipe.
template <bool FP8, int FILL>
__global__ void __launch_bounds__(128, 1) wg_k(int cls, u32 nb, int cont, u64 period, u64 ns, Stamp* out, u32* sink) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ u64 s_base, s_t0;
  __shared__ int s_stop;
  using T = std::conditional_t<FP8, float, u32>;
  constexpr int ABYTES = 64 * 32, BBYTES = N * 32;
  uint8_t* A = smem;
  uint8_t* B = smem + KS * ABYTES;
  const u32 seed = blockIdx.x * 0x10000u;
  for (int i = threadIdx.x; i < KS * (ABYTES + BBYTES) / 4; i += 128) reinterpret_cast<u32*>(smem)[i] = word(seed + i, cls, FP8);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  uint64_t da[KS], db[KS];
#pragma unroll
  for (int k = 0; k < KS; ++k) { da[k] = desc(A + k * ABYTES, 128, 256); db[k] = desc(B + k * BBYTES, 128, 256); }
  T d[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) d[i] = 0;
  float f[4] = {1.f, 2.f, 3.f, 4.f};
  if (threadIdx.x == 0) { s_t0 = gtimer(); s_base = clk(); }
  __syncthreads();
  const u64 t0 = s_t0, base = s_base, tend = t0 + ns;
  const u64 check = cont || period >= CHECK_CYCLES ? 1 : CHECK_CYCLES / period;
  u64 ops = 0, tns = 0, tcyc = 0;
  for (u64 k = 1;; ++k) {
    u64 ta = 0, ca = 0;
    if (threadIdx.x == 0 && !cont) { ta = gtimer(); ca = clk(); }
    for (u32 j = 0; j < nb; ++j) {
      pass(d);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int q = 0; q < WB<FP8>; ++q) {
        if constexpr (FP8) wgmma_e4m3_m64n256k32(d, da[q % KS], db[q % KS]);
        else wgmma_s8s8_m64n256k32(d, da[q % KS], db[q % KS]);
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      if constexpr (FP8) asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      pass(d);
    }
    ops += nb * WB<FP8>;
    if (!cont) {
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      pass(d);
      if (threadIdx.x == 0) { tns += gtimer() - ta; tcyc += clk() - ca; }
      fill_until<FILL>(base + k * period, f);
    }
    if (k % check == 0) {
      bool stop;
      stop_check(&s_stop, tend, stop);
      if (stop) break;
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  u32 x = __float_as_uint(f[0] + f[1] + f[2] + f[3]);
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    if constexpr (FP8) x ^= __float_as_uint(d[i]);
    else x ^= d[i];
  }
  if (x == 0x9e3779b9u) sink[blockIdx.x * 128 + threadIdx.x] = x;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{t0, gtimer(), base, clk(), ops, tns, tcyc, smid(), 0};
}

constexpr int SW = 8;  // warps a CTA
constexpr int SA = 8;  // accumulators a warp
constexpr double SYNC_MACS = 16.0 * 8 * 32;

__device__ __forceinline__ void mma_s8(u32* c, const u32* a, const u32* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// nb batches of SA mma.syncs a warp a period; cont: back to back, no fill
template <int FILL>
__global__ void __launch_bounds__(SW * 32, 1) sync_k(int cls, u32 nb, int cont, u64 period, u64 ns, Stamp* out, u32* sink) {
  __shared__ u64 s_base, s_t0;
  __shared__ int s_stop;
  const u32 seed = (blockIdx.x * SW * 32 + threadIdx.x) * 32;
  u32 a[KS][4], b[KS][2], c[SA][4];
#pragma unroll
  for (int q = 0; q < KS; ++q) {
#pragma unroll
    for (int i = 0; i < 4; ++i) a[q][i] = word(seed + q * 6 + i, cls, false);
#pragma unroll
    for (int i = 0; i < 2; ++i) b[q][i] = word(seed + q * 6 + 4 + i, cls, false);
  }
#pragma unroll
  for (int u = 0; u < SA; ++u)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[u][i] = 0;
  float f[4] = {1.f, 2.f, 3.f, 4.f};
  if (threadIdx.x == 0) { s_t0 = gtimer(); s_base = clk(); }
  __syncthreads();
  const u64 t0 = s_t0, base = s_base, tend = t0 + ns;
  const u64 check = cont || period >= CHECK_CYCLES ? 1 : CHECK_CYCLES / period;
  u64 ops = 0, tns = 0, tcyc = 0;
  for (u64 k = 1;; ++k) {
    u64 ta = 0, ca = 0;
    if (threadIdx.x == 0 && !cont) { ta = gtimer(); ca = clk(); }
    for (u32 j = 0; j < nb; ++j) {
#pragma unroll
      for (int u = 0; u < SA; ++u) mma_s8(c[u], a[u % KS], b[u % KS]);
    }
    ops += nb * SA;
    if (!cont) {
      if (threadIdx.x == 0) {
        u32 y = 0;  // warp 0's last results, so its tensor phase ends when they land
#pragma unroll
        for (int u = 0; u < SA; ++u) y ^= c[u][0];
        if (y == 0x9e3779b9u) sink[0] = y;
        tns += gtimer() - ta;
        tcyc += clk() - ca;
      }
      fill_until<FILL>(base + k * period, f);
    }
    if (k % check == 0) {
      bool stop;
      stop_check(&s_stop, tend, stop);
      if (stop) break;
    }
  }
  u32 x = __float_as_uint(f[0] + f[1] + f[2] + f[3]);
#pragma unroll
  for (int u = 0; u < SA; ++u)
#pragma unroll
    for (int i = 0; i < 4; ++i) x ^= c[u][i];
  if (x == 0x9e3779b9u) sink[blockIdx.x * SW * 32 + threadIdx.x] = x;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{t0, gtimer(), base, clk(), ops, tns, tcyc, smid(), 0};
}

using KernelFn = void (*)(int, u32, int, u64, u64, Stamp*, u32*);

struct Arm {
  KernelFn k;
  int threads, units;  // units: the CTA's issuers, each running nb batches a period
  size_t smem;
  double macs;  // a unit's operation
  int batch;    // a unit's operations a batch
};

static Arm arm_for(const std::string& name, int fill) {
  const size_t wsmem = (size_t)KS * (64 * 32 + N * 32);
  if (name == "wg_s8") return {fill ? wg_k<false, ALU> : wg_k<false, IDLE>, 128, 1, wsmem, WG_MACS, WB<false>};
  if (name == "wg_e4m3") return {fill ? wg_k<true, ALU> : wg_k<true, IDLE>, 128, 1, wsmem, WG_MACS, WB<true>};
  if (name == "sync_s8") return {fill ? sync_k<ALU> : sync_k<IDLE>, SW * 32, SW, 0, SYNC_MACS, SA};
  fprintf(stderr, "ARM: wg_s8, wg_e4m3 or sync_s8\n");
  exit(2);
}

static double unix_now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct Point {
  double host_t0, host_t1, event_s, macs, cyc, ns, tcyc, tns, gt_s;
};

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "duty ARM CLASS FILL BURST SECS DUTY...\n");
    return 2;
  }
  const std::string aname = argv[1], cname = argv[2], fname = argv[3];
  int cls = -1;
  for (int c = 0; c < 6; ++c)
    if (cname == CLS_NAMES[c]) cls = c;
  if (cls < 0 || (aname == "wg_e4m3" && cls > UNIFORM)) {
    fprintf(stderr, "CLASS: zero, const, uniform, s2, u2 or u4 (wg_e4m3: zero, const or uniform)\n");
    return 2;
  }
  if (fname != "idle" && fname != "alu") {
    fprintf(stderr, "FILL: idle or alu\n");
    return 2;
  }
  const int fill = fname == "alu" ? ALU : IDLE;
  const u64 burst = strtoull(argv[4], nullptr, 10);
  const double secs = atof(argv[5]);
  std::vector<double> duties;
  for (int i = 6; i < argc; ++i) duties.push_back(atof(argv[i]));
  const Arm arm = arm_for(aname, fill);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const int nsm = p.multiProcessorCount;
  if (arm.smem) CK(cudaFuncSetAttribute(arm.k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arm.smem));
  Stamp* sd;
  u32* sink;
  CK(cudaMalloc(&sd, nsm * sizeof(Stamp)));
  CK(cudaMalloc(&sink, (size_t)nsm * arm.threads * 4));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  auto run = [&](u32 nb, int cont, u64 period, double s) {
    Point pt{};
    pt.host_t0 = unix_now();
    CK(cudaEventRecord(e0));
    arm.k<<<nsm, arm.threads, arm.smem>>>(cls, nb, cont, period, (u64)(s * 1e9), sd, sink);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    pt.host_t1 = unix_now();
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    pt.event_s = ms * 1e-3;
    std::vector<Stamp> st(nsm);
    CK(cudaMemcpy(st.data(), sd, st.size() * sizeof(Stamp), cudaMemcpyDeviceToHost));
    u64 a = ~0ull, b = 0;
    for (auto& x : st) {
      pt.macs += (double)x.ops * arm.units * arm.macs;
      pt.cyc += (double)(x.c1 - x.c0);
      pt.ns += (double)(x.t1 - x.t0);
      pt.tcyc += (double)x.tcyc;
      pt.tns += (double)x.tns;
      a = std::min(a, x.t0);
      b = std::max(b, x.t1);
    }
    pt.gt_s = (b - a) * 1e-9;
    return pt;
  };

  // a CTA's batch, in MACs; continuous runs (the calibration, and duty 100) check for the end about every
  // max(BURST, CHECK_CYCLES) cycles, at the full rate's guess
  const double batch_macs = arm.units * arm.macs * arm.batch;
  const double guess = aname == "sync_s8" ? 2048.0 : 4096.0;
  const u32 cont_nb =
      std::max<u32>(1, (u32)std::lround(guess * (double)std::max<u64>(burst, CHECK_CYCLES) / batch_macs));
  run(cont_nb, 1, 0, 0.05);
  const Point cal = run(cont_nb, 1, 0, 0.3);
  const double r100 = cal.macs / cal.cyc;  // MACs a SM-cycle, continuous
  const u32 nb = std::max<u32>(1, (u32)std::lround(burst * r100 / batch_macs));
  printf("{\"arm\": \"%s\", \"class\": \"%s\", \"fill\": \"%s\", \"burst\": %llu, \"calibrate\": true, \"sms\": %d, "
         "\"nb\": %u, \"macs_per_smclk\": %.2f, \"clock64_mhz\": %.2f, \"tmacs_per_s\": %.3f}\n",
         aname.c_str(), cname.c_str(), fname.c_str(), burst, nsm, nb, r100, cal.cyc / cal.ns * 1e3,
         cal.macs / cal.event_s * 1e-12);
  fflush(stdout);

  for (double duty : duties) {
    const bool cont = duty >= 100;
    const u64 period = cont ? 0 : (u64)std::llround(nb * batch_macs / (duty / 100 * r100));
    const Point pt = run(cont ? cont_nb : nb, cont, period, secs);
    const double rate = pt.macs / pt.event_s;
    const double fcyc = pt.cyc - pt.tcyc, fns = pt.ns - pt.tns;
    printf("{\"arm\": \"%s\", \"class\": \"%s\", \"fill\": \"%s\", \"burst\": %llu, \"duty\": %.0f, \"nb\": %u, "
           "\"period\": %llu, \"cont\": %s, \"secs\": %.2f, \"host_t0\": %.3f, \"host_t1\": %.3f, \"event_s\": %.6f, "
           "\"gt_s\": %.6f, \"macs\": %.6e, \"tmacs_per_s\": %.4f, \"mhz_equiv\": %.2f, \"over_bound\": %s, "
           "\"clock64_mhz\": %.2f, \"tensor_mhz\": %.2f, \"fill_mhz\": %.2f, \"macs_per_smclk\": %.2f, "
           "\"duty_achieved\": %.4f}\n",
           aname.c_str(), cname.c_str(), fname.c_str(), burst, duty, cont ? cont_nb : nb, period, cont ? "true" : "false",
           secs, pt.host_t0, pt.host_t1, pt.event_s, pt.gt_s, pt.macs, rate * 1e-12, rate / (nsm * 4096.0) * 1e-6,
           rate > BOUND ? "true" : "false", pt.cyc / pt.ns * 1e3,
           cont ? pt.cyc / pt.ns * 1e3 : (pt.tns > 0 ? pt.tcyc / pt.tns * 1e3 : 0.0),
           cont ? 0.0 : (fns > 0 ? fcyc / fns * 1e3 : 0.0), pt.macs / pt.cyc, pt.macs / pt.cyc / r100);
    fflush(stdout);
  }
  return 0;
}
