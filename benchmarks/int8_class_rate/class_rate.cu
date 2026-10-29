// The H200's sustained int8 rate per operand class, alone and beside SHA-512 paced at a ratio of compressions a T MAC.
// Timings only: every rate and clock is the kernels' own clock64 over %globaltimer, and nothing else on the card is read.
//
// Operand classes (Verity.OperandClass), each byte drawn from a hash of its position: s8 uniform on [-128, 127], s2 on
// [-2, 1], u2 on [0, 3], u4 on [0, 15], zero. Kernels:
//   wgmma  one warpgroup an SM issuing wgmma m64n256k32 back to back on A and B resident in
//          shared memory (KS k-slices, cycled), the tensor pipe's issue limit with no loads. ENC is the instruction's operand
//          types: s8s8 (the class's bytes), u8u8 or s8u8 (the offset re-encoding: each value plus OFF, read as u8; s8's OFF
//          is 128, s2's 2, the unsigned classes' 0).
//   lt     cuBLASLt's int8 GEMM, M = N = K = 8192 into int32 (TN), operands in HBM, a stamp kernel between GEMMs; sampled
//          entries of C checked against the host's.
// Each RATIO (M compressions a T MAC) is an arm: 0 the tensor alone, else the tensor with SHA-512 (sha512x.cuh's compress
// chained, 2 CTAs of 256 threads an SM) on a second stream, each warp paced by %globaltimer to RATIO times the alone arm's
// rate; the arm reports the ratio it achieved over the hash's window and the tensor's rate inside that window. Where the
// hash falls short of RATIO beside the whole tensor, wgmma tries again with the tensor paced (`tensor_paced`, a fraction
// of the alone rate) at the fraction the shortfall names and halfway from it to the whole, leaving the hash room. Where
// the hash overshoots (the tensor slowed beside it), wgmma tries again with the hash paced to the rate the tensor kept.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -I. -I../torch_gpu/sha512_rows -o class_rate
//        class_rate.cu -lcublasLt
// Run:   class_rate wgmma CLASS ENC SECONDS RATIO...
//        class_rate lt CLASS SECONDS RATIO...
// One JSON line an arm on stdout.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "sha512x.cuh"
#include "wgmma_i8.cuh"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)
#define LT(x)                                                                                    \
  do {                                                                                           \
    cublasStatus_t s_ = (x);                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                           \
      fprintf(stderr, "%s:%d %s: cublas status %d\n", __FILE__, __LINE__, #x, (int)s_);          \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

using u64 = unsigned long long;
using u32 = uint32_t;

struct Stamp {
  u64 t0, t1, c0, c1;
  u32 sm, n;
};

enum Cls { S8 = 0, S2 = 1, U2 = 2, U4 = 3, ZERO = 4 };
static const char* CLS_NAMES[] = {"s8", "s2", "u2", "u4", "zero"};
static const int CLS_OFF[] = {128, 2, 0, 0, 0};

__host__ __device__ inline u32 mix(u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// four operand bytes of class `cls` at word i, each value plus `off` (the offset re-encoding), as bytes
__host__ __device__ inline u32 word(u32 i, int cls, int off) {
  const u32 x = mix(i * 2654435761u + 0x9e3779b9u);
  u32 w = 0;
  for (int j = 0; j < 4; ++j) {
    const u32 b = (x >> (8 * j)) & 0xffu;
    int v = cls == S8 ? (int)(int8_t)b : cls == S2 ? (int)(b & 3u) - 2 : cls == U2 ? (int)(b & 3u) : cls == U4 ? (int)(b & 15u) : 0;
    w |= (u32)((v + off) & 0xff) << (8 * j);
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

constexpr int HB = 4;  // compressions between a warp's pacing checks

// SHA-512 compressions chained per thread; with ns_per_batch > 0 each warp holds HB compressions to one batch per
// ns_per_batch of %globaltimer from its start
__global__ void hash_k(u64* io, int reps, u64 ns_per_batch, Stamp* out) {
  if (threadIdx.x == 0) { out[blockIdx.x].t0 = gtimer(); out[blockIdx.x].c0 = clock64(); }
  u64 start = gtimer();
  start = __shfl_sync(0xffffffffu, start, 0);
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
    if (ns_per_batch && (r + 1) % HB == 0) {
      const u64 target = start + (u64)((r + 1) / HB) * ns_per_batch;
      while (gtimer() < target) __nanosleep(500);
    }
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
  __syncthreads();
  if (threadIdx.x == 0) { out[blockIdx.x].t1 = gtimer(); out[blockIdx.x].c1 = clock64(); out[blockIdx.x].sm = smid(); }
}

__device__ __forceinline__ uint64_t desc(const void* p, u32 lbo, u32 sbo) {
  const u32 a = static_cast<u32>(__cvta_generic_to_shared(p));
  return (uint64_t)((a & 0x3FFFF) >> 4) | (uint64_t)((lbo >> 4) & 0x3FFF) << 16 | (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
}

constexpr int KS = 4;
constexpr int N = 256;
constexpr int LOGN = 16384;
constexpr int LOGMAX = 8192;

template <int ENC>
__device__ __forceinline__ void mma(uint32_t* d, uint64_t da, uint64_t db) {
  if constexpr (ENC == 0) wgmma_s8s8_m64n256k32(d, da, db);
  else if constexpr (ENC == 1) wgmma_u8u8_m64n256k32(d, da, db);
  else wgmma_s8u8_m64n256k32(d, da, db);
}

constexpr int TB = 64;  // iterations between the tensor's pacing checks

// ENC 2 (s8u8) keeps A signed: its A is the class's bytes and its B the offset bytes. With ns_per_batch > 0, the CTA
// holds each TB iterations to one batch per ns_per_batch of %globaltimer from its start.
template <int ENC>
__global__ void __launch_bounds__(128, 1) wg_k(int iters, int cls, int off, u64 ns_per_batch, Stamp* out, u64* log, u32* sink,
                                              int* started) {
  extern __shared__ __align__(1024) uint8_t smem[];
  constexpr int ABYTES = 64 * 32, BBYTES = N * 32;
  uint8_t* A = smem;
  uint8_t* B = smem + KS * ABYTES;
  const u32 seed = blockIdx.x * 0x10000u;
  for (int i = threadIdx.x; i < KS * ABYTES / 4; i += 128)
    reinterpret_cast<u32*>(A)[i] = word(seed + i, cls, ENC == 2 ? 0 : off);
  for (int i = threadIdx.x; i < KS * BBYTES / 4; i += 128)
    reinterpret_cast<u32*>(B)[i] = word(seed + 0x8000u + i, cls, off);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  uint64_t da[KS], db[KS];
#pragma unroll
  for (int k = 0; k < KS; ++k) { da[k] = desc(A + k * ABYTES, 128, 256); db[k] = desc(B + k * BBYTES, 128, 256); }
  u32 d[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) d[i] = 0;
  u64 t0 = 0, c0 = 0;
  if (threadIdx.x == 0) { atomicAdd(started, 1); t0 = gtimer(); c0 = clock64(); }
  int li = 0;
  const u64 start = gtimer();
  for (int it = 0; it < iters; ++it) {
    if (ns_per_batch && it % TB == 0) {
      if (threadIdx.x == 0) {
        const u64 target = start + (u64)(it / TB) * ns_per_batch;
        while (gtimer() < target) __nanosleep(200);
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < N / 2; ++i) asm volatile("" : "+r"(d[i])::"memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int k = 0; k < KS; ++k) mma<ENC>(d, da[k], db[k]);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < N / 2; ++i) asm volatile("" : "+r"(d[i])::"memory");
    if (threadIdx.x == 0 && (it % LOGN) == LOGN - 1 && li < LOGMAX) log[(u64)blockIdx.x * LOGMAX + li++] = gtimer();
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  u32 x = 0;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) x ^= d[i];
  if (x == 0x9e3779b9u) sink[blockIdx.x * 128 + threadIdx.x] = x;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{t0, gtimer(), c0, (u64)clock64(), smid(), (u32)li};
}

__global__ void wait_k(volatile int* started, int want) {
  while (*started < want) {
  }
}

// one %globaltimer reading into stamps[i]; at i = 0 it also releases wait_k
__global__ void stamp_k(u64* stamps, int i, int* started, int release) {
  stamps[i] = gtimer();
  if (i == 0) atomicExch(started, release);
}

__global__ void fill_k(u32* p, size_t words, u32 seed, int cls) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x)
    p[i] = word(seed + (u32)i, cls, 0);
}

struct Window {
  double t0, t1, mhz;
};

static Window window(const std::vector<Stamp>& s) {
  u64 a = ~0ull, b = 0;
  double cyc = 0, ns = 0;
  for (auto& x : s) {
    a = std::min(a, x.t0);
    b = std::max(b, x.t1);
    cyc += (double)(x.c1 - x.c0);
    ns += (double)(x.t1 - x.t0);
  }
  return {(double)a, (double)b, cyc / ns * 1e3};
}

// MAC/s summed over the CTAs' logs between w0 and w1 (ns of %globaltimer)
static double logged_rate(const std::vector<Stamp>& ts, const std::vector<u64>& log, double w0, double w1, double macs_per_iter) {
  double rate = 0;
  for (size_t c = 0; c < ts.size(); ++c) {
    const u64* L = log.data() + c * LOGMAX;
    int j0 = -1, j1 = -1;
    for (int j = 0; j < (int)ts[c].n; ++j) {
      if (L[j] >= w0 && j0 < 0) j0 = j;
      if (L[j] <= w1) j1 = j;
    }
    if (j0 >= 0 && j1 > j0) rate += (double)(j1 - j0) * LOGN * macs_per_iter / ((double)(L[j1] - L[j0]) * 1e-9);
  }
  return rate;
}

struct Hash {
  int nsm, hcta = 2, threads = 256;
  u64* io;
  Stamp* hs;
  void init(int sms) {
    nsm = sms;
    CK(cudaMalloc(&io, (size_t)hcta * nsm * threads * 8));
    CK(cudaMalloc(&hs, hcta * nsm * sizeof(Stamp)));
  }
  double lanes() const { return (double)hcta * nsm * threads; }
  // reps and the pace for `comps_per_s` compressions a second held for `secs`
  std::pair<int, u64> pace(double comps_per_s, double secs) const {
    const double per_thread = comps_per_s / lanes();
    const int reps = std::max(HB, (int)std::ceil(per_thread * secs / HB) * HB);
    return {reps, (u64)(HB * 1e9 / per_thread)};
  }
  void launch(cudaStream_t s, int reps, u64 ns_per_batch) {
    hash_k<<<hcta * nsm, threads, 0, s>>>(io, reps, ns_per_batch, hs);
    CK(cudaGetLastError());
  }
  std::vector<Stamp> stamps() {
    std::vector<Stamp> h(hcta * nsm);
    CK(cudaMemcpy(h.data(), hs, h.size() * sizeof(Stamp), cudaMemcpyDeviceToHost));
    return h;
  }
};

static int nsm;
static cudaStream_t sh, st;
static int* started;

static void streams() {
  int plo, phi;
  CK(cudaDeviceGetStreamPriorityRange(&plo, &phi));
  CK(cudaStreamCreateWithPriority(&sh, cudaStreamNonBlocking, phi));
  CK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, plo));
  CK(cudaMalloc(&started, 4));
}

static void beside_line(const char* kernel, const char* cls, const char* enc, double ratio, double secs, const Window& w,
                        double comps, double tensor_rate, double macs_per_smclk, bool covered, double paced = 1) {
  const double hs = (w.t1 - w.t0) * 1e-9;
  printf("{\"kernel\": \"%s\", \"class\": \"%s\", \"enc\": \"%s\", \"ratio\": %.3f, \"tensor_paced\": %.4f, "
         "\"secs\": %.1f, \"s\": %.3f, "
         "\"mhz\": %.1f, \"macs_per_smclk\": %.1f, \"tmacs_per_s\": %.3f, \"gcomp_s\": %.4f, \"ratio_achieved\": %.4f, "
         "\"covered\": %s}\n",
         kernel, cls, enc, ratio, paced, secs, hs, w.mhz, macs_per_smclk, tensor_rate * 1e-12, comps / hs * 1e-9,
         tensor_rate > 0 ? comps / (tensor_rate * hs) * 1e6 : 0.0, covered ? "true" : "false");
  fflush(stdout);
}

template <int ENC>
static void run_wgmma(int cls, double secs, const std::vector<double>& ratios, const char* enc) {
  const int off = ENC == 0 ? 0 : CLS_OFF[cls];
  const size_t smem = (size_t)KS * (64 * 32 + N * 32);
  CK(cudaFuncSetAttribute(wg_k<ENC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  const double macs_per_iter = 64.0 * N * 32 * KS;
  Stamp* tsd;
  u64* logd;
  u32* sink;
  CK(cudaMalloc(&tsd, nsm * sizeof(Stamp)));
  CK(cudaMalloc(&logd, (size_t)nsm * LOGMAX * 8));
  CK(cudaMalloc(&sink, (size_t)nsm * 128 * 4));
  Hash h;
  h.init(nsm);
  auto launch = [&](int iters, u64 pace = 0) {
    CK(cudaMemset(started, 0, 4));
    CK(cudaDeviceSynchronize());
    wg_k<ENC><<<nsm, 128, smem, st>>>(iters, cls, off, pace, tsd, logd, sink, started);
    CK(cudaGetLastError());
  };
  launch(1);
  CK(cudaDeviceSynchronize());
  auto get = [&]() {
    std::vector<Stamp> s(nsm);
    std::vector<u64> L((size_t)nsm * LOGMAX);
    CK(cudaMemcpy(s.data(), tsd, s.size() * sizeof(Stamp), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.data(), logd, L.size() * 8, cudaMemcpyDeviceToHost));
    return std::make_pair(s, L);
  };
  const int cap = LOGMAX * LOGN - LOGN;
  launch(4096);
  launch(LOGN * 4);
  CK(cudaDeviceSynchronize());
  Window wc = window(get().first);
  const double t_iter = (wc.t1 - wc.t0) * 1e-9 / (LOGN * 4);
  double alone = 0;
  for (double ratio : ratios) {
    if (ratio == 0) {
      const int iters = std::min((double)cap, secs / t_iter);
      launch(iters);
      CK(cudaDeviceSynchronize());
      auto [s, L] = get();
      Window w = window(s);
      const double sec = (w.t1 - w.t0) * 1e-9, macs = macs_per_iter * iters * nsm;
      alone = macs / sec;
      const double last = logged_rate(s, L, w.t0 + 0.5 * (w.t1 - w.t0), w.t1, macs_per_iter);
      printf("{\"kernel\": \"wgmma\", \"class\": \"%s\", \"enc\": \"%s\", \"ratio\": 0, \"secs\": %.1f, \"s\": %.3f, "
             "\"mhz\": %.1f, \"macs_per_smclk\": %.1f, \"tmacs_per_s\": %.3f, \"tmacs_per_s_last_half\": %.3f}\n",
             CLS_NAMES[cls], enc, secs, sec, w.mhz, alone / (nsm * w.mhz * 1e6), alone * 1e-12, last * 1e-12);
      fflush(stdout);
      continue;
    }
    const double ref = alone > 0 ? alone : macs_per_iter * nsm / t_iter;
    // the tensor at `frac` of `ref` (1: unpaced), the hash paced to `ratio` of `base` (0: of the tensor's target);
    // returns (tensor rate, achieved ratio)
    auto beside = [&](double frac, double base = 0) {
      const double rt = frac * ref;
      auto [reps, pace] = h.pace(ratio * 1e-6 * (base > 0 ? base : rt), secs);
      const u64 tpace = frac < 1 ? (u64)(TB * macs_per_iter / (rt / nsm) * 1e9) : 0;
      const int iters = std::min((double)cap, (1.25 * secs + 1.0) * (frac < 1 ? rt / nsm / macs_per_iter : 1 / t_iter));
      launch(iters, tpace);
      wait_k<<<1, 1, 0, sh>>>(started, nsm);
      h.launch(sh, reps, pace);
      CK(cudaDeviceSynchronize());
      auto [s, L] = get();
      Window w = window(h.stamps()), wt = window(s);
      const double tr = logged_rate(s, L, w.t0, w.t1, macs_per_iter), comps = h.lanes() * reps;
      const double hs = (w.t1 - w.t0) * 1e-9;
      beside_line("wgmma", CLS_NAMES[cls], enc, ratio, secs, w, comps, tr, tr / (nsm * w.mhz * 1e6),
                  wt.t0 <= w.t0 && wt.t1 >= w.t1, frac);
      return std::make_pair(tr, tr > 0 ? comps / (tr * hs) * 1e6 : 0.0);
    };
    auto [tr, got] = beside(1.0);
    if (got < 0.99 * ratio && tr > 0) {
      // the hash fell short beside the whole tensor: leave it room, at the fraction its shortfall names and halfway back
      const double f1 = std::min(0.999, tr / ref * got / ratio);
      beside(f1);
      beside((1 + f1) / 2);
    } else if (got > 1.02 * ratio && tr > 0) {
      // the tensor slowed beside the hash, so the hash overshot: pace it to the rate the tensor kept, 1% over
      beside(1.0, 1.01 * tr);
    }
  }
}

static void run_lt(int cls, double secs, const std::vector<double>& ratios) {
  const int M = 8192, Nn = 8192, K = 8192;
  const double mnk = (double)M * Nn * K;
  int8_t *A, *B;
  int32_t* C;
  CK(cudaMalloc(&A, (size_t)M * K));
  CK(cudaMalloc(&B, (size_t)Nn * K));
  CK(cudaMalloc(&C, (size_t)M * Nn * 4));
  fill_k<<<1024, 256>>>(reinterpret_cast<u32*>(A), (size_t)M * K / 4, 0x01000000u, cls);
  fill_k<<<1024, 256>>>(reinterpret_cast<u32*>(B), (size_t)Nn * K / 4, 0x41000000u, cls);
  CK(cudaGetLastError());
  cublasLtHandle_t lt;
  LT(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t T = CUBLAS_OP_T, Nop = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T)));
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &Nop, sizeof(Nop)));
  cublasLtMatrixLayout_t la, lb, lc;
  LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, M, K));   // A^T: A is M rows of K
  LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, Nn, K));  // B: N columns of K
  LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, Nn, M));
  const size_t wsb = 64 << 20;
  void* ws;
  CK(cudaMalloc(&ws, wsb));
  cublasLtMatmulPreference_t pref;
  LT(cublasLtMatmulPreferenceCreate(&pref));
  LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
  cublasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  LT(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &found));
  if (!found) { fprintf(stderr, "cuBLASLt: no int8 algorithm\n"); exit(1); }
  const int32_t one = 1, zero = 0;
  auto gemm = [&](cudaStream_t s) {
    LT(cublasLtMatmul(lt, op, &one, A, la, B, lb, &zero, C, lc, C, lc, &heur.algo, ws, wsb, s));
  };
  // the host's C at sampled entries
  std::vector<int8_t> hA((size_t)M * K), hB((size_t)Nn * K);
  CK(cudaMemcpy(hA.data(), A, hA.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hB.data(), B, hB.size(), cudaMemcpyDeviceToHost));
  std::mt19937 rng(7);
  std::vector<std::pair<int, int>> at;
  for (int i = 0; i < 32; ++i) at.push_back({(int)(rng() % M), (int)(rng() % Nn)});
  auto checked = [&]() {
    for (auto [r, c] : at) {
      int64_t want = 0;
      for (int k = 0; k < K; ++k) want += (int64_t)hA[(size_t)r * K + k] * hB[(size_t)c * K + k];
      int32_t got;
      CK(cudaMemcpy(&got, C + (size_t)c * M + r, 4, cudaMemcpyDeviceToHost));
      if ((int64_t)got != want) return false;
    }
    return true;
  };
  for (int i = 0; i < 3; ++i) gemm(st);
  CK(cudaDeviceSynchronize());
  const bool ok = checked();
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, st));
  for (int i = 0; i < 10; ++i) gemm(st);
  CK(cudaEventRecord(e1, st));
  CK(cudaEventSynchronize(e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  const double t_gemm = ms * 1e-3 / 10;
  const int maxn = (int)((1.25 * secs + 1.0) / t_gemm) + 2;
  u64* stamps;
  CK(cudaMalloc(&stamps, (size_t)(maxn + 1) * 8));
  Hash h;
  h.init(nsm);
  auto loop = [&](int n) {
    CK(cudaMemset(started, 0, 4));
    CK(cudaDeviceSynchronize());
    stamp_k<<<1, 1, 0, st>>>(stamps, 0, started, nsm);
    for (int i = 1; i <= n; ++i) {
      gemm(st);
      stamp_k<<<1, 1, 0, st>>>(stamps, i, started, nsm);
    }
    CK(cudaGetLastError());
  };
  auto get = [&](int n) {
    std::vector<u64> s(n + 1);
    CK(cudaMemcpy(s.data(), stamps, s.size() * 8, cudaMemcpyDeviceToHost));
    return s;
  };
  double alone = 0;
  for (double ratio : ratios) {
    if (ratio == 0) {
      const int n = std::max(2, (int)(secs / t_gemm));
      loop(n);
      CK(cudaDeviceSynchronize());
      auto s = get(n);
      const double sec = (s[n] - s[0]) * 1e-9;
      alone = n * mnk / sec;
      printf("{\"kernel\": \"lt\", \"class\": \"%s\", \"enc\": \"s8s8\", \"ratio\": 0, \"secs\": %.1f, \"s\": %.3f, "
             "\"tmacs_per_s\": %.3f, \"checked\": %s}\n",
             CLS_NAMES[cls], secs, sec, alone * 1e-12, ok && checked() ? "true" : "false");
      fflush(stdout);
      continue;
    }
    const double ref = alone > 0 ? alone : mnk / t_gemm;
    auto [reps, pace] = h.pace(ratio * 1e-6 * ref, secs);
    const int n = maxn - 2;
    loop(n);
    wait_k<<<1, 1, 0, sh>>>(started, nsm);
    h.launch(sh, reps, pace);
    CK(cudaDeviceSynchronize());
    auto s = get(n);
    Window w = window(h.stamps());
    int j0 = -1, j1 = -1;
    for (int j = 0; j <= n; ++j) {
      if (s[j] >= w.t0 && j0 < 0) j0 = j;
      if (s[j] <= w.t1) j1 = j;
    }
    const double tr = j0 >= 0 && j1 > j0 ? (j1 - j0) * mnk / ((s[j1] - s[j0]) * 1e-9) : 0;
    beside_line("lt", CLS_NAMES[cls], "s8s8", ratio, secs, w, h.lanes() * reps, tr, tr / (nsm * w.mhz * 1e6),
                s[0] <= w.t0 && s[n] >= w.t1);
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "class_rate wgmma CLASS ENC SECONDS RATIO... | class_rate lt CLASS SECONDS RATIO...\n");
    return 2;
  }
  const std::string kernel = argv[1], cname = argv[2];
  int cls = -1;
  for (int c = 0; c < 5; ++c)
    if (cname == CLS_NAMES[c]) cls = c;
  if (cls < 0) { fprintf(stderr, "CLASS: s8, s2, u2, u4 or zero\n"); return 2; }
  // a kernel's first launch under lazy loading waits for the running one, so a hash beside the tensor would run after it
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  nsm = p.multiProcessorCount;
  streams();
  {
    CK(cudaMemset(started, 0, 4));
    wait_k<<<1, 1, 0, sh>>>(started, 0);
    Hash warm;
    warm.init(nsm);
    warm.launch(sh, HB, 1);
    CK(cudaDeviceSynchronize());
  }
  const int first = kernel == "wgmma" ? 5 : 4;
  const double secs = atof(argv[first - 1]);
  std::vector<double> ratios;
  for (int i = first; i < argc; ++i) ratios.push_back(atof(argv[i]));
  if (kernel == "lt") {
    run_lt(cls, secs, ratios);
  } else if (kernel == "wgmma") {
    const std::string enc = argv[3];
    if (enc == "s8s8") run_wgmma<0>(cls, secs, ratios, "s8s8");
    else if (enc == "u8u8") run_wgmma<1>(cls, secs, ratios, "u8u8");
    else if (enc == "s8u8") run_wgmma<2>(cls, secs, ratios, "s8u8");
    else { fprintf(stderr, "ENC: s8s8, u8u8 or s8u8\n"); return 2; }
  } else {
    fprintf(stderr, "KERNEL: wgmma or lt\n");
    return 2;
  }
  return 0;
}
