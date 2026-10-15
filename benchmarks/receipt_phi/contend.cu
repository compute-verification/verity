// How much tensor throughput one interleaved CUDA instruction costs on sm_120, by instruction class.
//
// Each warp runs a loop of U = 16 m16n8k32 MMAs (FP8 e4m3 or INT8 s8; 8 independent accumulators, so every MMA's
// dependency is 8 MMAs back) with F filler instructions of one class spread evenly between them, so ρ = F/U fillers per
// MMA. Fillers run on 8 register chains and never read an accumulator, so they add issue work and nothing else. A
// block of 128·W threads (W warps on each of the SM's 4 schedulers) reserves the SM's whole shared memory, so one block
// runs per SM, and its clock64 span, after a __syncthreads on both ends, is the SM's time.
//
//   contend sweep ITERS REPS GRID   every type × class × F × W; ITERS/W loop iterations a warp (the same MMAs per
//                                   scheduler at every W); GRID = 1 for one SM, 0 for one block on every SM
//   contend one T CLS F W ITERS GRID   one launch of one kernel after a warm-up launch (for ncu); CLS = -1 is F = 0
//
// Results are JSON lines on stdout. Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 contend.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#define CK(x)                                                                                            \
  do {                                                                                                   \
    cudaError_t e_ = (x);                                                                                \
    if (e_ != cudaSuccess) {                                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));               \
      exit(3);                                                                                           \
    }                                                                                                    \
  } while (0)

__host__ __device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ void mma_ip(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_ip(uint32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t bits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ uint32_t bits(uint32_t x) { return x; }

constexpr int U = 16;
constexpr int NC = 10;
static const char* CLS_NAME[NC] = {"iadd3", "lop3", "shf", "imad", "prmt", "ffma", "lds", "idp4a", "iadd3r2", "ffma3"};
__host__ __device__ constexpr bool is_float(int c) { return c == 5 || c == 9; }
constexpr int NFV = 8;
constexpr int FV[NFV] = {2, 4, 8, 16, 32, 64, 128, 256};

// Filler i works chain u = i mod 8 and reads chains u^1 and u^2, whose values change between two fillers of one chain
// whenever F >= 8, so ptxas can't fold two fillers into one. sel (PRMT's selector) is a kernel argument for the same
// reason: two PRMTs with constant selectors compose into one.
template <int CLS>
__device__ __forceinline__ void fill(int i, uint32_t (&y)[8], float (&f)[8], uint32_t sa, uint32_t sel) {
  const int u = i & 7, v = u ^ 1, w = u ^ 2;
  if constexpr (CLS == 0) y[u] = y[u] + y[v] + y[w];
  else if constexpr (CLS == 1) y[u] = y[u] ^ (y[v] & y[w]);
  else if constexpr (CLS == 2) y[u] = __funnelshift_l(y[u], y[v], 13);
  else if constexpr (CLS == 3) y[u] = y[u] * y[v] + y[w];
  else if constexpr (CLS == 4) y[u] = __byte_perm(y[u], y[v], sel);
  else if constexpr (CLS == 5) f[u] = fmaf(f[u], 0.75f, f[v]);
  else if constexpr (CLS == 6) asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(y[u]) : "r"(sa));
  else if constexpr (CLS == 7) y[u] = (uint32_t)__dp4a((int)y[v], (int)y[w], (int)y[u]);
  else if constexpr (CLS == 8) y[u] = y[u] + y[v];
  else f[u] = fmaf(f[u], f[v], f[w]);
}

template <int T, int CLS, int F>
__global__ void __launch_bounds__(1024, 1) contend_k(uint32_t* out, int iters, uint32_t seed, uint32_t sel,
                                                     long long* clk) {
  using Acc = std::conditional_t<T == 0, float, uint32_t>;
  extern __shared__ uint32_t smem[];
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t ma = T == 0 ? 0x07070707u : 0x3F3F3F3Fu;
  const uint32_t oa = T == 0 ? 0x38383838u : 0u, ob = T == 0 ? 0x30303030u : 0u;
  uint32_t a[4], b[2], y[8];
  float f[8];
  Acc acc[8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = oa | (lowbias32(seed ^ (tid * 16 + i)) & ma);
#pragma unroll
  for (int i = 0; i < 2; ++i) b[i] = ob | (lowbias32(seed ^ (tid * 16 + 4 + i)) & ma);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    y[j] = lowbias32(seed ^ (tid * 16 + 8 + j)) | 1u;
    f[j] = __uint_as_float(0x3f800000u | (y[j] & 0x7fffffu));
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = Acc(0);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) asm volatile("" : "+r"(a[i]));
  asm volatile("" : "+r"(b[0]), "+r"(b[1]));
#pragma unroll
  for (int j = 0; j < 8; ++j) asm volatile("" : "+r"(y[j]), "+f"(f[j]));
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem) + (threadIdx.x & 31) * 4;
  __syncthreads();
  const long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int m = 0; m < U; ++m) {
      mma_ip(acc[m & 7], a, b);
#pragma unroll
      for (int i = m * F / U; i < (m + 1) * F / U; ++i) fill<CLS>(i, y, f, sa, sel);
    }
  }
  __syncthreads();
  const long long t1 = clock64();
  uint32_t x = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x ^= is_float(CLS) ? __float_as_uint(f[j]) : y[j];
#pragma unroll
    for (int r = 0; r < 4; ++r) x ^= bits(acc[j][r]);
  }
  out[tid] = x;
  if (threadIdx.x == 0) {
    clk[2 * blockIdx.x] = t0;
    clk[2 * blockIdx.x + 1] = t1;
  }
}

// ---------------------------------------------------------------------------------------------------------------- host
using KFn = void (*)(uint32_t*, int, uint32_t, uint32_t, long long*);
using Row = std::array<KFn, NFV>;

template <int T, int C, size_t... I>
constexpr Row row(std::index_sequence<I...>) {
  return {contend_k<T, C, FV[I]>...};
}
template <int T, size_t... C>
constexpr std::array<Row, NC> table(std::index_sequence<C...>) {
  return {row<T, (int)C>(std::make_index_sequence<NFV>{})...};
}
static const std::array<Row, NC> TAB[2] = {table<0>(std::make_index_sequence<NC>{}),
                                           table<1>(std::make_index_sequence<NC>{})};
static const KFn BASE[2] = {contend_k<0, 0, 0>, contend_k<1, 0, 0>};
static const char* TYPE_NAME[2] = {"fp8", "s8"};

struct Res {
  double med, lo, hi, ms;
  int blocks;
  bool ok;
};

// One block per SM: it reserves the SM's whole shared memory.
static Res run(KFn fn, int W, int iters, int reps, int grid, uint32_t seed) {
  int dev = 0, sms = 0, per_smp = 0, reserved = 0, optin = 0, per_sm = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&per_smp, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const int smem = std::min(per_smp - reserved, optin), tpb = 128 * W;
  CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, tpb, smem));
  if (per_sm != 1) return {0, 0, 0, 0, 0, false};
  const int blocks = grid > 0 ? grid : sms;
  uint32_t* out;
  long long* clk;
  CK(cudaMalloc(&out, (size_t)blocks * tpb * sizeof(uint32_t)));
  CK(cudaMalloc(&clk, (size_t)blocks * 2 * sizeof(long long)));
  std::vector<long long> h(2 * blocks);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  fn<<<blocks, tpb, smem>>>(out, iters / 4 + 1, seed, 0x5140u, clk);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<double> med, lo, hi, ms;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    fn<<<blocks, tpb, smem>>>(out, iters, seed + r, 0x5140u, clk);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float t = 0;
    CK(cudaEventElapsedTime(&t, e0, e1));
    CK(cudaMemcpy(h.data(), clk, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    std::vector<double> span(blocks);
    for (int b = 0; b < blocks; ++b) span[b] = (double)(h[2 * b + 1] - h[2 * b]);
    std::sort(span.begin(), span.end());
    med.push_back(span[blocks / 2]);
    lo.push_back(span.front());
    hi.push_back(span.back());
    ms.push_back(t);
  }
  std::sort(med.begin(), med.end());
  std::sort(ms.begin(), ms.end());
  CK(cudaFree(out));
  CK(cudaFree(clk));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return {med[reps / 2], *std::min_element(lo.begin(), lo.end()), *std::max_element(hi.begin(), hi.end()),
          ms[reps / 2], blocks, true};
}

static void emit(int t, int c, int f, int W, int iters, int reps, int grid, KFn fn, const Res& r) {
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, fn));
  const double mmas = (double)W * iters * U;  // per scheduler
  printf("{\"type\":\"%s\",\"cls\":\"%s\",\"F\":%d,\"rho\":%.4f,\"W\":%d,\"grid\":%d,\"blocks\":%d,\"iters\":%d,"
         "\"reps\":%d,\"regs\":%d,\"local_bytes\":%zu,\"ok\":%s",
         TYPE_NAME[t], c < 0 ? "none" : CLS_NAME[c], f, (double)f / U, W, grid, r.blocks, iters, reps, fa.numRegs,
         fa.localSizeBytes, r.ok ? "true" : "false");
  if (r.ok)
    printf(",\"cycles\":%.0f,\"cycles_lo\":%.0f,\"cycles_hi\":%.0f,\"ms\":%.5f,\"mhz\":%.2f,\"cyc_per_mma\":%.5f,"
           "\"tensor_eff\":%.5f",
           r.med, r.lo, r.hi, r.ms, r.med / (r.ms * 1e3), r.med / mmas, 16.0 * mmas / r.med);
  printf("}\n");
  fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc >= 5 && !strcmp(argv[1], "sweep")) {
    const int iters = atoi(argv[2]), reps = atoi(argv[3]), grid = atoi(argv[4]);
    const int WS[4] = {1, 2, 4, 8};
    for (int t = 0; t < 2; ++t)
      for (int W : WS) {
        const int it = iters / W;
        emit(t, -1, 0, W, it, reps, grid, BASE[t], run(BASE[t], W, it, reps, grid, 0x5EED0000u + W));
        for (int c = 0; c < NC; ++c)
          for (int fi = 0; fi < NFV; ++fi) {
            KFn fn = TAB[t][c][fi];
            emit(t, c, FV[fi], W, it, reps, grid, fn, run(fn, W, it, reps, grid, 0x5EED0000u + 97 * c + fi));
          }
        emit(t, -1, 0, W, it, reps, grid, BASE[t], run(BASE[t], W, it, reps, grid, 0x5EED1000u + W));
      }
    return 0;
  }
  if (argc >= 8 && !strcmp(argv[1], "one")) {
    const int t = atoi(argv[2]), c = atoi(argv[3]), f = atoi(argv[4]), W = atoi(argv[5]), it = atoi(argv[6]),
              grid = atoi(argv[7]);
    int fi = -1;
    for (int i = 0; i < NFV; ++i)
      if (FV[i] == f) fi = i;
    if (t < 0 || t > 1 || c >= NC || (c >= 0 && fi < 0)) {
      fprintf(stderr, "no kernel T=%d CLS=%d F=%d\n", t, c, f);
      return 2;
    }
    KFn fn = c < 0 ? BASE[t] : TAB[t][c][fi];
    emit(t, c, c < 0 ? 0 : f, W, it, 1, grid, fn, run(fn, W, it, 1, grid, 0x5EED2000u));
    return 0;
  }
  fprintf(stderr, "usage: contend sweep ITERS REPS GRID | contend one T CLS F W ITERS GRID\n");
  return 2;
}
