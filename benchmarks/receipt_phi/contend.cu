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
//   contend split ITERS REPS        the same on one SM with the fillers in their own warps: of each scheduler's W
//                                   warps, W/2 run only the MMAs and W/2 only the fillers, 2·ITERS/W iterations each
//   contend one T CLS F W ITERS GRID [SPLIT]   one launch after a warm-up launch (for ncu); CLS = -1 is F = 0
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
constexpr int NC = 16;
static const char* CLS_NAME[NC] = {"iadd3",  "lop3",   "shf",    "imad", "prmt",          "ffma",
                                   "lds",    "idp4a",  "iadd3r2", "ffma3", "iadd3c",       "iadd3i",
                                   "uiadd3", "rot1",   "mix_iadd3_shf", "mix_iadd3_lds"};
__host__ __device__ constexpr bool is_float(int c) { return c == 5 || c == 9; }
constexpr int NFV = 8;
constexpr int FV[NFV] = {2, 4, 8, 16, 32, 64, 128, 256};

// Filler i works chain u = i mod 8 and reads chains u^1 and u^2, whose values change between two fillers of one chain
// whenever F >= 8, so ptxas can't fold two fillers into one. sel (PRMT's selector, iadd3c's addend) is a kernel
// argument for the same reason: two PRMTs with constant selectors compose into one. uiadd3 is iadd3 on chains that
// start from seed alone, so they are warp-uniform and ptxas can put them on the uniform datapath.
template <int CLS>
__device__ __forceinline__ void fill(int i, uint32_t (&y)[8], float (&f)[8], uint32_t sa, uint32_t sel) {
  const int u = i & 7, v = u ^ 1, w = u ^ 2;
  if constexpr (CLS == 14 || CLS == 15) {
    if ((i & 1) == 0) y[u] = y[u] + y[v] + y[w];
    else if constexpr (CLS == 14) y[u] = __funnelshift_l(y[u], y[v], 13);
    else asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(y[u]) : "r"(sa));
  } else if constexpr (CLS == 0 || CLS == 12) y[u] = y[u] + y[v] + y[w];
  else if constexpr (CLS == 1) y[u] = y[u] ^ (y[v] & y[w]);
  else if constexpr (CLS == 2) y[u] = __funnelshift_l(y[u], y[v], 13);
  else if constexpr (CLS == 3) y[u] = y[u] * y[v] + y[w];
  else if constexpr (CLS == 4) y[u] = __byte_perm(y[u], y[v], sel);
  else if constexpr (CLS == 5) f[u] = fmaf(f[u], 0.75f, f[v]);
  else if constexpr (CLS == 6) asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(y[u]) : "r"(sa));
  else if constexpr (CLS == 7) y[u] = (uint32_t)__dp4a((int)y[v], (int)y[w], (int)y[u]);
  else if constexpr (CLS == 8) y[u] = y[u] + y[v];
  else if constexpr (CLS == 9) f[u] = fmaf(f[u], f[v], f[w]);
  else if constexpr (CLS == 10) y[u] = y[u] + y[v] + sel;
  else if constexpr (CLS == 11) y[u] = y[u] + y[v] + 0x2a5u;
  else asm volatile("shf.l.wrap.b32 %0, %1, %1, 7;" : "=r"(y[u]) : "r"(y[u]));
}

// split = 0: every warp runs the fused loop. split = 1: warps whose id has bit 2 set (half of each scheduler's warps)
// run only the F fillers per iteration and the rest only the 16 MMAs; clk gets each group's last end time too.
template <int T, int CLS, int F>
__global__ void __launch_bounds__(1024, 1) contend_k(uint32_t* out, int iters, uint32_t seed, uint32_t sel, int split,
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
    y[j] = lowbias32(seed ^ ((CLS == 12 ? 0u : tid * 16) + 8 + j)) | 1u;
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
  unsigned long long* endt = reinterpret_cast<unsigned long long*>(smem + 1024);
  const int fw = split && ((threadIdx.x >> 7) & 1);
  if (threadIdx.x == 0) endt[0] = endt[1] = 0;
  __syncthreads();
  const long long t0 = clock64();
  if (!split) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int m = 0; m < U; ++m) {
        mma_ip(acc[m & 7], a, b);
#pragma unroll
        for (int i = m * F / U; i < (m + 1) * F / U; ++i) fill<CLS>(i, y, f, sa, sel);
      }
    }
  } else if (!fw) {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int m = 0; m < U; ++m) mma_ip(acc[m & 7], a, b);
    }
  } else {
#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int i = 0; i < F; ++i) fill<CLS>(i, y, f, sa, sel);
    }
  }
  if ((threadIdx.x & 31) == 0) atomicMax(&endt[fw], (unsigned long long)clock64());
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
    clk[4 * blockIdx.x] = t0;
    clk[4 * blockIdx.x + 1] = t1;
    clk[4 * blockIdx.x + 2] = (long long)endt[0];
    clk[4 * blockIdx.x + 3] = (long long)endt[1];
  }
}

// ---------------------------------------------------------------------------------------------------------------- host
using KFn = void (*)(uint32_t*, int, uint32_t, uint32_t, int, long long*);
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
  double med, lo, hi, ms, mma, fill;  // mma, fill: each split group's span (median over reps of the median block)
  int blocks;
  bool ok;
};

// One block per SM: it reserves the SM's whole shared memory.
static Res run(KFn fn, int W, int iters, int reps, int grid, uint32_t seed, int split = 0) {
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
  if (per_sm != 1) return {0, 0, 0, 0, 0, 0, 0, false};
  const int blocks = grid > 0 ? grid : sms;
  uint32_t* out;
  long long* clk;
  CK(cudaMalloc(&out, (size_t)blocks * tpb * sizeof(uint32_t)));
  CK(cudaMalloc(&clk, (size_t)blocks * 4 * sizeof(long long)));
  std::vector<long long> h(4 * blocks);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  fn<<<blocks, tpb, smem>>>(out, iters / 4 + 1, seed, 0x5140u, split, clk);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<double> med, lo, hi, ms, mma, fill;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    fn<<<blocks, tpb, smem>>>(out, iters, seed + r, 0x5140u, split, clk);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float t = 0;
    CK(cudaEventElapsedTime(&t, e0, e1));
    CK(cudaMemcpy(h.data(), clk, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    std::vector<double> span(blocks), sm(blocks), sf(blocks);
    for (int b = 0; b < blocks; ++b) {
      span[b] = (double)(h[4 * b + 1] - h[4 * b]);
      sm[b] = (double)(h[4 * b + 2] - h[4 * b]);
      sf[b] = split ? (double)(h[4 * b + 3] - h[4 * b]) : 0.0;
    }
    std::sort(span.begin(), span.end());
    std::sort(sm.begin(), sm.end());
    std::sort(sf.begin(), sf.end());
    med.push_back(span[blocks / 2]);
    lo.push_back(span.front());
    hi.push_back(span.back());
    mma.push_back(sm[blocks / 2]);
    fill.push_back(sf[blocks / 2]);
    ms.push_back(t);
  }
  for (auto* v : {&med, &ms, &mma, &fill}) std::sort(v->begin(), v->end());
  CK(cudaFree(out));
  CK(cudaFree(clk));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return {med[reps / 2],  *std::min_element(lo.begin(), lo.end()), *std::max_element(hi.begin(), hi.end()),
          ms[reps / 2],   mma[reps / 2], fill[reps / 2], blocks, true};
}

// cyc_per_mma is the MMA warps' span (the block's, unsplit) over the MMAs each scheduler runs.
static void emit(int t, int c, int f, int W, int iters, int reps, int grid, int split, KFn fn, const Res& r) {
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, fn));
  const double mmas = (double)(split ? W / 2 : W) * iters * U;
  printf("{\"type\":\"%s\",\"cls\":\"%s\",\"F\":%d,\"rho\":%.4f,\"W\":%d,\"split\":%d,\"grid\":%d,\"blocks\":%d,"
         "\"iters\":%d,\"reps\":%d,\"regs\":%d,\"local_bytes\":%zu,\"ok\":%s",
         TYPE_NAME[t], c < 0 ? "none" : CLS_NAME[c], f, (double)f / U, W, split, grid, r.blocks, iters, reps,
         fa.numRegs, fa.localSizeBytes, r.ok ? "true" : "false");
  if (r.ok) {
    const double span = split ? r.mma : r.med;
    printf(",\"cycles\":%.0f,\"cycles_lo\":%.0f,\"cycles_hi\":%.0f,\"mma_cycles\":%.0f,\"fill_cycles\":%.0f,"
           "\"ms\":%.5f,\"mhz\":%.2f,\"cyc_per_mma\":%.5f,\"tensor_eff\":%.5f",
           r.med, r.lo, r.hi, r.mma, r.fill, r.ms, r.med / (r.ms * 1e3), span / mmas, 16.0 * mmas / span);
  }
  printf("}\n");
  fflush(stdout);
}

int main(int argc, char** argv) {
  const bool sweep = argc >= 5 && !strcmp(argv[1], "sweep"), split = argc >= 4 && !strcmp(argv[1], "split");
  if (sweep || split) {
    const int iters = atoi(argv[2]), reps = atoi(argv[3]), grid = sweep ? atoi(argv[4]) : 1;
    const std::vector<int> WS = sweep ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{2, 4, 8};
    const int s = split ? 1 : 0;
    for (int t = 0; t < 2; ++t)
      for (int W : WS) {
        const int it = split ? 2 * iters / W : iters / W;
        emit(t, -1, 0, W, it, reps, grid, s, BASE[t], run(BASE[t], W, it, reps, grid, 0x5EED0000u + W, s));
        for (int c = 0; c < NC; ++c)
          for (int fi = 0; fi < NFV; ++fi) {
            KFn fn = TAB[t][c][fi];
            emit(t, c, FV[fi], W, it, reps, grid, s, fn, run(fn, W, it, reps, grid, 0x5EED0000u + 97 * c + fi, s));
          }
        emit(t, -1, 0, W, it, reps, grid, s, BASE[t], run(BASE[t], W, it, reps, grid, 0x5EED1000u + W, s));
      }
    return 0;
  }
  if (argc >= 8 && !strcmp(argv[1], "one")) {
    const int t = atoi(argv[2]), c = atoi(argv[3]), f = atoi(argv[4]), W = atoi(argv[5]), it = atoi(argv[6]),
              grid = atoi(argv[7]), s = argc >= 9 ? atoi(argv[8]) : 0;
    int fi = -1;
    for (int i = 0; i < NFV; ++i)
      if (FV[i] == f) fi = i;
    if (t < 0 || t > 1 || c >= NC || (c >= 0 && fi < 0)) {
      fprintf(stderr, "no kernel T=%d CLS=%d F=%d\n", t, c, f);
      return 2;
    }
    KFn fn = c < 0 ? BASE[t] : TAB[t][c][fi];
    emit(t, c, c < 0 ? 0 : f, W, it, 1, grid, s, fn, run(fn, W, it, 1, grid, 0x5EED2000u, s));
    return 0;
  }
  fprintf(stderr, "usage: contend sweep ITERS REPS GRID | contend split ITERS REPS | contend one T CLS F W ITERS GRID"
                  " [SPLIT]\n");
  return 2;
}
