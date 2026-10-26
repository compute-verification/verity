// Shared helpers for the sm_120 census probes: error checks, one-block-per-SM launches timed by every block's clock64,
// and JSON lines on stdout. A census kernel brackets its timed loop with CLK_BEGIN / CLK_END(rec), so each block records
// the SM cycles its loop took (the block's slowest warp, after a barrier) and the SM it ran on.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "ERR %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));     \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

struct BlockClk {
  long long cyc;
  long long t0_ns, t1_ns;  // %globaltimer at the loop's start and end (thread 0)
  int smid;
  int pad;
};

__device__ __forceinline__ int smid_reg() {
  int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

#define CLK_BEGIN          \
  __syncthreads();         \
  long long _t0 = clock64(); \
  long long _g0 = gtimer();
#define CLK_END(rec)                                                                     \
  __syncthreads();                                                                       \
  if (threadIdx.x == 0) {                                                                \
    long long _t1 = clock64();                                                           \
    rec[blockIdx.x].cyc = _t1 - _t0;                                                     \
    rec[blockIdx.x].t0_ns = _g0;                                                         \
    rec[blockIdx.x].t1_ns = gtimer();                                                    \
    rec[blockIdx.x].smid = smid_reg();                                                   \
  }

struct Timing {
  double ms = 0;        // CUDA-event time of the launch
  double cyc_med = 0;   // median over blocks of the loop's SM cycles
  double cyc_min = 0, cyc_max = 0;
  int blocks = 0, distinct_sms = 0;
  double mhz = 0;       // cyc_max over the launch's event time: the SM clock the loop ran at
};

inline int num_sms() {
  int d = 0, n = 0;
  CK(cudaGetDevice(&d));
  CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d));
  return n;
}

inline Timing summarize(const std::vector<BlockClk>& h, double ms) {
  Timing t;
  t.ms = ms;
  t.blocks = (int)h.size();
  std::vector<double> c;
  std::vector<int> s;
  for (auto& r : h) { c.push_back((double)r.cyc); s.push_back(r.smid); }
  std::sort(c.begin(), c.end());
  std::sort(s.begin(), s.end());
  t.distinct_sms = (int)(std::unique(s.begin(), s.end()) - s.begin());
  t.cyc_med = c[c.size() / 2];
  t.cyc_min = c.front();
  t.cyc_max = c.back();
  t.mhz = ms > 0 ? t.cyc_max / (ms * 1e3) : 0;
  return t;
}

// Launch `launch()` (which must write `nblocks` records into d_rec) once to warm up, then `reps` times; return the rep
// with the smallest median block cycles.
template <class F>
Timing run_timed(F launch, int nblocks, BlockClk* d_rec, int reps = 3) {
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
    Timing t = summarize(h, ms);
    if (t.cyc_med < best.cyc_med) best = t;
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best;
}

// Event-timed best of `reps` (after one warm-up), for probes that measure wall time rather than SM cycles.
template <class F>
double best_ms(F launch, int reps = 5) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  launch();
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  double best = 1e300;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(e0));
    launch();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, (double)ms);
  }
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return best;
}

inline uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

inline void print_timing(const Timing& t) {
  printf("\"ms\":%.6f,\"cyc_med\":%.0f,\"cyc_min\":%.0f,\"cyc_max\":%.0f,\"blocks\":%d,\"distinct_sms\":%d,\"mhz\":%.1f", t.ms,
         t.cyc_med, t.cyc_min, t.cyc_max, t.blocks, t.distinct_sms, t.mhz);
}

// Dynamic shared memory that keeps a second block of the same kernel off an SM (sm_120: 100 KB a SM).
constexpr int ONE_BLOCK_SMEM = 60 * 1024;
