// The control for hashcat's SHA-512 rate: sha512x.cuh's default compression (sx::compress<0, ROLL>) timed in the
// shapes of hashcat's -m 1700 optimized kernel (OpenCL/m01700_a3-optimized.cl, v7.1.2), so the difference between our raw
// rate and hashcat's can be attributed to what each form leaves out. Each thread runs REPS compressions a launch:
//
//   chain    sha_rate's (commit_gpu.cu): a variable block a compression, chained from the last state, every word live
//   iv       a candidate a compression from the constant IV (OPTI_TYPE_PRECOMPUTE_INIT): W0 the candidate, W1..W6 and W15
//            read from memory (hashcat's w[1..6] and its length word), W7..W14 zero at compile time
//            (OPTI_TYPE_ZERO_BYTE: m01700's `w7_t = 0 ... we_t = 0`), every output word read
//   skip     iv with only d and h read, as m01700's compare reads digest[3] and digest[7] (OPTI_TYPE_EARLY_SKIP): d and h
//            after round 79 are a and e after round 76, so the last three rounds and their three schedule words are dead
//
// ROLL is sha512x.cuh's (1: rounds 16..79 a loop of 16-round bodies; 0: straight-line, as hashcat's kernel is).
//
//   hashcat_forms time LAUNCHES  one JSON line a form: compressions a second, the SM clock (NVML) after each launch
//   hashcat_forms shapes LAUNCHES THREADS...
//                                chain (both ROLLs) and skip straight-line at each CTA size, in whole waves of the CTAs an
//                                SM holds (the occupancy API), with their registers and resident warps an SM
//   hashcat_forms check          the SM count, and every word of iv's first compression in threads 0 and 1 (the 7-byte
//                                candidates 0 and 2^20, the IV fed forward), for hashcat_summary.py to compare with
//                                hashlib.sha512
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sha512x.cuh"

using u64 = uint64_t;

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

enum Form { CHAIN = 0, IV = 1, SKIP = 2 };

// the candidate of thread t's r-th compression: 7 big-endian bytes and the pad byte 0x80, as hashcat's w0 for a 7-byte
// password (OPTS_TYPE_PT_ADD80)
__device__ __forceinline__ u64 candidate(u64 t, int r) { return ((t << 20) + (u64)r) << 8 | 0x80; }

template <int FORM, int ROLL>
__global__ void forms(const u64* __restrict__ tail, u64* out, int reps) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 acc = 0;
  if constexpr (FORM == CHAIN) {
    u64 st[8], w[16];
    sx::iv(st);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
    for (int r = 0; r < reps; ++r) {
#pragma unroll
      for (int i = 0; i < 8; ++i) w[i] ^= st[i];
      sx::compress<0, ROLL>(st, w);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) acc ^= st[i];
  } else {
    const u64 w1 = tail[0], w2 = tail[1], w3 = tail[2], w4 = tail[3], w5 = tail[4], w6 = tail[5], w15 = tail[6];
    for (int r = 0; r < reps; ++r) {
      u64 st[8], w[16] = {candidate(t, r), w1, w2, w3, w4, w5, w6, 0, 0, 0, 0, 0, 0, 0, 0, w15};
      sx::iv(st);
      sx::compress<0, ROLL>(st, w);
      if constexpr (FORM == IV) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc ^= st[i];
      } else {
        acc ^= st[3] ^ st[7];
      }
    }
  }
  if (acc == 0x0123456789abcdefull) out[t] = acc;
}

// iv's first compression in threads 0 and 1, every word
__global__ void first(const u64* __restrict__ tail, u64* out) {
  const u64 t = threadIdx.x;
  u64 st[8], w[16] = {candidate(t, 0), tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], 0, 0, 0, 0, 0, 0, 0, 0, tail[6]};
  sx::iv(st);
  sx::compress<0, 1>(st, w);
  for (int i = 0; i < 8; ++i) out[t * 8 + i] = st[i];
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Nv {
  nvmlDevice_t d{};
  bool ok = false;
  Nv() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    const char* u = getenv("GPU_LEASE_UUID");
    ok = (u && nvmlDeviceGetHandleByUUID(u, &d) == NVML_SUCCESS) || nvmlDeviceGetHandleByIndex_v2(0, &d) == NVML_SUCCESS;
  }
  double sm() {
    unsigned c = 0;
    if (ok) nvmlDeviceGetClockInfo(d, NVML_CLOCK_SM, &c);
    return c;
  }
};

template <int FORM, int ROLL>
static void time_form(const char* name, const u64* tail, u64* out, int launches, int threads, int blocks, int reps, Nv& nv,
                      const char* extra = "") {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> ms, mhz;
  for (int i = 0; i <= launches; ++i) {
    CK(cudaEventRecord(e0));
    forms<FORM, ROLL><<<blocks, threads>>>(tail, out, reps);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float m;
    CK(cudaEventElapsedTime(&m, e0, e1));
    if (i) { ms.push_back(m); mhz.push_back(nv.sm()); }
  }
  const double comps = (double)threads * blocks * reps, tm = median(ms);
  printf("{\"form\": \"%s\", \"roll\": %d, \"compressions\": %.0f, \"ms\": %.5f, \"gcomp_per_s\": %.4f, \"mhz\": %.0f%s}\n", name,
         ROLL, comps, tm, comps / tm * 1e-6, median(mhz), extra);
  fflush(stdout);
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
}

// the form at `threads` a CTA, in WAVES whole waves of the CTAs an SM holds at once (no partial last wave)
template <int FORM, int ROLL>
static void time_shape(const char* name, const u64* tail, u64* out, int launches, int threads, int sms, int reps, Nv& nv) {
  constexpr int WAVES = 4;
  int per_sm = 0;
  cudaFuncAttributes fa;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, forms<FORM, ROLL>, threads, 0));
  CK(cudaFuncGetAttributes(&fa, forms<FORM, ROLL>));
  if (per_sm < 1) return;
  char extra[160];
  snprintf(extra, sizeof extra, ", \"threads\": %d, \"ctas_an_sm\": %d, \"warps_an_sm\": %d, \"regs\": %d, \"waves\": %d", threads,
           per_sm, per_sm * threads / 32, fa.numRegs, WAVES);
  time_form<FORM, ROLL>(name, tail, out, launches, threads, sms * per_sm * WAVES, reps, nv, extra);
}

static const char* USAGE = "usage: hashcat_forms time LAUNCHES | shapes LAUNCHES THREADS... | check\n";

int main(int argc, char** argv) {
  if (argc < 2) { fputs(USAGE, stderr); return 2; }
  int dev;
  cudaDeviceProp prop;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  const int threads = 256, blocks = prop.multiProcessorCount * 8, reps = 256;
  const u64 tail_h[7] = {0, 0, 0, 0, 0, 0, 56};  // a 7-byte candidate: W1..W6 zero, W15 its bit length
  u64 *tail, *out;
  CK(cudaMalloc(&tail, sizeof tail_h));
  CK(cudaMemcpy(tail, tail_h, sizeof tail_h, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&out, (size_t)prop.multiProcessorCount * 2048 * 4 * 8));
  if (!strcmp(argv[1], "shapes") && argc >= 4) {
    const int launches = atoi(argv[2]), sms = prop.multiProcessorCount;
    Nv nv;
    for (int a = 3; a < argc; ++a) {
      const int th = atoi(argv[a]);
      time_shape<CHAIN, 1>("chain", tail, out, launches, th, sms, reps, nv);
      time_shape<CHAIN, 0>("chain", tail, out, launches, th, sms, reps, nv);
      time_shape<SKIP, 0>("skip", tail, out, launches, th, sms, reps, nv);
    }
    return 0;
  }
  if (!strcmp(argv[1], "check")) {
    first<<<1, 2>>>(tail, out);
    CK(cudaGetLastError());
    u64 h[16];
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    printf("{\"sms\": %d, \"device\": \"%s\"}\n", prop.multiProcessorCount, prop.name);
    for (int t = 0; t < 2; ++t) {
      printf("{\"thread\": %d, \"candidate\": %llu, \"digest\": \"", t, (unsigned long long)((u64)t << 20));
      for (int i = 0; i < 8; ++i) printf("%016llx", (unsigned long long)h[t * 8 + i]);
      printf("\"}\n");
    }
    return 0;
  }
  if (strcmp(argv[1], "time") || argc < 3) { fputs(USAGE, stderr); return 2; }
  const int launches = atoi(argv[2]);
  Nv nv;
  time_form<CHAIN, 1>("chain", tail, out, launches, threads, blocks, reps, nv);
  time_form<CHAIN, 0>("chain", tail, out, launches, threads, blocks, reps, nv);
  time_form<IV, 1>("iv", tail, out, launches, threads, blocks, reps, nv);
  time_form<IV, 0>("iv", tail, out, launches, threads, blocks, reps, nv);
  time_form<SKIP, 1>("skip", tail, out, launches, threads, blocks, reps, nv);
  time_form<SKIP, 0>("skip", tail, out, launches, threads, blocks, reps, nv);
  return 0;
}
