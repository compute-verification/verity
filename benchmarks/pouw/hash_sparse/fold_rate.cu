// The multiply-xor fold N <- (N ^ D_t) k_t on one H200, timed in an int8 MMA loop: each k32 step's accumulator words
// are the D_t, k_t a per-step odd u32 read from global memory. Variants, each the same grid and steps:
//   mma        mma.sync m16n8k32 s8 on TILES independent accumulator sets a warp, no fold
//   mma_fold   the same, folding every accumulator word after each step (LOP3 + IMUL a word)
//   word_fold  no MMA: the zero-data word D = R + C (IADD3), then the fold (three ops a word)
//   word       no MMA: the IADD3 alone
// Prints one JSON line a variant: ms (median of REPS), words a second, and the checksum of the states.
// usage: fold_rate STEPS REPS
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

constexpr int TILES = 8;
constexpr int WARPS = 8;

__device__ __forceinline__ void mma(uint32_t* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int V>
__global__ void __launch_bounds__(32 * WARPS) k_fold(const uint32_t* __restrict__ ks, const uint32_t* __restrict__ seed,
                                                     uint32_t* __restrict__ out, int steps) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4], b[2], c[TILES][4], n[TILES][4];
  for (int i = 0; i < 4; i++) a[i] = seed[(tid * 7 + i) & 1023];
  for (int i = 0; i < 2; i++) b[i] = seed[(tid * 5 + 11 + i) & 1023];
#pragma unroll
  for (int t = 0; t < TILES; t++)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      c[t][i] = seed[(tid + 13 * t + i) & 1023];
      n[t][i] = seed[(tid * 3 + 17 * t + i) & 1023];
    }
  const uint32_t r = seed[tid & 1023];
  for (int s = 0; s < steps; s++) {
    const uint32_t k = __ldg(ks + s);
    if (V == 0 || V == 1) {
#pragma unroll
      for (int t = 0; t < TILES; t++) mma(c[t], a, b);
    }
#pragma unroll
    for (int t = 0; t < TILES; t++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (V == 2 || V == 3) c[t][i] = c[t][i] + r + k;
        if (V == 1 || V == 2) n[t][i] = (n[t][i] ^ c[t][i]) * k;
      }
  }
  uint32_t acc = 0;
#pragma unroll
  for (int t = 0; t < TILES; t++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc ^= (V == 1 || V == 2) ? n[t][i] : c[t][i];
  out[tid] = acc;
}

template <int V>
static void run(const char* name, int blocks, int steps, int reps, const uint32_t* ks, const uint32_t* seed,
                uint32_t* out) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  k_fold<V><<<blocks, 32 * WARPS>>>(ks, seed, out, steps);
  cudaDeviceSynchronize();
  std::vector<float> ms;
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(e0);
    k_fold<V><<<blocks, 32 * WARPS>>>(ks, seed, out, steps);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float m;
    cudaEventElapsedTime(&m, e0, e1);
    ms.push_back(m);
  }
  std::sort(ms.begin(), ms.end());
  const size_t threads = (size_t)blocks * 32 * WARPS;
  std::vector<uint32_t> h(threads);
  cudaMemcpy(h.data(), out, threads * 4, cudaMemcpyDeviceToHost);
  uint64_t sum = 0;
  for (auto v : h) sum += v;
  const double words = (double)threads * TILES * 4 * steps;
  const double macs = (V == 0 || V == 1) ? (double)blocks * WARPS * TILES * 4096.0 * steps : 0.0;
  const float med = ms[ms.size() / 2];
  printf("{\"variant\":\"%s\",\"blocks\":%d,\"steps\":%d,\"ms\":%.4f,\"ms_min\":%.4f,\"words_s\":%.4e,\"macs_s\":%.4e,"
         "\"checksum\":%llu,\"err\":\"%s\"}\n",
         name, blocks, steps, med, ms[0], words / (med * 1e-3), macs / (med * 1e-3), (unsigned long long)sum,
         cudaGetErrorString(cudaGetLastError()));
}

int main(int argc, char** argv) {
  const int steps = argc > 1 ? atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? atoi(argv[2]) : 7;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  std::vector<uint32_t> hk(steps), hs(1024);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (auto& v : hk) { x = x * 6364136223846793005ull + 1442695040888963407ull; v = (uint32_t)(x >> 32) | 1u; }
  for (auto& v : hs) { x = x * 6364136223846793005ull + 1442695040888963407ull; v = (uint32_t)(x >> 32); }
  uint32_t *ks, *seed, *out;
  cudaMalloc(&ks, steps * 4);
  cudaMalloc(&seed, 1024 * 4);
  cudaMemcpy(ks, hk.data(), steps * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(seed, hs.data(), 1024 * 4, cudaMemcpyHostToDevice);
  for (int per_sm : {1, 2, 4}) {
    const int blocks = sms * per_sm;
    cudaMalloc(&out, (size_t)blocks * 32 * WARPS * 4);
    run<0>("mma", blocks, steps, reps, ks, seed, out);
    run<1>("mma_fold", blocks, steps, reps, ks, seed, out);
    run<2>("word_fold", blocks, steps, reps, ks, seed, out);
    run<3>("word", blocks, steps, reps, ks, seed, out);
    cudaFree(out);
  }
  return 0;
}
