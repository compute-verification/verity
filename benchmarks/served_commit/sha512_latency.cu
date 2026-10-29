// One SHA-512 compression's latency on the H200, through the committer's own `sha::compress` (sha512.cuh): one thread (or
// one per lane, or one warp an SM) runs n compressions one after another and times them with the SM clock, and the kernel's
// wall time is taken for n = 1 (the launch floor) and larger n. Mode 0 keeps each block's message independent of the state (a
// row's Merkle–Damgård chain); mode 1 feeds the state back as the next message (a tree path, a seed from a root). Prints one
// JSON object.
//
//     nvcc -O3 -std=c++17 -arch=sm_90a -DSHA_ROLL=1 -I integrations/vllm/verity_vllm/commit/committer sha512_latency.cu
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "sha512.cuh"

__global__ void chain(uint64_t* out, long long* cycles, int n, int mode) {
  uint64_t st[8], w[16];
  sha::iv(st);
  const uint64_t seed = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  long long t0 = clock64();
#pragma unroll 1
  for (int b = 0; b < n; ++b) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = mode ? st[i & 7] ^ (uint64_t)i : seed * 0x9E3779B97F4A7C15ull + (uint64_t)(b * 16 + i);
    sha::compress(st, w);
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
#pragma unroll
  for (int i = 0; i < 8; ++i) out[seed * 8 + i] = st[i];
}

static float wall_us(int grid, int block, int n, int mode, uint64_t* out, long long* cyc, int reps) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  chain<<<grid, block>>>(out, cyc, n, mode);
  cudaDeviceSynchronize();
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    chain<<<grid, block>>>(out, cyc, n, mode);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    if (ms * 1e3f < best) best = ms * 1e3f;
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

int main() {
  uint64_t* out;
  long long* cyc;
  cudaMalloc(&out, 132 * 32 * 8 * sizeof(uint64_t));
  cudaMalloc(&cyc, 132 * sizeof(long long));
  int mhz = 0;
  cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0);
  printf("{\"sha_roll\": %d, \"max_clock_khz\": %d, \"rows\": [\n", SHA_ROLL, mhz);
  const int shapes[3][2] = {{1, 1}, {1, 32}, {132, 32}};
  const int ns[4] = {1, 16, 64, 256};
  bool first = true;
  for (int mode = 0; mode < 2; ++mode)
    for (auto& s : shapes)
      for (int n : ns) {
        float us = wall_us(s[0], s[1], n, mode, out, cyc, 20);
        long long c[132];
        cudaMemcpy(c, cyc, s[0] * sizeof(long long), cudaMemcpyDeviceToHost);
        long long cmax = 0;
        for (int i = 0; i < s[0]; ++i) cmax = c[i] > cmax ? c[i] : cmax;
        printf("%s{\"mode\": %d, \"grid\": %d, \"block\": %d, \"n\": %d, \"wall_us_best\": %.3f, \"cycles_max\": %lld}",
               first ? "" : ",\n", mode, s[0], s[1], n, us, cmax);
        first = false;
      }
  cudaError_t e = cudaGetLastError();
  printf("\n], \"cuda_error\": \"%s\"}\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
