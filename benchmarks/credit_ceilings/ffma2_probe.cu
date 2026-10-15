// Packed FP32 FMA (fma.rn.f32x2, FFMA2 on sm_100) on sm_120: does it build, and what does it sustain beside FFMA?
// Built separately from credit_ceilings.cu so a toolkit that refuses it leaves the other routes running.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void __launch_bounds__(1024) k_ffma2(float* out, int iters, long long* cyc) {
  unsigned long long f[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float lo = (float)(threadIdx.x + r), hi = lo + 0.5f;
    f[r] = ((unsigned long long)__float_as_uint(hi) << 32) | __float_as_uint(lo);
  }
  const unsigned long long b = 0x3F8001003F800100ull, c = 0x3F0000003F000000ull;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int r = 0; r < 8; r++) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[r]) : "l"(b), "l"(c));
  }
  __syncthreads();
  if (threadIdx.x == 0) cyc[blockIdx.x] = clock64() - t0;
  unsigned long long s = 0;
#pragma unroll
  for (int r = 0; r < 8; r++) s ^= f[r];
  if (s == 0x7FFFFFFFull) out[0] = 1.f;
}

int main() {
  int nsm;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  float* out;
  long long* cyc;
  cudaMalloc(&out, 4);
  cudaMalloc(&cyc, sizeof(long long) * nsm);
  const int iters = 4096;
  k_ffma2<<<nsm, 1024>>>(out, iters, cyc);
  cudaDeviceSynchronize();
  k_ffma2<<<nsm, 1024>>>(out, iters, cyc);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<long long> h(nsm);
  cudaMemcpy(h.data(), cyc, sizeof(long long) * nsm, cudaMemcpyDeviceToHost);
  std::sort(h.begin(), h.end());
  double per_clk = 32.0 * 8 * iters / (double)h[nsm / 2];
  printf("{\"kind\":\"probe\",\"ops\":\"ffma2\",\"err\":\"%s\",\"warp_instr_per_sm_clk\":%.3f,\"mac_equiv_per_sm_clk\":%.1f}\n",
         cudaGetErrorString(e), per_clk, per_clk * 32 * 2);
  return 0;
}
