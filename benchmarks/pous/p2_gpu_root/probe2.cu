// Latencies inside one SM that bound a squaring step's phases: __syncthreads for 128..1024 threads, a dependent chain of
// shared loads, and a dependent chain of IMAD.WIDE. One CTA; device cycles (clock64). Output: JSON lines.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s\n", cudaGetErrorString(e_)); return 1; } } while (0)

__global__ void bar(int n, unsigned long long *o) {
  uint64_t c0 = clock64();
  for (int t = 0; t < n; t++) __syncthreads();
  uint64_t c1 = clock64();
  if (threadIdx.x == 0) o[0] = c1 - c0;
}
__global__ void lds_chain(int n, unsigned long long *o) {
  __shared__ uint32_t s[1024];
  for (int q = threadIdx.x; q < 1024; q += blockDim.x) s[q] = (q * 33 + 7) & 1023;
  __syncthreads();
  uint32_t x = threadIdx.x;
  uint64_t c0 = clock64();
  for (int t = 0; t < n; t++) x = s[x];
  uint64_t c1 = clock64();
  if (threadIdx.x == 0) { o[0] = c1 - c0; o[1] = x; }
}
__global__ void imad_chain(int n, uint32_t a, unsigned long long *o) {
  uint64_t acc = threadIdx.x;
  uint64_t c0 = clock64();
  for (int t = 0; t < n; t++) asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(acc) : "r"(a), "r"((uint32_t)acc));
  uint64_t c1 = clock64();
  if (threadIdx.x == 0) { o[0] = c1 - c0; o[1] = acc; }
}
__global__ void shfl_chain(int n, unsigned long long *o) {
  uint32_t x = threadIdx.x;
  uint64_t c0 = clock64();
  for (int t = 0; t < n; t++) x = __shfl_xor_sync(0xffffffffu, x, 1) + 1;
  uint64_t c1 = clock64();
  if (threadIdx.x == 0) { o[0] = c1 - c0; o[1] = x; }
}

int main() {
  unsigned long long *d, h[2]; CK(cudaMalloc(&d, 16));
  const int n = 100000;
  for (int t : {32, 128, 256, 512, 1024}) {
    bar<<<1, t>>>(n, d); bar<<<1, t>>>(n, d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"syncthreads\",\"threads\":%d,\"cyc\":%.2f}\n", t, (double)h[0] / n);
  }
  for (int t : {32, 1024}) {
    lds_chain<<<1, t>>>(n, d); lds_chain<<<1, t>>>(n, d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"lds_dependent_chain\",\"threads\":%d,\"cyc\":%.2f}\n", t, (double)h[0] / n);
    imad_chain<<<1, t>>>(n, 3, d); imad_chain<<<1, t>>>(n, 3, d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"imad_wide_dependent_chain\",\"threads\":%d,\"cyc\":%.2f}\n", t, (double)h[0] / n);
    shfl_chain<<<1, t>>>(n, d); shfl_chain<<<1, t>>>(n, d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"shfl_dependent_chain\",\"threads\":%d,\"cyc\":%.2f}\n", t, (double)h[0] / n);
  }
  return 0;
}
