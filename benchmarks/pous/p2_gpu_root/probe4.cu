// Tensor-core rates on this SM: mma.sync int8 (m16n8k32, s8 x s8 -> s32, 4,096 MACs) and FP64 (m8n8k4, 256 MACs), each
// as a throughput (W warps, 4 independent accumulators per warp) and as a dependent chain (one warp, one accumulator).
// Device cycles (clock64). Output: JSON lines.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f64(double (&d)[2], double a, double b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[0]), "+d"(d[1]) : "d"(a), "d"(b));
}

template <int ACC>
__global__ void k_s8(int n, unsigned long long *out, int *sink) {
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x01020304u ^ lane, 0x05060708u, 0x090a0b0cu, 0x0d0e0f10u}, b[2] = {0x11121314u, 0x15161718u ^ lane};
  int d[ACC][4] = {};
  unsigned long long c0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < n; i++) {
    #pragma unroll
    for (int q = 0; q < ACC; q++) mma_s8(d[q], a, b);
  }
  unsigned long long c1 = clock64();
  int s = 0;
  for (int q = 0; q < ACC; q++) s += d[q][0] + d[q][1] + d[q][2] + d[q][3];
  if (threadIdx.x == 0) out[0] = c1 - c0;
  if (s == 0x7fffffff) *sink = s;
}

template <int ACC>
__global__ void k_f64(int n, unsigned long long *out, double *sink) {
  const int lane = threadIdx.x & 31;
  double a = 1.0 + lane * 1e-3, b = 1.0 - lane * 1e-3;
  double d[ACC][2] = {};
  unsigned long long c0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < n; i++) {
    #pragma unroll
    for (int q = 0; q < ACC; q++) mma_f64(d[q], a, b);
  }
  unsigned long long c1 = clock64();
  double s = 0;
  for (int q = 0; q < ACC; q++) s += d[q][0] + d[q][1];
  if (threadIdx.x == 0) out[0] = c1 - c0;
  if (s == -1.0) *sink = s;
}

int main() {
  unsigned long long *dout; int *si; double *sd;
  CK(cudaMalloc(&dout, 8)); CK(cudaMalloc(&si, 4)); CK(cudaMalloc(&sd, 8));
  const int n = 4096;
  unsigned long long h;
  for (int w : {1, 4, 8, 16}) {
    k_s8<4><<<1, 32 * w>>>(n, dout, si); CK(cudaDeviceSynchronize());
    k_s8<4><<<1, 32 * w>>>(n, dout, si); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"mma_s8_m16n8k32\",\"warps\":%d,\"macs_per_sm_per_clk\":%.1f}\n", w, 4096.0 * 4 * n * w / h);
    k_f64<4><<<1, 32 * w>>>(n, dout, sd); CK(cudaDeviceSynchronize());
    k_f64<4><<<1, 32 * w>>>(n, dout, sd); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"mma_f64_m8n8k4\",\"warps\":%d,\"macs_per_sm_per_clk\":%.2f}\n", w, 256.0 * 4 * n * w / h);
  }
  k_s8<1><<<1, 32>>>(n, dout, si); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
  printf("{\"probe\":\"mma_s8_m16n8k32_dependent\",\"cyc\":%.2f}\n", (double)h / n);
  k_f64<1><<<1, 32>>>(n, dout, sd); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
  printf("{\"probe\":\"mma_f64_m8n8k4_dependent\",\"cyc\":%.2f}\n", (double)h / n);
  return 0;
}
