// Axiom probe: is device memory one shared budget for the SMs and the copy engines? Times, best of 5:
//   read:        an SM read kernel over 8 GiB (128-bit non-coherent loads);
//   d2d:         cudaMemcpyAsync device to device, 2 GiB (bytes read and bytes written reported);
//   read+d2d:    both at once on two streams, total DRAM bytes moved over the union's time;
//   d2h:         cudaMemcpyAsync device to pinned host, 1 GiB (PCIe, a copy engine);
//   read+d2h:    both at once.
// Falsifies "every DRAM byte any engine moves is charged to one per-device budget" if a union beats the best single
// route's DRAM bytes a second by more than a few percent.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* sink, int reps) {
  unsigned acc = 0;
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < reps; r++)
    for (size_t i = tid; i < n; i += nth) { uint4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void k_fill(uint4* p, size_t n) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nth = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += nth) {
    unsigned h = (unsigned)i * 0x9e3779b9u;
    p[i] = make_uint4(h ^ 0x85ebca6bu, h * 3u, h ^ (h >> 13), ~h);
  }
}

int main() {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const size_t RB = 8ull << 30, CB = 2ull << 30, HB = 1ull << 30;
  uint4 *r, *cs, *cd;
  void* h;
  unsigned* sink;
  CK(cudaMalloc(&r, RB)); CK(cudaMalloc(&cs, CB)); CK(cudaMalloc(&cd, CB)); CK(cudaMalloc(&sink, 4));
  CK(cudaMallocHost(&h, HB));
  k_fill<<<sms * 4, 512>>>(r, RB / 16); k_fill<<<sms * 4, 512>>>(cs, CB / 16);
  CK(cudaDeviceSynchronize());
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t a0, a1, b0, b1;
  CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1)); CK(cudaEventCreate(&b0)); CK(cudaEventCreate(&b1));
  // mode: 0 read, 1 d2d, 2 read+d2d, 3 d2h, 4 read+d2h. Bytes are DRAM bytes: a d2d copy reads and writes CB each.
  const char* names[5] = {"read", "d2d", "read+d2d", "d2h", "read+d2h"};
  for (int mode = 0; mode < 5; mode++) {
    double best = 1e30, bytes = 0, ms_read = 0, ms_copy = 0;
    for (int rep = 0; rep < 6; rep++) {
      bool rd = mode == 0 || mode == 2 || mode == 4, cp = mode != 0;
      int reads = mode == 0 ? 1 : 3;  // under a copy, the read kernel runs long enough to cover it
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(a0, s1)); CK(cudaEventRecord(b0, s2));
      if (rd) k_read<<<sms * 8, 512, 0, s1>>>(r, RB / 16, sink, reads);
      if (cp && (mode == 1 || mode == 2)) CK(cudaMemcpyAsync(cd, cs, CB, cudaMemcpyDeviceToDevice, s2));
      if (cp && (mode == 3 || mode == 4)) CK(cudaMemcpyAsync(h, cs, HB, cudaMemcpyDeviceToHost, s2));
      CK(cudaEventRecord(a1, s1)); CK(cudaEventRecord(b1, s2));
      CK(cudaEventSynchronize(a1)); CK(cudaEventSynchronize(b1));
      CK(cudaGetLastError());
      float t_r = 0, t_c = 0;
      if (rd) CK(cudaEventElapsedTime(&t_r, a0, a1));
      if (cp) CK(cudaEventElapsedTime(&t_c, b0, b1));
      if (rep == 0) continue;
      double copy_bytes = (mode == 1 || mode == 2) ? 2.0 * CB : (mode >= 3 ? (double)HB : 0);
      double read_bytes = rd ? (double)RB * reads : 0;
      // the union's DRAM bytes over the union's span (both streams start together): a lower bound on the joint rate
      double span = t_r > t_c ? t_r : t_c;
      double rate = (read_bytes + copy_bytes) / (span / 1e3);
      if (1.0 / rate < best) { best = 1.0 / rate; bytes = read_bytes + copy_bytes; ms_read = t_r; ms_copy = t_c; }
    }
    printf("{\"probe\":\"dram\",\"mode\":\"%s\",\"dram_bytes\":%.0f,\"read_ms\":%.3f,\"copy_ms\":%.3f,"
           "\"dram_bytes_per_s\":%.4e}\n", names[mode], bytes, ms_read, ms_copy, 1.0 / best);
  }
  return 0;
}
