// wgbench.py's clock: each SM's %clock64 and %globaltimer, one CTA an SM (its shared memory keeps a second CTA off), so
// the SM clock over a span of launches on the same stream is the difference of two stamps.
#include <cstdint>
#include <cuda_runtime.h>

constexpr int STAMP_SMEM = 150 * 1024;

extern "C" __global__ void stamp_k(unsigned long long* out) {
  extern __shared__ uint8_t keep_off[];
  unsigned long long c, t;
  uint32_t sm;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  if (threadIdx.x == 0) {
    keep_off[0] = 0;
    out[3 * blockIdx.x] = sm;
    out[3 * blockIdx.x + 1] = c;
    out[3 * blockIdx.x + 2] = t;
  }
}

extern "C" int stamp(void* out, int sms, void* stream) {
  if (cudaFuncSetAttribute(stamp_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAMP_SMEM) != cudaSuccess) return 1;
  stamp_k<<<sms, 32, STAMP_SMEM, static_cast<cudaStream_t>(stream)>>>(static_cast<unsigned long long*>(out));
  return cudaGetLastError() != cudaSuccess;
}
