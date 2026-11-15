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

// the SMs idle (asleep) for `cycles` of their own clock, one CTA an SM, each adding the clocks it spent to acc[smid]: rests
// between a GEMM's calls lower the card's mean power, so its clock rises toward a served step's
extern "C" __global__ void rest_k(long long cycles, unsigned long long* acc) {
  extern __shared__ uint8_t keep_off[];
  if (threadIdx.x == 0) {
    keep_off[0] = 0;
    const long long c0 = clock64();
    while (clock64() - c0 < cycles) __nanosleep(500);
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    atomicAdd(acc + sm, static_cast<unsigned long long>(clock64() - c0));
  }
}

extern "C" int stamp(void* out, int sms, void* stream) {
  if (cudaFuncSetAttribute(stamp_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAMP_SMEM) != cudaSuccess) return 1;
  stamp_k<<<sms, 32, STAMP_SMEM, static_cast<cudaStream_t>(stream)>>>(static_cast<unsigned long long*>(out));
  return cudaGetLastError() != cudaSuccess;
}

extern "C" int rest(long long cycles, void* acc, int sms, void* stream) {
  if (cudaFuncSetAttribute(rest_k, cudaFuncAttributeMaxDynamicSharedMemorySize, STAMP_SMEM) != cudaSuccess) return 1;
  rest_k<<<sms, 32, STAMP_SMEM, static_cast<cudaStream_t>(stream)>>>(cycles, static_cast<unsigned long long*>(acc));
  return cudaGetLastError() != cudaSuccess;
}
