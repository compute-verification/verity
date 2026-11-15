// The units' receipts on the device (receipts.cuh), one thread a unit, on the caller's stream; built by sketch.py with
// nvcc. p2r_receipts returns 0, or the CUDA error of its launch.
#include <cuda_runtime.h>

#include "receipts.cuh"

#ifndef SX_V
#define SX_V 0
#endif
#ifndef SX_ROLL
#define SX_ROLL 1
#endif

namespace {

constexpr int THREADS = 256;

struct Mid {
  uint64_t w[8];
};

__global__ void __launch_bounds__(THREADS) k_receipts(const Mid mid, const uint64_t* __restrict__ sigma, uint64_t g0,
                                                      uint64_t n, uint64_t* __restrict__ y) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  uint64_t s[8], d[8];
  const ulonglong2* in = reinterpret_cast<const ulonglong2*>(sigma + 8 * t);
  for (int i = 0; i < 4; ++i) {
    const ulonglong2 v = __ldg(in + i);
    s[2 * i] = v.x;
    s[2 * i + 1] = v.y;
  }
  p2r::receipt<SX_V, SX_ROLL>(mid.w, g0 + t, s, d);
  ulonglong2* out = reinterpret_cast<ulonglong2*>(y + 8 * t);
  for (int i = 0; i < 4; ++i) out[i] = make_ulonglong2(sx::bswap64(d[2 * i]), sx::bswap64(d[2 * i + 1]));
}

}  // namespace

extern "C" {

void p2r_midstate(const uint8_t* block, uint64_t* out) { p2r::midstate(block, out); }

// y[t] (64 bytes) = unit g0 + t's receipt over sigma[t] (64 bytes), t < n; sigma and y 16-byte aligned on the device
int p2r_receipts(void* stream, const uint64_t* mid, const uint64_t* sigma, uint64_t g0, uint64_t n, uint64_t* y) {
  Mid m;
  for (int i = 0; i < 8; ++i) m.w[i] = mid[i];
  if (n) k_receipts<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS, 0, (cudaStream_t)stream>>>(m, sigma, g0, n, y);
  return (int)cudaGetLastError();
}

}
