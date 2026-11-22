// The fill kernel (stream.cuh), launched on the caller's stream; built by engine.device_lib with nvcc. sxs_fill returns 0,
// a refusal (stream.cuh's REFUSED_*, below 0) before any launch, or the CUDA error of its launch. A stream that is
// capturing a graph is refused: a replay would write the captured counters again.
#include <cuda_runtime.h>

#include "stream.cuh"

using namespace sxs;

namespace {

constexpr u32 THREADS = 256;
constexpr u64 GRID_MAX = 1ull << 20;

struct D {
  static __device__ __forceinline__ void compress(u64* st, u64* w) { sx::compress<0, 1>(st, w); }
};

__global__ void __launch_bounds__(THREADS) fill(const Head h, u64 c0, u64 n, u64* out) {
  for (u64 t = (u64)blockIdx.x * THREADS + threadIdx.x; t < n; t += (u64)gridDim.x * THREADS) {
    u64 st[8];
    block<D>(h, c0 + t, st);
#pragma unroll
    for (int k = 0; k < 8; ++k) out[8 * t + k] = sx::bswap64(st[k]);
  }
}

}  // namespace

extern "C" {

// out[192 i .. 192 i + 192) = blocks 3 (first + i) .. 3 (first + i) + 2 at `head` for i < leaves; out is the device's, its
// address a multiple of 8
int sxs_fill(void* stream, const uint8_t* head, uint32_t len, uint64_t first, uint64_t leaves, uint8_t* out) {
  Head h;
  if (head_of(head, len, h)) return REFUSED_HEAD;
  if (!range_ok(first, leaves)) return REFUSED_RANGE;
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing((cudaStream_t)stream, &capture) != cudaSuccess) return (int)cudaGetLastError();
  if (capture != cudaStreamCaptureStatusNone) return REFUSED_CAPTURE;
  if (leaves == 0) return 0;
  const u64 n = LEAF_BLOCKS * leaves, blocks = (n + THREADS - 1) / THREADS;
  fill<<<(unsigned)(blocks < GRID_MAX ? blocks : GRID_MAX), THREADS, 0, (cudaStream_t)stream>>>(
      h, LEAF_BLOCKS * first, n, reinterpret_cast<u64*>(out));
  return (int)cudaGetLastError();
}

}  // extern "C"
