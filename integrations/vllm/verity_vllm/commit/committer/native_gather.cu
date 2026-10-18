// R10 acq-native, SECONDARY bet: one gather-copy kernel per legal collection point (owner 2's lifetime table, out/gen/r10/acq-life/lifetimes.md):
// the staged tensors of a window are copied into the arena by ONE launch instead of one cudaMemcpyAsync each.  Descriptors live in a
// host-mapped pinned ring (the kernel reads them over PCIe; no H2D copy call).  src == nullptr means "write zeros" (alignment gaps).
#include <cuda_runtime.h>
#include <cstdint>

struct GatherDesc {
  const char* src;
  char* dst;
  int64_t nbytes;       // logical bytes (rows * width)
  int64_t width;        // contiguous row bytes (== nbytes when contiguous)
  int64_t pitch;        // source row pitch in bytes (== width when contiguous)
  int64_t skip;         // P0 windowed staging: logical byte offset of this segment inside the (pitched) tensor; 0 for whole tensors
};

__global__ void gather_copy_kernel(const GatherDesc* __restrict__ d, int n) {
  const GatherDesc g = d[blockIdx.y];
  const int64_t nb = g.nbytes;
  const bool contiguous = (g.pitch == g.width);
  const bool vec16 = contiguous && ((reinterpret_cast<uintptr_t>(g.src) & 15) == 0) && ((reinterpret_cast<uintptr_t>(g.dst) & 15) == 0) && (nb % 16 == 0);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nthr = static_cast<int64_t>(gridDim.x) * blockDim.x;
  if (g.src == nullptr) {
    for (int64_t i = tid; i < nb; i += nthr) g.dst[i] = 0;
    return;
  }
  if (vec16) {
    const uint4* s = reinterpret_cast<const uint4*>(g.src);
    uint4* t = reinterpret_cast<uint4*>(g.dst);
    for (int64_t i = tid; i < nb / 16; i += nthr) t[i] = s[i];
    return;
  }
  if (contiguous) {
    for (int64_t i = tid; i < nb; i += nthr) g.dst[i] = g.src[i];
    return;
  }
  // 2-D pitched source (RoPE q/k views in prefill): logical byte i -> row i / width, col i % width
  const bool row16 = (g.skip % 16 == 0) && (g.width % 16 == 0) && (g.pitch % 16 == 0) && ((reinterpret_cast<uintptr_t>(g.src) & 15) == 0) && ((reinterpret_cast<uintptr_t>(g.dst) & 15) == 0) && (nb % 16 == 0);
  if (row16) {
    const int64_t wv = g.width / 16, sv = g.skip / 16;
    for (int64_t v = tid; v < nb / 16; v += nthr) {
      const int64_t j = v + sv;
      const int64_t row = j / wv, col = j - row * wv;
      *reinterpret_cast<uint4*>(g.dst + v * 16) = *reinterpret_cast<const uint4*>(g.src + row * g.pitch + col * 16);
    }
  } else {
    for (int64_t i = tid; i < nb; i += nthr) {
      const int64_t j = i + g.skip;
      const int64_t row = j / g.width, col = j - row * g.width;
      g.dst[i] = g.src[row * g.pitch + col];
    }
  }
}

extern "C" int verity_gather_launch(const void* d_desc, int n, int blocks_per_desc, cudaStream_t stream) {
  if (n <= 0) return 0;
  dim3 grid(blocks_per_desc, n);
  gather_copy_kernel<<<grid, 256, 0, stream>>>(reinterpret_cast<const GatherDesc*>(d_desc), n);
  return static_cast<int>(cudaGetLastError());
}
