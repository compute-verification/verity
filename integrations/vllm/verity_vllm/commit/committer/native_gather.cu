// One gather-copy kernel per legal collection point: the staged tensors of a window are copied into the arena by ONE launch instead of one cudaMemcpyAsync each.  Descriptors live in a
// host-mapped pinned ring; the collector copies a flush's descriptors to a device ring ahead of the launch (or, capturing a graph, passes
// the host-mapped ring, read over PCIe).  src == nullptr means "write zeros" (alignment gaps).
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
  __shared__ GatherDesc gs;                // one read of the descriptor a block, not one a warp
  if (threadIdx.x == 0) gs = d[blockIdx.y];
  __syncthreads();
  const GatherDesc g = gs;
  const int64_t nb = g.nbytes;
  const bool contiguous = (g.pitch == g.width);
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nthr = static_cast<int64_t>(gridDim.x) * blockDim.x;
  if (g.src == nullptr) {
    for (int64_t i = tid; i < nb; i += nthr) g.dst[i] = 0;
    return;
  }
  const uintptr_t sa = reinterpret_cast<uintptr_t>(g.src) & 15, da = reinterpret_cast<uintptr_t>(g.dst) & 15;
  if (contiguous && sa == da) {
    // 16 B words between a byte head (up to the first 16 B boundary of both) and a byte tail: the stream packs tensors at 4 B, so after
    // the first tensor whose length is no multiple of 16 most destinations are not 16 B aligned
    const int64_t head = static_cast<int64_t>((16 - da) & 15) < nb ? static_cast<int64_t>((16 - da) & 15) : nb;
    const int64_t n16 = (nb - head) / 16;
    if (tid < head) g.dst[tid] = g.src[tid];
    const uint4* s = reinterpret_cast<const uint4*>(g.src + head);
    uint4* t = reinterpret_cast<uint4*>(g.dst + head);
    for (int64_t i = tid; i < n16; i += nthr) t[i] = s[i];
    for (int64_t i = head + n16 * 16 + tid; i < nb; i += nthr) g.dst[i] = g.src[i];
    return;
  }
  if (contiguous && ((sa | da) & 3) == 0) {
    // offsets differ by r 4 B words mod 16: 16 B stores to the aligned destination, each joined from two aligned 16 B source loads (the
    // second is the next thread's first, so DRAM reads each byte once).  Both loaded 16 B blocks hold a byte of the source range: no
    // read leaves its pages
    const int64_t head = static_cast<int64_t>((16 - da) & 15) < nb ? static_cast<int64_t>((16 - da) & 15) : nb;
    const int64_t n16 = (nb - head) / 16;
    if (tid < head) g.dst[tid] = g.src[tid];
    const int r = static_cast<int>(((sa + head) & 15) >> 2);
    const uint4* s = reinterpret_cast<const uint4*>(g.src + head - 4 * r);
    uint4* t = reinterpret_cast<uint4*>(g.dst + head);
    if (r == 1) {
      for (int64_t i = tid; i < n16; i += nthr) { const uint4 a = s[i], b = s[i + 1]; t[i] = make_uint4(a.y, a.z, a.w, b.x); }
    } else if (r == 2) {
      for (int64_t i = tid; i < n16; i += nthr) { const uint4 a = s[i], b = s[i + 1]; t[i] = make_uint4(a.z, a.w, b.x, b.y); }
    } else {
      for (int64_t i = tid; i < n16; i += nthr) { const uint4 a = s[i], b = s[i + 1]; t[i] = make_uint4(a.w, b.x, b.y, b.z); }
    }
    for (int64_t i = head + n16 * 16 + tid; i < nb; i += nthr) g.dst[i] = g.src[i];
    return;
  }
  if (contiguous) {
    for (int64_t i = tid; i < nb; i += nthr) g.dst[i] = g.src[i];
    return;
  }
  // 2-D pitched source (RoPE q/k views in prefill): logical byte i -> row i / width, col i % width
  const bool row16 = (g.skip % 16 == 0) && (g.width % 16 == 0) && (g.pitch % 16 == 0) && sa == 0 && da == 0 && (nb % 16 == 0);
  const bool row4 = (g.skip % 4 == 0) && (g.width % 4 == 0) && (g.pitch % 4 == 0) && ((sa | da) & 3) == 0 && (nb % 4 == 0);
  if (row16) {
    const int64_t wv = g.width / 16, sv = g.skip / 16;
    for (int64_t v = tid; v < nb / 16; v += nthr) {
      const int64_t j = v + sv;
      const int64_t row = j / wv, col = j - row * wv;
      *reinterpret_cast<uint4*>(g.dst + v * 16) = *reinterpret_cast<const uint4*>(g.src + row * g.pitch + col * 16);
    }
  } else if (row4) {
    const int64_t wv = g.width / 4, sv = g.skip / 4;
    for (int64_t v = tid; v < nb / 4; v += nthr) {
      const int64_t j = v + sv;
      const int64_t row = j / wv, col = j - row * wv;
      *reinterpret_cast<uint32_t*>(g.dst + v * 4) = *reinterpret_cast<const uint32_t*>(g.src + row * g.pitch + col * 4);
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
