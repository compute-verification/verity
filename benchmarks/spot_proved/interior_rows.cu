// `interior_rows.cuh` on the device, enqueued on the caller's stream (`interior_gpu.Interior`): one launch a batch of pieces'
// leaves, then one a tree level over every port's pieces.
#include <cuda_runtime.h>

#include "interior_rows.cuh"

using namespace introws;

__global__ void k_midstates(const uint64_t* prefixes, uint64_t* ms) {
  if (blockIdx.x == 0 && threadIdx.x == 0) midstates(prefixes, ms);
}

__global__ void k_leaves(const Seg* s, uint32_t S, uint64_t total, const uint64_t* hm, const uint64_t* ms) {
  const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total) return;
  uint64_t m[16];
  for (int k = 0; k < 16; ++k) m[k] = ms[k];
  seg_leaf(s, S, g, hm, m);
}

__global__ void k_nodes(const Seg* s, uint32_t S, uint64_t total) {
  const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g < total) seg_node(s, S, g);
}

static unsigned blocks(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// The midstates after the two prefix blocks (``prefixes``, 32 words) into ``ms`` (16 words), both device pointers.
extern "C" int ir_midstates(void* stream, const uint64_t* prefixes, uint64_t* ms) {
  k_midstates<<<1, 32, 0, (cudaStream_t)stream>>>(prefixes, ms);
  return (int)cudaGetLastError();
}

// The ``total`` rows of the ``S`` pieces ``s`` (a device array): each row's b || c and leaf. ``hm`` (192 x 256 x 8 words)
// and ``ms`` (16) are device pointers.
extern "C" int ir_leaves(void* stream, const Seg* s, uint32_t S, uint64_t total, const uint64_t* hm, const uint64_t* ms) {
  if (S == 0 || total == 0) return -1;
  k_leaves<<<blocks(total, 64), 64, 0, (cudaStream_t)stream>>>(s, S, total, hm, ms);
  return (int)cudaGetLastError();
}

// The ``total`` pads or nodes of one tree level over the ``S`` pieces ``s`` (a device array), once the level below is
// enqueued.
extern "C" int ir_nodes(void* stream, const Seg* s, uint32_t S, uint64_t total) {
  if (S == 0 || total == 0) return -1;
  k_nodes<<<blocks(total, 128), 128, 0, (cudaStream_t)stream>>>(s, S, total);
  return (int)cudaGetLastError();
}
