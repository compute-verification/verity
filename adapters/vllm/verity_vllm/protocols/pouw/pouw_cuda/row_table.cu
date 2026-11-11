// `row_table.cuh` on the device: a step's row tables hashed on the caller's stream, from device pointers only, so a serving loop
// launches it on a side stream behind an event and records the step's root when that stream's work completes
// (`row_tables_device.step_roots`). Four kernels: every segment digest, every row's leaf, every table's tree (one block each), and
// the step's tree (one block).
#include <cuda_runtime.h>

#include "row_table.cuh"

using namespace rowtab;

__global__ void k_midstates(const uint64_t* prefixes, uint64_t* ms) {
  if (blockIdx.x == 0 && threadIdx.x == 0) midstates(prefixes, ms);
}

__global__ void k_segments(const Tab* tabs, uint32_t T, const uint32_t* perm, uint64_t S, uint64_t* dig) {
  const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= S) return;
  uint64_t d[8];
  step_segment(tabs, T, perm, i, d);
#pragma unroll
  for (int k = 0; k < 8; ++k) dig[8 * i + k] = d[k];
}

__global__ void k_rows(const Tab* tabs, Args a, uint32_t N, const uint64_t* dig, const uint64_t* hm, const uint64_t* ms,
                       uint64_t* leaves) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= N) return;
  uint64_t m[16], o[8];
  for (int k = 0; k < 16; ++k) m[k] = ms[k];
  step_row(tabs, a, g, dig, hm, m, o);
#pragma unroll
  for (int k = 0; k < 8; ++k) leaves[8 * (size_t)g + k] = o[k];
}

// The frame tree over ``n`` leaves ``lv`` under ``did`` by one block, its levels in ``sc`` (``W - 1`` nodes): the root.
__device__ void block_tree(const uint64_t did[8], uint32_t n, const uint64_t* lv, uint64_t* sc, uint64_t* root) {
  const uint32_t W = width(n), d = depth(W);
  if (W == 1) {
    if (threadIdx.x < 8) root[threadIdx.x] = lv[threadIdx.x];
    return;
  }
  for (uint32_t j = threadIdx.x; j < W / 2; j += blockDim.x) {
    uint64_t c[2][8];
    for (int h = 0; h < 2; ++h) {
      const uint32_t r = 2 * j + h;
      if (r < n) {
        for (int k = 0; k < 8; ++k) c[h][k] = lv[8 * (size_t)r + k];
      } else {
        frame_pad(did, r, c[h]);
      }
    }
    frame_node(did, 0, j, c[0], c[1], sc + 8 * (size_t)(level_at(W, 1) + j));
  }
  __syncthreads();
  for (uint32_t l = 2; l <= d; ++l) {
    const uint64_t* below = sc + 8 * (size_t)level_at(W, l - 1);
    for (uint32_t j = threadIdx.x; j < (W >> l); j += blockDim.x)
      frame_node(did, l - 1, j, below + 16 * (size_t)j, below + 16 * (size_t)j + 8, sc + 8 * (size_t)(level_at(W, l) + j));
    __syncthreads();
  }
  if (threadIdx.x < 8) root[threadIdx.x] = sc[8 * (size_t)level_at(W, d) + threadIdx.x];
}

__global__ void k_trees(const Tab* tabs, const uint64_t* leaves, uint64_t* tree, uint64_t* out) {
  const Tab& t = tabs[blockIdx.x];
  block_tree(t.did, t.n, leaves + 8 * (size_t)t.row0, tree + 8 * (size_t)t.tree0, out + 8 * (1 + (size_t)blockIdx.x));
}

// ``sc``: the step's leaves (T) then its tree's levels (W - 1).
__global__ void k_step(Args a, uint64_t* sc, uint64_t* out) {
  for (uint32_t i = threadIdx.x; i < a.T; i += blockDim.x) frame_leaf(a.did, i, ROWTAB_ROOTS_SCHEMA, out + 8 * (1 + (size_t)i), sc + 8 * (size_t)i);
  __syncthreads();
  block_tree(a.did, a.T, sc, sc + 8 * (size_t)a.T, out);
}

static unsigned blocks(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// The midstates after the two prefix blocks (``prefixes``, 32 words) into ``ms`` (16 words), both device pointers.
extern "C" int rt_midstates(void* stream, const uint64_t* prefixes, uint64_t* ms) {
  k_midstates<<<1, 32, 0, (cudaStream_t)stream>>>(prefixes, ms);
  return (int)cudaGetLastError();
}

// The step's root then its tables' roots into ``out`` ((T + 1) x 8 words), enqueued on ``stream``. Device pointers: ``tabs``
// (T), ``perm`` (N), ``hm`` (192 x 256 x 8 words), ``ms`` (16), the scratch ``dig`` (S x 8), ``leaves`` (N x 8), ``tree``
// (the tables' widths summed, x 8) and ``step`` ((T + width(T)) x 8). ``a`` is host memory, read before this returns.
extern "C" int rt_step(void* stream, const Tab* tabs, const uint32_t* perm, uint64_t S, uint32_t N, const uint64_t* hm,
                       const uint64_t* ms, const Args* a, uint64_t* dig, uint64_t* leaves, uint64_t* tree, uint64_t* step,
                       uint64_t* out) {
  if (a->T == 0 || S == 0 || N == 0) return -1;
  cudaStream_t st = (cudaStream_t)stream;
  k_segments<<<blocks(S, 128), 128, 0, st>>>(tabs, a->T, perm, S, dig);
  k_rows<<<blocks(N, 64), 64, 0, st>>>(tabs, *a, N, dig, hm, ms, leaves);
  k_trees<<<a->T, 128, 0, st>>>(tabs, leaves, tree, out);
  k_step<<<1, 128, 0, st>>>(*a, step, out);
  return (int)cudaGetLastError();
}
