// `registered_rows.cuh` on the device: a registered value's ``b || c`` strings and every level of its tree, enqueued on the
// caller's stream from device pointers (`pouw_h1_registered_device.device_commit`). Rows go in chunks (`rr_leaves`, each
// launch the rows its buffer holds), then the pads and the levels (`rr_tree`): one launch a wide level, then one block for
// the levels of at most `TOP` nodes.
#include <cuda_runtime.h>

#include "registered_rows.cuh"

using namespace regrows;

constexpr uint32_t TOP = 1024;

__global__ void k_midstates(const uint64_t* prefixes, uint64_t* ms) {
  if (blockIdx.x == 0 && threadIdx.x == 0) midstates(prefixes, ms);
}

__global__ void k_leaves(Table t, uint32_t j0, uint32_t count, const uint64_t* hm, const uint64_t* ms, uint64_t* bcs,
                         uint64_t* levels) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t j = j0 + i;
  uint64_t m[16], bc[16], lf[8];
  for (int k = 0; k < 16; ++k) m[k] = ms[k];
  row_leaf(t, j, (const uint8_t*)t.rows + (size_t)t.row_bytes * i, hm, m, bc, lf);
#pragma unroll
  for (int k = 0; k < 16; ++k) bcs[16 * (size_t)j + k] = bc[k];
#pragma unroll
  for (int k = 0; k < 8; ++k) levels[8 * (size_t)j + k] = lf[k];
}

__global__ void k_pads(Table t, uint64_t* levels) {
  const uint32_t r = t.n + blockIdx.x * blockDim.x + threadIdx.x;
  if (r < t.W) pad(t, r, levels + 8 * (size_t)r);
}

__global__ void k_level(Table t, uint32_t l, uint64_t* levels) {
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < (t.W >> l)) node(t, l, j, levels);
}

__global__ void k_top(Table t, uint32_t l0, uint64_t* levels) {
  for (uint32_t l = l0; l <= t.depth; ++l) {
    for (uint32_t j = threadIdx.x; j < (t.W >> l); j += blockDim.x) node(t, l, j, levels);
    __syncthreads();
  }
}

__global__ void k_public_salts(Pub p, uint64_t* out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) public_salt(p, p.first + i, out + 24 * (size_t)i);
}

static unsigned blocks(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// The ``p->n`` public salts from position ``p->first`` (``p`` host memory, read before this returns) into ``out`` (n x 24
// words, a device pointer), each its 192 bytes.
extern "C" int rr_public_salts(void* stream, const Pub* p, uint64_t* out) {
  if (p->n == 0 || p->len > 8 * PUB_WORDS) return -1;
  k_public_salts<<<blocks(p->n, 128), 128, 0, (cudaStream_t)stream>>>(*p, out);
  return (int)cudaGetLastError();
}

// The midstates after the two prefix blocks (``prefixes``, 32 words) into ``ms`` (16 words), both device pointers.
extern "C" int rr_midstates(void* stream, const uint64_t* prefixes, uint64_t* ms) {
  k_midstates<<<1, 32, 0, (cudaStream_t)stream>>>(prefixes, ms);
  return (int)cudaGetLastError();
}

// Rows j0 .. j0 + count - 1 of the value ``t`` (host memory, read before this returns; its ``rows`` the device bytes of
// those rows, its ``salts`` the device salts or 0): their b || c into ``bcs`` (n x 16 words) and their leaves into level 0
// of ``levels`` ((2 W - 1) x 8 words). ``hm`` (192 x 256 x 8 words) and ``ms`` (16) are device pointers.
extern "C" int rr_leaves(void* stream, const Table* t, uint32_t j0, uint32_t count, const uint64_t* hm, const uint64_t* ms,
                         uint64_t* bcs, uint64_t* levels) {
  if (count == 0 || (uint64_t)j0 + count > t->n) return -1;
  k_leaves<<<blocks(count, 64), 64, 0, (cudaStream_t)stream>>>(*t, j0, count, hm, ms, bcs, levels);
  return (int)cudaGetLastError();
}

// The pads past the n leaves and every level above, in ``levels``, once every row's leaf is enqueued.
extern "C" int rr_tree(void* stream, const Table* t, uint64_t* levels) {
  cudaStream_t st = (cudaStream_t)stream;
  if (t->W > t->n) k_pads<<<blocks(t->W - t->n, 128), 128, 0, st>>>(*t, levels);
  uint32_t l = 1;
  for (; l <= t->depth && (t->W >> l) > TOP; ++l) k_level<<<blocks(t->W >> l, 128), 128, 0, st>>>(*t, l, levels);
  if (l <= t->depth) k_top<<<1, 256, 0, st>>>(*t, l, levels);
  return (int)cudaGetLastError();
}
