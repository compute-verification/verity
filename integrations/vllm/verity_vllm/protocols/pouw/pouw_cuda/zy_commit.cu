// `zy_commit.cuh` on the device: a call's z/y records committed for its z table and its registered value from one read of
// each record, enqueued on the caller's stream from device pointers (`pouw_h1_registered_device.device_hash`'s ``also``).
// `zy_records` writes the call table's leaves, the value's b || c strings and its leaves; `zy_call_tree` the call table's pads
// and levels (one launch a wide level, then one block for the levels of at most `TOP` nodes); the value's pads and levels are
// `registered_rows.cu`'s `rr_tree`. The plan and the nibble table are built on the host (`zy_plan`, `zy_nibbles`).
#include <cuda_runtime.h>

#include "zy_commit.cuh"

using namespace zyc;

constexpr uint32_t TOP = 1024;

__global__ void __launch_bounds__(128) k_records(const Plan* __restrict__ p, const uint8_t* __restrict__ tab,
                                                 const uint8_t* __restrict__ rows, uint32_t n, u64* call, u64* bcs,
                                                 u64* levels) {
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;
  const rc::NibTab<true> T{tab};
  u64 cl[8], bc[16], rl[8];
  record(*p, T, u, rows + (size_t)p->R * u, cl, bc, rl);
#pragma unroll
  for (int k = 0; k < 8; ++k) call[8 * (size_t)u + k] = cl[k];
#pragma unroll
  for (int k = 0; k < 16; ++k) bcs[16 * (size_t)u + k] = bc[k];
#pragma unroll
  for (int k = 0; k < 8; ++k) levels[8 * (size_t)u + k] = rl[k];
}

__global__ void k_pads(const Plan* __restrict__ p, uint32_t W, u64* levels) {
  const uint32_t r = p->n + blockIdx.x * blockDim.x + threadIdx.x;
  if (r < W) call_pad(*p, r, levels + 8 * (size_t)r);
}

__global__ void k_level(const Plan* __restrict__ p, uint32_t W, uint32_t l, u64* levels) {
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < (W >> l)) call_node(*p, W, l, j, levels);
}

__global__ void k_top(const Plan* __restrict__ p, uint32_t W, uint32_t depth, uint32_t l0, u64* levels) {
  for (uint32_t l = l0; l <= depth; ++l) {
    for (uint32_t j = threadIdx.x; j < (W >> l); j += blockDim.x) call_node(*p, W, l, j, levels);
    __syncthreads();
  }
}

static unsigned blocks(uint64_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// The plan of ``a`` into ``p`` (host memory, sizeof(Plan) bytes, `zy_plan_bytes`): 0, or -1 for what `plan` refuses.
extern "C" int zy_plan(const Args* a, Plan* p) { return plan(*a, *p); }

extern "C" uint64_t zy_plan_bytes() { return sizeof(Plan); }
extern "C" uint64_t zy_args_bytes() { return sizeof(Args); }

// The hm96 key's (256 bytes) nibble table into ``tab`` (host memory, `rc::TN_BYTES`).
extern "C" void zy_nibbles(const uint8_t* key, uint8_t* tab) {
  for (uint32_t t = 0; t < 512; ++t) rc::nib_entry(key, t, tab);
}

// Records 0 .. n - 1 at ``rows`` (n x R bytes, 16-byte aligned): the call table's leaves into level 0 of ``call`` ((2 W - 1)
// x 8 words), the value's b || c into ``bcs`` (n x 16 words) and its leaves into level 0 of ``levels`` ((2 W - 1) x 8). ``p``
// (the plan, `zy_plan`'s, copied there) and ``tab`` (the nibble table) are device pointers; ``n`` the plan's.
extern "C" int zy_records(void* stream, const Plan* p, uint32_t n, const uint8_t* tab, const uint8_t* rows, u64* call,
                          u64* bcs, u64* levels) {
  if (n == 0 || (uintptr_t)rows % 16 || (uintptr_t)tab % 16) return -1;
  k_records<<<blocks(n, 128), 128, 0, (cudaStream_t)stream>>>(p, tab, rows, n, call, bcs, levels);
  return (int)cudaGetLastError();
}

// The call table's pads past its n leaves and every level above, in ``call``, once `zy_records` is enqueued: its root at node
// 2 W - 2.
extern "C" int zy_call_tree(void* stream, const Plan* p, uint32_t n, uint32_t W, uint32_t depth, u64* call) {
  cudaStream_t st = (cudaStream_t)stream;
  if (n == 0 || W < n || (W & (W - 1)) || (1ull << depth) != W) return -1;
  if (W > n) k_pads<<<blocks(W - n, 128), 128, 0, st>>>(p, W, call);
  uint32_t l = 1;
  for (; l <= depth && (W >> l) > TOP; ++l) k_level<<<blocks(W >> l, 128), 128, 0, st>>>(p, W, l, call);
  if (l <= depth) k_top<<<1, 256, 0, st>>>(p, W, depth, l, call);
  return (int)cudaGetLastError();
}
