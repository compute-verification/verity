// pd.cuh and the hash_h3.cuh / hash_fast.cuh kernels built for the host (host_test.py): the CUDA builtins they use, and
// `pd_run`, which runs a 1-D grid's CTAs in order, a CTA's threads as host threads under one barrier (__syncthreads) and a
// barrier per warp (__syncwarp).  Buffers are host memory: Args' pointers are host addresses.
//   g++ -O2 -std=c++20 -pthread -shared -fPIC -o pd_host.so pd_host.cpp
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using std::min;
#define PD_HOST
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __global__
#define __constant__ static const
#define __launch_bounds__(...)
#define __shared__ static
struct Dim3 { unsigned x = 0, y = 1, z = 1; };
static Dim3 blockIdx, blockDim;
static thread_local Dim3 threadIdx;
static std::barrier<>* g_barrier;
static std::vector<std::unique_ptr<std::barrier<>>> g_warps;
static void __syncthreads() { g_barrier->arrive_and_wait(); }
static void __syncwarp() { g_warps[threadIdx.x / 32]->arrive_and_wait(); }
static void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static void __trap() { std::fprintf(stderr, "__trap\n"); std::abort(); }
static long long clock64() { return 0; }
static unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v); }
static uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t s) {
  return (uint32_t)(((((uint64_t)hi << 32) | lo) << (s & 31)) >> 32);
}
static uint64_t g_lanes[1024];
static uint64_t __shfl_xor_sync(unsigned, uint64_t v, int o) {   // every lane of the warp calls it, as on the device
  g_lanes[threadIdx.x] = v;
  __syncwarp();
  const uint64_t x = g_lanes[threadIdx.x ^ o];
  __syncwarp();
  return x;
}
static int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
static int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
struct uint4 { uint32_t x, y, z, w; };
static uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return {x, y, z, w}; }

#include "pd.cuh"

namespace {
struct Kernel { const char* name; void (*fn)(const Args); };
#define PD_KERNELS(V) {"pd_commit_" #V, pd_commit_##V}, {"pd_tree_" #V, pd_tree_##V}, {"pd_tiles_" #V, pd_tiles_##V}, \
                      {"pd_probe_" #V, pd_probe_##V},
const Kernel KERNELS[] = {
    {"s512_heads", s512_heads},   {"pd_heads", pd_heads},     {"s512_row_segs", s512_row_segs},
    {"s512_folds", s512_folds},   {"hm_leaves", hm_leaves},   {"s512_frame_leaves", s512_frame_leaves},
    {"s512_pads", s512_pads},     {"s512_nodes", s512_nodes}, {"s512_tree_small", s512_tree_small},
    {"s512_seeds", s512_seeds},   {"s512_tile_segs", s512_tile_segs}, {"s512_tile_leaves", s512_tile_leaves},
    {"s512_rows", s512_rows},     PD_VARIANT_LIST(PD_KERNELS)};
}  // namespace

extern "C" int pd_run(const char* name, int grid, int block, const Args* a) {
  const Kernel* k = nullptr;
  for (const auto& x : KERNELS)
    if (!std::strcmp(x.name, name)) k = &x;
  if (!k || block % 32) return 1;
  blockDim.x = (unsigned)block;
  for (int b = 0; b < grid; b++) {
    blockIdx.x = (unsigned)b;
    std::barrier<> bar(block);
    g_barrier = &bar;
    g_warps.clear();
    for (int w = 0; w < block / 32; w++) g_warps.push_back(std::make_unique<std::barrier<>>(32));
    std::vector<std::thread> ts;
    for (int t = 0; t < block; t++)
      ts.emplace_back([k, a, t] {
        threadIdx.x = (unsigned)t;
        k->fn(*a);
      });
    for (auto& t : ts) t.join();
  }
  return 0;
}
