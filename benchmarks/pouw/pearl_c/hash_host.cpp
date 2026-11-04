// hash_h3.cuh and pearl_c_sm120's hash_fast.cuh built for the host (tests/test_pearl_c_hash_host.py): the CUDA builtins they
// use, and `launch`, which runs a 1-D grid's CTAs in order; a CTA's threads run in order too, or as host threads under one
// barrier for a kernel that calls __syncthreads (hash_rows, s512_rows, s512_tile_leaves, s512_tree_small).  Buffers are host
// memory: Args' pointers are host addresses.
//   g++ -O2 -std=c++20 -pthread -shared -fPIC -I benchmarks/pouw/pearl_c -o hash_host.so hash_host.cpp
#include <algorithm>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using std::min;
#define __device__
#define __forceinline__ inline
#define __global__
#define __constant__ static const
#define __launch_bounds__(...)
#define __shared__ static
struct Dim3 { unsigned x = 0, y = 1, z = 1; };
static Dim3 blockIdx, blockDim;
static thread_local Dim3 threadIdx;
static std::barrier<>* g_barrier;
static void __syncthreads() { g_barrier->arrive_and_wait(); }
static uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t s) {
  return (uint32_t)(((((uint64_t)hi << 32) | lo) << (s & 31)) >> 32);
}
struct uint4 { uint32_t x, y, z, w; };
static uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return {x, y, z, w}; }

#include "hash_h3.cuh"
#include "../pearl_c_sm120/hash_fast.cuh"

namespace {
struct Kernel { const char* name; void (*fn)(const Args); bool syncs; };
const Kernel KERNELS[] = {{"s512_rows", s512_rows, true},          {"s512_seeds", s512_seeds, false},
                          {"s512_line_raws", s512_line_raws, false}, {"s512_tile_leaves", s512_tile_leaves, true},
                          {"s512_pads", s512_pads, false},         {"s512_nodes", s512_nodes, false},
                          {"s512_tree_small", s512_tree_small, true}, {"hm_leaves", hm_leaves, false},
                          {"s512_tickets", s512_tickets, false},     {"s512_frame_leaves", s512_frame_leaves, false},
                          {"s512_heads", s512_heads, false},         {"s512_row_segs", s512_row_segs, false},
                          {"s512_tile_segs", s512_tile_segs, false}, {"s512_folds", s512_folds, false}};
constexpr int COUNT = sizeof KERNELS / sizeof *KERNELS;
}  // namespace

extern "C" const char* kernel_name(int i) { return i >= 0 && i < COUNT ? KERNELS[i].name : nullptr; }

extern "C" int launch(const char* name, unsigned grid, unsigned block, const Args* a) {
  const Kernel* k = nullptr;
  for (const Kernel& e : KERNELS)
    if (!strcmp(e.name, name)) k = &e;
  if (!k || !grid || !block || block > 1024) return 1;
  blockDim.x = block;
  for (unsigned b = 0; b < grid; b++) {
    blockIdx.x = b;
    if (!k->syncs) {
      for (unsigned t = 0; t < block; t++) {
        threadIdx.x = t;
        k->fn(*a);
      }
      continue;
    }
    auto bar = std::make_unique<std::barrier<>>(block);
    g_barrier = bar.get();
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < block; t++) threads.emplace_back([k, a, t] { threadIdx.x = t; k->fn(*a); });
    for (std::thread& th : threads) th.join();
    g_barrier = nullptr;
  }
  return 0;
}
