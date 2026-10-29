// hash_decode.cuh and the hash_h3.cuh / hash_fast.cuh kernels built for the host (tests/test_pearl_c_hash_decode.py):
// the CUDA builtins they use, and
// `pd_run`, which runs a 1-D grid's CTAs in order, a CTA's threads as host threads under one barrier (__syncthreads), a
// barrier per warp (__syncwarp) and the named barriers (pd_bar_sync / pd_bar_arrive, each made at its first use with its
// thread count).  Buffers are host memory: Args' pointers are host addresses.
//   g++ -O2 -std=c++20 -pthread -shared -fPIC -o hash_decode_host.so hash_decode_host.cpp
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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
static std::mutex g_named_mu;
static std::unique_ptr<std::barrier<>> g_named[16];
static std::barrier<>& pd_named(int id, int n) {
  std::lock_guard<std::mutex> lock(g_named_mu);
  if (!g_named[id]) g_named[id] = std::make_unique<std::barrier<>>(n);
  return *g_named[id];
}
static void pd_bar_sync(int id, int n) { pd_named(id, n).arrive_and_wait(); }
static void pd_bar_arrive(int id, int n) { (void)pd_named(id, n).arrive(); }
static void __threadfence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static void __trap() { std::fprintf(stderr, "__trap\n"); std::abort(); }
static long long clock64() {   // monotonic, so the stamps' order is checked on the host
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
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
static float __uint_as_float(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static float __fmul_rn(float a, float b) { return a * b; }   // x86-64's SSE: IEEE single, nearest-even
static float __fdiv_rn(float a, float b) { return a / b; }
static uint16_t pd_f2bf(float f) {   // __float2bfloat16_rn: nearest-even, NaN quieted
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static uint8_t pd_e4m3(float f) {   // __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3): nearest-even, saturating to 448
  const uint8_t sign = std::signbit(f) ? 0x80 : 0;
  const double a = std::fabs((double)f);
  if (std::isnan(f)) return 0x7F;
  if (a >= 448.0) return sign | 0x7E;
  if (a == 0.0) return sign;
  int e;
  std::frexp(a, &e);
  const int E = e - 1;                                  // a = 1.m · 2^E
  if (E < -6) return sign | (uint8_t)std::nearbyint(std::ldexp(a, 9));   // subnormal: m · 2^-9 (8 is the least normal)
  double q = std::nearbyint(std::ldexp(a, 3 - E));     // in [8, 16]
  int ex = E;
  if (q == 16.0) { q = 8.0; ex++; }
  const int code = (ex + 7) << 3 | ((int)q - 8);
  return sign | (uint8_t)std::min(code, 0x7E);
}

#include "hash_decode.cuh"
// pd_commit_t's Z form, which pearl_c_sm90.cu's pd_commit_g takes: pd_commit_zero_V
#define PD_ZERO_INSTANCE(V) static void pd_commit_zero_##V(const Args a) { pd_commit_t<V, true>(a, blockIdx.x); }
PD_VARIANT_LIST(PD_ZERO_INSTANCE)
// another file's kernels beside these (pearl_c_sm90/hash_prefill_host.cpp): PD_HOST_EXTRA the file, PD_HOST_EXTRA_INSTANCES
// their definitions, PD_HOST_EXTRA_KERNELS their KERNELS entries
#ifdef PD_HOST_EXTRA
#include PD_HOST_EXTRA
PD_HOST_EXTRA_INSTANCES
#else
#define PD_HOST_EXTRA_KERNELS
#endif

namespace {
struct Kernel { const char* name; void (*fn)(const Args); };
#define PD_KERNELS(V) {"pd_commit_" #V, pd_commit_##V}, {"pd_tree_" #V, pd_tree_##V}, {"pd_tiles_" #V, pd_tiles_##V}, \
                      {"pd_probe_" #V, pd_probe_##V}, {"pd_commit_zero_" #V, pd_commit_zero_##V},
const Kernel KERNELS[] = {
    {"s512_heads", s512_heads},   {"pd_heads", pd_heads},     {"s512_row_segs", s512_row_segs},
    {"s512_folds", s512_folds},   {"hm_leaves", hm_leaves},   {"s512_frame_leaves", s512_frame_leaves},
    {"s512_pads", s512_pads},     {"s512_nodes", s512_nodes}, {"s512_tree_small", s512_tree_small},
    {"s512_seeds", s512_seeds},   {"s512_tile_segs", s512_tile_segs}, {"s512_tile_leaves", s512_tile_leaves},
    {"s512_rows", s512_rows},     PD_VARIANT_LIST(PD_KERNELS) PD_HOST_EXTRA_KERNELS};
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
    for (auto& n : g_named) n.reset();
  }
  return 0;
}
