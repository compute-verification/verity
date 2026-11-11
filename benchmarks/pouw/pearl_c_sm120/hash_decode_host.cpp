// hash_decode.cuh and the hash_h3.cuh / hash_fast.cuh kernels built for the host (tests/test_pearl_c_hash_decode.py):
// the CUDA builtins they use, and
// `pd_run`, which runs a 1-D grid's CTAs in order, a CTA's threads under one barrier (__syncthreads), a barrier per warp
// (__syncwarp) and the named barriers (pd_bar_sync / pd_bar_arrive, each made at its first use with its thread count).
// A CTA's threads are contexts on the caller's thread (`Fiber`), each run in thread order until it waits at a barrier or
// returns, so a barrier costs a context switch whatever else holds the machine; a round of the CTA in which no thread
// arrives at a barrier or returns is a deadlock, which pd_run reports.  Buffers are host memory: Args' pointers are host
// addresses.
//   g++ -O2 -std=c++20 -pthread -shared -fPIC -o hash_decode_host.so hash_decode_host.cpp
#include <sys/mman.h>
#include <ucontext.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
static Dim3 blockIdx, blockDim, threadIdx;
static void pd_fiber();
#if defined(__x86_64__)
// a switch between pd_run and a CTA's thread without swapcontext's signal-mask system call: the callee-saved registers,
// MXCSR and the x87 control word pushed on the stack being left, whose pointer is saved in *from
asm(".text\n.p2align 4\n.hidden pd_switch\n.globl pd_switch\n.type pd_switch,@function\npd_switch:\n"
    "pushq %rbp\npushq %rbx\npushq %r12\npushq %r13\npushq %r14\npushq %r15\nsubq $16, %rsp\nstmxcsr 8(%rsp)\nfnstcw (%rsp)\n"
    "movq %rsp, (%rdi)\nmovq %rsi, %rsp\n"
    "fldcw (%rsp)\nldmxcsr 8(%rsp)\naddq $16, %rsp\npopq %r15\npopq %r14\npopq %r13\npopq %r12\npopq %rbx\npopq %rbp\nret\n"
    ".size pd_switch, .-pd_switch\n");
extern "C" void pd_switch(void** from, void* to);
struct Fiber {
  void* sp;
  bool done = false;
};
static void* g_sched;
static Fiber* g_self;
static void pd_yield() { pd_switch(&g_self->sp, g_sched); }
static void pd_resume(Fiber& f) { pd_switch(&g_sched, f.sp); }
static void pd_start(Fiber& f, char* stack, size_t size) {   // a frame pd_switch's return enters pd_fiber from
  auto* sp = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(stack + size) & ~uintptr_t{15});
  *--sp = 0;
  *--sp = reinterpret_cast<uint64_t>(&pd_fiber);
  for (int r = 0; r < 6; r++) *--sp = 0;
  sp -= 2;
  sp[0] = 0x037F;   // the x87 control word and MXCSR a thread starts with: their defaults
  sp[1] = 0x1F80;
  f.sp = sp;
}
static void pd_exit() { pd_yield(); }
#else
struct Fiber {
  ucontext_t ctx;
  bool done = false;
};
static ucontext_t g_sched;
static Fiber* g_self;
static void pd_yield() { swapcontext(&g_self->ctx, &g_sched); }
static void pd_resume(Fiber& f) { swapcontext(&g_sched, &f.ctx); }
static void pd_start(Fiber& f, char* stack, size_t size) {
  getcontext(&f.ctx);
  f.ctx.uc_stack.ss_sp = stack;
  f.ctx.uc_stack.ss_size = size;
  f.ctx.uc_link = &g_sched;
  makecontext(&f.ctx, pd_fiber, 0);
}
static void pd_exit() {}
#endif
static unsigned long g_progress;   // arrivals and returns
struct Barrier {
  int n, arrived = 0;
  unsigned gen = 0;
  explicit Barrier(int n) : n(n) {}
  void arrive() {
    g_progress++;
    if (++arrived == n) {
      arrived = 0;
      gen++;
    }
  }
  void arrive_and_wait() {
    const unsigned g = gen;
    arrive();
    while (gen == g) pd_yield();
  }
};
static Barrier* g_barrier;
static std::vector<Barrier> g_warps;
static std::unique_ptr<Barrier> g_named[16];
static Barrier& pd_named(int id, int n) {
  if (!g_named[id]) g_named[id] = std::make_unique<Barrier>(n);
  return *g_named[id];
}
static void __syncthreads() { g_barrier->arrive_and_wait(); }
static void __syncwarp() { g_warps[threadIdx.x / 32].arrive_and_wait(); }
static void pd_bar_sync(int id, int n) { pd_named(id, n).arrive_and_wait(); }
static void pd_bar_arrive(int id, int n) { pd_named(id, n).arrive(); }
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
const Kernel* g_kernel;
const Args* g_args;
}  // namespace

static void pd_fiber() {
  g_kernel->fn(*g_args);
  g_self->done = true;
  g_progress++;
  pd_exit();
}

extern "C" int pd_run(const char* name, int grid, int block, const Args* a) {
  const Kernel* k = nullptr;
  for (const auto& x : KERNELS)
    if (!std::strcmp(x.name, name)) k = &x;
  if (!k || block % 32 || block > 1024) return 1;
  // each thread's stack, above a guard page
  constexpr size_t STACK = 1 << 20, PAGE = 4096;
  const size_t span = (STACK + PAGE) * (size_t)block;
  void* mem = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (mem == MAP_FAILED) return 3;
  char* stacks = static_cast<char*>(mem);
  for (int t = 0; t < block; t++) mprotect(stacks + (STACK + PAGE) * t, PAGE, PROT_NONE);
  g_kernel = k;
  g_args = a;
  blockDim.x = (unsigned)block;
  std::vector<Fiber> fs(block);
  int rc = 0;
  for (int b = 0; b < grid && !rc; b++) {
    blockIdx.x = (unsigned)b;
    Barrier bar(block);
    g_barrier = &bar;
    g_warps.assign(block / 32, Barrier(32));
    for (int t = 0; t < block; t++) {
      fs[t].done = false;
      pd_start(fs[t], stacks + (STACK + PAGE) * t + PAGE, STACK);
    }
    for (int left = block; left && !rc;) {
      const unsigned long before = g_progress;
      for (int t = 0; t < block; t++) {
        if (fs[t].done) continue;
        g_self = &fs[t];
        threadIdx.x = (unsigned)t;
        pd_resume(fs[t]);
        left -= fs[t].done;
      }
      if (left && g_progress == before) {
        std::fprintf(stderr, "pd_run: %s's CTA %d deadlocked: %d of its %d threads wait at a barrier no thread reaches\n",
                     name, b, left, block);
        rc = 2;
      }
    }
    for (auto& n : g_named) n.reset();
  }
  munmap(mem, span);
  return rc;
}
