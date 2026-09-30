"""A host build of `hash.cuh`, `hash_sm120.cuh` and `h1_standin.cu` that runs its kernels thread for thread, for byte tests of the -h1 kernels without a GPU.

Every CUDA thread of a block is a host thread; `__syncthreads` and `__syncwarp` are one barrier over the block (stronger than
a warp's, which the kernels here allow: every thread of a block reaches the same syncs), which a thread leaves when it
returns. Blocks run one after another. `build()` compiles it with g++ (C++20) and returns the library; `launch()` runs a
kernel over a grid with `Args` built from host buffers.
"""
from __future__ import annotations

import ctypes as C
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
KERNELS = ("hash_msg", "hash_msg_v", "hash_leaf", "hash_leaf_p", "hash_rows", "hash_rows_w", "hash_pad", "hash_node_keys", "hash_nodes",
           "hash_tree_leaves", "hash_tree_small", "hash_rows_wt", "lat_chain", "derive_rows_hash", "derive_rows_wide",
           "hash_rows_blk", "hash_rows_seg", "derive_rows", "standin_gemm",
           "b3s_level_keys", "hash_rows_b3s", "hash_leaves_b3s", "b3_seeds")

PRELUDE = r"""
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#define __device__
#define __host__
#define __constant__ static
#define __shared__ static
#define __global__
#define __launch_bounds__(...)
#define __forceinline__ inline
struct uint4 { uint32_t x, y, z, w; };
static inline uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
static inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, int n) { n &= 31; return n ? (lo >> n) | (hi << (32 - n)) : lo; }
static inline uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, int n) { n &= 31; return n ? (hi << n) | (lo >> (32 - n)) : hi; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
struct Dim { unsigned x = 0, y = 0, z = 0; };
static thread_local Dim threadIdx;
static Dim blockIdx, blockDim, gridDim;
// a block's barrier that a returning thread leaves (drop), releasing the others if they were the rest of the phase
struct TwinBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int expected, arrived = 0;
  unsigned gen = 0;
  explicit TwinBarrier(int n) : expected(n) {}
  void wait() {
    std::unique_lock<std::mutex> l(mu);
    const unsigned g = gen;
    if (++arrived == expected) { arrived = 0; gen++; cv.notify_all(); }
    else cv.wait(l, [&] { return gen != g; });
  }
  void drop() {
    std::unique_lock<std::mutex> l(mu);
    if (--expected > 0 && arrived == expected) { arrived = 0; gen++; cv.notify_all(); }
  }
};
static TwinBarrier* twin_bar;
static inline void __syncthreads() { twin_bar->wait(); }
static inline void __syncwarp(unsigned = 0xffffffffu) { twin_bar->wait(); }
static uint32_t twin_x[1024];
// a warp shuffle: every thread of the block posts its value, then reads its source lane's within its warp
static inline uint32_t __shfl_sync(unsigned, uint32_t v, int src) {
  twin_x[threadIdx.x] = v;
  twin_bar->wait();
  const uint32_t r = twin_x[(threadIdx.x & ~31u) + (unsigned)(src & 31)];
  twin_bar->wait();
  return r;
}
static inline uint32_t __shfl_xor_sync(unsigned m, uint32_t v, int o) { return __shfl_sync(m, v, (int)(threadIdx.x & 31) ^ o); }
static inline long long clock64() { return 0; }
static inline void __threadfence() {}
static inline void __threadfence_block() {}
static inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
static std::vector<uint32_t> twin_dyn;
"""


def _source() -> str:
    body = ((HERE / "hash.cuh").read_text() + (HERE / "hash_sm120.cuh").read_text() + (HERE / "hash_h2.cuh").read_text()
            + (HERE / "h1_standin.cu").read_text()).replace(
        "extern __shared__ uint32_t sh[];", "uint32_t* sh = twin_dyn.data();").replace("#pragma once", "")
    runs = "".join(
        f'extern "C" void run_{k}(const Args* a, int grid, int block, int smem) {{\n'
        f"  twin_dyn.assign(smem / 4 + 1, 0u); blockDim.x = block; gridDim.x = grid;\n"
        f"  for (int b = 0; b < grid; b++) {{\n"
        f"    blockIdx.x = b; TwinBarrier bar(block); twin_bar = &bar; std::vector<std::thread> ts;\n"
        f"    for (int t = 0; t < block; t++) ts.emplace_back([&, t] {{ threadIdx.x = t; {k}(*a); bar.drop(); }});\n"
        f"    for (auto& th : ts) th.join();\n  }}\n}}\n" for k in KERNELS)
    return PRELUDE + re.sub(r'#include "hash(_sm120|_h2)?.cuh"', "", body) + runs


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


@lru_cache(maxsize=1)
def build() -> C.CDLL:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise RuntimeError("no C++ compiler")
    out = Path(tempfile.mkdtemp(prefix="h1-simt-"))
    (out / "twin.cpp").write_text(_source())
    subprocess.run([cxx, "-O1", "-std=c++20", "-pthread", "-shared", "-fPIC", "-o", str(out / "twin.so"), str(out / "twin.cpp")],
                   check=True, capture_output=True, text=True)
    return C.CDLL(str(out / "twin.so"))


def buffer(data: bytes | int):
    return C.create_string_buffer(data, len(data)) if isinstance(data, bytes) else C.create_string_buffer(max(data, 16))


def launch(kernel: str, grid: int, block: int, ptrs, ints, smem: int = 0) -> None:
    a = Args()
    for j, p in enumerate(ptrs):
        a.p[j] = C.addressof(p)
    for j, v in enumerate(ints):
        a.i[j] = v
    getattr(build(), f"run_{kernel}")(C.byref(a), grid, block, smem)


def rows_w_words(rb: int) -> int:
    """`hash_sm120.cuh`'s rows_w_words: shared words per warp of hash_rows_w (the leaf header is one 64-byte block)."""
    return 8 * ((64 + rb + 1023) // 1024)
