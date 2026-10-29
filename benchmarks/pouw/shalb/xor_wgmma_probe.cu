// Honest's xor-m epilogue on the served GEMM's form, sm_90a: a WGMMA m64n8k32 s8 a step accumulating in place (D is
// C, as np_kernel's ssn.cu issues it), waited each step, and over a block of M steps nothing (base) or the XOR of the
// block's words (xor), the block's four XORs a thread stored. Built to SASS only and counted by xor_epilogue_sass.py;
// never launched.
#include <cstdint>

enum WMode { WBASE, WXOR };

__device__ __forceinline__ void wgmma_n8(uint32_t (&d)[4], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0, %1, %2, %3}, %4, %5, p;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "l"(da), "l"(db), "r"(1));
}

template <int M, WMode MODE>
__global__ void __launch_bounds__(128) wprobe(const uint64_t* __restrict__ desc, uint32_t* __restrict__ out,
                                              int blocks) {
  __shared__ uint32_t sm[4 * 128];
  uint32_t d[4] = {0, 0, 0, 0};
  const int tid = threadIdx.x;
  for (int blk = 0; blk < blocks; ++blk) {
    uint32_t x[4] = {0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < M; ++t) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      wgmma_n8(d, desc[2 * (blk * M + t)], desc[2 * (blk * M + t) + 1]);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      if constexpr (MODE == WXOR) {
#pragma unroll
        for (int k = 0; k < 4; ++k) x[k] = t == 0 ? d[k] : x[k] ^ d[k];
      }
    }
    if constexpr (MODE == WXOR) {
#pragma unroll
      for (int k = 0; k < 4; ++k) sm[k * 128 + tid] = x[k];
    }
    __syncthreads();
  }
  out[tid] = d[0] ^ d[1] ^ d[2] ^ d[3] ^ sm[tid];
}

#define INST(M, MODE) template __global__ void wprobe<M, MODE>(const uint64_t*, uint32_t*, int);
INST(16, WBASE)
INST(16, WXOR)
INST(11, WBASE)
INST(11, WXOR)
