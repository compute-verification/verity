// One call's PRE words a loop trip through shal1::call_pre (sha512_l1.cuh), for counting its SASS with shalb_sass.py
// (NAME=SASS:1): what an L1 call computes once for all its leaves. Each trip loads one call's words 6–14 and stores
// its PRE words. Built only to be disassembled; never launched.
#include <cstdint>

#include "sha512_l1.cuh"

extern "C" __global__ void probe(const uint64_t* __restrict__ call, uint64_t* __restrict__ pre, int ncalls) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
  for (int c = 0; c < ncalls; ++c) {
    const size_t k = t * (size_t)ncalls + c;
    uint64_t cw[9], pw[shal1::PRE];
#pragma unroll
    for (int i = 0; i < 9; ++i) cw[i] = call[k * 9 + i];
    shal1::call_pre(cw, pw);
#pragma unroll
    for (int i = 0; i < shal1::PRE; ++i) pre[k * shal1::PRE + i] = pw[i];
  }
}
