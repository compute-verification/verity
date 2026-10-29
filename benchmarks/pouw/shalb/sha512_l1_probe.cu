// One L1 leaf a loop trip through shal1::leaf (sha512_l1.cuh), for counting its SASS with shalb_sass.py (count it as
// NAME=SASS:10, ten compressions a trip). Each thread is one call: it loads the call's words once, before the loop, and
// each trip loads one leaf's rank and position words and reads its value. Built only to be disassembled; never
// launched.
#include <cstdint>

#include "sha512_l1.cuh"

extern "C" __global__ void probe(const uint64_t* __restrict__ call, const uint64_t* __restrict__ item,
                                 const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nleaves) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t cw[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) cw[i] = call[t * 9 + i];
#pragma unroll 1
  for (int b = 0; b < nleaves; ++b) {
    const size_t l = t * (size_t)nleaves + b;
    uint64_t st[8];
    shal1::leaf(cw, item[2 * l], item[2 * l + 1], in + l * shal1::VALUE_WORDS, st);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[l * 8 + i] = st[i];
  }
}
