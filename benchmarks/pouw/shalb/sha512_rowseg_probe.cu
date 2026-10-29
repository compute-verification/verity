// One row-seg segment a loop trip through shaseg::segment_from (sha512_rowseg.cuh), for counting its SASS with
// shalb_sass.py (count it as NAME=SASS:10, ten compressions a trip). Each thread is one call whose PRE words are
// computed once for all its rows; each trip loads them again (volatile, so they stay loads rather than registers held
// across trips), the row's words 13 and 14 of block 1 and the segment's words. Built only to be disassembled; never
// launched.
#include <cstdint>

#include "sha512_rowseg.cuh"

extern "C" __global__ void probe(const uint64_t* __restrict__ pre, const uint64_t* __restrict__ item,
                                 const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nsegments) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const volatile uint64_t* p = pre + t * shaseg::PRE;
#pragma unroll 1
  for (int b = 0; b < nsegments; ++b) {
    const size_t l = t * (size_t)nsegments + b;
    uint64_t st[8];
    shaseg::segment_from(p, item[2 * l], item[2 * l + 1], in + l * shaseg::SEGMENT_WORDS, st);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[l * 8 + i] = st[i];
  }
}
