// One SHA-512 compression a loop trip, through the committer's own `sha::compress` (sha512.cuh), for counting its SASS:
// the loop body is a 128-byte block's 16 loads plus one compression, so its instructions less the loads are what the
// committer issues a compression. Built only to be disassembled (shalb_sass.sh); never launched.
#include <cstdint>

#include "sha512.cuh"

extern "C" __global__ void probe(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nblocks) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t st[8];
  sha::iv(st);
  const uint64_t* p = in + t * 16 * (size_t)nblocks;
#pragma unroll 1
  for (int b = 0; b < nblocks; ++b) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = p[(size_t)b * 16 + i];
    sha::compress(st, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[t * 8 + i] = st[i];
}
