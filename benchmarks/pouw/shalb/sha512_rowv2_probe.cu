// One 1 KiB row/v2 chunk a loop trip through shav2::chunk (sha512_rowv2.cuh), for counting its SASS with shalb_sass.py
// (count it as NAME=SASS:9, nine compressions a trip: eight data blocks and the padding block). Built only to be
// disassembled; never launched.
#include <cstdint>

#include "sha512_rowv2.cuh"

extern "C" __global__ void probe(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nchunks) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
  for (int b = 0; b < nchunks; ++b) {
    const size_t l = t * (size_t)nchunks + b;
    uint64_t st[8];
    shav2::chunk<8, 8192>(in + l * 128, st);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[l * 8 + i] = st[i];
  }
}
