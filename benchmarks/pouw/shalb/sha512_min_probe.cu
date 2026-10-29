// One SHA-512 compression a loop trip through shamin::compress, in the harness of the committer's probe
// (sha512_probe.cu), for counting its SASS with shalb_sass.py. SHA_CVC=1 takes the chaining value as the IV, a
// compile-time constant, every trip, and stores each trip's digest. SHA_PAD=BITS compresses the padding block of a
// BITS-bit block-aligned message, which reads no block, from a chaining value loaded per thread (else the whole loop
// is warp-uniform and runs on the uniform datapath). Built only to be disassembled; never launched.
#include <cstdint>

#include "sha512_min.cuh"

#ifndef SHA_CVC
#define SHA_CVC 0
#endif
#ifndef SHA_PAD
#define SHA_PAD 0
#endif

extern "C" __global__ void probe(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nblocks) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t st[8];
  const uint64_t* p = in + t * 16 * (size_t)nblocks;
#pragma unroll
  for (int i = 0; i < 8; ++i) st[i] = SHA_PAD ? p[i] : shamin::IVt(i);
#pragma unroll 1
  for (int b = 0; b < nblocks; ++b) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = p[(size_t)b * 16 + i];
    shamin::compress<SHA_CVC, SHA_PAD>(st, w);
#if SHA_CVC
#pragma unroll
    for (int i = 0; i < 8; ++i) out[((size_t)b * gridDim.x * blockDim.x + t) * 8 + i] = st[i];
#endif
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[t * 8 + i] = st[i];
}
