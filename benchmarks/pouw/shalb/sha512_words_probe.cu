// One item of the words committer a loop trip (sha512_words.cuh): a row/v2 chunk of WORDS_NB data blocks from its
// prefix's constant midstate and its padding block, unrolled; its frame-v3 leaf; and a level-0 node over that leaf and a
// loaded sibling. Count it with shalb_sass.py as NAME=SASS:N, N = WORDS_NB + 5 compressions a trip. The 16 KiB chunk
// (WORDS_NB 128) is the same chain at 64 more pairs; shalb_words_gpu.sh extrapolates in pairs from WORDS_NB 16 and 32. Built
// only to be disassembled; never launched.
#include <cstdint>

#include "sha512_words.cuh"

#ifndef WORDS_NB
#define WORDS_NB 8
#endif
#ifndef WORDS_PART  // 0: the item; 1: the chunk alone; 2: the chunk and its leaf
#define WORDS_PART 0
#endif

extern "C" __global__ void probe(const __grid_constant__ shaw::Pre p, const uint64_t* __restrict__ in,
                                 const uint64_t* __restrict__ sib, uint64_t* __restrict__ out, int n) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  constexpr uint64_t BITS = 1024ull * WORDS_NB;
#pragma unroll 1
  for (int b = 0; b < n; ++b) {
    const size_t l = t * (size_t)n + b;
    uint64_t d[8], lf[8], r[8], nd[8];
    shav2::chunk<WORDS_NB, BITS + 1024, shamin::RowV2Mid<BITS>>(in + l * 16 * WORDS_NB, d);
#if WORDS_PART == 1
    for (int i = 0; i < 8; ++i) nd[i] = d[i];
#else
    shaw::leaf(p.leaf, 65536 + l, d, lf);
#if WORDS_PART == 2
    for (int i = 0; i < 8; ++i) nd[i] = lf[i];
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = sib[l * 8 + i];
    shaw::node(p, 65536 + l, lf, r, nd);
#endif
#endif
#pragma unroll
    for (int i = 0; i < 8; ++i) out[l * 8 + i] = nd[i];
  }
}
