// seeds.noise_block on the device: block t of seed u's stream, SHA-512(NOISE_TAG || seed_u || u64be t), one compression (96
// message bytes), one thread a block.  `receipts.Cubin` loads it; build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin -o noise.cubin noise.cu
//
//   h1_noise        n seeds' blocks 0 .. nb - 1 each: a unit's seed_j (h1tile.noise) or σ_R (h1tile.fold_keys)
//   h1_noise_calls  J10-R's call-scale streams (h1tile.j10r_noise): call c's seed_c, blocks 0 .. starts[c + 1] - starts[c] - 1,
//                   the calls' streams end to end, so a step's calls of one K take one launch
#include "../../../verity/ml/kernels/sha512.cuh"
typedef unsigned long long u64;
typedef unsigned int u32;

// block t of `seed`'s stream into out (16 big-endian halves), or with sink its halves' XOR into *out
__device__ __forceinline__ void noise_block(const u64* __restrict__ seed, u64 t, u64 tag0, u64 tag1, u64 tag2,
                                            u32* __restrict__ out, int sink) {
  u64 w[16];
  w[0] = tag0;
  w[1] = tag1;
  w[2] = tag2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = seed[j];
  w[11] = t;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
  u64 s[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = sx::IV_DEV[j];
  sx::compress<0, 2>(s, w);
  if (sink) {
    u32 x = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) x ^= (u32)(s[j] >> 32) ^ (u32)s[j];
    *out = x;
    return;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[2 * j] = (u32)(s[j] >> 32);
    out[2 * j + 1] = (u32)s[j];
  }
}

// seeds: n x 8 big-endian 64-bit words; tag0..2: NOISE_TAG's three words.  out: n x nb x 16 big-endian halves, or with sink
// n x nb words, each block's XOR of its halves.
extern "C" __global__ void h1_noise(const u64* __restrict__ seeds, u64 n, u32 nb, u64 tag0, u64 tag1, u64 tag2,
                                    u32* __restrict__ out, int sink) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nb) return;
  noise_block(seeds + 8 * (i / nb), i % nb, tag0, tag1, tag2, out + (sink ? i : 16 * i), sink);
}

// seeds: ncalls x 8 big-endian 64-bit words, call c's seed_c; starts: ncalls + 1 ascending, starts[0] = 0, call c's stream
// is blocks i in [starts[c], starts[c + 1]) of the launch, its block i - starts[c].  out: starts[ncalls] x 16 halves, or with
// sink starts[ncalls] words.
extern "C" __global__ void h1_noise_calls(const u64* __restrict__ seeds, const u64* __restrict__ starts, u32 ncalls, u64 tag0,
                                          u64 tag1, u64 tag2, u32* __restrict__ out, int sink) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= starts[ncalls]) return;
  u32 lo = 0, hi = ncalls;
  while (hi - lo > 1) {
    const u32 mid = (lo + hi) >> 1;
    if (starts[mid] <= i) lo = mid;
    else hi = mid;
  }
  noise_block(seeds + 8 * (u64)lo, i - starts[lo], tag0, tag1, tag2, out + (sink ? i : 16 * i), sink);
}
