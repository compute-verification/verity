// seeds.noise_block on the device: block t of seed u's stream, SHA-512(NOISE_TAG || seed_u || u64be t), one compression (96
// message bytes), one thread a block.  `receipts.Cubin` loads it; build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I benchmarks/codec_tensor -cubin -o noise.cubin noise.cu
namespace sh {
typedef unsigned long long u64;
#include "sha512.cuh"
}  // namespace sh
typedef unsigned long long u64;
typedef unsigned int u32;

// seeds: n x 8 big-endian 64-bit words; tag0..2: NOISE_TAG's three words.  out: n x nb x 16 big-endian halves, or with sink
// n x nb words, each block's XOR of its halves.
extern "C" __global__ void h1_noise(const u64* __restrict__ seeds, u64 n, u32 nb, u64 tag0, u64 tag1, u64 tag2,
                                    u32* __restrict__ out, int sink) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * nb) return;
  const u64 u = i / nb;
  const u64 t = i % nb;
  u64 w[16];
  w[0] = tag0;
  w[1] = tag1;
  w[2] = tag2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = seeds[8 * u + j];
  w[11] = t;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
  u64 s[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = sh::kSha512Iv[j];
  sh::sha512_compress(s, w);
  if (sink) {
    u32 x = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) x ^= (u32)(s[j] >> 32) ^ (u32)s[j];
    out[i] = x;
    return;
  }
  u32* o = out + 16 * i;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[2 * j] = (u32)(s[j] >> 32);
    o[2 * j + 1] = (u32)s[j];
  }
}
