// The SHA-512 noise as separate passes (sha_noise.cuh's generator), shared by sgemm.cu and pvgemm.cu; included after
// mgemm.cu. sigma is fresh for every instance, from the seed.
#pragma once

struct Sig { uint64_t w[4]; };
static uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static Sig sigma(uint32_t seed, uint32_t inst) {
  uint64_t x = (uint64_t)seed << 32 | inst;
  Sig s;
  for (auto& w : s.w) w = splitmix64(x);
  return s;
}

// int8 in [lo, lo + span), or E4M3 in [-8, 8]
template <int FMT> __global__ void __launch_bounds__(256) sfill(uint8_t* X, size_t count, uint32_t seed, int lo = -119, int span = 239) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t h = mix32(seed ^ mix32((uint32_t)i) ^ mix32((uint32_t)(i >> 32) + 0x51u));
    X[i] = (uint8_t)enc<FMT>(FMT == I8 ? (int)(h % (uint32_t)span) + lo : (int)(h % 17) - 8);
  }
}

// the separate pass over both operands in one grid: thread p < pa is A's patch p, the rest B's; each chunk of the patch
// loaded, its noise added, stored to the edited copy
template <int NS, bool F8>
__global__ void __launch_bounds__(256) sha_edit(const uint8_t* __restrict__ A, uint8_t* __restrict__ XA, const uint8_t* __restrict__ B,
                                                uint8_t* __restrict__ XB, uint32_t pa, uint32_t pb, uint32_t KB,
                                                const __grid_constant__ Sig sg) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pa + pb) return;
  const bool isA = i < pa;
  const uint32_t p = isA ? i : i - pa;
  const uint8_t* X = isA ? A : B;
  uint8_t* Xp = isA ? XA : XB;
  const size_t k = (size_t)KB * 128, base = (size_t)(p / KB) * 4 * k + (size_t)(p % KB) * 128;
  sha_patch<NS>(sg.w, 2 * (NS - 1) + (isA ? 0 : 1), p, [&](int c, const uint32_t(&n)[4]) {
    const size_t off = base + (size_t)(c / 8) * k + (c % 8) * 16;
    const uint4 x = *reinterpret_cast<const uint4*>(X + off);
    *reinterpret_cast<uint4*>(Xp + off) =
        make_uint4(add_noise<F8>(x.x, n[0]), add_noise<F8>(x.y, n[1]), add_noise<F8>(x.z, n[2]), add_noise<F8>(x.w, n[3]));
  });
}
// the generation alone: each patch's noise folded to one word, G[p] = xor over chunks c of rotl(n0 ^ n1 ^ n2 ^ n3, c)
template <int NS>
__global__ void __launch_bounds__(256) sha_gen(uint32_t* __restrict__ G, uint32_t pa, uint32_t pb, const __grid_constant__ Sig sg) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pa + pb) return;
  const bool isA = i < pa;
  uint32_t x = 0;
  sha_patch<NS>(sg.w, 2 * (NS - 1) + (isA ? 0 : 1), isA ? i : i - pa,
                [&](int c, const uint32_t(&n)[4]) { x ^= __funnelshift_l(n[0] ^ n[1] ^ n[2] ^ n[3], n[0] ^ n[1] ^ n[2] ^ n[3], c); });
  G[i] = x;
}
// the noise itself into a rows x (128 KB) buffer of int8 (side 0, A's tag): the shared E of pvgemm.cu
template <int NS>
__global__ void __launch_bounds__(256) sha_fill(uint8_t* __restrict__ E, uint32_t patches, uint32_t KB, const __grid_constant__ Sig sg) {
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= patches) return;
  const size_t k = (size_t)KB * 128, base = (size_t)(p / KB) * 4 * k + (size_t)(p % KB) * 128;
  sha_patch<NS>(sg.w, 2 * (NS - 1), p, [&](int c, const uint32_t(&n)[4]) {
    *reinterpret_cast<uint4*>(E + base + (size_t)(c / 8) * k + (c % 8) * 16) = make_uint4(n[0], n[1], n[2], n[3]);
  });
}
