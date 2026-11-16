// ffill: the H200's FFMA filler beside the LI-CP-W GEMM, dither-2s-licpw-redux-l32's [filler] (ffill_ref.py states it
// and is the reference). A thread takes one step on each of three of its accumulator words a MMA: u = IMAD(w, one, BIAS),
// f = u read as binary32, x = fma.rn.f32(f, ±s, x), y = fma.rn.f32(f, ±t, y), the sign alternating step by step. Both
// FMAs are an explicit fma.rn.f32 without .ftz (SASS FFMA: subnormals kept, every NaN 0x7FFFFFFF), so neither contraction
// nor -fmad decides what runs; the negated s and t are words of their own (the sign bit flipped once), not a per-step op.
#pragma once
#include <cstdint>

constexpr uint32_t FF_BIAS = 0x4B400000u;
constexpr uint64_t FF_KEYS = 0x46464D4146494C4Cull;  // "FFMAFILL"
constexpr int FF_STEPS = 3;                           // filler steps a thread a MMA
constexpr int FF_REGS[FF_STEPS] = {0, 43, 86};        // ffill_ref.REGS: the accumulator words of the 128 a thread they read

__host__ __device__ inline uint64_t ff_smix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__host__ __device__ inline uint64_t ff_seed(uint64_t c0, uint64_t c1, uint64_t tile) {
  return ff_smix(ff_smix(c0 ^ ff_smix(tile)) ^ c1);
}
// (s, t, x0, y0) as binary32 words: s and t odd significands in (1, 2), t != s; x0 and y0 in [1, 2).
__host__ __device__ inline void ff_coins(uint64_t c0, uint64_t c1, uint64_t tile, uint32_t w[4]) {
  const uint64_t k = ff_seed(c0, c1, tile), r = ff_smix(k ^ FF_KEYS), q = ff_smix(k ^ FF_KEYS ^ 1ull);
  w[0] = 0x3F800001u | ((uint32_t)r & 0x7FFFFEu);
  w[1] = 0x3F800001u | ((uint32_t)(r >> 32) & 0x7FFFFEu);
  if (w[1] == w[0]) w[1] ^= 2u;
  w[2] = 0x3F800000u | ((uint32_t)q & 0x7FFFFFu);
  w[3] = 0x3F800000u | ((uint32_t)(q >> 32) & 0x7FFFFFu);
}

#ifdef __CUDACC__
__device__ __forceinline__ float ff_fma(float a, float b, float c) {
  float d;
  asm volatile("fma.rn.f32 %0,%1,%2,%3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ uint32_t ff_imad(uint32_t w, uint32_t one) {
  uint32_t u;
  asm volatile("mad.lo.u32 %0,%1,%2,%3;" : "=r"(u) : "r"(w), "r"(one), "r"(FF_BIAS));
  return u;
}
struct FfChain {
  float x, y, s, t, ns, nt;
  uint32_t one;  // a runtime 1: IMAD(w, one, BIAS) stays an IMAD
  __device__ FfChain(const uint32_t c[4], uint32_t one_) : one(one_) {
    s = __uint_as_float(c[0]);
    t = __uint_as_float(c[1]);
    ns = __uint_as_float(c[0] ^ 0x80000000u);
    nt = __uint_as_float(c[1] ^ 0x80000000u);
    x = __uint_as_float(c[2]);
    y = __uint_as_float(c[3]);
  }
  // step n on word w, NEG = n & 1
  template <int NEG>
  __device__ __forceinline__ void step(uint32_t w) {
    const float f = __uint_as_float(ff_imad(w, one));
    x = ff_fma(f, NEG ? ns : s, x);
    y = ff_fma(f, NEG ? nt : t, y);
  }
  // the three steps of one MMA's words, steps 3m .. 3m + 2: their signs alternate from (m & 1)
  template <int ODD>
  __device__ __forceinline__ void mma(uint32_t w0, uint32_t w1, uint32_t w2) {
    step<ODD>(w0);
    step<!ODD>(w1);
    step<ODD>(w2);
  }
};
#endif
