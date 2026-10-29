// sx::compress's rounds for a hash warpgroup that shares its SM's register file with MMA warpgroups (setmaxnreg): the same
// FIPS 180-4 rounds as sha512x.cuh's compress<V, ROLL> (its sigmas, Ch, Maj and constants), split so the caller chooses where
// each part of the state lives:
//
//   rounds<V, ROLL>(v, W)        the 80 rounds over the working variables v[0..7] (a .. h) and the chunk W, without the
//                                feed-forward: the caller adds v to the chaining value, which can stay in shared memory
//                                through the rounds (16 fewer live registers)
//   rounds2<V>(v0, W0, v1, W1)   two chunks' rounds interleaved round by round in ROLL 1's loop: the instruction-level
//                                parallelism of two chains for a warp alone on its scheduler, for twice the state
//
// W is the chunk's 16 words, overwritten by the schedule as compress's w is: a u64 array (registers), or Strided<S> over
// shared memory, word j of a thread's chunk at p[j * S] ([word][thread]: a warp's accesses to one word fall in distinct
// banks), 32 fewer live registers for the schedule's loads and stores.
#pragma once
#include "sha512x.cuh"

namespace sx {

template <int S> struct Strided {
  u64* p;
  SX_HD u64& operator[](int j) const { return p[j * S]; }
};

#ifdef __CUDACC__
// the 128 bytes at p (16-byte aligned, shared or global) as a chunk's 16 big-endian words, into W[0..15]; the chunk's
// 16-byte unit i at p + 16 (i ^ sw) (sw < 8: a row stored with its units XOR-swizzled)
template <class Sched>
__device__ __forceinline__ void load_chunk(const uint8_t* p, Sched W, int sw = 0) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint4 u = q[i ^ sw];
    W[2 * i] = be64(u.x, u.y);
    W[2 * i + 1] = be64(u.z, u.w);
  }
}
#endif

#define SXL_ROUND(a, b, c, d, e, f, g, h, kw)                                                                              \
  do {                                                                                                                     \
    const u64 t1_ = (h) + S1<M1>(e) + ch(e, f, g) + (kw);                                                                  \
    (d) += t1_;                                                                                                            \
    (h) = t1_ + S0<M0>(a) + maj(a, b, c);                                                                                  \
  } while (0)
#define SXL_W(X, j) (X[(j) & 15] += s1<m1>(X[((j) + 14) & 15]) + X[((j) + 9) & 15] + s0<m0>(X[((j) + 1) & 15]))
#define SXL_RAW(X, j) X[(j)]

template <int V, int ROLL, class Sched>
SX_HD void rounds(u64* v, Sched W) {
  static_assert(ROLL == 0 || ROLL == 1, "ROLL 0 (straight-line) or 1 (rounds 16..79 a loop of 16-round bodies)");
  constexpr int M0 = V & 3, M1 = (V >> 2) & 3, m0 = (V >> 4) & 3, m1 = (V >> 6) & 3;
  const u64* K = kconst();
  u64 a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5], g = v[6], h = v[7];
#define SXL_EIGHT(tk, tw, wf)                                                                                              \
  SXL_ROUND(a, b, c, d, e, f, g, h, K[(tk) + 0] + wf(W, (tw) + 0));                                                        \
  SXL_ROUND(h, a, b, c, d, e, f, g, K[(tk) + 1] + wf(W, (tw) + 1));                                                        \
  SXL_ROUND(g, h, a, b, c, d, e, f, K[(tk) + 2] + wf(W, (tw) + 2));                                                        \
  SXL_ROUND(f, g, h, a, b, c, d, e, K[(tk) + 3] + wf(W, (tw) + 3));                                                        \
  SXL_ROUND(e, f, g, h, a, b, c, d, K[(tk) + 4] + wf(W, (tw) + 4));                                                        \
  SXL_ROUND(d, e, f, g, h, a, b, c, K[(tk) + 5] + wf(W, (tw) + 5));                                                        \
  SXL_ROUND(c, d, e, f, g, h, a, b, K[(tk) + 6] + wf(W, (tw) + 6));                                                        \
  SXL_ROUND(b, c, d, e, f, g, h, a, K[(tk) + 7] + wf(W, (tw) + 7))
  SXL_EIGHT(0, 0, SXL_RAW);
  SXL_EIGHT(8, 8, SXL_RAW);
  if constexpr (ROLL) {
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
    for (int t = 16; t < 80; t += 16) {
      SXL_EIGHT(t, 0, SXL_W);
      SXL_EIGHT(t + 8, 8, SXL_W);
    }
  } else {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int t = 16; t < 80; t += 16) {
      SXL_EIGHT(t, 0, SXL_W);
      SXL_EIGHT(t + 8, 8, SXL_W);
    }
  }
#undef SXL_EIGHT
  v[0] = a; v[1] = b; v[2] = c; v[3] = d; v[4] = e; v[5] = f; v[6] = g; v[7] = h;
}

template <int V, class Sched>
SX_HD void rounds2(u64* v0, Sched W0, u64* v1, Sched W1) {
  constexpr int M0 = V & 3, M1 = (V >> 2) & 3, m0 = (V >> 4) & 3, m1 = (V >> 6) & 3;
  const u64* K = kconst();
  u64 a0 = v0[0], b0 = v0[1], c0 = v0[2], d0 = v0[3], e0 = v0[4], f0 = v0[5], g0 = v0[6], h0 = v0[7];
  u64 a1 = v1[0], b1 = v1[1], c1 = v1[2], d1 = v1[3], e1 = v1[4], f1 = v1[5], g1 = v1[6], h1 = v1[7];
#define SXL_ROUND2(a, b, c, d, e, f, g, h, t, j, wf)                                                                       \
  SXL_ROUND(a##0, b##0, c##0, d##0, e##0, f##0, g##0, h##0, K[t] + wf(W0, j));                                             \
  SXL_ROUND(a##1, b##1, c##1, d##1, e##1, f##1, g##1, h##1, K[t] + wf(W1, j))
#define SXL_EIGHT2(tk, tw, wf)                                                                                             \
  SXL_ROUND2(a, b, c, d, e, f, g, h, (tk) + 0, (tw) + 0, wf);                                                              \
  SXL_ROUND2(h, a, b, c, d, e, f, g, (tk) + 1, (tw) + 1, wf);                                                              \
  SXL_ROUND2(g, h, a, b, c, d, e, f, (tk) + 2, (tw) + 2, wf);                                                              \
  SXL_ROUND2(f, g, h, a, b, c, d, e, (tk) + 3, (tw) + 3, wf);                                                              \
  SXL_ROUND2(e, f, g, h, a, b, c, d, (tk) + 4, (tw) + 4, wf);                                                              \
  SXL_ROUND2(d, e, f, g, h, a, b, c, (tk) + 5, (tw) + 5, wf);                                                              \
  SXL_ROUND2(c, d, e, f, g, h, a, b, (tk) + 6, (tw) + 6, wf);                                                              \
  SXL_ROUND2(b, c, d, e, f, g, h, a, (tk) + 7, (tw) + 7, wf)
  SXL_EIGHT2(0, 0, SXL_RAW);
  SXL_EIGHT2(8, 8, SXL_RAW);
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (int t = 16; t < 80; t += 16) {
    SXL_EIGHT2(t, 0, SXL_W);
    SXL_EIGHT2(t + 8, 8, SXL_W);
  }
#undef SXL_EIGHT2
#undef SXL_ROUND2
  v0[0] = a0; v0[1] = b0; v0[2] = c0; v0[3] = d0; v0[4] = e0; v0[5] = f0; v0[6] = g0; v0[7] = h0;
  v1[0] = a1; v1[1] = b1; v1[2] = c1; v1[3] = d1; v1[4] = e1; v1[5] = f1; v1[6] = g1; v1[7] = h1;
}

#undef SXL_RAW
#undef SXL_W
#undef SXL_ROUND

}  // namespace sx
