// P2's squaring on one GPU, CGBN (NVlabs) multi-precision, which ../p2_v1/p2v1.cu builds on (-DP2_LIBRARY): with
// p = 2^W - C0 (p = 3 mod 4), a^2 mod p, then p - a^2 when a is odd. From the efficient-cryptography campaign
// (bc-28faaf0e: the tuned kernel of run r20260928-065407-16f0).
// P2_PARTS=1 (default) squares with CGBN's cgbn_sqr_wide; 2 or 4 is the tuned Karatsuba path below.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

#ifndef P2_BITS
#error "define P2_BITS"
#endif
#ifndef P2_TPI
#error "define P2_TPI"
#endif
#ifndef P2_TPB
#define P2_TPB 128
#endif
#ifndef P2_PARTS
#define P2_PARTS 1
#endif

static const uint32_t NL = P2_BITS / 32;
typedef cgbn_context_t<P2_TPI> context_t;
typedef cgbn_env_t<context_t, P2_BITS> env_t;
typedef env_t::cgbn_t bn_t;
typedef env_t::cgbn_wide_t bnw_t;
typedef cgbn_mem_t<P2_BITS> mem_t;

#ifdef __CUDA_ARCH__
__device__ __forceinline__ void sel(bn_t &r, bool take_a, const bn_t &a, const bn_t &b) {
  #pragma unroll
  for (int i = 0; i < env_t::LIMBS; i++) r._limbs[i] = take_a ? a._limbs[i] : b._limbs[i];
}

// r (< 2^W) + v' * 2^W where v = v' * c0 < 2^32  ->  r reduced mod p
__device__ __forceinline__ void fold32(env_t &env, bn_t &r, uint32_t v, uint32_t c0) {
  bn_t tmp;
  int32_t carry = cgbn_add_ui32(env, r, r, v);
  cgbn_add_ui32(env, tmp, r, c0);                 // when carry, r < 2^32 so this cannot wrap
  sel(r, carry != 0, tmp, r);
  int32_t ge_p = cgbn_add_ui32(env, tmp, r, c0);  // r >= p  <=>  r + c0 >= 2^W; then r - p = low(r + c0)
  sel(r, ge_p != 0, tmp, r);
}

// a^2 mod p, then p - a^2 when a is odd (the parity sign fix)
__device__ __forceinline__ void sqr_fix(env_t &env, bn_t &r, const bn_t &a, uint32_t c0) {
  bnw_t w;
  bool odd = cgbn_get_ui32(env, a) & 1;
  cgbn_sqr_wide(env, w, a);
  uint32_t hi_c = cgbn_mul_ui32(env, w._high, w._high, c0);
  int32_t carry = cgbn_add(env, r, w._low, w._high);
  fold32(env, r, (hi_c + (uint32_t)carry) * c0, c0);
  cgbn_bitwise_complement(env, w._low, r);        // p - r = (2^W - 1 - r) - (c0 - 1)
  cgbn_sub_ui32(env, w._low, w._low, c0 - 1);
  sel(r, odd && !cgbn_equals_ui32(env, r, 0), w._low, r);
}

#endif

#if P2_PARTS > 1
// Tuned path: every w-bit value is P2_PARTS parts of w/P bits, each a CGBN number of the part env (same TPI), so the
// squaring can split without moving limbs between threads. a^2 is Karatsuba on parts, recursively:
//   a = lo + hi*B:  a^2 = lo^2 + (lo^2 + hi^2 - (lo - hi)^2)*B + hi^2*B^2,  i.e. 3 half squarings instead of 4 half
// products; the cross term 2*lo*hi is formed once. P2_SYM (P = 2 only) forms it as lo*hi doubled instead.
// Carries returned by CGBN are group-uniform, so branching on them only diverges between instances.
typedef cgbn_env_t<context_t, P2_BITS / P2_PARTS> penv_t;
typedef penv_t::cgbn_t pbn_t;
typedef penv_t::cgbn_wide_t pbnw_t;
typedef cgbn_mem_t<P2_BITS / P2_PARTS> pmem_t;
static const uint32_t NP = NL / P2_PARTS;
static_assert(NL % P2_PARTS == 0 && (P2_PARTS & (P2_PARTS - 1)) == 0, "parts must be a power of 2 dividing w/32");

#ifdef __CUDA_ARCH__
template <int N> __device__ __forceinline__ void psel(pbn_t *r, bool take_a, const pbn_t *a, const pbn_t *b) {
  #pragma unroll
  for (int k = 0; k < N; k++)
    #pragma unroll
    for (int i = 0; i < penv_t::LIMBS; i++) r[k]._limbs[i] = take_a ? a[k]._limbs[i] : b[k]._limbs[i];
}
template <int N> __device__ __forceinline__ uint32_t add_n(penv_t &e, pbn_t *r, const pbn_t *a, const pbn_t *b) {
  uint32_t c = cgbn_add(e, r[0], a[0], b[0]) != 0;
  #pragma unroll
  for (int k = 1; k < N; k++) {
    uint32_t c1 = cgbn_add(e, r[k], a[k], b[k]) != 0;
    if (c) c1 += cgbn_add_ui32(e, r[k], r[k], 1) != 0;
    c = c1;
  }
  return c;
}
template <int N> __device__ __forceinline__ uint32_t sub_n(penv_t &e, pbn_t *r, const pbn_t *a, const pbn_t *b) {
  uint32_t c = cgbn_sub(e, r[0], a[0], b[0]) != 0;
  #pragma unroll
  for (int k = 1; k < N; k++) {
    uint32_t c1 = cgbn_sub(e, r[k], a[k], b[k]) != 0;
    if (c) c1 += cgbn_sub_ui32(e, r[k], r[k], 1) != 0;
    c = c1;
  }
  return c;
}
// r += v (in place); returns the carry out of the top part
template <int N> __device__ __forceinline__ uint32_t add1_n(penv_t &e, pbn_t *r, uint32_t v) {
  uint32_t c = cgbn_add_ui32(e, r[0], r[0], v) != 0;
  #pragma unroll
  for (int k = 1; k < N; k++)
    if (c) c = cgbn_add_ui32(e, r[k], r[k], 1) != 0;
  return c;
}
template <int N> __device__ __forceinline__ uint32_t sub1_n(penv_t &e, pbn_t *r, uint32_t v) {
  uint32_t c = cgbn_sub_ui32(e, r[0], r[0], v) != 0;
  #pragma unroll
  for (int k = 1; k < N; k++)
    if (c) c = cgbn_sub_ui32(e, r[k], r[k], 1) != 0;
  return c;
}

template <int Q> struct ksq {  // z[0..2Q) = a[0..Q)^2
  static __device__ __forceinline__ void sq(penv_t &e, pbn_t *z, const pbn_t *a) {
    const int H = Q / 2;
    pbn_t m[2 * H];
    uint32_t mc;
#if defined(P2_SYM)
    static_assert(Q == 2, "P2_SYM is one level");
    pbnw_t w;
    cgbn_mul_wide(e, w, a[0], a[1]);
    m[0] = w._low; m[1] = w._high;
    mc = add_n<2>(e, m, m, m);                      // m = 2*lo*hi
    ksq<H>::sq(e, z, a);
    ksq<H>::sq(e, z + 2 * H, a + H);
#else
    pbn_t d[H];
#if defined(P2_KSEL)
    pbn_t d2[H];                                    // d = |lo - hi| as both differences and a select
    uint32_t bw = sub_n<H>(e, d, a, a + H);
    sub_n<H>(e, d2, a + H, a);
    psel<H>(d, bw != 0, d2, d);
#else
    if (sub_n<H>(e, d, a, a + H)) {                 // d = |lo - hi|
      #pragma unroll
      for (int k = 0; k < H; k++) cgbn_bitwise_complement(e, d[k], d[k]);
      add1_n<H>(e, d, 1);
    }
#endif
    ksq<H>::sq(e, m, d);
    ksq<H>::sq(e, z, a);
    uint32_t bb = sub_n<2 * H>(e, m, z, m);         // lo^2 - d^2 (may borrow; the sum below is >= 0)
    ksq<H>::sq(e, z + 2 * H, a + H);
    mc = add_n<2 * H>(e, m, m, z + 2 * H) - bb;     // m + mc*B^2 = 2*lo*hi, mc in {0, 1}
#endif
    uint32_t c = add_n<2 * H>(e, z + H, z + H, m);
    add1_n<H>(e, z + 3 * H, c + mc);
  }
};
template <> struct ksq<1> {
  static __device__ __forceinline__ void sq(penv_t &e, pbn_t *z, const pbn_t *a) {
    pbnw_t w;
    cgbn_sqr_wide(e, w, a[0]);
    z[0] = w._low; z[1] = w._high;
  }
};

__device__ __forceinline__ void pfold32(penv_t &e, pbn_t *r, uint32_t v, uint32_t c0) {
  pbn_t tmp[P2_PARTS];
  if (add1_n<P2_PARTS>(e, r, v)) add1_n<P2_PARTS>(e, r, c0);   // after a wrap r < 2^32, so + c0 cannot wrap
  #pragma unroll
  for (int k = 0; k < P2_PARTS; k++) tmp[k] = r[k];
  uint32_t ge_p = add1_n<P2_PARTS>(e, tmp, c0);                 // r >= p  <=>  r + c0 >= 2^W
  psel<P2_PARTS>(r, ge_p != 0, tmp, r);
}

__device__ __forceinline__ void psqr_fix(penv_t &e, pbn_t *r, const pbn_t *a, uint32_t c0) {
  pbn_t z[2 * P2_PARTS];
  bool odd = cgbn_get_ui32(e, a[0]) & 1;
  ksq<P2_PARTS>::sq(e, z, a);
  uint32_t h = 0, c = 0;                                        // r = lo + c0*hi, parts carried upward
  #pragma unroll
  for (int k = 0; k < P2_PARTS; k++) {
    pbn_t x;
    uint32_t hk = cgbn_mul_ui32(e, x, z[P2_PARTS + k], c0);
    uint32_t c1 = cgbn_add(e, r[k], z[k], x) != 0;
    c = c1 + (cgbn_add_ui32(e, r[k], r[k], h + c) != 0);
    h = hk;
  }
  pfold32(e, r, (h + c) * c0, c0);
  bool zero = true;
  #pragma unroll
  for (int k = 0; k < P2_PARTS; k++) {
    zero = zero && cgbn_equals_ui32(e, r[k], 0);
    cgbn_bitwise_complement(e, z[k], r[k]);                     // p - r = (2^W - 1 - r) - (c0 - 1)
  }
  sub1_n<P2_PARTS>(e, z, c0 - 1);
  psel<P2_PARTS>(r, odd && !zero, z, r);
}
#endif
#endif
