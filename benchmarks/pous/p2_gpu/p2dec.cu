// P2 public decode throughput on one GPU, CGBN (NVlabs) multi-precision.
// From the efficient-cryptography campaign (its tuned
// kernel). Added for benchmarks/pous: -DP2_DERIVE expands fresh per-block keys on the
// device (M1's key schedule, verity.core.protocols.pous.schemes.p2 v2): per segment s a 32-byte seed, per block j of the segment a ChaCha8
// stream with nonce j, read as big-endian W/8-byte words: K_j the first and t_j the second. The spec rejection-samples t_j
// below p; a second word >= p (probability C0/2^W) makes the run exit 3 instead of drawing the third. Only the seeds cross
// from the host, so nothing key-sized is stored. The shared K and t_j = t0 + j*step (M2) stay the default build. C may be
// read from a file (argv[6]) instead of filled on the device; on the device it is little-endian 32-bit limbs.
// Per block j, with p = 2^W - C0 (p = 3 mod 4), public mask K, tweak t_j = t0 + j*step mod p:
//   s = c^2 mod p, s = p - s if c odd;  r = s^K if s^K < p else s;
//   y = r^2 mod p, y = p - y if r odd;  m = y - t_j mod p.
// Build: nvcc -O3 -arch=sm_89 -DP2_BITS=5504 -DP2_TPI=8 [-DXMP_IMAD] [-DP2_PARTS=2|4 [-DP2_SYM]] -I CGBN/include -o p2dec p2dec.cu
//   P2_PARTS=1 (default) squares with CGBN's cgbn_sqr_wide; 2 or 4 is the tuned Karatsuba path below.
// Run:   p2dec params.bin specials.bin sample.bin payload_bytes reps [c.bin [seeds.bin]]   (seeds.bin with -DP2_DERIVE:
//        u32 segments, u32 blocks per segment, then 32 bytes per segment)
// -DP2_LIBRARY leaves out this file's kernels and main, so another file (../p2_v1/p2v1.cu) can build on its squaring.
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
#ifndef P2_UNROLL
#define P2_UNROLL 0
#endif
#ifndef P2_PARTS
#define P2_PARTS 1
#endif
#if P2_PARTS == 1
#define IMPL "cgbn"
#elif defined(P2_SYM)
#define IMPL "sym"
#elif defined(P2_KSEL)
#define IMPL "xkara"
#else
#define IMPL "kara"
#endif
#if defined(XMP_IMAD)
#define MAD "imad"
#elif defined(XMP_XMAD)
#define MAD "xmad"
#else
#define MAD "wmad"
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const uint32_t NL = P2_BITS / 32;
typedef cgbn_context_t<P2_TPI> context_t;
typedef cgbn_env_t<context_t, P2_BITS> env_t;
typedef env_t::cgbn_t bn_t;
typedef env_t::cgbn_wide_t bnw_t;
typedef cgbn_mem_t<P2_BITS> mem_t;

struct params_t { mem_t K, t0, step; uint32_t c0; };
#ifdef P2_DERIVE
#define KEY_ARGS , const uint32_t *__restrict__ seeds, uint32_t bps, uint32_t *__restrict__ reject
#define KEY_PASS , dseeds, bps, dreject
#define KEYS "derived"
#else
#define KEY_ARGS
#define KEY_PASS
#define KEYS "shared"
#endif

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

// the same with v < 2^64
__device__ __forceinline__ void fold64(env_t &env, bn_t &r, uint64_t v, uint32_t c0) {
  bn_t tmp;
  cgbn_set_ui32(env, tmp, (uint32_t)v);
  cgbn_insert_bits_ui32(env, tmp, tmp, 32, 32, (uint32_t)(v >> 32));
  int32_t carry = cgbn_add(env, r, r, tmp);
  fold32(env, r, carry ? c0 : 0, c0);
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

#ifdef P2_DERIVE
// ChaCha8 (Bernstein: 64-bit block counter, then 64-bit nonce), one 64-byte block into out[16]
#define P2_ROTL(v, n) (((v) << (n)) | ((v) >> (32 - (n))))
#define P2_QR(a, b, c, d) x[a] += x[b]; x[d] ^= x[a]; x[d] = P2_ROTL(x[d], 16); x[c] += x[d]; x[b] ^= x[c]; x[b] = P2_ROTL(x[b], 12); \
                          x[a] += x[b]; x[d] ^= x[a]; x[d] = P2_ROTL(x[d], 8);  x[c] += x[d]; x[b] ^= x[c]; x[b] = P2_ROTL(x[b], 7);
__device__ __forceinline__ void chacha8_block(const uint32_t *__restrict__ key, uint64_t ctr, uint64_t nonce, uint32_t *out) {
  uint32_t x[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3], key[4], key[5], key[6],
                    key[7], (uint32_t)ctr, (uint32_t)(ctr >> 32), (uint32_t)nonce, (uint32_t)(nonce >> 32)};
  uint32_t s[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) s[i] = x[i];
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    P2_QR(0, 4, 8, 12) P2_QR(1, 5, 9, 13) P2_QR(2, 6, 10, 14) P2_QR(3, 7, 11, 15)
    P2_QR(0, 5, 10, 15) P2_QR(1, 6, 11, 12) P2_QR(2, 7, 8, 13) P2_QR(3, 4, 9, 14)
  }
  #pragma unroll
  for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}
static const uint32_t KSB = (2 * NL + 15) / 16;   // ChaCha blocks per data block: K_j then t_j

// keystream word g lands where cgbn_load finds big-endian K_j (words 0..NL-1) and t_j (NL..2NL-1) as little-endian limbs
__device__ __forceinline__ uint32_t be_slot(uint32_t g) {
  return g < NL ? NL - 1 - g : g < 2 * NL ? 3 * NL - 1 - g : g;
}
#endif

#if defined(P2_DERIVE) && P2_PARTS == 1
#error "P2_DERIVE is the tuned path's (P2_PARTS 2 or 4)"
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

#ifndef P2_LIBRARY
__global__ void decode_kernel(const params_t *__restrict__ prm, const mem_t *__restrict__ cin, mem_t *__restrict__ mout,
                              uint32_t n KEY_ARGS) {
#ifdef __CUDA_ARCH__
  context_t ctx(cgbn_no_checks);
  penv_t e(ctx.env<penv_t>());
  const uint32_t c0 = prm->c0;
  pbn_t a[P2_PARTS], b[P2_PARTS], x[P2_PARTS];
  const uint32_t stride = (gridDim.x * blockDim.x) / P2_TPI;
#ifdef P2_DERIVE
  static_assert(P2_TPI == 32, "P2_DERIVE: one instance per warp");
  __shared__ uint32_t ksm[P2_TPB / 32][KSB * 16];
  uint32_t *ks = ksm[threadIdx.x / 32];
  const uint32_t lane = threadIdx.x % 32;
#endif
  for (uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) / P2_TPI; j < n; j += stride) {
#ifdef P2_DERIVE
    __syncwarp();                                               // the previous block's keys are read
    for (uint32_t i = lane; i < KSB; i += 32) {
      uint32_t z[16];
      chacha8_block(&seeds[8 * (j / bps)], i, j % bps, z);
      #pragma unroll
      for (int q = 0; q < 16; q++) ks[be_slot(16 * i + q)] = __byte_perm(z[q], 0, 0x0123);
    }
    __syncwarp();
#endif
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_load(e, a[k], (pmem_t *)&cin[j]._limbs[k * NP]);
    psqr_fix(e, b, a, c0);                                      // b = s
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) {
#ifdef P2_DERIVE
      cgbn_load(e, x[k], (pmem_t *)&ks[k * NP]);               // K_j
#else
      cgbn_load(e, x[k], (pmem_t *)&prm->K._limbs[k * NP]);
#endif
      cgbn_bitwise_xor(e, x[k], b[k], x[k]);
      a[k] = x[k];
    }
    uint32_t ge_p = add1_n<P2_PARTS>(e, a, c0);                 // s^K >= p  <=>  (s^K) + c0 >= 2^W
    psel<P2_PARTS>(b, ge_p == 0, x, b);                         // b = r
    psqr_fix(e, a, b, c0);                                      // a = y
#ifdef P2_DERIVE
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_load(e, b[k], (pmem_t *)&ks[NL + k * NP]);       // b = t_j, if below p
    {
      pbn_t tt[P2_PARTS];
      #pragma unroll
      for (int k = 0; k < P2_PARTS; k++) tt[k] = b[k];
      if (add1_n<P2_PARTS>(e, tt, c0) && lane == 0) atomicOr(reject, 1u);   // t >= p  <=>  t + c0 >= 2^W
    }
#else
    uint32_t h = 0, c = 0;                                      // b = j*step, top word h + c
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) {
      cgbn_load(e, x[k], (pmem_t *)&prm->step._limbs[k * NP]);
      uint32_t hk = cgbn_mul_ui32(e, b[k], x[k], j);
      c = cgbn_add_ui32(e, b[k], b[k], h + c) != 0;             // h < j, so h + c < 2^32
      h = hk;
    }
    uint64_t v = (uint64_t)(h + c) * c0;
    cgbn_set_ui32(e, x[0], (uint32_t)v);
    cgbn_insert_bits_ui32(e, x[0], x[0], 32, 32, (uint32_t)(v >> 32));
    c = cgbn_add(e, b[0], b[0], x[0]) != 0;
    #pragma unroll
    for (int k = 1; k < P2_PARTS; k++)
      if (c) c = cgbn_add_ui32(e, b[k], b[k], 1) != 0;
    pfold32(e, b, c ? c0 : 0, c0);
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_load(e, x[k], (pmem_t *)&prm->t0._limbs[k * NP]);
    c = add_n<P2_PARTS>(e, b, b, x);
    pfold32(e, b, c ? c0 : 0, c0);                              // b = t_j
#endif
    if (sub_n<P2_PARTS>(e, a, a, b)) sub1_n<P2_PARTS>(e, a, c0); // m = y - t_j (+ 2^W - c0 on borrow)
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_store(e, (pmem_t *)&mout[j]._limbs[k * NP], a[k]);
  }
#endif
}
#endif
#else
#ifndef P2_LIBRARY
__global__ void decode_kernel(const params_t *__restrict__ prm, const mem_t *__restrict__ cin, mem_t *__restrict__ mout,
                              uint32_t n KEY_ARGS) {
#ifdef __CUDA_ARCH__
  context_t ctx(cgbn_no_checks);
  env_t env(ctx.env<env_t>());
  const uint32_t c0 = prm->c0;
  bn_t a, b, x;
  const uint32_t stride = (gridDim.x * blockDim.x) / P2_TPI;
  for (uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) / P2_TPI; j < n; j += stride) {
    cgbn_load(env, a, (mem_t *)&cin[j]);
    sqr_fix(env, b, a, c0);                          // b = s
    cgbn_load(env, x, (mem_t *)&prm->K);
    cgbn_bitwise_xor(env, x, b, x);
    int32_t ge_p = cgbn_add_ui32(env, a, x, c0);     // s^K >= p  <=>  (s^K) + c0 >= 2^W
    sel(b, ge_p == 0, x, b);                         // b = r
    sqr_fix(env, a, b, c0);                          // a = y
    cgbn_load(env, x, (mem_t *)&prm->step);          // t_j = t0 + j*step mod p
    uint32_t hi = cgbn_mul_ui32(env, b, x, j);
    fold64(env, b, (uint64_t)hi * c0, c0);
    cgbn_load(env, x, (mem_t *)&prm->t0);
    int32_t carry = cgbn_add(env, b, b, x);
    fold32(env, b, carry ? c0 : 0, c0);
    int32_t borrow = cgbn_sub(env, a, a, b);         // m = y - t_j mod p
    cgbn_sub_ui32(env, x, a, c0);                    // (y - t + 2^W) - c0 = y - t + p
    sel(a, borrow != 0, x, a);
    cgbn_store(env, &mout[j], a);
  }
#endif
}
#endif
#endif

#ifndef P2_LIBRARY
__device__ __forceinline__ uint64_t splitmix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

__global__ void fill_kernel(mem_t *c, uint32_t n, uint64_t seed) {
  uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t total = (uint64_t)n * NL;
  for (uint64_t k = tid; k < total; k += (uint64_t)gridDim.x * blockDim.x) {
    uint32_t limb = (uint32_t)(k % NL);
    uint32_t v = (uint32_t)splitmix(seed ^ (k * 0x2545F4914F6CDD1Dull));
    if (limb == NL - 1 && v == 0xFFFFFFFFu) v = 0xFFFFFFFEu;  // keeps c < p (c0 < 2^32)
    c[k / NL]._limbs[limb] = v;
  }
}

static void read_all(const char *path, std::vector<uint8_t> &buf) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  buf.resize(sz);
  if (fread(buf.data(), 1, sz, f) != (size_t)sz) { perror(path); exit(1); }
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s params.bin specials.bin sample.bin payload_bytes reps [c.bin [seeds.bin]]\n", argv[0]); return 2; }
  std::vector<uint8_t> pb, sb;
  read_all(argv[1], pb);
  read_all(argv[2], sb);
  uint32_t w, c0;
  memcpy(&w, pb.data(), 4); memcpy(&c0, pb.data() + 4, 4);
  if (w != P2_BITS || pb.size() != 8 + 3 * NL * 4 || c0 >= 65536) { fprintf(stderr, "bad params for P2_BITS=%d\n", P2_BITS); return 2; }
  params_t hp;
  memcpy(hp.K._limbs, pb.data() + 8, NL * 4);
  memcpy(hp.t0._limbs, pb.data() + 8 + NL * 4, NL * 4);
  memcpy(hp.step._limbs, pb.data() + 8 + 2 * NL * 4, NL * 4);
  hp.c0 = c0;
  double payload = atof(argv[4]);
  int reps = atoi(argv[5]);
  uint32_t n = (uint32_t)(payload / (P2_BITS / 8));
  size_t bytes = (size_t)n * sizeof(mem_t);

  uint32_t nspec; memcpy(&nspec, sb.data(), 4);
  const size_t rec = 8 + NL * 4;
  if (sb.size() != 4 + nspec * rec) { fprintf(stderr, "bad specials\n"); return 2; }

  int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  params_t *dp; mem_t *dc, *dm;
  CK(cudaMalloc(&dp, sizeof(params_t)));
  CK(cudaMalloc(&dc, bytes));
  CK(cudaMalloc(&dm, bytes));
  CK(cudaMemcpy(dp, &hp, sizeof(params_t), cudaMemcpyHostToDevice));
  if (argc > 6) {
    std::vector<uint8_t> cbuf; read_all(argv[6], cbuf);
    if (cbuf.size() < bytes) { fprintf(stderr, "c.bin holds %zu bytes, not %zu\n", cbuf.size(), bytes); return 2; }
    CK(cudaMemcpy(dc, cbuf.data(), bytes, cudaMemcpyHostToDevice));
  } else {
    fill_kernel<<<prop.multiProcessorCount * 8, 256>>>(dc, n, 0x5032474055ull ^ P2_BITS);
    CK(cudaGetLastError());
  }
#ifdef P2_DERIVE
  if (argc < 8) { fprintf(stderr, "-DP2_DERIVE needs c.bin and seeds.bin\n"); return 2; }
  uint32_t *dseeds, bps;
  {
    std::vector<uint8_t> sd; read_all(argv[7], sd);
    uint32_t segs; memcpy(&segs, sd.data(), 4); memcpy(&bps, sd.data() + 4, 4);
    if (sd.size() != 8 + (size_t)segs * 32 || bps == 0 || (uint64_t)segs * bps < n) {
      fprintf(stderr, "seeds.bin: %u segments of %u blocks, %zu bytes, for %u blocks\n", segs, bps, sd.size(), n); return 2;
    }
    CK(cudaMalloc(&dseeds, (size_t)segs * 32));
    CK(cudaMemcpy(dseeds, sd.data() + 8, (size_t)segs * 32, cudaMemcpyHostToDevice));
  }
  uint32_t *dreject;
  CK(cudaMalloc(&dreject, 4));
  CK(cudaMemset(dreject, 0, 4));
#endif
  std::vector<uint64_t> idx;
  for (uint32_t k = 0; k < nspec; k++) {
    uint64_t j; memcpy(&j, sb.data() + 4 + k * rec, 8);
    if (j >= n) { fprintf(stderr, "special index %llu >= n\n", (unsigned long long)j); return 2; }
    CK(cudaMemcpy(&dc[j], sb.data() + 4 + k * rec + 8, NL * 4, cudaMemcpyHostToDevice));
    idx.push_back(j);
  }

  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, decode_kernel));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, decode_kernel, P2_TPB, 0));
  uint64_t full = ((uint64_t)n * P2_TPI + P2_TPB - 1) / P2_TPB;
  uint32_t grid = (uint32_t)std::min<uint64_t>(full, (uint64_t)occ * prop.multiProcessorCount * 4);

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  decode_kernel<<<grid, P2_TPB>>>(dp, dc, dm, n KEY_PASS);  // warm-up
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> ms;
  for (int i = 0; i < reps; i++) {
    CK(cudaEventRecord(e0));
    decode_kernel<<<grid, P2_TPB>>>(dp, dc, dm, n KEY_PASS);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
  }
  CK(cudaGetLastError());
  std::sort(ms.begin(), ms.end());

  // host<->device copy rate over the same bytes (pinned), for the end-to-end bound
  void *hbuf; CK(cudaMallocHost(&hbuf, bytes));
  CK(cudaEventRecord(e0)); CK(cudaMemcpy(dm, hbuf, bytes, cudaMemcpyHostToDevice));
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float h2d; CK(cudaEventElapsedTime(&h2d, e0, e1));
  CK(cudaFreeHost(hbuf));
  decode_kernel<<<grid, P2_TPB>>>(dp, dc, dm, n KEY_PASS);  // restore outputs overwritten by the copy
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
#ifdef P2_DERIVE
  uint32_t rejected = 0;
  CK(cudaMemcpy(&rejected, dreject, 4, cudaMemcpyDeviceToHost));
  if (rejected) { fprintf(stderr, "a block's t_j word is >= p: the spec draws the next word, which this kernel does not\n"); return 3; }
#endif

  for (uint32_t k = 0; k < 1024 && k < n; k++) idx.push_back(k);
  uint64_t lcg = 12345;
  for (int k = 0; k < 1024; k++) { lcg = lcg * 6364136223846793005ull + 1442695040888963407ull; idx.push_back((lcg >> 11) % n); }
  idx.push_back(n - 1);
  FILE *f = fopen(argv[3], "wb");
  uint32_t hdr[2] = {P2_BITS, (uint32_t)idx.size()};
  fwrite(hdr, 4, 2, f);
  std::vector<uint32_t> cb(NL), mb(NL);
  for (uint64_t j : idx) {
    CK(cudaMemcpy(cb.data(), &dc[j], NL * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(mb.data(), &dm[j], NL * 4, cudaMemcpyDeviceToHost));
    fwrite(&j, 8, 1, f); fwrite(cb.data(), 4, NL, f); fwrite(mb.data(), 4, NL, f);
  }
  fclose(f);

  double stored = (double)n * (P2_BITS / 8), paybytes = (double)n * ((P2_BITS - 16) / 8);
  float med = ms[ms.size() / 2];
  printf("{\"w\":%d,\"c0\":%u,\"keys\":\"" KEYS "\",\"impl\":\"%s\",\"parts\":%d,\"tpi\":%d,\"mad\":\"%s\",\"unroll\":%d,\"tpb\":%d,\"blocks\":%u,\"stored_bytes\":%.0f,"
         "\"grid\":%u,\"occ_blocks_per_sm\":%d,\"regs\":%d,\"local_bytes\":%zu,\"sms\":%d,"
         "\"ms_min\":%.4f,\"ms_med\":%.4f,\"GBps_stored_min\":%.3f,\"GBps_stored_med\":%.3f,\"GBps_payload_min\":%.3f,"
         "\"h2d_GBps\":%.2f,\"gpu\":\"%s\"}\n",
         P2_BITS, c0, IMPL, P2_PARTS, P2_TPI, MAD, P2_UNROLL, P2_TPB, n, stored, grid, occ, fa.numRegs, fa.localSizeBytes, prop.multiProcessorCount,
         ms[0], med, stored / ms[0] / 1e6, stored / med / 1e6, paybytes / ms[0] / 1e6, bytes / h2d / 1e6, prop.name);
  return 0;
}
#endif
