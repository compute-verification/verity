// P2 v1 on the GPU (the POUS store's docs/efficient-crypto/p2-spec-draft.md, frozen v1), as a shared library for ctypes:
// decode (two signed squarings on the tuned Karatsuba path of ../p2_gpu/p2dec.cu) and encode (two roots), on rows of
// fixed-width big-endian W/8-byte integers, with each block's mask K and tweak t given as rows too (the harness expands them
// on the device by SHAKE256 with the spec's v1 framing, before these kernels run, and frees them after).
//
//   decode  m = d(sigma_K(d(c))) - t mod p        d(z) = z^2 mod p, negated when z is odd (p2dec.cu's psqr_fix)
//   encode  c = rho(sigma_K(rho(m + t mod p)))    rho(y) = the parity-canonical z with d(z) = y
//   rho     s = y^e, e = (p + 1) / 4 = 2^a (2^n - 1) + r  (the host passes a, n, r): 2^n - 1 by an addition chain on the
//           bits of n, then a squarings and y^r; about W squarings and 25 multiplications, each reduced by 2^W = c0 mod p
//           (sqr_fix's fold); then the even member of {s, p - s} if d of it is y, else the odd one, else an error
//   sigma   sigma_K(r) = r xor K if that is below p, else r
//
// Every value is one CGBN instance of 32 threads (TPI 32), P2_TPB threads a block. Rows are staged through shared memory,
// big-endian words to little-endian limbs. flags: bit 0, a tweak t >= p (the spec draws the next candidate, which this
// library leaves to its caller); bit 1, a y with no root (p is not a prime 3 mod 4, or the input is not canonical).
// Build (Makefile): nvcc -DP2_BITS=16448 -DP2_TPI=32 -DP2_PARTS=2 -DP2_SYM -DP2_TPB=64 -DXMP_WMAD -DP2_LIBRARY -shared.
#include "../p2_gpu/p2dec.cu"

#if !defined(P2_LIBRARY) || P2_PARTS != 2 || P2_TPI != 32
#error "p2v1.cu builds on p2dec.cu's tuned path as a library: -DP2_LIBRARY -DP2_PARTS=2 -DP2_TPI=32"
#endif

#define P2V1_WARPS (P2_TPB / 32)

#ifdef P2_FAST
#include "p2sq.cuh"
#endif
#ifdef P2_KEYS
#include "p2keys.cuh"
#endif

#ifdef __CUDA_ARCH__
// big-endian row of NL words -> little-endian limbs
__device__ __forceinline__ void load_be(uint32_t *sm, const uint32_t *__restrict__ row, uint32_t lane) {
  for (uint32_t g = lane; g < NL; g += 32) sm[NL - 1 - g] = __byte_perm(row[g], 0, 0x0123);
}
__device__ __forceinline__ void store_be(uint32_t *__restrict__ row, const uint32_t *sm, uint32_t lane) {
  for (uint32_t g = lane; g < NL; g += 32) row[g] = __byte_perm(sm[NL - 1 - g], 0, 0x0123);
}

// r = a * b mod p, a and b canonical: sqr_fix's reduction of the wide product, without its parity negation
__device__ __forceinline__ void mul_p(env_t &env, bn_t &r, const bn_t &a, const bn_t &b, uint32_t c0) {
  bnw_t w;
  cgbn_mul_wide(env, w, a, b);
  uint32_t hi_c = cgbn_mul_ui32(env, w._high, w._high, c0);
  int32_t carry = cgbn_add(env, r, w._low, w._high);
  fold32(env, r, (hi_c + (uint32_t)carry) * c0, c0);
}
__device__ __forceinline__ void sqr_p(env_t &env, bn_t &r, const bn_t &a, uint32_t c0) {
  bnw_t w;
  cgbn_sqr_wide(env, w, a);
  uint32_t hi_c = cgbn_mul_ui32(env, w._high, w._high, c0);
  int32_t carry = cgbn_add(env, r, w._low, w._high);
  fold32(env, r, (hi_c + (uint32_t)carry) * c0, c0);
}

// s = y^(2^a (2^n - 1) + r) mod p
__device__ void pow_e(env_t &env, bn_t &s, const bn_t &y, uint32_t a, uint32_t n, uint32_t r, uint32_t c0) {
  bn_t f, t;
  cgbn_set(env, f, y);                                        // f = y^(2^k - 1), k = 1
  uint32_t k = 1;
  for (int bit = 30 - __clz(n); bit >= 0; bit--) {
    cgbn_set(env, t, f);
    for (uint32_t i = 0; i < k; i++) sqr_p(env, t, t, c0);
    mul_p(env, f, t, f, c0);                                  // f(2k) = f(k)^(2^k) f(k)
    k *= 2;
    if ((n >> bit) & 1) {
      sqr_p(env, f, f, c0);
      mul_p(env, f, f, y, c0);                                // f(k + 1) = f(k)^2 y
      k += 1;
    }
  }
  for (uint32_t i = 0; i < a; i++) sqr_p(env, f, f, c0);
  if (r == 0) {
    cgbn_set(env, s, f);
    return;
  }
  cgbn_set(env, t, y);                                        // t = y^r, left to right
  for (int bit = 30 - __clz(r); bit >= 0; bit--) {
    sqr_p(env, t, t, c0);
    if ((r >> bit) & 1) mul_p(env, t, t, y, c0);
  }
  mul_p(env, s, f, t, c0);
}

// z with d(z) = y; false when there is none
__device__ bool rho(env_t &env, bn_t &z, const bn_t &y, const bn_t &p, uint32_t a, uint32_t n, uint32_t r, uint32_t c0) {
  if (cgbn_equals_ui32(env, y, 0)) {
    cgbn_set_ui32(env, z, 0);
    return true;
  }
  bn_t s, other, even, odd, dz;
  pow_e(env, s, y, a, n, r, c0);
  cgbn_sub(env, other, p, s);
  bool s_even = (cgbn_get_ui32(env, s) & 1) == 0;
  sel(even, s_even, s, other);
  sel(odd, s_even, other, s);
  sqr_fix(env, dz, even, c0);
  if (cgbn_equals(env, dz, y)) {
    cgbn_set(env, z, even);
    return true;
  }
  sqr_fix(env, dz, odd, c0);
  if (cgbn_equals(env, dz, y)) {
    cgbn_set(env, z, odd);
    return true;
  }
  return false;
}
#endif

__global__ void decode_v1_kernel(const uint32_t *__restrict__ cin, uint32_t *__restrict__ mout, const uint32_t *__restrict__ kin,
                                 const uint32_t *__restrict__ tin, uint32_t n, uint32_t c0, uint32_t *__restrict__ flags) {
#ifdef __CUDA_ARCH__
  context_t ctx(cgbn_no_checks);
  penv_t e(ctx.env<penv_t>());
  __shared__ uint32_t smem[P2V1_WARPS][3][NL];
  uint32_t (*sm)[NL] = smem[threadIdx.x / 32];
  const uint32_t lane = threadIdx.x % 32;
  pbn_t a[P2_PARTS], b[P2_PARTS], x[P2_PARTS];
  const uint32_t stride = (gridDim.x * blockDim.x) / P2_TPI;
  for (uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) / P2_TPI; j < n; j += stride) {
    __syncwarp();
    load_be(sm[0], cin + (size_t)j * NL, lane);
    load_be(sm[1], kin + (size_t)j * NL, lane);
    load_be(sm[2], tin + (size_t)j * NL, lane);
    __syncwarp();
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_load(e, a[k], (pmem_t *)&sm[0][k * NP]);
    psqr_fix(e, b, a, c0);                                      // b = d(c)
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) {
      cgbn_load(e, x[k], (pmem_t *)&sm[1][k * NP]);             // K
      cgbn_bitwise_xor(e, x[k], b[k], x[k]);
      a[k] = x[k];
    }
    uint32_t ge_p = add1_n<P2_PARTS>(e, a, c0);                 // (d(c) ^ K) >= p  <=>  + c0 carries out of 2^W
    psel<P2_PARTS>(b, ge_p == 0, x, b);                         // b = sigma_K(d(c))
    psqr_fix(e, a, b, c0);                                      // a = d(b)
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_load(e, b[k], (pmem_t *)&sm[2][k * NP]);   // t
    {
      pbn_t tt[P2_PARTS];
      #pragma unroll
      for (int k = 0; k < P2_PARTS; k++) tt[k] = b[k];
      if (add1_n<P2_PARTS>(e, tt, c0) && lane == 0) atomicOr(flags, 1u);
    }
    if (sub_n<P2_PARTS>(e, a, a, b)) sub1_n<P2_PARTS>(e, a, c0); // m = a - t mod p
    __syncwarp();
    #pragma unroll
    for (int k = 0; k < P2_PARTS; k++) cgbn_store(e, (pmem_t *)&sm[0][k * NP], a[k]);
    __syncwarp();
    store_be(mout + (size_t)j * NL, sm[0], lane);
  }
#endif
}

__global__ void encode_v1_kernel(const uint32_t *__restrict__ min, uint32_t *__restrict__ cout, const uint32_t *__restrict__ kin,
                                 const uint32_t *__restrict__ tin, uint32_t n, uint32_t c0, uint32_t ea, uint32_t en, uint32_t er,
                                 uint32_t *__restrict__ flags) {
#ifdef __CUDA_ARCH__
  context_t ctx(cgbn_no_checks);
  env_t env(ctx.env<env_t>());
  __shared__ uint32_t smem[P2V1_WARPS][3][NL];
  uint32_t (*sm)[NL] = smem[threadIdx.x / 32];
  const uint32_t lane = threadIdx.x % 32;
  bn_t p, m, kk, t, y, z, x, tmp;
  cgbn_set_ui32(env, p, 0);
  cgbn_sub_ui32(env, p, p, c0);                                // p = 2^W - c0 (wraps from 0)
  const uint32_t stride = (gridDim.x * blockDim.x) / P2_TPI;
  for (uint32_t j = (blockIdx.x * blockDim.x + threadIdx.x) / P2_TPI; j < n; j += stride) {
    __syncwarp();
    load_be(sm[0], min + (size_t)j * NL, lane);
    load_be(sm[1], kin + (size_t)j * NL, lane);
    load_be(sm[2], tin + (size_t)j * NL, lane);
    __syncwarp();
    cgbn_load(env, m, (mem_t *)sm[0]);
    cgbn_load(env, kk, (mem_t *)sm[1]);
    cgbn_load(env, t, (mem_t *)sm[2]);
    if (cgbn_add_ui32(env, tmp, t, c0) && lane == 0) atomicOr(flags, 1u);   // t >= p
    int32_t carry = cgbn_add(env, y, m, t);                    // y = m + t mod p: past 2^W, 2^W = c0; else at most one p
    if (carry) {
      cgbn_add_ui32(env, y, y, c0);
    } else if (cgbn_add_ui32(env, tmp, y, c0)) {
      cgbn_set(env, y, tmp);
    }
    bool ok = rho(env, z, y, p, ea, en, er, c0);
    cgbn_bitwise_xor(env, x, z, kk);                            // sigma_K
    if (!cgbn_add_ui32(env, tmp, x, c0)) cgbn_set(env, z, x);
    ok = rho(env, y, z, p, ea, en, er, c0) && ok;
    if (!ok && lane == 0) atomicOr(flags, 2u);
    __syncwarp();
    cgbn_store(env, (mem_t *)sm[0], y);
    __syncwarp();
    store_be(cout + (size_t)j * NL, sm[0], lane);
  }
#endif
}

static uint32_t grid_for(const void *kernel, uint32_t n) {
  int dev = 0, sms = 0, occ = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, P2_TPB, 0);
  uint64_t full = ((uint64_t)n * P2_TPI + P2_TPB - 1) / P2_TPB;
  uint64_t cap = (uint64_t)(occ > 0 ? occ : 1) * (sms > 0 ? sms : 1) * 4;
  return (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(full, cap));
}

extern "C" int p2v1_width_bits() { return P2_BITS; }

extern "C" int p2v1_decode(const void *c, void *m, const void *k, const void *t, uint32_t n, uint32_t c0, void *flags,
                           void *stream) {
  if (n == 0) return 0;
#ifdef P2_FAST
  {
    int dev = 0, sms = 0, occ = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void *)decode_sq_kernel, 32 * P2SQ_WARPS, 0);
    uint64_t full = ((uint64_t)n + P2SQ_WARPS - 1) / P2SQ_WARPS, cap = (uint64_t)(occ > 0 ? occ : 1) * (sms > 0 ? sms : 1);
    decode_sq_kernel<<<(uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(full, cap)), 32 * P2SQ_WARPS, 0, (cudaStream_t)stream>>>(
        (const uint32_t *)c, (uint32_t *)m, (const uint32_t *)k, (const uint32_t *)t, n, c0, (uint32_t *)flags);
    return (int)cudaGetLastError();
  }
#endif
  decode_v1_kernel<<<grid_for((const void *)decode_v1_kernel, n), P2_TPB, 0, (cudaStream_t)stream>>>(
      (const uint32_t *)c, (uint32_t *)m, (const uint32_t *)k, (const uint32_t *)t, n, c0, (uint32_t *)flags);
  return (int)cudaGetLastError();
}

extern "C" int p2v1_encode(const void *m, void *c, const void *k, const void *t, uint32_t n, uint32_t c0, uint32_t ea,
                           uint32_t en, uint32_t er, void *flags, void *stream) {
  if (n == 0) return 0;
  encode_v1_kernel<<<grid_for((const void *)encode_v1_kernel, n), P2_TPB, 0, (cudaStream_t)stream>>>(
      (const uint32_t *)m, (uint32_t *)c, (const uint32_t *)k, (const uint32_t *)t, n, c0, ea, en, er, (uint32_t *)flags);
  return (int)cudaGetLastError();
}

#ifdef P2_KEYS
// K and t rows (as gpu.Kernels.keys) of blocks first .. first + count - 1, under the expansion prefix; -1 if it doesn't fit
extern "C" int p2v1_keys(const uint8_t *prefix, int plen, uint64_t first, uint32_t count, uint64_t per_segment, void *k, void *t,
                         void *stream) {
  p2keys::Mid mid;
  if (!p2keys::midstate(prefix, plen, mid)) return -1;
  if (count == 0) return 0;
  const uint32_t per = 32 * p2keys::WARPS;
  p2keys::keys_kernel<<<(count + per - 1) / per, per, 0, (cudaStream_t)stream>>>(mid, first, count, per_segment, (uint64_t *)k,
                                                                                  (uint64_t *)t);
  return (int)cudaGetLastError();
}
#endif
