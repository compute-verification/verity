// SHA-512 compression latency and throughput on the device: each thread chains d <- SHA-512(d) (a 64-byte message, one
// compression from the IV) n times and writes its last digest, which shabench.py checks against hashlib.
//   V 0  `sha512.cuh`'s sha512_compress (the rounds 16 at a time, `unroll 1` over the five passes: K read by index)
//   V 1  the same rounds fully unrolled (every K at a fixed constant-bank offset)
//   V 2  V 1 on 32-bit halves: the rotations as funnel shifts of the two halves, the sums by add.cc / addc
#include <cstdint>
#include <cuda_runtime.h>

namespace sh {
typedef unsigned long long u64;
#include "../../codec_tensor/sha512.cuh"
}  // namespace sh

typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ void compress_unrolled(u64 s[8], u64 w[16]) {
  u64 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const u64 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (sh::rotr64(w15, 1) ^ sh::rotr64(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] +
                (sh::rotr64(w2, 19) ^ sh::rotr64(w2, 61) ^ (w2 >> 6));
      }
      const u64 t1 = h + (sh::rotr64(e, 14) ^ sh::rotr64(e, 18) ^ sh::rotr64(e, 41)) + ((e & f) ^ (~e & g)) +
                     sh::kSha512K[r + t] + w[t];
      const u64 t2 = (sh::rotr64(a, 28) ^ sh::rotr64(a, 34) ^ sh::rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

struct W2 {
  u32 lo, hi;
};
__device__ __forceinline__ W2 mk(u64 x) { return {(u32)x, (u32)(x >> 32)}; }
__device__ __forceinline__ u64 un(W2 x) { return ((u64)x.hi << 32) | x.lo; }
template <int N>
__device__ __forceinline__ W2 rotr(W2 x) {
  if constexpr (N < 32)
    return {__funnelshift_r(x.lo, x.hi, N), __funnelshift_r(x.hi, x.lo, N)};
  else
    return {__funnelshift_r(x.hi, x.lo, N - 32), __funnelshift_r(x.lo, x.hi, N - 32)};
}
template <int N>
__device__ __forceinline__ W2 shr(W2 x) {
  return {__funnelshift_r(x.lo, x.hi, N), x.hi >> N};
}
__device__ __forceinline__ W2 x3(W2 a, W2 b, W2 c) { return {a.lo ^ b.lo ^ c.lo, a.hi ^ b.hi ^ c.hi}; }
__device__ __forceinline__ W2 add(W2 a, W2 b) {
  W2 r;
  asm("add.cc.u32 %0, %2, %4;\n\taddc.u32 %1, %3, %5;" : "=r"(r.lo), "=r"(r.hi) : "r"(a.lo), "r"(a.hi), "r"(b.lo), "r"(b.hi));
  return r;
}
__device__ __forceinline__ W2 ch(W2 e, W2 f, W2 g) { return {(e.lo & f.lo) ^ (~e.lo & g.lo), (e.hi & f.hi) ^ (~e.hi & g.hi)}; }
__device__ __forceinline__ W2 maj(W2 a, W2 b, W2 c) {
  return {(a.lo & b.lo) ^ (a.lo & c.lo) ^ (b.lo & c.lo), (a.hi & b.hi) ^ (a.hi & c.hi) ^ (b.hi & c.hi)};
}

__device__ __forceinline__ void compress_halves(u64 s[8], u64 win[16]) {
  W2 w[16], v[8];
#pragma unroll
  for (int j = 0; j < 16; ++j) w[j] = mk(win[j]);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = mk(s[j]);
#pragma unroll
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const W2 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] = add(add(w[t], x3(rotr<1>(w15), rotr<8>(w15), shr<7>(w15))),
                   add(w[(t + 9) & 15], x3(rotr<19>(w2), rotr<61>(w2), shr<6>(w2))));
      }
      const W2 e = v[4], a = v[0];
      const W2 kw = add(mk(sh::kSha512K[r + t]), w[t]);
      const W2 t1 = add(add(v[7], kw), add(x3(rotr<14>(e), rotr<18>(e), rotr<41>(e)), ch(e, v[5], v[6])));
      const W2 t2 = add(x3(rotr<28>(a), rotr<34>(a), rotr<39>(a)), maj(a, v[1], v[2]));
      v[7] = v[6];
      v[6] = v[5];
      v[5] = e;
      v[4] = add(v[3], t1);
      v[3] = v[2];
      v[2] = v[1];
      v[1] = a;
      v[0] = add(t1, t2);
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] += un(v[j]);
}

template <int V>
__global__ void chain_k(int n, const u64* __restrict__ d0, u64* __restrict__ out, long long* __restrict__ cyc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  u64 d[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) d[j] = d0[8 * i + j];
  const long long t0 = clock64();
  for (int k = 0; k < n; ++k) {
    u64 w[16], s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = d[j];
    w[8] = 0x8000000000000000ULL;
#pragma unroll
    for (int j = 9; j < 15; ++j) w[j] = 0;
    w[15] = 512;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = sh::kSha512Iv[j];
    if constexpr (V == 0)
      sh::sha512_compress(s, w);
    else if constexpr (V == 1)
      compress_unrolled(s, w);
    else
      compress_halves(s, w);
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = s[j];
  }
  const long long t1 = clock64();
#pragma unroll
  for (int j = 0; j < 8; ++j) out[8 * i + j] = d[j];
  cyc[i] = t1 - t0;
}

extern "C" int sha_chain(int v, int blocks, int threads, int n, const u64* d0, u64* out, long long* cyc, cudaStream_t st) {
  if (v == 0)
    chain_k<0><<<blocks, threads, 0, st>>>(n, d0, out, cyc);
  else if (v == 1)
    chain_k<1><<<blocks, threads, 0, st>>>(n, d0, out, cyc);
  else if (v == 2)
    chain_k<2><<<blocks, threads, 0, st>>>(n, d0, out, cyc);
  else
    return 1;
  return (int)cudaGetLastError();
}
