// SHA-512 compression latency and throughput on the device: each thread chains d <- SHA-512(d) (a 64-byte message, one
// compression from the IV) n times and writes its last digest, which shabench.py checks against hashlib.
//   V 0  `sha512.cuh`'s sha512_compress (the rounds 16 at a time, `unroll 1` over the five passes: K read by index)
//   V 1  the same rounds fully unrolled (every K at a fixed constant-bank offset)
//   V 2  V 1 on 32-bit halves: the rotations as funnel shifts of the two halves, the sums by add.cc / addc
//   V 3..11  V 2 with some rotations on the FMA pipe (rotr_f): sm_90's ALU pipe (SHF, LOP3, IADD3) is 16 lanes a
//        sub-partition and its FMA-heavy pipe (IMAD) another 16, so moving rotations there can raise a compression's
//        throughput where the ALU pipe binds. V 3, 4, 5: 2, 4, 6 of a round's Sigma rotations (rotr_f's F 1); V 6: 4
//        and 2 of the schedule's sigma rotations (F 1); V 7, 8, 9, 10, 11: V 3, 4, 5, 6 and 6 with 4 sigma (F 2)
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

// rotr<N> on the FMA pipe, by m = 2^k (k = 32 - N % 32) read from constant memory, which ptxas can't fold back into
// shifts. With lo and hi the halves (swapped for N > 32), the rotation's low word is (hi << k) | (lo >> n) and its high
// word (hi >> n) | (lo << k), each an OR of disjoint bits, so a sum.
//   F 1: p = lo m, q = hi m + p.hi (mul.wide.u32 twice), high word q.hi + p.lo by an IMAD (the addend pair costs ptxas
//        a move and a zero)
//   F 2: p = lo m, then mad.lo(hi, m, p.hi) and mad.hi(hi, m, p.lo) (SASS's IMAD.HI adds a pair, so the same juggling)
//   F 3: c = {mul.hi(lo, m), mul.lo(lo, m')} = {lo >> n, lo << k} straight into a pair, then one mad.wide.u32
//        hi m + c: both words, three FMA-pipe instructions and no moves. m' is m from a second table, so ptxas can't merge
//        the two multiplies of lo into one IMAD.WIDE (whose halves land in the pair's other order)
__constant__ u32 kP2[33] = {1u,       2u,       4u,       8u,        16u,       32u,       64u,       128u,       256u,
                            512u,     1024u,    2048u,    4096u,     8192u,     16384u,    32768u,    65536u,     131072u,
                            262144u,  524288u,  1048576u, 2097152u,  4194304u,  8388608u,  16777216u, 33554432u,
                            67108864u, 134217728u, 268435456u, 536870912u, 1073741824u, 2147483648u, 1u};
__constant__ u32 kP2b[32] = {1u,       2u,       4u,       8u,        16u,       32u,       64u,       128u,       256u,
                             512u,     1024u,    2048u,    4096u,     8192u,     16384u,    32768u,    65536u,     131072u,
                             262144u,  524288u,  1048576u, 2097152u,  4194304u,  8388608u,  16777216u, 33554432u,
                             67108864u, 134217728u, 268435456u, 536870912u, 1073741824u, 2147483648u};
template <int N, int F>
__device__ __forceinline__ W2 rotr_f(W2 x) {
  if constexpr (F == 0) {
    return rotr<N>(x);
  } else {
    constexpr int n = N > 32 ? N - 32 : N, k = 32 - n;
    const u32 lo = N > 32 ? x.hi : x.lo, hi = N > 32 ? x.lo : x.hi, m = kP2[k];
    W2 r;
    if constexpr (F == 1) {
      const u64 p = (u64)lo * m;
      const u64 q = (u64)hi * m + (u32)(p >> 32);
      r = {(u32)q, (u32)p * kP2[32] + (u32)(q >> 32)};
    } else if constexpr (F == 2) {
      const u64 p = (u64)lo * m;
      asm("mad.lo.u32 %0, %2, %3, %4;\n\tmad.hi.u32 %1, %2, %3, %5;"
          : "=r"(r.lo), "=r"(r.hi) : "r"(hi), "r"(m), "r"((u32)(p >> 32)), "r"((u32)p));
    } else {
      asm("{\n\t.reg .u32 cl, ch;\n\t.reg .u64 c;\n\t"
          "mul.hi.u32 cl, %2, %3;\n\tmul.lo.u32 ch, %2, %4;\n\tmov.b64 c, {cl, ch};\n\t"
          "mad.wide.u32 c, %5, %3, c;\n\tmov.b64 {%0, %1}, c;\n\t}"
          : "=r"(r.lo), "=r"(r.hi) : "r"(lo), "r"(m), "r"(kP2b[k]), "r"(hi));
    }
    return r;
  }
}

// compress_halves with RS of each round's six Sigma rotations and RW of the schedule's four sigma rotations as rotr_f<., F>
// (Sigma1's and Sigma0's taken alternately, sigma0's and sigma1's likewise)
template <int RS, int RW, int F>
__device__ __forceinline__ void compress_halves_f(u64 s[8], u64 win[16]) {
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
        w[t] = add(add(w[t], x3(rotr_f<1, (0 < RW) * F>(w15), rotr_f<8, (2 < RW) * F>(w15), shr<7>(w15))),
                   add(w[(t + 9) & 15], x3(rotr_f<19, (1 < RW) * F>(w2), rotr_f<61, (3 < RW) * F>(w2), shr<6>(w2))));
      }
      const W2 e = v[4], a = v[0];
      const W2 kw = add(mk(sh::kSha512K[r + t]), w[t]);
      const W2 S1 = x3(rotr_f<14, (0 < RS) * F>(e), rotr_f<18, (2 < RS) * F>(e), rotr_f<41, (4 < RS) * F>(e));
      const W2 S0 = x3(rotr_f<28, (1 < RS) * F>(a), rotr_f<34, (3 < RS) * F>(a), rotr_f<39, (5 < RS) * F>(a));
      const W2 t1 = add(add(v[7], kw), add(S1, ch(e, v[5], v[6])));
      const W2 t2 = add(S0, maj(a, v[1], v[2]));
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

// V 3.. : compress_halves_f<VRS[V], VRW[V], VF[V]>
constexpr int NV = 17;
constexpr int VRS[NV] = {0, 0, 0, 2, 4, 6, 4, 2, 4, 6, 4, 6, 2, 4, 6, 4, 6};
constexpr int VRW[NV] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 4, 0, 0, 0, 2, 4};
constexpr int VF[NV] = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3};

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
    else if constexpr (V == 2)
      compress_halves(s, w);
    else
      compress_halves_f<VRS[V], VRW[V], VF[V]>(s, w);
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = s[j];
  }
  const long long t1 = clock64();
#pragma unroll
  for (int j = 0; j < 8; ++j) out[8 * i + j] = d[j];
  cyc[i] = t1 - t0;
}

template <int V>
static void launch(int v, int blocks, int threads, int n, const u64* d0, u64* out, long long* cyc, cudaStream_t st) {
  if (v == V)
    chain_k<V><<<blocks, threads, 0, st>>>(n, d0, out, cyc);
  else if constexpr (V + 1 < NV)
    launch<V + 1>(v, blocks, threads, n, d0, out, cyc, st);
}

extern "C" int sha_chain(int v, int blocks, int threads, int n, const u64* d0, u64* out, long long* cyc, cudaStream_t st) {
  if (v < 0 || v >= NV) return 1;
  launch<0>(v, blocks, threads, n, d0, out, cyc, st);
  return (int)cudaGetLastError();
}
