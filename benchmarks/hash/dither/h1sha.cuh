// SHA-512 on sm_90a / sm_100a for H1's per-unit streams (pouw.circuit.seeds): a seed's noise block t is
// SHA-512(NOISE_TAG || seed || u64be t), one compression of a 96-byte message whose words 0..10 are the seed's. So rounds
// 0..10 (the midstate) and every schedule term free of word 11 are a seed's, computed once (NoiseMid), and a block runs
// rounds 11..79 on them with immediate constants.
#pragma once
#include <cstdint>
typedef unsigned long long u64;
typedef unsigned int u32;
namespace sh {
typedef unsigned long long u64;
#include "../../codec_tensor/sha512.cuh"
}  // namespace sh

// NOISE_TAG = b"verity/pouw/noise/v1".ljust(24, b"\0") as three big-endian words
constexpr u64 TAG0 = 0x7665726974792f70ULL, TAG1 = 0x6f75772f6e6f6973ULL, TAG2 = 0x652f763100000000ULL;

__device__ constexpr u64 KC[80] = WT_SHA512_K;
__device__ constexpr u64 IVC[8] = WT_SHA512_IV;

__device__ __forceinline__ u64 rotr(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

#ifndef H1_PIPE
#define H1_PIPE 0
#endif
#if H1_PIPE
// H1_PIPE's bits move rotations off the integer ALU pipe, which takes every SHF, LOP3 and IADD3 and alone bounds the
// compression (2,837 of its 2,967 instructions), onto the IMAD pipe, which issues beside it: 4 Sigma1's, 8 Sigma0's,
// 16 sigma0's, 32 sigma1's (and 64, in h1u.cu, the fold's rotate-and-add). A 32-bit funnel shift becomes two IMADs by a
// power of two read from __constant__ memory, which ptxas can't see, so it can't turn them back into a shift. (The
// adds can't move: ptxas splits mad.wide.u32's 64-bit add into IADD3 and IMAD.X, so the low word's carry stays an IADD3.)
__constant__ u32 PW2[32] = {1u << 0,  1u << 1,  1u << 2,  1u << 3,  1u << 4,  1u << 5,  1u << 6,  1u << 7,
                            1u << 8,  1u << 9,  1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
                            1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
                            1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31};
constexpr bool P_S1 = H1_PIPE & 4, P_S0 = H1_PIPE & 8, P_s0 = H1_PIPE & 16, P_s1 = H1_PIPE & 32;

// the low word of (b:a) >> n, 0 < n < 32, as hi(a 2^(32-n)) + b 2^(32-n) (their bits are disjoint)
__device__ __forceinline__ u32 fsr_imad(u32 a, u32 b, int n) {
  const u32 m = PW2[32 - n];
  u32 r;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(m), "r"(b * m));
  return r;
}
template <bool F>
__device__ __forceinline__ u64 rotp(u64 x, int n) {
  if (!F) return rotr(x, n);
  u32 lo = u32(x), hi = u32(x >> 32);
  if (n >= 32) {
    const u32 t = lo;
    lo = hi;
    hi = t;
    n -= 32;
  }
  return (u64(fsr_imad(hi, lo, n)) << 32) | fsr_imad(lo, hi, n);
}
template <bool F>
__device__ __forceinline__ u64 shrp(u64 x, int n) {
  if (!F) return x >> n;
  const u32 lo = u32(x), hi = u32(x >> 32);
  return (u64(__umulhi(hi, PW2[32 - n])) << 32) | fsr_imad(lo, hi, n);
}
__device__ __forceinline__ u64 S0(u64 a) { return rotp<P_S0>(a, 28) ^ rotp<P_S0>(a, 34) ^ rotp<P_S0>(a, 39); }
__device__ __forceinline__ u64 S1(u64 e) { return rotp<P_S1>(e, 14) ^ rotp<P_S1>(e, 18) ^ rotp<P_S1>(e, 41); }
__device__ __forceinline__ u64 s0(u64 w) { return rotp<P_s0>(w, 1) ^ rotp<P_s0>(w, 8) ^ shrp<P_s0>(w, 7); }
__device__ __forceinline__ u64 s1(u64 w) { return rotp<P_s1>(w, 19) ^ rotp<P_s1>(w, 61) ^ shrp<P_s1>(w, 6); }
#else
__device__ __forceinline__ u64 S0(u64 a) { return rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39); }
__device__ __forceinline__ u64 S1(u64 e) { return rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41); }
__device__ __forceinline__ u64 s0(u64 w) { return rotr(w, 1) ^ rotr(w, 8) ^ (w >> 7); }
__device__ __forceinline__ u64 s1(u64 w) { return rotr(w, 19) ^ rotr(w, 61) ^ (w >> 6); }
#endif
__device__ __forceinline__ u64 chf(u64 e, u64 f, u64 g) { return g ^ (e & (f ^ g)); }
__device__ __forceinline__ u64 maj(u64 a, u64 b, u64 c) { return (a & b) | (c & (a | b)); }

// DEP.d[i]: whether schedule word i depends on word 11 (the block counter); words 12..15 are the padding's constants
struct DepT {
  bool d[80];
  constexpr DepT() : d() {
    for (int i = 0; i < 80; ++i) d[i] = i == 11 || (i >= 16 && (d[i - 2] || d[i - 7] || d[i - 15] || d[i - 16]));
  }
};
__device__ constexpr DepT DEP{};
#define dep(i) DEP.d[i]
// from word NC on, every schedule term depends on t (so the seed's terms stop there)
constexpr int NC = 40;
constexpr bool nc_ok() {
  DepT t;
  for (int i = NC; i < 80; ++i)
    if (!(t.d[i - 2] && t.d[i - 7] && t.d[i - 15] && t.d[i - 16])) return false;
  return true;
}
static_assert(nc_ok(), "a schedule word past NC has a term free of the block counter");

// state rotation by naming: round r's (a..h) are v[(8 - r) & 7 ..]
#define RND(r, wv)                                                                                    \
  {                                                                                                   \
    u64& a = v[(8 - (r)) & 7]; u64& b = v[(9 - (r)) & 7]; u64& c = v[(10 - (r)) & 7];                 \
    u64& d = v[(11 - (r)) & 7]; u64& e = v[(12 - (r)) & 7]; u64& f = v[(13 - (r)) & 7];               \
    u64& g = v[(14 - (r)) & 7]; u64& h = v[(15 - (r)) & 7];                                           \
    const u64 t1 = h + S1(e) + chf(e, f, g) + KC[r] + (wv);                                           \
    d += t1;                                                                                          \
    h = t1 + S0(a) + maj(a, b, c);                                                                    \
  }

// a seed's noise stream: the state after rounds 0..10, and c[i - 16], schedule word i's terms free of t (all of it where
// !dep(i)), 16 <= i < NC
struct NoiseMid {
  u64 mid[8];
  u64 c[NC - 16];
};

__device__ __forceinline__ void noise_mid(const u64* seed, NoiseMid& m) {
  u64 w[16];
  w[0] = TAG0;
  w[1] = TAG1;
  w[2] = TAG2;
#pragma unroll
  for (int j = 0; j < 8; ++j) w[3 + j] = seed[j];
  w[11] = 0;
  w[12] = 0x8000000000000000ULL;
  w[13] = 0;
  w[14] = 0;
  w[15] = 96 * 8;
  u64 c[NC];
#pragma unroll
  for (int i = 16; i < NC; ++i) {
    u64 x = 0;
    if (!dep(i - 2)) x += s1(i - 2 < 16 ? w[i - 2] : c[i - 2]);
    if (!dep(i - 7)) x += i - 7 < 16 ? w[i - 7] : c[i - 7];
    if (!dep(i - 15)) x += s0(i - 15 < 16 ? w[i - 15] : c[i - 15]);
    if (!dep(i - 16)) x += i - 16 < 16 ? w[i - 16] : c[i - 16];
    c[i] = x;
    m.c[i - 16] = x;
  }
  u64 v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = IVC[j];
#pragma unroll
  for (int r = 0; r < 11; ++r) RND(r, w[r]);
#pragma unroll
  for (int j = 0; j < 8; ++j) m.mid[j] = v[j];
}

// block t of the stream: its digest's eight big-endian words; m may live in shared memory (read once a block)
__device__ __forceinline__ void noise_block(const NoiseMid& m, u64 t, u64 s[8]) {
  u64 v[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) v[q] = m.mid[q];
  u64 W[80];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = i == 11 ? t : i == 12 ? 0x8000000000000000ULL : i == 15 ? 96 * 8 : 0;
#pragma unroll
  for (int i = 16; i < 80; ++i) {
    if (!dep(i)) {
      W[i] = m.c[i - 16];
    } else {
      u64 x = i < NC ? m.c[i - 16] : 0;
      if (dep(i - 2)) x += s1(W[i - 2]);
      if (dep(i - 7)) x += W[i - 7];
      if (dep(i - 15)) x += s0(W[i - 15]);
      if (dep(i - 16)) x += W[i - 16];
      W[i] = x;
    }
  }
#pragma unroll
  for (int r = 11; r < 80; ++r) RND(r, W[r]);
  // after 80 rounds the names are back in place: v[0] is a
#pragma unroll
  for (int q = 0; q < 8; ++q) s[q] = IVC[q] + v[q];
}

// a whole message's digest from its 64-bit words (len16 x 16 of them, padding included), the rolled compression
__device__ __forceinline__ void sha512_words(const u64* msg, int blocks, u64 s[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = IVC[j];
#pragma unroll 1
  for (int b = 0; b < blocks; ++b) {
    u64 w[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) w[j] = msg[16 * b + j];
    sh::sha512_compress(s, w);
  }
}
