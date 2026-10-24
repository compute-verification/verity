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

// H1S_ROT (a mask) takes SHA-512's rotations off the funnel shifter (SHF.W, the ALU's quarter-rate heavy pipe on
// sm_120, which the noise blocks' rounds keep busy): bit 0 sigma0's rotation by 8 as two byte permutes; bit 1 the
// schedule's sigma0 and sigma1 (rotations and shifts) as 32 x 32 -> 64-bit multiplies by powers of two on the FMA pipe
// (x 2^m is {x << m, x >> (32 - m)}: each half's two pieces of a rotation in one IMAD.WIDE); bit 2 Sigma1's, bit 3
// Sigma0's. Every form is the same function.
#ifndef H1S_ROT
#define H1S_ROT 0
#endif
template <int M>
__device__ __forceinline__ void mulw(u32 x, u32& lo, u32& hi) {
  asm("{\n\t.reg .u64 t;\n\tmul.wide.u32 t, %2, %3;\n\tmov.b64 {%0, %1}, t;\n\t}" : "=r"(lo), "=r"(hi) : "r"(x), "n"(1u << M));
}
template <int N>
__device__ __forceinline__ u64 rotr_m(u64 x) {
  static_assert(N % 32 != 0, "a rotation by 32 is the halves' swap");
  u32 lo = (u32)x, hi = (u32)(x >> 32);
  if constexpr (N > 32) {
    const u32 t = lo;
    lo = hi;
    hi = t;
  }
  u32 al, ah, bl, bh;
  mulw<32 - N % 32>(lo, al, ah);
  mulw<32 - N % 32>(hi, bl, bh);
  return ((u64)(bh | al) << 32) | (ah | bl);
}
template <int N>
__device__ __forceinline__ u64 shr_m(u64 x) {
  static_assert(N > 0 && N < 32, "a shift below 32");
  u32 al, ah, bl, bh;
  mulw<32 - N>((u32)x, al, ah);
  mulw<32 - N>((u32)(x >> 32), bl, bh);
  return ((u64)bh << 32) | (ah | bl);
}
__device__ __forceinline__ u64 rotr8_p(u64 x) {
  const u32 lo = (u32)x, hi = (u32)(x >> 32);
  return ((u64)__byte_perm(hi, lo, 0x4321) << 32) | __byte_perm(lo, hi, 0x4321);
}

#if H1S_ROT & 8
__device__ __forceinline__ u64 S0(u64 a) { return rotr_m<28>(a) ^ rotr_m<34>(a) ^ rotr_m<39>(a); }
#else
__device__ __forceinline__ u64 S0(u64 a) { return rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39); }
#endif
#if H1S_ROT & 4
__device__ __forceinline__ u64 S1(u64 e) { return rotr_m<14>(e) ^ rotr_m<18>(e) ^ rotr_m<41>(e); }
#else
__device__ __forceinline__ u64 S1(u64 e) { return rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41); }
#endif
#if H1S_ROT & 1
#define H1S_ROT8(w) rotr8_p(w)
#elif H1S_ROT & 2
#define H1S_ROT8(w) rotr_m<8>(w)
#else
#define H1S_ROT8(w) rotr(w, 8)
#endif
#if H1S_ROT & 2
__device__ __forceinline__ u64 s0(u64 w) { return rotr_m<1>(w) ^ H1S_ROT8(w) ^ shr_m<7>(w); }
__device__ __forceinline__ u64 s1(u64 w) { return rotr_m<19>(w) ^ rotr_m<61>(w) ^ shr_m<6>(w); }
#else
__device__ __forceinline__ u64 s0(u64 w) { return rotr(w, 1) ^ H1S_ROT8(w) ^ (w >> 7); }
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

// blocks ta and tb of one stream (noise_block's of each), their two compressions' rounds interleaved: two independent
// chains for a warp whose few neighbours on its scheduler leave one chain's latency exposed
#define RNDV(V, r, wv)                                                                                \
  {                                                                                                   \
    u64& a = V[(8 - (r)) & 7]; u64& b = V[(9 - (r)) & 7]; u64& c = V[(10 - (r)) & 7];                 \
    u64& d = V[(11 - (r)) & 7]; u64& e = V[(12 - (r)) & 7]; u64& f = V[(13 - (r)) & 7];               \
    u64& g = V[(14 - (r)) & 7]; u64& h = V[(15 - (r)) & 7];                                           \
    const u64 t1 = h + S1(e) + chf(e, f, g) + KC[r] + (wv);                                           \
    d += t1;                                                                                          \
    h = t1 + S0(a) + maj(a, b, c);                                                                    \
  }
__device__ __forceinline__ void noise_block2(const NoiseMid& m, u64 ta, u64 tb, u64 sa[8], u64 sb[8]) {
  u64 va[8], vb[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) va[q] = vb[q] = m.mid[q];
  // schedule word r computed at round r, so only the last 16 of each stream's words are live
  u64 WA[80], WB[80];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    WA[i] = i == 11 ? ta : i == 12 ? 0x8000000000000000ULL : i == 15 ? 96 * 8 : 0;
    WB[i] = i == 11 ? tb : WA[i];
  }
#pragma unroll
  for (int r = 11; r < 80; ++r) {
    if (r >= 16) {
      if (!dep(r)) {
        WA[r] = WB[r] = m.c[r - 16];
      } else {
        const u64 x = r < NC ? m.c[r - 16] : 0;
        u64 xa = x, xb = x;
        if (dep(r - 2)) { xa += s1(WA[r - 2]); xb += s1(WB[r - 2]); }
        if (dep(r - 7)) { xa += WA[r - 7]; xb += WB[r - 7]; }
        if (dep(r - 15)) { xa += s0(WA[r - 15]); xb += s0(WB[r - 15]); }
        if (dep(r - 16)) { xa += WA[r - 16]; xb += WB[r - 16]; }
        WA[r] = xa;
        WB[r] = xb;
      }
    }
    RNDV(va, r, WA[r]);
    RNDV(vb, r, WB[r]);
  }
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    sa[q] = IVC[q] + va[q];
    sb[q] = IVC[q] + vb[q];
  }
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
