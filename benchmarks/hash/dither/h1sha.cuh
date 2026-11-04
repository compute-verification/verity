// SHA-512 on sm_90a / sm_100a for H1's per-unit streams (pouw.circuit.seeds): a seed's noise chunk t is
// SHA-512(NOISE_TAG || seed || u64be t), one compression of a 96-byte message whose words 0..10 are the seed's. So rounds
// 0..10 (the midstate) and every schedule term free of word 11 are a seed's, computed once (NoiseMid), and a block runs
// rounds 11..79 on them with immediate constants.
#pragma once
#include <cstdint>
typedef unsigned long long u64;
typedef unsigned int u32;
#include "../../../verity/ml/kernels/sha512.cuh"

// NOISE_TAG = b"verity/pouw/noise/v1".ljust(24, b"\0") as three big-endian words
constexpr u64 TAG0 = 0x7665726974792f70ULL, TAG1 = 0x6f75772f6e6f6973ULL, TAG2 = 0x652f763100000000ULL;

__device__ constexpr u64 KC[80] = {SX_K_LIST};

__device__ __forceinline__ u64 rotr(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

// H1S_ROTF = R (0..6) puts R of each round's six Sigma rotations, and H1S_ROTFW = R (0..4) R of the schedule's four sigma
// rotations, on the FMA pipe (rotr_f) instead of two funnel shifts on the ALU pipe: sm_90's ALU pipe (SHF, LOP3, IADD3)
// is 16 lanes a sub-partition and its IMAD pipe another 16, and a unit's hashing is nearly all ALU-pipe work, so moving
// some of it can raise the units' throughput where the ALU pipe binds. Both 0 is the funnel
// shifts' code as before. Served on the H200, R = 6 is slower than 0: Dither's T 64 step +20% with today's units, +6%
// with J10-R's. rotr_f<N>: with lo and hi x's halves (swapped for N > 32), n = N mod 32 and m = 2^(32 - n)
// read from constant memory (so ptxas can't fold it back into shifts), c = {lo >> n, lo << 32 - n} = {mul.hi(lo, m),
// mul.lo(lo, m')} and the rotation is hi m + c (one mad.wide.u32): its words (hi << 32 - n) | (lo >> n) and
// (hi >> n) | (lo << 32 - n) are sums of disjoint bits. m' is m from a second table, so ptxas can't merge the two
// multiplies of lo into one IMAD.WIDE, whose halves land in the pair's other order.
#ifndef H1S_ROTF
#define H1S_ROTF 0
#endif
#ifndef H1S_ROTFW
#define H1S_ROTFW 0
#endif
static_assert(H1S_ROTF >= 0 && H1S_ROTF <= 6 && H1S_ROTFW >= 0 && H1S_ROTFW <= 4, "H1S_ROTF is 0..6, H1S_ROTFW 0..4");
#if H1S_ROTF || H1S_ROTFW
#define H1S_P2_INIT                                                                                                  \
  {1u,        2u,        4u,         8u,         16u,        32u,      64u,      128u,     256u,      512u,      \
   1024u,     2048u,     4096u,      8192u,      16384u,     32768u,   65536u,   131072u,  262144u,   524288u,   \
   1048576u,  2097152u,  4194304u,   8388608u,   16777216u,  33554432u, 67108864u, 134217728u, 268435456u,         \
   536870912u, 1073741824u, 2147483648u}
__constant__ u32 H1S_P2[32] = H1S_P2_INIT;
__constant__ u32 H1S_P2B[32] = H1S_P2_INIT;
template <int N>
__device__ __forceinline__ u64 rotr_f(u64 x) {
  constexpr int n = N > 32 ? N - 32 : N, k = 32 - n;
  const u32 xl = (u32)x, xh = (u32)(x >> 32), lo = N > 32 ? xh : xl, hi = N > 32 ? xl : xh;
  u64 r;
  asm("{\n\t.reg .u32 cl, ch;\n\t.reg .u64 c;\n\t"
      "mul.hi.u32 cl, %1, %2;\n\tmul.lo.u32 ch, %1, %3;\n\tmov.b64 c, {cl, ch};\n\t"
      "mad.wide.u32 %0, %4, %2, c;\n\t}"
      : "=l"(r) : "r"(lo), "r"(H1S_P2[k]), "r"(H1S_P2B[k]), "r"(hi));
  return r;
}
template <int N, bool F>
__device__ __forceinline__ u64 rot(u64 x) {
  if constexpr (F) return rotr_f<N>(x);
  return rotr(x, N);
}

__device__ __forceinline__ u64 S0(u64 a) {
  return rot<28, (1 < H1S_ROTF)>(a) ^ rot<34, (3 < H1S_ROTF)>(a) ^ rot<39, (5 < H1S_ROTF)>(a);
}
__device__ __forceinline__ u64 S1(u64 e) {
  return rot<14, (0 < H1S_ROTF)>(e) ^ rot<18, (2 < H1S_ROTF)>(e) ^ rot<41, (4 < H1S_ROTF)>(e);
}
__device__ __forceinline__ u64 s0(u64 w) { return rot<1, (0 < H1S_ROTFW)>(w) ^ rot<8, (2 < H1S_ROTFW)>(w) ^ (w >> 7); }
__device__ __forceinline__ u64 s1(u64 w) { return rot<19, (1 < H1S_ROTFW)>(w) ^ rot<61, (3 < H1S_ROTFW)>(w) ^ (w >> 6); }
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
  sx::iv(v);
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
  u64 iv[8];
  sx::iv(iv);
#pragma unroll
  for (int q = 0; q < 8; ++q) s[q] = iv[q] + v[q];
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
  u64 iv[8];
  sx::iv(iv);
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    sa[q] = iv[q] + va[q];
    sb[q] = iv[q] + vb[q];
  }
}

// a whole message's digest from its 64-bit words (len16 x 16 of them, padding included), the rolled compression
__device__ __forceinline__ void sha512_words(const u64* msg, int blocks, u64 s[8]) {
  sx::iv(s);
#pragma unroll 1
  for (int b = 0; b < blocks; ++b) {
    u64 w[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) w[j] = msg[16 * b + j];
    sx::compress<0, 1>(s, w);
  }
}
