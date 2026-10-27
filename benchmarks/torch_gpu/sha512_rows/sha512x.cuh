// FIPS 180-4 SHA-512's compression for the serve-time commit (engine.py), on the device and the host, every word a
// big-endian u64. `compress<V, ROLL>(st, w)` has `sha::compress`'s contract (benchmarks/pouw/commit_gpu/sha512.cuh): it
// advances the chaining value st by the chunk w, and the schedule overwrites w.
//
// The rounds are p2keys4.cuh's (benchmarks/pous/p2_v1/p2keys4.cuh, `chunk`; copied, not included: the names a .. h rotate
// through the round macro instead of moving, Ch is g ^ (e & (f ^ g)), the schedule runs in place over 16 words), with a
// whole message schedule instead of P2's mostly zero counter chunk.
//
// V picks the pipe of each sigma's shifts, two bits each (S0 | S1 << 2 | s0 << 4 | s1 << 6): 0 funnel shifts on the integer
// ALU, 1 the high half of each rotation and shift as mul.hi / mad.lo by 2^k on the FMA pipe and the low half a funnel shift,
// 2 both halves on the FMA pipe, 3 four FMA-pipe instructions a rotation and three a shift, none on the ALU (rotr3). On sm_90
// IMAD issues to the FMA pipe, beside the ALU's SHF, LOP3 and IADD3, so a balanced V issues more than one integer pipe's worth
// a cycle. The multiply form is commit_gpu's rotr_fma; the power of two comes from constant memory so ptxas can't fold the
// multiply back into a shift. On the host every V is the plain function.
//
// ROLL = 1 runs rounds 16..79 as a loop of 16-round bodies (a quarter of the code: a lone warp otherwise waits on
// instruction fetch), 0 straight-line, 2 all 80 rounds as codec_tensor/sha512.cuh's sha512_compress, statement for
// statement: two thirds of ROLL 1's SASS, faster than it in dither's h1c and h1r, slower in its h1u.
#pragma once
#include <cstdint>
#include <cstring>

// a host build's own definitions stand (hash_host.cpp's __constant__ is static const)
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __constant__
#define __constant__
#endif
#endif

#define SX_HD __host__ __device__ __forceinline__

namespace sx {

using u64 = uint64_t;
using u32 = uint32_t;

#define SX_K_LIST                                                                                                          \
  0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,       \
      0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,   \
      0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,   \
      0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,   \
      0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,   \
      0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,   \
      0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,   \
      0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,   \
      0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,   \
      0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,   \
      0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,   \
      0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,   \
      0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,   \
      0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,   \
      0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,   \
      0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull

// SX_K_LIST stays defined for a constexpr copy, whose round constants are immediates (dither's h1sha.cuh)
__constant__ u64 K_DEV[80] = {SX_K_LIST};
static const u64 K_HOST[80] = {SX_K_LIST};

__constant__ u32 POW2[33] = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1u << 10, 1u << 11, 1u << 12, 1u << 13,
                             1u << 14, 1u << 15, 1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22,
                             1u << 23, 1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31, 0u};
// the IV as constant operands, for a caller slower with iv's immediates (dither's h1r, its noise from a midstate)
__constant__ u64 IV_DEV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                              0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                              0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
SX_HD const u64* kconst() {
#ifdef __CUDA_ARCH__
  return K_DEV;
#else
  return K_HOST;
#endif
}

// a 64-bit unsigned word type: uint64_t, or unsigned long long where a caller's arrays are that
template <class W> constexpr bool word64 = sizeof(W) == 8 && W(0) < W(-1);

template <class W> SX_HD void iv(W* st) {
  static_assert(word64<W>, "64-bit unsigned words");
  st[0] = 0x6a09e667f3bcc908ull; st[1] = 0xbb67ae8584caa73bull; st[2] = 0x3c6ef372fe94f82bull; st[3] = 0xa54ff53a5f1d36f1ull;
  st[4] = 0x510e527fade682d1ull; st[5] = 0x9b05688c2b3e6c1full; st[6] = 0x1f83d9abfb41bd6bull; st[7] = 0x5be0cd19137e2179ull;
}

SX_HD u32 bswap32(u32 x) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, 0, 0x0123);
#else
  return __builtin_bswap32(x);
#endif
}
// the big-endian u64 of 8 bytes loaded as two little-endian u32
SX_HD u64 be64(u32 lo, u32 hi) { return (u64)bswap32(lo) << 32 | bswap32(hi); }
SX_HD u64 bswap64(u64 x) { return be64((u32)x, (u32)(x >> 32)); }

#ifdef __CUDA_ARCH__
__device__ __forceinline__ u32 mulhi(u32 x, u32 m) {
  u32 r;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(m));
  return r;
}
__device__ __forceinline__ u32 madlo(u32 y, u32 m, u32 t) {
  u32 r;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(y), "r"(m), "r"(t));
  return r;
}
__device__ __forceinline__ u64 mulwide(u32 x, u32 m) {
  u64 r;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(r) : "r"(x), "r"(m));
  return r;
}
// (hi:lo) rotated right by n (0 < n < 32) as (rhi:rlo), and shi = hi >> n, all on the FMA pipe: with m = 2^(32 - n), one
// mul.wide each gives hi * m = (hi >> n : hi << (32 - n)) and lo * m = (lo >> n : lo << (32 - n)), and rhi =
// (lo << (32 - n)) + (hi >> n) and rlo = (lo >> n) + (hi << (32 - n)), also x >> n's low word, are each a sum of disjoint
// bit ranges, added by mad.lo with a multiplier of 1. Fewer instructions cost more in SASS: ptxas splits a mad.wide with a
// 64-bit addend into a mul.wide and an IADD3, and gives a mad.hi's addend two moves (IMAD.HI adds a register pair).
template <int n>
__device__ __forceinline__ void rotr3(u32 lo, u32 hi, u32& rhi, u32& rlo, u32& shi) {
  const u32 m = POW2[32 - n], one = POW2[0];
  const u64 p = mulwide(hi, m), q = mulwide(lo, m);
  shi = (u32)(p >> 32);
  rhi = madlo((u32)q, one, shi);
  rlo = madlo((u32)(q >> 32), one, (u32)p);
}
#endif

// x rotated right by N (0 < N < 64, N != 32), its shifts on the pipe MODE names
template <int MODE, int N>
SX_HD u64 rotr(u64 x) {
#ifdef __CUDA_ARCH__
  if constexpr (MODE == 3) {
    u32 lo = (u32)x, hi = (u32)(x >> 32), rhi, rlo, shi;
    if constexpr (N > 32) { const u32 t = lo; lo = hi; hi = t; }
    rotr3<(N > 32 ? N - 32 : N)>(lo, hi, rhi, rlo, shi);
    return (u64)rhi << 32 | rlo;
  } else if constexpr (MODE != 0) {
    u32 lo = (u32)x, hi = (u32)(x >> 32);
    if constexpr (N > 32) { const u32 t = lo; lo = hi; hi = t; }
    constexpr int n = N > 32 ? N - 32 : N;
    const u32 m = POW2[32 - n];
    const u32 rhi = madlo(lo, m, mulhi(hi, m));
    const u32 rlo = MODE == 2 ? madlo(hi, m, mulhi(lo, m)) : __funnelshift_r(lo, hi, n);
    return (u64)rhi << 32 | rlo;
  }
#endif
  return (x >> N) | (x << (64 - N));
}

// x >> N (0 < N < 32)
template <int MODE, int N>
SX_HD u64 shr(u64 x) {
#ifdef __CUDA_ARCH__
  if constexpr (MODE == 3) {
    u32 rhi, rlo, shi;
    rotr3<N>((u32)x, (u32)(x >> 32), rhi, rlo, shi);
    return (u64)shi << 32 | rlo;
  } else if constexpr (MODE != 0) {
    const u32 lo = (u32)x, hi = (u32)(x >> 32), m = POW2[32 - N];
    const u32 rhi = mulhi(hi, m);
    const u32 rlo = MODE == 2 ? madlo(hi, m, mulhi(lo, m)) : __funnelshift_r(lo, hi, N);
    return (u64)rhi << 32 | rlo;
  }
#endif
  return x >> N;
}

template <int M> SX_HD u64 S0(u64 a) { return rotr<M, 28>(a) ^ rotr<M, 34>(a) ^ rotr<M, 39>(a); }
template <int M> SX_HD u64 S1(u64 e) { return rotr<M, 14>(e) ^ rotr<M, 18>(e) ^ rotr<M, 41>(e); }
template <int M> SX_HD u64 s0(u64 w) { return rotr<M, 1>(w) ^ rotr<M, 8>(w) ^ shr<M, 7>(w); }
template <int M> SX_HD u64 s1(u64 w) { return rotr<M, 19>(w) ^ rotr<M, 61>(w) ^ shr<M, 6>(w); }
SX_HD u64 ch(u64 e, u64 f, u64 g) { return g ^ (e & (f ^ g)); }
SX_HD u64 maj(u64 a, u64 b, u64 c) { return (a & b) | (c & (a | b)); }

template <int V, int ROLL, class Word>
SX_HD void compress(Word* st, Word* W) {
  static_assert(word64<Word>, "64-bit unsigned words");
  constexpr int M0 = V & 3, M1 = (V >> 2) & 3, m0 = (V >> 4) & 3, m1 = (V >> 6) & 3;
  const u64* K = kconst();
  u64 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#define SX_ROUND(a, b, c, d, e, f, g, h, kw)                                                                               \
  do {                                                                                                                     \
    const u64 t1_ = (h) + S1<M1>(e) + ch(e, f, g) + (kw);                                                                  \
    (d) += t1_;                                                                                                            \
    (h) = t1_ + S0<M0>(a) + maj(a, b, c);                                                                                  \
  } while (0)
#define SX_W(j) (W[(j) & 15] += s1<m1>(W[((j) + 14) & 15]) + W[((j) + 9) & 15] + s0<m0>(W[((j) + 1) & 15]))
  // rounds tk .. tk + 7 on schedule words tw .. tw + 7; wf(j) is round j's word
#define SX_EIGHT(tk, tw, wf)                                                                                               \
  SX_ROUND(a, b, c, d, e, f, g, h, K[(tk) + 0] + wf((tw) + 0));                                                            \
  SX_ROUND(h, a, b, c, d, e, f, g, K[(tk) + 1] + wf((tw) + 1));                                                            \
  SX_ROUND(g, h, a, b, c, d, e, f, K[(tk) + 2] + wf((tw) + 2));                                                            \
  SX_ROUND(f, g, h, a, b, c, d, e, K[(tk) + 3] + wf((tw) + 3));                                                            \
  SX_ROUND(e, f, g, h, a, b, c, d, K[(tk) + 4] + wf((tw) + 4));                                                            \
  SX_ROUND(d, e, f, g, h, a, b, c, K[(tk) + 5] + wf((tw) + 5));                                                            \
  SX_ROUND(c, d, e, f, g, h, a, b, K[(tk) + 6] + wf((tw) + 6));                                                            \
  SX_ROUND(b, c, d, e, f, g, h, a, K[(tk) + 7] + wf((tw) + 7))
#define SX_RAW(j) W[(j)]
  if constexpr (ROLL == 2) {
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
    for (int r = 0; r < 80; r += 16) {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (int t = 0; t < 16; ++t) {
        if (r) W[t] += s0<m0>(W[(t + 1) & 15]) + W[(t + 9) & 15] + s1<m1>(W[(t + 14) & 15]);
        const u64 t1 = h + S1<M1>(e) + ((e & f) ^ (~e & g)) + K[r + t] + W[t];
        const u64 t2 = S0<M0>(a) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
      }
    }
  } else {
    SX_EIGHT(0, 0, SX_RAW);
    SX_EIGHT(8, 8, SX_RAW);
    if constexpr (ROLL) {
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
      for (int t = 16; t < 80; t += 16) {
        SX_EIGHT(t, 0, SX_W);
        SX_EIGHT(t + 8, 8, SX_W);
      }
    } else {
      SX_EIGHT(16, 0, SX_W);
      SX_EIGHT(24, 8, SX_W);
      SX_EIGHT(32, 0, SX_W);
      SX_EIGHT(40, 8, SX_W);
      SX_EIGHT(48, 0, SX_W);
      SX_EIGHT(56, 8, SX_W);
      SX_EIGHT(64, 0, SX_W);
      SX_EIGHT(72, 8, SX_W);
    }
  }
#undef SX_RAW
#undef SX_EIGHT
#undef SX_W
#undef SX_ROUND
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// K[t] + W[t] for the chunk that ends a 256-byte message (0x80, zeros, the bit length 2048): every word is a constant, so
// the whole schedule is (hm96's h ends with this chunk)
#define SX_KW_PAD256_LIST                                                                                              \
  0xc28a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,                          \
      0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,                      \
      0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,                      \
      0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692e94ull,                      \
      0x649b69c19ef14ad2ull, 0xf0be4786384f6603ull, 0x11c1adc68b8cd5b9ull, 0x240ca1ec77ae9c65ull,                      \
      0x3df12c6f5b2b0295ull, 0x8a7485aaeebaec83ull, 0xdcb4cbddcd4a0514ull, 0x76f988da859153d5ull,                      \
      0x9c4761d2eea767ebull, 0xaa35d66d41be3364ull, 0xe153c8abb9126168ull, 0x1f638fc7e2ef2f28ull,                      \
      0x692570a455e2d00dull, 0x5b83d78905a3b687ull, 0x5f75a7a48ba84f02ull, 0x66cb4d902419302aull,                      \
      0x27e8ff16d9def4fcull, 0x31deadd7acf94f06ull, 0x4865f165bf7d7aaaull, 0x6c60f495411c8347ull,                      \
      0x0f85fc3af2085fbeull, 0x21660a2af9acf803ull, 0xc272cbb84d8a6a11ull, 0xd28c166752e28881ull,                      \
      0x9378463cace01b32ull, 0x62fb7bfb31387799ull, 0xe663fdc140b72139ull, 0x8ac6a6f028b7547eull,                      \
      0x319b8bee593abeeeull, 0x27f7ca2cdcfe9b1cull, 0x8cda485d8bca9aebull, 0xe9d09c09ee420668ull,                      \
      0xa1f8e3c50aafc84aull, 0x968146382170b10dull, 0x290b6fde6dc7c47aull, 0x85be4a15267185a2ull,                      \
      0xd9863caf8b055b5dull, 0x6cf1a4c45b9a9afaull, 0x5e4f2aaaf504ec15ull, 0xbaa4b690a16e887dull,                      \
      0xd6deefafa6e9b80eull, 0xdaaa2b21fbdcdfe1ull, 0x7c27540e4c494849ull, 0x7eedd22ad2df02e3ull,                      \
      0xddee79d9e75f1cc4ull, 0xd111d2bce9610b19ull, 0x9fb24be9b8d7371cull, 0x78f7fd060071ddb4ull,                      \
      0x4cb381c27994ef77ull, 0x0ac1b70a8ce453a9ull, 0x790d6d8995529634ull, 0xfa975f11f18abfbdull,                      \
      0xbb17b998273ce801ull, 0x01a0bacb85aa2c16ull, 0xc7291aa80f5eae0eull, 0x9c4ffed7fc1252ecull,                      \
      0x0ea1f738a106aa51ull, 0x1647c833b33903b5ull, 0xb05a0c5531353616ull, 0xc72da5195d3a71fcull,                      \
      0xaa484e02078b8d74ull, 0xa0af66f3caa8eeaeull, 0x5729dd1286eb91f0ull, 0x76e07546e6f85510ull

__constant__ u64 KW_PAD256_DEV[80] = {SX_KW_PAD256_LIST};
static const u64 KW_PAD256_HOST[80] = {SX_KW_PAD256_LIST};
#undef SX_KW_PAD256_LIST

SX_HD const u64* kw_pad256() {
#ifdef __CUDA_ARCH__
  return KW_PAD256_DEV;
#else
  return KW_PAD256_HOST;
#endif
}

// compress's rounds alone, over a chunk whose K[t] + W[t] are given (kw_pad256): the chaining value compress gives over that
// chunk, without its schedule
template <int V, int ROLL>
SX_HD void compress_kw(u64* st, const u64* KW) {
  constexpr int M0 = V & 3, M1 = (V >> 2) & 3;
  u64 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#define SX_ROUND(a, b, c, d, e, f, g, h, kw)                                                                               \
  do {                                                                                                                     \
    const u64 t1_ = (h) + S1<M1>(e) + ch(e, f, g) + (kw);                                                                  \
    (d) += t1_;                                                                                                            \
    (h) = t1_ + S0<M0>(a) + maj(a, b, c);                                                                                  \
  } while (0)
#define SX_EIGHT_KW(t)                                                                                                     \
  SX_ROUND(a, b, c, d, e, f, g, h, KW[(t) + 0]);                                                                           \
  SX_ROUND(h, a, b, c, d, e, f, g, KW[(t) + 1]);                                                                           \
  SX_ROUND(g, h, a, b, c, d, e, f, KW[(t) + 2]);                                                                           \
  SX_ROUND(f, g, h, a, b, c, d, e, KW[(t) + 3]);                                                                           \
  SX_ROUND(e, f, g, h, a, b, c, d, KW[(t) + 4]);                                                                           \
  SX_ROUND(d, e, f, g, h, a, b, c, KW[(t) + 5]);                                                                           \
  SX_ROUND(c, d, e, f, g, h, a, b, KW[(t) + 6]);                                                                           \
  SX_ROUND(b, c, d, e, f, g, h, a, KW[(t) + 7])
  if constexpr (ROLL) {
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
    for (int t = 0; t < 80; t += 16) {
      SX_EIGHT_KW(t);
      SX_EIGHT_KW(t + 8);
    }
  } else {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int t = 0; t < 80; t += 8) {
      SX_EIGHT_KW(t);
    }
  }
#undef SX_EIGHT_KW
#undef SX_ROUND
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// compress_kw's K[t] + W[t] for the block w (w is not changed): for a block that every slot hashes alike, computed once
SX_HD void kw_of(const u64* w, u64* kw) {
  const u64* K = kconst();
  u64 W[16];
  for (int t = 0; t < 16; ++t) W[t] = w[t];
  for (int t = 0; t < 80; ++t) {
    if (t >= 16) W[t & 15] += s1<0>(W[(t + 14) & 15]) + W[(t + 9) & 15] + s0<0>(W[(t + 1) & 15]);
    kw[t] = K[t] + W[t & 15];
  }
}

}  // namespace sx
