// PoUW's device hashing on sm_120, one header for the FP8 and FP4 streams: TurboSHAKE128 (RFC 9861, the binding), SHA-256
// (FIPS 180-4, the comparison) and BLAKE3 (Pearl-C's seeds, noise lines and keyed tickets), with the pieces a GEMM
// epilogue calls to hash FP32 accumulator words where they sit. Built from #449's `pearl_c.cu` (TurboSHAKE128, the keyed BLAKE3
// block), #468's `ncp2_tiles.cuh` (the SHA-256 compression) and the H100 `hash90` (Keccak's LOP3 form).
//
// Every function compiles for the host (g++ -x c++, no CUDA) and the device (nvcc, sm_120a). `pouw_hash_bench.cu check` prints
// the same vectors from both, and `check.py` holds them to hashlib, the `blake3` package, verity.ml.crypto.turboshake and
// Pearl-C's own `leaf` (benchmarks/pouw/tests/test_pouw_hash.py on the host; `run.sh` on the device before any timing).
//
// Words and lanes. A lane is 8 message bytes, little-endian: lane_u32x2(lo, hi) puts the word with the lower column (the
// first 4 bytes) in the low half, as Pearl-C's le32 C~ and U words sit in its messages. A SHA-256 word is big-endian.
//
// State footprints, in 32-bit registers per thread (the rates and ptxas's counts are in pouw_hash_bench.cu's rows):
//   TurboSHAKE128 / Keccak   50 (25 lanes), about 70 with theta's temporaries; no shared or local memory
//   SHA-256                  8 chaining + 16 schedule words, about 40 in a compression
//   BLAKE3                   16 working + 16 message words, about 40 in a compression
//   quad_msg_digest<A>       the sponge (50) and the fragments not yet sent (32 A floats at the start, 8 fewer per group),
//                            plus 1 KB of shared memory per warp (the transpose's scratch, reused by every group)
//   Ts128Stream              208 bytes wherever it lives; keep it in shared memory (a runtime lane position into a register
//                            array would go to local memory)
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifndef __CUDACC__
#include <string.h>
#endif

#ifdef __CUDACC__
#define PH_HD __host__ __device__ __forceinline__
#else
#define PH_HD inline
#endif

namespace ph {

// ---- words and lanes ------------------------------------------------------------------------------------------------------

PH_HD uint64_t lane_u32x2(uint32_t lo, uint32_t hi) { return (uint64_t)hi << 32 | lo; }
PH_HD uint32_t f32_bits(float x) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(x);
#else
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
#endif
}
PH_HD uint64_t lane_f32x2(float lo, float hi) { return lane_u32x2(f32_bits(lo), f32_bits(hi)); }
PH_HD uint32_t bswap32(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, 0, 0x0123);
#else
  return x >> 24 | (x >> 8 & 0xFF00u) | (x << 8 & 0xFF0000u) | x << 24;
#endif
}
PH_HD uint64_t le64(const uint8_t* p) {
  uint64_t x = 0;
  for (int i = 7; i >= 0; --i) x = x << 8 | p[i];
  return x;
}

// ---- Keccak-p[1600, n_r] ----------------------------------------------------------------------------------------------------

PH_HD uint64_t rol64(uint64_t x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }
// The same as two funnel shifts (SHF.L.W), which ptxas doesn't choose on its own for every rotation: on sm_120a it lowers some
// plain rotations to IMAD.SHL + LOP3 + SHF (Keccak-p12: 2,976 instructions, 324 of them IMAD.SHL, against about 2,200 as SHF).
PH_HD uint64_t rol64_shf(uint64_t x, int n) {
#ifdef __CUDA_ARCH__
  const uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
  if (n == 0) return x;
  if (n == 32) return lane_u32x2(hi, lo);
  if (n < 32) return lane_u32x2(__funnelshift_l(hi, lo, n), __funnelshift_l(lo, hi, n));
  return lane_u32x2(__funnelshift_l(lo, hi, n - 32), __funnelshift_l(hi, lo, n - 32));
#else
  return rol64(x, n);
#endif
}
// one LOP3 per 32-bit half; the host evaluates the same truth table (bit a·4 + b·2 + c of LUT)
template <int LUT>
PH_HD uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
#ifdef __CUDA_ARCH__
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(LUT));
  return r;
#else
  uint32_t r = 0;
  for (int k = 0; k < 8; ++k)
    if (LUT >> k & 1) r |= (k & 4 ? a : ~a) & (k & 2 ? b : ~b) & (k & 1 ? c : ~c);
  return r;
#endif
}
template <int LUT>
PH_HD uint64_t lop3_64(uint64_t a, uint64_t b, uint64_t c) {
  return (uint64_t)lop3<LUT>((uint32_t)(a >> 32), (uint32_t)(b >> 32), (uint32_t)(c >> 32)) << 32 |
         lop3<LUT>((uint32_t)a, (uint32_t)b, (uint32_t)c);
}
PH_HD uint64_t xor3(uint64_t a, uint64_t b, uint64_t c) { return lop3_64<0x96>(a, b, c); }
PH_HD uint64_t chi(uint64_t a, uint64_t b, uint64_t c) { return lop3_64<0xD2>(a, b, c); }  // a ^ (~b & c)

// Rounds 24 − ROUNDS … 23 of Keccak-f[1600] on 25 lanes (index x + 5y): ROUNDS = 12 is TurboSHAKE128's permutation, 24
// Keccak-f (SHAKE). Theta's parity, theta's apply and chi are three-input LOP3s (hash90: 4,309 instructions per Keccak-f on
// sm_90a against 4,651 for ptxas's own fusion of the plain expressions). SHF: rotations as funnel shifts (rol64_shf), the
// default (PH_KECCAK_SHF): on sm_120a with CUDA 12.9, 2,320 instructions per Keccak-p12 (1,509 LOP3, 696 SHF) against 2,976
// (1,833 LOP3, 696 SHF, 329 IMAD) as ptxas lowers the plain rotations; the rates' p12s / p12i rows time both. PH_KECCAK_INLINE: define it (before this header)
// as __host__ __device__ __noinline__ where code size matters more than speed (the check kernels).
#ifndef PH_KECCAK_SHF
#define PH_KECCAK_SHF true
#endif
#ifndef PH_KECCAK_INLINE
#define PH_KECCAK_INLINE PH_HD
#endif
template <int ROUNDS, bool SHF = PH_KECCAK_SHF>
PH_KECCAK_INLINE void keccak_p(uint64_t s[25]) {
  const uint64_t RC[24] = {0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
                           0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
                           0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
                           0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
                           0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
                           0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};
  const int ROT[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int PI[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll
  for (int r = 24 - ROUNDS; r < 24; ++r) {
    uint64_t bc[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) bc[i] = xor3(xor3(s[i], s[i + 5], s[i + 10]), s[i + 15], s[i + 20]);
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint64_t p = bc[(i + 4) % 5], q = SHF ? rol64_shf(bc[(i + 1) % 5], 1) : rol64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) s[j + i] = xor3(s[j + i], p, q);
    }
    uint64_t t = s[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      const int j = PI[i];
      const uint64_t x = s[j];
      s[j] = SHF ? rol64_shf(t, ROT[i]) : rol64(t, ROT[i]);
      t = x;
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
      const uint64_t b0 = s[j], b1 = s[j + 1], b2 = s[j + 2], b3 = s[j + 3], b4 = s[j + 4];
      s[j] = chi(b0, b1, b2);
      s[j + 1] = chi(b1, b2, b3);
      s[j + 2] = chi(b2, b3, b4);
      s[j + 3] = chi(b3, b4, b0);
      s[j + 4] = chi(b4, b0, b1);
    }
    s[0] ^= RC[r];
  }
}

// ---- TurboSHAKE128: Keccak-p[1600, 12] at a rate of 21 lanes (168 bytes), 32-byte digests --------------------------------

constexpr int TS_RATE = 21;

// One rate block: s ^= lane(0 … 20), then the permutation. `lane` is called with compile-time indices once unrolled, so it may
// read registers.
template <class Lane>
PH_HD void ts128_block(uint64_t s[25], const Lane& lane) {
#pragma unroll
  for (int w = 0; w < TS_RATE; ++w) s[w] ^= lane(w);
  keccak_p<12>(s);
}
// The last block: lane(0 … R − 1), the domain byte D (0x01 … 0x7F) after them, 0x80 in the block's last byte, the
// permutation; the digest is lanes 0 … 3.
template <int R, class Lane>
PH_HD void ts128_final(uint64_t s[25], uint8_t D, const Lane& lane, uint64_t out[4]) {
  static_assert(R >= 0 && R < TS_RATE, "a last block holds 0 … 20 lanes");
#pragma unroll
  for (int w = 0; w < R; ++w) s[w] ^= lane(w);
  s[R] ^= D;
  s[TS_RATE - 1] ^= 0x8000000000000000ull;
  keccak_p<12>(s);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = s[i];
}
// TurboSHAKE128(M, D, 32) of NL lanes, lane(i) = M's bytes 8i … 8i + 7. Up to 3 blocks are unrolled (lane may read registers);
// longer messages loop over blocks (lane should read memory).
template <int NL, class Lane>
PH_HD void ts128(uint8_t D, const Lane& lane, uint64_t out[4]) {
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
  constexpr int FULL = NL / TS_RATE;
  if constexpr (FULL <= 3) {
#pragma unroll
    for (int b = 0; b < FULL; ++b) ts128_block(s, [&](int w) { return lane(TS_RATE * b + w); });
  } else {
#pragma unroll 1
    for (int b = 0; b < FULL; ++b) ts128_block(s, [&](int w) { return lane(TS_RATE * b + w); });
  }
  ts128_final<NL % TS_RATE>(s, D, [&](int w) { return lane(TS_RATE * FULL + w); }, out);
}

// A sponge with a runtime lane position, for a digest that several threads or passes feed in turn (a tile's leaf, band by
// band). Keep it in shared memory: each permutation loads it into registers and stores it back.
struct Ts128Stream {
  uint64_t s[25];
  uint32_t pos;
};
PH_HD void tss_init(Ts128Stream& t) {
  for (int i = 0; i < 25; ++i) t.s[i] = 0;
  t.pos = 0;
}
PH_HD void tss_permute(Ts128Stream& t) {
  uint64_t r[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) r[i] = t.s[i];
  keccak_p<12>(r);
#pragma unroll
  for (int i = 0; i < 25; ++i) t.s[i] = r[i];
}
PH_HD void tss_lane(Ts128Stream& t, uint64_t x) {
  t.s[t.pos] ^= x;
  if (++t.pos == TS_RATE) {
    tss_permute(t);
    t.pos = 0;
  }
}
PH_HD void tss_final(Ts128Stream& t, uint8_t D, uint64_t out[4]) {
  t.s[t.pos] ^= D;
  t.s[TS_RATE - 1] ^= 0x8000000000000000ull;
  tss_permute(t);
  for (int i = 0; i < 4; ++i) out[i] = t.s[i];
}

// Any Keccak sponge on bytes (the vectors' reference form): rate RATE bytes, ROUNDS rounds, the domain byte ds, outlen bytes out.
// (24, 168, 0x1F) is SHAKE128, (24, 136, 0x1F) SHAKE256, (12, 168, D) TurboSHAKE128.
template <int ROUNDS, int RATE>
PH_HD void sponge_bytes(const uint8_t* m, size_t len, uint8_t ds, uint8_t* out, size_t outlen) {
  uint64_t s[25];
  for (int i = 0; i < 25; ++i) s[i] = 0;
  size_t off = 0;
  for (; len - off >= RATE; off += RATE) {
    for (int w = 0; w < RATE / 8; ++w) s[w] ^= le64(m + off + 8 * w);
    keccak_p<ROUNDS>(s);
  }
  uint8_t blk[RATE];
  for (int i = 0; i < RATE; ++i) blk[i] = i < (int)(len - off) ? m[off + i] : 0;
  blk[len - off] ^= ds;
  blk[RATE - 1] ^= 0x80;
  for (int w = 0; w < RATE / 8; ++w) s[w] ^= le64(blk + 8 * w);
  keccak_p<ROUNDS>(s);
  for (size_t o = 0; o < outlen;) {
    for (int w = 0; w < RATE / 8 && o < outlen; ++w)
      for (int b = 0; b < 8 && o < outlen; ++b) out[o++] = (uint8_t)(s[w] >> (8 * b));
    if (o < outlen) keccak_p<ROUNDS>(s);
  }
}

// ---- SHA-256 ----------------------------------------------------------------------------------------------------------------

PH_HD uint32_t rotr32(uint32_t x, int n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << (32 - n));
#endif
}
PH_HD void sha256_iv(uint32_t h[8]) {
  const uint32_t IV[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = IV[i];
}
// One compression of 16 big-endian words (#468's `tile_compress`): h + K + W summed a round ahead, Ch and Maj as one
// three-input operation each, each message word scheduled a round before its use, the working variables renamed, not moved.
PH_HD void sha256_compress(uint32_t h[8], const uint32_t blk[16]) {
  const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[16], s[8];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = blk[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = h[i];
  uint32_t hkw = s[7] + K[0] + w[0];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const uint32_t a = s[(0 - i) & 7], b = s[(1 - i) & 7], c = s[(2 - i) & 7];
    const uint32_t e = s[(4 - i) & 7], f = s[(5 - i) & 7], g = s[(6 - i) & 7];
    if (i + 1 >= 16 && i + 1 < 64) {
      const int j = i + 1;
      const uint32_t w15 = w[(j + 1) & 15], w2 = w[(j + 14) & 15];
      w[j & 15] += (rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3)) + w[(j + 9) & 15] +
                   (rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10));
    }
    const uint32_t t1 = (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + (g ^ (e & (f ^ g))) + hkw;
    if (i + 1 < 64) hkw = g + K[i + 1] + w[(i + 1) & 15];
    s[(3 - i) & 7] += t1;
    s[(7 - i) & 7] = t1 + (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) | (c & (a | b)));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] += s[i];
}
// SHA-256 of NW big-endian words word(0 … NW − 1), padding fixed at compile time; the digest is h, big-endian.
template <int NW, class Word>
PH_HD void sha256(const Word& word, uint32_t h[8]) {
  sha256_iv(h);
  constexpr int FULL = NW / 16, REM = NW % 16;
  constexpr uint64_t BITS = (uint64_t)NW * 32;
#pragma unroll 1
  for (int b = 0; b < FULL; ++b) {
    uint32_t blk[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) blk[i] = word(16 * b + i);
    sha256_compress(h, blk);
  }
  uint32_t blk[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) blk[i] = i < REM ? word(16 * FULL + i) : i == REM ? 0x80000000u : 0;
  if constexpr (REM > 13) {
    sha256_compress(h, blk);
#pragma unroll
    for (int i = 0; i < 16; ++i) blk[i] = 0;
  }
  blk[14] = (uint32_t)(BITS >> 32);
  blk[15] = (uint32_t)BITS;
  sha256_compress(h, blk);
}
PH_HD void sha256_bytes(const uint8_t* m, size_t len, uint8_t out[32]) {
  uint32_t h[8];
  sha256_iv(h);
  uint8_t blk[128];
  for (size_t off = 0;; off += 64) {
    const size_t left = len - off;
    int nb = 1;
    if (left >= 64) {
      for (int i = 0; i < 64; ++i) blk[i] = m[off + i];
    } else {
      for (int i = 0; i < 128; ++i) blk[i] = i < (int)left ? m[off + i] : 0;
      blk[left] = 0x80;
      nb = left < 56 ? 1 : 2;
      for (int i = 0; i < 8; ++i) blk[nb * 64 - 1 - i] = (uint8_t)((uint64_t)len * 8 >> (8 * i));
    }
    for (int q = 0; q < nb; ++q) {
      uint32_t w[16];
      for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t)blk[64 * q + 4 * i] << 24 | (uint32_t)blk[64 * q + 4 * i + 1] << 16 | (uint32_t)blk[64 * q + 4 * i + 2] << 8 |
               blk[64 * q + 4 * i + 3];
      sha256_compress(h, w);
    }
    if (left < 64) break;
  }
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(h[i >> 2] >> (24 - 8 * (i & 3)));
}

// ---- BLAKE3 -----------------------------------------------------------------------------------------------------------------

constexpr uint32_t B3_CHUNK_START = 1, B3_CHUNK_END = 2, B3_ROOT = 8, B3_KEYED_HASH = 16;
PH_HD void b3_g(uint32_t v[16], int a, int b, int c, int d, uint32_t x, uint32_t y) {
  v[a] = v[a] + v[b] + x;
  v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + y;
  v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 7);
}
// One compression: chaining value cv, 16 little-endian message words, the block counter, the block's length in bytes and the
// flags; out = the first 8 words of the output (the next chaining value, or the digest on a ROOT block).
PH_HD void blake3_compress(const uint32_t cv[8], const uint32_t msg[16], uint64_t counter, uint32_t len, uint32_t flags,
                           uint32_t out[8]) {
  const uint32_t IV[4] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A};
  const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t v[16], m[16], t[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = cv[i];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[8 + i] = IV[i];
  v[12] = (uint32_t)counter;
  v[13] = (uint32_t)(counter >> 32);
  v[14] = len;
  v[15] = flags;
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    b3_g(v, 0, 4, 8, 12, m[0], m[1]);
    b3_g(v, 1, 5, 9, 13, m[2], m[3]);
    b3_g(v, 2, 6, 10, 14, m[4], m[5]);
    b3_g(v, 3, 7, 11, 15, m[6], m[7]);
    b3_g(v, 0, 5, 10, 15, m[8], m[9]);
    b3_g(v, 1, 6, 11, 12, m[10], m[11]);
    b3_g(v, 2, 7, 8, 13, m[12], m[13]);
    b3_g(v, 3, 4, 9, 14, m[14], m[15]);
#pragma unroll
    for (int i = 0; i < 16; ++i) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}
// The keyed hash of one 64-byte message (a single-block chunk: CHUNK_START | CHUNK_END | ROOT | KEYED_HASH): Pearl's noise
// lines and tickets. key: 8 little-endian words.
PH_HD void blake3_keyed_block(const uint32_t key[8], const uint32_t msg[16], uint32_t out[8]) {
  blake3_compress(key, msg, 0, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED_HASH, out);
}
// BLAKE3 of a message of at most one chunk (1,024 bytes): keyed when key is non-null (Pearl-C's seeds, `keyed(key, m)`),
// plain otherwise; 32 bytes out.
PH_HD void blake3_chunk(const uint32_t* key, const uint8_t* m, size_t len, uint8_t out[32]) {
  const uint32_t IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
  uint32_t cv[8];
  for (int i = 0; i < 8; ++i) cv[i] = key ? key[i] : IV[i];
  const uint32_t mode = key ? B3_KEYED_HASH : 0;
  const size_t blocks = len ? (len + 63) / 64 : 1;
  for (size_t b = 0; b < blocks; ++b) {
    uint32_t w[16];
    const size_t off = 64 * b, n = len - off < 64 ? len - off : 64;
    for (int i = 0; i < 16; ++i) {
      w[i] = 0;
      for (int j = 0; j < 4; ++j)
        if ((size_t)(4 * i + j) < n) w[i] |= (uint32_t)m[off + 4 * i + j] << (8 * j);
    }
    const uint32_t flags = mode | (b == 0 ? B3_CHUNK_START : 0) | (b + 1 == blocks ? B3_CHUNK_END | B3_ROOT : 0);
    blake3_compress(cv, w, 0, (uint32_t)n, flags, cv);
  }
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(cv[i >> 2] >> (8 * (i & 3)));
}

// BLAKE3's tree over n = 2^j whole chunks (the chunks' chaining values, then parent nodes pairwise in index order, ROOT on the
// top one; n = 1: the chunk itself is the root). A chunk is `blocks` 64-byte blocks: 16 is BLAKE3 itself (1,024-byte chunks);
// 1 is the fine tree of 64-byte leaves (a candidate form for decode, not BLAKE3's hash of the bytes). key: blake3_chunk's
// (IV unkeyed, mode 0; the key with B3_KEYED_HASH). word(i): the i-th little-endian word of the chunk.
constexpr uint32_t B3_PARENT = 4;
PH_HD void b3_iv(uint32_t cv[8]) {
  const uint32_t IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
  for (int i = 0; i < 8; ++i) cv[i] = IV[i];
}
template <class Word>
PH_HD void blake3_chunk_cv(const uint32_t key[8], uint32_t mode, uint64_t counter, int blocks, uint32_t root, const Word& word,
                           uint32_t cv[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key[i];
#pragma unroll 1
  for (int b = 0; b < blocks; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = word(16 * b + i);
    const uint32_t flags = mode | (b == 0 ? B3_CHUNK_START : 0) | (b + 1 == blocks ? B3_CHUNK_END | root : 0);
    blake3_compress(cv, m, counter, 64, flags, cv);
  }
}
PH_HD void blake3_parent(const uint32_t key[8], uint32_t mode, const uint32_t l[8], const uint32_t r[8], uint32_t root,
                         uint32_t out[8]) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = l[i], m[8 + i] = r[i];
  blake3_compress(key, m, 0, 64, mode | B3_PARENT | root, out);
}
// One thread, n <= MAXN chunks; word(i) is the whole input's i-th word.
template <int MAXN, class Word>
PH_HD void blake3_tree(const uint32_t key[8], uint32_t mode, int blocks, int n, const Word& word, uint32_t out[8]) {
  uint32_t cv[MAXN][8];
  const int per = 16 * blocks;
#pragma unroll 1
  for (int c = 0; c < n; ++c) blake3_chunk_cv(key, mode, c, blocks, n == 1 ? B3_ROOT : 0, [&](int i) { return word(per * c + i); }, cv[c]);
#pragma unroll 1
  for (int w = n; w > 1; w /= 2)
#pragma unroll 1
    for (int i = 0; i < w / 2; ++i) blake3_parent(key, mode, cv[2 * i], cv[2 * i + 1], w == 2 ? B3_ROOT : 0, cv[i]);
  for (int i = 0; i < 8; ++i) out[i] = cv[0][i];
}

#ifdef __CUDACC__
// The same tree on a warp: 32 / N trees of N = 2^j chunks, lane l taking chunk l % N of tree l / N, the tree's lanes reducing
// through shuffles (depth: blocks compressions, then log2 N parents). Every lane calls it (converged); word(c, i): chunk c's
// i-th word of this lane's tree. Lane l with l % N == 0 gets its tree's root.
template <int N, class Word>
__device__ __forceinline__ void warp_blake3_tree(const uint32_t key[8], uint32_t mode, int blocks, const Word& word, uint32_t out[8]) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "a power of two within a warp");
  const int c = (threadIdx.x & 31) % N;
  blake3_chunk_cv(key, mode, c, blocks, N == 1 ? B3_ROOT : 0, [&](int i) { return word(c, i); }, out);
#pragma unroll
  for (int s = 1; s < N; s *= 2) {
    uint32_t r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = __shfl_down_sync(0xffffffffu, out[i], s);
    if (c % (2 * s) == 0) blake3_parent(key, mode, out, r, 2 * s == N ? B3_ROOT : 0, out);
  }
}
// The fine tree (64-byte leaves, blocks = 1) of N = 2^j leaves on a block of N threads (N a multiple of 32, at most 1,024):
// thread t compresses leaf t, each warp reduces its 32, and warp 0 the warps' roots; sh: N / 32 x 8 words of shared memory.
// Every thread calls it; thread 0 gets the root. Depth 1 + log2 N compressions.
template <int N, class Word>
__device__ __forceinline__ void block_blake3_fine(const uint32_t key[8], uint32_t mode, const Word& word, uint32_t* sh, uint32_t out[8]) {
  static_assert(N >= 64 && N <= 1024 && (N & (N - 1)) == 0, "two warps or more, a power of two");
  const int t = threadIdx.x, lane = t & 31;
  blake3_chunk_cv(key, mode, t, 1, 0, [&](int i) { return word(t, i); }, out);
#pragma unroll
  for (int s = 1; s < 32; s *= 2) {
    uint32_t r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = __shfl_down_sync(0xffffffffu, out[i], s);
    if (lane % (2 * s) == 0) blake3_parent(key, mode, out, r, 0, out);
  }
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < 8; ++i) sh[8 * (t >> 5) + i] = out[i];
  __syncthreads();
  if (t < 32) {
    constexpr int W = N / 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = sh[8 * (lane % W) + i];
#pragma unroll
    for (int s = 1; s < W; s *= 2) {
      uint32_t r[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) r[i] = __shfl_down_sync(0xffffffffu, out[i], s);
      if (lane % (2 * s) == 0) blake3_parent(key, mode, out, r, 2 * s == W ? B3_ROOT : 0, out);
    }
  }
  __syncthreads();
}
#endif

// ---- in a GEMM epilogue: mma.sync's FP32 accumulators, hashed where they sit -------------------------------------------------
//
// The m16n8 accumulator fragment of every mma.sync m16n8kK with FP32 accumulation (FP8 k32, FP4 k64, BF16 k16): lane
// 4g + t holds c[0], c[1] at row g, columns 2t, 2t + 1 and c[2], c[3] at row g + 8, the same columns. A region is m16 x n64
// (eight n8 blocks side by side), and its messages are its 16 rows x two 32-column chunks: 32 messages, one per lane. Each
// message is A arrays' 32 words of its row and chunk, array by array (Pearl-C: A = 2, C~ then U, D = 0x21). The quad of lanes
// 4g … 4g + 3 holds the 4 messages of rows g and g + 8, 8 lanes of each; lane 4g + t hashes message (row g + 8 (t & 1),
// chunk t >> 1). The quad swaps its lanes through 1 KB of shared memory per warp, 4 message lanes per group, so every register
// index is fixed at compile time and neither side needs a select. frag(a, j, i) returns array a's fragment c[i] of n8 block j.

PH_HD int quad_row(int lane) { return (lane >> 2) + 8 * (lane & 1); }
PH_HD int quad_chunk(int lane) { return (lane >> 1) & 1; }
// scratch slot of message m (the owner's t), lane e of the group, quad: conflict-free for both halves (writes: fixed m, e = t;
// reads: m = t, fixed e), 128 slots.
PH_HD int quad_slot(int m, int e, int quad) { return m * 32 + (quad >> 2) * 16 + ((e ^ m) << 2) + (quad & 3); }
// A lane's writes for group q (message lanes 4q … 4q + 3): its lane of each of the quad's 4 messages.
template <class Frag>
PH_HD void quad_put(const Frag& frag, int q, int quad, int t, uint64_t* scratch) {
  const int a = q >> 2, jj = q & 3;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int j = 4 * (m >> 1) + jj, i = 2 * (m & 1);
    scratch[quad_slot(m, t, quad)] = lane_f32x2(frag(a, j, i), frag(a, j, i + 1));
  }
}
// A lane's reads for group q: message lanes 4q … 4q + 3 of its own message, absorbed into s (a permutation after each 21st).
PH_HD void quad_get(int q, int quad, int t, const uint64_t* scratch, uint64_t s[25]) {
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int L = 4 * q + e;
    s[L % TS_RATE] ^= scratch[quad_slot(t, e, quad)];
    if (L % TS_RATE == TS_RATE - 1) keccak_p<12>(s);
  }
}
template <int A>
PH_HD void quad_final(uint64_t s[25], uint8_t D, uint64_t out[4]) {
  constexpr int NL = 16 * A;
  s[NL % TS_RATE] ^= D;
  s[TS_RATE - 1] ^= 0x8000000000000000ull;
  keccak_p<12>(s);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = s[i];
}

#ifdef __CUDACC__
// Every lane of the warp calls it (converged); scratch: 1 KB of shared memory for this warp, 8-byte aligned. Lane 4g + t gets
// the digest of message (quad_row(lane), quad_chunk(lane)).
template <int A, class Frag>
__device__ __forceinline__ void quad_msg_digest(const Frag& frag, uint8_t D, uint64_t* scratch, uint64_t out[4]) {
  const int lane = threadIdx.x & 31, quad = lane >> 2, t = lane & 3;
  uint64_t s[25];
#pragma unroll
  for (int i = 0; i < 25; ++i) s[i] = 0;
#pragma unroll
  for (int q = 0; q < 4 * A; ++q) {
    quad_put(frag, q, quad, t, scratch);
    __syncwarp();
    quad_get(q, quad, t, scratch, s);
    __syncwarp();
  }
  quad_final<A>(s, D, out);
}
// The same messages under keyed BLAKE3 (a candidate for hashing format -h1, P5: one chunk of 4·A blocks of 64 bytes, key
// B3_KEYED_HASH, the message bytes TurboSHAKE128's): two groups of the transpose are one block, compressed as it completes.
template <int A, class Frag>
__device__ __forceinline__ void quad_msg_digest_b3(const Frag& frag, const uint32_t key[8], uint64_t* scratch, uint32_t out[8]) {
  const int lane = threadIdx.x & 31, quad = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = key[i];
  uint32_t w[16];
#pragma unroll
  for (int q = 0; q < 4 * A; ++q) {
    quad_put(frag, q, quad, t, scratch);
    __syncwarp();
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint64_t x = scratch[quad_slot(t, e, quad)];
      w[8 * (q & 1) + 2 * e] = (uint32_t)x, w[8 * (q & 1) + 2 * e + 1] = (uint32_t)(x >> 32);
    }
    __syncwarp();
    if (q & 1) {
      const int b = q >> 1;
      blake3_compress(out, w, 0, 64, B3_KEYED_HASH | (b == 0 ? B3_CHUNK_START : 0) | (b == 2 * A - 1 ? B3_CHUNK_END | B3_ROOT : 0), out);
    }
  }
}
#endif

// The same, one warp emulated on the host: frag(lane, a, j, i); out[lane] is lane's digest.
template <int A, class Frag>
inline void quad_msg_digest_warp(const Frag& frag, uint8_t D, uint64_t out[32][4]) {
  uint64_t scratch[128], s[32][25] = {};
  for (int q = 0; q < 4 * A; ++q) {
    for (int lane = 0; lane < 32; ++lane)
      quad_put([&](int a, int j, int i) { return frag(lane, a, j, i); }, q, lane >> 2, lane & 3, scratch);
    for (int lane = 0; lane < 32; ++lane) quad_get(q, lane >> 2, lane & 3, scratch, s[lane]);
  }
  for (int lane = 0; lane < 32; ++lane) quad_final<A>(s[lane], D, out[lane]);
}

// ---- Pearl-C (`pearl-c-h100-v0/v1`, verity.core.protocols.pouw.schemes.pearl_c): which hash binds which words -----------------------------

namespace pearlc {
constexpr uint8_t D_MESSAGE = 0x21, D_LEAF = 0x22, D_ROWS = 0x23;
constexpr int TILE = 64, CHUNK = 32, MSG_LANES = 32, DIGESTS = TILE * TILE / CHUNK, LEAF_LANES = 4 * DIGESTS;

// d_rh = TurboSHAKE128(le32 C~ of the chunk ‖ le32 U of the chunk, 0x21): lane(0 … 15) C~'s word pairs, lane(16 … 31) U's.
template <class Lane>
PH_HD void msg_digest(const Lane& lane, uint64_t d[4]) {
  ts128<MSG_LANES>(D_MESSAGE, lane, d);
}
// The tile's leaf: TurboSHAKE128 of its 128 message digests, rows in order and 2 chunks each (digest 2r + h), 0x22; lane(i) is
// lane i % 4 of digest i / 4. 25 permutations, one serial chain.
template <class Lane>
PH_HD void tile_leaf(const Lane& lane, uint64_t leaf[4]) {
  ts128<LEAF_LANES>(D_LEAF, lane, leaf);
}
// Where a region's digests go: region ρ (tile rows 16ρ … 16ρ + 15) of a 64 x 64 tile, lane's message.
PH_HD int digest_index(int region, int lane) { return 2 * (16 * region + quad_row(lane)) + quad_chunk(lane); }
// A warp tile one chunk wide (16·MI rows x 32 columns, MI even; GPU 1's sm_120 GEMM holds 64 x 32) pairs its m16 blocks into
// regions: region p's chunk h is m16 block 2p + h, so frag(a, j, i) is array a's c[i] of m16 block 2p + (j >> 2), n8 block
// j & 3, and lane's digest is of the warp's row narrow_row(p, lane) and its one chunk.
PH_HD int narrow_row(int p, int lane) { return 16 * (2 * p + quad_chunk(lane)) + quad_row(lane); }
// A's commitment, modeled: TurboSHAKE128 of one row segment of NL lanes (FP32 rows: 128, FP10 rows: 160), 0x23.
template <int NL, class Lane>
PH_HD void row_segment(const Lane& lane, uint64_t d[4]) {
  ts128<NL>(D_ROWS, lane, d);
}
// Pearl's ticket of a 16 x 16 subtile of C~: lane (rb, cb) folds its 4 x 4 block row-major, lane <- rotl32(lane·0x9E3779B1 +
// bits, 13), and the 16 lanes are one keyed BLAKE3 block under key = subkey("jackpot", seed_A). word(r, c): C~'s bits.
template <class Word>
PH_HD void ticket(const uint32_t key[8], const Word& word, uint32_t out[8]) {
  uint32_t z[16];
#pragma unroll
  for (int l = 0; l < 16; ++l) {
    uint32_t v = 0;
#pragma unroll
    for (int f = 0; f < 16; ++f) {
      v = v * 0x9E3779B1u + word(4 * (l >> 2) + (f >> 2), 4 * (l & 3) + (f & 3));
      v = rotr32(v, 19);
    }
    z[l] = v;
  }
  blake3_keyed_block(key, z, out);
}
}  // namespace pearlc

}  // namespace ph
