// See receipt.h.
#include "receipt.h"

#include <immintrin.h>

#include <cstring>

static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

static uint64_t seed64() {
  uint64_t s;
  memcpy(&s, SEED, 8);
  return s;
}

// ---- SHA-256 (FIPS 180-4), for the CPU leaves ----
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void sha256_block(uint32_t h[8], const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
    uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
}
void sha256(const uint8_t* m, size_t n, uint8_t out[32]) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t i = 0;
  for (; i + 64 <= n; i += 64) sha256_block(h, m + i);
  uint8_t t[128] = {0};
  size_t r = n - i;
  memcpy(t, m + i, r);
  t[r] = 0x80;
  size_t len = r + 9 <= 64 ? 64 : 128;
  uint64_t bits = (uint64_t)n * 8;
  for (int k = 0; k < 8; ++k) t[len - 1 - k] = (uint8_t)(bits >> (8 * k));
  sha256_block(h, t);
  if (len == 128) sha256_block(h, t + 64);
  for (int k = 0; k < 8; ++k)
    out[4 * k] = h[k] >> 24, out[4 * k + 1] = h[k] >> 16, out[4 * k + 2] = h[k] >> 8, out[4 * k + 3] = h[k];
}

void cpu_keys(int8_t* K, int ks) {  // K[l*256 + j]
  for (int l = 0; l < ks; ++l)
    for (int j = 0; j < 256; ++j)
      K[l * 256 + j] = (int8_t)(splitmix64(seed64() ^ (0x6300000000ull + l * 32 + j / 8)) >> (8 * (j % 8)));
}
void cpu_mask(const uint32_t* z, uint8_t* M) {
  uint64_t h = seed64();
  for (int t = 0; t < 16; ++t) h = splitmix64(h ^ z[t]);
  for (int i = 0; i < 512; ++i) {
    uint64_t v = splitmix64(h ^ (0x6d00000000ull + i));
    memcpy(M + 8 * i, &v, 8);
  }
}

// NB blocks at once (NB streams from DRAM), KS sketches each; every key and mask line is read once for all NB blocks
template <int KS, int NB>
static void sketch(const uint8_t* const* xb, const uint8_t* M, const int8_t* K, int32_t* const* s) {
  __m512i acc[NB][KS];
  for (int b = 0; b < NB; ++b)
    for (int l = 0; l < KS; ++l) acc[b][l] = _mm512_setzero_si512();
  for (int c = 0; c < 64; c += 4) {
#pragma GCC unroll 4
    for (int q = 0; q < 4; ++q) {
      const __m512i m = _mm512_load_si512(M + 64 * (c + q));
#pragma GCC unroll 4
      for (int b = 0; b < NB; ++b) {
        const __m512i d = _mm512_xor_si512(_mm512_load_si512(xb[b] + 64 * (c + q)), m);
#pragma GCC unroll 16
        for (int l = 0; l < KS; ++l) acc[b][l] = _mm512_dpbusd_epi32(acc[b][l], d, _mm512_load_si512(K + l * 256 + 64 * q));
      }
    }
  }
  for (int b = 0; b < NB; ++b)
    for (int l = 0; l < KS; ++l) s[b][l] = _mm512_reduce_add_epi32(acc[b][l]);
}

void cpu_group(const uint8_t* x, const uint8_t* M, uint64_t zseq, uint64_t group, const int8_t* K, int ks,
               uint8_t leaf[32], int32_t* sk) {
  if (ks == 4)
    for (int i = 0; i < 4; ++i) {  // blocks i, i + 4, i + 8, i + 12 together: four 16 KiB streams
      const uint8_t* xb[4] = {x + 4096 * i, x + 4096 * (i + 4), x + 4096 * (i + 8), x + 4096 * (i + 12)};
      int32_t* s[4] = {sk + 4 * i, sk + 4 * (i + 4), sk + 4 * (i + 8), sk + 4 * (i + 12)};
      sketch<4, 4>(xb, M, K, s);
    }
  else
    for (int b = 0; b < 16; ++b) {
      const uint8_t* xb[1] = {x + 4096 * b};
      int32_t* s[1] = {sk + 16 * b};
      sketch<16, 1>(xb, M, K, s);
    }
  uint8_t msg[32 + 16 + 16 * 16 * 4];
  memcpy(msg, SEED, 32);
  memcpy(msg + 32, &group, 8);
  memcpy(msg + 40, &zseq, 8);
  memcpy(msg + 48, sk, 16 * ks * 4);
  sha256(msg, 48 + 16 * ks * 4, leaf);
}

uint64_t adv_chunk(const void* v, size_t n) {
  const __m512i* p = (const __m512i*)v;
  size_t q = n / 4;
  __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
  for (size_t i = 0; i < q; i += 2) {
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + q + i));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + 2 * q + i));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + 3 * q + i));
    a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i + 1));
    a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + q + i + 1));
    a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + 2 * q + i + 1));
    a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + 3 * q + i + 1));
  }
  return _mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3)));
}

