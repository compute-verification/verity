// `ncp-v2`'s committed values, one unit at a time, as functions that compile for the host (g++, the CPU gate) and the device
// (nvcc, `ncp2_kernels.cu`). Each is `pouw_native`'s computation of the same values; `pouw_device.first_difference` holds
// the two to the same leaves. Hashes: SHA-256 and SHA-512 (FIPS 180-4), Keccak-p[1600, n_r] with SHAKE256 (FIPS 202) and
// TurboSHAKE128 (RFC 9861). Floating point: every step is one IEEE binary32 operation, rounded to nearest, without
// contraction (the host builds with -ffp-contract=off; the device uses the _rn intrinsics).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#define HDM __host__ __device__ __forceinline__
#else
#define HD static inline
#define HDM inline
#endif

namespace ncp2 {

constexpr int D = 16;
constexpr int BETA = 128;
constexpr uint32_t F32_NAN_PTX = 0x7FFFFFFFu;
constexpr uint16_t BF16_NAN = 0x7FC0u;

// ---- SHA-256 ------------------------------------------------------------------------------------------------------------

HD uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256 {
  uint32_t h[8];
  uint32_t w[16];     // the block being filled, as big-endian words
  uint32_t fill;      // bytes in the block
  uint64_t len;       // bytes absorbed
};

HD void sha256_compress(uint32_t h[8], const uint32_t blk[16]) {
  const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = blk[i];
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t wi;
    if (i < 16) {
      wi = w[i];
    } else {
      uint32_t w15 = w[(i + 1) & 15], w2 = w[(i + 14) & 15];
      uint32_t s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3), s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10);
      wi = w[i & 15] = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
    }
    uint32_t t1 = hh + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + wi;
    uint32_t t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

HD void sha256_init(Sha256& s) {
  const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  for (int i = 0; i < 8; ++i) s.h[i] = iv[i];
  for (int i = 0; i < 16; ++i) s.w[i] = 0;
  s.fill = 0;
  s.len = 0;
}

HD void sha256_byte(Sha256& s, uint8_t b) {
  s.w[s.fill >> 2] |= (uint32_t)b << (24 - 8 * (s.fill & 3));
  ++s.len;
  if (++s.fill == 64) {
    sha256_compress(s.h, s.w);
    for (int i = 0; i < 16; ++i) s.w[i] = 0;
    s.fill = 0;
  }
}

HD void sha256_bytes(Sha256& s, const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) sha256_byte(s, p[i]);
}

HD void sha256_u32le(Sha256& s, uint32_t v) {
  for (int i = 0; i < 4; ++i) sha256_byte(s, (uint8_t)(v >> (8 * i)));
}

HD void sha256_final(Sha256& s, uint8_t out[32]) {
  uint64_t bits = s.len * 8;
  sha256_byte(s, 0x80);
  while (s.fill != 56) sha256_byte(s, 0);
  for (int i = 7; i >= 0; --i) sha256_byte(s, (uint8_t)(bits >> (8 * i)));
  for (int i = 0; i < 32; ++i) out[i] = (uint8_t)(s.h[i >> 2] >> (24 - 8 * (i & 3)));
}

// ---- SHA-512 ------------------------------------------------------------------------------------------------------------

HD uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

struct Sha512 {
  uint64_t h[8];
  uint64_t w[16];
  uint32_t fill;
  uint64_t len;
};

HD void sha512_compress(uint64_t h[8], const uint64_t blk[16]) {
  const uint64_t K[80] = {
      0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
      0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
      0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
      0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
      0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
      0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
      0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
      0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
      0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
      0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
      0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
      0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
      0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
      0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
      0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
      0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = blk[i];
  uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
  for (int i = 0; i < 80; ++i) {
    uint64_t wi;
    if (i < 16) {
      wi = w[i];
    } else {
      uint64_t w15 = w[(i + 1) & 15], w2 = w[(i + 14) & 15];
      uint64_t s0 = rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7), s1 = rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6);
      wi = w[i & 15] = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
    }
    uint64_t t1 = hh + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) + K[i] + wi;
    uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

HD void sha512_init(Sha512& s) {
  const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                          0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  for (int i = 0; i < 8; ++i) s.h[i] = iv[i];
  for (int i = 0; i < 16; ++i) s.w[i] = 0;
  s.fill = 0;
  s.len = 0;
}

HD void sha512_byte(Sha512& s, uint8_t b) {
  s.w[s.fill >> 3] |= (uint64_t)b << (56 - 8 * (s.fill & 7));
  ++s.len;
  if (++s.fill == 128) {
    sha512_compress(s.h, s.w);
    for (int i = 0; i < 16; ++i) s.w[i] = 0;
    s.fill = 0;
  }
}

HD void sha512_bytes(Sha512& s, const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) sha512_byte(s, p[i]);
}

HD void sha512_final(Sha512& s, uint8_t out[64]) {
  uint64_t bits = s.len * 8;
  sha512_byte(s, 0x80);
  while (s.fill != 112) sha512_byte(s, 0);
  for (int i = 0; i < 8; ++i) sha512_byte(s, 0);                   // the length's high 64 bits
  for (int i = 7; i >= 0; --i) sha512_byte(s, (uint8_t)(bits >> (8 * i)));
  for (int i = 0; i < 64; ++i) out[i] = (uint8_t)(s.h[i >> 3] >> (56 - 8 * (i & 7)));
}

// ---- Keccak-p[1600, n_r]: SHAKE256 (24 rounds) and TurboSHAKE128 (the last 12) --------------------------------------------

HD uint64_t rotl64(uint64_t x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }

template <int ROUNDS>
HD void keccak_p(uint64_t s[25]) {
  const uint64_t RC[24] = {
      0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull, 0x000000000000808Bull,
      0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull, 0x000000000000008Aull, 0x0000000000000088ull,
      0x0000000080008009ull, 0x000000008000000Aull, 0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull,
      0x8000000000008003ull, 0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
      0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};
  const int ROT[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int PI[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
#pragma unroll
  for (int r = 24 - ROUNDS; r < 24; ++r) {
    uint64_t c[5];
#pragma unroll
    for (int x = 0; x < 5; ++x) c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
#pragma unroll
    for (int x = 0; x < 5; ++x) {
      uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
#pragma unroll
      for (int y = 0; y < 25; y += 5) s[y + x] ^= d;
    }
    uint64_t t = s[1];
#pragma unroll
    for (int i = 0; i < 24; ++i) {
      uint64_t u = s[PI[i]];
      s[PI[i]] = rotl64(t, ROT[i]);
      t = u;
    }
#pragma unroll
    for (int y = 0; y < 25; y += 5) {
      uint64_t row[5];
#pragma unroll
      for (int x = 0; x < 5; ++x) row[x] = s[y + x];
#pragma unroll
      for (int x = 0; x < 5; ++x) s[y + x] = row[x] ^ (~row[(x + 1) % 5] & row[(x + 2) % 5]);
    }
    s[0] ^= RC[r];
  }
}

// A sponge over Keccak-p[1600, ROUNDS] at RATE bytes, padded with the domain byte ``ds`` and the final 0x80.
template <int ROUNDS, int RATE>
struct Sponge {
  uint64_t s[25];
  uint32_t pos;

  HDM void init() {
    for (int i = 0; i < 25; ++i) s[i] = 0;
    pos = 0;
  }
  HDM void byte(uint8_t b) {
    s[pos >> 3] ^= (uint64_t)b << (8 * (pos & 7));
    if (++pos == RATE) {
      keccak_p<ROUNDS>(s);
      pos = 0;
    }
  }
  HDM void bytes(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) byte(p[i]);
  }
  HDM void u32le(uint32_t v) {
    for (int i = 0; i < 4; ++i) byte((uint8_t)(v >> (8 * i)));
  }
  HDM void u64le(uint64_t v) {
    for (int i = 0; i < 8; ++i) byte((uint8_t)(v >> (8 * i)));
  }
  HDM void pad(uint8_t ds) {
    s[pos >> 3] ^= (uint64_t)ds << (8 * (pos & 7));
    s[(RATE - 1) >> 3] ^= (uint64_t)0x80 << (8 * ((RATE - 1) & 7));
    keccak_p<ROUNDS>(s);
    pos = 0;
  }
  HDM uint8_t squeeze() {
    if (pos == RATE) {
      keccak_p<ROUNDS>(s);
      pos = 0;
    }
    uint8_t b = (uint8_t)(s[pos >> 3] >> (8 * (pos & 7)));
    ++pos;
    return b;
  }
};

using Shake256 = Sponge<24, 136>;
using TurboShake128 = Sponge<12, 168>;
constexpr uint8_t SHAKE_DS = 0x1F;

// ---- ncp-v2 (`verity.ml.protocols.pouw.circuit.ncp2`'s docstring; `pouw_native`) --------------------------------------------------------

// Tags, as `ncp2` and `pouw_circuit` spell them (the unit tag's and the node tag's NUL included).
#define NCP2_TAG_ROW "ncp2-row"
#define NCP2_TAG_STRIP "ncp2-stp"
#define NCP2_TAG_NODE "ncp2-nod"
#define NCP2_TAG_KEY "ncp2-key"
#define NCP2_TAG_E1 "ncp2-E1\0"
#define NCP2_TAG_F1 "ncp2-F1\0"
#define NCP2_UNIT_TAG "verity/pouw/ncp-v2/unit/v1\0"
#define NCP2_NODE_TAG "verity/pouw/ncp-v2/units-node/v1\0"
constexpr int UNIT_TAG_LEN = 27;
constexpr int NODE_TAG_LEN = 33;

HD int32_t noise(uint8_t b) { return (int32_t)(b & 63) - 31; }

template <class H>
HD void index_word(H& h, uint32_t i) {
  uint64_t v = ((uint64_t)i << 32) | i;
  for (int r = 0; r < 5; ++r) h.u64le(v);
}

// E₁ row ``rowid``: noise(SHAKE256(i ×5 ‖ "ncp2-E1\0" ‖ K)[:k]).
HD void e1_row(const uint8_t key[32], uint32_t rowid, int k, int8_t* e) {
  Shake256 x;
  x.init();
  index_word(x, rowid);
  x.bytes((const uint8_t*)NCP2_TAG_E1, 8);
  x.bytes(key, 32);
  x.pad(SHAKE_DS);
  for (int l = 0; l < k; ++l) e[l] = (int8_t)noise(x.squeeze());
}

// F₁ column ``colid``: noise(SHAKE256(j ×5 ‖ "ncp2-F1\0" ‖ salt ‖ weight id)[:k]).
HD void f1_col(const uint8_t salt[32], uint64_t weight, uint32_t colid, int k, int8_t* f) {
  Shake256 x;
  x.init();
  index_word(x, colid);
  x.bytes((const uint8_t*)NCP2_TAG_F1, 8);
  x.bytes(salt, 32);
  x.u64le(weight);
  x.pad(SHAKE_DS);
  for (int l = 0; l < k; ++l) f[l] = (int8_t)noise(x.squeeze());
}

// X + β for one row: [A + E₁ | A + E₁∘π | −(A + E₁ + E₁∘π)] + β, as bytes.
HD void form_x_row(const int8_t* a, const int8_t* e, const uint32_t* pi, int k, uint8_t* x) {
  for (int l = 0; l < k; ++l) {
    int32_t al = a[l], el = e[l], ep = e[pi[l]];
    x[l] = (uint8_t)(al + el + BETA);
    x[k + l] = (uint8_t)(al + ep + BETA);
    x[2 * k + l] = (uint8_t)(BETA - (al + el + ep));
  }
}

// Y for one column: [B + F₁ | B − F₁∘π⁻¹ | B + F₁ − F₁∘π⁻¹] as s8 bytes; returns c₀ = −β·Σ Y mod 2³².
HD uint32_t form_y_col(const int8_t* b, const int8_t* f, const uint32_t* inv, int k, uint8_t* y) {
  int64_t sum = 0;
  for (int l = 0; l < k; ++l) {
    int32_t bl = b[l], fl = f[l], fi = f[inv[l]];
    y[l] = (uint8_t)(bl + fl);
    y[k + l] = (uint8_t)(bl - fi);
    y[2 * k + l] = (uint8_t)(bl + fl - fi);
    sum += (int8_t)y[l] + (int8_t)y[k + l] + (int8_t)y[2 * k + l];
  }
  return (uint32_t)(-(int64_t)BETA * sum);
}

// One output (r, c) of a tile: route U from c₀, every 16-deep checkpoint, written to out[t · stride].
HD void tile_entry(const uint8_t* x, const uint8_t* y, uint32_t c0, int steps, uint32_t* out, size_t stride) {
  uint32_t run = c0;
  for (int t = 0; t < steps; ++t) {
    int32_t acc = 0;
    for (int d = 0; d < D; ++d) acc += (int32_t)x[D * t + d] * (int32_t)(int8_t)y[D * t + d];
    run += (uint32_t)acc;
    out[(size_t)t * stride] = run;
  }
}

// ---- dequantization: Bf16Add(F32ToBf16Rn(F32Mul(F32Mul(s, t), I32ToF32Rn(z))), bias) ----------------------------------------

HD float as_f32(uint32_t u) {
  union { uint32_t u; float f; } v;
  v.u = u;
  return v.f;
}

HD uint32_t as_u32(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;
  return v.u;
}

HD bool is_nan32(uint32_t u) { return (u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu) != 0; }

HD float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}

HD float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}

HD float i32_to_f32_rn(int32_t z) {
#ifdef __CUDA_ARCH__
  return __int2float_rn(z);
#else
  return (float)z;
#endif
}

// `F32Mul_v2`: RN(a·b), every NaN F32_NAN_PTX.
HD uint32_t f32_mul(uint32_t a, uint32_t b) {
  uint32_t r = as_u32(mul_rn(as_f32(a), as_f32(b)));
  return is_nan32(r) ? F32_NAN_PTX : r;
}

// `F32ToBf16Rn_v1`: RNE, every NaN 0x7FC0.
HD uint16_t f32_to_bf16_rn(uint32_t u) {
  if (is_nan32(u)) return BF16_NAN;
  return (uint16_t)(((uint64_t)u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

HD uint16_t dequant(uint32_t z, uint32_t s, uint32_t t, uint16_t bias) {
  uint32_t zf = as_u32(i32_to_f32_rn((int32_t)z));
  uint16_t v = f32_to_bf16_rn(f32_mul(f32_mul(s, t), zf));
  return f32_to_bf16_rn(as_u32(add_rn(as_f32((uint32_t)v << 16), as_f32((uint32_t)bias << 16))));
}

// ---- leaves --------------------------------------------------------------------------------------------------------------

HD void unit_leaf_begin(Sha256& h) {
  sha256_init(h);
  sha256_bytes(h, (const uint8_t*)NCP2_UNIT_TAG, UNIT_TAG_LEN);
}

// `pouw_circuit.unit_leaf` over ``n`` words at a stride.
HD void unit_leaf(const uint32_t* w, size_t n, size_t stride, uint8_t out[32]) {
  Sha256 h;
  unit_leaf_begin(h);
  for (size_t i = 0; i < n; ++i) sha256_u32le(h, w[i * stride]);
  sha256_final(h, out);
}

// A unit whose outputs are bytes, each one output word (form_x, form_y's Y).
HD void unit_leaf_bytes(Sha256& h, const uint8_t* b, size_t n) {
  for (size_t i = 0; i < n; ++i) sha256_u32le(h, b[i]);
}

// A digest unit's outputs are its big-endian words (`pouw_native._be`).
HD void unit_leaf_digest(const uint8_t d[64], uint8_t out[32]) {
  Sha256 h;
  unit_leaf_begin(h);
  for (int i = 0; i < 16; ++i)
    sha256_u32le(h, ((uint32_t)d[4 * i] << 24) | ((uint32_t)d[4 * i + 1] << 16) | ((uint32_t)d[4 * i + 2] << 8) | d[4 * i + 3]);
  sha256_final(h, out);
}

// The same word leaf under TurboSHAKE128 (32 bytes, domain byte 0x1F): the audit's A22 alternative, not committed.
HD void unit_leaf_turbo(const uint32_t* w, size_t n, size_t stride, uint8_t out[32]) {
  TurboShake128 h;
  h.init();
  h.bytes((const uint8_t*)NCP2_UNIT_TAG, UNIT_TAG_LEN);
  for (size_t i = 0; i < n; ++i) h.u32le(w[i * stride]);
  h.pad(SHAKE_DS);
  for (int i = 0; i < 32; ++i) out[i] = h.squeeze();
}

// The same word leaf under SHA-512 (64 bytes): A22's decided leaf hash, not committed.
HD void unit_leaf_sha512(const uint32_t* w, size_t n, size_t stride, uint8_t out[64]) {
  Sha512 h;
  sha512_init(h);
  sha512_bytes(h, (const uint8_t*)NCP2_UNIT_TAG, UNIT_TAG_LEN);
  for (size_t i = 0; i < n; ++i)
    for (int b = 0; b < 4; ++b) sha512_byte(h, (uint8_t)(w[i * stride] >> (8 * b)));
  sha512_final(h, out);
}

// One message hashed: kind 0 SHA-256 (32 bytes), 1 SHA-512 (64), 2 SHAKE256 and 3 TurboSHAKE128 (``outlen`` bytes, domain
// byte ``ds``). Returns 0, 1 for a SHA-2 width other than its own, 2 for an unknown kind.
HD int hash_one(int kind, int ds, const uint8_t* p, size_t len, size_t outlen, uint8_t* o) {
  if (kind == 0) {
    if (outlen != 32) return 1;
    Sha256 h;
    sha256_init(h);
    sha256_bytes(h, p, len);
    sha256_final(h, o);
  } else if (kind == 1) {
    if (outlen != 64) return 1;
    Sha512 h;
    sha512_init(h);
    sha512_bytes(h, p, len);
    sha512_final(h, o);
  } else if (kind == 2) {
    Shake256 h;
    h.init();
    h.bytes(p, len);
    h.pad((uint8_t)ds);
    for (size_t j = 0; j < outlen; ++j) o[j] = h.squeeze();
  } else if (kind == 3) {
    TurboShake128 h;
    h.init();
    h.bytes(p, len);
    h.pad((uint8_t)ds);
    for (size_t j = 0; j < outlen; ++j) o[j] = h.squeeze();
  } else {
    return 2;
  }
  return 0;
}

HD void units_node(const uint8_t* left, const uint8_t* right, uint8_t out[32]) {
  Sha256 h;
  sha256_init(h);
  sha256_bytes(h, (const uint8_t*)NCP2_NODE_TAG, NODE_TAG_LEN);
  sha256_bytes(h, left, 32);
  sha256_bytes(h, right, 32);
  sha256_final(h, out);
}

// ---- the call's key --------------------------------------------------------------------------------------------------------

HD void row_digest(const int8_t* row, int k, uint8_t out[64]) {
  Sha512 h;
  sha512_init(h);
  sha512_bytes(h, (const uint8_t*)NCP2_TAG_ROW, 8);
  sha512_bytes(h, (const uint8_t*)row, (size_t)k);
  sha512_final(h, out);
}

HD void strip_digest(const uint8_t rows[D * 64], uint8_t out[64]) {
  Sha512 h;
  sha512_init(h);
  sha512_bytes(h, (const uint8_t*)NCP2_TAG_STRIP, 8);
  sha512_bytes(h, rows, D * 64);
  sha512_final(h, out);
}

HD void tree_node(const uint8_t* left, const uint8_t* right, uint8_t out[64]) {
  Sha512 h;
  sha512_init(h);
  sha512_bytes(h, (const uint8_t*)NCP2_TAG_NODE, 8);
  sha512_bytes(h, left, 64);
  sha512_bytes(h, right, 64);
  sha512_final(h, out);
}

// The RFC 6962 tree over strips lo … hi − 1 (`ncp2.rfc6962`): nodes appended to ``nodes`` (64 bytes each) in emission order
// (after both subtrees), ``*count`` of them; returns the root, a strip's own digest or a node's. Iterative, so the device
// stack stays static: a frame per level, for at most 2^19 strips (`ncp2_commit` refuses more).
HD const uint8_t* rfc6962(const uint8_t* strips, int lo, int hi, uint8_t* nodes, int* count) {
  struct Frame { int lo, hi, split; const uint8_t* left; };
  Frame st[21];
  int top = 0;
  const uint8_t* done = nullptr;
  st[0] = {lo, hi, 0, nullptr};
  while (true) {
    Frame& f = st[top];
    if (done == nullptr) {
      if (f.hi - f.lo == 1) {
        done = strips + 64 * (size_t)f.lo;
      } else {
        int split = 1;
        while (split * 2 < f.hi - f.lo) split *= 2;
        f.split = split;
        st[++top] = {f.lo, f.lo + split, 0, nullptr};
        continue;
      }
    } else if (f.left == nullptr) {
      f.left = done;
      done = nullptr;
      st[++top] = {f.lo + f.split, f.hi, 0, nullptr};
      continue;
    } else {
      uint8_t* out = nodes + 64 * (size_t)(*count)++;
      tree_node(f.left, done, out);
      done = out;
    }
    if (top == 0) return done;
    --top;
  }
}

// K = SHAKE256(D_A as lanes ‖ "ncp2-key" ‖ salt ‖ forward ‖ position ‖ weight id)[:32].
HD void call_key(const uint8_t salt[32], uint64_t forward, uint64_t position, uint64_t weight, const uint8_t da[64],
                 uint8_t out[32]) {
  Shake256 x;
  x.init();
  for (int w = 0; w < 8; ++w)
    for (int i = 7; i >= 0; --i) x.byte(da[8 * w + i]);
  x.bytes((const uint8_t*)NCP2_TAG_KEY, 8);
  x.bytes(salt, 32);
  x.u64le(forward);
  x.u64le(position);
  x.u64le(weight);
  x.pad(SHAKE_DS);
  for (int i = 0; i < 32; ++i) out[i] = x.squeeze();
}

}  // namespace ncp2
