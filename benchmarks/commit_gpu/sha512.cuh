// FIPS 180-4 SHA-512's compression for the committer, on the device and the host, every word a big-endian u64.
//
// Three forms of one compression, all the same function: `compress` (a block of 16 words, the schedule in place),
// `compress_from<R0>` (rounds R0..79 from a state the host advanced through rounds 0..R0-1 of a block whose words 0..R0-1
// are fixed, the feed-forward still adding the chaining value) and `compress_kw` (a fixed block, whose schedule plus round
// constants the host precomputed into a __constant__ table: no schedule on the device).
//
// SHA_ADD selects how the 64-bit adds are issued: 0 lets ptxas emit IADD.64 (half rate on sm_120, ISSUE r6), 1 issues them
// as 32-bit pairs (add.cc / addc: IADD3 and IADD3.X, full rate). Rotations are funnel-shift pairs either way.
#pragma once
#include <cstdint>

#ifndef SHA_ADD
#define SHA_ADD 1
#endif
#define HD __host__ __device__ __forceinline__

namespace sha {

#ifdef __CUDA_ARCH__
#define SHA_K_TABLE K_dev
#else
#define SHA_K_TABLE K_host
#endif

#define SHA_K_VALUES                                                                                                         \
  0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,        \
      0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,    \
      0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,    \
      0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,    \
      0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,    \
      0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,    \
      0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,    \
      0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,    \
      0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,    \
      0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,    \
      0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,    \
      0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,    \
      0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,    \
      0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,    \
      0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,    \
      0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull

__constant__ uint64_t K_dev[80] = {SHA_K_VALUES};
static const uint64_t K_host[80] = {SHA_K_VALUES};

HD void iv(uint64_t* st) {
  st[0] = 0x6a09e667f3bcc908ull; st[1] = 0xbb67ae8584caa73bull; st[2] = 0x3c6ef372fe94f82bull; st[3] = 0xa54ff53a5f1d36f1ull;
  st[4] = 0x510e527fade682d1ull; st[5] = 0x9b05688c2b3e6c1full; st[6] = 0x1f83d9abfb41bd6bull; st[7] = 0x5be0cd19137e2179ull;
}

HD uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

HD uint64_t add(uint64_t a, uint64_t b) {
#if SHA_ADD == 1 && defined(__CUDA_ARCH__)
  uint64_t r;
  asm("{.reg .u32 al, ah, bl, bh, rl, rh;\n\t"
      "mov.b64 {al, ah}, %1;\n\tmov.b64 {bl, bh}, %2;\n\t"
      "add.cc.u32 rl, al, bl;\n\taddc.u32 rh, ah, bh;\n\t"
      "mov.b64 %0, {rl, rh};}"
      : "=l"(r) : "l"(a), "l"(b));
  return r;
#else
  return a + b;
#endif
}

HD uint64_t add3(uint64_t a, uint64_t b, uint64_t c) {
#if SHA_ADD == 1 && defined(__CUDA_ARCH__)
  uint64_t r;
  asm("{.reg .u32 al, ah, bl, bh, cl, ch, rl, rh;\n\t"
      "mov.b64 {al, ah}, %1;\n\tmov.b64 {bl, bh}, %2;\n\tmov.b64 {cl, ch}, %3;\n\t"
      "add.cc.u32 rl, al, bl;\n\taddc.u32 rh, ah, bh;\n\t"
      "add.cc.u32 rl, rl, cl;\n\taddc.u32 rh, rh, ch;\n\t"
      "mov.b64 %0, {rl, rh};}"
      : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
#else
  return a + b + c;
#endif
}

HD uint64_t S0(uint64_t a) { return rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39); }
HD uint64_t S1(uint64_t e) { return rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41); }
HD uint64_t s0(uint64_t w) { return rotr(w, 1) ^ rotr(w, 8) ^ (w >> 7); }
HD uint64_t s1(uint64_t w) { return rotr(w, 19) ^ rotr(w, 61) ^ (w >> 6); }
HD uint64_t ch(uint64_t e, uint64_t f, uint64_t g) { return (e & f) ^ (~e & g); }
HD uint64_t maj(uint64_t a, uint64_t b, uint64_t c) { return (a & b) | (c & (a | b)); }

// Rounds R0..79 of the block w (all 16 words given; the schedule overwrites w), from round-R0 state m, feeding forward cv.
template <int R0>
HD void compress_from(uint64_t* cv, const uint64_t* m, uint64_t* w) {
  uint64_t a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7];
#pragma unroll
  for (int i = 0; i < 80; ++i) {
    if (i < 16) {
      if (i >= R0) {
        const uint64_t kw = add(SHA_K_TABLE[i], w[i]);
        const uint64_t t1 = add3(h, S1(e), ch(e, f, g));
        const uint64_t t1b = add(t1, kw);
        const uint64_t mj = maj(a, b, c);
        h = g; g = f; f = e; e = add(d, t1b);
        d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
      }
    } else {
      const uint64_t wi = add3(add(s1(w[(i - 2) & 15]), w[(i - 7) & 15]), s0(w[(i - 15) & 15]), w[i & 15]);
      w[i & 15] = wi;
      const uint64_t t1 = add3(h, S1(e), ch(e, f, g));
      const uint64_t t1b = add3(t1, SHA_K_TABLE[i], wi);
      const uint64_t mj = maj(a, b, c);
      h = g; g = f; f = e; e = add(d, t1b);
      d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
    }
  }
  cv[0] = add(cv[0], a); cv[1] = add(cv[1], b); cv[2] = add(cv[2], c); cv[3] = add(cv[3], d);
  cv[4] = add(cv[4], e); cv[5] = add(cv[5], f); cv[6] = add(cv[6], g); cv[7] = add(cv[7], h);
}

HD void compress(uint64_t* st, uint64_t* w) { compress_from<0>(st, st, w); }

// A fixed block: kw[i] = K[i] + W[i] for the block's whole schedule, precomputed (host `schedule_kw`).
HD void compress_kw(uint64_t* st, const uint64_t* kw) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 80; ++i) {
    const uint64_t t1 = add3(h, S1(e), ch(e, f, g));
    const uint64_t t1b = add(t1, kw[i]);
    const uint64_t mj = maj(a, b, c);
    h = g; g = f; f = e; e = add(d, t1b);
    d = c; c = b; b = a; a = add3(t1b, S0(b), mj);
  }
  st[0] = add(st[0], a); st[1] = add(st[1], b); st[2] = add(st[2], c); st[3] = add(st[3], d);
  st[4] = add(st[4], e); st[5] = add(st[5], f); st[6] = add(st[6], g); st[7] = add(st[7], h);
}

// --- host helpers -------------------------------------------------------------------------------------------------

inline void schedule_kw(const uint64_t* block, uint64_t* kw) {
  uint64_t w[80];
  for (int i = 0; i < 16; ++i) w[i] = block[i];
  for (int i = 16; i < 80; ++i) w[i] = s1(w[i - 2]) + w[i - 7] + s0(w[i - 15]) + w[i - 16];
  for (int i = 0; i < 80; ++i) kw[i] = K_host[i] + w[i];
}

// The state after rounds 0..r-1 of block (from chaining value cv): the m that compress_from<r> starts from.
inline void rounds_prefix(const uint64_t* cv, const uint64_t* block, int r, uint64_t* m) {
  uint64_t a = cv[0], b = cv[1], c = cv[2], d = cv[3], e = cv[4], f = cv[5], g = cv[6], h = cv[7];
  for (int i = 0; i < r; ++i) {
    const uint64_t t1 = h + S1(e) + ch(e, f, g) + K_host[i] + block[i];
    const uint64_t t2 = S0(a) + maj(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  m[0] = a; m[1] = b; m[2] = c; m[3] = d; m[4] = e; m[5] = f; m[6] = g; m[7] = h;
}

inline uint64_t be64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = v << 8 | p[i];
  return v;
}
inline void put_be64(uint8_t* p, uint64_t v) {
  for (int i = 7; i >= 0; --i) { p[i] = (uint8_t)v; v >>= 8; }
}

// SHA-512 of a whole message on the host (for the constants the device starts from, and the self-test).
inline void digest(const uint8_t* msg, size_t len, uint8_t* out) {
  uint64_t st[8], w[16];
  iv(st);
  const size_t total = ((len + 17 + 127) / 128) * 128;
  for (size_t off = 0; off < total; off += 128) {
    uint8_t blk[128];
    for (int i = 0; i < 128; ++i) {
      const size_t j = off + i;
      blk[i] = j < len ? msg[j] : j == len ? 0x80 : 0;
    }
    if (off + 128 == total) put_be64(blk + 120, (uint64_t)len * 8);
    for (int i = 0; i < 16; ++i) w[i] = be64(blk + 8 * i);
    compress(st, w);
  }
  for (int i = 0; i < 8; ++i) put_be64(out + 8 * i, st[i]);
}

}  // namespace sha
