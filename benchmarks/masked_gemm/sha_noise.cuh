// The edit's noise expanded from sigma by SHA-512 in counter mode (RS-D1b; one hash, SHA-512 with its whole output):
// block (tag, j) = SHA-512(sigma || BE64(tag << 56 | j)), 40 bytes, one compression. The unit is a patch, 4 rows x 128
// columns of the stacked operand (rows x k), p = (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 + col % 128.
//   NS 1, S = {+1, -1}: block j = p, entry e's bit is bit e % 8 of digest byte e / 8, noise 1 - 2 bit (1 bit an entry)
//   NS 2, S = {-8..8}: blocks j = 8 p + r, r = 0, 1, ...; their 64-bit words (big-endian digest words) in order, each
//         word w < 6 17^15 accepted and read as 15 base-17 digits of w mod 17^15, least significant first; entry e is
//         the e-th digit of the accepted words, noise digit - 8 (about 5 bits an entry, 15 digits a 64-bit word)
// tag = 2 (NS - 1) + side (0 for A, 1 for B). The generator hands its patch's noise to apply(c, n) a 16-byte chunk at a
// time, c = 0..31 (row c / 8 of the patch, column chunk c % 8), n its 16 noise values as 4 words of int8 bytes.
#pragma once
#include "../pouw/pearl_c/sha512.cuh"

DEV void sha_block(const uint64_t* sg, uint64_t ctr, uint64_t* st) {
  uint64_t w[16];
  w[0] = sg[0]; w[1] = sg[1]; w[2] = sg[2]; w[3] = sg[3]; w[4] = ctr; w[5] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 6; i < 15; i++) w[i] = 0;
  w[15] = 320;
  rs_iv(st);
  rs_compress(st, w);
}
// 4 bits to 4 int8 bytes, 1 - 2 bit
DEV uint32_t pm1_word(uint32_t nib) { return 0x01010101u | (((nib * 0x00204081u) & 0x01010101u) * 0xFEu); }
constexpr uint64_t P17_7 = 410338673ull, P17_15 = 2862423051509815793ull, U17_LIM = 6 * P17_15;

template <int NS, class F> DEV void sha_patch(const uint64_t* sg, uint32_t tag, uint32_t p, F&& apply) {
  if constexpr (NS == 1) {
    uint64_t st[8];
    sha_block(sg, (uint64_t)tag << 56 | p, st);
#pragma unroll
    for (int c = 0; c < 32; c++) {
      const uint32_t sh = 48 - 16 * (c % 4);
      const uint32_t b16 = (uint32_t)(st[c / 4] >> sh) & 0xFFFFu;   // digest bytes 2c (high) and 2c + 1 (low)
      const uint32_t bits = (b16 >> 8) | ((b16 & 0xFF) << 8);       // byte 2c's bits are entries 16c..16c+7
      const uint32_t n[4] = {pm1_word(bits & 0xF), pm1_word((bits >> 4) & 0xF), pm1_word((bits >> 8) & 0xF), pm1_word(bits >> 12)};
      apply(c, n);
    }
  } else {
    uint64_t st[8];
    int r = 0, wi = 8;   // block r of the patch, word wi of it (8: the next block is due)
    uint32_t buf[4] = {0, 0, 0, 0};
    int pos = 0, c = 0;
    auto put = [&](uint32_t d) {
      const uint32_t x = ((d - 8) & 0xFFu) << (8 * (pos & 3));
      const int q = pos >> 2;
      if (q == 0) buf[0] |= x; else if (q == 1) buf[1] |= x; else if (q == 2) buf[2] |= x; else buf[3] |= x;
      if (++pos == 16) {
        apply(c, buf);
        c++; pos = 0; buf[0] = buf[1] = buf[2] = buf[3] = 0;
      }
    };
    auto put7 = [&](uint32_t v) {
#pragma unroll
      for (int i = 0; i < 7; i++) { const uint32_t q = v / 17u; put(v - 17u * q); v = q; }
    };
    while (c < 32) {
      if (wi == 8) {
        if (r == 8) break;   // 64 words without 35 accepted: never in practice; the gate would see the zero noise
        sha_block(sg, (uint64_t)tag << 56 | ((uint64_t)p * 8 + r), st);
        r++; wi = 0;
      }
      uint64_t w = st[0];
#pragma unroll
      for (int i = 1; i < 8; i++) if (wi == i) w = st[i];
      wi++;
      if (w >= U17_LIM) continue;
      w %= P17_15;
      const uint64_t q1 = w / P17_7, a = q1 / P17_7;
      put7((uint32_t)(w - q1 * P17_7));
      if (c >= 32) break;
      put7((uint32_t)(q1 - a * P17_7));
      if (c >= 32) break;
      put((uint32_t)a);
    }
  }
}
// x + n bytewise: int8 (SWAR, no carry between bytes) or E4M3 (through f16, exact for |x + n| <= 16)
template <bool F8> DEV uint32_t add_noise(uint32_t x, uint32_t n) {
  if constexpr (!F8) {
    return ((x & 0x7F7F7F7Fu) + (n & 0x7F7F7F7Fu)) ^ ((x ^ n) & 0x80808080u);
  } else {
    const __half2 n0 = __halves2half2(__int2half_rn((int)(int8_t)(n & 0xFF)), __int2half_rn((int)(int8_t)((n >> 8) & 0xFF)));
    const __half2 n1 = __halves2half2(__int2half_rn((int)(int8_t)((n >> 16) & 0xFF)), __int2half_rn((int)(int8_t)(n >> 24)));
    __half2_raw a = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(x & 0xFFFF), __NV_E4M3);
    __half2_raw b = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(x >> 16), __NV_E4M3);
    __half2 sa = __hadd2(*reinterpret_cast<__half2*>(&a), n0), sb = __hadd2(*reinterpret_cast<__half2*>(&b), n1);
    const uint32_t ra = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&sa), __NV_SATFINITE, __NV_E4M3);
    const uint32_t rb = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&sb), __NV_SATFINITE, __NV_E4M3);
    return (ra & 0xFFFF) | (rb << 16);
  }
}
