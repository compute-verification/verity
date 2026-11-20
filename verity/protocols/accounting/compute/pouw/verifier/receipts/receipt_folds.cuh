// PoUW's receipt folds in C++ (CUDA device and host), the same as receipt_folds.py: each fold's step, the key rules, and
// the start states and keys of the schedules that run them. Every word is a uint32 and all arithmetic wraps.
// receipt_conform.py holds this header to vectors.json (tests/test_receipt_folds.py builds conform_host.cpp with g++, and
// with nvcc where it is installed).
//
// Soundness (receipt_folds.py FOLDS; README.md):
//   duo    needs (D) past four words: one, two and three proved (Lean Pouw.SecurityProofs.Receipt.DuoWordInjective,
//          DuoPairNoKernel, DuoTripleNoKernel), four at both schedules' keys (DuoQuadNoKernel)
//   xin13  needs (RT) and (G3) (the referee's Part IX, r11); keys under key_ok_r9, key_ok_r11 for r11's Theorem 1
//   li     needs (RT), (G3) and (HO)
//   ram    binds pairs only (a step reads (d0, d1) through Q = a0 d0 + d1)
//   xin7   retired (pfill.cuh refuses -DPFILL_XIN)
#pragma once
#include <stdint.h>

// The key rules are RF_RULE, inline but not forced: forced, lambda8's 128-step loop changes the code of every kernel
// that derives keys (pfill.cuh's r9 builds keep their SASS only this way).
#ifdef __CUDACC__
#define RF_HD __host__ __device__ __forceinline__
#define RF_RULE __host__ __device__ inline
#else
#define RF_HD inline
#define RF_RULE inline
#endif

// r in 1..31
RF_HD uint32_t rf_rotl(uint32_t x, int r) {
#ifdef __CUDA_ARCH__
  return __funnelshift_l(x, x, r);
#else
  return (x << r) | (x >> (32 - r));
#endif
}
RF_HD uint32_t rf_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}

// ---- steps
RF_HD uint32_t rf_ram_step(uint32_t y, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t a0, uint32_t a1) {
  y = (rf_rotl(y, 7) + d0) * a0 + d1;
  return (rf_rotl(y, 19) + d2) * a1 + d3;
}
RF_HD uint32_t rf_xin13_step(uint32_t y, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t a0, uint32_t a1) {
  y = (rf_rotl(y, 13) ^ d0) * a0 + d1;
  return (rf_rotl(y, 19) ^ d2) * a1 + d3;
}
RF_HD uint32_t rf_xin7_step(uint32_t y, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t a0, uint32_t a1) {
  y = (rf_rotl(y, 7) ^ d0) * a0 + d1;
  return (rf_rotl(y, 19) ^ d2) * a1 + d3;
}
RF_HD uint32_t rf_li_step(uint32_t y, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t a0, uint32_t a1,
                          uint32_t k0, uint32_t k1) {
  y = (y ^ d0 ^ k0) * a0 + d1;
  return (y ^ d2 ^ k1) * a1 + d3;
}
RF_HD void rf_duo_word(uint32_t& y, uint32_t& z, uint32_t d, uint32_t k, uint32_t a, int r) {
  const uint32_t u = y ^ d ^ k;
  y = u * a + z;
  z = rf_rotl(u, r) + y;
}
// pfill's duo step: words d0..d3 with keys (k0, a0), (k1, a1), (k0, a0), (k1, a1), rotations 5, 11, 17, 29
RF_HD void rf_duo_step(uint32_t& y, uint32_t& z, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t a0,
                       uint32_t a1, uint32_t k0, uint32_t k1) {
  rf_duo_word(y, z, d0, k0, a0, 5);
  rf_duo_word(y, z, d1, k1, a1, 11);
  rf_duo_word(y, z, d2, k0, a0, 17);
  rf_duo_word(y, z, d3, k1, a1, 29);
}
RF_HD uint32_t rf_duo_final(uint32_t y, uint32_t z) { return y ^ rf_rotl(z, 16); }

// ---- key rules
RF_RULE bool rf_key_ok(uint32_t a) {
  const uint32_t m = a & 0xFFFFFFu, r = a & 31u;
  return (a & 1u) && m > 256u && m < (1u << 24) - 256u && r != 1u && r != 31u;
}
// a odd, and a b mod 2^30 outside [-128, 128] for every b in 1..128 (neither a nor -a in Λ8 mod 2^30)
RF_RULE bool rf_lambda8_ok(uint32_t a) {
  if (!(a & 1u)) return false;
  uint32_t t = 0;
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
  for (int b = 1; b <= 128; b++) {
    t += a;
    if (((t + 128u) & 0x3FFFFFFFu) <= 256u) return false;
  }
  return true;
}
RF_RULE bool rf_key_ok_r9(uint32_t a) { return rf_key_ok(a) && rf_lambda8_ok(a); }
// for each j < 19, the signed residue of a mod 2^(31 - j) is at least 2^(19 - j) in size
RF_RULE bool rf_kj19(uint32_t a) {
  for (int j = 0; j < 19; j++) {
    const uint32_t mod = 1u << (31 - j), v = a & (mod - 1u), lim = 1u << (19 - j);
    const uint32_t mag = v >= (mod >> 1) ? mod - v : v;
    if (mag < lim) return false;
  }
  return true;
}
RF_RULE bool rf_key_ok_r11(uint32_t a) { return rf_key_ok_r9(a) && rf_kj19(a); }

// ---- schedules' starts and keys
RF_HD uint32_t rf_pfill_y0(uint32_t pos) { return pos; }
RF_HD uint32_t rf_pfill_z0(uint32_t pos) { return pos * 0x9E3779B9u; }
// fgemm.cu's lane-0 duo keys at position t = 4 mi + ni
RF_HD uint32_t rf_served_ka(int t) { return rf_mix32(0xA0000000u + 2u * t) | 1u; }
RF_HD uint32_t rf_served_kk(int t) { return rf_mix32(0xB0000000u + 2u * t + 1u); }
RF_HD int rf_served_rot(int t) { return ((t * 7 + 5) % 31) + 1; }
// a served duo chain's start from its index ((tile 512 + ctid) nw + w) 8 + j
RF_HD uint32_t rf_served_y0(uint32_t seed, uint32_t idx) { return rf_mix32(seed ^ rf_mix32(idx)); }
RF_HD uint32_t rf_served_z0(uint32_t y0) { return rf_mix32(y0 + 0x9E3779B9u); }
RF_HD uint32_t rf_egemm_key_candidate(uint32_t seed, uint32_t i) { return rf_mix32(seed ^ (0x5A5A0000u + i)); }
RF_HD uint32_t rf_egemm_xin13_y0(uint32_t tile, uint32_t ctid, int c) { return (tile * 512u + ctid) * 4u + c; }
RF_HD uint32_t rf_e2e_key_candidate(uint32_t i) { return rf_mix32(0xC0000000u + i) | 1u; }
