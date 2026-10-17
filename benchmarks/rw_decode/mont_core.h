// Montgomery squaring mod an n-bit RSA modulus N, one word a thread, and the RW-SMS decode built on it: the public
// direction of a trusted third party's trapdoor encoding of the weights (rw.py is the encoder and the reference).
//
// A square is x -> x^2 R^-1 mod N, R = 2^n, canonical in [0, N). Four forms:
//   FORM 0  SOS: the column square (rate_squaring/sq.cu's `sqr_cols_n`, each unordered pair once), then NL rows of
//           Montgomery reduction in place, each row a low-products carry chain and a high one a limb up.
//   FORM 1  FORM 0 for a half-fixed modulus N = 5 mod 2^(32 KF), KF = NL / 2 - 1 (Lenstra's predetermined portion;
//           p = 3, q = 7 mod 8 keep it a Williams integer): a reduction row multiplies by N's NL - KF + 1 nonzero limbs.
//   FORM 2  CIOS: per limb a_i, t += a_i a, then t += m N and a shift; 2 NL + 2 registers of state in place of 2 NL + NL.
//   FORM 3  FORM 2 for the half-fixed modulus.
// N, n' = -N^-1 mod 2^32, the store-wide XOR mask K and the additive mask t sit in constant memory, so a product
// takes N's limb as a constant-bank operand.
//
// Built for the device by mont.cu and for the host by host_check.cpp, where each carry-chain instruction is emulated
// with a carry flag in a variable, so the arithmetic can be checked without a GPU.
#pragma once
#include <cstdint>

#ifndef MB_BITS
#define MB_BITS 1024
#endif

#ifdef __CUDACC__
#define HD __device__ __forceinline__
#define CONST __constant__
#else
#define HD inline
#define CONST
#endif

namespace mb {

using u32 = uint32_t;
using u64 = uint64_t;
constexpr int NL = MB_BITS / 32;
constexpr int KF = NL / 2 - 1;
static_assert(MB_BITS % 64 == 0 && KF >= 2, "n a multiple of 64, at least 192");

CONST u32 cN[NL];
CONST u32 cNp;
CONST u32 cK[NL];
CONST u32 cT[NL];

#ifdef __CUDACC__
HD u32 add_cc(u32 a, u32 b) { u32 r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 addc_cc(u32 a, u32 b) { u32 r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 addc(u32 a, u32 b) { u32 r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 sub_cc(u32 a, u32 b) { u32 r; asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 subc_cc(u32 a, u32 b) { u32 r; asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 subc(u32 a, u32 b) { u32 r; asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
HD u32 madlo_cc(u32 a, u32 b, u32 c) { u32 r; asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
HD u32 madclo_cc(u32 a, u32 b, u32 c) { u32 r; asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
HD u32 madhi_cc(u32 a, u32 b, u32 c) { u32 r; asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
HD u32 madchi_cc(u32 a, u32 b, u32 c) { u32 r; asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
HD void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
HD void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}
#else
inline u32 CF = 0;
HD u32 add_cc(u32 a, u32 b) { u64 s = (u64)a + b; CF = (u32)(s >> 32); return (u32)s; }
HD u32 addc_cc(u32 a, u32 b) { u64 s = (u64)a + b + CF; CF = (u32)(s >> 32); return (u32)s; }
HD u32 addc(u32 a, u32 b) { return a + b + CF; }
HD u32 sub_cc(u32 a, u32 b) { CF = a < b; return a - b; }
HD u32 subc_cc(u32 a, u32 b) { u64 s = (u64)b + CF; CF = (u64)a < s; return (u32)((u64)a - s); }
HD u32 subc(u32 a, u32 b) { return a - b - CF; }
HD u32 madlo_cc(u32 a, u32 b, u32 c) { return add_cc((u32)((u64)a * b), c); }
HD u32 madclo_cc(u32 a, u32 b, u32 c) { return addc_cc((u32)((u64)a * b), c); }
HD u32 madhi_cc(u32 a, u32 b, u32 c) { return add_cc((u32)(((u64)a * b) >> 32), c); }
HD u32 madchi_cc(u32 a, u32 b, u32 c) { return addc_cc((u32)(((u64)a * b) >> 32), c); }
HD void mac(u32 *r, u32 a, u32 b) {
  const u64 p = (u64)a * b;
  r[0] = add_cc(r[0], (u32)p);
  r[1] = addc_cc(r[1], (u32)(p >> 32));
  r[2] = addc(r[2], 0u);
}
HD void add3(u32 *r, u32 a, u32 b) {
  r[0] = add_cc(r[0], a);
  r[1] = addc_cc(r[1], b);
  r[2] = addc(r[2], 0u);
}
#endif

// N's limb j is known zero in the half-fixed form (FORM 1, 3): 1 <= j < KF.
template <int FORM>
HD bool nz(int j) { return !(FORM & 1) || j == 0 || j >= KF; }

// sq.cu's column square: column m's off-diagonal products, doubled, its diagonal, and the carry in.
HD void sqr_cols(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = 0; m < 2 * NL - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = (m < NL ? 0 : m - NL + 1); 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    add3(r, c0, c1);
    t[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  t[2 * NL - 1] = c0;
}

// x = v - N if v = top 2^n + v[OFF .. OFF + NL) >= N, else that; v < 2N.
template <int OFF = 0, int LEN>
HD void cond_sub(const u32 (&v)[LEN], u32 top, u32 (&x)[NL]) {
  u32 d[NL];
  d[0] = sub_cc(v[OFF], cN[0]);
#pragma unroll
  for (int j = 1; j < NL; ++j) d[j] = subc_cc(v[OFF + j], cN[j]);
  const u32 keep = subc(top, 0u);
#pragma unroll
  for (int j = 0; j < NL; ++j) x[j] = (v[OFF + j] & keep) | (d[j] & ~keep);
}

// Row i of the reduction: t += m N 2^(32 i), m = t_i n'; the carry out of the low chain and of the high chain wait at
// limb i + NL + 1 in `carry` (at most 2), which the next row's low chain takes in.
template <int FORM>
HD void sos_sq(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 t[2 * NL];
  sqr_cols(a, t);
  u32 carry = 0;
#pragma unroll
  for (int i = 0; i < NL; ++i) {
    const u32 m = t[i] * cNp;
    t[i] = madlo_cc(m, cN[0], t[i]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[i + j] = nz<FORM>(j) ? madclo_cc(m, cN[j], t[i + j]) : addc_cc(t[i + j], 0u);
    t[i + NL] = addc_cc(t[i + NL], carry);
    carry = addc(0u, 0u);
    t[i + 1] = madhi_cc(m, cN[0], t[i + 1]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[i + 1 + j] = nz<FORM>(j) ? madchi_cc(m, cN[j], t[i + 1 + j]) : addc_cc(t[i + 1 + j], 0u);
    carry = addc(carry, 0u);
  }
  cond_sub<NL>(t, carry, x);
}

template <int FORM>
HD void cios_sq(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 t[NL + 2];
#pragma unroll
  for (int k = 0; k < NL + 2; ++k) t[k] = 0;
#pragma unroll
  for (int i = 0; i < NL; ++i) {
    const u32 ai = a[i];
    t[0] = madlo_cc(ai, a[0], t[0]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[j] = madclo_cc(ai, a[j], t[j]);
    t[NL] = addc_cc(t[NL], 0u);
    t[NL + 1] = addc(t[NL + 1], 0u);
    t[1] = madhi_cc(ai, a[0], t[1]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[j + 1] = madchi_cc(ai, a[j], t[j + 1]);
    t[NL + 1] = addc(t[NL + 1], 0u);
    const u32 m = t[0] * cNp;
    t[0] = madlo_cc(m, cN[0], t[0]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[j] = nz<FORM>(j) ? madclo_cc(m, cN[j], t[j]) : addc_cc(t[j], 0u);
    t[NL] = addc_cc(t[NL], 0u);
    t[NL + 1] = addc(t[NL + 1], 0u);
    t[1] = madhi_cc(m, cN[0], t[1]);
#pragma unroll
    for (int j = 1; j < NL; ++j) t[j + 1] = nz<FORM>(j) ? madchi_cc(m, cN[j], t[j + 1]) : addc_cc(t[j + 1], 0u);
    t[NL + 1] = addc(t[NL + 1], 0u);
#pragma unroll
    for (int k = 0; k <= NL; ++k) t[k] = t[k + 1];
    t[NL + 1] = 0;
  }
  cond_sub(t, t[NL], x);
}

template <int FORM>
HD void mont_sq(const u32 (&a)[NL], u32 (&x)[NL]) {
  if (FORM < 2) sos_sq<FORM>(a, x);
  else cios_sq<FORM>(a, x);
}

// x = (-1)^e 2^f x mod N for tweak bits (e, f) = (tw & 1, tw >> 1 & 1), branch-free; x in [0, N).
HD void tweak(u32 (&x)[NL], u32 tw) {
  const u32 fm = 0u - ((tw >> 1) & 1u), em = 0u - (tw & 1u);
  u32 s[NL];
  s[0] = x[0] << 1;
#pragma unroll
  for (int j = 1; j < NL; ++j) s[j] = (x[j] << 1) | (x[j - 1] >> 31);
  const u32 top = x[NL - 1] >> 31;
  u32 d[NL];
  cond_sub(s, top, d);
#pragma unroll
  for (int j = 0; j < NL; ++j) x[j] = (x[j] & ~fm) | (d[j] & fm);
  d[0] = sub_cc(cN[0], x[0]);
#pragma unroll
  for (int j = 1; j < NL; ++j) d[j] = subc_cc(cN[j], x[j]);
#pragma unroll
  for (int j = 0; j < NL; ++j) x[j] = (x[j] & ~em) | (d[j] & em);
}

// The RW-SMS decode of record j: c (NL limbs, c < N) and its tag byte (a1 = tag & 3, a2 = tag >> 2 & 3, h = tag >> 4
// & 1). v = a2 c^2 R^-1, z = v + h N, r = z xor K, y = a1 r^2 R^-1, m = y - (t + j) mod N. The two rounds are one
// loop, so a kernel holds one copy of the square.
template <int FORM>
HD void decode(const u32 (&c)[NL], u32 tag, u32 j, u32 (&m)[NL]) {
  u32 x[NL], y[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = c[k];
#pragma unroll 1
  for (int round = 0; round < 2; ++round) {
    mont_sq<FORM>(x, y);
    tweak(y, round ? tag & 3u : (tag >> 2) & 3u);
    const u32 hm = round ? 0u : 0u - ((tag >> 4) & 1u), km = round ? 0u : ~0u;
    x[0] = add_cc(y[0], cN[0] & hm);
#pragma unroll
    for (int k = 1; k < NL; ++k) x[k] = addc_cc(y[k], cN[k] & hm);
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] ^= cK[k] & km;
  }
  u32 tj[NL];
  tj[0] = add_cc(cT[0], j);
#pragma unroll
  for (int k = 1; k < NL; ++k) tj[k] = addc_cc(cT[k], 0u);
  m[0] = sub_cc(y[0], tj[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) m[k] = subc_cc(y[k], tj[k]);
  const u32 neg = subc(0u, 0u);
  m[0] = add_cc(m[0], cN[0] & neg);
#pragma unroll
  for (int k = 1; k < NL; ++k) m[k] = addc_cc(m[k], cN[k] & neg);
}

}  // namespace mb
