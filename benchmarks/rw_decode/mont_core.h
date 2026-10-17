// Montgomery squaring mod an n-bit RSA modulus N, one word a thread, and the RW-SMS decode built on it: the public
// direction of a trusted third party's trapdoor encoding of the weights (rw.py is the encoder and the reference).
//
// A square is x -> x^2 R^-1 mod N, R = 2^n, canonical in [0, N). Six forms:
//   FORM 0  SOS: the column square (rate_squaring/sq.cu's `sqr_cols_n`, each unordered pair once), then NL rows of
//           Montgomery reduction in place, each row a low-products carry chain and a high one a limb up.
//   FORM 1  FORM 0 for a half-fixed modulus N = 5 mod 2^(32 KF), KF = NL / 2 - 1 (Lenstra's predetermined portion;
//           p = 3, q = 7 mod 8 keep it a Williams integer): a reduction row multiplies by N's NL - KF + 1 nonzero limbs.
//   FORM 2  CIOS: per limb a_i, t += a_i a, then t += m N and a shift; 2 NL + 2 registers of state in place of 2 NL + NL.
//   FORM 3  FORM 2 for the half-fixed modulus.
//   FORM 4  a Blum modulus N = 1 mod 2^(n/2) (q's low half is p^-1; no Williams tweaks: the encoder pads the payload
//           until each value is a square): the column square, then two Montgomery steps by 2^(n/2), each m = -T mod
//           2^(n/2) (a negation) and one (n/2 x n/2)-bit product m N_hi column by column. NL^2 products a square in
//           place of FORM 0's 1.5 NL^2, and no row waits on the one before.
//   FORM 5  FORM 4 with one Karatsuba level over the column square (sq.cu's `sqr_kara_n`).
// N, n' = -N^-1 mod 2^32, the store-wide XOR mask K, the additive mask t and the block-index key k sit in constant
// memory, so a product takes N's limb as a constant-bank operand. Record j's additive mask is t + (k j mod 2^128): a
// prover who picks W before the salt cannot plant two blocks with equal y (equal records it could store once), which
// t + j would let it do with u_j' = u_j + j - j'.
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
CONST u32 cKj[4];

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
HD u32 madchi(u32 a, u32 b, u32 c) { u32 r; asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
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
HD u32 madchi(u32 a, u32 b, u32 c) { return addc((u32)(((u64)a * b) >> 32), c); }
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
HD bool nz(int j) { return !(FORM == 1 || FORM == 3) || j == 0 || j >= KF; }

// sq.cu's column square: column m's off-diagonal products, doubled, its diagonal, and the carry in.
template <int N>
HD void sqr_cols_n(const u32 (&a)[N], u32 (&t)[2 * N]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = 0; m < 2 * N - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = (m < N ? 0 : m - N + 1); 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    add3(r, c0, c1);
    t[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  t[2 * N - 1] = c0;
}

HD void sqr_cols(const u32 (&a)[NL], u32 (&t)[2 * NL]) { sqr_cols_n<NL>(a, t); }

// sq.cu's one Karatsuba level (FOLD 4): a = a1 2^(n/2) + a0, a^2 = C 2^n + (A + C - D) 2^(n/2) + A, A = a0^2,
// C = a1^2, D = (a0 - a1)^2, three half-width column squares.
HD void sqr_kara(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
  constexpr int H = NL / 2;
  u32 x0[H], x1[H], d[H], e[H];
#pragma unroll
  for (int k = 0; k < H; ++k) {
    x0[k] = a[k];
    x1[k] = a[H + k];
  }
  d[0] = sub_cc(x0[0], x1[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) d[k] = subc_cc(x0[k], x1[k]);
  const u32 mask = subc(0u, 0u);
  e[0] = sub_cc(x1[0], x0[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) e[k] = subc_cc(x1[k], x0[k]);
#pragma unroll
  for (int k = 0; k < H; ++k) d[k] = (d[k] & ~mask) | (e[k] & mask);
  u32 A[2 * H], C[2 * H], D[2 * H], M[2 * H];
  sqr_cols_n<H>(x0, A);
  sqr_cols_n<H>(x1, C);
  sqr_cols_n<H>(d, D);
  M[0] = add_cc(A[0], C[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) M[k] = addc_cc(A[k], C[k]);
  u32 m = addc(0u, 0u);
  M[0] = sub_cc(M[0], D[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) M[k] = subc_cc(M[k], D[k]);
  m = subc(m, 0u);
#pragma unroll
  for (int k = 0; k < 2 * H; ++k) {
    t[k] = A[k];
    t[2 * H + k] = C[k];
  }
  t[H] = add_cc(t[H], M[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) t[H + k] = addc_cc(t[H + k], M[k]);
  t[3 * H] = addc_cc(t[3 * H], m);
#pragma unroll
  for (int k = 3 * H + 1; k < 4 * H - 1; ++k) t[k] = addc_cc(t[k], 0u);
  t[4 * H - 1] = addc(t[4 * H - 1], 0u);
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

// One Montgomery step by 2^(32 H) for N = 1 mod 2^(32 H), H = NL / 2, on the digit at limb OFF: m = -T_d mod 2^(32 H)
// (so n' = -1), T + m N = T + m + m N_hi 2^(32 H) and T_d + m is 2^(32 H) unless T_d = 0, so the digit above takes that
// carry and m N_hi, column by column. Returns the carry out of limb 2 NL - 1 (0 for the first step).
template <int OFF>
HD u32 half_step(u32 (&t)[2 * NL]) {
  constexpr int H = NL / 2;
  u32 m[H];
  m[0] = add_cc(~t[OFF], 1u);
#pragma unroll
  for (int k = 1; k < H; ++k) m[k] = addc_cc(~t[OFF + k], 0u);
  u32 c0 = 1u - addc(0u, 0u), c1 = 0;
#pragma unroll
  for (int k = 0; k < 2 * H - 1; ++k) {
    u32 r[3] = {c0, c1, 0};
#pragma unroll
    for (int i = (k < H ? 0 : k - H + 1); i <= (k < H ? k : H - 1); ++i) mac(r, m[i], cN[H + k - i]);
    add3(r, t[OFF + H + k], 0u);
    t[OFF + H + k] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  constexpr int P = OFF + 3 * H - 1;
  t[P] = add_cc(t[P], c0);
  if constexpr (P + 1 < 2 * NL) {
    t[P + 1] = addc_cc(t[P + 1], c1);
#pragma unroll
    for (int k = P + 2; k < 2 * NL; ++k) t[k] = addc_cc(t[k], 0u);
    return 0u;
  } else {
    return addc(c1, 0u);
  }
}

template <int FORM>
HD void half_sq(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 t[2 * NL];
  if (FORM == 5) sqr_kara(a, t);
  else sqr_cols(a, t);
  half_step<0>(t);
  const u32 top = half_step<NL / 2>(t);
  cond_sub<NL>(t, top, x);
}

template <int FORM>
HD void mont_sq(const u32 (&a)[NL], u32 (&x)[NL]) {
  if (FORM < 2) sos_sq<FORM>(a, x);
  else if (FORM < 4) cios_sq<FORM>(a, x);
  else half_sq<FORM>(a, x);
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
// & 1; FORM 4 and 5 have no tweaks, so a1 = a2 = 0). v = a2 c^2 R^-1, z = v + h N, r = z xor K, y = a1 r^2 R^-1,
// m = y - t_j mod N with t_j = t + (k j mod 2^128). The two rounds are one loop, so a kernel holds one copy of the
// square.
template <int FORM>
HD void decode(const u32 (&c)[NL], u32 tag, u32 j, u32 (&m)[NL]) {
  u32 x[NL], y[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = c[k];
#pragma unroll 1
  for (int round = 0; round < 2; ++round) {
    mont_sq<FORM>(x, y);
    if (FORM < 4) tweak(y, round ? tag & 3u : (tag >> 2) & 3u);
    const u32 hm = round ? 0u : 0u - ((tag >> 4) & 1u), km = round ? 0u : ~0u;
    x[0] = add_cc(y[0], cN[0] & hm);
#pragma unroll
    for (int k = 1; k < NL; ++k) x[k] = addc_cc(y[k], cN[k] & hm);
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] ^= cK[k] & km;
  }
  u32 f[4], tj[NL];
  f[0] = cKj[0] * j;
  f[1] = madhi_cc(cKj[0], j, cKj[1] * j);
  f[2] = madchi_cc(cKj[1], j, cKj[2] * j);
  f[3] = madchi(cKj[2], j, cKj[3] * j);
  tj[0] = add_cc(cT[0], f[0]);
#pragma unroll
  for (int k = 1; k < 4; ++k) tj[k] = addc_cc(cT[k], f[k]);
#pragma unroll
  for (int k = 4; k < NL; ++k) tj[k] = addc_cc(cT[k], 0u);
  m[0] = sub_cc(y[0], tj[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) m[k] = subc_cc(y[k], tj[k]);
  const u32 neg = subc(0u, 0u);
  m[0] = add_cc(m[0], cN[0] & neg);
#pragma unroll
  for (int k = 1; k < NL; ++k) m[k] = addc_cc(m[k], cN[k] & neg);
}

}  // namespace mb
