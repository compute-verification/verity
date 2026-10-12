// Squarings mod p = 2^512 - 569 on the CUDA cores (sm_120), a thread holding ILP words, for the adversary's floor on
// Sloth's encode (a root is (p+1)/4, about 510 squarings). Every form squares column by column (ra-word-codec's and
// ra-codec-tensor's `sqr_full`, the red team's form 0: each unordered pair once, doubled, a three-word accumulator) and
// differs in the fold 2^512 = 569:
//   FOLD 0  the red team's `fold_cols` (C u64 carries), ptxas's MOV / IADD.64 churn included: the baseline.
//   FOLD 1  `kv.cu`'s PTX carry chains: the low products' chain, then the high ones' a limb up.
//   FOLD 2  FOLD 1 with each multiply by 569 = 2^9 + 2^5 + 2^4 + 2^3 + 1 as shifts and adds on the integer ALU, so the
//           FMA-heavy pipe (the binding one, 84% busy at 512 bits) runs only the square's 136 IMAD.WIDE.
// ILP interleaves that many independent words a thread, to fill the pipe past the carry chains' latency.
//   FOLD 3  the fold inside the column square: the high columns first, then each low column takes 569 h_m as one more
//           product, so the fold is 16 IMAD.WIDE and the columns' carries.
//   FOLD 4  one level of Karatsuba over the column square (the red team's `sqr_kara`), then FOLD 1's fold.
//   FOLD 5  the off-diagonal products' columns chained by their carries, one doubling, the squares in one chain,
//           then FOLD 1's fold: fewer integer-ALU instructions beside the same multiplies.
//   FOLD 6  FOLD 4's Karatsuba level over FOLD 5's 8-limb squares.
//   FOLD 7  FOLD 5's square, then a fold of one 64-bit multiply-add a limb.
//   FOLD 8, 9   FOLD 6 held to 6 and 7 blocks of 128 a SM (80 and 72 registers), where its 96 registers allow 5.
//   FOLD 10     FOLD 5 held to 8 blocks a SM (64 registers), where its 72 allow 7.
//   FOLD 11, 12 two and three Karatsuba levels over FOLD 5's squares (NL/4- and NL/8-limb), then FOLD 1's fold.
// All keep x lazily below 2^512; the host reduces below p. Other widths build with -DSQ_BITS=768 -DSQ_C=825 or
// -DSQ_BITS=1024 -DSQ_C=105 (`kv.py`'s primes), with the forms that matter there (FOLD 1, 5, 6, 7, 11, 12 at ILP 1).
//
//   sq gate FOLD ILP K N           N words from a fixed seed, K squarings: "x0 xK" in hex, canonical
//   sq time FOLD ILP K TPB WAVES REPS   timed launches at the occupancy's blocks a SM times WAVES
//   sq probe P W A ITERS WAVES     the pipe probe (below), its digests checked against the host's, then timed
//   sq clock                       the SM clock off a timed clock64 spin
#include <cuda_runtime.h>

#ifndef SQ_BITS
#define SQ_BITS 512
#define SQ_C 569
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sq {

using u32 = uint32_t;
using u64 = uint64_t;
constexpr int NL = SQ_BITS / 32;
constexpr u32 CF = SQ_C;
static_assert(SQ_BITS % 64 == 0 && CF % 4 == 1 && (u64)CF * CF < (1ull << 31), "p = 2^SQ_BITS - SQ_C = 3 mod 4, SQ_C small");

__device__ __forceinline__ void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}

__device__ __forceinline__ void sqr_cols(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
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

// FOLD 0: the red team's sqfloor.cu `fold_cols`, verbatim.
__device__ __forceinline__ void fold_cols(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u64 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += (u64)t[NL + k] * CF + t[k];
    x[k] = (u32)v;
    v >>= 32;
  }
  v *= CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += x[k];
    x[k] = (u32)v;
    v >>= 32;
  }
  if (v) {
    u64 z = CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      z += x[k];
      x[k] = (u32)z;
      z >>= 32;
    }
  }
}

// One PTX instruction each, so a carry chain is a run of them; volatile keeps anything that would clobber the carry
// flag out from between the links (`kv.cu`'s).
__device__ __forceinline__ u32 add_cc(u32 a, u32 b) { u32 r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc_cc(u32 a, u32 b) { u32 r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc(u32 a, u32 b) { u32 r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 madlo_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madclo_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madhi_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madchi_cc(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ u32 madchi(u32 a, u32 b, u32 c) {
  u32 r; asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}

// x += a (a < 2^32) into NL limbs; returns the carry out of 2^512. The carry leaves limb 0 with probability a / 2^32
// (every caller's a is below c^2 + c < 2^20), so the rest of the chain is a branch.
__device__ __forceinline__ u32 add_small(u32 (&x)[NL], u32 a) {
  x[0] += a;
  if (x[0] >= a) return 0u;
  x[1] = add_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
  return addc(0u, 0u);
}

// FOLD 1: `kv.cu`'s fold. x = lo + c hi: the low products' chain, then the high ones' a limb up, whose carry word
// h <= c; then h c (< 2^20) into the low limbs, and c once more if that carried out. x < 2^512.
__device__ __forceinline__ void fold_ptx(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  x[0] = madlo_cc(t[NL], CF, t[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = madclo_cc(t[NL + k], CF, t[k]);
  u32 h = addc(0u, 0u);
  x[1] = madhi_cc(t[NL], CF, x[1]);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = madchi_cc(t[NL + k - 1], CF, x[k]);
  h = madchi(t[2 * NL - 1], CF, h);
  if (add_small(x, h * CF)) add_small(x, CF);
}

// v 569 = v 2^9 + v 2^5 + v 2^4 + v 2^3 + v as (lo, hi), hi < 2^10, in shifts and adds only (no IMAD). The 64-bit
// sum of five shifted copies of v; volatile keeps ptxas from folding it back into a multiply.
__device__ __forceinline__ void mul569(u32 v, u32 &lo, u32 &hi) {
  u32 a, b;
  asm volatile(
      "{\n\t.reg .u32 s9l, s9h, s5l, s5h, s4l, s4h, s3l, s3h;\n\t"
      "shl.b32 s9l, %2, 9;\n\tshr.b32 s9h, %2, 23;\n\t"
      "shl.b32 s5l, %2, 5;\n\tshr.b32 s5h, %2, 27;\n\t"
      "shl.b32 s4l, %2, 4;\n\tshr.b32 s4h, %2, 28;\n\t"
      "shl.b32 s3l, %2, 3;\n\tshr.b32 s3h, %2, 29;\n\t"
      "add.cc.u32 %0, s9l, s5l;\n\taddc.u32 %1, s9h, s5h;\n\t"
      "add.cc.u32 %0, %0, s4l;\n\taddc.u32 %1, %1, s4h;\n\t"
      "add.cc.u32 %0, %0, s3l;\n\taddc.u32 %1, %1, s3h;\n\t"
      "add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;\n\t}"
      : "=r"(a), "=r"(b) : "r"(v));
  lo = a;
  hi = b;
}

// FOLD 2: FOLD 1's arithmetic with the multiplies by c as shifts: x = lo + sum_k (P_k lo) 2^(32k) + (P_k hi) 2^(32(k+1)),
// P_k = t[NL + k] c. The low words' chain, then the high words' a limb up, then h c and c as in FOLD 1.
__device__ __forceinline__ void fold_shift(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u32 pl[NL], ph[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) mul569(t[NL + k], pl[k], ph[k]);
  x[0] = add_cc(t[0], pl[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = addc_cc(t[k], pl[k]);
  u32 h = addc(0u, 0u);
  x[1] = add_cc(x[1], ph[0]);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], ph[k - 1]);
  h = addc(h, ph[NL - 1]);
  u32 hl, hh;
  mul569(h, hl, hh);
  if (add_small(x, hl)) add_small(x, CF);
}

// x += lo + hi 2^32 into NL limbs; returns the carry out of 2^512. Past limb 1 the carry is a branch.
__device__ __forceinline__ u32 add_small2(u32 (&x)[NL], u32 lo, u32 hi) {
  x[0] = add_cc(x[0], lo);
  x[1] = addc_cc(x[1], hi);
  if (!addc(0u, 0u)) return 0u;
  x[2] = add_cc(x[2], 1u);
#pragma unroll
  for (int k = 3; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
  return addc(0u, 0u);
}

// FOLD 3: the fold inside the column square. The high columns first, alone (no carry in from column NL - 1), give
// h = (their sum) < 2^512 since h 2^512 <= a^2; then each low column m takes h_m c as one more product after its
// doubling, so x = lo + c h comes out of the columns' own carries. Column NL - 1 carries out e = (c1, c0) at 2^512:
// the low columns' raw sums, below 17 2^544 + c 2^512, so e < 2^37 and e c < 2^47 folds as two words.
__device__ __forceinline__ void sqr_fused(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 h[NL];
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = NL; m < 2 * NL - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = m - NL + 1; 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    add3(r, c0, c1);
    h[m - NL] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  h[NL - 1] = c0;
  c0 = 0;
  c1 = 0;
#pragma unroll
  for (int m = 0; m < NL; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = 0; 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    r[2] = (r[2] << 1) | (r[1] >> 31);
    r[1] = (r[1] << 1) | (r[0] >> 31);
    r[0] <<= 1;
    if ((m & 1) == 0) mac(r, a[m / 2], a[m / 2]);
    mac(r, h[m], CF);
    add3(r, c0, c1);
    x[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  if (add_small2(x, c0 * CF, __umulhi(c0, CF) + c1 * CF)) add_small(x, CF);
}

// FOLD 5: the square with one doubling. The off-diagonal products alone, column by column, each column's accumulator
// starting at the previous one's carry (so a column ends in no add), give S = sum_{i<j} a_i a_j 2^(32(i+j)) < 2^1023;
// then y = 2 S + sum a_i^2 2^(64 i) in one carry chain (a funnel shift a word for the doubling, the squares as
// multiply-adds), then FOLD 1's fold.
template <int N>
__device__ __forceinline__ void sqr_dbl_n(const u32 (&a)[N], u32 (&y)[2 * N]) {
  u32 s[2 * N];
  u32 c0 = 0, c1 = 0;
  s[0] = 0;
#pragma unroll
  for (int m = 1; m < 2 * N - 2; ++m) {
    u32 r[3] = {c0, c1, 0};
#pragma unroll
    for (int i = (m < N ? 0 : m - N + 1); 2 * i < m; ++i) mac(r, a[i], a[m - i]);
    s[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  s[2 * N - 2] = c0;
  s[2 * N - 1] = c1;
  u32 d[2 * N];
  d[0] = 0;
#pragma unroll
  for (int k = 1; k < 2 * N; ++k) d[k] = __funnelshift_l(s[k - 1], s[k], 1);
  y[0] = madlo_cc(a[0], a[0], d[0]);
  y[1] = madchi_cc(a[0], a[0], d[1]);
#pragma unroll
  for (int i = 1; i < N; ++i) {
    y[2 * i] = madclo_cc(a[i], a[i], d[2 * i]);
    y[2 * i + 1] = madchi_cc(a[i], a[i], d[2 * i + 1]);
  }
}

__device__ __forceinline__ void sqr_dbl(const u32 (&a)[NL], u32 (&y)[2 * NL]) { sqr_dbl_n<NL>(a, y); }

// FOLD 7's fold: x_k = t_k + c t_(NL+k) + the previous limb's high word, each limb one 64-bit multiply-add (one
// IMAD.WIDE in place of FOLD 1's IMAD and IMAD.HI) and a two-word add; the last high word h <= c folds as in FOLD 1.
__device__ __forceinline__ void fold_wide(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u32 h = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    const u64 p = (u64)t[NL + k] * CF + t[k];
    const u32 lo = (u32)p, hi = (u32)(p >> 32);
    x[k] = add_cc(lo, h);
    h = addc(hi, 0u);
  }
  if (add_small(x, h * CF)) add_small(x, CF);
}

template <int N>
__device__ __forceinline__ void sqr_cols_n(const u32 (&a)[N], u32 (&t)[2 * N]) {
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

__device__ __forceinline__ u32 sub_cc(u32 a, u32 b) { u32 r; asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc_cc(u32 a, u32 b) { u32 r; asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc(u32 a, u32 b) { u32 r; asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }

// FOLD 4: one level of Karatsuba (the red team's `sqr_kara` at 512 bits): a = a1 2^256 + a0, a^2 = C 2^512 +
// (A + C - D) 2^256 + A with A = a0^2, C = a1^2, D = (a0 - a1)^2, three 8-limb column squares (108 products in place of
// 136), then FOLD 1's fold. M = A + C - D = 2 a0 a1 < 2^513: 16 limbs and a top bit m. Over N limbs, LEV levels deep:
// each half's square is LEV - 1 levels of the same, down to FOLD 5's (DBL) or the plain column square.
template <int N, bool DBL, int LEV>
__device__ __forceinline__ void sqr_kara_n(const u32 (&a)[N], u32 (&t)[2 * N]) {
  constexpr int H = N / 2;
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
  if constexpr (LEV > 1) {
    sqr_kara_n<H, DBL, LEV - 1>(x0, A);
    sqr_kara_n<H, DBL, LEV - 1>(x1, C);
    sqr_kara_n<H, DBL, LEV - 1>(d, D);
  } else if constexpr (DBL) {
    sqr_dbl_n<H>(x0, A);
    sqr_dbl_n<H>(x1, C);
    sqr_dbl_n<H>(d, D);
  } else {
    sqr_cols_n<H>(x0, A);
    sqr_cols_n<H>(x1, C);
    sqr_cols_n<H>(d, D);
  }
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

template <bool DBL, int LEV = 1>
__device__ __forceinline__ void sqr_kara(const u32 (&a)[NL], u32 (&t)[2 * NL]) { sqr_kara_n<NL, DBL, LEV>(a, t); }

template <int FOLD>
__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  if (FOLD == 3) {
    sqr_fused(a, x);
    return;
  }
  u32 t[2 * NL];
  if (FOLD == 4) sqr_kara<false>(a, t);
  else if (FOLD == 6 || FOLD == 8 || FOLD == 9) sqr_kara<true>(a, t);
  else if (FOLD == 11) sqr_kara<true, 2>(a, t);
  else if (FOLD == 12) sqr_kara<true, 3>(a, t);
  else if (FOLD == 5 || FOLD == 7 || FOLD == 10) sqr_dbl(a, t);
  else sqr_cols(a, t);
  if (FOLD == 0) fold_cols(t, x);
  else if (FOLD == 2) fold_shift(t, x);
  else if (FOLD == 7) fold_wide(t, x);
  else fold_ptx(t, x);
}

template <int FOLD, int ILP>
__device__ __forceinline__ void chain_body(u32 *xs, int n, int K) {
  const int w0 = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;
  if (w0 >= n) return;
  u32 x[ILP][NL];
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int k = 0; k < NL; ++k) x[j][k] = xs[(size_t)k * n + w0 + j];
#pragma unroll 1
  for (int r = 0; r < K; ++r) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      u32 y[NL];
      sqr<FOLD>(x[j], y);
#pragma unroll
      for (int k = 0; k < NL; ++k) x[j][k] = y[k];
    }
  }
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int k = 0; k < NL; ++k) xs[(size_t)k * n + w0 + j] = x[j][k];
}

// ptxas allocates differently under an explicit minimum of blocks, even 1, so only FOLD 8-10 name one.
template <int FOLD, int ILP>
__global__ void __launch_bounds__(128) chain(u32 *xs, int n, int K) { chain_body<FOLD, ILP>(xs, n, K); }

template <int FOLD, int ILP, int MIN_BLOCKS>
__global__ void __launch_bounds__(128, MIN_BLOCKS) chain_held(u32 *xs, int n, int K) { chain_body<FOLD, ILP>(xs, n, K); }

// The pipe probe: what fraction of the FMA-heavy pipe a warp keeps busy with the column square's instructions when no
// carry chain orders them. PROBE 0: W 64-bit multiply-adds (`mad.wide.u32`) and A 32-bit adds an iteration over CH
// independent accumulators each. PROBE 1: W of the square's `mac` (IMAD.WIDE.U32 with its carry into a third word).
constexpr int CH = 8;

template <int PROBE, int W, int A>
__host__ __device__ inline u64 probe_run(u32 tid, int iters) {
  u64 acc[CH];
  u32 m[CH], s[CH], r2[CH];
  for (int c = 0; c < CH; ++c) {
    const u32 v = tid * 0x9E3779B9u + 0x85EBCA6Bu * (c + 1);
    acc[c] = v | (u64)(v * 0xC2B2AE35u) << 32;
    m[c] = v * 0x27D4EB2Fu;
    s[c] = v ^ 0x165667B1u;
    r2[c] = 0;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < W; ++i) {
      const int c = i % CH;
      const u32 b = (u32)acc[(i + 3) % CH], k = m[(i + 5) % CH];
#ifdef __CUDA_ARCH__
      if (PROBE == 0) {
        asm volatile("mad.wide.u32 %0, %1, %2, %0;" : "+l"(acc[c]) : "r"(b), "r"(k));
      } else {
        u32 lo = (u32)acc[c], hi = (u32)(acc[c] >> 32);
        asm volatile("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
                     : "+r"(lo), "+r"(hi), "+r"(r2[c]) : "r"(b), "r"(k));
        acc[c] = lo | (u64)hi << 32;
      }
#else
      const u64 p = (u64)b * k, v = acc[c] + p;
      if (PROBE == 1) r2[c] += v < p;
      acc[c] = v;
#endif
    }
#pragma unroll
    for (int i = 0; i < A; ++i) {
#ifdef __CUDA_ARCH__
      asm volatile("add.u32 %0, %0, %1;" : "+r"(s[i % CH]) : "r"(m[(i + 1) % CH]));
#else
      s[i % CH] += m[(i + 1) % CH];
#endif
    }
  }
  u64 h = 0;
  for (int c = 0; c < CH; ++c) h = h * 0x100000001B3ull ^ acc[c] ^ (u64)s[c] << 17 ^ (u64)r2[c] << 41;
  return h;
}

template <int PROBE, int W, int A>
__global__ void __launch_bounds__(128) probe(u64 *out, int iters) {
  const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = probe_run<PROBE, W, A>(tid, iters);
}

__global__ void spin(long long cycles, long long *out) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
  *out = clock64() - t0;
}

}  // namespace sq

static uint64_t splitmix(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      exit(3);                                                                        \
    }                                                                                 \
  } while (0)

static void fill(std::vector<uint32_t> &h, int n) {
  uint64_t s = 0x5eed512ull;
  h.assign((size_t)sq::NL * n, 0);
  for (int w = 0; w < n; ++w)
    for (int k = 0; k < sq::NL; ++k) h[(size_t)k * n + w] = (uint32_t)splitmix(s);
}

using Kernel = void (*)(uint32_t *, int, int);
static Kernel kernel(int fold, int ilp) {
#define K_(F, I) if (fold == F && ilp == I) return sq::chain<F, I>;
#if SQ_BITS == 512
  K_(0, 1) K_(0, 2) K_(1, 1) K_(1, 2) K_(1, 4) K_(2, 1) K_(2, 2) K_(2, 4) K_(3, 1) K_(3, 2) K_(4, 1) K_(4, 2) K_(5, 1) K_(5, 2) K_(6, 1) K_(7, 1) K_(7, 2)
#define H_(F, I, B) if (fold == F && ilp == I) return sq::chain_held<F, I, B>;
  H_(8, 1, 6) H_(9, 1, 7) H_(10, 1, 8)
#undef H_
#endif
  K_(1, 1) K_(5, 1) K_(6, 1) K_(7, 1) K_(11, 1) K_(12, 1)
#undef K_
  fprintf(stderr, "no kernel for fold %d ilp %d\n", fold, ilp);
  exit(2);
}

using Probe = void (*)(uint64_t *, int);
using ProbeHost = uint64_t (*)(uint32_t, int);
static void probe_kernel(int p, int w, int a, Probe &k, ProbeHost &h) {
#define P_(P, W, A) if (p == P && w == W && a == A) { k = sq::probe<P, W, A>; h = sq::probe_run<P, W, A>; return; }
  P_(0, 16, 0) P_(0, 16, 16) P_(0, 16, 32) P_(1, 16, 0) P_(1, 16, 16)
#undef P_
  fprintf(stderr, "no probe %d %d %d\n", p, w, a);
  exit(2);
}

static void launch(int fold, int ilp, uint32_t *d, int n, int K, int tpb) {
  const int threads = n / ilp;
  kernel(fold, ilp)<<<(threads + tpb - 1) / tpb, tpb>>>(d, n, K);
  CK(cudaGetLastError());
}

// x mod p for x < 2^512, as hex (most significant limb first).
static void print_canon(const std::vector<uint32_t> &h, int n, int w) {
  uint32_t x[sq::NL], z[sq::NL];
  for (int k = 0; k < sq::NL; ++k) x[k] = h[(size_t)k * n + w];
  uint64_t v = sq::CF;
  for (int k = 0; k < sq::NL; ++k) {
    v += x[k];
    z[k] = (uint32_t)v;
    v >>= 32;
  }
  const uint32_t *y = v ? z : x;
  for (int k = sq::NL - 1; k >= 0; --k) printf("%08x", y[k]);
}

#ifndef SQ_NO_MAIN
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  if (!strcmp(argv[1], "clock")) {
    long long *d;
    CK(cudaMalloc(&d, sizeof(long long)));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    sq::spin<<<1, 1>>>(1LL << 26, d);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    sq::spin<<<1, 1>>>(1LL << 30, d);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    long long cyc;
    CK(cudaMemcpy(&cyc, d, sizeof cyc, cudaMemcpyDeviceToHost));
    printf("{\"clock_mhz\": %.1f}\n", cyc / (ms * 1e3));
    return 0;
  }
  if (!strcmp(argv[1], "gate") && argc == 6) {
    const int fold = atoi(argv[2]), ilp = atoi(argv[3]), K = atoi(argv[4]), n = atoi(argv[5]);
    std::vector<uint32_t> h0, h;
    fill(h0, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h0.size() * 4));
    CK(cudaMemcpy(d, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice));
    launch(fold, ilp, d, n, K, 128);
    CK(cudaDeviceSynchronize());
    h.resize(h0.size());
    CK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
    for (int w = 0; w < n; ++w) {
      printf("%d %d %d %u ", fold * 10 + ilp, K, SQ_BITS, sq::CF);
      print_canon(h0, n, w);
      printf(" ");
      print_canon(h, n, w);
      printf("\n");
    }
    return 0;
  }
  if (!strcmp(argv[1], "probe") && argc == 7) {
    const int p = atoi(argv[2]), w = atoi(argv[3]), a = atoi(argv[4]), iters = atoi(argv[5]), waves = atoi(argv[6]);
    Probe k;
    ProbeHost host;
    probe_kernel(p, w, a, k, host);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, 128, 0));
    const int n = prop.multiProcessorCount * per_sm * 128 * waves;
    uint64_t *d;
    CK(cudaMalloc(&d, (size_t)n * 8));
    k<<<n / 128, 128>>>(d, 3);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> got(n);
    CK(cudaMemcpy(got.data(), d, (size_t)n * 8, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int t = 0; t < n; t += 997) bad += got[t] != host((uint32_t)t, 3);
    if (bad) {
      printf("{\"probe\": %d, \"w\": %d, \"a\": %d, \"gate_bad\": %d}\n", p, w, a, bad);
      return 1;
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    k<<<n / 128, 128>>>(d, iters / 10 + 1);
    CK(cudaEventRecord(e0));
    k<<<n / 128, 128>>>(d, iters);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double warp_iters_per_s_sm = (double)n / 32 * iters / (ms * 1e-3) / prop.multiProcessorCount;
    printf("{\"probe\": %d, \"w\": %d, \"a\": %d, \"iters\": %d, \"blocks_per_sm\": %d, \"threads\": %d, \"gate_bad\": %d, "
           "\"wide_warp_inst_per_s_sm\": %.4e, \"add_warp_inst_per_s_sm\": %.4e}\n",
           p, w, a, iters, per_sm, n, bad, warp_iters_per_s_sm * w, warp_iters_per_s_sm * a);
    return bad ? 1 : 0;
  }
  if (!strcmp(argv[1], "time") && argc == 8) {
    const int fold = atoi(argv[2]), ilp = atoi(argv[3]), K = atoi(argv[4]), tpb = atoi(argv[5]),
              waves = atoi(argv[6]), reps = atoi(argv[7]);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel(fold, ilp), tpb, 0));
    cudaFuncAttributes attr;
    CK(cudaFuncGetAttributes(&attr, kernel(fold, ilp)));
    const int n = prop.multiProcessorCount * per_sm * tpb * waves * ilp;
    std::vector<uint32_t> h;
    fill(h, n);
    uint32_t *d;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    launch(fold, ilp, d, n, K / 10 + 1, tpb);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> rate;
    for (int r = 0; r < reps; ++r) {
      CK(cudaEventRecord(a));
      launch(fold, ilp, d, n, K, tpb);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      rate.push_back((double)n * K / (ms * 1e-3));
    }
    std::sort(rate.begin(), rate.end());
    printf("{\"bits\": %d, \"fold\": %d, \"ilp\": %d, \"K\": %d, \"tpb\": %d, \"waves\": %d, \"blocks_per_sm\": %d, "
           "\"regs\": %d, \"words\": %d, \"sms\": %d, \"sq_per_s_max\": %.4e, \"sq_per_s_median\": %.4e}\n",
           SQ_BITS, fold, ilp, K, tpb, waves, per_sm, attr.numRegs, n, prop.multiProcessorCount, rate.back(),
           rate[rate.size() / 2]);
    return 0;
  }
  return 2;
}
#endif
