// The KV cache's codec on the RTX PRO 6000 (sm_120): Sloth over one wide word, `kv.py` the reference. One build per
// field, KV_FIELD = 512, 768, 1024, 1280 or 1536 (p = 2^n - c, n = 32 NL), any k up to KMAX at run time.
//
// A thread a word, NL 32-bit limbs: the square column by column, each unordered pair once and doubled
// (`benchmarks/codec_tensor/wt.cu`'s), then 2^n = c folded twice, a lazy value below 2^n; canonical (< p) only where a
// comparison, the sign rule or an addition reads it. Past 768 bits (STREAM) each high column is folded into the low
// limbs as it is produced, so the 2 NL-limb product is never held, with the same result. The root is `kv.py`'s chain:
// x_M = a^(2^M - 1) by M's bits from the top, then LOW_BITS squarings with a multiply by a at LOW's set bits.
//
// Kernels (extern "C" launchers at the end): encode and decode of a flat store of words (NL limbs each, contiguous), the
// decode written out or XORed into a checksum (the read-and-decode rate); the read of a store; compute-only chains for the
// squaring (ILP 1 or 2 words a thread), decode and encode rates; a spin kernel to queue timed launches behind.
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef KV_FIELD
#define KV_FIELD 512
#endif

namespace kv {

using u32 = uint32_t;
using u64 = uint64_t;

#if KV_FIELD == 512
constexpr int NBITS = 512;
constexpr u32 CF = 569;
#elif KV_FIELD == 768
constexpr int NBITS = 768;
constexpr u32 CF = 825;
#elif KV_FIELD == 1024
constexpr int NBITS = 1024;
constexpr u32 CF = 105;
#elif KV_FIELD == 1280
constexpr int NBITS = 1280;
constexpr u32 CF = 1665;
#elif KV_FIELD == 1536
constexpr int NBITS = 1536;
constexpr u32 CF = 3453;
#else
#error "KV_FIELD is one of kv.FIELDS"
#endif
constexpr int NL = NBITS / 32;
constexpr int KMAX = 8;
constexpr int bitlen(int x) { return x ? 1 + bitlen(x >> 1) : 0; }
constexpr int LOW_MIN = 8;
constexpr int LOW_BITS = bitlen((CF - 1) / 4) > LOW_MIN ? bitlen((CF - 1) / 4) : LOW_MIN;
constexpr int CHAIN_M = NBITS - 2 - LOW_BITS;
constexpr u32 LOW = (1u << LOW_BITS) - (CF - 1) / 4;
constexpr int MBITS = bitlen(CHAIN_M);
constexpr int WPB = 128;
constexpr bool STREAM = NL > 24;
#ifndef KV_KARATSUBA
#define KV_KARATSUBA 0
#endif
constexpr bool KARATSUBA = KV_KARATSUBA;
static_assert(NBITS % 32 == 0 && NL % 4 == 0 && CF % 4 == 1 && CF < (1u << 16) && (CF - 1) / 4 < (1u << LOW_BITS),
              "p = 2^n - c");

__constant__ u32 kC[KMAX][NL];
__constant__ u32 kTau[NL];

__device__ __forceinline__ void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}

__device__ __forceinline__ void sqr_full(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
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

__device__ __forceinline__ void mul_full(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&t)[2 * NL]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int m = 0; m < 2 * NL - 1; ++m) {
    u32 r[3] = {0, 0, 0};
#pragma unroll
    for (int i = (m < NL ? 0 : m - NL + 1); i <= (m < NL ? m : NL - 1); ++i) mac(r, a[i], b[m - i]);
    add3(r, c0, c1);
    t[m] = r[0];
    c0 = r[1];
    c1 = r[2];
  }
  t[2 * NL - 1] = c0;
}

// One PTX instruction each, so a carry chain is a run of them (`benchmarks/codec_tensor/wt.cu`'s): volatile keeps the
// compiler from moving anything between the links that would clobber the carry flag.
__device__ __forceinline__ u32 add_cc(u32 a, u32 b) { u32 r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc_cc(u32 a, u32 b) { u32 r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc(u32 a, u32 b) { u32 r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 sub_cc(u32 a, u32 b) { u32 r; asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc_cc(u32 a, u32 b) { u32 r; asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc(u32 a, u32 b) { u32 r; asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
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

// x += a (a < 2^32) into NL limbs; returns the carry out of 2^n. The carry leaves limb 0 with probability a / 2^32
// (every caller's a is at most c^2 + c < 2^24), so the rest of the chain is a branch.
__device__ __forceinline__ u32 add_small(u32 (&x)[NL], u32 a) {
  x[0] += a;
  if (x[0] >= a) return 0u;
  x[1] = add_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
  return addc(0u, 0u);
}

// x = t mod p lazily for t < 2^(2n): lo + c hi (the low products' chain, then the high ones' a limb up), whose carry
// word h <= c; then h c (< 2^20) into the low limbs, and c once more if that carried out (the limbs are then below
// 2^20). x < 2^n.
__device__ __forceinline__ void fold(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
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

__device__ __forceinline__ void copy(u32 (&x)[NL], const u32 (&a)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = a[k];
}

// Column m >= NL of a product, v, folded into the low limbs y as c v, the carry word f running up them.
__device__ __forceinline__ void fold_column(u32 (&y)[NL], int j, u32 v, u32 &f) {
  const u64 s = (u64)y[j] + (u64)v * CF + f;
  y[j] = (u32)s;
  f = (u32)(s >> 32);
}

// Column m of the square of a[OFF..OFF + N) before its carry in: each pair once, doubled, and the square term.
template <int N, int OFF, int LEN>
__device__ __forceinline__ void sqr_column(const u32 (&a)[LEN], int m, u32 (&r)[3]) {
  r[0] = r[1] = r[2] = 0;
#pragma unroll
  for (int i = (m < N ? 0 : m - N + 1); 2 * i < m; ++i) mac(r, a[OFF + i], a[OFF + m - i]);
  r[2] = (r[2] << 1) | (r[1] >> 31);
  r[1] = (r[1] << 1) | (r[0] >> 31);
  r[0] <<= 1;
  if ((m & 1) == 0) mac(r, a[OFF + m / 2], a[OFF + m / 2]);
}

// One level of Karatsuba (KV_KARATSUBA=1): a = a0 + a1 B with B = 2^(n/2), so a^2 = L + M B + H B^2 with L = a0^2,
// H = a1^2, M = L + H - D, D = (a0 - a1)^2, and B^2 = c mod p: three half squarings, 3/4 of the products. Their
// columns are made side by side (three carry chains a column), y = L + c H + M's low half B as they come, and M's
// high half, which lands at B^2 and up, folds in as c times it at the end, then y's carry words as in `fold`.
__device__ __forceinline__ void sqr_k1(const u32 (&a)[NL], u32 (&x)[NL]) {
  constexpr int H = NL / 2;
  u32 d[H], bw = 0;
#pragma unroll
  for (int i = 0; i < H; ++i) {
    const u64 s = (u64)a[i] - a[H + i] - bw;
    d[i] = (u32)s;
    bw = (u32)(s >> 63);
  }
  const u32 neg = 0u - bw;
  u32 cy = bw;
#pragma unroll
  for (int i = 0; i < H; ++i) {
    const u64 s = (u64)(d[i] ^ neg) + cy;
    d[i] = (u32)s;
    cy = (u32)(s >> 32);
  }
  u32 y[NL], ml[H], mh[H + 1];
  u32 l0 = 0, l1 = 0, h0 = 0, h1 = 0, e0 = 0, e1 = 0, fy = 0, cp = 0, bd = 0;
#pragma unroll
  for (int m = 0; m < NL; ++m) {
    u32 lv = l0, hv = h0, dv = e0;
    if (m < NL - 1) {
      u32 r[3];
      sqr_column<H, 0>(a, m, r);
      add3(r, l0, l1);
      lv = r[0], l0 = r[1], l1 = r[2];
      sqr_column<H, H>(a, m, r);
      add3(r, h0, h1);
      hv = r[0], h0 = r[1], h1 = r[2];
      sqr_column<H, 0>(d, m, r);
      add3(r, e0, e1);
      dv = r[0], e0 = r[1], e1 = r[2];
    }
    // y_m = l + c h + fy (+ M_(m - H)) and M_m = (l + h + cp) - d - bd, each a carry chain of its own in one block
    u32 yv, fn, mv, cn, bn;
    asm("mad.lo.cc.u32 %0, %2, %3, %4;\n\tmadc.hi.u32 %1, %2, %3, 0;\n\tadd.cc.u32 %0, %0, %5;\n\taddc.u32 %1, %1, 0;"
        : "=r"(yv), "=r"(fn) : "r"(hv), "n"(CF), "r"(lv), "r"(fy));
    if (m >= H) asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, %1, 0;" : "+r"(yv), "+r"(fn) : "r"(ml[m - H]));
    asm("add.cc.u32 %0, %3, %4;\n\taddc.u32 %1, 0, 0;\n\tadd.cc.u32 %0, %0, %5;\n\taddc.u32 %1, %1, 0;\n\t"
        "sub.cc.u32 %0, %0, %6;\n\tsubc.u32 %2, 0, 0;\n\tsub.cc.u32 %0, %0, %7;\n\tsubc.u32 %2, %2, 0;"
        : "=r"(mv), "=r"(cn), "=r"(bn) : "r"(lv), "r"(hv), "r"(cp), "r"(dv), "r"(bd));
    y[m] = yv;
    fy = fn;
    cp = cn;
    bd = bn & 1;
    if (m < H) ml[m] = mv;
    else mh[m - H] = mv;
  }
  mh[H] = cp - bd;
  u32 g = 0;
#pragma unroll
  for (int m = 0; m < NL; ++m) {
    u32 gn;
    if (m <= H)
      asm("mad.lo.cc.u32 %0, %2, %3, %0;\n\tmadc.hi.u32 %1, %2, %3, 0;\n\tadd.cc.u32 %0, %0, %4;\n\taddc.u32 %1, %1, 0;"
          : "+r"(y[m]), "=r"(gn) : "r"(mh[m]), "n"(CF), "r"(g));
    else
      asm("add.cc.u32 %0, %0, %2;\n\taddc.u32 %1, 0, 0;" : "+r"(y[m]), "=r"(gn) : "r"(g));
    g = gn;
  }
  if (add_small(y, (g + fy) * CF)) add_small(y, CF);
  copy(x, y);
}

__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  if constexpr (KARATSUBA) {
    sqr_k1(a, x);
  } else if constexpr (STREAM) {
    u32 y[NL];
    u32 c0 = 0, c1 = 0, f = 0;
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
      if (m < NL) y[m] = r[0];
      else fold_column(y, m - NL, r[0], f);
      c0 = r[1];
      c1 = r[2];
    }
    fold_column(y, NL - 1, c0, f);
    if (add_small(y, f * CF)) add_small(y, CF);
    copy(x, y);
  } else {
    u32 t[2 * NL];
    sqr_full(a, t);
    fold(t, x);
  }
}

__device__ __forceinline__ void mul(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  if constexpr (STREAM) {
    u32 y[NL];
    u32 c0 = 0, c1 = 0, f = 0;
#pragma unroll
    for (int m = 0; m < 2 * NL - 1; ++m) {
      u32 r[3] = {0, 0, 0};
#pragma unroll
      for (int i = (m < NL ? 0 : m - NL + 1); i <= (m < NL ? m : NL - 1); ++i) mac(r, a[i], b[m - i]);
      add3(r, c0, c1);
      if (m < NL) y[m] = r[0];
      else fold_column(y, m - NL, r[0], f);
      c0 = r[1];
      c1 = r[2];
    }
    fold_column(y, NL - 1, c0, f);
    if (add_small(y, f * CF)) add_small(y, CF);
    copy(x, y);
  } else {
    u32 t[2 * NL];
    mul_full(a, b, t);
    fold(t, x);
  }
}

// x mod p for x < 2^n: x - p = x + c - 2^n when that carries out, which needs x's top limb all ones (p's limbs above
// the lowest are, c < 2^32).
__device__ __forceinline__ void canon(u32 (&x)[NL]) {
  if (x[NL - 1] != 0xffffffffu) return;
  u32 z[NL];
  z[0] = add_cc(x[0], CF);
#pragma unroll
  for (int k = 1; k < NL; ++k) z[k] = addc_cc(x[k], 0u);
  const bool ge = addc(0u, 0u) != 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? z[k] : x[k];
}

// x + b mod p, both canonical: s = x + b < 2p. When s carried out of 2^n, s - p = (s - 2^n) + c < p, which carries out
// of limb 0 only rarely (add_small); otherwise s < 2^n and canon takes off p if s >= p.
__device__ __forceinline__ void add_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  x[0] = add_cc(x[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = addc_cc(x[k], b[k]);
  add_small(x, CF & (0u - addc(0u, 0u)));
  canon(x);
}

// x - b mod p, both canonical: on a borrow, r = x - b + 2^n >= c + 1 and x - b + p = r - c, which borrows past limb 0
// only when r's low limb is below c.
__device__ __forceinline__ void sub_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  x[0] = sub_cc(x[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = subc_cc(x[k], b[k]);
  const u32 d = CF & subc(0u, 0u), lo = x[0];
  x[0] = lo - d;
  if (lo >= d) return;
  x[1] = sub_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = subc_cc(x[k], 0u);
}

// s = p - s when on, s canonical (0 becomes p). p's limbs above the lowest are all ones, so p - s complements them
// unless the lowest limb borrows (s[0] > 2^32 - c, rare).
__device__ __forceinline__ void neg_if(u32 (&s)[NL], bool on) {
  if (on && s[0] > 0u - CF) {
    s[0] = sub_cc(0u - CF, s[0]);
#pragma unroll
    for (int k = 1; k < NL - 1; ++k) s[k] = subc_cc(0xffffffffu, s[k]);
    s[NL - 1] = subc(0xffffffffu, s[NL - 1]);
    return;
  }
  const u32 m = 0u - (u32)on;
  s[0] = on ? (0u - CF) - s[0] : s[0];
#pragma unroll
  for (int k = 1; k < NL; ++k) s[k] ^= m;
}

__device__ __forceinline__ bool eq(const u32 (&a)[NL], const u32 (&b)[NL]) {
  u32 d = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) d |= a[k] ^ b[k];
  return d == 0;
}

// a^((p+1)/4), lazily; a < 2^n.
__device__ __forceinline__ void root(const u32 (&a)[NL], u32 (&x)[NL]) {
  copy(x, a);
  int cur = 1;
#pragma unroll 1
  for (int b = MBITS - 2; b >= 0; --b) {
    u32 s[NL];
    copy(s, x);
#pragma unroll 1
    for (int i = 0; i < cur; ++i) sqr(x, x);
    mul(x, s, x);
    cur *= 2;
    if ((CHAIN_M >> b) & 1) {
      sqr(x, x);
      mul(x, a, x);
      ++cur;
    }
  }
#pragma unroll 1
  for (int b = LOW_BITS - 1; b >= 0; --b) {
    sqr(x, x);
    if ((LOW >> b) & 1) mul(x, a, x);
  }
}

__device__ __forceinline__ void rho(u32 (&x)[NL]) {
  u32 s[NL], q[NL];
  root(x, s);
  canon(s);
  sqr(s, q);
  canon(q);
  const bool square = eq(q, x);
  neg_if(s, (s[0] & 1) != (square ? 0u : 1u));
  copy(x, s);
}

// y odd is nonzero, so its square is too.
__device__ __forceinline__ void rho_inv(u32 (&y)[NL]) {
  const bool odd = y[0] & 1;
  u32 q[NL];
  sqr(y, q);
  canon(q);
  neg_if(q, odd);
  copy(y, q);
}

// t(i) = tau i mod p: tau i < 2^(n + 64) (the high word's products only when i >= 2^32), its two top limbs h folded in
// as c h, canonical.
__device__ __forceinline__ void tweak(u64 i, u32 (&t)[NL]) {
  const u32 ilo = (u32)i, ihi = (u32)(i >> 32);
  u32 r[NL + 2];
#pragma unroll
  for (int k = 0; k < NL; ++k) r[k] = kTau[k] * ilo;
  r[1] = madhi_cc(kTau[0], ilo, r[1]);
#pragma unroll
  for (int k = 2; k < NL; ++k) r[k] = madchi_cc(kTau[k - 1], ilo, r[k]);
  r[NL] = madchi(kTau[NL - 1], ilo, 0u);
  r[NL + 1] = 0;
  if (ihi) {
    r[1] = madlo_cc(kTau[0], ihi, r[1]);
#pragma unroll
    for (int k = 1; k < NL; ++k) r[k + 1] = madclo_cc(kTau[k], ihi, r[k + 1]);
    r[NL + 1] = addc(0u, 0u);
    r[2] = madhi_cc(kTau[0], ihi, r[2]);
#pragma unroll
    for (int k = 1; k < NL - 1; ++k) r[k + 2] = madchi_cc(kTau[k], ihi, r[k + 2]);
    r[NL + 1] = madchi(kTau[NL - 1], ihi, r[NL + 1]);
  }
  t[0] = madlo_cc(r[NL], CF, r[0]);
  t[1] = madclo_cc(r[NL + 1], CF, r[1]);
#pragma unroll
  for (int k = 2; k < NL; ++k) t[k] = addc_cc(r[k], 0u);
  u32 co = addc(0u, 0u);
  t[1] = madhi_cc(r[NL], CF, t[1]);
  t[2] = madchi_cc(r[NL + 1], CF, t[2]);
#pragma unroll
  for (int k = 3; k < NL; ++k) t[k] = addc_cc(t[k], 0u);
  co = addc(co, 0u);
  if (co) add_small(t, CF);
  canon(t);
}

__device__ __forceinline__ void round_constant(int j, u32 (&c)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) c[k] = kC[j][k];
}

// Past 768 bits the tweak is recomputed each round, so it isn't held through the root.
__device__ __forceinline__ void encode_word(u32 (&x)[NL], u64 i, int k) {
  u32 t[NL];
  if (!STREAM) tweak(i, t);
#pragma unroll 1
  for (int j = 0; j < k; ++j) {
    u32 c[NL];
    round_constant(j, c);
    add_mod(x, c);
    if (STREAM) tweak(i, t);
    add_mod(x, t);
    rho(x);
  }
}

// The decode given its word's tweak t = t(i): loops that step i by a constant step t by one add_mod.
__device__ __forceinline__ void decode_word_t(u32 (&x)[NL], const u32 (&t)[NL], int k) {
#pragma unroll 1
  for (int j = k - 1; j >= 0; --j) {
    u32 c[NL];
    round_constant(j, c);
    rho_inv(x);
    sub_mod(x, c);
    sub_mod(x, t);
  }
}

__device__ __forceinline__ void decode_word(u32 (&x)[NL], u64 i, int k) {
  u32 t[NL];
  tweak(i, t);
  decode_word_t(x, t, k);
}

// t += tau d mod p, tau d read from shared memory (every thread's step the same).
__device__ __forceinline__ void tweak_step(u32 (&t)[NL], const u32 *td) {
  u32 s[NL];
#pragma unroll
  for (int l = 0; l < NL; ++l) s[l] = td[l];
  add_mod(t, s);
}

__device__ __forceinline__ void load(const u32 *__restrict__ p, u32 (&x)[NL]) {
  const uint4 *q = reinterpret_cast<const uint4 *>(p);
#pragma unroll
  for (int k = 0; k < NL / 4; ++k) {
    const uint4 v = q[k];
    x[4 * k] = v.x;
    x[4 * k + 1] = v.y;
    x[4 * k + 2] = v.z;
    x[4 * k + 3] = v.w;
  }
}

__device__ __forceinline__ void store(u32 *__restrict__ p, const u32 (&x)[NL]) {
  uint4 *q = reinterpret_cast<uint4 *>(p);
#pragma unroll
  for (int k = 0; k < NL / 4; ++k) q[k] = make_uint4(x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]);
}

__device__ __forceinline__ void xor_out(u32 acc, unsigned long long *xr) {
#pragma unroll
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) atomicXor(xr, (unsigned long long)acc);
}

__global__ void __launch_bounds__(WPB) encode_kernel(const u32 *__restrict__ in, u32 *__restrict__ out, u64 n, u64 base,
                                                     int k) {
  for (u64 w = (u64)blockIdx.x * WPB + threadIdx.x; w < n; w += (u64)gridDim.x * WPB) {
    u32 x[NL];
    load(in + w * NL, x);
    encode_word(x, base + w, k);
    store(out + w * NL, x);
  }
}

// out null: the decoded words XORed into xr (the read-and-decode rate). The grid-stride loop steps each thread's tweak.
__global__ void __launch_bounds__(WPB) decode_kernel(const u32 *__restrict__ in, u32 *__restrict__ out, u64 n, u64 base,
                                                     int k, unsigned long long *xr) {
  __shared__ u32 td[NL];
  if (threadIdx.x == 0) {
    u32 s[NL];
    tweak((u64)gridDim.x * WPB, s);
#pragma unroll
    for (int l = 0; l < NL; ++l) td[l] = s[l];
  }
  __syncthreads();
  u32 acc = 0, t[NL];
  const u64 w0 = (u64)blockIdx.x * WPB + threadIdx.x;
  if (w0 < n) tweak(base + w0, t);
  for (u64 w = w0; w < n; w += (u64)gridDim.x * WPB) {
    u32 x[NL];
    load(in + w * NL, x);
    if (w != w0) tweak_step(t, td);
    decode_word_t(x, t, k);
    if (out) {
      store(out + w * NL, x);
    } else {
#pragma unroll
      for (int l = 0; l < NL; ++l) acc ^= x[l];
    }
  }
  if (!out) xor_out(acc, xr);
}

// The field's primitives a word a thread, for the gate at their rare carries: op 0 add_mod(a, b), 1 sub_mod(a, b),
// 2 neg_if(a, b odd), 3 canon(a), 4 sqr(a).
__global__ void __launch_bounds__(WPB) prim_kernel(int op, const u32 *__restrict__ a, const u32 *__restrict__ b,
                                                   u32 *__restrict__ out, int n) {
  const int w = blockIdx.x * WPB + threadIdx.x;
  if (w >= n) return;
  u32 x[NL], y[NL];
  load(a + (u64)w * NL, x);
  load(b + (u64)w * NL, y);
  if (op == 0) add_mod(x, y);
  else if (op == 1) sub_mod(x, y);
  else if (op == 2) neg_if(x, y[0] & 1);
  else if (op == 3) canon(x);
  else {
    sqr(x, y);
    copy(x, y);
  }
  store(out + (u64)w * NL, x);
}

__global__ void __launch_bounds__(256) read_kernel(const uint4 *__restrict__ s, u64 n16, unsigned long long *xr) {
  u32 acc = 0;
  for (u64 q = (u64)blockIdx.x * 256 + threadIdx.x; q < n16; q += (u64)gridDim.x * 256) {
    const uint4 v = __ldcs(s + q);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  xor_out(acc, xr);
}

// what 0: `steps` squarings of ILP words a thread; 1: `steps` decodes of k rounds; 2: `steps` encodes of k rounds, each
// step's input the previous output and the index advancing. The words start canonical (top limb below p's).
template <int WHAT, int ILP>
__global__ void __launch_bounds__(WPB) chain_kernel(int k, u64 steps, unsigned long long *xr) {
  const u64 gid = (u64)blockIdx.x * WPB + threadIdx.x;
  u32 x[ILP][NL];
#pragma unroll
  for (int w = 0; w < ILP; ++w) {
    u32 h = (u32)(gid * 2654435761u) ^ (0x9e3779b9u * (w + 1));
#pragma unroll
    for (int l = 0; l < NL; ++l) {
      h = h * 1664525u + 1013904223u;
      x[w][l] = h;
    }
    x[w][NL - 1] &= 0x7fffffffu;
  }
  if constexpr (WHAT == 0) {
#pragma unroll 1
    for (u64 s = 0; s < steps; ++s) {
#pragma unroll
      for (int w = 0; w < ILP; ++w) sqr(x[w], x[w]);
    }
  } else if constexpr (WHAT == 1) {
#pragma unroll 1
    for (u64 s = 0; s < steps; ++s) decode_word(x[0], gid + s, k);
  } else {
#pragma unroll 1
    for (u64 s = 0; s < steps; ++s) encode_word(x[0], gid + s, k);
  }
  u32 acc = 0;
#pragma unroll
  for (int w = 0; w < ILP; ++w) {
#pragma unroll
    for (int l = 0; l < NL; ++l) acc ^= x[w][l];
  }
  xor_out(acc, xr);
}

__global__ void spin_kernel(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

// Decode attention at Llama-3.1-70B's shape: one query token a sequence, 64 query heads in 8 groups sharing a KV head
// of 128 dims, over the KV cache in kv.py's token layout: token slot s = seq * ctx + t at layer l of `layers` is one
// token-layer of K then V (4,096 bytes), plain, or encoded as TLW words at word index (s * layers + l) * TLW + w (kv.py's
// with layers = 80). A CTA is one split of a sequence's tokens, its warp g KV head g and its 8 query heads, each lane 4
// of the 128 dims. Tiles of TT tokens are copied into shared memory a tile ahead (cp.async); the encoded form decodes a
// tile's words in place there, a thread a word, and then runs the plain form's arithmetic on the payloads, so the two
// agree bit for bit. q is bf16 (vLLM's), pre-scaled by log2(e)/sqrt(128), so scores are in base 2; each split leaves its
// unnormalized output, max and sum, which attn_combine merges. Shared memory stays under half an SM's 100 KB, so two
// CTAs fit an SM. Past 768 bits (STREAM) the decode's registers allow one, and the encoded form copies and decodes DT = 2
// tiles at a time (1,536 bits' 22 words a token-layer would leave most of a tile's threads idle), each tile then
// folded in turn as in the plain form.
#ifndef KV_QHEADS
#define KV_QHEADS 64
#endif
constexpr int HEADS = KV_QHEADS, KV_HEADS = 8, GROUP = HEADS / KV_HEADS, HEAD_DIM = 128;
constexpr int TL_BYTES = 2 * KV_HEADS * HEAD_DIM * 2;
constexpr int WB = NBITS / 8;
constexpr int TLW = (TL_BYTES + WB - 1) / WB;
constexpr int TLE = TLW * WB;
constexpr int TT = 32 / GROUP;
constexpr int AT = 32 * KV_HEADS;
constexpr int PART = HEADS * (HEAD_DIM + 2);
static_assert(GROUP * TT == 32 && GROUP <= 8 && TLE % 16 == 0 && WB % 16 == 0,
              "a warp's scores, one a lane; 16-byte copies");

template <bool ENC>
__host__ __device__ constexpr int attn_dt() { return ENC && STREAM ? 2 : 1; }
template <bool ENC>
constexpr int attn_smem() {
  return 2 * attn_dt<ENC>() * TT * (ENC ? TLE : TL_BYTES) + HEADS * HEAD_DIM * 2;
}
// Two CTAs an SM where their tiles fit (70B's 64 query heads, to 768 bits), else one (8B's 32 take tiles of 8 tokens).
constexpr int ATTN_MINB = !STREAM && attn_smem<true>() + 1024 <= 50 * 1024 ? 2 : 1;
static_assert(attn_smem<true>() + 1024 <= 99 * 1024, "a CTA's tiles fit an SM");

__device__ __forceinline__ void bf16x4(const uint8_t *p, float (&v)[4]) {
  const uint2 x = *reinterpret_cast<const uint2 *>(p);
  v[0] = __uint_as_float(x.x << 16);
  v[1] = __uint_as_float(x.x & 0xffff0000u);
  v[2] = __uint_as_float(x.y << 16);
  v[3] = __uint_as_float(x.y & 0xffff0000u);
}

__device__ __forceinline__ void copy16(void *dst, const void *src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
}

// One tile of nt tokens (token t's K at src + t * RAW, its V TL_BYTES / 2 on) folded into warp g's 8 heads: scores in
// base 2, the online softmax's max and sum, and the unnormalized output, a lane 4 of the 128 dims.
template <int RAW>
__device__ __forceinline__ void attn_tile(const uint8_t *src, const uint8_t *qs, int g, int lane, int nt,
                                          float (&acc)[GROUP][4], float &m_own, float &l_own) {
  float qf[GROUP][4];
#pragma unroll
  for (int h = 0; h < GROUP; ++h) bf16x4(qs + ((g * GROUP + h) * HEAD_DIM + lane * 4) * 2, qf[h]);
  float s[GROUP * TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    float kk[4];
    bf16x4(src + t * RAW + g * HEAD_DIM * 2 + lane * 8, kk);
#pragma unroll
    for (int h = 0; h < GROUP; ++h)
      s[h * TT + t] = __fmaf_rn(qf[h][3], kk[3],
                                __fmaf_rn(qf[h][2], kk[2], __fmaf_rn(qf[h][1], kk[1], __fmul_rn(qf[h][0], kk[0]))));
  }
  // reduce-scatter: lane L ends with the whole sum of s[L], the score of head L / TT at token L % TT
#pragma unroll
  for (int o = 16, n = 32; o >= 1; o >>= 1, n >>= 1) {
    const bool up = lane & o;
#pragma unroll
    for (int i = 0; i < n / 2; ++i) {
      const float send = up ? s[i] : s[i + n / 2], keep = up ? s[i + n / 2] : s[i];
      s[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
  }
  const float sc = lane % TT < nt ? s[0] : -INFINITY;
  float gm = sc;
#pragma unroll
  for (int o = 1; o < TT; o <<= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, o));
  const float m_new = fmaxf(m_own, gm);
  const float scale = exp2f(m_own - m_new), p = exp2f(sc - m_new);
  float ps = p;
#pragma unroll
  for (int o = 1; o < TT; o <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
  l_own = __fmaf_rn(l_own, scale, ps);
  m_own = m_new;
#pragma unroll
  for (int h = 0; h < GROUP; ++h) {
    const float sh = __shfl_sync(0xffffffffu, scale, h * TT);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[h][i] = __fmul_rn(acc[h][i], sh);
  }
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    if (t < nt) {
      float vv[4];
      bf16x4(src + t * RAW + TL_BYTES / 2 + g * HEAD_DIM * 2 + lane * 8, vv);
#pragma unroll
      for (int h = 0; h < GROUP; ++h) {
        const float ph = __shfl_sync(0xffffffffu, p, h * TT + t);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][i] = __fmaf_rn(ph, vv[i], acc[h][i]);
      }
    }
  }
}

__device__ __forceinline__ void attn_out(float *o, int g, int lane, const float (&acc)[GROUP][4], float m_own,
                                         float l_own) {
#pragma unroll
  for (int h = 0; h < GROUP; ++h)
    reinterpret_cast<float4 *>(o + (g * GROUP + h) * HEAD_DIM)[lane] = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
  if (lane % TT == 0) {
    o[HEADS * HEAD_DIM + g * GROUP + lane / TT] = m_own;
    o[HEADS * HEAD_DIM + HEADS + g * GROUP + lane / TT] = l_own;
  }
}

template <bool ENC>
__global__ void __launch_bounds__(AT, ATTN_MINB) attn_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                     float *__restrict__ part, int ctx, int layers, int layer,
                                                     int per_split, int splits, int k) {
  constexpr int RAW = ENC ? TLE : TL_BYTES, ST = attn_dt<ENC>() * TT;
  extern __shared__ __align__(16) uint8_t sm[];
  uint8_t *raw = sm;
  uint8_t *qs = sm + 2 * ST * RAW;
  const int seq = blockIdx.x / splits, split = blockIdx.x % splits;
  const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
  const int g = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < HEADS * HEAD_DIM / 8; i += AT)
    reinterpret_cast<uint4 *>(qs)[i] = reinterpret_cast<const uint4 *>(q + (size_t)seq * HEADS * HEAD_DIM)[i];
  float acc[GROUP][4] = {};
  float m_own = -INFINITY, l_own = 0.f;
  const int ntiles = t1 > t0 ? (t1 - t0 + ST - 1) / ST : 0;
  const auto issue = [&](int tile) {
    const int bt = t0 + tile * ST, nt = min(ST, t1 - bt);
    uint8_t *dst = raw + (tile & 1) * ST * RAW;
    for (int c = threadIdx.x; c < nt * (RAW / 16); c += AT) {
      const int t = c / (RAW / 16), o = c % (RAW / 16);
      const size_t s = (size_t)seq * ctx + bt + t;
      copy16(dst + t * RAW + 16 * o, kv + (s * layers + layer) * RAW + 16 * o);
    }
    asm volatile("cp.async.commit_group;");
  };
  if (ntiles) issue(0);
  for (int tile = 0; tile < ntiles; ++tile) {
    if (tile + 1 < ntiles) issue(tile + 1);
    else asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    const int bt = t0 + tile * ST, nt = min(ST, t1 - bt);
    uint8_t *src = raw + (tile & 1) * ST * RAW;
    if (ENC) {
      for (int u = threadIdx.x; u < nt * TLW; u += AT) {
        const int t = u / TLW, w = u % TLW;
        u32 x[NL];
        u32 *wp = reinterpret_cast<u32 *>(src + t * TLE + w * WB);
        load(wp, x);
        decode_word(x, (((u64)seq * ctx + bt + t) * layers + layer) * TLW + w, k);
        store(wp, x);
      }
      __syncthreads();
    }
#pragma unroll
    for (int d = 0; d < ST / TT; ++d)
      if (nt > d * TT) attn_tile<RAW>(src + d * TT * RAW, qs, g, lane, min(TT, nt - d * TT), acc, m_own, l_own);
    __syncthreads();
  }
  attn_out(part + ((size_t)seq * splits + split) * PART, g, lane, acc, m_own, l_own);
}

// The warp-specialized encoded form: AT attention threads as in attn_kernel, and DW decoder warps that copy each tile
// into a ring of NBUF buffers PRE tiles ahead (cp.async, each thread its own words' chunks), decode their words in place
// and hand the buffer over (full); the attention warps hand it back (empty). The decode of later tiles runs while the
// attention warps work, and p768's 172 words a tile fill 6 warps.
constexpr int NBUF = 4 * TT * TLE + HEADS * HEAD_DIM * 2 + 64 <= 99 * 1024 ? 4 : 2, PRE = NBUF / 2;
constexpr int DW = (TT * TLW + 31) / 32;
constexpr int WT = AT + 32 * DW;
constexpr int ws_smem() { return NBUF * TT * TLE + HEADS * HEAD_DIM * 2 + 2 * NBUF * 8; }
static_assert(ws_smem() <= 99 * 1024 && PRE < NBUF, "the ring fits an SM's shared memory");

__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t *bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t *bar, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(
                   smem_u32(bar)),
               "r"(parity)
               : "memory");
}

__global__ void __launch_bounds__(WT, 1) attn_ws_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                        float *__restrict__ part, int ctx, int layers, int layer,
                                                        int per_split, int splits, int k) {
  extern __shared__ __align__(16) uint8_t sm[];
  uint8_t *raw = sm;
  uint8_t *qs = sm + NBUF * TT * TLE;
  uint64_t *full = reinterpret_cast<uint64_t *>(qs + HEADS * HEAD_DIM * 2);
  uint64_t *empty = full + NBUF;
  const int seq = blockIdx.x / splits, split = blockIdx.x % splits;
  const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
  const int ntiles = t1 > t0 ? (t1 - t0 + TT - 1) / TT : 0;
  if (threadIdx.x == 0) {
    for (int b = 0; b < NBUF; ++b) {
      mbar_init(full + b, 32 * DW);
      mbar_init(empty + b, AT);
    }
  }
  for (int i = threadIdx.x; i < HEADS * HEAD_DIM / 8; i += WT)
    reinterpret_cast<uint4 *>(qs)[i] = reinterpret_cast<const uint4 *>(q + (size_t)seq * HEADS * HEAD_DIM)[i];
  __syncthreads();
  if (threadIdx.x >= AT) {
    const int d = threadIdx.x - AT;
    const auto issue = [&](int tile) {
      if (tile >= ntiles) return;
      if (tile >= NBUF) mbar_wait(empty + tile % NBUF, (tile / NBUF - 1) & 1);
      const int bt = t0 + tile * TT, nt = min(TT, t1 - bt);
      uint8_t *dst = raw + (tile % NBUF) * TT * TLE;
      for (int u = d; u < nt * TLW; u += 32 * DW) {
        const int t = u / TLW, w = u % TLW;
        const uint8_t *src = kv + (((size_t)seq * ctx + bt + t) * layers + layer) * TLE + w * WB;
#pragma unroll
        for (int c = 0; c < WB / 16; ++c) copy16(dst + t * TLE + w * WB + 16 * c, src + 16 * c);
      }
    };
    for (int j = 0; j < PRE; ++j) {
      issue(j);
      asm volatile("cp.async.commit_group;");
    }
    for (int j = 0; j < ntiles; ++j) {
      issue(j + PRE);
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group %0;" ::"n"(PRE));
      const int bt = t0 + j * TT, nt = min(TT, t1 - bt);
      uint8_t *src = raw + (j % NBUF) * TT * TLE;
      for (int u = d; u < nt * TLW; u += 32 * DW) {
        const int t = u / TLW, w = u % TLW;
        u32 x[NL];
        u32 *wp = reinterpret_cast<u32 *>(src + t * TLE + w * WB);
        load(wp, x);
        decode_word(x, (((u64)seq * ctx + bt + t) * layers + layer) * TLW + w, k);
        store(wp, x);
      }
      mbar_arrive(full + j % NBUF);
    }
    return;
  }
  const int g = threadIdx.x / 32, lane = threadIdx.x % 32;
  float acc[GROUP][4] = {};
  float m_own = -INFINITY, l_own = 0.f;
  for (int j = 0; j < ntiles; ++j) {
    const int b = j % NBUF;
    mbar_wait(full + b, (j / NBUF) & 1);
    attn_tile<TLE>(raw + b * TT * TLE, qs, g, lane, min(TT, t1 - t0 - j * TT), acc, m_own, l_own);
    mbar_arrive(empty + b);
  }
  attn_out(part + ((size_t)seq * splits + split) * PART, g, lane, acc, m_own, l_own);
}

// The tensor-core form, for fields whose words tile a KV head's 256 bytes of K, and of V (p512, p1024): a CTA is one
// split of one sequence's tokens at one KV head g, in tiles of MT tokens, token t's 512 bytes of head g (K, then V) at
// t * MTS in shared memory (16 bytes past 512, so ldmatrix's 8 rows land in distinct banks). Each of MW warps takes 16
// tokens of a tile on the tensor cores: S = Q K^T (bf16 mma, fp32 accumulate; rows the group's heads, zero past GROUP),
// the online softmax by head, and O^T += V^T P^T with P rounded to bf16 (P^T's fragment is S's) and its sum taken over
// the rounded P, so the output is a weighted mean. Each warp leaves its partials as a split of its own. The encoded form
// decodes a tile's words in place before the warps fold it, a thread a word, so plain and encoded agree bit for bit; the
// CUDA cores do the decode and little else.
#ifndef KV_MT
#define KV_MT 32
#endif
constexpr int HB = HEAD_DIM * 2;
constexpr bool MMA_OK = HB % WB == 0;
constexpr int MT = KV_MT, MW = MT / 16, MTS = 2 * HB + 16;
constexpr int HW = MMA_OK ? 2 * HB / WB : 1;
constexpr int MTH = MT * HW > 32 * MW ? MT * HW : 32 * MW;
constexpr int mma_smem() { return 2 * MT * MTS; }
static_assert(MT % 16 == 0 && MTS % 16 == 0 && MTH % 32 == 0 && mma_smem() <= 99 * 1024, "whole mma tiles");

__device__ __forceinline__ void ldsm4(unsigned a, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}

__device__ __forceinline__ void ldsm4t(unsigned a, u32 &r0, u32 &r1, u32 &r2, u32 &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}

__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// {lo, hi} rounded to bf16 and packed (lo in the low half), and the rounded values.
__device__ __forceinline__ u32 bf16x2_rn(float lo, float hi, float &rlo, float &rhi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  rlo = __uint_as_float(r << 16);
  rhi = __uint_as_float(r & 0xffff0000u);
  return r;
}

// Q's A fragments (rows the group's heads, k the dims), zero past GROUP.
__device__ __forceinline__ void mma_load_q(const uint16_t *q, int seq, int g, int lane, u32 (&qa)[8][2]) {
  const int gid = lane >> 2, t4 = lane & 3;
  const u32 *qp = reinterpret_cast<const u32 *>(q + ((size_t)seq * HEADS + g * GROUP + (gid < GROUP ? gid : 0)) * HEAD_DIM);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    qa[ks][0] = gid < GROUP ? qp[8 * ks + t4] : 0u;
    qa[ks][1] = gid < GROUP ? qp[8 * ks + 4 + t4] : 0u;
  }
}

// A warp's 16 tokens at shared address src, nt of them valid (the rest zero, so P V stays finite). SW: the tile's rows
// are 2 HB bytes, unpadded, and the 16-byte chunk c of token t is stored at chunk c ^ (t & 7) (src a multiple of 8
// tokens into its tile), so ldmatrix's 8 rows still fall in distinct banks. o holds NMT of the 8 blocks of 16 dims,
// from block mt0 on.
template <bool SW = false, int NMT = 8>
__device__ __forceinline__ void mma_chunk(unsigned src, const u32 (&qa)[8][2], int lane, int nt, float (&o)[NMT][4],
                                          float &m_own, float &l_own, int mt0 = 0) {
  const int t4 = lane & 3, mi = lane >> 3, r8 = lane & 7;
  const unsigned row = src + (8 * (mi >> 1) + r8) * (SW ? 2 * HB : MTS);
  const unsigned c0 = SW ? 16 * ((mi & 1) ^ (r8 & 1)) + 32 * (r8 >> 1) : 16 * (mi & 1);
  const auto at = [&](int c) { return row + (SW ? c0 ^ (32 * c) : c0 + 32 * c); };
  float s[2][4] = {};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    u32 b00, b01, b10, b11;
    ldsm4(at(ks), b00, b01, b10, b11);
    mma16816(s[0], qa[ks][0], 0u, qa[ks][1], 0u, b00, b01);
    mma16816(s[1], qa[ks][0], 0u, qa[ks][1], 0u, b10, b11);
  }
  float mx = -INFINITY;
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      if (8 * j + 2 * t4 + e >= nt) s[j][e] = -INFINITY;
      mx = fmaxf(mx, s[j][e]);
    }
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  const float m_new = fmaxf(m_own, mx), sc = exp2f(m_own - m_new);
  float pl[2][2];
  u32 pb[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) pb[j] = bf16x2_rn(exp2f(s[j][0] - m_new), exp2f(s[j][1] - m_new), pl[j][0], pl[j][1]);
  l_own = __fmaf_rn(l_own, sc, (pl[0][0] + pl[0][1]) + (pl[1][0] + pl[1][1]));
  m_own = m_new;
  const float s0 = __shfl_sync(0xffffffffu, sc, 8 * t4), s1 = __shfl_sync(0xffffffffu, sc, 8 * t4 + 4);
#pragma unroll
  for (int mt = 0; mt < NMT; ++mt) {
    u32 a0, a1, a2, a3;
    ldsm4t(at(mt0 + mt) + HB, a0, a1, a2, a3);
    o[mt][0] = __fmul_rn(o[mt][0], s0);
    o[mt][1] = __fmul_rn(o[mt][1], s1);
    o[mt][2] = __fmul_rn(o[mt][2], s0);
    o[mt][3] = __fmul_rn(o[mt][3], s1);
    mma16816(o[mt], a0, a1, a2, a3, pb[0], pb[1]);
  }
}

// A warp's partials: o[h][d] for the group's heads (o's NMT blocks of 16 dims from mt0 on), then, if ml, their max
// (finite, -1e30 for no tokens) and sum.
template <int NMT = 8>
__device__ __forceinline__ void mma_out(float *op, int g, int lane, const float (&o)[NMT][4], float m_own, float l_own,
                                        int mt0 = 0, bool ml = true) {
  const int gid = lane >> 2, t4 = lane & 3;
  l_own += __shfl_xor_sync(0xffffffffu, l_own, 1);
  l_own += __shfl_xor_sync(0xffffffffu, l_own, 2);
#pragma unroll
  for (int mt = 0; mt < NMT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int h = 2 * t4 + (e & 1), d = 16 * (mt0 + mt) + gid + 8 * (e >> 1);
      if (h < GROUP) op[(g * GROUP + h) * HEAD_DIM + d] = o[mt][e];
    }
  if (ml && t4 == 0 && gid < GROUP) {
    op[HEADS * HEAD_DIM + g * GROUP + gid] = l_own > 0.f ? m_own : -1e30f;
    op[HEADS * HEAD_DIM + HEADS + g * GROUP + gid] = l_own;
  }
}

// Head g's K and V of a tile's nt tokens into dst (cp.async), the rows past nt zeroed.
__device__ __forceinline__ void mma_issue(uint8_t *dst, const uint8_t *kv, int raw, size_t slot0, int layers, int layer,
                                          int g, int nt, int tid, int nthreads) {
  for (int c = tid; c < nt * 32; c += nthreads) {
    const int t = c >> 5, x = c & 31;
    copy16(dst + t * MTS + 16 * x,
           kv + ((slot0 + t) * layers + layer) * raw + (x < 16 ? 0 : TL_BYTES / 2 - HB) + g * HB + 16 * x);
  }
  for (int c = nt * 32 + tid; c < MT * 32; c += nthreads)
    *reinterpret_cast<uint4 *>(dst + (c >> 5) * MTS + 16 * (c & 31)) = make_uint4(0u, 0u, 0u, 0u);
}

template <bool ENC>
__global__ void __launch_bounds__(MTH, 2) attn_mma_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                         float *__restrict__ part, int ctx, int layers, int layer,
                                                         int per_split, int splits, int k) {
  if constexpr (MMA_OK) {
    constexpr int RAW = ENC ? TLE : TL_BYTES;
    extern __shared__ __align__(16) uint8_t sm[];
    const int g = blockIdx.x % KV_HEADS, sq = blockIdx.x / KV_HEADS, seq = sq / splits, split = sq % splits;
    const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
    const int ntiles = t1 > t0 ? (t1 - t0 + MT - 1) / MT : 0;
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    u32 qa[8][2];
    float o[8][4] = {};
    float m_own = -INFINITY, l_own = 0.f;
    if (w < MW) mma_load_q(q, seq, g, lane, qa);
    const auto issue = [&](int tile) {
      const int bt = t0 + tile * MT;
      mma_issue(sm + (tile & 1) * MT * MTS, kv, RAW, (size_t)seq * ctx + bt, layers, layer, g, min(MT, t1 - bt),
                threadIdx.x, MTH);
      asm volatile("cp.async.commit_group;");
    };
    if (ntiles) issue(0);
    for (int tile = 0; tile < ntiles; ++tile) {
      if (tile + 1 < ntiles) issue(tile + 1);
      else asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 1;");
      __syncthreads();
      const int bt = t0 + tile * MT, nt = min(MT, t1 - bt);
      uint8_t *src = sm + (tile & 1) * MT * MTS;
      if (ENC) {
        const int t = threadIdx.x % MT, j = threadIdx.x / MT;
        if (t < nt && j < HW) {
          u32 x[NL];
          u32 *wp = reinterpret_cast<u32 *>(src + t * MTS + j * WB);
          load(wp, x);
          const int wi = (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + g * (HB / WB) + j;
          decode_word(x, (((u64)seq * ctx + bt + t) * layers + layer) * TLW + wi, k);
          store(wp, x);
        }
        __syncthreads();
      }
      if (w < MW && nt > 16 * w)
        mma_chunk(smem_u32(src + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
      __syncthreads();
    }
    if (w < MW) mma_out(part + ((size_t)sq * MW + w) * PART, g, lane, o, m_own, l_own);
  }
}

// The warp-specialized tensor-core form: MDW decoder warps, each lane a word of every tile (8 lanes a token's 8 words),
// load it from global memory a tile ahead into registers, decode it and store its payload into a ring of MNB tiles
// (full); the MW MMA warps fold each tile as in attn_mma_kernel and hand it back (empty). Shared memory holds only
// decoded tiles, so two CTAs fit an SM, and the decode never waits on the tensor cores' barrier.
constexpr int MNB = 2, MDW = MMA_OK ? MT * HW / 32 : 1, MWT = 32 * (MDW + MW);
constexpr int mws_smem() { return MNB * MT * MTS + 2 * MNB * 8 + NL * 4; }
static_assert(MT * HW % 32 == 0 && 32 % HW == 0, "whole decoder warps, a token's words in one warp");

__device__ __forceinline__ void ldg_word(const u32 *p, u32 (&x)[NL]) {
  const uint4 *q = reinterpret_cast<const uint4 *>(p);
#pragma unroll
  for (int k = 0; k < NL / 4; ++k) {
    const uint4 v = __ldcs(q + k);
    x[4 * k] = v.x;
    x[4 * k + 1] = v.y;
    x[4 * k + 2] = v.z;
    x[4 * k + 3] = v.w;
  }
}

__global__ void __launch_bounds__(MWT, 2) attn_mma_ws_kernel(const uint8_t *__restrict__ kv,
                                                            const uint16_t *__restrict__ q, float *__restrict__ part,
                                                            int ctx, int layers, int layer, int per_split, int splits,
                                                            int k) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
    u32 *td = reinterpret_cast<u32 *>(empty + MNB);
    const int g = blockIdx.x % KV_HEADS, sq = blockIdx.x / KV_HEADS, seq = sq / splits, split = sq % splits;
    const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
    const int ntiles = t1 > t0 ? (t1 - t0 + MT - 1) / MT : 0;
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * MDW);
        mbar_init(empty + b, 32 * MW);
      }
      u32 s[NL];
      tweak((u64)MT * layers * TLW, s);
#pragma unroll
      for (int l = 0; l < NL; ++l) td[l] = s[l];
    }
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int wi = (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + g * (HB / WB) + j;
      const auto word = [&](int tile) { return (((u64)seq * ctx + t0 + tile * MT + t) * layers + layer) * TLW + wi; };
      u32 nx[NL], tw[NL];
      tweak(word(0), tw);
      if (ntiles && t < min(MT, t1 - t0)) ldg_word(reinterpret_cast<const u32 *>(kv) + word(0) * NL, nx);
      for (int tile = 0; tile < ntiles; ++tile) {
        const int nt = min(MT, t1 - t0 - tile * MT);
        u32 x[NL];
        copy(x, nx);
        if (tile + 1 < ntiles && t < min(MT, t1 - t0 - (tile + 1) * MT))
          ldg_word(reinterpret_cast<const u32 *>(kv) + word(tile + 1) * NL, nx);
        if (tile) tweak_step(tw, td);
        if (t < nt) decode_word_t(x, tw, k);
        else for (int l = 0; l < NL; ++l) x[l] = 0u;
        if (tile >= MNB) mbar_wait(empty + tile % MNB, (tile / MNB - 1) & 1);
        store(reinterpret_cast<u32 *>(sm + (tile % MNB) * MT * MTS + t * MTS + j * WB), x);
        mbar_arrive(full + tile % MNB);
      }
      return;
    }
    u32 qa[8][2];
    float o[8][4] = {};
    float m_own = -INFINITY, l_own = 0.f;
    mma_load_q(q, seq, g, lane, qa);
    for (int tile = 0; tile < ntiles; ++tile) {
      const int b = tile % MNB, nt = min(MT, t1 - t0 - tile * MT);
      mbar_wait(full + b, (tile / MNB) & 1);
      if (nt > 16 * w)
        mma_chunk(smem_u32(sm + b * MT * MTS + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
      mbar_arrive(empty + b);
    }
    mma_out(part + ((size_t)sq * MW + w) * PART, g, lane, o, m_own, l_own);
  }
}

// The persistent warp-specialized form: gridDim.x CTAs (a CTA pair an SM at most once over), CTA c folding tiles
// [c N / G, (c + 1) N / G) of the N tiles in (sequence, KV head, tile) order, so the CTAs get the same tiles within one
// and none starts or drains per pair: a decoder lane loads the next pair's first word during the last tile's decode.
// A pair (sequence, head) folded by several CTAs leaves one partial each in its slot e, the CTA's index among them; the
// CTA folding the pair's last tile marks slots e + 1 .. sp - 1 empty, so attn_combine reads sp slots a pair. Its plain
// form (ENC = false) stores the words undecoded, the same partition, so the two agree bit for bit.
template <bool ENC>
__global__ void __launch_bounds__(MWT, 2) attn_pers_kernel(const uint8_t *__restrict__ kv,
                                                          const uint16_t *__restrict__ q, float *__restrict__ part,
                                                          int batch, int ctx, int layers, int layer, int sp, int k) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
    u32 *td = reinterpret_cast<u32 *>(empty + MNB);
    const int T = (ctx + MT - 1) / MT;
    const u64 N = (u64)batch * KV_HEADS * T, G = gridDim.x, c = blockIdx.x;
    const u64 i0 = c * N / G;
    const int n = (int)((c + 1) * N / G - i0);
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * MDW);
        mbar_init(empty + b, 32 * MW);
      }
      if (ENC) {
        u32 s[NL];
        tweak((u64)MT * layers * TLW, s);
#pragma unroll
        for (int l = 0; l < NL; ++l) td[l] = s[l];
      }
    }
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int wo = (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + j;
      const auto word = [&](int p, int tile) {
        return (((u64)(p / KV_HEADS) * ctx + tile * MT + t) * layers + layer) * TLW + wo + p % KV_HEADS * (HB / WB);
      };
      int p = (int)(i0 / T), tile = (int)(i0 % T), np = p, ntile = tile;
      u32 nx[NL], tw[NL];
      if (n && tile * MT + t < ctx) ldg_word(reinterpret_cast<const u32 *>(kv) + word(p, tile) * NL, nx);
#pragma unroll 1
      for (int li = 0; li < n; ++li) {
        u32 x[NL];
        copy(x, nx);
        if (++ntile == T) ntile = 0, ++np;
        if (li + 1 < n && ntile * MT + t < ctx)
          ldg_word(reinterpret_cast<const u32 *>(kv) + word(np, ntile) * NL, nx);
        if (ENC) {
          if (li == 0 || tile == 0) tweak(word(p, tile), tw);
          else tweak_step(tw, td);
        }
        if (tile * MT + t < ctx) {
          if (ENC) decode_word_t(x, tw, k);
        } else {
#pragma unroll
          for (int l = 0; l < NL; ++l) x[l] = 0u;
        }
        if (li >= MNB) mbar_wait(empty + li % MNB, (li / MNB - 1) & 1);
        store(reinterpret_cast<u32 *>(sm + (li % MNB) * MT * MTS + t * MTS + j * WB), x);
        mbar_arrive(full + li % MNB);
        p = np, tile = ntile;
      }
      return;
    }
    u32 qa[8][2];
    float o[8][4];
    float m_own = -INFINITY, l_own = 0.f;
    int p = (int)(i0 / T), tile = (int)(i0 % T), e = 0;
#pragma unroll 1
    for (int li = 0; li < n; ++li) {
      const int seq = p / KV_HEADS, g = p % KV_HEADS;
      if (li == 0 || tile == 0) {
        mma_load_q(q, seq, g, lane, qa);
#pragma unroll
        for (int mt = 0; mt < 8; ++mt)
#pragma unroll
          for (int v = 0; v < 4; ++v) o[mt][v] = 0.f;
        m_own = -INFINITY, l_own = 0.f;
        e = (int)(c - (((u64)p * T + 1) * G - 1) / N);
      }
      const int b = li % MNB, nt = min(MT, ctx - tile * MT);
      mbar_wait(full + b, (li / MNB) & 1);
      if (nt > 16 * w)
        mma_chunk(smem_u32(sm + b * MT * MTS + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
      mbar_arrive(empty + b);
      if (tile == T - 1 || li == n - 1) {
        mma_out(part + (((size_t)seq * sp + e) * MW + w) * PART, g, lane, o, m_own, l_own);
        if (tile == T - 1)
          for (int s = e + 1; s < sp; ++s) {
            float *op = part + (((size_t)seq * sp + s) * MW + w) * PART;
            for (int v = lane; v < GROUP * HEAD_DIM; v += 32) op[g * GROUP * HEAD_DIM + v] = 0.f;
            if (lane < GROUP) {
              op[HEADS * HEAD_DIM + g * GROUP + lane] = -1e30f;
              op[HEADS * HEAD_DIM + HEADS + g * GROUP + lane] = 0.f;
            }
          }
      }
      if (++tile == T) tile = 0, ++p;
    }
  }
}

// The warp-specialized form at the standalone decode's occupancy: ODW = 12 decoder warps and the MW MMA warps a CTA, two
// CTAs an SM at 72 registers, so 24 decoder warps an SM. A decoder lane holds no next word in registers: it loads its
// word of a tile of OT tokens, decodes it and stores it, and the SM's other warps hide the load, as in decode_kernel.
// The ring of MNB tiles is unpadded and swizzled (mma_chunk's SW), so two CTAs' rings fit an SM. Each MMA warp folds
// every chunk of 16 tokens into OMT of the 8 blocks of 16 dims (both compute the scores; the dims split keeps their
// registers within 72), so a CTA leaves one split's partials. Its plain form (ENC = false) stores the words undecoded,
// so the two agree bit for bit. The ring keeps a CTA's decoder warps within a tile of each other, so they wait on their
// loads together; with KV_OCC_PF (default 1) the first lane of each run of a token's K or V words prefetches the next
// tile's run into the L2, which holds no registers.
#ifndef KV_OCC_PF
#define KV_OCC_PF 1
#endif
constexpr bool OPF = KV_OCC_PF;
constexpr int ODW = 12, OT = MMA_OK ? 32 * ODW / HW : 16, OC = OT / 16, OWT = 32 * (ODW + MW), OMT = 8 / MW;
constexpr int occ_smem() { return MNB * OT * 2 * HB + 2 * MNB * 8 + NL * 4; }
static_assert(!MMA_OK || (OT % 16 == 0 && WB % 16 == 0), "whole chunks, 16-byte stores");

template <bool ENC>
__global__ void __launch_bounds__(OWT, 2) attn_occ_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                         float *__restrict__ part, int ctx, int layers, int layer,
                                                         int per_split, int splits, int k) {
  if constexpr (MMA_OK) {
    constexpr int TB = OT * 2 * HB;
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * TB), *empty = full + MNB;
    u32 *td = reinterpret_cast<u32 *>(empty + MNB);
    const int g = blockIdx.x % KV_HEADS, sq = blockIdx.x / KV_HEADS, seq = sq / splits, split = sq % splits;
    const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
    const int ntiles = t1 > t0 ? (t1 - t0 + OT - 1) / OT : 0;
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * ODW);
        mbar_init(empty + b, 32 * MW);
      }
      if (ENC) {
        u32 s[NL];
        tweak((u64)OT * layers * TLW, s);
#pragma unroll
        for (int l = 0; l < NL; ++l) td[l] = s[l];
      }
    }
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int wi = (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + g * (HB / WB) + j;
      const u64 w0 = (((u64)seq * ctx + t0 + t) * layers + layer) * TLW + wi;
      const u32 *src = reinterpret_cast<const u32 *>(kv) + w0 * NL;
      const u64 step = (u64)OT * layers * TLW * NL;
      u32 tw[NL];
      if (ENC) tweak(w0, tw);
      uint8_t *row = sm + t * 2 * HB;
#pragma unroll 1
      for (int tile = 0; tile < ntiles; ++tile, src += step) {
        const int nt = min(OT, t1 - t0 - tile * OT);
        u32 x[NL];
        if (ENC && tile) tweak_step(tw, td);
        if (t < nt) {
          ldg_word(src, x);
          if (OPF && (j & (HW / 2 - 1)) == 0 && t < t1 - t0 - (tile + 1) * OT)
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src + step), "n"(HB) : "memory");
          if (ENC) decode_word_t(x, tw, k);
        } else {
#pragma unroll
          for (int l = 0; l < NL; ++l) x[l] = 0u;
        }
        if (tile >= MNB) mbar_wait(empty + tile % MNB, (tile / MNB - 1) & 1);
        uint8_t *dst = row + (tile % MNB) * TB;
#pragma unroll
        for (int c = 0; c < WB / 16; ++c)
          *reinterpret_cast<uint4 *>(dst + 16 * ((WB / 16 * j + c) ^ (t & 7))) =
              make_uint4(x[4 * c], x[4 * c + 1], x[4 * c + 2], x[4 * c + 3]);
        mbar_arrive(full + tile % MNB);
      }
      return;
    }
    u32 qa[8][2];
    float o[OMT][4] = {};
    float m_own = -INFINITY, l_own = 0.f;
    mma_load_q(q, seq, g, lane, qa);
#pragma unroll 1
    for (int tile = 0; tile < ntiles; ++tile) {
      const int b = tile % MNB, nt = min(OT, t1 - t0 - tile * OT);
      mbar_wait(full + b, (tile / MNB) & 1);
#pragma unroll 1
      for (int c = 0; c < OC && nt > 16 * c; ++c)
        mma_chunk<true, OMT>(smem_u32(sm + b * TB + 16 * c * 2 * HB), qa, lane, min(16, nt - 16 * c), o, m_own, l_own,
                             OMT * w);
      mbar_arrive(empty + b);
    }
    mma_out<OMT>(part + (size_t)sq * PART, g, lane, o, m_own, l_own, OMT * w, w == 0);
  }
}

// A block a sequence's head: warp w folds splits w, w + CW, ... online, a lane 4 dims, and the warps' folds merge in
// shared memory. Every split holds at least one token, so warp 0's max is finite.
constexpr int CW = 8;

__global__ void __launch_bounds__(32 * CW) attn_combine(const float *__restrict__ part, float *__restrict__ out,
                                                        int splits) {
  __shared__ float4 sacc[CW][32];
  __shared__ float sm_[CW], sl[CW];
  const int seq = blockIdx.x / HEADS, h = blockIdx.x % HEADS, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float *P = part + (size_t)seq * splits * PART;
  float m = -INFINITY, l = 0.f;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = w; s < splits; s += CW) {
    const float *ps = P + (size_t)s * PART;
    const float ms = ps[HEADS * HEAD_DIM + h], ls = ps[HEADS * HEAD_DIM + HEADS + h];
    const float4 o = reinterpret_cast<const float4 *>(ps + h * HEAD_DIM)[lane];
    const float mn = fmaxf(m, ms), x = exp2f(m - mn), y = exp2f(ms - mn);
    l = l * x + ls * y;
    a = make_float4(a.x * x + o.x * y, a.y * x + o.y * y, a.z * x + o.z * y, a.w * x + o.w * y);
    m = mn;
  }
  sacc[w][lane] = a;
  if (lane == 0) { sm_[w] = m; sl[w] = l; }
  __syncthreads();
  if (w) return;
  float M = sm_[0];
#pragma unroll
  for (int v = 1; v < CW; ++v) M = fmaxf(M, sm_[v]);
  float L = 0.f;
  float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int v = 0; v < CW; ++v) {
    const float x = exp2f(sm_[v] - M);
    const float4 o = sacc[v][lane];
    L += sl[v] * x;
    A = make_float4(A.x + o.x * x, A.y + o.y * x, A.z + o.z * x, A.w + o.w * x);
  }
  reinterpret_cast<float4 *>(out + ((size_t)seq * HEADS + h) * HEAD_DIM)[lane] =
      make_float4(A.x / L, A.y / L, A.z / L, A.w / L);
}

}  // namespace kv

static int launched() { return (int)cudaPeekAtLastError(); }

static int blocks(uint64_t n, int grid) {
  return grid > 0 ? grid : (int)((n + kv::WPB - 1) / kv::WPB);
}

extern "C" {

int kv_bits() { return kv::NBITS; }

int kv_kmax() { return kv::KMAX; }

int kv_karatsuba() { return kv::KARATSUBA; }

// c: k round constants of NL little-endian limbs each, then tau.
int kv_set_key(const uint32_t *c, int k, const uint32_t *tau) {
  if (k < 1 || k > kv::KMAX) return -1;
  if (cudaMemcpyToSymbol(kv::kC, c, sizeof(uint32_t) * kv::NL * k) != cudaSuccess) return launched();
  if (cudaMemcpyToSymbol(kv::kTau, tau, sizeof(uint32_t) * kv::NL) != cudaSuccess) return launched();
  return 0;
}

// grid <= 0: a thread a word.
int kv_encode(const void *in, void *out, uint64_t n, uint64_t base, int k, int grid, cudaStream_t s) {
  if (k < 1 || k > kv::KMAX) return -1;
  kv::encode_kernel<<<blocks(n, grid), kv::WPB, 0, s>>>((const uint32_t *)in, (uint32_t *)out, n, base, k);
  return launched();
}

// The encode's preferred shared memory carveout (percent of the SM's L1 and shared memory, -1 the driver's choice). An
// SM holding the encode's CTAs takes another kernel's CTAs only when its carveout fits their shared memory, so beside a
// GEMM the encode asks for the most.
int kv_encode_carveout(int pct) {
  return (int)cudaFuncSetAttribute(kv::encode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, pct);
}

int kv_decode(const void *in, void *out, uint64_t n, uint64_t base, int k, int grid, unsigned long long *xr,
              cudaStream_t s) {
  if (k < 1 || k > kv::KMAX || (!out && !xr)) return -1;
  kv::decode_kernel<<<blocks(n, grid), kv::WPB, 0, s>>>((const uint32_t *)in, (uint32_t *)out, n, base, k, xr);
  return launched();
}

// kv_decode at most ctas CTAs an SM (padded with dynamic shared memory the kernel never reads): the read-and-decode
// at a given decoder occupancy. *got is the CTAs an SM the occupancy calculator grants.
int kv_decode_occ(const void *in, uint64_t n, uint64_t base, int k, int grid, unsigned long long *xr, int ctas,
                  int *got, cudaStream_t s) {
  if (k < 1 || k > kv::KMAX || ctas < 1 || grid < 1) return -1;
  int dev, per_sm, reserved;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, kv::decode_kernel);
  const int smem = (per_sm / ctas - reserved - (int)fa.sharedSizeBytes) / 128 * 128;
  if (smem < 0) return -1;
  cudaFuncSetAttribute(kv::decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(got, kv::decode_kernel, kv::WPB, smem);
  kv::decode_kernel<<<grid, kv::WPB, smem, s>>>((const uint32_t *)in, nullptr, n, base, k, xr);
  return launched();
}

int kv_prim(int op, const void *a, const void *b, void *out, int n, cudaStream_t s) {
  if (op < 0 || op > 4 || n < 1) return -1;
  kv::prim_kernel<<<(n + kv::WPB - 1) / kv::WPB, kv::WPB, 0, s>>>(op, (const uint32_t *)a, (const uint32_t *)b,
                                                                  (uint32_t *)out, n);
  return launched();
}

int kv_read(const void *store, uint64_t bytes, unsigned long long *xr, int grid, cudaStream_t s) {
  kv::read_kernel<<<grid, 256, 0, s>>>((const uint4 *)store, bytes / 16, xr);
  return launched();
}

int kv_chain(int what, int k, int ilp, uint64_t steps, int grid, unsigned long long *xr, cudaStream_t s) {
  if (what < 0 || what > 2 || ((what > 0) && (k < 1 || k > kv::KMAX))) return -1;
  if (what == 0 && ilp == 1) kv::chain_kernel<0, 1><<<grid, kv::WPB, 0, s>>>(k, steps, xr);
  else if (what == 0 && ilp == 2) kv::chain_kernel<0, 2><<<grid, kv::WPB, 0, s>>>(k, steps, xr);
  else if (what == 1) kv::chain_kernel<1, 1><<<grid, kv::WPB, 0, s>>>(k, steps, xr);
  else if (what == 2) kv::chain_kernel<2, 1><<<grid, kv::WPB, 0, s>>>(k, steps, xr);
  else return -1;
  return launched();
}

int kv_spin(long long cycles, cudaStream_t s) {
  kv::spin_kernel<<<1, 1, 0, s>>>(cycles);
  return launched();
}

// heads, KV heads, head dim, words a token-layer, its stored bytes, tokens a tile, floats a split's partials; the
// tensor-core form's warps a CTA (each its own split's partials), whether this field has it, and its tokens a tile.
void kv_attn_shape(int *out) {
  out[0] = kv::HEADS; out[1] = kv::KV_HEADS; out[2] = kv::HEAD_DIM; out[3] = kv::TLW; out[4] = kv::TLE; out[5] = kv::TT;
  out[6] = kv::PART; out[7] = kv::MW; out[8] = kv::MMA_OK; out[9] = kv::MT;
}

// The CTAs an SM the occupancy calculator grants kv_attn's tensor-core forms 5 and 7 (warp-specialized, and at the
// decode's occupancy), and their decoder warps a CTA: out[0..3].
int kv_attn_occ(int *out) {
  if (!kv::MMA_OK) return -2;
  cudaFuncSetAttribute(kv::attn_mma_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mws_smem());
  cudaFuncSetAttribute(kv::attn_occ_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::occ_smem());
  cudaFuncSetAttribute(kv::attn_occ_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, kv::attn_mma_ws_kernel, kv::MWT, kv::mws_smem());
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(out + 2, kv::attn_occ_kernel<true>, kv::OWT, kv::occ_smem());
  out[1] = kv::MDW;
  out[3] = kv::ODW;
  return launched();
}

// One layer's decode attention for `batch` sequences of `ctx` tokens (attn_kernel's layout): kv plain (encoded = 0) or
// its words (encoded = 1, k rounds; encoded = 2 the warp-specialized form, the same output); on the tensor cores,
// encoded = 3 plain, 4 its words and 5 its words in the warp-specialized form (4 and 5 the same output as 3, not as 0);
// encoded = 6 the tensor-core form at the decode's occupancy, plain, and 7 its words (the same output as 6); encoded =
// 8 the persistent warp-specialized form, plain, and 9 its words (the same output as 8), `splits` times the CTAs the
// SMs hold at once, fewer if a pair would span more than 16 of them.
// q bf16 [batch][HEADS][128] pre-scaled by log2(e)/sqrt(128), out fp32 [batch][HEADS][128], part at least batch *
// splits * kv_attn_shape's out[6] floats (times out[7] from 3 on; batch * 16 * out[6] * out[7] for 8 and 9). The tokens
// are split into at most `splits` ranges of whole tiles.
int kv_attn(const void *kv_, int encoded, const uint16_t *q, float *out, float *part, int batch, int ctx, int layers,
            int layer, int splits, int k, cudaStream_t s) {
  if (batch < 1 || ctx < 1 || splits < 1 || layer < 0 || layer >= layers || encoded < 0 || encoded > 9 ||
      (encoded != 0 && encoded != 3 && encoded != 6 && encoded != 8 && (k < 1 || k > kv::KMAX)))
    return -1;
  if (encoded >= 8) {
    if (!kv::MMA_OK) return -2;
    static int slots = 0;
    if (!slots) {
      int dev, sms, occ;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      for (auto *f : {kv::attn_pers_kernel<true>, kv::attn_pers_kernel<false>})
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mws_smem());
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kv::attn_pers_kernel<true>, kv::MWT, kv::mws_smem());
      slots = sms * occ;
    }
    constexpr long long MAX_SP = 16;
    const long long T = (ctx + kv::MT - 1) / kv::MT, N = (long long)batch * kv::KV_HEADS * T;
    long long G = std::min((long long)splits * slots, N);
    const long long per_min = std::max(1LL, (T - 1 + MAX_SP - 2) / (MAX_SP - 1));
    G = std::min(G, N / per_min);
    const long long per = N / G, sp = (T - 1 + per - 1) / per + 1;
    (encoded == 9 ? kv::attn_pers_kernel<true> : kv::attn_pers_kernel<false>)<<<(int)G, kv::MWT, kv::mws_smem(), s>>>(
        static_cast<const uint8_t *>(kv_), q, part, batch, ctx, layers, layer, (int)sp, k);
    kv::attn_combine<<<batch * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, (int)sp * kv::MW);
    return launched();
  }
  const int tile = encoded >= 6 ? kv::OT : encoded >= 3 ? kv::MT : kv::TT;
  const int per = (ctx + splits - 1) / splits, per_split = (per + tile - 1) / tile * tile;
  const int sp = (ctx + per_split - 1) / per_split;
  const auto *p = static_cast<const uint8_t *>(kv_);
  if (encoded >= 3) {
    if (!kv::MMA_OK) return -2;
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(kv::attn_mma_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mma_smem());
      cudaFuncSetAttribute(kv::attn_mma_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mma_smem());
      cudaFuncSetAttribute(kv::attn_mma_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mws_smem());
      for (auto *f : {kv::attn_occ_kernel<true>, kv::attn_occ_kernel<false>}) {
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::occ_smem());
        cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
      }
      set = true;
    }
    const int grid = batch * sp * kv::KV_HEADS;
    if (encoded >= 6)
      (encoded == 7 ? kv::attn_occ_kernel<true> : kv::attn_occ_kernel<false>)<<<grid, kv::OWT, kv::occ_smem(), s>>>(
          p, q, part, ctx, layers, layer, per_split, sp, k);
    else if (encoded == 5)
      kv::attn_mma_ws_kernel<<<grid, kv::MWT, kv::mws_smem(), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k);
    else if (encoded == 4)
      kv::attn_mma_kernel<true><<<grid, kv::MTH, kv::mma_smem(), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k);
    else
      kv::attn_mma_kernel<false><<<grid, kv::MTH, kv::mma_smem(), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k);
    kv::attn_combine<<<batch * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, encoded >= 6 ? sp : sp * kv::MW);
    return launched();
  }
  if (encoded == 2) {
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(kv::attn_ws_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::ws_smem());
      set = true;
    }
    kv::attn_ws_kernel<<<batch * sp, kv::WT, kv::ws_smem(), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k);
  } else if (encoded) {
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(kv::attn_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::attn_smem<true>());
      set = true;
    }
    kv::attn_kernel<true><<<batch * sp, kv::AT, kv::attn_smem<true>(), s>>>(p, q, part, ctx, layers, layer, per_split,
                                                                            sp, k);
  } else {
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(kv::attn_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::attn_smem<false>());
      set = true;
    }
    kv::attn_kernel<false><<<batch * sp, kv::AT, kv::attn_smem<false>(), s>>>(p, q, part, ctx, layers, layer,
                                                                             per_split, sp, k);
  }
  kv::attn_combine<<<batch * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, sp);
  return launched();
}
}
