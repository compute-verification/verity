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
#ifndef KV_FOLD6
#define KV_FOLD6 0
#endif
constexpr bool FOLD6 = KV_FOLD6;
static_assert(!FOLD6 || (NL % 2 == 0 && !KARATSUBA), "FOLD 6 halves the limbs, and is its own Karatsuba level");
// KV_MASK=1: `kv.py`'s masked word, σ_K (kMask) between consecutive rounds.
#ifndef KV_MASK
#define KV_MASK 0
#endif
constexpr bool MASK = KV_MASK;
static_assert(NBITS % 32 == 0 && NL % 4 == 0 && CF % 4 == 1 && CF < (1u << 16) && (CF - 1) / 4 < (1u << LOW_BITS),
              "p = 2^n - c");

__constant__ u32 kC[KMAX][NL];
__constant__ u32 kTau[NL];
__constant__ u32 kMask[NL];

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

// KV_FOLD6=1: ra-squaring-ceiling's FOLD 6 (`benchmarks/rate_squaring/sq.cu`), the adversary's best squaring at 768 and
// 1,024 bits. sqr_dbl_n is its FOLD 5 square: the off-diagonal products column by column, each column's accumulator
// starting at the previous one's carry, then y = 2 S + sum a_i^2 2^(64 i) in one carry chain.
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

// FOLD 6's square: one Karatsuba level, a = a1 B + a0, a^2 = C B^2 + (A + C - D) B + A with A = a0^2, C = a1^2,
// D = |a0 - a1|^2, each a FOLD 5 square; then `fold`, which is FOLD 1's.
__device__ __forceinline__ void sqr_fold6(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
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
  sqr_dbl_n<H>(x0, A);
  sqr_dbl_n<H>(x1, C);
  sqr_dbl_n<H>(d, D);
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

__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  if constexpr (FOLD6) {
    u32 t[2 * NL];
    sqr_fold6(a, t);
    fold(t, x);
  } else if constexpr (KARATSUBA) {
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

// σ_K(x) = x ^ K when that is below p, else x, x canonical: x ^ K >= p needs its limbs above the lowest all ones and
// its lowest at least 2^32 - c, when the XOR is undone. Its top limb all ones (probability 2^-32) is the one test on
// the fast path.
__device__ __forceinline__ void sigma(u32 (&x)[NL], const u32 (&m)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] ^= m[k];
  if (__builtin_expect(x[NL - 1] != 0xffffffffu, 1)) return;
  u32 all = x[1];
#pragma unroll
  for (int k = 2; k < NL - 1; ++k) all &= x[k];
  if (all == 0xffffffffu && x[0] >= 0u - CF) {
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] ^= m[k];
  }
}

// Past 768 bits the tweak is recomputed each round, so it isn't held through the root.
__device__ __forceinline__ void encode_word(u32 (&x)[NL], u64 i, int k) {
  u32 t[NL];
  if (!STREAM) tweak(i, t);
#pragma unroll 1
  for (int j = 0; j < k; ++j) {
    u32 c[NL];
    if (MASK && j > 0) sigma(x, kMask);
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
    if (MASK && j > 0) sigma(x, kMask);
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
// 2 neg_if(a, b odd), 3 canon(a), 4 sqr(a), 5 sigma(a) with b as K.
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
  else if (op == 5) sigma(x, y);
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
// tiles at a time where two fit (1,536 bits' 22 words a token-layer would leave most of a tile's threads idle; 8B's
// 8-token tiles of 4 KiB at 1,024 bits don't fit two), each tile then folded in turn as in the plain form.
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
__host__ __device__ constexpr int attn_dt() {
  return ENC && STREAM && 4 * TT * TLE + HEADS * HEAD_DIM * 2 + 1024 <= 99 * 1024 ? 2 : 1;
}
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

// Receipts inside mma-ws (RC = 1 unbound, 2 bound). A block is RT tiles of one (sequence, KV head) pair, at absolute
// tile offsets of RT. Each decoder lane feeds its raw word of each tile, before decoding it, to mma.m16n8k32.s8 as the A
// operand: step s (0..RS - 1, RS = NL / 4) is registers x[4s..4s+3], so the warp's 16 A rows are its 16 vectors of
// 32 RS bytes a tile (ra-sketch-tag's fragment order: A register 2h + e is row g + 8e, column 16h + 4tq + b of lane
// 4g + tq, byte b), and its B operand, the 16 sketch columns, is register q = 2 blk + h of step s, loaded from shared
// memory: word (512 i mod ZW) + 128 s + 4 l + q of the tile's B slot (RC = 1: offset 0, not 512 i). kRcB holds a
// block's sign rows, word 128 s + 4 l + q of tile τ (kv_receipt.rows): ±1 bytes (RC = 1), or 0x00/0xFF masks M (RC = 2,
// ro-bound-sketch/v0, only at 64-byte words,
// B = R ⊕ M with R's word the z word of the same index, z the MMA warps' PV accumulators after tile j - RLAG: word
// (w 16 + 2 g + t4) 32 + 4 mt + e of warp w's o[mt][e] at lane 4g + t4, t4 < GROUP / 2; R = 0 for a CTA's first RLAG
// tiles). The MMA warps fill tile j's slot after folding tile j - RLAG (the first RLAG before their first tile), so the
// decoder warps spend a tile RS LDS and 2 RS IMMA on the sketch. A block's sketch, 16 int32 a vector, goes to
// rc_sk[vector][16], vector ((pair NBLK + block) MDW + warp) 16 + row; rc_tag_kernel hashes each into its receipt.
#ifndef KV_RT
#define KV_RT 16
#endif
#ifndef KV_RLAG
#define KV_RLAG MNB
#endif
constexpr int RT = KV_RT, RLAG = KV_RLAG, ZB = RLAG, ZW = MW * 8 * (GROUP / 2) * 32;
static_assert(RLAG >= 1 && (ZW & (ZW - 1)) == 0 && ZW >= 512 && GROUP % 2 == 0, "B slots outlive their readers");
constexpr int rc_smem(int rc) { return mws_smem() + (rc ? ZB * ZW * 4 + ZB * 8 : 0); }
// RC = 1's B slot is a tile's RCW sign words at offset 0; RC = 2's z indexing assumes 512 of them (64-byte words).
constexpr int RS = NL / 4, RCW = 128 * RS;
constexpr bool RC1_OK = MMA_OK && RCW <= ZW, RC2_OK = MMA_OK && RS == 4;
__device__ uint4 kRcB[2][RT * RCW / 4];
__constant__ uint32_t kRcKey[2][8];

__device__ __forceinline__ void imma16832(int (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One tile of a lane's sketch: br its B words in the tile's slot (shared), step s at br + 128 s.
__device__ __forceinline__ void rc_sketch(const u32 (&x)[NL], int (&acc)[2][4], const u32 *br) {
#pragma unroll
  for (int s = 0; s < RS; ++s) {
    const uint4 b = *reinterpret_cast<const uint4 *>(br + 128 * s);
    imma16832(acc[0], x[4 * s], x[4 * s + 1], x[4 * s + 2], x[4 * s + 3], b.x, b.y);
    imma16832(acc[1], x[4 * s], x[4 * s + 1], x[4 * s + 2], x[4 * s + 3], b.z, b.w);
  }
}

// The MMA warps' share of tile jt's B slot bs: R = 0, so B is kRcB's rows (both halves of ZW for RC = 2).
template <int RC>
__device__ __forceinline__ void rc_fill(u32 *bs, int jt, int mlane) {
  const uint4 *rows = kRcB[RC - 1] + jt % RT * (RCW / 4);
#pragma unroll
  for (int v = mlane; v < (RC == 2 ? ZW : RCW) / 4; v += 32 * MW)
    reinterpret_cast<uint4 *>(bs)[v] = rows[v % (RCW / 4)];
}

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

template <int RC = 0>
__global__ void __launch_bounds__(MWT, 2) attn_mma_ws_kernel(const uint8_t *__restrict__ kv,
                                                            const uint16_t *__restrict__ q, float *__restrict__ part,
                                                            int ctx, int layers, int layer, int per_split, int splits,
                                                            int k, int *__restrict__ rc_sk = nullptr,
                                                            u32 *__restrict__ zdump = nullptr) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
    u32 *td = reinterpret_cast<u32 *>(empty + MNB);
    u32 *bbuf = td + NL;
    uint64_t *bfull = reinterpret_cast<uint64_t *>(bbuf + ZB * ZW);
    const int g = blockIdx.x % KV_HEADS, sq = blockIdx.x / KV_HEADS, seq = sq / splits, split = sq % splits;
    const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
    const int ntiles = t1 > t0 ? (t1 - t0 + MT - 1) / MT : 0;
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * MDW);
        mbar_init(empty + b, 32 * MW);
      }
      if (RC > 0)
        for (int b = 0; b < ZB; ++b) mbar_init(bfull + b, 32 * MW);
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
      int acc[2][4] = {};
      tweak(word(0), tw);
      if (ntiles && t < min(MT, t1 - t0)) ldg_word(reinterpret_cast<const u32 *>(kv) + word(0) * NL, nx);
      for (int tile = 0; tile < ntiles; ++tile) {
        const int nt = min(MT, t1 - t0 - tile * MT);
        u32 x[NL];
        copy(x, nx);
        if (tile + 1 < ntiles && t < min(MT, t1 - t0 - (tile + 1) * MT))
          ldg_word(reinterpret_cast<const u32 *>(kv) + word(tile + 1) * NL, nx);
        if (tile) tweak_step(tw, td);
        if constexpr (RC > 0) {
          const int i = w - MW;
          if (t >= nt)
            for (int l = 0; l < NL; ++l) x[l] = 0u;
          mbar_wait(bfull + tile % ZB, (tile / ZB) & 1);
          rc_sketch(x, acc, bbuf + tile % ZB * ZW + (RC == 2 ? 512 * i & (ZW - 1) : 0) + 4 * lane);
          if (tile % RT == RT - 1 || tile == ntiles - 1) {
            const int nblk = (ctx + RT * MT - 1) / (RT * MT), blk = (t0 / MT + tile) / RT, gid = lane >> 2, t4 = lane & 3;
            int *vp = rc_sk + ((((size_t)seq * KV_HEADS + g) * nblk + blk) * MDW + i) * 16 * 16;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              *reinterpret_cast<int2 *>(vp + (gid + 8 * h) * 16 + 2 * t4) = make_int2(acc[0][2 * h], acc[0][2 * h + 1]);
              *reinterpret_cast<int2 *>(vp + (gid + 8 * h) * 16 + 8 + 2 * t4) =
                  make_int2(acc[1][2 * h], acc[1][2 * h + 1]);
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[0][v] = acc[1][v] = 0;
          }
        }
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
    const int rem = t1 - t0;
    if constexpr (RC > 0) {
      for (int jt = 0; jt < RLAG && jt * MT < rem; ++jt) {
        rc_fill<RC>(bbuf + jt * ZW, jt, 32 * w + lane);
        mbar_arrive(bfull + jt);
      }
    }
    for (int tile = 0; tile * MT < rem; ++tile) {
      const int b = tile % MNB, nt = min(MT, rem - tile * MT);
      mbar_wait(full + b, (tile / MNB) & 1);
      if (nt > 16 * w)
        mma_chunk(smem_u32(sm + b * MT * MTS + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
      mbar_arrive(empty + b);
      if constexpr (RC > 0) {
        const int jt = tile + RLAG;
        const bool next = jt * MT < rem;
        u32 *bs = bbuf + tile % ZB * ZW;
        if (RC == 1 && next) rc_fill<1>(bs, jt, 32 * w + lane);
        if constexpr (RC == 2) {
          constexpr int ZL = GROUP / 2;
          const int gid = lane >> 2, t4 = lane & 3;
          if (t4 < ZL) {
            const int zi = (w * 8 * ZL + gid * ZL + t4) * 32;
            const uint4 *rows = kRcB[1] + jt % RT * (RCW / 4) + (zi & (RCW - 1)) / 4;
            u32 *zd = zdump ? zdump + ((size_t)blockIdx.x * (per_split / MT) + tile) * ZW + zi : nullptr;
#pragma unroll
            for (int mt = 0; mt < 8; ++mt) {
              const uint4 v = make_uint4(__float_as_uint(o[mt][0]), __float_as_uint(o[mt][1]),
                                         __float_as_uint(o[mt][2]), __float_as_uint(o[mt][3]));
              if (next) {
                const uint4 m = rows[mt];
                *reinterpret_cast<uint4 *>(bs + zi + 4 * mt) = make_uint4(v.x ^ m.x, v.y ^ m.y, v.z ^ m.z, v.w ^ m.w);
              }
              if (zd) *reinterpret_cast<uint4 *>(zd + 4 * mt) = v;
            }
          }
        }
        if (next) mbar_arrive(bfull + tile % ZB);
      }
    }
    mma_out(part + ((size_t)sq * MW + w) * PART, g, lane, o, m_own, l_own);
  }
}

// A receipt a vector: BLAKE3-keyed(kRcKey[rc - 1], u64be(jbase + v) ‖ rc_sk[v] as 16 int32 LE), 32 bytes.
#define B3_IV0 0x6A09E667u
#define B3_IV1 0xBB67AE85u
#define B3_IV2 0x3C6EF372u
#define B3_IV3 0xA54FF53Au
constexpr u32 B3_CHUNK_START = 1, B3_CHUNK_END = 2, B3_ROOT = 8, B3_KEYED = 16;

#define B3_G(a, b, c, d, x, y)   \
  a += b + (x);                  \
  d = __funnelshift_r(d ^ a, d ^ a, 16); \
  c += d;                        \
  b = __funnelshift_r(b ^ c, b ^ c, 12); \
  a += b + (y);                  \
  d = __funnelshift_r(d ^ a, d ^ a, 8);  \
  c += d;                        \
  b = __funnelshift_r(b ^ c, b ^ c, 7)

// One compression of the message words m (permuted in place) from chaining value cv (`tagger.cu`'s b3_compress).
__device__ __forceinline__ void b3_compress(const u32 cv[8], u32 m[16], u32 len, u32 flags, u32 out[8],
                                            u32 counter = 0) {
  u32 v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  u32 v8 = B3_IV0, v9 = B3_IV1, v10 = B3_IV2, v11 = B3_IV3, v12 = counter, v13 = 0, v14 = len, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    B3_G(v0, v4, v8, v12, m[0], m[1]);
    B3_G(v1, v5, v9, v13, m[2], m[3]);
    B3_G(v2, v6, v10, v14, m[4], m[5]);
    B3_G(v3, v7, v11, v15, m[6], m[7]);
    B3_G(v0, v5, v10, v15, m[8], m[9]);
    B3_G(v1, v6, v11, v12, m[10], m[11]);
    B3_G(v2, v7, v8, v13, m[12], m[13]);
    B3_G(v3, v4, v9, v14, m[14], m[15]);
    if (r < 6) {
      const u32 t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
  out[0] = v0 ^ v8, out[1] = v1 ^ v9, out[2] = v2 ^ v10, out[3] = v3 ^ v11;
  out[4] = v4 ^ v12, out[5] = v5 ^ v13, out[6] = v6 ^ v14, out[7] = v7 ^ v15;
}

__global__ void __launch_bounds__(128) rc_tag_kernel(int rc, const int *__restrict__ sk, u32 *__restrict__ tags, u64 n,
                                                     u64 jbase) {
  const u64 v = (u64)blockIdx.x * 128 + threadIdx.x;
  if (v >= n) return;
  const uint4 *p = reinterpret_cast<const uint4 *>(sk + v * 16);
  u32 m[16], key[8], cv[8], out[8];
  const u64 j = jbase + v;
  m[0] = __byte_perm((u32)(j >> 32), 0, 0x0123), m[1] = __byte_perm((u32)j, 0, 0x0123);
  const uint4 a = p[0], b = p[1], c = p[2], d = p[3];
  m[2] = a.x, m[3] = a.y, m[4] = a.z, m[5] = a.w, m[6] = b.x, m[7] = b.y, m[8] = b.z, m[9] = b.w;
  m[10] = c.x, m[11] = c.y, m[12] = c.z, m[13] = c.w, m[14] = d.x, m[15] = d.y;
#pragma unroll
  for (int i = 0; i < 8; ++i) key[i] = kRcKey[rc - 1][i];
  b3_compress(key, m, 64, B3_KEYED | B3_CHUNK_START, cv);
  m[0] = d.z, m[1] = d.w;
#pragma unroll
  for (int i = 2; i < 16; ++i) m[i] = 0u;
  b3_compress(cv, m, 8, B3_KEYED | B3_CHUNK_END | B3_ROOT, out);
  uint4 *o = reinterpret_cast<uint4 *>(tags + v * 8);
  o[0] = make_uint4(out[0], out[1], out[2], out[3]);
  o[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// A receipt a decoder warp's block of 16 vectors, the candidate that hashes 18 blocks where 16 vector receipts hash 32:
// BLAKE3-keyed(kRcKey[rc - 1], u64be(jbase + u) ‖ rc_sk[16 u .. 16 u + 15] as 256 int32 LE), 1,032 bytes, so chunk 0's
// 16 blocks (each the previous block's last two sketch words and 14 of its own), chunk 1's last 8 bytes, and their
// parent.
__global__ void __launch_bounds__(128) rc_tag16_kernel(int rc, const int *__restrict__ sk, u32 *__restrict__ tags, u64 n,
                                                       u64 jbase) {
  constexpr u32 B3_PARENT = 4;
  const u64 u = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;
  const uint4 *p = reinterpret_cast<const uint4 *>(sk + u * 256);
  u32 m[16], key[8], cv[8], c1[8], out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key[i] = kRcKey[rc - 1][i];
  const u64 j = jbase + u;
  u32 h0 = __byte_perm((u32)(j >> 32), 0, 0x0123), h1 = __byte_perm((u32)j, 0, 0x0123);
#pragma unroll 1
  for (int t = 0; t < 16; ++t) {
    const uint4 a = p[4 * t], b = p[4 * t + 1], c = p[4 * t + 2], d = p[4 * t + 3];
    m[0] = h0, m[1] = h1, m[2] = a.x, m[3] = a.y, m[4] = a.z, m[5] = a.w, m[6] = b.x, m[7] = b.y, m[8] = b.z;
    m[9] = b.w, m[10] = c.x, m[11] = c.y, m[12] = c.z, m[13] = c.w, m[14] = d.x, m[15] = d.y;
    h0 = d.z, h1 = d.w;
    b3_compress(cv, m, 64, B3_KEYED | (t == 0 ? B3_CHUNK_START : 0) | (t == 15 ? B3_CHUNK_END : 0), out);
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = out[i];
  }
  m[0] = h0, m[1] = h1;
#pragma unroll
  for (int i = 2; i < 16; ++i) m[i] = 0u;
  b3_compress(key, m, 8, B3_KEYED | B3_CHUNK_START | B3_CHUNK_END, c1, 1);
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = cv[i], m[8 + i] = c1[i];
  b3_compress(key, m, 64, B3_KEYED | B3_PARENT | B3_ROOT, out);
  uint4 *o = reinterpret_cast<uint4 *>(tags + u * 8);
  o[0] = make_uint4(out[0], out[1], out[2], out[3]);
  o[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// Two receipts a block, over 8 vectors each: BLAKE3-keyed(kRcKey[rc - 1], u64be(jbase + u) ‖ rc_sk[8 u .. 8 u + 7] as
// 128 int32 LE), 520 bytes in one chunk, so 9 blocks: the same 18 compressions a block of 16 as rc_tag16_kernel, twice
// the threads and half the chain.
__global__ void __launch_bounds__(128) rc_tag8_kernel(int rc, const int *__restrict__ sk, u32 *__restrict__ tags, u64 n,
                                                      u64 jbase) {
  const u64 u = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;
  const uint4 *p = reinterpret_cast<const uint4 *>(sk + u * 128);
  u32 m[16], cv[8], out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = kRcKey[rc - 1][i];
  const u64 j = jbase + u;
  u32 h0 = __byte_perm((u32)(j >> 32), 0, 0x0123), h1 = __byte_perm((u32)j, 0, 0x0123);
#pragma unroll 1
  for (int t = 0; t < 8; ++t) {
    const uint4 a = p[4 * t], b = p[4 * t + 1], c = p[4 * t + 2], d = p[4 * t + 3];
    m[0] = h0, m[1] = h1, m[2] = a.x, m[3] = a.y, m[4] = a.z, m[5] = a.w, m[6] = b.x, m[7] = b.y, m[8] = b.z;
    m[9] = b.w, m[10] = c.x, m[11] = c.y, m[12] = c.z, m[13] = c.w, m[14] = d.x, m[15] = d.y;
    h0 = d.z, h1 = d.w;
    b3_compress(cv, m, 64, B3_KEYED | (t == 0 ? B3_CHUNK_START : 0), out);
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = out[i];
  }
  m[0] = h0, m[1] = h1;
#pragma unroll
  for (int i = 2; i < 16; ++i) m[i] = 0u;
  b3_compress(cv, m, 8, B3_KEYED | B3_CHUNK_END | B3_ROOT, out);
  uint4 *o = reinterpret_cast<uint4 *>(tags + u * 8);
  o[0] = make_uint4(out[0], out[1], out[2], out[3]);
  o[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

#define KV_SHA512_K                                                                                                  \
  {0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull, \
   0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull, \
   0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, \
   0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull, \
   0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull, \
   0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, \
   0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, \
   0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull, \
   0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull, \
   0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, \
   0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, \
   0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull, \
   0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull, \
   0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, \
   0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, \
   0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull}
__constant__ u64 kSha512K[80] = KV_SHA512_K;
#ifndef __CUDA_ARCH__
static const u64 kSha512KHost[80] = KV_SHA512_K;
#endif
__constant__ u64 kRcMid[8];

__host__ __device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

// One SHA-512 compression of the big-endian words w (clobbered) into s (`tagger.cu`'s compress).
__host__ __device__ __forceinline__ void sha512_compress(u64 s[8], u64 w[16]) {
#ifdef __CUDA_ARCH__
  const u64 *K = kSha512K;
#else
  const u64 *K = kSha512KHost;
#endif
  u64 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (r) {
        const u64 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
        w[t] += (rotr64(w15, 1) ^ rotr64(w15, 8) ^ (w15 >> 7)) + w[(t + 9) & 15] +
                (rotr64(w2, 19) ^ rotr64(w2, 61) ^ (w2 >> 6));
      }
      const u64 t1 = h + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) + K[r + t] + w[t];
      const u64 t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
  }
  s[0] += a, s[1] += b, s[2] += c, s[3] += d, s[4] += e, s[5] += f, s[6] += g, s[7] += h;
}

__device__ __forceinline__ u64 be64(u32 lo, u32 hi) {
  return (u64)__byte_perm(lo, 0, 0x0123) << 32 | __byte_perm(hi, 0, 0x0123);
}

// The half-block receipt as SHA-512, the remote driver's hash (receipts.py's y, seed-first): SHA512(P ‖ u64be(jbase + u)
// ‖ rc_sk[8 u .. 8 u + 7] as 128 int32 LE), P one 128-byte block whose midstate kRcMid holds (kv_rc_sha_set). After P
// the message is 520 bytes, so 5 compressions with the padding; the digest's 64 bytes go to tags[16 u ..].
__global__ void __launch_bounds__(128) rc_sha8_kernel(const int *__restrict__ sk, u32 *__restrict__ tags, u64 n,
                                                      u64 jbase) {
  const u64 u = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;
  const uint4 *p = reinterpret_cast<const uint4 *>(sk + u * 128);
  u64 s[8], w[16], c = jbase + u;
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kRcMid[i];
#pragma unroll 1
  for (int t = 0; t < 4; ++t) {
    w[0] = c;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const uint4 q = p[8 * t + v];
      w[1 + 2 * v] = be64(q.x, q.y);
      if (v < 7) w[2 + 2 * v] = be64(q.z, q.w); else c = be64(q.z, q.w);
    }
    sha512_compress(s, w);
  }
  w[0] = c, w[1] = 0x8000000000000000ull;
#pragma unroll
  for (int i = 2; i < 15; ++i) w[i] = 0;
  w[15] = (128 + 520) * 8;
  sha512_compress(s, w);
  uint4 *o = reinterpret_cast<uint4 *>(tags + u * 16);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const u64 x = s[2 * i], y = s[2 * i + 1];
    o[i] = make_uint4(__byte_perm((u32)(x >> 32), 0, 0x0123), __byte_perm((u32)x, 0, 0x0123),
                      __byte_perm((u32)(y >> 32), 0, 0x0123), __byte_perm((u32)y, 0, 0x0123));
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

// The dynamic persistent form of mma-ws: a CTA pair an SM, each CTA folding mma-ws's CTAs' work items (sequence,
// split, KV head; item (seq splits + split) KV_HEADS + g) in the order the counter *ctr (zero at launch) hands them out,
// so no CTA starts or drains per item and none waits behind a static share. Item n + 1's id is in it[(n + 1) % 4] at
// item n's start: thread 0 takes items 0 and 1 before the loop, and the first decoder thread takes item n + 2 at item
// n's start and publishes it at item n + 1's, so the atomic's latency is a whole item. The decoder warps read it after
// a named barrier, the MMA warps after the full barrier of the item's first tile; a decoder lane loads item n + 1's
// first word during item n's last tile. Each item's partials are mma-ws's at the same splits, bit for bit.
constexpr int dyn_smem() { return mws_smem() + 4 * 4; }

__global__ void __launch_bounds__(MWT, 2) attn_dyn_kernel(const uint8_t *__restrict__ kv,
                                                         const uint16_t *__restrict__ q, float *__restrict__ part,
                                                         int ctx, int layers, int layer, int per_split, int splits,
                                                         int k, int nitems, unsigned long long *__restrict__ ctr) {
  if constexpr (MMA_OK) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint64_t *full = reinterpret_cast<uint64_t *>(sm + MNB * MT * MTS), *empty = full + MNB;
    u32 *td = reinterpret_cast<u32 *>(empty + MNB);
    volatile int *it = reinterpret_cast<volatile int *>(td + NL);
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const auto tokens = [&](int item) {
      const int t0 = item / KV_HEADS % splits * per_split;
      return min(ctx, t0 + per_split) - t0;
    };
    if (threadIdx.x == 0) {
      for (int b = 0; b < MNB; ++b) {
        mbar_init(full + b, 32 * MDW);
        mbar_init(empty + b, 32 * MW);
      }
      u32 s[NL];
      tweak((u64)MT * layers * TLW, s);
#pragma unroll
      for (int l = 0; l < NL; ++l) td[l] = s[l];
      it[0] = (int)atomicAdd(ctr, 1ULL);
      it[1] = (int)atomicAdd(ctr, 1ULL);
    }
    __syncthreads();
    if (w >= MW) {
      const int d = threadIdx.x - 32 * MW, t = d / HW, j = d % HW;
      const int wj = (j < HW / 2 ? 0 : TL_BYTES / 2 / WB - HW / 2) + j;
      const u64 stride = (u64)MT * layers * TLW;
      const auto word = [&](int item) {
        const int sq = item / KV_HEADS;
        return (((u64)(sq / splits) * ctx + sq % splits * per_split + t) * layers + layer) * TLW + wj +
               item % KV_HEADS * (HB / WB);
      };
      const u32 *kw = reinterpret_cast<const u32 *>(kv);
      int cur = it[0], gt = 0;
      unsigned long long pend = 0;
      u32 nx[NL], tw[NL];
      u64 w0 = cur < nitems ? word(cur) : 0;
      if (cur < nitems && t < min(MT, tokens(cur))) ldg_word(kw + w0 * NL, nx);
#pragma unroll 1
      for (int n = 0; cur < nitems; ++n) {
        if (d == 0) {
          if (n) it[(n + 1) & 3] = (int)pend;
          pend = atomicAdd(ctr, 1ULL);
        }
        asm volatile("bar.sync 1, %0;" ::"n"(32 * MDW) : "memory");
        const int nxt = it[(n + 1) & 3], rem = tokens(cur), nrem = nxt < nitems ? tokens(nxt) : 0;
        const u64 nw0 = nxt < nitems ? word(nxt) : 0;
        tweak(w0, tw);
        for (int tile = 0; tile * MT < rem; ++tile, ++gt) {
          const int nt = min(MT, rem - tile * MT);
          u32 x[NL];
          copy(x, nx);
          if ((tile + 1) * MT < rem) {
            if (t < min(MT, rem - (tile + 1) * MT)) ldg_word(kw + (w0 + (tile + 1) * stride) * NL, nx);
          } else if (t < min(MT, nrem)) {
            ldg_word(kw + nw0 * NL, nx);
          }
          if (tile) tweak_step(tw, td);
          if (t < nt) decode_word_t(x, tw, k);
          else
#pragma unroll
            for (int l = 0; l < NL; ++l) x[l] = 0u;
          if (gt >= MNB) mbar_wait(empty + gt % MNB, (gt / MNB - 1) & 1);
          store(reinterpret_cast<u32 *>(sm + (gt % MNB) * MT * MTS + t * MTS + j * WB), x);
          mbar_arrive(full + gt % MNB);
        }
        cur = nxt, w0 = nw0;
      }
      return;
    }
    u32 qa[8][2];
    float o[8][4];
    int cur = it[0], gt = 0;
#pragma unroll 1
    for (int n = 0; cur < nitems; ++n) {
      const int g = cur % KV_HEADS, sq = cur / KV_HEADS, rem = tokens(cur);
      float m_own = -INFINITY, l_own = 0.f;
#pragma unroll
      for (int mt = 0; mt < 8; ++mt)
#pragma unroll
        for (int v = 0; v < 4; ++v) o[mt][v] = 0.f;
      mma_load_q(q, sq / splits, g, lane, qa);
      int nxt = nitems;
#pragma unroll 1
      for (int tile = 0; tile * MT < rem; ++tile, ++gt) {
        const int b = gt % MNB, nt = min(MT, rem - tile * MT);
        mbar_wait(full + b, (gt / MNB) & 1);
        if (tile == 0) nxt = it[(n + 1) & 3];
        if (nt > 16 * w)
          mma_chunk(smem_u32(sm + b * MT * MTS + 16 * w * MTS), qa, lane, min(16, nt - 16 * w), o, m_own, l_own);
        mbar_arrive(empty + b);
      }
      mma_out(part + ((size_t)sq * MW + w) * PART, g, lane, o, m_own, l_own);
      cur = nxt;
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

int kv_mask() { return kv::MASK; }

// K: NL little-endian limbs, a b-bit constant (not reduced mod p).
int kv_set_mask(const uint32_t *mask) {
  if (cudaMemcpyToSymbol(kv::kMask, mask, sizeof(uint32_t) * kv::NL) != cudaSuccess) return launched();
  return 0;
}

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
  if (op < 0 || op > 5 || n < 1) return -1;
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
  cudaFuncSetAttribute(kv::attn_mma_ws_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mws_smem());
  cudaFuncSetAttribute(kv::attn_occ_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::occ_smem());
  cudaFuncSetAttribute(kv::attn_occ_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, kv::attn_mma_ws_kernel<0>, kv::MWT, kv::mws_smem());
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(out + 2, kv::attn_occ_kernel<true>, kv::OWT, kv::occ_smem());
  out[1] = kv::MDW;
  out[3] = kv::ODW;
  return launched();
}

static int *rc_sk = nullptr;
static uint32_t *rc_tags = nullptr, *rc_zdump = nullptr;

// Receipts' shape: tiles a block, the bound form's lag in tiles, z's words, decoder warps a CTA, tokens a tile, the
// sketch's steps a tile (a word's bytes / 16), and the forms this build runs (bit rc - 1).
void kv_rc_shape(int *out) {
  out[0] = kv::RT; out[1] = kv::RLAG; out[2] = kv::ZW; out[3] = kv::MDW; out[4] = kv::MT; out[5] = kv::RS;
  out[6] = (kv::RC1_OK ? 1 : 0) | (kv::RC2_OK ? 2 : 0);
}

// kv_attn forms 10 (rc = 1) and 11 (rc = 2): a block's sign rows (RT * RCW u32, kv_receipt.rows) and the BLAKE3 key
// (8 u32).
int kv_rc_set(int rc, const uint32_t *rows, const uint32_t *key) {
  if (rc < 1 || rc > 2) return -1;
  const size_t ns = sizeof(uint4) * kv::RT * kv::RCW / 4;
  if (cudaMemcpyToSymbol(kv::kRcB, rows, ns, (rc - 1) * ns) != cudaSuccess) return launched();
  if (cudaMemcpyToSymbol(kv::kRcKey, key, 32, (rc - 1) * 32) != cudaSuccess) return launched();
  return 0;
}

// Forms 10 and 11's outputs: sketches (16 int32 a vector), receipts (8 u32 a vector, nullptr none) and form 11's z
// (ZW words a tile of each CTA, nullptr none).
int kv_rc_buffers(int *sk, uint32_t *tags, uint32_t *zdump) {
  rc_sk = sk, rc_tags = tags, rc_zdump = zdump;
  return 0;
}

// The receipts of n vectors of kv_rc_buffers' sketches, alone.
int kv_rc_tag(int rc, uint64_t n, cudaStream_t s) {
  if (rc < 1 || rc > 2 || !rc_sk || !rc_tags) return -1;
  kv::rc_tag_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(rc, rc_sk, rc_tags, n, 0);
  return launched();
}

// rc_tag16_kernel's receipts of the n blocks of 16 vectors of kv_rc_buffers' sketches, into the first n receipt slots.
// 64 threads a CTA, so one CTA fits on an SM beside mma-ws's two (61,440 of its 65,536 registers).
int kv_rc_tag16(int rc, uint64_t n, cudaStream_t s) {
  if (rc < 1 || rc > 2 || !rc_sk || !rc_tags) return -1;
  kv::rc_tag16_kernel<<<(unsigned)((n + 63) / 64), 64, 0, s>>>(rc, rc_sk, rc_tags, n, 0);
  return launched();
}

// rc_tag8_kernel's receipts of the n halves (8 vectors) of blocks of kv_rc_buffers' sketches, 64 threads a CTA.
int kv_rc_tag8(int rc, uint64_t n, cudaStream_t s) {
  if (rc < 1 || rc > 2 || !rc_sk || !rc_tags) return -1;
  kv::rc_tag8_kernel<<<(unsigned)((n + 63) / 64), 64, 0, s>>>(rc, rc_sk, rc_tags, n, 0);
  return launched();
}

// rc_sha8_kernel's prefix P, one 128-byte block (kv_receipt.sha_prefix): its midstate, computed once a pass.
int kv_rc_sha_set(const uint8_t *prefix) {
  uint64_t s[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                   0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull}, w[16];
  for (int i = 0; i < 16; ++i) {
    w[i] = 0;
    for (int b = 0; b < 8; ++b) w[i] = w[i] << 8 | prefix[8 * i + b];
  }
  kv::sha512_compress(s, w);
  if (cudaMemcpyToSymbol(kv::kRcMid, s, sizeof s) != cudaSuccess) return launched();
  return 0;
}

// rc_sha8_kernel's receipts of the n halves (8 vectors) of blocks of kv_rc_buffers' sketches, 64 bytes each in the
// receipt slots (tags[16 u ..]), 64 threads a CTA.
int kv_rc_sha8(uint64_t n, cudaStream_t s) {
  if (!rc_sk || !rc_tags) return -1;
  kv::rc_sha8_kernel<<<(unsigned)((n + 63) / 64), 64, 0, s>>>(rc_sk, rc_tags, n, 0);
  return launched();
}

// Holds the first `bytes` of kv_rc_buffers' sketches in L2 on stream s (an access policy window over as much of the
// device's persisting L2 as they need), so attention's sketch writes and the receipts' kernel's reads stay off DRAM;
// bytes = 0 clears it. out: the window's bytes and the persisting L2 set aside.
int kv_rc_l2(uint64_t bytes, cudaStream_t s, uint64_t *out) {
  int dev = 0, max_persist = 0, max_window = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, dev);
  cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, dev);
  const size_t win = bytes && rc_sk ? std::min<size_t>(bytes, (size_t)max_window) : 0;
  const size_t persist = std::min<size_t>(win, (size_t)max_persist);
  if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist) != cudaSuccess) return launched();
  cudaStreamAttrValue a = {};
  a.accessPolicyWindow.base_ptr = win ? rc_sk : nullptr;
  a.accessPolicyWindow.num_bytes = win;
  a.accessPolicyWindow.hitRatio = win ? std::min(1.0f, (float)persist / (float)win) : 0.f;
  a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  if (cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &a) != cudaSuccess) return launched();
  if (!win && cudaCtxResetPersistingL2Cache() != cudaSuccess) return launched();
  out[0] = win, out[1] = persist;
  return launched();
}

// One layer's decode attention for `batch` sequences of `ctx` tokens (attn_kernel's layout): kv plain (encoded = 0) or
// its words (encoded = 1, k rounds; encoded = 2 the warp-specialized form, the same output); on the tensor cores,
// encoded = 3 plain, 4 its words and 5 its words in the warp-specialized form (4 and 5 the same output as 3, not as 0);
// encoded = 6 the tensor-core form at the decode's occupancy, plain, and 7 its words (the same output as 6); encoded =
// 8 the persistent warp-specialized form, plain, and 9 its words (the same output as 8), `splits` times the CTAs the
// SMs hold at once, fewer if a pair would span more than 16 of them; encoded = 10 and 11 mma-ws with the unbound and the
// bound receipt (the same output as 5 at the same split, in whole blocks of RT tiles), into kv_rc_buffers' sketches,
// and their receipts if it has a receipt buffer; encoded = 12 mma-ws's work items folded by a CTA pair an SM as a
// counter hands them out (the same output as 5 at the same split).
// q bf16 [batch][HEADS][128] pre-scaled by log2(e)/sqrt(128), out fp32 [batch][HEADS][128], part at least batch *
// splits * kv_attn_shape's out[6] floats (times out[7] from 3 on; batch * 16 * out[6] * out[7] for 8 and 9). The tokens
// are split into at most `splits` ranges of whole tiles.
int kv_attn(const void *kv_, int encoded, const uint16_t *q, float *out, float *part, int batch, int ctx, int layers,
            int layer, int splits, int k, cudaStream_t s) {
  if (batch < 1 || ctx < 1 || splits < 1 || layer < 0 || layer >= layers || encoded < 0 || encoded > 12 ||
      (encoded != 0 && encoded != 3 && encoded != 6 && encoded != 8 && (k < 1 || k > kv::KMAX)))
    return -1;
  if (encoded == 12) {
    if (!kv::MMA_OK) return -2;
    static int slots = 0;
    static unsigned long long *ctr = nullptr;
    if (!slots) {
      int dev, sms, occ;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      cudaFuncSetAttribute(kv::attn_dyn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::dyn_smem());
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kv::attn_dyn_kernel, kv::MWT, kv::dyn_smem());
      if (occ < 1 || cudaMalloc(&ctr, sizeof *ctr) != cudaSuccess) return -3;
      slots = sms * occ;
    }
    const int per = (ctx + splits - 1) / splits, per_split = (per + kv::MT - 1) / kv::MT * kv::MT;
    const int sp = (ctx + per_split - 1) / per_split, nitems = batch * sp * kv::KV_HEADS;
    if (cudaMemsetAsync(ctr, 0, sizeof *ctr, s) != cudaSuccess) return launched();
    kv::attn_dyn_kernel<<<std::min(slots, nitems), kv::MWT, kv::dyn_smem(), s>>>(
        static_cast<const uint8_t *>(kv_), q, part, ctx, layers, layer, per_split, sp, k, nitems, ctr);
    kv::attn_combine<<<batch * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, sp * kv::MW);
    return launched();
  }
  if (encoded >= 10) {
    if (!kv::MMA_OK || (encoded == 10 && !kv::RC1_OK) || (encoded == 11 && !kv::RC2_OK)) return -2;
    if (!rc_sk) return -1;
    static bool set = false;
    if (!set) {
      cudaFuncSetAttribute(kv::attn_mma_ws_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::rc_smem(1));
      cudaFuncSetAttribute(kv::attn_mma_ws_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::rc_smem(2));
      set = true;
    }
    const int bt = kv::RT * kv::MT, per = (ctx + splits - 1) / splits, per_split = (per + bt - 1) / bt * bt;
    const int sp = (ctx + per_split - 1) / per_split, grid = batch * sp * kv::KV_HEADS, rc = encoded - 9;
    const auto *p = static_cast<const uint8_t *>(kv_);
    if (rc == 1)
      kv::attn_mma_ws_kernel<1><<<grid, kv::MWT, kv::rc_smem(1), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k,
                                                                       rc_sk, nullptr);
    else
      kv::attn_mma_ws_kernel<2><<<grid, kv::MWT, kv::rc_smem(2), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k,
                                                                       rc_sk, rc_zdump);
    kv::attn_combine<<<batch * kv::HEADS, 32 * kv::CW, 0, s>>>(part, out, sp * kv::MW);
    if (rc_tags) {
      const uint64_t n = (uint64_t)batch * kv::KV_HEADS * ((ctx + bt - 1) / bt) * kv::MDW * 16;
      kv::rc_tag_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(rc, rc_sk, rc_tags, n, 0);
    }
    return launched();
  }
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
      cudaFuncSetAttribute(kv::attn_mma_ws_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kv::mws_smem());
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
      kv::attn_mma_ws_kernel<0><<<grid, kv::MWT, kv::mws_smem(), s>>>(p, q, part, ctx, layers, layer, per_split, sp, k);
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
