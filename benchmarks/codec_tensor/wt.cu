// Sloth over one wide word on the RTX PRO 6000 (sm_120): `wide.py` is the reference. One build per field, WT_FIELD = n
// (255, 511, 521, 767, 1023, 1279, or the full-slot 512 and 768; p = 2^n - c as in `wide.FIELDS`).
//
// Two squarings mod p:
//   CUDA cores   a thread per word, NL 32-bit limbs; the square column by column (mad.lo.cc / madc.hi.cc / addc, each
//                unordered pair once, doubled), then two folds 2^n = c: a lazy value < 2^n + 2^25.
//   tensor cores (S = 32U bytes only) a warp per word, its bytes the limbs; the square's byte coefficients are 2 (U + 1)
//                int8 mma.m16n8k32 (`wide.tc_plan`): A_w[r][v] = a_(32w + r - v) from four byte-shifted copies of the
//                reversed bytes in shared memory (one aligned 32-bit load a register), B[v][s] = a_(32(s - w) + v) (one
//                load and a byte swap). The coefficients go through shared memory to the owner layout (lane l holds bytes
//                U l .. U l + U - 1, a digit of 8U bits), are folded at byte S (2^(8S) = 2c mod p), and the carries are
//                resolved across lanes: shuffles while they are large, then a carry-lookahead over two ballots.
//
// Kernels (extern "C" launchers at the end): encode / decode of a flat store (words of S bytes, contiguous) on CUDA cores
// or (decode) tensor cores; the read of a store; compute-only chains for the squaring, decode and encode rates on CUDA
// cores, the squaring on tensor cores, and both at once in one CTA (the adversary's rate with both units); the FP8 GEMM in
// its warp-specialized form (`benchmarks/word_codec/wc.cu`'s gemm_ws) on plain fragments, with a knob of extra int8 MMAs
// or IMADs a tile (the tensor-core and CUDA-core budget of the memory-bound step), and fused with the decode of each task.
#include <cstdint>
#include <cuda_runtime.h>

#ifndef WT_FIELD
#define WT_FIELD 1279
#endif

namespace wt {

using u32 = uint32_t;
using u64 = uint64_t;

#if WT_FIELD == 255
constexpr int NBITS = 255, TW = 265;
constexpr u32 CF = 765;
#elif WT_FIELD == 511
constexpr int NBITS = 511, TW = 131;
constexpr u32 CF = 481;
#elif WT_FIELD == 521
constexpr int NBITS = 521, TW = 128;
constexpr u32 CF = 1;
#elif WT_FIELD == 767
constexpr int NBITS = 767, TW = 87;
constexpr u32 CF = 1081;
#elif WT_FIELD == 1023
constexpr int NBITS = 1023, TW = 65;
constexpr u32 CF = 361;
#elif WT_FIELD == 1279
constexpr int NBITS = 1279, TW = 52;
constexpr u32 CF = 1;
#elif WT_FIELD == 512
constexpr int NBITS = 512, TW = 128;
constexpr u32 CF = 569;
#elif WT_FIELD == 768
constexpr int NBITS = 768, TW = 86;
constexpr u32 CF = 825;
#else
#error "WT_FIELD is one of wide.FIELDS"
#endif
// A full-slot field (32 | n) has no spare top bits: values are lazy below 2^n, and the payload is the whole word.
constexpr bool FULL_SLOT = NBITS % 32 == 0;
constexpr int NL = (NBITS + 31) / 32, S = (NBITS + 7) / 8, PB = FULL_SLOT ? S : (NBITS - 1) / 8, U = (S + 31) / 32;
constexpr int Q = NBITS / 32, R = NBITS % 32;
constexpr u32 TOPMASK = (1u << R) - 1;
constexpr bool TC_OK = S == 32 * U && NBITS == 8 * S - 1;
static_assert(FULL_SLOT ? Q == NL : (R > 0 && Q == NL - 1), "the top limb holds bits 32Q .. n - 1");
static_assert(CF % 4 == 1 && CF < 4096, "p = 3 mod 4, c small");
static_assert(TW * PB >= 8192 && (TW * S) % 16 == 0, "a task's words");
constexpr int MAXK = 2;
constexpr int TASK_BYTES = 8192;
constexpr int SB = TW * S;                              // a task's stored bytes

__constant__ u32 kC[MAXK][NL];
__constant__ u32 kTau[NL];
__constant__ u32 kP[NL];
#ifdef WT_DEBUG_DUMP
__device__ uint8_t *g_dbg;   // the fused MMA warp's operand bytes, TASK_BYTES a store task
#endif

__host__ __device__ constexpr int bitlen(u64 x) { return x ? 1 + bitlen(x >> 1) : 0; }

// --- CUDA cores: a thread per word ----------------------------------------------------------------------------------

__device__ __forceinline__ void mac(u32 *r, u32 a, u32 b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void add3(u32 *r, u32 a, u32 b) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, 0;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b));
}

// t = a^2, a < 2^(32 NL): column m sums each unordered pair once, doubles, adds the diagonal and the carry in.
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

// x = t mod p, lazily: (t mod 2^n) + c (t >> n), twice. t < 2^(2n + 2) gives x < 2^n + 2^25; at full slot, t < 2^(2n)
// gives x < 2^n, the second fold's carry out of 2^n (then x < c^2) added back as c.
__device__ __forceinline__ void fold(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  if constexpr (FULL_SLOT) {
    u64 carry = 0;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      const u64 v = (u64)t[NL + k] * CF + t[k] + carry;
      x[k] = (u32)v;
      carry = v >> 32;
    }
    u64 c = carry * CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      c += x[k];
      x[k] = (u32)c;
      c >>= 32;
    }
    x[0] += (u32)c * CF;
  } else {
    constexpr int QY = Q % NL;
    u32 y[NL + 1];
    u64 carry = 0;
#pragma unroll
    for (int k = 0; k <= NL; ++k) {
      const u32 nxt = Q + k + 1 < 2 * NL ? t[(Q + k + 1) % (2 * NL)] : 0u;
      const u32 hi = __funnelshift_r(t[(Q + k) % (2 * NL)], nxt, R);
      const u32 lo = k < Q ? t[k] : (k == Q ? (t[QY] & TOPMASK) : 0u);
      const u64 v = (u64)hi * CF + lo + carry;
      y[k] = (u32)v;
      carry = v >> 32;
    }
    u64 c = (u64)__funnelshift_r(y[QY], y[QY + 1], R) * CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      c += k == Q ? (y[QY] & TOPMASK) : y[k];
      x[k] = (u32)c;
      c >>= 32;
    }
  }
}

__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 t[2 * NL];
  sqr_full(a, t);
  fold(t, x);
}

__device__ __forceinline__ void mul(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  u32 t[2 * NL];
  mul_full(a, b, t);
  fold(t, x);
}

__device__ __forceinline__ void copy(u32 (&d)[NL], const u32 (&s)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) d[k] = s[k];
}

// x < 2p to [0, p): x - p = x + c - 2^n when x + c >= 2^n.
__device__ __forceinline__ void canon(u32 (&x)[NL]) {
  u32 y[NL];
  u64 c = CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += x[k];
    y[k] = (u32)c;
    c >>= 32;
  }
  bool ge;
  if constexpr (FULL_SLOT) {
    ge = c != 0;
  } else {
    ge = (y[Q % NL] >> R) != 0;
    y[Q % NL] &= TOPMASK;
  }
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? y[k] : x[k];
}

// x = a - b mod p, a and b canonical.
__device__ __forceinline__ void submod(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  u64 br = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    const u64 v = (u64)a[k] - b[k] - br;
    x[k] = (u32)v;
    br = (v >> 32) & 1;
  }
  const u32 m = 0u - (u32)br;
  u64 c = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += (u64)x[k] + (kP[k] & m);
    x[k] = (u32)c;
    c >>= 32;
  }
}

// x = a + b mod p, a and b canonical. At full slot a + b may pass 2^n, and is then at least p.
__device__ __forceinline__ void addmod(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  u64 c = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += (u64)a[k] + b[k];
    x[k] = (u32)c;
    c >>= 32;
  }
  if constexpr (FULL_SLOT) {
    u32 y[NL];
    u64 d = CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      d += x[k];
      y[k] = (u32)d;
      d >>= 32;
    }
    const bool ge = (c | d) != 0;
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] = ge ? y[k] : x[k];
  } else {
    canon(x);
  }
}

__device__ __forceinline__ bool is_zero(const u32 (&x)[NL]) {
  u32 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) v |= x[k];
  return v == 0;
}

__device__ __forceinline__ bool equal(const u32 (&a)[NL], const u32 (&b)[NL]) {
  u32 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) v |= a[k] ^ b[k];
  return v == 0;
}

// x = p - x for x canonical (0 stays 0).
__device__ __forceinline__ void negate(u32 (&x)[NL]) {
  const bool z = is_zero(x);
  u64 br = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    const u64 v = (u64)kP[k] - x[k] - br;
    x[k] = z ? 0u : (u32)v;
    br = (v >> 32) & 1;
  }
}

// Carry chains in PTX (one SASS instruction a limb, against two to three for the u64 forms above). The flag lives
// between consecutive asm volatile statements, which the compiler keeps in order.
__device__ __forceinline__ u32 add_cc(u32 a, u32 b) { u32 r; asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc_cc(u32 a, u32 b) { u32 r; asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 addc(u32 a, u32 b) { u32 r; asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 sub_cc(u32 a, u32 b) { u32 r; asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc_cc(u32 a, u32 b) { u32 r; asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 subc(u32 a, u32 b) { u32 r; asm volatile("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }

// x < 2p to [0, p), as canon.
__device__ __forceinline__ void canon_cc(u32 (&x)[NL]) {
  u32 y[NL];
  y[0] = add_cc(x[0], CF);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) y[k] = addc_cc(x[k], 0u);
  bool ge;
  if constexpr (FULL_SLOT) {
    y[NL - 1] = addc_cc(x[NL - 1], 0u);
    ge = addc(0u, 0u) != 0;
  } else {
    y[NL - 1] = addc(x[NL - 1], 0u);
    ge = (y[Q % NL] >> R) != 0;
    y[Q % NL] &= TOPMASK;
  }
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? y[k] : x[k];
}

// x = a + b mod p, a + b < 2p, as addmod.
__device__ __forceinline__ void addmod_cc(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  u32 s[NL], y[NL];
  s[0] = add_cc(a[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) s[k] = addc_cc(a[k], b[k]);
  const u32 co = addc(0u, 0u);
  y[0] = add_cc(s[0], CF);
#pragma unroll
  for (int k = 1; k < NL; ++k) y[k] = addc_cc(s[k], 0u);
  const bool ge = (co | addc(0u, 0u)) != 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? y[k] : s[k];
}

// x = a + b, no carry out (a + b < 2^(32 NL)).
__device__ __forceinline__ void add_nc(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  x[0] = add_cc(a[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) x[k] = addc_cc(a[k], b[k]);
  x[NL - 1] = addc(a[NL - 1], b[NL - 1]);
}

// q = p - q when neg, q canonical (so 0 becomes p).
__device__ __forceinline__ void neg_if(u32 (&q)[NL], bool neg) {
  u32 z[NL];
  z[0] = sub_cc(kP[0], q[0]);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) z[k] = subc_cc(kP[k], q[k]);
  z[NL - 1] = subc(kP[NL - 1], q[NL - 1]);
#pragma unroll
  for (int k = 0; k < NL; ++k) q[k] = neg ? z[k] : q[k];
}

// x = a - b mod p, a and b canonical, as submod.
__device__ __forceinline__ void submod_cc(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  x[0] = sub_cc(a[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = subc_cc(a[k], b[k]);
  const u32 m = subc(0u, 0u);
  x[0] = add_cc(x[0], kP[0] & m);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) x[k] = addc_cc(x[k], kP[k] & m);
  x[NL - 1] = addc(x[NL - 1], kP[NL - 1] & m);
}

__device__ __forceinline__ void sqr_n(u32 (&x)[NL], int n) {
#pragma unroll 1
  for (int i = 0; i < n; ++i) {
    u32 y[NL];
    sqr(x, y);
    copy(x, y);
  }
}

// s = a^((p+1)/4), a canonical; s lazy. (p+1)/4 = 2^(n-2) - d, d = (c-1)/4: for d > 0, with 2^m > d,
// s = (a^(2^L - 1))^(2^m) a^(2^m - d), L = n - 2 - m; a^(2^L - 1) along L's bits (e_(2j) = e_j^(2^j) e_j,
// e_(j+1) = e_j^2 a), a^(2^m - d) by square and multiply.
__device__ __forceinline__ void pow_root(const u32 (&a)[NL], u32 (&s)[NL]) {
  constexpr u32 D = (CF - 1) / 4;
  if constexpr (D == 0) {
    copy(s, a);
    sqr_n(s, NBITS - 2);
  } else {
    constexpr int M = bitlen(D), L = NBITS - 2 - M, LB = bitlen(L);
    constexpr u32 E = (1u << M) - D;
    constexpr int EB = bitlen(E);
    u32 r[NL], t[NL];
    copy(r, a);
    int cur = 1;
#pragma unroll 1
    for (int bit = LB - 2; bit >= 0; --bit) {
      copy(t, r);
      sqr_n(t, cur);
      mul(t, r, r);
      cur *= 2;
      if ((L >> bit) & 1) {
        sqr(r, t);
        mul(t, a, r);
        cur += 1;
      }
    }
    sqr_n(r, M);
    u32 h[NL];
    copy(h, a);
#pragma unroll 1
    for (int bit = EB - 2; bit >= 0; --bit) {
      sqr(h, t);
      if ((E >> bit) & 1) mul(t, a, h);
      else copy(h, t);
    }
    mul(r, h, s);
  }
}

// rho(a), a canonical: the even root of a square, the odd root of a non-square.
__device__ __forceinline__ void rho(const u32 (&a)[NL], u32 (&y)[NL]) {
  u32 s[NL], q[NL];
  pow_root(a, s);
  canon(s);
  sqr(s, q);
  canon(q);
  const bool square = equal(q, a);
  const bool odd = s[0] & 1;
  if (odd == square) negate(s);
  copy(y, s);
}

// rho^-1(y), y canonical: y^2, negated when y is odd.
__device__ __forceinline__ void rho_inv(const u32 (&y)[NL], u32 (&x)[NL]) {
  sqr(y, x);
  canon(x);
  if (y[0] & 1) negate(x);
}

__device__ __forceinline__ void load_key(int j, u32 (&c)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) c[k] = kC[j][k];
}

template <int K>
__device__ __forceinline__ void decode_word(u32 (&y)[NL], const u32 (&t)[NL]) {
#pragma unroll
  for (int j = K - 1; j >= 0; --j) {
    u32 c[NL], x[NL];
    load_key(j, c);
    rho_inv(y, x);
    submod(x, c, y);
    submod(y, t, x);
    copy(y, x);
  }
}

// One round's decode against w = -(c_0 + t) mod p (canonical), which the caller keeps in place of t:
// rho^-1(y) - c_0 - t = (y^2 negated when y is odd) + w.
__device__ __forceinline__ void decode_word_w(u32 (&y)[NL], const u32 (&w)[NL]) {
  u32 q[NL];
  sqr(y, q);
  canon_cc(q);
  neg_if(q, y[0] & 1);
  if constexpr (FULL_SLOT) {
    addmod_cc(q, w, y);
  } else {
    add_nc(q, w, y);
    canon_cc(y);
  }
}

// w = -(c_0 + t) mod p.
__device__ __forceinline__ void w_of(const u32 (&t)[NL], u32 (&w)[NL]) {
  u32 c[NL];
  load_key(0, c);
  addmod(c, t, w);
  negate(w);
}

template <int K>
__device__ __forceinline__ void encode_word(u32 (&x)[NL], const u32 (&t)[NL]) {
#pragma unroll 1
  for (int j = 0; j < K; ++j) {
    u32 c[NL], u[NL], a[NL];
    load_key(j, c);
    addmod(c, t, u);
    addmod(x, u, a);
    rho(a, x);
  }
}

// x = a i mod p, a canonical, i < 2^64.
__device__ __forceinline__ void mul_small(const u32 (&a)[NL], u64 i, u32 (&x)[NL]) {
  u32 t[2 * NL];
  const u32 lo = (u32)i, hi = (u32)(i >> 32);
  u64 c = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += (u64)a[k] * lo;
    t[k] = (u32)c;
    c >>= 32;
  }
  t[NL] = (u32)c;
  c = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += (u64)a[k] * hi + t[k + 1];
    t[k + 1] = (u32)c;
    c >>= 32;
  }
  t[NL + 1] = (u32)c;
#pragma unroll
  for (int k = NL + 2; k < 2 * NL; ++k) t[k] = 0;
  fold(t, x);
  canon(x);
}

__device__ __forceinline__ void tweak(u64 i, u32 (&t)[NL]) {
  u32 a[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) a[k] = kTau[k];
  mul_small(a, i, t);
}

// --- a flat store: words of S bytes, contiguous; a CTA stages WPB words through shared memory -------------------------

constexpr int WPB = 128;
constexpr int STAGE_BYTES = ((WPB * S + 15) / 16) * 16;

__device__ __forceinline__ void stage_words(const uint8_t *__restrict__ store, u64 first, u64 n, uint8_t *sm) {
  const u64 words = n - first < (u64)WPB ? n - first : (u64)WPB;
  const uint4 *src = reinterpret_cast<const uint4 *>(store + first * S);
  uint4 *dst = reinterpret_cast<uint4 *>(sm);
  const int vecs = (int)((words * S + 15) / 16);
  for (int e = threadIdx.x; e < vecs; e += blockDim.x) dst[e] = __ldcs(src + e);
}

// A word's limbs from shared memory (b 16-byte aligned when S % 16 == 0, else 2-byte aligned).
__device__ __forceinline__ void load_word_smem(const uint8_t *b, u32 (&x)[NL]) {
  if constexpr (S % 16 == 0) {
    const uint4 *w = reinterpret_cast<const uint4 *>(b);
#pragma unroll
    for (int k = 0; k < NL / 4; ++k) {
      const uint4 v = w[k];
      x[4 * k] = v.x; x[4 * k + 1] = v.y; x[4 * k + 2] = v.z; x[4 * k + 3] = v.w;
    }
  } else {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(b);
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      const u32 lo = 2 * k < S / 2 ? h[2 * k] : 0u, hi = 2 * k + 1 < S / 2 ? h[2 * k + 1] : 0u;
      x[k] = lo | hi << 16;
    }
  }
}

__device__ __forceinline__ u64 fold64(const u32 (&x)[NL]) {
  u64 h = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) h ^= (u64)x[k] << (32 * (k & 1));
  return h;
}

// mode 0: decode and XOR (checksum); mode 2: also write the result's limbs (out: NL u32 a word).
template <int K, int MODE>
__global__ void __launch_bounds__(WPB) decode_kernel(const uint8_t *__restrict__ store, u32 *__restrict__ out, u64 n,
                                                     u64 base, unsigned long long *xr) {
  __shared__ __align__(16) uint8_t sm[STAGE_BYTES];
  u64 acc = 0;
  u32 t[NL], step[NL];
  {
    u32 a[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) a[k] = kTau[k];
    mul_small(a, (u64)gridDim.x * WPB, step);
    mul_small(a, base + (u64)blockIdx.x * WPB + threadIdx.x, t);
  }
  if constexpr (K == 1) {
    u32 w[NL];
    w_of(t, w);
    copy(t, w);
  }
  for (u64 first = (u64)blockIdx.x * WPB; first < n; first += (u64)gridDim.x * WPB) {
    __syncthreads();
    stage_words(store, first, n, sm);
    __syncthreads();
    const u64 i = first + threadIdx.x;
    if (i < n) {
      u32 y[NL];
      load_word_smem(sm + threadIdx.x * S, y);
      if constexpr (K == 1) decode_word_w(y, t);
      else decode_word<K>(y, t);
      if (MODE == 2) {
#pragma unroll
        for (int k = 0; k < NL; ++k) out[i * NL + k] = y[k];
      }
      acc ^= fold64(y);
    }
    u32 t2[NL];
    if constexpr (K == 1) submod_cc(t, step, t2);
    else addmod(t, step, t2);
    copy(t, t2);
  }
  atomicXor(xr, (unsigned long long)acc);
}

__global__ void __launch_bounds__(256) read_kernel(const uint4 *__restrict__ store, u64 n16, unsigned long long *xr) {
  u64 acc = 0;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (u64)gridDim.x * blockDim.x) {
    const uint4 v = __ldcs(store + i);
    acc ^= ((u64)v.x | (u64)v.y << 32) ^ ((u64)v.z << 7 | (u64)v.w << 39);
  }
  atomicXor(xr, (unsigned long long)acc);
}

// in: NL u32 limbs a payload; store: S bytes a word.
template <int K>
__global__ void __launch_bounds__(WPB) encode_kernel(const u32 *__restrict__ in, uint8_t *__restrict__ store, u64 n,
                                                     u64 base) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) {
    u32 x[NL], t[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) x[k] = in[i * NL + k];
    tweak(base + i, t);
    encode_word<K>(x, t);
    uint8_t *o = store + i * S;
#pragma unroll
    for (int b = 0; b < S; ++b) o[b] = (uint8_t)(x[b / 4] >> (8 * (b % 4)));
  }
}

// --- compute-only chains, CUDA cores ----------------------------------------------------------------------------------

__device__ __forceinline__ void seed_word(u64 id, u32 (&x)[NL]) {
  u64 s = id * 0x9E3779B97F4A7C15ull + 1;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    s ^= s >> 29; s *= 0xBF58476D1CE4E5B9ull; s ^= s >> 32;
    x[k] = (u32)s;
  }
  if constexpr (FULL_SLOT) x[NL - 1] &= 0x7FFFFFFFu;
  else x[Q % NL] &= TOPMASK >> 1;
}

// what 0: `steps` squarings of ILP words a thread (lazy); 1: `steps` decodes of K rounds; 2: `steps` encodes of K rounds.
template <int WHAT, int K, int ILP>
__global__ void __launch_bounds__(128, 1) chain_kernel(u64 steps, unsigned long long *xr) {
  const u64 id = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u32 x[ILP][NL], t[NL];
#pragma unroll
  for (int w = 0; w < ILP; ++w) seed_word(id * ILP + w, x[w]);
  if (WHAT != 0) {
    canon(x[0]);
    tweak(id, t);
  }
#pragma unroll 1
  for (u64 s = 0; s < steps; ++s) {
    if (WHAT == 0) {
#pragma unroll
      for (int w = 0; w < ILP; ++w) {
        u32 y[NL];
        sqr(x[w], y);
        copy(x[w], y);
      }
    } else if (WHAT == 1) {
      if constexpr (K == 1) decode_word_w(x[0], t);
      else decode_word<K>(x[0], t);
    } else {
      encode_word<K>(x[0], t);
    }
  }
  u64 h = 0;
#pragma unroll
  for (int w = 0; w < ILP; ++w) h ^= fold64(x[w]) * (2 * w + 1);
  atomicXor(xr, (unsigned long long)h);
}

// --- tensor cores: a warp per word (S = 32U) --------------------------------------------------------------------------

constexpr u32 FULL = 0xFFFFFFFFu;
constexpr int DB = 8 * U;                               // digit bits a lane
constexpr u64 MASK = DB == 64 ? ~0ull : (1ull << DB) - 1;
constexpr u32 FW = 2 * CF;                              // 2^(8S) = 2^(n+1) = 2c (mod p) when S = 32U
constexpr int SHUF_ROUNDS = U == 1 ? 11 : U == 2 ? 6 : U == 3 ? 5 : U == 4 ? 4 : 3;
static_assert(U <= 5, "the plan is wide.tc_plan's for U <= 5");

__device__ __forceinline__ void mma_u8(int32_t (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int NGROUPS = U <= 4 ? 1 : 2;
__host__ __device__ constexpr int tc_cb(int g) { return g == 0 ? 0 : 4; }
__host__ __device__ constexpr int tc_group(int w) { return w + U <= 8 ? 0 : 1; }

// The reversed bytes G[x] = a_(GM - x), GM = 32U + 35 (3 mod 4), zero outside the word, in four copies G_s[y] = G[y + s]
// so a lane's A register is one aligned load; B registers come from G_0 byte-swapped.
constexpr int GM = 32 * U + 35;
constexpr int GLEN = 32 * U + 68;
constexpr int GSTRIDE = ((GLEN + 31) / 32) * 32 + 16;
constexpr int CSTRIDE = 36;                             // coefficient slots a column (conflict-free C stores)
constexpr int CO = CSTRIDE * 8;                         // slots a group
constexpr int TC_WORD_BYTES = 4 * GSTRIDE + 4 * CO * NGROUPS;
static_assert(GM % 4 == 3, "B's loads are aligned");
static_assert(GSTRIDE % 16 == 0, "copies are aligned");

struct TcScratch {
  uint8_t *g;
  int32_t *co;
};

__device__ __forceinline__ TcScratch tc_scratch(uint8_t *base) {
  TcScratch s;
  s.g = base;
  s.co = reinterpret_cast<int32_t *>(base + 4 * GSTRIDE);
  return s;
}

__device__ __forceinline__ void tc_init(TcScratch s, int lane) {
  u32 *g = reinterpret_cast<u32 *>(s.g);
  for (int e = lane; e < GSTRIDE; e += 32) g[e] = 0;
  for (int e = lane; e < CO * NGROUPS; e += 32) s.co[e] = 0;
  __syncwarp();
}

// Lane l's digit (bytes U l .. U l + U - 1) into the copies: byte i goes to G[GM - i], copy c at GM - i - c.
__device__ __forceinline__ void tc_put(TcScratch s, u64 v, int lane) {
#pragma unroll
  for (int b = 0; b < U; ++b) {
    const int i = U * lane + b;
    const uint8_t byte = (uint8_t)(v >> (8 * b));
#pragma unroll
    for (int c = 0; c < 4; ++c) s.g[c * GSTRIDE + GM - i - c] = byte;
  }
  __syncwarp();
}

__device__ __forceinline__ int tc_slot(int m, int cb) { return CSTRIDE * ((m >> 5) - cb) + (m & 31); }

// The square's byte coefficients into s.co: group grp's accumulator holds columns cb .. cb + 7 (coefficient
// m = 32 (cb + col) + r at slot CSTRIDE col + r).
__device__ __forceinline__ void tc_square_coeffs(TcScratch s, int lane) {
  const int g = lane >> 2, q = lane & 3;
  const int sig = (GM - g) & 3;
  const u32 *ga = reinterpret_cast<const u32 *>(s.g + sig * GSTRIDE);
  const u32 *g0 = reinterpret_cast<const u32 *>(s.g);
  // A_w[h] reg e = a_base .. a_(base - 3), base = 16 j + g + 8 (e & 1) - 4q, j = 2w + h - (e >> 1) in [-1, 2U + 1]
  u32 areg[2 * U + 3][2];
#pragma unroll
  for (int j = -1; j <= 2 * U + 1; ++j)
#pragma unroll
    for (int par = 0; par < 2; ++par) {
      const int base = 16 * j + g + 8 * par - 4 * q;
      areg[j + 1][par] = ga[(GM - base - sig) >> 2];
    }
  // B for delta = cb - w in [-U, 0]: column s = cb + g, k = 4q + b (reg 0), 16 + 4q + b (reg 1); a_x .. a_(x+3) is the
  // byte swap of G_0's aligned word at GM - x - 3.
  u32 breg[U + 1][2];
#pragma unroll
  for (int dd = 0; dd <= U; ++dd) {
    const int x0 = 32 * (g - dd) + 4 * q;
    const int y0 = GM - x0 - 3, y1 = GM - x0 - 19;
    breg[dd][0] = (y0 >= 0 && y0 < GLEN) ? __byte_perm(g0[y0 >> 2], 0, 0x0123) : 0u;
    breg[dd][1] = (y1 >= 0 && y1 < GLEN) ? __byte_perm(g0[y1 >> 2], 0, 0x0123) : 0u;
  }
  int32_t acc[NGROUPS][2][4];
#pragma unroll
  for (int a = 0; a < NGROUPS; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) acc[a][h][0] = acc[a][h][1] = acc[a][h][2] = acc[a][h][3] = 0;
#pragma unroll
  for (int w = 0; w <= U; ++w) {
    const int grp = tc_group(w), dd = w - tc_cb(grp);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 a[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = areg[2 * w + h - (e >> 1) + 1][e & 1];
      mma_u8(acc[grp][h], a, breg[dd][0], breg[dd][1]);
    }
  }
  // C fragment: reg e at row g + 8 (e >> 1) (+ 16 h), column 2q + (e & 1).
#pragma unroll
  for (int grp = 0; grp < NGROUPS; ++grp)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = 2 * q + (e & 1), r = 16 * h + g + 8 * (e >> 1);
        s.co[grp * CO + CSTRIDE * col + r] = acc[grp][h][e];
      }
  __syncwarp();
}

__device__ __forceinline__ int64_t tc_coeff(TcScratch s, int m) {   // c_m, 0 <= m < 64U
  int64_t c = m < 256 ? s.co[tc_slot(m, 0)] : 0;
  if (NGROUPS == 2 && m >= 32 * tc_cb(1)) c += s.co[CO + tc_slot(m, tc_cb(1))];
  return c;
}

// Digits < 2^(DB + 1) to digits < 2^DB by a carry-lookahead; `out` the carry out of lane 31.
__device__ __forceinline__ u64 lookahead(u64 v, int lane, u32 &out) {
  const u32 gen = __ballot_sync(FULL, v > MASK);
  v &= MASK;
  const u32 prop = __ballot_sync(FULL, v == MASK);
  const u64 a = (u64)(gen | prop);
  const u64 sm = a + gen;
  const u32 cin = (u32)((sm ^ a ^ gen) >> lane) & 1u;
  out = (u32)(sm >> 32);
  return (v + cin) & MASK;
}

// v + k in lane 0 (and k's bits past DB in lane 1), k < 2^16.
__device__ __forceinline__ u64 add_low(u64 v, int lane, u32 k) {
  if (lane == 0) v += k & MASK;
  if (DB < 16 && lane == 1) v += (u64)k >> DB;
  return v;
}

// The square mod p in the owner layout, lazily: the coefficients folded at byte S, carried; digits < 2^DB (value
// < 2^(8S)), congruent to a^2.
__device__ __forceinline__ u64 tc_square(TcScratch s, u64 v, int lane) {
  tc_put(s, v, lane);
  tc_square_coeffs(s, lane);
  u64 acc = 0;
#pragma unroll
  for (int b = U - 1; b >= 0; --b) {
    const int k = U * lane + b;
    const int64_t d = tc_coeff(s, k) + (int64_t)FW * tc_coeff(s, k + S);
    acc = (acc << 8) + (u64)d;
  }
#pragma unroll
  for (int round = 0; round < SHUF_ROUNDS; ++round) {
    const u64 c = acc >> DB;
    acc &= MASK;
    u64 in = __shfl_up_sync(FULL, c, 1);
    const u64 top = __shfl_sync(FULL, c, 31);
    if (lane == 0) in = top * FW;
    acc += in;
  }
  u32 out;
  acc = lookahead(acc, lane, out);
  if (out) {
    acc = lookahead(add_low(acc, lane, FW), lane, out);
    if (out) acc = lookahead(add_low(acc, lane, FW), lane, out);
  }
  return acc;
}

// v < 2^(8S) = 2^(n+1) to [0, p): fold bit n (x c), then subtract p once if v + c >= 2^n.
__device__ __forceinline__ u64 tc_canon(u64 v, int lane) {
  const u32 hb = (u32)(__shfl_sync(FULL, v, 31) >> (DB - 1)) & 1u;
  u32 out;
  if (lane == 31) v &= MASK >> 1;
  if (hb) v = lookahead(add_low(v, lane, CF), lane, out);
  const u64 w = lookahead(add_low(v, lane, CF), lane, out);
  const u32 ge = (u32)(__shfl_sync(FULL, w, 31) >> (DB - 1)) & 1u;
  if (ge) v = lane == 31 ? (w & (MASK >> 1)) : w;
  return v;
}

__device__ __forceinline__ u64 lane_bits(const u32 *x, int lane) {   // lane's digit of an NL-limb value
  u64 v = 0;
#pragma unroll
  for (int b = U - 1; b >= 0; --b) {
    const int i = U * lane + b;
    v = (v << 8) | ((x[i >> 2] >> (8 * (i & 3))) & 0xFFu);
  }
  return v;
}

// a - b as digits, `neg` when a < b (the borrow out of lane 31).
__device__ __forceinline__ u64 sub_raw(u64 a, u64 b, int lane, bool &neg) {
  const u32 gen = __ballot_sync(FULL, a < b), prop = __ballot_sync(FULL, a == b);
  const u64 x = (u64)(gen | prop);
  const u64 sm = x + gen;
  const u32 bin = (u32)((sm ^ x ^ gen) >> lane) & 1u;
  neg = (sm >> 32) != 0;
  return (a - b - bin) & MASK;
}

__device__ __forceinline__ u64 tc_sub(u64 a, u64 b, u64 pv, int lane) {   // a - b mod p, canonical
  bool neg;
  u64 d = sub_raw(a, b, lane, neg);
  if (neg) {
    u32 out;
    d = lookahead(d + pv, lane, out);
  }
  return d;
}

__device__ __forceinline__ u64 tc_addmod(u64 a, u64 b, u64 pv, int lane) {   // a + b mod p, canonical
  u32 out;
  u64 sm = lookahead(a + b, lane, out);
  bool neg;
  const u64 d = sub_raw(sm, pv, lane, neg);
  return neg ? sm : d;
}

__device__ __forceinline__ u64 tc_negate(u64 a, u64 pv, int lane) {   // p - a, a canonical (0 stays 0)
  bool neg;
  const u64 d = sub_raw(pv, a, lane, neg);
  return __ballot_sync(FULL, a != 0) ? d : 0;
}

template <int K>
__device__ __forceinline__ u64 tc_decode(TcScratch s, u64 y, const u64 (&cj)[MAXK], u64 t, u64 pv, int lane) {
#pragma unroll
  for (int j = K - 1; j >= 0; --j) {
    const bool odd = __shfl_sync(FULL, y, 0) & 1;
    u64 x = tc_canon(tc_square(s, y, lane), lane);
    if (odd) x = tc_negate(x, pv, lane);
    y = tc_sub(tc_sub(x, cj[j], pv, lane), t, pv, lane);
  }
  return y;
}

constexpr int TC_WARPS = 4;

// A warp per word: words gw, gw + warps, ...; mode 0: XOR checksum; mode 2: also the result's limbs.
template <int K, int MODE>
__global__ void __launch_bounds__(TC_WARPS * 32) tc_decode_kernel(const uint8_t *__restrict__ store, u32 *__restrict__ out,
                                                                 u64 n, u64 base, unsigned long long *xr) {
  __shared__ __align__(16) uint8_t scratch[TC_WARPS][TC_WORD_BYTES];
  __shared__ u32 limbs[TC_WARPS][2][NL];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u64 gw = (u64)blockIdx.x * TC_WARPS + warp, nw = (u64)gridDim.x * TC_WARPS;
  TcScratch s = tc_scratch(scratch[warp]);
  tc_init(s, lane);
  if (lane == 0) {
    u32 a[NL], t0[NL], st[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) a[k] = kTau[k];
    mul_small(a, base + gw, t0);
    mul_small(a, nw, st);
#pragma unroll
    for (int k = 0; k < NL; ++k) { limbs[warp][0][k] = t0[k]; limbs[warp][1][k] = st[k]; }
  }
  __syncwarp();
  u64 t = lane_bits(limbs[warp][0], lane);
  const u64 step = lane_bits(limbs[warp][1], lane);
  u32 pl[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) pl[k] = kP[k];
  const u64 pv = lane_bits(pl, lane);
  u64 cj[MAXK];
#pragma unroll
  for (int j = 0; j < MAXK; ++j) {
    u32 c[NL];
    load_key(j, c);
    cj[j] = lane_bits(c, lane);
  }
  u64 acc = 0;
  for (u64 i = gw; i < n; i += nw) {
    const uint8_t *w = store + i * S + U * lane;
    u64 y = 0;
#pragma unroll
    for (int b = U - 1; b >= 0; --b) y = (y << 8) | __ldcs(w + b);
    y = tc_decode<K>(s, y, cj, t, pv, lane);
    acc ^= y * 0x9E3779B97F4A7C15ull + lane;
    if (MODE == 2) {
      uint8_t *o = reinterpret_cast<uint8_t *>(out + i * NL) + U * lane;
#pragma unroll
      for (int b = 0; b < U; ++b) o[b] = (uint8_t)(y >> (8 * b));
    }
    t = tc_addmod(t, step, pv, lane);
  }
  atomicXor(xr, (unsigned long long)acc);
}

// The tensor cores' squaring rate: W independent words a warp, each squared `steps` times (lazy, carried).
template <int W>
__global__ void __launch_bounds__(TC_WARPS * 32) tc_chain_kernel(u64 steps, unsigned long long *xr) {
  __shared__ __align__(16) uint8_t scratch[TC_WARPS][W][TC_WORD_BYTES];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  TcScratch s[W];
  u64 v[W];
  const u64 id = (u64)blockIdx.x * TC_WARPS + warp;
#pragma unroll
  for (int w = 0; w < W; ++w) {
    s[w] = tc_scratch(scratch[warp][w]);
    tc_init(s[w], lane);
    u64 z = (id * W + w) * 0x9E3779B97F4A7C15ull + lane * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 31;
    v[w] = z & MASK & (lane == 31 ? MASK >> 1 : MASK);
  }
#pragma unroll 1
  for (u64 st = 0; st < steps; ++st) {
#pragma unroll
    for (int w = 0; w < W; ++w) v[w] = tc_square(s[w], v[w], lane);
  }
  u64 acc = 0;
#pragma unroll
  for (int w = 0; w < W; ++w) acc ^= v[w] * (w + 1);
  atomicXor(xr, (unsigned long long)acc);
}

// Both units at once: warps [0, cw) square on CUDA cores (a word a thread, sc steps), warps [cw, 8) on tensor cores (a
// word a warp, stc steps).
constexpr int MIX_WARPS = 8;
__global__ void __launch_bounds__(MIX_WARPS * 32, 1) mix_kernel(u64 sc, u64 stc, int cw, unsigned long long *xr) {
  extern __shared__ __align__(16) uint8_t mix_smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  u64 h = 0;
  if (warp < cw) {
    u32 x[NL];
    seed_word((u64)blockIdx.x * blockDim.x + threadIdx.x, x);
#pragma unroll 1
    for (u64 st = 0; st < sc; ++st) {
      u32 y[NL];
      sqr(x, y);
      copy(x, y);
    }
    h = fold64(x);
  } else {
    TcScratch s = tc_scratch(mix_smem + (warp - cw) * TC_WORD_BYTES);
    tc_init(s, lane);
    u64 z = ((u64)blockIdx.x * MIX_WARPS + warp) * 0x9E3779B97F4A7C15ull + lane * 0xBF58476D1CE4E5B9ull;
    z ^= z >> 31;
    u64 v = z & MASK & (lane == 31 ? MASK >> 1 : MASK);
#pragma unroll 1
    for (u64 st = 0; st < stc; ++st) v = tc_square(s, v, lane);
    h = v;
  }
  atomicXor(xr, (unsigned long long)h);
}

// Gate for the tensor-core squaring: in/out NL limbs a word, canonical in and out.
__global__ void __launch_bounds__(32) tc_square_gate(const u32 *__restrict__ in, u32 *__restrict__ out, int n) {
  __shared__ __align__(16) uint8_t scratch[TC_WORD_BYTES];
  __shared__ __align__(16) u32 word[NL];
  const int lane = threadIdx.x;
  TcScratch s = tc_scratch(scratch);
  tc_init(s, lane);
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    for (int k = lane; k < NL; k += 32) word[k] = in[(u64)i * NL + k];
    __syncwarp();
    u64 v = lane_bits(word, lane);
    v = tc_canon(tc_square(s, v, lane), lane);
    __syncwarp();
    uint8_t *wb = reinterpret_cast<uint8_t *>(word);
#pragma unroll
    for (int b = 0; b < U; ++b) wb[U * lane + b] = (uint8_t)(v >> (8 * b));
    __syncwarp();
    for (int k = lane; k < NL; k += 32) out[(u64)i * NL + k] = word[k];
    __syncwarp();
  }
}

// --- the FP8 GEMM: warp-specialized, plain or fused -------------------------------------------------------------------
//
// As `benchmarks/word_codec/wc.cu`'s gemm_ws: a CTA keeps one 512-column chunk of x in shared memory and takes the 16-row
// blocks nb = g0 + i groups; the last warp's lane 0 bulk-copies task i into stage i mod stages, consumer warp c takes
// tasks c, c + C, ... MODE 0: plain fragments (8,192 bytes a stage); MODE 1: plain, plus `knob` int8 MMAs a tile on
// the tile's own registers; MODE 2: plain, plus `knob` IMADs a tile a thread. The fused decode is fused_kernel.

constexpr int LANES = 32, TASK_TILES = 16, TASK_COLS = 512, XS_STRIDE = TASK_COLS + 16;
constexpr int WS_MAX_CONSUMERS = 8;

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint4 &a, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u32 smem_u32(const void *p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u64 *bar, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u64 *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(u64 *bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(u64 *bar, u32 parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(
          smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ bool mbar_test(u64 *bar, u32 parity) {
  u32 r;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(r)
      : "r"(smem_u32(bar)), "r"(parity)
      : "memory");
  return r != 0;
}
__device__ __forceinline__ bool mbar_try(u64 *bar, u32 parity, u32 ns) {   // waits at most about ns
  u32 r;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(r)
      : "r"(smem_u32(bar)), "r"(parity), "r"(ns)
      : "memory");
  return r != 0;
}
__device__ __forceinline__ void bulk_load(void *dst, const void *src, u32 bytes, u64 *bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}

template <int NB8>
__device__ __forceinline__ void load_xs(uint8_t (*xs)[XS_STRIDE], const uint8_t *__restrict__ x, int M, int Kd, int kc) {
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += blockDim.x) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < M) v = __ldg(reinterpret_cast<const uint4 *>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4 *>(&xs[b][16 * c]) = v;
  }
}

// Tile tt of a task: the warp's A fragment a (rows nb 16 .., columns 32 tt ..) against x's batch groups.
template <int NB8>
__device__ __forceinline__ void mma_tile(float (&acc)[NB8][4], const u32 (&a)[4], const uint8_t (*xs)[XS_STRIDE],
                                         int tt, int g, int tq) {
#pragma unroll
  for (int j = 0; j < NB8; ++j) {
    const u32 b0 = *reinterpret_cast<const u32 *>(&xs[8 * j + g][32 * tt + 4 * tq]);
    const u32 b1 = *reinterpret_cast<const u32 *>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
    mma_e4m3(acc[j], a, b0, b1);
  }
}

template <int NB8, bool PARTIALS>
__device__ __forceinline__ void store_task(const float (&acc)[NB8][4], float *__restrict__ y, int M, int N, int kc,
                                           int nb, int g, int tq) {
#pragma unroll
  for (int j = 0; j < NB8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
      if (b < M) {
        if constexpr (PARTIALS) y[((size_t)kc * M + b) * N + n] = acc[j][e];
        else atomicAdd(y + (size_t)b * N + n, acc[j][e]);
      }
    }
  }
}

template <int NB8>
__host__ __device__ constexpr int ws_smem_bytes(int stages) {
  return stages * TASK_BYTES + NB8 * 8 * XS_STRIDE + 2 * stages * 8;
}

// An SM's registers are four sub-partitions of 16K, a warp on one: WS_MAX_CONSUMERS + 1 = 9 warps put three on one, so
// 168 registers a thread.
template <int MODE, int NB8, bool PARTIALS>
__global__ void __maxnreg__(168) gemm_ws_kernel(const void *__restrict__ wstore, const uint8_t *__restrict__ x,
                                                float *__restrict__ y, int M, int N, int Kd, int groups, int consumers,
                                                int stages, int knob, unsigned long long *xr) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * TASK_BYTES);
  u64 *full = reinterpret_cast<u64 *>(smem + stages * TASK_BYTES + NB8 * 8 * XS_STRIDE);
  u64 *empty = full + stages;
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  load_xs<NB8>(xs, x, M, Kd, kc);
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntasks = nbs > g0 ? (nbs - g0 + groups - 1) / groups : 0;

  if (warp == consumers) {
    if (lane == 0) {
      for (int i = 0; i < ntasks; ++i) {
        const int s = i % stages;
        if (i >= stages) mbar_wait(empty + s, (i / stages - 1) & 1);
        const u64 q = (u64)(g0 + i * groups) * nchunks + kc;
        mbar_expect_tx(full + s, TASK_BYTES);
        bulk_load(smem + s * TASK_BYTES, static_cast<const uint8_t *>(wstore) + q * TASK_BYTES, TASK_BYTES, full + s);
      }
    }
    return;
  }

  u32 sink = 0;
  int32_t dacc[4][4] = {};
  for (int i = warp; i < ntasks; i += consumers) {
    const int s = i % stages, nb = g0 + i * groups;
    const uint8_t *st = smem + s * TASK_BYTES;
    mbar_wait(full + s, (i / stages) & 1);
    float acc[NB8][4];
#pragma unroll
    for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const uint4 *sp = reinterpret_cast<const uint4 *>(st) + lane;
    uint4 wv[TASK_TILES];
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) wv[tt] = sp[tt * LANES];
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + s);
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) {
      const u32 a[4] = {wv[tt].x, wv[tt].y, wv[tt].z, wv[tt].w};
      mma_tile<NB8>(acc, a, xs, tt, g, tq);
      if constexpr (MODE == 1) {
#pragma unroll 1
        for (int e = 0; e < knob; e += 4) {
          mma_u8(dacc[0], a, a[0], a[1]);
          mma_u8(dacc[1], a, a[1], a[2]);
          mma_u8(dacc[2], a, a[2], a[3]);
          mma_u8(dacc[3], a, a[3], a[0]);
        }
      } else if constexpr (MODE == 2) {
        u32 z[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) z[c] = a[c & 3] + c;
#pragma unroll 1
        for (int e = 0; e < knob; e += 8) {
#pragma unroll
          for (int c = 0; c < 8; ++c) z[c] = z[c] * a[(c + 1) & 3] + a[(c + 2) & 3];
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) sink ^= z[c];
      }
    }
    store_task<NB8, PARTIALS>(acc, y, M, N, kc, nb, g, tq);
  }
  if (MODE == 1 || MODE == 2) {
    u32 h = sink;
#pragma unroll
    for (int c = 0; c < 4; ++c) h ^= (u32)(dacc[c][0] ^ dacc[c][1] ^ dacc[c][2] ^ dacc[c][3]);
    if (h == 0x12345679u) atomicXor(xr, (unsigned long long)h);
  }
}

// --- ro-blake3/v1 (verity/core/protocols/bandwidth/tags.py) ----------------------------------------------------------
//
// y_j = BLAKE3-keyed(K, u64be j ‖ E_j) over E's blocks of TAG_BLOCK bytes, so a full block's message is four whole
// chunks: a quad of lanes hashes one block, a chunk a lane, and joins the chunks' chaining values with two parent
// compressions over shuffles (lanes 0 and 2, then lane 0). Every add of the compression is an IMAD by kOne, which moves
// the adds to the FMA pipe (ra-tag-fast's V1, +32% on sm_120); kOne sits in constant memory so ptxas can't fold it.
// leaf_c = SHA512(prefix(leaf) ‖ u64be c ‖ 16 tags), five compressions from the prefix's midstate; the root is the
// RFC 6962 SHA-512 tree over the leaves, a level at a time with the odd last node promoted (the same tree).
constexpr int TAG_BLOCK = 4088;
constexpr u32 B3_START = 1, B3_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16;
constexpr u32 B3_IV0 = 0x6A09E667u, B3_IV1 = 0xBB67AE85u, B3_IV2 = 0x3C6EF372u, B3_IV3 = 0xA54FF53Au;
__constant__ u32 kTagKey[8];
__constant__ u32 kOne;

__device__ __forceinline__ u32 imad(u32 a, u32 b, u32 c) {
  u32 d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ u32 bswap32(u32 x) { return __byte_perm(x, 0, 0x0123); }

__device__ __forceinline__ void b3g(u32 &a, u32 &b, u32 &c, u32 &d, u32 x, u32 y, u32 one) {
  a = imad(b, one, imad(x, one, a));
  d = __byte_perm(d ^ a, 0, 0x1032);
  c = imad(d, one, c);
  b = __funnelshift_r(b ^ c, b ^ c, 12);
  a = imad(b, one, imad(y, one, a));
  d = __byte_perm(d ^ a, 0, 0x0321);
  c = imad(d, one, c);
  b = __funnelshift_r(b ^ c, b ^ c, 7);
}

// One compression of m from cv; out (may alias cv) gets the next chaining value, or under ROOT the hash.
__device__ __forceinline__ void b3(const u32 (&cv)[8], const u32 (&min)[16], u32 counter, u32 len, u32 flags,
                                   u32 (&out)[8], u32 one) {
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = min[i];
  u32 v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  u32 v8 = B3_IV0, v9 = B3_IV1, v10 = B3_IV2, v11 = B3_IV3, v12 = counter, v13 = 0, v14 = len, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    b3g(v0, v4, v8, v12, m[0], m[1], one);
    b3g(v1, v5, v9, v13, m[2], m[3], one);
    b3g(v2, v6, v10, v14, m[4], m[5], one);
    b3g(v3, v7, v11, v15, m[6], m[7], one);
    b3g(v0, v5, v10, v15, m[8], m[9], one);
    b3g(v1, v6, v11, v12, m[10], m[11], one);
    b3g(v2, v7, v8, v13, m[12], m[13], one);
    b3g(v3, v4, v9, v14, m[14], m[15], one);
    if (r < 6) {
      const u32 t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
  out[0] = v0 ^ v8;
  out[1] = v1 ^ v9;
  out[2] = v2 ^ v10;
  out[3] = v3 ^ v11;
  out[4] = v4 ^ v12;
  out[5] = v5 ^ v13;
  out[6] = v6 ^ v14;
  out[7] = v7 ^ v15;
}

// Message block k of chunk c of a block's message (u64be j ‖ the block's bytes at e, len bytes in all, a multiple of 8).
__device__ __forceinline__ void tag_words(const uint2 *__restrict__ e, u64 j, int len, int c, int k, u32 (&m)[16]) {
#pragma unroll
  for (int h = 0; h < 8; ++h) {
    const int o = 1024 * c + 64 * k + 8 * h;
    uint2 v = make_uint2(0u, 0u);
    if (o == 0)
      v = make_uint2(bswap32((u32)(j >> 32)), bswap32((u32)j));
    else if (o < len)
      v = __ldg(e + (o >> 3) - 1);
    m[2 * h] = v.x;
    m[2 * h + 1] = v.y;
  }
}

// Block jl's tag (j = its index in E) in lane 0 of the quad, whose lane c hashes chunk c; E's `bytes` end the last block
// short. A quad past the end loads nothing and its tag is not stored.
__device__ __forceinline__ void tag_quad(const uint8_t *__restrict__ E, u64 bytes, u64 jl, u64 j, int c, u32 one,
                                         u32 (&y)[8]) {
  const u64 e0 = jl * TAG_BLOCK;
  const int len = e0 < bytes ? (int)(bytes - e0 < (u64)TAG_BLOCK ? bytes - e0 : (u64)TAG_BLOCK) + 8 : 0;
  const int nch = (len + 1023) >> 10, clen = min(1024, len - 1024 * c), nmb = clen > 0 ? (clen + 63) >> 6 : 0;
  const uint2 *e = reinterpret_cast<const uint2 *>(E + (e0 < bytes ? e0 : 0));
  u32 key[8], cv[8], nx[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key[i] = kTagKey[i];
  tag_words(e, j, len, c, 0, nx);
#pragma unroll 1
  for (int k = 0; k < nmb; ++k) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = nx[i];
    if (k + 1 < nmb) tag_words(e, j, len, c, k + 1, nx);
    const u32 fl = B3_KEYED | (k == 0 ? B3_START : 0) | (k == nmb - 1 ? B3_END | (nch == 1 ? B3_ROOT : 0) : 0);
    b3(cv, m, (u32)c, (u32)min(64, clen - 64 * k), fl, cv, one);
  }
  u32 m[16], p[8], r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m[i] = cv[i];
    m[8 + i] = __shfl_down_sync(FULL, cv[i], 1);
  }
  b3(key, m, 0, 64, B3_KEYED | B3_PARENT | (nch == 2 ? B3_ROOT : 0), p, one);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    m[i] = p[i];
    m[8 + i] = __shfl_down_sync(FULL, nch == 3 ? cv[i] : p[i], 2);
  }
  b3(key, m, 0, 64, B3_KEYED | B3_PARENT | B3_ROOT, r, one);
#pragma unroll
  for (int i = 0; i < 8; ++i) y[i] = nch == 1 ? cv[i] : nch == 2 ? p[i] : r[i];
}

// A CTA's tagging: its warps take the CTA's units from the shared counter *next, the tag warps from the start and the
// others once their own work is done; unit u is the 8 blocks from 8 (u grid + cta), a quad each, so the device's tag
// front moves through E as its decode front does. tags[8 jl ..] gets block jl's tag as little-endian words.
__device__ __noinline__ void tag_warp(const uint8_t *__restrict__ E, u64 bytes, u64 jbase, u32 *next,
                                      u32 *__restrict__ tags) {
  const int lane = threadIdx.x & 31, quad = lane >> 2, c = lane & 3;
  const u64 nblk = (bytes + TAG_BLOCK - 1) / TAG_BLOCK;
  const u32 one = kOne;
  for (;;) {
    u32 v = 0;
    if (lane == 0) v = atomicAdd(next, 1u);
    const u64 u = __shfl_sync(FULL, v, 0);
    const u64 jl0 = 8 * (u * gridDim.x + blockIdx.x);
    if (jl0 >= nblk) break;
    const u64 jl = jl0 + quad;
    u32 y[8];
    tag_quad(E, bytes, jl, jbase + jl, c, one, y);
    if (c == 0 && jl < nblk) {
      uint4 *o = reinterpret_cast<uint4 *>(tags + 8 * jl);
      o[0] = make_uint4(y[0], y[1], y[2], y[3]);
      o[1] = make_uint4(y[4], y[5], y[6], y[7]);
    }
  }
}

// Compute-only pipe rates: `steps` times eight independent ops a thread, op 0 mad.wide.u32 (IMAD.WIDE.U32), 1 mad.lo.u32
// (IMAD), 2 add.u32 (IADD3), 3 the squaring's mac (mad.lo.cc, madc.hi.cc, addc): the floor a squaring's products set.
template <int OP>
__global__ void __launch_bounds__(128) pipe_kernel(u64 steps, unsigned long long *xr) {
  const u32 id = blockIdx.x * blockDim.x + threadIdx.x, b = id | 1;
  u32 a[8], r[8][3];
  u64 acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    a[i] = id * 0x9E3779B9u + i;
    acc[i] = a[i];
    r[i][0] = r[i][1] = r[i][2] = a[i];
  }
#pragma unroll 1
  for (u64 s = 0; s < steps; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (OP == 0)
        asm volatile("{\n .reg .u32 lo, hi;\n mov.b64 {lo, hi}, %0;\n mad.wide.u32 %0, lo, %1, %0;\n}" : "+l"(acc[i])
                     : "r"(b));
      if constexpr (OP == 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(b), "r"(id));
      if constexpr (OP == 2) asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(b));
      if constexpr (OP == 3) mac(r[i], r[i][1], b);
    }
  }
  u64 h = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) h ^= acc[i] ^ a[i] ^ r[i][0] ^ ((u64)r[i][1] << 17) ^ ((u64)r[i][2] << 40);
  if (h == 0x123456789ull) atomicXor(xr, h);
}

// Compute-only: `steps` chained compressions a thread (the tag's ALU rate, which sets tau).
__global__ void __launch_bounds__(128) b3_chain_kernel(u64 steps, unsigned long long *xr) {
  const u32 id = blockIdx.x * blockDim.x + threadIdx.x, one = kOne;
  u32 cv[8], m[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = kTagKey[i] ^ (id * 0x9E3779B9u + i);
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = id + 0x01000193u * i;
#pragma unroll 1
  for (u64 s = 0; s < steps; ++s) b3(cv, m, (u32)s, 64, B3_KEYED, cv, one);
  if ((cv[0] ^ cv[5]) == 0x12345679u) atomicXor(xr, (unsigned long long)cv[1]);
}

#define WT_SHA512_K                                                                                              \
  {0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, \
   0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, \
   0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, \
   0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, \
   0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL, \
   0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, \
   0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, \
   0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL, \
   0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL, \
   0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, \
   0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, \
   0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL, \
   0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL, \
   0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, \
   0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, \
   0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL}
#define WT_SHA512_IV                                                                                            \
  {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, \
   0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL}
__constant__ u64 kSha512K[80] = WT_SHA512_K;
__constant__ u64 kSha512Iv[8] = WT_SHA512_IV;
[[maybe_unused]] static const u64 SHA512_K_HOST[80] = WT_SHA512_K;
static const u64 SHA512_IV_HOST[8] = WT_SHA512_IV;

struct Mid {
  u64 h[8];
};

__host__ __device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

// One SHA-512 compression of the big-endian words w (clobbered) into s (`benchmarks/bandwidth/tagger.cu`'s).
__host__ __device__ __forceinline__ void sha512_compress(u64 s[8], u64 w[16]) {
#ifdef __CUDA_ARCH__
  const u64 *K = kSha512K;
#else
  const u64 *K = SHA512_K_HOST;
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
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

// The leaves over the tags of blocks [0, nblk), a thread a leaf; each digest as SHA-512's 8 state words.
__global__ void tag_leaves_kernel(const u32 *__restrict__ tags, u64 nblk, Mid leafmid, u64 *__restrict__ leaves) {
  const u64 leaf = blockIdx.x * (u64)blockDim.x + threadIdx.x, first = leaf * 16;
  if (first >= nblk) return;
  const int n = (int)(nblk - first < 16 ? nblk - first : 16);
  const int data = 1 + 4 * n, nb = (8 * data + 17 + 127) / 128;
  const u32 *T = tags + first * 8;
  u64 s[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = leafmid.h[i];
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i;
      u64 v = 0;
      if (q == 0)
        v = leaf;
      else if (q < data)
        v = ((u64)bswap32(T[2 * (q - 1)]) << 32) | bswap32(T[2 * (q - 1) + 1]);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nb - 1) v = (u64)(128 + 8 + 32 * n) * 8;
      w[i] = v;
    }
    sha512_compress(s, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) leaves[leaf * 8 + i] = s[i];
}

// SHA512(0x01 ‖ L ‖ R): 129 bytes, two compressions from the IV.
__device__ __forceinline__ void node_hash(const u64 *L, const u64 *R, u64 *out) {
  u64 w[16], s[8], prev = 1;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const u64 cur = k < 8 ? L[k] : R[k - 8];
    w[k] = (prev << 56) | (cur >> 8);
    prev = cur;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kSha512Iv[i];
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = 0;
  w[0] = (prev << 56) | (0x80ULL << 48);
  w[15] = 129 * 8;
  sha512_compress(s, w);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i];
}

__global__ void __launch_bounds__(128) tag_level_kernel(const u64 *__restrict__ in, u64 n, u64 *__restrict__ out) {
  const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i >= (n + 1) / 2) return;
  if (2 * i + 1 < n)
    node_hash(in + 16 * i, in + 16 * i + 8, out + 8 * i);
  else
#pragma unroll
    for (int k = 0; k < 8; ++k) out[8 * i + k] = in[16 * i + k];
}

// --- the fused decode GEMM --------------------------------------------------------------------------------------------
//
// Warp D + 1's lane 0 bulk-copies task i's TW stored words (SB bytes) into stage i mod stages; the D decoder warps
// decode them in place; warp D, the MMA warp, takes the tasks in order and runs the plain kernel's 16 tiles on each.
// The warps after D + 1, if any, tag the layer's store (`tag_warp`), read through L2 behind the bulk copies.
// The CTA's tasks are one sequence of words (task i's word w is word TW i + w), cut into units of 32: decoder warp d
// takes units d, d + D, ..., lane l a unit's word l. A lane loads its word, decodes it, and writes its payload (stage
// bytes PB w ..) only once every word of the task is loaded, since a payload overwrites the stored bytes of the words
// before it. Barriers a stage: full (the copy), ready (full relayed in task order), loaded (TW lanes), decoded (TW
// lanes), empty (the MMA warp's lanes).
//
// A parity wait is right only while its barrier is at most one phase behind. Bulk copies may complete out of order, so
// a decoder that skipped task i - stages cannot read that copy's completion off `full`; the copy lane, which observes
// every copy in order, relays each onto `ready`, and the decoders wait on that. The host's bounds do the rest: D at
// least the units a task spans (no warp holds two units of one task, which would deadlock on `loaded`), and at least
// ceil(32 D / TW) stages (a decoder's tasks advance by at most that from one unit to its next, so the stage's previous
// task is one it held, or an earlier one, whose relay came first). One MMA warp keeps `decoded` and `empty` in order.
// The barrier orders generic accesses only, so the MMA warp arrives on `empty` once its MMAs have consumed the stage's
// loads: an earlier arrive lets the next bulk copy (async proxy) overwrite the stage under loads still in flight.
//
// An SM's registers fit FUSED_WARPS warps at the decode's register count (a sub-partition holds 16K: four warps at
// 128 registers, three at 168, two at 255).
constexpr int FUSED_WARPS = NL <= 17 ? 16 : (NL <= 24 ? 12 : 8);
constexpr int FUSED_SPAN = (TW + 30) / 32 + 1;            // the most units a task's words touch

template <int NB8, int SKK>
__host__ __device__ constexpr int fused_smem_bytes(int stages) {
  return stages * SB + NB8 * 8 * XS_STRIDE + 6 * stages * 8 + 8 + 64 * SKK;
}

// Bytes [o, o + PB) of the stage, o = PB w, get y's low PB bytes as whole aligned words: the lane writes each word
// that starts inside its payload, taking the bytes past it from the next lane's first word (nxt). Past a task's last
// word they are padding (TW PB + 3 <= SB, and every word of the task is loaded by now). Words shared with another warp's
// lane go byte by byte: lane 0's head (its word's first bytes, in the word lane 31 of the warp before owns) and lane
// 31's tail (unless its word is the task's last).
constexpr int PAY_E = 4 * NL - PB;                      // y's bytes past the payload; 0 at full slot
static_assert(PAY_E == 0 ? PB == S : (PAY_E <= 3 && TW * PB + 3 <= SB), "the payload's last limb and the stage's padding");
__device__ __forceinline__ u32 top_byte(const u32 (&y)[NL], int b) {   // y's byte b, 4 (NL - 2) <= b < 4 NL
  return ((b >= 4 * (NL - 1) ? y[NL - 1] : y[NL - 2]) >> (8 * (b & 3))) & 0xFFu;
}
__device__ __forceinline__ void put_payload_w(uint8_t *st, int w, int lane, const u32 (&y)[NL], u32 nxt) {
  const int o = PB * w, d = (4 - (o & 3)) & 3, nw = (PB - d + 3) >> 2;
  const bool split = lane == LANES - 1 && w != TW - 1 && ((o + PB) & 3) != 0;
  const int whole = nw - (split ? 1 : 0);
  u32 yx[NL + 1];
#pragma unroll
  for (int k = 0; k < NL - 1; ++k) yx[k] = y[k];
  constexpr int E = PAY_E > 0 ? PAY_E : 1;               // unused at full slot
  yx[NL - 1] = (y[NL - 1] & ((1u << (8 * (4 - E))) - 1u)) | (nxt << (8 * (4 - E)));
  yx[NL] = nxt >> (8 * E);
  u32 *sw = reinterpret_cast<u32 *>(st) + ((o + d) >> 2);
#pragma unroll
  for (int k = 0; k < NL; ++k)
    if (k < whole) sw[k] = __funnelshift_r(yx[k], yx[k + 1], 8 * d);
  if (split) {
    const int a = o + d + 4 * (nw - 1);
#pragma unroll
    for (int b = 0; b < 3; ++b)
      if (a + b < o + PB) st[a + b] = (uint8_t)top_byte(y, a + b - o);
  }
  if (lane == 0 && w != 0) {
#pragma unroll
    for (int b = 0; b < 3; ++b)
      if (b < d) st[o + b] = (uint8_t)(y[0] >> (8 * b));
  }
}

// The decoder warps' loop (fused_kernel's comment). At k = 1 a lane keeps w = -(c_0 + t) in place of its tweak t and
// steps it down. A payload's generic stores reach the stage before the next bulk copy into it (async proxy) only
// behind a proxy fence. At full slot a payload is its own word's bytes, so a lane writes it in place without `loaded`.
template <int K>
__device__ __forceinline__ void decode_tasks(uint8_t *smem, u64 *ready, u64 *loaded, u64 *decoded, u64 *sketched,
                                             int stages, int ntasks, u64 q0, u64 gq, u64 base, int decoders, int warp,
                                             int lane, const u32 *__restrict__ tsteps) {
  const int adv = 32 * decoders / TW, rem = 32 * decoders % TW;
  int i = (32 * warp + lane) / TW, w = (32 * warp + lane) % TW;
  u32 t[NL];
  tweak(base + (q0 + (u64)i * gq) * TW + w, t);
  if constexpr (K == 1) {
    u32 wn[NL];
    w_of(t, wn);
    copy(t, wn);
  }
  for (;;) {
    const int lo = __shfl_sync(FULL, i, 0);
    if (lo >= ntasks) break;
    const int hi = min(__shfl_sync(FULL, i, LANES - 1), ntasks - 1);
    const bool on = i < ntasks;
    const int s = i % stages;
    uint8_t *st = smem + s * SB;
    mbar_wait(ready + lo % stages, (lo / stages) & 1);
    if (hi != lo) mbar_wait(ready + hi % stages, (hi / stages) & 1);
    u32 yw[NL];
    yw[0] = 0;
    if (on) {
      load_word_smem(st + w * S, yw);
      if constexpr (PAY_E > 0) mbar_arrive(loaded + s);
      if constexpr (K == 1) decode_word_w(yw, t);
      else decode_word<K>(yw, t);
    }
    if constexpr (PAY_E == 0) {
      if (sketched) {
        mbar_wait(sketched + lo % stages, (lo / stages) & 1);
        if (hi != lo) mbar_wait(sketched + hi % stages, (hi / stages) & 1);
      }
      if (on) {
        uint4 *o = reinterpret_cast<uint4 *>(st + w * S);
#pragma unroll
        for (int k = 0; k < NL / 4; ++k) o[k] = make_uint4(yw[4 * k], yw[4 * k + 1], yw[4 * k + 2], yw[4 * k + 3]);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        mbar_arrive(decoded + s);
      }
    } else {
      const u32 nxt = __shfl_down_sync(FULL, yw[0], 1);
      mbar_wait(loaded + lo % stages, (lo / stages) & 1);
      if (hi != lo) mbar_wait(loaded + hi % stages, (hi / stages) & 1);
      if (sketched) {
        mbar_wait(sketched + lo % stages, (lo / stages) & 1);
        if (hi != lo) mbar_wait(sketched + hi % stages, (hi / stages) & 1);
      }
      if (on) {
        put_payload_w(st, w, lane, yw, nxt);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        mbar_arrive(decoded + s);
      }
    }
    const bool wrap = w + rem >= TW;
    u32 step[NL], t2[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) step[k] = __ldg(tsteps + (wrap ? NL : 0) + k);
    if constexpr (K == 1) submod_cc(t, step, t2);
    else addmod(t, step, t2);
    copy(t, t2);
    w += wrap ? rem - TW : rem;
    i += wrap ? adv + 1 : adv;
  }
}

// --- ro-sketch-blake3/v0 (sketching.py) -------------------------------------------------------------------------------
//
// Sketch warp sw of nsw takes steps sw, sw + nsw, ... of each task (a step is 512 stored bytes, lane l's 16 at 16 l the
// int8 A fragment, zero past SB) and runs k_s / 8 mma.m16n8k32.s8 on each against the signs (`sign_fragments`: a
// lane's sk_words u32 a step, one prmt a register), keeping the tile's 16 × k_s sums in its accumulators. A lane arrives
// on `sketched` once its loads of the task have landed (a use of every loaded register before the arrive), and the
// decoders write payloads only behind it. At a tile's end (skt tasks, or the CTA's last) the warps add their sums in
// skbuf; sk_out, if given, gets the tile's 16 × k_s int32 at 16 k_s tile, and a lane a vector hashes y_v into
// tags[8 v ..] (tag index jbase + v). The CTA's tiles past its last task are zero vectors, and are tagged as such.
constexpr int SK_STEP = 512, SK_M = (SB + SK_STEP - 1) / SK_STEP, SK_HOLD = 5;
template <int KS>
__host__ __device__ constexpr int sk_words() {
  return (KS / 4 + 5) / 6 == 3 ? 4 : (KS / 4 + 5) / 6;
}

__device__ __forceinline__ u32 sk_sign(u32 sel, int e) {
  return e == 0 ? __byte_perm(0xFF01FF01u, 0xFF01FF01u, sel)
                : e == 1 ? __byte_perm(0xFFFF0101u, 0xFFFF0101u, sel) : __byte_perm(0x01010101u, 0xFFFFFFFFu, sel);
}
__device__ __forceinline__ void sk_bar(int nsw) { asm volatile("bar.sync 1, %0;\n" ::"r"(32 * nsw) : "memory"); }

template <int KS>
__device__ __forceinline__ void sk_tag(const int *w, u64 j, u32 *__restrict__ out) {
  constexpr int LEN = 8 + 4 * KS, NMB = (LEN + 63) / 64;
  const u32 one = kOne;
  u32 cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = kTagKey[i];
#pragma unroll 1
  for (int b = 0; b < NMB; ++b) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * b + i;
      m[i] = q == 0 ? bswap32((u32)(j >> 32)) : q == 1 ? bswap32((u32)j) : q - 2 < KS ? (u32)w[q - 2] : 0u;
    }
    b3(cv, m, 0, b == NMB - 1 ? LEN - 64 * b : 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == NMB - 1 ? B3_END | B3_ROOT : 0),
       cv, one);
  }
  uint4 *o = reinterpret_cast<uint4 *>(out);
  o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
  o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
}

template <int KS>
__device__ __forceinline__ void sk_flush(int (&acc)[KS / 8][4], int *skbuf, u64 tile, int sw, int nsw, int lane,
                                         int *__restrict__ sk_out, u32 *__restrict__ tags, u64 jbase) {
  const int g = lane >> 2, tq = lane & 3, t = 32 * sw + lane;
#pragma unroll
  for (int b = 0; b < KS / 8; ++b) {
    int *r0 = skbuf + g * KS + 8 * b + 2 * tq, *r1 = r0 + 8 * KS;
    atomicAdd(r0, acc[b][0]);
    atomicAdd(r0 + 1, acc[b][1]);
    atomicAdd(r1, acc[b][2]);
    atomicAdd(r1 + 1, acc[b][3]);
    acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0;
  }
  sk_bar(nsw);
  if (sk_out)
    for (int e = t; e < 4 * KS; e += 32 * nsw)
      reinterpret_cast<int4 *>(sk_out + tile * 16 * KS)[e] = reinterpret_cast<const int4 *>(skbuf)[e];
  const int r = lane * nsw + sw;
  if (tags && r < 16) sk_tag<KS>(skbuf + r * KS, jbase + 16 * tile + r, tags + 8 * (16 * tile + r));
  sk_bar(nsw);
  for (int e = t; e < 16 * KS; e += 32 * nsw) skbuf[e] = 0;
  sk_bar(nsw);
}

template <int KS>
__device__ __noinline__ void sketch_warp(const uint8_t *smem, u64 *ready, u64 *sketched, int *skbuf, int stages,
                                         int ntasks, int sw, int nsw, const u32 *__restrict__ signs,
                                         int *__restrict__ sk_out, u32 *__restrict__ tags, u64 jbase, int skt,
                                         int tpc) {
  constexpr int KB = KS / 8, WP = sk_words<KS>();
  const int lane = threadIdx.x & 31;
  int acc[KB][4];
#pragma unroll
  for (int b = 0; b < KB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0;
  for (int i = 0; i < ntasks; ++i) {
    const int s = i % stages, ii = i % skt;
    const uint8_t *st = smem + s * SB;
    const u32 *sg = signs + ((size_t)ii * SK_M * LANES + lane) * WP;
    mbar_wait(ready + s, (i / stages) & 1);
#pragma unroll 1
    for (int s0 = sw; s0 < SK_M; s0 += SK_HOLD * nsw) {
      uint4 a[SK_HOLD];
      u32 w[SK_HOLD][WP];
#pragma unroll
      for (int h = 0; h < SK_HOLD; ++h) {
        const int p = s0 + h * nsw, off = SK_STEP * p + 16 * lane;
        a[h] = make_uint4(0u, 0u, 0u, 0u);
        if (p < SK_M && off < SB) a[h] = *reinterpret_cast<const uint4 *>(st + off);
#pragma unroll
        for (int q = 0; q < WP; ++q) w[h][q] = 0u;
        if (p < SK_M) {
          const u32 *sp = sg + (size_t)p * LANES * WP;
          if constexpr (WP == 4) {
            const uint4 v = __ldg(reinterpret_cast<const uint4 *>(sp));
            w[h][0] = v.x, w[h][1] = v.y, w[h][2] = v.z, w[h][3] = v.w;
          } else if constexpr (WP == 2) {
            const uint2 v = __ldg(reinterpret_cast<const uint2 *>(sp));
            w[h][0] = v.x, w[h][1] = v.y;
          } else {
            w[h][0] = __ldg(sp);
          }
        }
      }
      if (s0 + SK_HOLD * nsw >= SK_M) {
        u32 dep = 0;
#pragma unroll
        for (int h = 0; h < SK_HOLD; ++h) dep ^= a[h].x ^ a[h].y ^ a[h].z ^ a[h].w;
        asm volatile("" ::"r"(dep));
        mbar_arrive(sketched + s);
      }
#pragma unroll
      for (int h = 0; h < SK_HOLD; ++h) {
        if (s0 + h * nsw < SK_M) {
#pragma unroll
          for (int b = 0; b < KB; ++b) {
            const int r0 = 2 * b, r1 = 2 * b + 1;
            const u32 b0 = sk_sign((r0 % 6) / 3 ? w[h][r0 / 6] >> 16 : w[h][r0 / 6], r0 % 3);
            const u32 b1 = sk_sign((r1 % 6) / 3 ? w[h][r1 / 6] >> 16 : w[h][r1 / 6], r1 % 3);
            mma_s8(acc[b], a[h], b0, b1);
          }
        }
      }
    }
    if (ii == skt - 1 || i == ntasks - 1)
      sk_flush<KS>(acc, skbuf, (u64)blockIdx.x * tpc + i / skt, sw, nsw, lane, sk_out, tags, jbase);
  }
  for (int j = (ntasks + skt - 1) / skt; j < tpc; ++j)
    sk_flush<KS>(acc, skbuf, (u64)blockIdx.x * tpc + j, sw, nsw, lane, sk_out, tags, jbase);
}

// tsteps: tau (32 D mod TW + A G TW) and tau (32 D mod TW - TW + (A + 1) G TW) mod p, A = floor(32 D / TW), G the
// store's tasks between the CTA's consecutive ones: the tweak's step to a lane's next unit, its task advancing A or
// A + 1.
// With SKK > 0, warps D + 2 .. D + 1 + sketchers sketch (sketch_warp) into sk_out and tags (ro-sketch-blake3/v0,
// index tag_base + v), and there are no tag warps.
template <int K, int NB8, bool PARTIALS, int SKK>
__global__ void __launch_bounds__(FUSED_WARPS * 32, 1) fused_kernel(
    const uint8_t *__restrict__ wstore, const uint8_t *__restrict__ x, float *__restrict__ y, int M, int N, int Kd,
    u64 base, int groups, int decoders, int stages, const u32 *__restrict__ tsteps, u32 *__restrict__ tags,
    u64 tag_base, const u32 *__restrict__ signs, int *__restrict__ sk_out, int skt, int sketchers) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * SB);
  u64 *full = reinterpret_cast<u64 *>(smem + stages * SB + NB8 * 8 * XS_STRIDE);
  u64 *loaded = full + stages, *decoded = loaded + stages, *empty = decoded + stages, *ready = empty + stages;
  u64 *sketched = ready + stages;
  u32 *tagq = reinterpret_cast<u32 *>(sketched + stages);
  int *skbuf = reinterpret_cast<int *>(tagq + 2);
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  load_xs<NB8>(xs, x, M, Kd, kc);
  for (int e = threadIdx.x; e < 16 * SKK; e += blockDim.x) skbuf[e] = 0;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      mbar_init(full + s, 1);
      mbar_init(loaded + s, TW);
      mbar_init(decoded + s, TW);
      mbar_init(empty + s, LANES);
      mbar_init(ready + s, 1);
      mbar_init(sketched + s, 32 * (SKK > 0 ? sketchers : 1));
    }
    *tagq = 0;
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntasks = nbs > g0 ? (nbs - g0 + groups - 1) / groups : 0;
  const u64 q0 = (u64)g0 * nchunks + kc, gq = (u64)groups * nchunks;   // task i is the store's task q0 + i gq

  if (warp == decoders + 1) {
    if (lane == 0) {
      int r = 0;   // the next copy to relay
      for (int i = 0; i < ntasks; ++i) {
        const int s = i % stages;
        if (i >= stages)
          while (!mbar_try(empty + s, (i / stages - 1) & 1, 128))
            while (r < i && mbar_test(full + r % stages, (r / stages) & 1)) mbar_arrive(ready + r++ % stages);
        mbar_expect_tx(full + s, SB);
        bulk_load(smem + s * SB, wstore + (q0 + i * gq) * SB, SB, full + s);
      }
      for (; r < ntasks; ++r) {
        mbar_wait(full + r % stages, (r / stages) & 1);
        mbar_arrive(ready + r % stages);
      }
    }
    __syncwarp();
  } else if (warp == decoders) {
    const int g = lane >> 2, tq = lane & 3;
    for (int i = 0; i < ntasks; ++i) {
      const int s = i % stages;
      mbar_wait(decoded + s, (i / stages) & 1);
      float acc[NB8][4];
#pragma unroll
      for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      const uint4 *sp = reinterpret_cast<const uint4 *>(smem + s * SB) + lane;
      uint4 wv[TASK_TILES];
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) wv[tt] = sp[tt * LANES];
#ifdef WT_DEBUG_DUMP
      if (g_dbg) {
        uint4 *o = reinterpret_cast<uint4 *>(g_dbg + (q0 + (u64)i * gq) * TASK_BYTES) + lane;
#pragma unroll
        for (int tt = 0; tt < TASK_TILES; ++tt) o[tt * LANES] = wv[tt];
      }
#endif
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) {
        const u32 a[4] = {wv[tt].x, wv[tt].y, wv[tt].z, wv[tt].w};
        mma_tile<NB8>(acc, a, xs, tt, g, tq);
      }
      mbar_arrive(empty + s);
      store_task<NB8, PARTIALS>(acc, y, M, N, kc, g0 + i * groups, g, tq);
    }
  } else if (warp < decoders) {
    decode_tasks<K>(smem, ready, loaded, decoded, SKK > 0 ? sketched : nullptr, stages, ntasks, q0, gq, base, decoders,
                    warp, lane, tsteps);
  } else if (SKK > 0 && warp < decoders + 2 + sketchers) {
    if constexpr (SKK > 0) {
      const int tpc = ((nbs + groups - 1) / groups + skt - 1) / skt;
      sketch_warp<SKK>(smem, ready, sketched, skbuf, stages, ntasks, warp - decoders - 2, sketchers, signs, sk_out, tags,
                       tag_base, skt, tpc);
    }
  }
  if constexpr (SKK == 0)
    if (tags) tag_warp(wstore, (u64)nbs * nchunks * SB, tag_base, tagq, tags);
}

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

__global__ void empty_kernel() {}

}  // namespace wt

using namespace wt;

// The groups (16-row blocks a CTA's chunk advances by) and the launch: groups <= 0 takes as many as fill the device at
// the occupancy the shared memory and registers allow. *launched gets the grid; with w == nullptr nothing is launched
// (the fused kernel's caller needs the groups for its tweak steps first).
template <typename Kern, typename Launch>
static int launch_gemm(Kern kern, int smem, int threads, int groups, int nchunks, int nbs, bool dry, int *launched,
                       Launch launch) {
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 2;
  int gr = groups;
  if (gr <= 0) {
    int per_sm = 0, dev = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, threads, smem)) return 2;
    if (per_sm < 1) return 6;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    gr = (per_sm * sms) / nchunks;
    if (gr < 1) gr = 1;
    if (gr > nbs) gr = nbs;
  }
  if (launched) *launched = gr * nchunks;
  if (dry) return 0;
  launch(gr);
  return cudaGetLastError() ? 4 : 0;
}

#define WT_K_SWITCH(k, F) switch (k) { case 1: F(1); break; case 2: F(2); break; default: return 1; }

extern "C" {

int wt_constants(int *out) {
  out[0] = NBITS; out[1] = NL; out[2] = S; out[3] = PB; out[4] = TW; out[5] = U; out[6] = (int)CF; out[7] = WPB;
  out[8] = TC_OK ? 2 * (U + 1) : 0; out[9] = TC_WORD_BYTES; out[10] = MAXK; out[11] = SB; out[12] = FUSED_WARPS;
  out[13] = FUSED_SPAN; out[14] = TAG_BLOCK;
  return 0;
}

// c (k elements of NL limbs), tau and p in host memory.
int wt_set_key(const u32 *c, int k, const u32 *tau, const u32 *p) {
  if (k < 1 || k > MAXK) return 1;
  u32 cc[MAXK][NL] = {};
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < NL; ++i) cc[j][i] = c[j * NL + i];
  if (cudaMemcpyToSymbol(kC, cc, sizeof cc) || cudaMemcpyToSymbol(kTau, tau, NL * 4) || cudaMemcpyToSymbol(kP, p, NL * 4))
    return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

int wt_encode(const u32 *in, uint8_t *store, u64 n, u64 base, int k, int grid, cudaStream_t s) {
#define F(K) encode_kernel<K><<<grid, WPB, 0, s>>>(in, store, n, base)
  WT_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

int wt_decode(const uint8_t *store, u32 *out, u64 n, u64 base, int k, unsigned long long *xr, int grid, cudaStream_t s) {
#define F(K) do { if (out) decode_kernel<K, 2><<<grid, WPB, 0, s>>>(store, out, n, base, xr); \
                  else decode_kernel<K, 0><<<grid, WPB, 0, s>>>(store, out, n, base, xr); } while (0)
  WT_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

int wt_read(const void *store, u64 bytes, unsigned long long *xr, int grid, cudaStream_t s) {
  read_kernel<<<grid, 256, 0, s>>>(reinterpret_cast<const uint4 *>(store), bytes / 16, xr);
  return cudaGetLastError() ? 4 : 0;
}

int wt_tc_decode(const uint8_t *store, u32 *out, u64 n, u64 base, int k, unsigned long long *xr, int grid,
                 cudaStream_t s) {
  if constexpr (!TC_OK) {
    return 5;
  } else {
#define F(K) do { if (out) tc_decode_kernel<K, 2><<<grid, TC_WARPS * 32, 0, s>>>(store, out, n, base, xr); \
                  else tc_decode_kernel<K, 0><<<grid, TC_WARPS * 32, 0, s>>>(store, out, n, base, xr); } while (0)
    WT_K_SWITCH(k, F)
#undef F
    return cudaGetLastError() ? 4 : 0;
  }
}

int wt_tc_square_gate(const u32 *in, u32 *out, int n, cudaStream_t s) {
  if constexpr (!TC_OK) return 5;
  tc_square_gate<<<n < 1024 ? n : 1024, 32, 0, s>>>(in, out, n);
  return cudaGetLastError() ? 4 : 0;
}

// what 0: CUDA-core squarings (ilp words a thread); 1: CUDA-core decodes; 2: CUDA-core encodes; 3: tensor-core
// squarings (ilp words a warp). steps a thread (or warp). block: threads a CTA for 0-2.
int wt_chain(int what, int k, int ilp, u64 steps, unsigned long long *xr, int grid, cudaStream_t s) {
  if (what == 0) {
    if (ilp == 1) chain_kernel<0, 1, 1><<<grid, 128, 0, s>>>(steps, xr);
    else if (ilp == 2) chain_kernel<0, 1, 2><<<grid, 128, 0, s>>>(steps, xr);
    else return 1;
  } else if (what == 1) {
#define F(K) chain_kernel<1, K, 1><<<grid, 128, 0, s>>>(steps, xr)
    WT_K_SWITCH(k, F)
#undef F
  } else if (what == 2) {
#define F(K) chain_kernel<2, K, 1><<<grid, 128, 0, s>>>(steps, xr)
    WT_K_SWITCH(k, F)
#undef F
  } else if (what == 4) {
    b3_chain_kernel<<<grid, 128, 0, s>>>(steps, xr);
  } else if (what == 5) {
    if (ilp == 0) pipe_kernel<0><<<grid, 128, 0, s>>>(steps, xr);
    else if (ilp == 1) pipe_kernel<1><<<grid, 128, 0, s>>>(steps, xr);
    else if (ilp == 2) pipe_kernel<2><<<grid, 128, 0, s>>>(steps, xr);
    else if (ilp == 3) pipe_kernel<3><<<grid, 128, 0, s>>>(steps, xr);
    else return 1;
  } else if (what == 3) {
    if constexpr (!TC_OK) return 5;
    if (ilp == 1) tc_chain_kernel<1><<<grid, TC_WARPS * 32, 0, s>>>(steps, xr);
    else if (ilp == 2) tc_chain_kernel<2><<<grid, TC_WARPS * 32, 0, s>>>(steps, xr);
    else return 1;
  } else {
    return 1;
  }
  return cudaGetLastError() ? 4 : 0;
}

int wt_mix(u64 sc, u64 stc, int cw, unsigned long long *xr, int grid, cudaStream_t s) {
  if constexpr (!TC_OK) return 5;
  if (cw < 0 || cw > MIX_WARPS) return 1;
  const int smem = (MIX_WARPS - cw) * TC_WORD_BYTES;
  if (smem > 48 * 1024 && cudaFuncSetAttribute(mix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 2;
  mix_kernel<<<grid, MIX_WARPS * 32, smem, s>>>(sc, stc, cw, xr);
  return cudaGetLastError() ? 4 : 0;
}

// mode 0 plain, 1 plain + knob MMAs a tile, 2 plain + knob IMADs a tile.
int wt_gemm(int mode, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, int partials, int consumers,
            int stages, int groups, int knob, unsigned long long *xr, int *launched, cudaStream_t s) {
  if (Kd % TASK_COLS || N % 16 || M < 1 || M > 32 || consumers < 1 || stages < 1 || consumers > WS_MAX_CONSUMERS ||
      mode < 0 || mode > 2)
    return 1;
  const int nb8 = M <= 8 ? 1 : (M <= 16 ? 2 : 4);
  const int nchunks = Kd / TASK_COLS, nbs = N / 16, threads = (consumers + 1) * 32;
  auto go = [&](auto kern, int smem) -> int {
    return launch_gemm(kern, smem, threads, groups, nchunks, nbs, w == nullptr, launched, [&](int gr) {
      kern<<<gr * nchunks, threads, smem, s>>>(w, x, y, M, N, Kd, gr, consumers, stages, knob, xr);
    });
  };
#define G(MO, NB) \
  return partials ? go(gemm_ws_kernel<MO, NB, true>, ws_smem_bytes<NB>(stages)) \
                  : go(gemm_ws_kernel<MO, NB, false>, ws_smem_bytes<NB>(stages))
#define GN(MO) do { if (nb8 == 1) G(MO, 1); else if (nb8 == 2) G(MO, 2); else G(MO, 4); } while (0)
  if (mode == 0) GN(0);
  if (mode == 1) GN(1);
  GN(2);
#undef GN
#undef G
}

// The fused decode GEMM at k rounds: `decoders` decoder warps, an MMA warp, a producer and `taggers` tag warps (block
// jl's ro-blake3/v1 tag, at index tag_base + jl, into tags[8 jl ..]), `stages` stored stages; tsteps as fused_kernel's.
// wt_fused_sketch instead has `sketchers` sketch warps (k = 1, no tag warps): ro-sketch-blake3/v0 at k_s = sketch_k
// over tiles of skt tasks, the signs as sketching.sign_fragments, each vector v's tag at tag_base + v into
// tags[8 v ..] and, if sk_out, its sketch at sk_out[k_s v ..]; tags and sk_out hold sketch_vectors(...) vectors.
int wt_fused_sketch(int k, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, u64 base, int partials,
                    int decoders, int stages, int groups, int taggers, const u32 *tsteps, u32 *tags, u64 tag_base,
                    int sketch_k, int sketchers, const u32 *signs, int *sk_out, int skt, int *launched,
                    cudaStream_t s) {
  if (Kd % TASK_COLS || N % 16 || M < 1 || M > 32 || k < 1 || k > MAXK || decoders < FUSED_SPAN || taggers < 0 ||
      decoders + 2 + taggers + sketchers > FUSED_WARPS || stages < (32 * decoders + TW - 1) / TW ||
      (w && taggers && !tags))
    return 1;
  if (sketch_k ? (sketch_k != 16 && sketch_k != 64) || k != 1 || taggers || sketchers < 1 || sketchers > 4 ||
                     sketchers > SK_M || skt < 1 || (w && (!signs || !(tags || sk_out)))
               : sketchers != 0)
    return 1;
  const int nb8 = M <= 8 ? 1 : (M <= 16 ? 2 : 4);
  const int nchunks = Kd / TASK_COLS, nbs = N / 16, threads = (decoders + 2 + taggers + sketchers) * 32;
  auto go = [&](auto kern, int smem) -> int {
    return launch_gemm(kern, smem, threads, groups, nchunks, nbs, w == nullptr, launched, [&](int gr) {
      kern<<<gr * nchunks, threads, smem, s>>>(static_cast<const uint8_t *>(w), x, y, M, N, Kd, base, gr, decoders,
                                               stages, tsteps, tags, tag_base, signs, sk_out, skt, sketchers);
    });
  };
#define G(KK, NB, SK) \
  return partials ? go(fused_kernel<KK, NB, true, SK>, fused_smem_bytes<NB, SK>(stages)) \
                  : go(fused_kernel<KK, NB, false, SK>, fused_smem_bytes<NB, SK>(stages))
#define GN(KK, SK) do { if (nb8 == 1) G(KK, 1, SK); else if (nb8 == 2) G(KK, 2, SK); else G(KK, 4, SK); } while (0)
  if (sketch_k == 16) GN(1, 16);
  if (sketch_k == 64) GN(1, 64);
  if (k == 1) GN(1, 0);
  GN(2, 0);
#undef GN
#undef G
}

int wt_fused(int k, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, u64 base, int partials,
             int decoders, int stages, int groups, int taggers, const u32 *tsteps, u32 *tags, u64 tag_base,
             int *launched, cudaStream_t s) {
  return wt_fused_sketch(k, w, x, y, M, N, Kd, base, partials, decoders, stages, groups, taggers, tsteps, tags, tag_base,
                         0, 0, nullptr, nullptr, 1, launched, s);
}

// The vectors a sketched launch tags: 16 a tile, ceil(ceil(nbs / groups) / skt) tiles a CTA.
u64 wt_sketch_vectors(int N, int Kd, int groups, int skt) {
  const u64 nbs = N / 16, per = (nbs + groups - 1) / groups;
  return (u64)groups * (Kd / TASK_COLS) * ((per + skt - 1) / skt) * 16;
}

// ro-blake3/v1 for one window: K as BLAKE3's 8 little-endian key words, and the leaves' prefix block
// Z(LEAF_DOMAIN ‖ frame(name) ‖ frame(σ)) (128 bytes), whose midstate the leaves start from.
static Mid g_leafmid;

int wt_set_dbg(uint8_t *p) {
#ifdef WT_DEBUG_DUMP
  return cudaMemcpyToSymbol(g_dbg, &p, sizeof p) ? 2 : 0;
#else
  return p ? 5 : 0;
#endif
}

int wt_set_tag(const u32 *key, const uint8_t *leaf_prefix) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int b = 0; b < 8; ++b) v = (v << 8) | leaf_prefix[8 * i + b];
    w[i] = v;
  }
  for (int i = 0; i < 8; ++i) g_leafmid.h[i] = SHA512_IV_HOST[i];
  sha512_compress(g_leafmid.h, w);
  const u32 one = 1;
  if (cudaMemcpyToSymbol(kTagKey, key, 8 * 4) || cudaMemcpyToSymbol(kOne, &one, 4)) return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

// leaves (ceil(nblk / 16) digests of 8 u64) from the tags of blocks [0, nblk).
int wt_tag_leaves(const u32 *tags, u64 nblk, u64 *leaves, cudaStream_t s) {
  const u64 n = (nblk + 15) / 16;
  tag_leaves_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(tags, nblk, g_leafmid, leaves);
  return cudaGetLastError() ? 4 : 0;
}

// The tree over the n leaves at nodes[0 ..], each level after the one below it; *root gets the root's node index.
int wt_tag_tree(u64 *nodes, u64 n, unsigned long long *root, cudaStream_t s) {
  u64 off = 0;
  while (n > 1) {
    const u64 up = (n + 1) / 2;
    tag_level_kernel<<<(unsigned)((up + 127) / 128), 128, 0, s>>>(nodes + 8 * off, n, nodes + 8 * (off + n));
    off += n;
    n = up;
  }
  *root = off;
  return cudaGetLastError() ? 4 : 0;
}

int wt_spin(long long cycles, cudaStream_t s) {
  spin_kernel<<<1, 1, 0, s>>>(cycles);
  return cudaGetLastError() ? 4 : 0;
}

int wt_empty(int grid, cudaStream_t s) {
  empty_kernel<<<grid, 32, 0, s>>>();
  return cudaGetLastError() ? 4 : 0;
}

}  // extern "C"
