// Sloth over one wide word on the RTX PRO 6000 (sm_120): `wide.py` is the reference. One build per field, WT_FIELD = n
// (255, 511, 521, 767, 1023, 1279, or the full-slot 512, 768, 1024, 1536 and 2048; p = 2^n - c as in `wide.FIELDS`).
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
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#elif WT_FIELD == 1024
constexpr int NBITS = 1024, TW = 64;
constexpr u32 CF = 105;
#elif WT_FIELD == 1536
constexpr int NBITS = 1536, TW = 43;
constexpr u32 CF = 3453;
#elif WT_FIELD == 2048
constexpr int NBITS = 2048, TW = 32;
constexpr u32 CF = 1557;
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
constexpr int MAXK = 3;
constexpr int TASK_BYTES = 8192;
constexpr int SB = TW * S;                              // a task's stored bytes
// WT_MASK=1: `wide.py`'s masked word, σ_K (kMask) between consecutive rounds. WT_FOLD6=1: sqr is ra-squaring-ceiling's
// FOLD 6 (`benchmarks/word_codec/kv.cu`'s KV_FOLD6), its half squares ordered to fit the p1024 12-warp fused kernels;
// 2: its half squares folded as they are made (sqr_fold6_lean).
#ifndef WT_MASK
#define WT_MASK 0
#endif
#ifndef WT_FOLD6
#define WT_FOLD6 0
#endif
constexpr bool MASKED = WT_MASK, FOLD6 = WT_FOLD6;
static_assert(!MASKED || FULL_SLOT, "σ_K's test against p is the full-slot one");
static_assert(!FOLD6 || (FULL_SLOT && NL % 2 == 0), "FOLD 6 halves the limbs, and folds at full slot");

__constant__ u32 kC[MAXK][NL];
__constant__ u32 kD[MAXK][NL];   // (c_0 - c_j) mod p, the decode's round j against w = -(c_0 + t)
__constant__ u32 kTau[NL];
__constant__ u32 kP[NL];
__constant__ u32 kMask[NL];
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

// Carry chains in PTX (one SASS instruction a limb, against two to three for the u64 forms below). The flag lives
// between consecutive asm volatile statements, which the compiler keeps in order.
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

// x += a (a < 2^32) into NL limbs; returns the carry out of 2^(32 NL). The carry leaves limb 0 with probability
// a / 2^32 (every caller's a is at most c^2 + c < 2^24), so the rest of the chain is a branch.
__device__ __forceinline__ u32 add_small(u32 (&x)[NL], u32 a) {
  x[0] += a;
  if (x[0] >= a) return 0u;
  x[1] = add_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
  return addc(0u, 0u);
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
    if (add_small(x, (u32)carry * CF)) add_small(x, CF);
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

// At full slot, limb k of the lazy x takes t's columns k and k + NL at once (r and h, each pair once), so t is never
// held: x = sum (r_k + c h_k + carry) 2^(32k) = t mod p, then fold's second stage. The low columns' carries stay in
// that sum, so its carry out of 2^n, (c1, c0), is below (NL + 1) 2^32 + c, not c: (c1, c0) c < 2^50 goes back in.
// It needs fewer registers than sqr_full + fold but runs 1-16% slower (three multiplies by c a limb, not one), so
// only the fused decoders that need the registers use it (decode_word_fresh); sqr and mul, the encode's, do not.
// x must not be a.
__device__ __forceinline__ void fold_limb(u32 &x, u32 &c0, u32 &c1, const u32 (&r)[3], const u32 (&h)[3]) {
  u64 v = (u64)h[0] * CF + r[0] + c0;
  x = (u32)v;
  v = (v >> 32) + (u64)h[1] * CF + r[1] + c1;
  c0 = (u32)v;
  c1 = (u32)(v >> 32) + h[2] * CF + r[2];
}
__device__ __forceinline__ void fold_top(u32 (&x)[NL], u32 c0, u32 c1) {
  u64 c = (((u64)c1 << 32) | c0) * CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += x[k];
    x[k] = (u32)c;
    c >>= 32;
  }
  x[0] += (u32)c * CF;
}
__device__ __forceinline__ void dbl3(u32 (&r)[3]) {
  r[2] = (r[2] << 1) | (r[1] >> 31);
  r[1] = (r[1] << 1) | (r[0] >> 31);
  r[0] <<= 1;
}

__device__ __forceinline__ void sqr_fold(const u32 (&a)[NL], u32 (&x)[NL]) {
  u32 c0 = 0, c1 = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    u32 r[3] = {0, 0, 0}, h[3] = {0, 0, 0};
#pragma unroll
    for (int i = 0; 2 * i < k; ++i) mac(r, a[i], a[k - i]);
#pragma unroll
    for (int i = k + 1; 2 * i < k + NL; ++i) mac(h, a[i], a[k + NL - i]);
    dbl3(r);
    dbl3(h);
    if ((k & 1) == 0) mac(r, a[k / 2], a[k / 2]);
    if (((k + NL) & 1) == 0) mac(h, a[(k + NL) / 2], a[(k + NL) / 2]);
    fold_limb(x[k], c0, c1, r, h);
  }
  fold_top(x, c0, c1);
}

// WT_FOLD6=1: ra-squaring-ceiling's FOLD 6 (`benchmarks/rate_squaring/sq.cu`, as `benchmarks/word_codec/kv.cu`'s
// KV_FOLD6). sqr_dbl_n is its FOLD 5 square: the off-diagonal products column by column, each column's accumulator
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
// D = |a0 - a1|^2, each a FOLD 5 square. A and D come first and go into E = A_hi + (A - D) (2H limbs and a signed
// top word es) before C starts: a1 is remade from a0 and d (a0 - d, or a0 + d when a0 < a1) with a dependence on es
// ((es + 1) >> 2 = 0), so neither a1 nor A and D are held across C. Two half squares' columns are in flight at once
// and no third's inputs wait beside them, which the 12-warp fused decoders' 168 registers hold beside the tweak
// (A, C, then D spilled 16-32 bytes there). a^2 = A_lo + B (E + es B^2) + B C + B^2 C.
__device__ __forceinline__ void sqr_fold6(const u32 (&a)[NL], u32 (&t)[2 * NL]) {
  constexpr int H = NL / 2;
  u32 x0[H], d[H], e[H];
#pragma unroll
  for (int k = 0; k < H; ++k) x0[k] = a[k];
  d[0] = sub_cc(a[0], a[H]);
#pragma unroll
  for (int k = 1; k < H; ++k) d[k] = subc_cc(a[k], a[H + k]);
  const u32 mask = subc(0u, 0u);   // a0 < a1
  e[0] = sub_cc(a[H], a[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) e[k] = subc_cc(a[H + k], a[k]);
#pragma unroll
  for (int k = 0; k < H; ++k) d[k] = (d[k] & ~mask) | (e[k] & mask);
  u32 A[2 * H], D[2 * H], F[2 * H], E[2 * H];
  sqr_dbl_n<H>(x0, A);
  sqr_dbl_n<H>(d, D);
  F[0] = sub_cc(A[0], D[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) F[k] = subc_cc(A[k], D[k]);
  const u32 f = subc(0u, 0u);   // A - D's sign word
  E[0] = add_cc(A[H], F[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) E[k] = addc_cc(A[H + k], F[k]);
#pragma unroll
  for (int k = H; k < 2 * H; ++k) E[k] = addc_cc(F[k], 0u);
  const u32 es = addc(f, 0u);   // -1, 0 or 1
  // a1 = a0 + (d ^ nm) + (nm & 1): nm all ones (a0 - d) or 0 (a0 + d); nm + nm carries nm & 1 in.
  const u32 nm = ~mask | ((es + 1u) >> 2);
  u32 x1[H];
  x1[0] = add_cc(nm, nm);
  x1[0] = addc_cc(x0[0], d[0] ^ nm);
#pragma unroll
  for (int k = 1; k < H; ++k) x1[k] = addc_cc(x0[k], d[k] ^ nm);
  u32 C[2 * H];
  sqr_dbl_n<H>(x1, C);
#pragma unroll
  for (int k = 0; k < H; ++k) t[k] = A[k];
  t[H] = add_cc(E[0], C[0]);
#pragma unroll
  for (int k = 1; k < 2 * H; ++k) t[H + k] = addc_cc(E[k], C[k]);
  const u32 top = addc(es, 0u);   // es plus the carry, signed, sign-extended to 2^(64 H)
  const u32 ext = (u32)((int)top >> 31);
  t[3 * H] = top;
#pragma unroll
  for (int k = 3 * H + 1; k < 4 * H; ++k) t[k] = ext;
  t[2 * H] = add_cc(t[2 * H], C[0]);
#pragma unroll
  for (int k = 1; k < 2 * H - 1; ++k) t[2 * H + k] = addc_cc(t[2 * H + k], C[k]);
  t[4 * H - 1] = addc(t[4 * H - 1], C[2 * H - 1]);
}

// fold at full slot as FOLD 1's carry chains (kv.cu's fold): lo + c hi, the low products' chain then the high ones'
// a limb up, whose carry word h <= c; then h c into the low limbs, and c once more if that carried out. The same x.
__device__ __forceinline__ void fold_cc(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
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

__device__ __forceinline__ u32 sub_small(u32 (&x)[NL], u32 b) {
  const u32 lo = x[0];
  x[0] = lo - b;
  if (lo >= b) return 0u;
  x[1] = sub_cc(x[1], 1u);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = subc_cc(x[k], 0u);
  return subc(0u, 0u) & 1u;
}

// x[H..NL) += Y[0..H): B Y_lo; the carry into h.
template <int H>
__device__ __forceinline__ void add_lo_high(u32 (&x)[NL], const u32 (&Y)[NL], u32 &h) {
  x[H] = add_cc(x[H], Y[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) x[H + k] = addc_cc(x[H + k], Y[k]);
  h = addc(h, 0u);
}

// x[0..H] += c Y[H..2H): B Y_hi = c Y_hi (mod p); the carries out of limb H into cm.
template <int H>
__device__ __forceinline__ void add_hi_low(u32 (&x)[NL], const u32 (&Y)[NL], u32 &cm) {
  x[0] = madlo_cc(Y[H], CF, x[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) x[k] = madclo_cc(Y[H + k], CF, x[k]);
  x[H] = addc_cc(x[H], 0u);
  cm = addc(cm, 0u);
  x[1] = madhi_cc(Y[H], CF, x[1]);
#pragma unroll
  for (int k = 2; k <= H; ++k) x[k] = madchi_cc(Y[H + k - 1], CF, x[k]);
  cm = addc(cm, 0u);
}

// WT_FOLD6=2: FOLD 6's three half squares (A = a0^2, C = a1^2, D = |a0 - a1|^2 with B = 2^(n/2)), each folded into x as
// soon as it is made, so neither the double-width square nor the Karatsuba sum is held: B^2 = 2^n = c (mod p), so
// a^2 = A (1 + B) + C (c + B) - B D, and B Y = B Y_lo + c Y_hi for each. The carries out of limb H wait in cm and those
// out of 2^n in h (both signed, two's complement) until the end. x < 2^n, x = a^2 (mod p).
__device__ __forceinline__ void sqr_fold6_lean(const u32 (&a)[NL], u32 (&x)[NL]) {
  constexpr int H = NL / 2;
  u32 a0[H], a1[H], d[H], e[H];
#pragma unroll
  for (int k = 0; k < H; ++k) {
    a0[k] = a[k];
    a1[k] = a[H + k];
  }
  d[0] = sub_cc(a0[0], a1[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) d[k] = subc_cc(a0[k], a1[k]);
  const u32 mask = subc(0u, 0u);
  e[0] = sub_cc(a1[0], a0[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) e[k] = subc_cc(a1[k], a0[k]);
#pragma unroll
  for (int k = 0; k < H; ++k) d[k] = (d[k] & ~mask) | (e[k] & mask);
  u32 Y[NL], h = 0, cm = 0;
  // Each half square's inputs take h >> 16 (always 0: h <= 2c + 4) after the previous one is folded, so the compiler
  // makes them one at a time and holds one half square's columns, not three.
  sqr_dbl_n<H>(a0, Y);
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = Y[k];
  add_lo_high<H>(x, Y, h);
  add_hi_low<H>(x, Y, cm);
#pragma unroll
  for (int k = 0; k < H; ++k) a1[k] ^= (h + cm) >> 16;
  sqr_dbl_n<H>(a1, Y);
  x[0] = madlo_cc(Y[0], CF, x[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = madclo_cc(Y[k], CF, x[k]);
  h = addc(h, 0u);
  x[1] = madhi_cc(Y[0], CF, x[1]);
#pragma unroll
  for (int k = 2; k < NL; ++k) x[k] = madchi_cc(Y[k - 1], CF, x[k]);
  h = madchi(Y[NL - 1], CF, h);
  add_lo_high<H>(x, Y, h);
  add_hi_low<H>(x, Y, cm);
#pragma unroll
  for (int k = 0; k < H; ++k) d[k] ^= (h + cm) >> 16;
  sqr_dbl_n<H>(d, Y);
  x[H] = sub_cc(x[H], Y[0]);
#pragma unroll
  for (int k = 1; k < H; ++k) x[H + k] = subc_cc(x[H + k], Y[k]);
  h = subc(h, 0u);
  u32 P[H + 1];
#pragma unroll
  for (int k = 0; k < H; ++k) P[k] = Y[H + k] * CF;
  P[H] = 0u;
  P[1] = madhi_cc(Y[H], CF, P[1]);
#pragma unroll
  for (int k = 2; k < H; ++k) P[k] = madchi_cc(Y[H + k - 1], CF, P[k]);
  P[H] = madchi(Y[2 * H - 1], CF, 0u);
  x[0] = sub_cc(x[0], P[0]);
#pragma unroll
  for (int k = 1; k <= H; ++k) x[k] = subc_cc(x[k], P[k]);
  cm = subc(cm, 0u);
  const u32 ext = (u32)((int)cm >> 31);
  x[H + 1] = add_cc(x[H + 1], cm);
#pragma unroll
  for (int k = H + 2; k < NL; ++k) x[k] = addc_cc(x[k], ext);
  h = addc(h, ext);
  const int hs = (int)h;
  if (hs >= 0) {
    if (add_small(x, (u32)hs * CF)) add_small(x, CF);
  } else if (sub_small(x, (u32)(-hs) * CF)) {
    sub_small(x, CF);
  }
}

__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  if constexpr (WT_FOLD6 == 2) {
    sqr_fold6_lean(a, x);
    return;
  }
  u32 t[2 * NL];
  if constexpr (FOLD6) {
    sqr_fold6(a, t);
    fold_cc(t, x);
  } else {
    sqr_full(a, t);
    fold(t, x);
  }
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

// x < 2p to [0, p), as canon. At full slot x >= p needs x's top limb all ones (p's limbs above the lowest are, c < 2^32),
// with probability 2^-32 a word.
__device__ __forceinline__ void canon_cc(u32 (&x)[NL]) {
  if constexpr (FULL_SLOT) {
    if (x[NL - 1] != 0xffffffffu) return;
  }
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

// σ_K(x) = x ^ K when that is below p, else x, x canonical at full slot (kv.cu's sigma): x ^ K >= p needs its limbs
// above the lowest all ones and its lowest at least 2^32 - c, when the XOR is undone. Its top limb all ones
// (probability 2^-32) is the one test on the fast path.
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

__device__ __forceinline__ void sigma_k(u32 (&x)[NL]) {
  if constexpr (MASKED) {
    u32 m[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) m[k] = kMask[k];
    sigma(x, m);
  }
}

// x = a + b mod p, a + b < 2p, as addmod. At full slot, when s = a + b carries out of 2^n, s - p = (s - 2^n) + c < p
// carries out of limb 0 only rarely (add_small); otherwise s < 2^n and canon_cc takes off p if s >= p.
__device__ __forceinline__ void addmod_cc(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  if constexpr (FULL_SLOT) {
    x[0] = add_cc(a[0], b[0]);
#pragma unroll
    for (int k = 1; k < NL; ++k) x[k] = addc_cc(a[k], b[k]);
    add_small(x, CF & (0u - addc(0u, 0u)));
    canon_cc(x);
    return;
  }
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

// q = p - q when neg, q canonical (so 0 becomes p). At full slot p's limbs above the lowest are all ones, so p - q
// complements them unless the lowest limb borrows (q[0] > 2^32 - c, rare).
__device__ __forceinline__ void neg_if(u32 (&q)[NL], bool neg) {
  if constexpr (FULL_SLOT) {
    if (!neg || q[0] <= 0u - CF) {
      const u32 m = 0u - (u32)neg;
      q[0] = neg ? (0u - CF) - q[0] : q[0];
#pragma unroll
      for (int k = 1; k < NL; ++k) q[k] ^= m;
      return;
    }
    q[0] = sub_cc(0u - CF, q[0]);
#pragma unroll
    for (int k = 1; k < NL - 1; ++k) q[k] = subc_cc(0xffffffffu, q[k]);
    q[NL - 1] = subc(0xffffffffu, q[NL - 1]);
    return;
  }
  u32 z[NL];
  z[0] = sub_cc(kP[0], q[0]);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) z[k] = subc_cc(kP[k], q[k]);
  z[NL - 1] = subc(kP[NL - 1], q[NL - 1]);
#pragma unroll
  for (int k = 0; k < NL; ++k) q[k] = neg ? z[k] : q[k];
}

// x = a - b mod p, a and b canonical, as submod. At full slot, on a borrow r = a - b + 2^n >= c + 1 and a - b + p = r - c,
// which borrows past limb 0 only when r's low limb is below c.
__device__ __forceinline__ void submod_cc(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  x[0] = sub_cc(a[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = subc_cc(a[k], b[k]);
  if constexpr (FULL_SLOT) {
    const u32 d = CF & subc(0u, 0u), lo = x[0];
    x[0] = lo - d;
    if (lo >= d) return;
    x[1] = sub_cc(x[1], 1u);
#pragma unroll
    for (int k = 2; k < NL; ++k) x[k] = subc_cc(x[k], 0u);
    return;
  }
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

// x = a + b mod p, a + b < 2p, a and b below 2^n.
__device__ __forceinline__ void addmod_lt2p(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  if constexpr (FULL_SLOT) {
    addmod_cc(a, b, x);
  } else {
    add_nc(a, b, x);
    canon_cc(x);
  }
}

// One round's decode against w = -(c_0 + t) mod p (canonical), which the caller keeps in place of t:
// rho^-1(y) - c_0 - t = (y^2 negated when y is odd) + w.
__device__ __forceinline__ void decode_word_w(u32 (&y)[NL], const u32 (&w)[NL]) {
  u32 q[NL];
  sqr(y, q);
  canon_cc(q);
  neg_if(q, y[0] & 1);
  addmod_lt2p(q, w, y);
}

// Round j >= 1's rho^-1(y) - c_j - t from rho^-1(y) - c_0 - t (in y): plus kD[j] = (c_0 - c_j) mod p, then σ_K.
__device__ __forceinline__ void undo_round(u32 (&y)[NL], int j) {
  u32 d[NL], z[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) d[k] = kD[j][k];
  addmod_lt2p(y, d, z);
  copy(y, z);
  sigma_k(y);
}

// K rounds against w as decode_word_w, in one loop: peeling the last round puts a second copy of the round's code
// (1,800 instructions at p1024) in every decoder.
template <int K>
__device__ __forceinline__ void decode_word_wk(u32 (&y)[NL], const u32 (&w)[NL]) {
#pragma unroll 1
  for (int j = K - 1; j >= 0; --j) {
    decode_word_w(y, w);
    if (j > 0) undo_round(y, j);
  }
}

// u = (c_j + tau i) mod p, canonical, at full slot (2^n = c; FRESH_DECODE, its one caller, is full slot only) and
// without a double-width product: limb k sums c_j's, tau_k lo and tau_(k-1) hi into the carry (three words), and the
// part above 2^n (< 2^66) comes back times c.
__device__ __forceinline__ void ct_of(u64 i, int j, u32 (&u)[NL]) {
  const u32 lo = (u32)i, hi = (u32)(i >> 32);
  u32 r[3] = {0, 0, 0};
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    mac(r, kTau[k], lo);
    if (k > 0) mac(r, kTau[k - 1], hi);
    add3(r, kC[j][k], 0u);
    u[k] = r[0];
    r[0] = r[1];
    r[1] = r[2];
    r[2] = 0;
  }
  mac(r, kTau[NL - 1], hi);
  u64 v = (u64)r[0] * CF;
  const u32 a0 = (u32)v;
  v = (v >> 32) + (u64)r[1] * CF;
  const u32 a1 = (u32)v;
  const u32 a2 = (u32)(v >> 32) + r[2] * CF;
  u[0] = add_cc(u[0], a0);
  u[1] = addc_cc(u[1], a1);
  u[2] = addc_cc(u[2], a2);
#pragma unroll
  for (int k = 3; k < NL; ++k) u[k] = addc_cc(u[k], 0u);
  const u32 co = addc(0u, 0u);   // then u < c 2^66, so the c below carries at most into u[2]
  u[0] = add_cc(u[0], co * CF);
  u[1] = addc_cc(u[1], 0u);
  u[2] = addc(u[2], 0u);
  canon_cc(u);
}

// Round j's decode rho^-1(y) - c_j - t with c_j + t made from the word's index after the square (ct_of), so no tweak
// is held across words. F6: FOLD 6's square (sqr), else the column square.
template <bool F6>
__device__ __forceinline__ void decode_word_fresh(u32 (&y)[NL], u64 i, int j) {
  u32 q[NL], u[NL];
  if constexpr (F6) sqr(y, q);
  else sqr_fold(y, q);
  canon_cc(q);
  neg_if(q, y[0] & 1);
  ct_of(i, j, u);
  submod_cc(q, u, y);
}

// K rounds as decode_word_fresh, σ_K after each but the last. One loop over every round: with the last peeled, k = 2's
// one-trip loop unrolls into two rounds' straight line, which spills at 128 registers. At k >= 2 (only at 128
// registers, fresh_decode) the rounds square by columns even in a FOLD 6 build: lean FOLD 6 there spills 16 bytes.
template <int K>
__device__ __forceinline__ void decode_word_freshk(u32 (&y)[NL], u64 i) {
#pragma unroll 1
  for (int j = K - 1; j >= 0; --j) {
    decode_word_fresh<FOLD6 && K == 1>(y, i, j);
    if (j > 0) sigma_k(y);
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
    if (j > 0) sigma_k(x);
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
  {
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
      decode_word_wk<K>(y, t);
      if (MODE == 2) {
#pragma unroll
        for (int k = 0; k < NL; ++k) out[i * NL + k] = y[k];
      }
      acc ^= fold64(y);
    }
    u32 t2[NL];
    submod_cc(t, step, t2);
    copy(t, t2);
  }
  atomicXor(xr, (unsigned long long)acc);
}

// The decode's field primitives a word a thread, for the gate at their rare carries (`kv_twin.rare_cases`): op 0
// addmod_cc(a, b), 1 submod_cc(a, b), 2 neg_if(a, b odd), 3 canon_cc(a), 4 sqr(a), and at full slot 5 sigma(a) with b
// as K and 6 sqr_fold(a), the fused decoders' column square.
__global__ void __launch_bounds__(WPB) prim_kernel(int op, const u32 *__restrict__ a, const u32 *__restrict__ b,
                                                   u32 *__restrict__ out, int n) {
  const int w = blockIdx.x * WPB + threadIdx.x;
  if (w >= n) return;
  u32 x[NL], y[NL], z[NL];
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    x[k] = a[(u64)w * NL + k];
    y[k] = b[(u64)w * NL + k];
  }
  if (op == 0) {
    addmod_cc(x, y, z);
  } else if (op == 1) {
    submod_cc(x, y, z);
  } else if (op == 2) {
    copy(z, x);
    neg_if(z, y[0] & 1);
  } else if (op == 3) {
    copy(z, x);
    canon_cc(z);
  } else if (op == 4) {
    sqr(x, z);
  } else if constexpr (FULL_SLOT) {
    if (op == 5) {
      copy(z, x);
      sigma(z, y);
    } else {
      sqr_fold(x, z);
    }
  }
#pragma unroll
  for (int k = 0; k < NL; ++k) out[(u64)w * NL + k] = z[k];
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

// what 0: `steps` squarings of ILP words a thread (lazy, sqr); 1: `steps` decodes of K rounds; 2: `steps` encodes of K
// rounds; 3: as 0 with sqr_fold, the fused decoders' column square (full slot).
template <int WHAT, int K, int ILP>
__global__ void __launch_bounds__(128, 1) chain_kernel(u64 steps, unsigned long long *xr) {
  const u64 id = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u32 x[ILP][NL], t[NL];
#pragma unroll
  for (int w = 0; w < ILP; ++w) seed_word(id * ILP + w, x[w]);
  if (WHAT == 1 || WHAT == 2) {
    canon(x[0]);
    tweak(id, t);
  }
#pragma unroll 1
  for (u64 s = 0; s < steps; ++s) {
    if (WHAT == 0 || WHAT == 3) {
#pragma unroll
      for (int w = 0; w < ILP; ++w) {
        u32 y[NL];
        if constexpr (WHAT == 3) sqr_fold(x[w], y);
        else sqr(x[w], y);
        copy(x[w], y);
      }
    } else if (WHAT == 1) {
      decode_word_wk<K>(x[0], t);
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
static_assert(!TC_OK || U <= 5, "the plan is wide.tc_plan's for U <= 5");

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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // before `empty`, as fused_kernel's MMA warp
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

#include "sha512.cuh"

struct Mid {
  u64 h[8];
};

// The leaves' prefix midstate: wt_set_tag's (one window's), or pass_kernel's (its pass's).
__device__ Mid g_leafmid;

// The leaves over the tags of blocks [0, nblk), a thread a leaf; each digest as SHA-512's 8 state words.
__global__ void tag_leaves_kernel(const u32 *__restrict__ tags, u64 nblk, u64 *__restrict__ leaves) {
  const Mid leafmid = g_leafmid;
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

// --- the pass's receipt key (red team F29) ----------------------------------------------------------------------------
//
// Pass j's seed s_j = SHA512(PASS_DOMAIN ‖ σ ‖ u32be j ‖ root_(j−1))[:32] (sketching.pass_seed), and its receipt is
// ro-sketch-blake3/v0 under s_j: the signs K_S = derive(s_j, SIGN_DOMAIN, {k, n, tag}), the key and the leaves'
// prefix. So the ±1 matrix is fresh every pass, and no pass's receipts exist before the previous pass's root.
// wt_set_pass puts the messages' fixed bytes in kPass (zeros where s_j, j, the root, K_S and the row go); pass_kernel,
// one CTA on the stream after the previous commit, fills them in, so the chain needs no host round trip.
//
// A WT_RECEIPT = 1 build's receipt is sketch-sha512 instead (`sketch_receipt.py`, the remote bandwidth driver's kind):
// s_j is the whole 64-byte SHA-512 (`receipts.seed`), K_S = derive(s_j, "verity/remote-bandwidth/sketch-signs/v0",
// {"k": 16, "n": 4096, "tag": "sketch-sha512"}), the store's block j is the GEMM's vector j (n = 4096 at T = 8), and
// y_j = SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(s_j)) ‖ u64be j ‖ σ_j): one compression from the prefix's midstate
// (kRcptMid, the leaf slot's), 64 bytes a vector. Its leaves and tree are the driver's (reader.cu's rb_leaves, rb_tree).
#ifndef WT_RECEIPT
#define WT_RECEIPT 0
#endif
constexpr int KPASS_ROWS = 16, PASS_BITS_BYTES = 16 * 1024, PASS_THREADS = 256, SEED_BYTES = WT_RECEIPT ? 64 : 32;
struct PassMsgs {
  uint8_t seed[160], derive[256], tag[128], leaf[128], row[128];
  int seed_len, derive_len, row_len, derive_s, tag_s, leaf_s, row_key, row_c, row_ctr, ks, n;
};
struct PassState {
  uint8_t seed[64], ks[32];
  u32 key[8];
  u64 leafmid[8];
};
__constant__ PassMsgs kPass;
__device__ PassState g_pass;
__constant__ u64 kRcptMid[8];

// SHA-512 of the len bytes at m (big-endian words, the standard padding).
__device__ void sha512_bytes(const uint8_t *m, int len, u64 (&s)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = kSha512Iv[i];
  const int nb = (len + 17 + 127) / 128;
  for (int blk = 0; blk < nb; ++blk) {
    u64 w[16];
    for (int i = 0; i < 16; ++i) {
      u64 v = 0;
      for (int b = 0; b < 8; ++b) {
        const int o = 128 * blk + 8 * i + b;
        const u64 x = o < len ? m[o] : o == len ? 0x80u : 0u;
        v = (v << 8) | x;
      }
      w[i] = v;
    }
    if (blk == nb - 1) w[15] = (u64)len * 8;
    sha512_compress(s, w);
  }
}

__device__ __forceinline__ void be_bytes(const u64 (&s)[8], uint8_t *out, int len) {
  for (int b = 0; b < len; ++b) out[b] = (uint8_t)(s[b >> 3] >> (56 - 8 * (b & 7)));
}

__global__ void __launch_bounds__(PASS_THREADS) pass_kernel(unsigned j, const u64 *__restrict__ root,
                                                            u32 *__restrict__ signs, int t) {
  __shared__ uint8_t sd[64], ksd[32];
  __shared__ uint8_t bits[PASS_BITS_BYTES];
  const int tid = threadIdx.x, rb = kPass.n / 8, nb = (rb + 63) / 64;
  uint8_t m[256];
  u64 h[8];
  if (tid == 0) {
    for (int b = 0; b < kPass.seed_len; ++b) m[b] = kPass.seed[b];
    for (int b = 0; b < 4; ++b) m[64 + b] = (uint8_t)(j >> (24 - 8 * b));
    for (int b = 0; b < 64; ++b) m[68 + b] = (uint8_t)(root[b >> 3] >> (56 - 8 * (b & 7)));
    sha512_bytes(m, kPass.seed_len, h);
    be_bytes(h, sd, 64);
  }
  __syncthreads();
  if (tid == 0) {
    for (int b = 0; b < kPass.derive_len; ++b) m[b] = kPass.derive[b];
    for (int b = 0; b < SEED_BYTES; ++b) m[kPass.derive_s + b] = sd[b];
    sha512_bytes(m, kPass.derive_len, h);
    be_bytes(h, ksd, 32);
    for (int b = 0; b < 32; ++b) g_pass.ks[b] = ksd[b];
    for (int b = 0; b < 64; ++b) g_pass.seed[b] = sd[b];
  } else if (tid == 32) {
    for (int b = 0; b < 128; ++b) m[b] = kPass.tag[b];
    for (int b = 0; b < 32; ++b) m[kPass.tag_s + b] = sd[b];
    sha512_bytes(m, 128, h);
    uint8_t kb[32];
    be_bytes(h, kb, 32);
    for (int i = 0; i < 8; ++i) g_pass.key[i] = kb[4 * i] | kb[4 * i + 1] << 8 | kb[4 * i + 2] << 16 | (u32)kb[4 * i + 3] << 24;
  } else if (tid == 64) {
    for (int b = 0; b < 128; ++b) m[b] = kPass.leaf[b];
    for (int b = 0; b < SEED_BYTES; ++b) m[kPass.leaf_s + b] = sd[b];
    u64 w[16];
    for (int i = 0; i < 16; ++i) {
      u64 v = 0;
      for (int b = 0; b < 8; ++b) v = (v << 8) | m[8 * i + b];
      w[i] = v;
    }
    for (int i = 0; i < 8; ++i) h[i] = kSha512Iv[i];
    sha512_compress(h, w);
    for (int i = 0; i < 8; ++i) g_pass.leafmid[i] = g_leafmid.h[i] = h[i];
  }
  __syncthreads();
  for (int e = tid; e < kPass.ks * nb; e += PASS_THREADS) {   // row c's stream block b
    const int c = e / nb, b = e % nb;
    for (int x = 0; x < kPass.row_len; ++x) m[x] = kPass.row[x];
    for (int x = 0; x < 32; ++x) m[kPass.row_key + x] = ksd[x];
    m[kPass.row_c] = (uint8_t)c;
    for (int x = 0; x < 8; ++x) m[kPass.row_ctr + x] = (uint8_t)((u64)b >> (56 - 8 * x));
    sha512_bytes(m, kPass.row_len, h);
    uint8_t blk[64];
    be_bytes(h, blk, 64);
    for (int x = 0; x < 64 && 64 * b + x < rb; ++x) bits[c * rb + 64 * b + x] = blk[x];
  }
  __syncthreads();
  const int msteps = (SB + 511) / 512;
  for (int e = tid; e < t * msteps * 32; e += PASS_THREADS) {   // sketching.sign_fragments at k_s = 16 (a word a lane)
    const int lane = e & 31, st = (e >> 5) % msteps, ii = (e >> 5) / msteps, g = lane >> 2, tq = lane & 3;
    u32 wd = 0;
    for (int r = 0; r < 4; ++r)
      for (int q = 0; q < 4; ++q) {
        const int row = 8 * (r >> 1) + g, col = 32 * (ii * msteps + st) + 16 * (r & 1) + 4 * tq + q;
        const u32 bit = (bits[row * rb + (col >> 3)] >> (col & 7)) & 1u;
        wd |= bit << (16 * ((r % 6) / 3) + 4 * q + r % 3);
      }
    signs[e] = wd;
  }
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
// The barrier orders generic accesses only, so the MMA warp arrives on `empty` behind a proxy fence after its last load
// of the stage: without it the next bulk copy (async proxy) may overwrite the stage under loads still in flight.
//
// An SM's registers fit FUSED_WARPS warps at the decode's register count (a sub-partition holds 16K: four warps at
// 128 registers, three at 168, two at 255; 9 to 11 warps put three on one). The decode holds the word, its square and
// the tweak. At full slot and 24 < NL <= 48 the k = 1 decode holds neither the double-width square nor the tweak
// (FRESH_DECODE: sqr_fold, and c_0 + t made a word at a time), which fits 12 warps at NL = 48. Elsewhere it buys no
// warp and only costs time (p2048 at 8 warps: 2.92 against 2.59 untagged).
constexpr bool FRESH_DECODE = FULL_SLOT && NL > 24 && NL <= 48;
constexpr int FUSED_WARPS = FRESH_DECODE ? 12 : (NL <= 17 ? 16 : (NL <= 24 ? 12 : 8));
constexpr int FUSED_SPAN = (TW + 30) / 32 + 1;            // the most units a task's words touch

constexpr int SK_BUFS = 3;   // the sketch's tile buffers (sk_begin)
// sketch_warp hashes the tiles of a batch at once, a lane a vector (16 / SKW vectors of each tile a warp), from two
// sets of a batch's buffers.
template <int SKW>
__host__ __device__ constexpr int sk_batch() {
  return SKW ? 32 / (16 / SKW) : 1;
}
template <int SKW>
__host__ __device__ constexpr int sk_bufs() {
  return SKW ? 2 * sk_batch<SKW>() : SK_BUFS;
}
// A diagnostic build (WT_SK_DIAG, bits) drops parts of the sketch to time them, so its sketches and tags are wrong
// (bench.py gates only its partial sums): 1 the MMA warp's hash, 2 the decoders' MMAs and adds, 4 their sign loads.
#ifndef WT_SK_DIAG
#define WT_SK_DIAG 0
#endif
// sgs sign words (k_s = 16's signs, skt tasks' steps, a u32 a lane) after the sketch's buffers: wt_fused_sketch puts
// them there when they fit beside the stages, and otherwise the decoders load them from global memory (sgs = 0). The
// sketch warps' fragments are 4x the words, so they read them through L1 and L2 (sgs = 0 at SKW > 0).
template <int NB8, int SKK, int SKW>
__host__ __device__ constexpr int fused_smem_bytes(int stages, int sgs) {
  return stages * SB + NB8 * 8 * XS_STRIDE + 16 * ((6 * stages + 1) / 2) + 16 + sk_bufs<SKW>() * 64 * SKK + 4 * sgs;
}

// The CTA's warps: FUSED_WARPS, and with SKW sketch warps (sketch_warp) SKW more where the registers allow (the MMA
// warp's accumulators at NB8 = 4 don't fit 18 warps' 96 registers a thread, so there the sketch warps take decoders').
// At full slot the kernel fits 96 registers untagged and beside the sketch warps (the MMA warp holding half a task's A
// fragments at NB8 = 4), and a sub-partition holds five warps at 96: there 20, so 16 decoders put four on every
// sub-partition and the MMA warps, the producer and the sketch warps one on each (14 put four on two and three on two,
// and three a sub-partition leave its issue idle at k >= 2).
template <int NB8, int SKK, int SKW>
__host__ __device__ constexpr int fused_warps() {
  if (FULL_SLOT && NL <= 17 && (SKW > 0 || SKK == 0)) return 20;
  return FUSED_WARPS + (NB8 <= 2 ? SKW : 0);
}

// The MMA warps after the decoders: at NB8 = 4 (64 MMAs a task) and k <= 2 two, taking alternate tasks (one warp's
// MMAs at M = 32 run at 1.45 reads, above the k <= 2 decode; the k = 3 decode is slower, and wants the warp back),
// except where the MMA warp also hashes the decoders' sketches (SKK > 0, SKW = 0).
// The decoders' fresh decode (decode_word_freshk, c_j + t made each round) where four warps share a sub-partition
// (WARPS > 12, 128 registers a thread): the held w doesn't fit there at k >= 2 (p1024's 14 warps spilled 112-136
// bytes). At 12 warps (168 registers) the k >= 2 decoders hold w.
template <int K, int WARPS>
__host__ __device__ constexpr bool fresh_decode() {
  return FRESH_DECODE && (K == 1 || WARPS > 12);
}

template <int K, int NB8, int SKK, int SKW>
__host__ __device__ constexpr int fused_mma_warps() {
  return NB8 == 4 && K <= 2 && !(SKK > 0 && SKW == 0) ? 2 : 1;
}

template <int SKK, int SKW>
int fused_cap(int nb8) {
  return nb8 == 1 ? fused_warps<1, SKK, SKW>() : (nb8 == 2 ? fused_warps<2, SKK, SKW>() : fused_warps<4, SKK, SKW>());
}

template <int SKK, int SKW>
int fused_mma(int k, int nb8) {
  return nb8 == 4 && k <= 2 ? fused_mma_warps<1, 4, SKK, SKW>() : fused_mma_warps<3, 1, SKK, SKW>();
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

// --- ro-sketch-blake3/v0 (sketching.py) -------------------------------------------------------------------------------
//
// A step is 512 stored bytes of a task, lane l's 16 at 16 l the A fragment of k_s / 8 mma.m16n8k32.s8 against the
// signs (`sign_fragments`: a lane's sk_words u32 a step, one prmt a register); a tile is skt tasks, its 16 vectors the
// MMAs' rows. The decoder warps sketch the stored words they decode (`sk_begin`), so no warp is given over to it.
constexpr int SK_STEP = 512, SK_M = (SB + SK_STEP - 1) / SK_STEP, SK_CHUNKS = S / 16;
static_assert(SB % 16 == 0 && S % 16 == 0, "a step's lane loads (uint4) and the shared memory after the stages");
template <int KS>
__host__ __device__ constexpr int sk_words() {
  return (KS / 4 + 5) / 6 == 3 ? 4 : (KS / 4 + 5) / 6;
}

__device__ __forceinline__ u32 sk_sign(u32 sel, int e) {
  return e == 0 ? __byte_perm(0xFF01FF01u, 0xFF01FF01u, sel)
                : e == 1 ? __byte_perm(0xFFFF0101u, 0xFFFF0101u, sel) : __byte_perm(0x01010101u, 0xFFFFFFFFu, sel);
}

// A tile buffer's row r holds its column c at c ^ sk_swz(r % 8): the lanes of a C fragment's red (rows g, columns
// 8 b + 2 tq) then hit 32 distinct banks.
template <int KS>
__host__ __device__ constexpr int sk_swz(int g) {
  return KS >= 32 ? 8 * (g & 3) + (g >> 2) : 8 * ((g >> 1) & 1) + (g >> 2);
}

// The CTA's sketch state: SK_BUFS tile buffers of 16 x k_s int32 (tile tl's is tl % SK_BUFS).
constexpr int SK_NP = TW % 32 == 0 ? SK_CHUNKS : SK_CHUNKS + 1;   // the most steps a warp's units of a task touch
struct Sketch {
  int *buf;
  const u32 *__restrict__ signs;
  int *__restrict__ out;
  u32 *__restrict__ tags;
  u64 jbase, tile0;   // the tag index of vector 0; the CTA's first tile
  int skt, tpc, ntasks;
};

// The MMA warp's hash of a finished tile: SkJob holds lanes 0-15's chaining values (vector `lane` of tile `tile`),
// and sk_hash_block runs the next BLAKE3 block of all 16, so the hash runs in the MMA warp's waits on `decoded`. At
// the last block it writes the tags and zeroes the buffer. sk_start first copies the tile to out (if given).
struct SkJob {
  u32 cv[8];
  int tile, blk;   // blk < 0: no job
};

template <int KS>
__device__ __forceinline__ void sk_start(SkJob &J, int tile, int lane, const Sketch &k) {
  const int *b = k.buf + (tile % SK_BUFS) * 16 * KS;
  if (k.out)
    for (int e = lane; e < 16 * KS; e += 32)
      k.out[(k.tile0 + tile) * 16 * KS + e] = b[e / KS * KS + ((e % KS) ^ sk_swz<KS>((e / KS) & 7))];
#pragma unroll
  for (int i = 0; i < 8; ++i) J.cv[i] = kTagKey[i];
  J.tile = tile;
  J.blk = 0;
}

template <int KS>
__device__ __forceinline__ void sk_hash_block(SkJob &J, int lane, const Sketch &k) {
  constexpr int LEN = 8 + 4 * KS, NMB = (LEN + 63) / 64;
  int *b = k.buf + (J.tile % SK_BUFS) * 16 * KS;
  const int blk = J.blk;
  if (!(WT_SK_DIAG & 1) && k.tags && lane < 16) {
    const u64 j = k.jbase + 16 * (k.tile0 + J.tile) + lane;
    const int *w = b + lane * KS, swz = sk_swz<KS>(lane & 7);
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = 16 * blk + i - 2;
      const u32 x = q >= 0 && q < KS ? (u32)w[q ^ swz] : 0u;
      m[i] = blk == 0 && i == 0 ? bswap32((u32)(j >> 32)) : blk == 0 && i == 1 ? bswap32((u32)j) : x;
    }
    const bool last = blk == NMB - 1;
    b3(J.cv, m, 0, last ? LEN - 64 * blk : 64, B3_KEYED | (blk == 0 ? B3_START : 0) | (last ? B3_END | B3_ROOT : 0),
       J.cv, kOne);
    if (last) {
      uint4 *o = reinterpret_cast<uint4 *>(k.tags + 8 * (16 * (k.tile0 + J.tile) + lane));
      o[0] = make_uint4(J.cv[0], J.cv[1], J.cv[2], J.cv[3]);
      o[1] = make_uint4(J.cv[4], J.cv[5], J.cv[6], J.cv[7]);
    }
  }
  __syncwarp();
  J.blk = blk + 1;
  if (J.blk == NMB) {
    for (int e = lane; e < 16 * KS; e += 32) b[e] = 0;
    __syncwarp();
    J.blk = -1;
  }
}

template <int KS>
__device__ __forceinline__ void sk_finish(SkJob &J, int lane, const Sketch &k) {
  while (J.blk >= 0) sk_hash_block<KS>(J, lane, k);
}

// The warp's units of task x (the lanes with `mine`, consecutive, units ua .. ub - 1) hold chunks [ua, ub) S / 16 of
// x's stage: step p's lane l takes chunk 32 p + l if it is one of them and zero otherwise, so the warps holding a
// step's chunks add up to its sketch. sk_begin finds them and loads their signs (before the stage is ready), and sk_mma
// adds the warp's sums into the tile's buffer (while the warp holds the stage, before its lanes arrive on `decoded`).
// A warp's lanes are in at most two consecutive tasks, so tile tl's sums are all in once the MMA warp has waited on
// `decoded` for task min(skt (tl + 1), ntasks - 1); it starts the tile's hash after that task's arrive on `empty`
// (SkJob). Tile tl + SK_BUFS reuses the buffer, and no warp adds to it before its first task is ready, after the MMA
// warp's arrive on `empty` for task skt (tl + SK_BUFS) - stages: the MMA warp finishes the hash before that arrive,
// which comes after the hash's start while stages < (SK_BUFS - 1) skt.
template <int KS>
struct SkPart {
  u32 sw[SK_NP][sk_words<KS>()];
  int c0, c1, p0, tl;
};

template <int KS>
__device__ __forceinline__ void sk_begin(SkPart<KS> &P, int x, bool mine, int w, int lane, const Sketch &k) {
  constexpr int WP = sk_words<KS>();
  const unsigned bal = __ballot_sync(FULL, mine);
  const int f = __ffs(bal) - 1;
  P.c0 = SK_CHUNKS * __shfl_sync(FULL, w, f);
  P.c1 = SK_CHUNKS * (__shfl_sync(FULL, w, f + __popc(bal) - 1) + 1);
  P.tl = x / k.skt;
  P.p0 = P.c0 / 32;
  const u32 *sg = k.signs + ((size_t)((x % k.skt) * SK_M + P.p0) * LANES + lane) * WP;
#pragma unroll
  for (int h = 0; h < SK_NP; ++h) {
#pragma unroll
    for (int q = 0; q < WP; ++q) P.sw[h][q] = 0u;
    if (32 * (P.p0 + h) < P.c1) {
      const u32 *sp = sg + h * LANES * WP;
      if constexpr (WP == 4) {
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(sp));
        P.sw[h][0] = v.x, P.sw[h][1] = v.y, P.sw[h][2] = v.z, P.sw[h][3] = v.w;
      } else if constexpr (WP == 2) {
        const uint2 v = __ldg(reinterpret_cast<const uint2 *>(sp));
        P.sw[h][0] = v.x, P.sw[h][1] = v.y;
      } else {
        P.sw[h][0] = *sp;   // shared or global memory (fused_kernel's sgs)
      }
    }
  }
}

template <int KS>
__device__ __forceinline__ void sk_mma(const SkPart<KS> &P, const uint8_t *st, int lane, const Sketch &k) {
  constexpr int KB = KS / 8;
  int acc[KB][4];
#pragma unroll
  for (int b = 0; b < KB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0;
#pragma unroll
  for (int h = 0; h < SK_NP; ++h)
    if (32 * (P.p0 + h) < P.c1) {
      const int c = 32 * (P.p0 + h) + lane;
      const uint4 a = c >= P.c0 && c < P.c1 ? *reinterpret_cast<const uint4 *>(st + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int b = 0; b < KB; ++b) {
        const int r0 = 2 * b, r1 = 2 * b + 1;
        const u32 b0 = sk_sign((r0 % 6) / 3 ? P.sw[h][r0 / 6] >> 16 : P.sw[h][r0 / 6], r0 % 3);
        const u32 b1 = sk_sign((r1 % 6) / 3 ? P.sw[h][r1 / 6] >> 16 : P.sw[h][r1 / 6], r1 % 3);
        mma_s8(acc[b], a, b0, b1);
      }
    }
  int *buf = k.buf + (P.tl % SK_BUFS) * 16 * KS;
  const int g = lane >> 2, tq = lane & 3, swz = sk_swz<KS>(g);
#pragma unroll
  for (int b = 0; b < KB; ++b) {
    int *r0 = buf + g * KS, *r1 = r0 + 8 * KS;
    const int c = (8 * b + 2 * tq) ^ swz;
    atomicAdd(r0 + c, acc[b][0]);
    atomicAdd(r0 + (c ^ 1), acc[b][1]);
    atomicAdd(r1 + c, acc[b][2]);
    atomicAdd(r1 + (c ^ 1), acc[b][3]);
  }
}

// ro-sketch-blake3/v0 on SKW sketch warps of their own (after the producer), so the decoders and the MMA warp do none
// of it. Sketch warp q takes tasks ii = q, q + SKW, ... of each of the CTA's tiles: it loads the task's first B
// fragments (k.signs, the rest PF steps ahead of their MMAs), waits on `ready`, reads the stage's stored bytes (a
// step's A fragment a uint4 a lane, k_s / 8 MMAs) and arrives on
// `sketched`, which the decoders wait on before they overwrite the stage in place. Its sums for the tile stay in its
// MMAs' accumulators; at the tile's end the warps add them into the tile's buffer (tl % 2 NBT). At the end of each
// batch of NBT tiles (and of the CTA's last tile) they meet at named barrier 1, and warp q copies out, hashes (a lane
// a vector, from the window's or pass's kTagKey) and zeroes vectors 16 q / SKW .. 16 (q + 1) / SKW - 1 of each of the
// batch's tiles: a hash's instructions issue once for all 32 lanes, so the batch takes a quarter of a tile's issue
// each at SKW = 2. Batch b + 2's adds come after batch b + 1's barrier, which follows batch b's zeroing.
// Every warp runs every one of the CTA's tiles (tpc), its tasks or none, so the tiles past the CTA's tasks hash zeros.
//
// The B operand is SkB's (the interface a bound receipt drops into): SkB{}(m, r, ii, p, lane) is B register r
// (b0 at r = 2 blk, b1 at 2 blk + 1) of step p of the tile's task ii for this lane, from m, the lane's 4 words of the
// pass's fragment array at that step (k.signs, in sign_frags_kernel's order, a uint4 a lane a step). For
// ro-sketch-blake3/v0 the array holds the ±1 bytes and SignB takes them as they are; ro-bound-sketch/v0's would hold
// the 0x00 / 0xFF mask M in the same order, and its SkB would XOR the lane's bytes of expand(z) into it.
struct SignB {
  __device__ __forceinline__ u32 operator()(const uint4 &m, int r, int, int, int) const {
    return r == 0 ? m.x : r == 1 ? m.y : r == 2 ? m.z : m.w;
  }
};

// sketching.sign_fragments' words (k_s = 16, a word a lane a step) as the sketch warps' B fragments: word e's 4 B
// registers (the prmt each register's selector bits pick) at frags[e], so a sketch warp's step is one 16-byte load a
// lane and no sign arithmetic.
__global__ void sign_frags_kernel(const u32 *__restrict__ words, uint4 *__restrict__ frags, int n) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  const u32 sw = words[e];
  frags[e] = make_uint4(sk_sign(sw, 0), sk_sign(sw, 1), sk_sign(sw, 2), sk_sign(sw >> 16, 0));
}

template <int KS, int SKW, class SkB>
__device__ __forceinline__ void sketch_warp(const uint8_t *smem, u64 *ready, u64 *sketched, int stages, int q,
                                            int lane, const Sketch &k) {
  static_assert(sk_words<KS>() == 1 && 16 % SKW == 0, "a sign word a lane a step; whole vectors a warp");
  constexpr int KB = KS / 8, VW = 16 / SKW, NBT = sk_batch<SKW>(), LEN = 8 + 4 * KS;
  constexpr int PF = SK_M < 4 ? SK_M : 4;   // the steps whose fragments are in flight (L2 reads: 64 KiB at T = 8)
  const int g = lane >> 2, tq = lane & 3, swz = sk_swz<KS>(g);
#if !WT_RECEIPT
  constexpr int NMB = (LEN + 63) / 64;
  const u32 one = kOne;
#endif
  const SkB bop{};
  for (int tl = 0; tl < k.tpc; ++tl) {
    int acc[KB][4];
#pragma unroll
    for (int b = 0; b < KB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0;
    for (int ii = q; ii < k.skt; ii += SKW) {
      const int i = tl * k.skt + ii;
      if (i >= k.ntasks) break;
      const int s = i % stages;
      const uint4 *fg = reinterpret_cast<const uint4 *>(k.signs) + (size_t)ii * SK_M * LANES + lane;
      uint4 mq[PF];
#pragma unroll
      for (int p = 0; p < PF; ++p) mq[p] = __ldg(fg + p * LANES);
      mbar_wait(ready + s, (i / stages) & 1);
      const uint4 *st = reinterpret_cast<const uint4 *>(smem + (size_t)s * SB) + lane;
#pragma unroll
      for (int p = 0; p < SK_M; ++p) {
        const uint4 m = mq[p % PF];
        if (p + PF < SK_M) mq[p % PF] = __ldg(fg + (p + PF) * LANES);
        const uint4 a = SB % SK_STEP == 0 || SK_STEP * p + 16 * lane < SB ? st[p * (SK_STEP / 16)]
                                                                          : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int b = 0; b < KB; ++b) mma_s8(acc[b], a, bop(m, 2 * b, ii, p, lane), bop(m, 2 * b + 1, ii, p, lane));
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(sketched + s);
    }
    int *buf = k.buf + (tl % (2 * NBT)) * 16 * KS;
#pragma unroll
    for (int b = 0; b < KB; ++b) {
      int *r0 = buf + g * KS, *r1 = r0 + 8 * KS;
      const int c = (8 * b + 2 * tq) ^ swz;
      atomicAdd(r0 + c, acc[b][0]);
      atomicAdd(r0 + (c ^ 1), acc[b][1]);
      atomicAdd(r1 + c, acc[b][2]);
      atomicAdd(r1 + (c ^ 1), acc[b][3]);
    }
    if (tl % NBT != NBT - 1 && tl != k.tpc - 1) continue;
    asm volatile("bar.sync 1, %0;\n" ::"r"(32 * SKW) : "memory");
    const int tb = tl - tl % NBT, nt = tl - tb + 1;   // the batch's tiles, whose buffers follow one another
    int *bb = k.buf + (tb % (2 * NBT)) * 16 * KS;
    if (k.out)
      for (int e = lane; e < nt * VW * KS; e += 32) {
        const int t = e / (VW * KS), row = VW * q + e / KS % VW, col = e % KS;
        k.out[(16 * (k.tile0 + tb + t) + row) * KS + col] = bb[(16 * t + row) * KS + (col ^ sk_swz<KS>(row & 7))];
      }
    if (k.tags && lane < nt * VW) {
      const int t = lane / VW, r = VW * q + lane % VW, sz = sk_swz<KS>(r & 7);
      const int *w = bb + (16 * t + r) * KS;
      const u64 vi = 16 * (k.tile0 + tb + t) + r, j = k.jbase + vi;
#if WT_RECEIPT
      static_assert(KS == 16, "sketch-sha512's σ is 16 words");
      u64 h[8], m[16];
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = kRcptMid[i];
      m[0] = j;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        m[1 + i] = (u64)bswap32((u32)w[(2 * i) ^ sz]) << 32 | bswap32((u32)w[(2 * i + 1) ^ sz]);
      m[9] = 0x80ULL << 56;
#pragma unroll
      for (int i = 10; i < 15; ++i) m[i] = 0;
      m[15] = (128 + LEN) * 8;
      sha512_compress(h, m);
      // Lanes 2a and 2a + 1 hold vectors vi and vi + 1 and write their 128 bytes as four whole 32-byte sectors, each
      // half from one lane: a lane's own four 16-byte stores would each fill half a sector.
      // Sector s of the pair is words 4 (s % 2) … of vector vi + s / 2; the even lane writes its first 16 bytes and the
      // odd lane its last. Each lane keeps its own words 0, 1, 4, 5 (even) or 2, 3, 6, 7 (odd) and takes the other's.
      static_assert(VW % 2 == 0, "a lane pair's vectors are adjacent");
      const int odd = lane & 1, nl = nt * VW;
      const u32 act = nl >= 32 ? 0xffffffffu : (1u << nl) - 1;
      u64 mine[4], theirs[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int w0 = 4 * (i >> 1) + (i & 1);
        mine[i] = odd ? h[w0 + 2] : h[w0];
        theirs[i] = __shfl_xor_sync(act, odd ? h[w0] : h[w0 + 2], 1);
      }
      uint4 *o = reinterpret_cast<uint4 *>(k.tags + 16 * (vi & ~(u64)1));
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const bool own = (s < 2) != (odd == 1);
        const u64 a = own ? mine[2 * (s & 1)] : theirs[2 * (s & 1)];
        const u64 b = own ? mine[2 * (s & 1) + 1] : theirs[2 * (s & 1) + 1];
        o[2 * s + odd] = make_uint4((u32)a, (u32)(a >> 32), (u32)b, (u32)(b >> 32));
      }
#else
      u32 cv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) cv[i] = kTagKey[i];
#pragma unroll
      for (int blk = 0; blk < NMB; ++blk) {
        u32 m[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int x = 16 * blk + i - 2;
          const u32 v = x >= 0 && x < KS ? (u32)w[x ^ sz] : 0u;
          m[i] = blk == 0 && i == 0 ? bswap32((u32)(j >> 32)) : blk == 0 && i == 1 ? bswap32((u32)j) : v;
        }
        const bool last = blk == NMB - 1;
        b3(cv, m, 0, last ? LEN - 64 * blk : 64,
           B3_KEYED | (blk == 0 ? B3_START : 0) | (last ? B3_END | B3_ROOT : 0), cv, one);
      }
      uint4 *o = reinterpret_cast<uint4 *>(k.tags + 8 * vi);
      o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
      o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
#endif
    }
    __syncwarp();
    for (int e = lane; e < nt * VW * KS; e += 32) bb[(16 * (e / (VW * KS)) + VW * q) * KS + e % (VW * KS)] = 0;
    __syncwarp();
  }
}

// The decoder warps' loop (fused_kernel's comment). A lane keeps w = -(c_0 + t) in place of its tweak t and steps it
// down (decode_word_wk). A payload's generic stores reach the stage before the next bulk copy into it (async proxy) only
// behind a proxy fence. At full slot a payload is its own word's bytes, so a lane writes it in place without `loaded`.
// With SKK > 0 the warp sketches its units of lo and hi (sk_begin, sk_mma) before it loads its words: every read of
// the stage's stored bytes is then before the lane's arrive on `loaded` (the warp's, behind __syncwarp), and its
// sums are in before its lanes arrive on `decoded`. With SKW > 0 the sketch warps sketch instead (sketch_warp), and a
// lane writes its payload over the stored bytes only once its task's are `sketched`.
template <int K, int SKK, int SKW, bool FRESH>
__device__ __forceinline__ void decode_tasks(uint8_t *smem, u64 *ready, u64 *loaded, u64 *decoded, u64 *sketched,
                                             int stages, int ntasks, u64 q0, u64 gq, u64 base, int decoders, int warp,
                                             int lane, const u32 *__restrict__ tsteps, const Sketch &sk) {
  const int adv = 32 * decoders / TW, rem = 32 * decoders % TW;
  int i = (32 * warp + lane) / TW, w = (32 * warp + lane) % TW;
  u32 t[NL];
  if constexpr (!FRESH) {
    u32 wn[NL];
    tweak(base + (q0 + (u64)i * gq) * TW + w, t);
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
    constexpr bool SPLIT = TW % 32 != 0;   // else a warp's lanes are in one task
    SkPart<(SKK > 0 ? SKK : 16)> P0, P1;
    if constexpr (SKK > 0 && SKW == 0 && !(WT_SK_DIAG & 4)) {
      sk_begin(P0, lo, on && i == lo, w, lane, sk);
      if (SPLIT && hi != lo) sk_begin(P1, hi, on && i == hi, w, lane, sk);
    }
    mbar_wait(ready + lo % stages, (lo / stages) & 1);
    if (hi != lo) mbar_wait(ready + hi % stages, (hi / stages) & 1);
    if constexpr (SKK > 0 && SKW == 0 && !(WT_SK_DIAG & 2)) {
      sk_mma(P0, smem + (lo % stages) * SB, lane, sk);
      if (SPLIT && hi != lo) sk_mma(P1, smem + (hi % stages) * SB, lane, sk);
      __syncwarp();
    }
    u32 yw[NL];
    yw[0] = 0;
    if (on) {
      load_word_smem(st + w * S, yw);
      if constexpr (PAY_E > 0) mbar_arrive(loaded + s);
      if constexpr (FRESH) decode_word_freshk<K>(yw, base + (q0 + (u64)i * gq) * TW + w);
      else decode_word_wk<K>(yw, t);
    }
    if constexpr (PAY_E == 0) {
      if (on) {
        if constexpr (SKW > 0) mbar_wait(sketched + s, (i / stages) & 1);
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
      if (on) {
        if constexpr (SKW > 0) mbar_wait(sketched + s, (i / stages) & 1);
        put_payload_w(st, w, lane, yw, nxt);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        mbar_arrive(decoded + s);
      }
    }
    const bool wrap = w + rem >= TW;
    if constexpr (!FRESH) {
      u32 step[NL], t2[NL];
#pragma unroll
      for (int k = 0; k < NL; ++k) step[k] = __ldg(tsteps + (wrap ? NL : 0) + k);
      submod_cc(t, step, t2);
      copy(t, t2);
    }
    w += wrap ? rem - TW : rem;
    i += wrap ? adv + 1 : adv;
  }
}

// tsteps: tau (32 D mod TW + A G TW) and tau (32 D mod TW - TW + (A + 1) G TW) mod p, A = floor(32 D / TW), G the
// store's tasks between the CTA's consecutive ones: the tweak's step to a lane's next unit, its task advancing A or
// A + 1.
// With SKK > 0 the decoders sketch (sk_begin) and the MMA warp hashes each tile (SkJob) into sk_out and tags
// (ro-sketch-blake3/v0, index tag_base + v), and there are no tag warps; with SKW > 0 as well, SKW sketch warps after
// the producer do all of it (sketch_warp).
// WARPS, the warps the launch bounds allow, is fused_warps' or fewer (wt_fused_sketch).
template <int K, int NB8, bool PARTIALS, int SKK, int SKW, int WARPS = fused_warps<NB8, SKK, SKW>()>
__global__ void __launch_bounds__(WARPS * 32, 1) fused_kernel(
    const uint8_t *__restrict__ wstore, const uint8_t *__restrict__ x, float *__restrict__ y, int M, int N, int Kd,
    u64 base, int groups, int decoders, int stages, const u32 *__restrict__ tsteps, u32 *__restrict__ tags,
    u64 tag_base, const u32 *__restrict__ signs, int *__restrict__ sk_out, int skt, int sgs) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * SB);
  u64 *full = reinterpret_cast<u64 *>(smem + stages * SB + NB8 * 8 * XS_STRIDE);
  u64 *loaded = full + stages, *decoded = loaded + stages, *empty = decoded + stages, *ready = empty + stages;
  u64 *sketched = ready + stages;
  u32 *tagq = reinterpret_cast<u32 *>(full + 2 * ((6 * stages + 1) / 2));
  int *skbuf = reinterpret_cast<int *>(tagq + 4);   // 16-byte aligned
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  constexpr int MW = fused_mma_warps<K, NB8, SKK, SKW>();
  load_xs<NB8>(xs, x, M, Kd, kc);
  for (int e = threadIdx.x; e < sk_bufs<SKW>() * 16 * SKK; e += blockDim.x) skbuf[e] = 0;
  u32 *sksg = reinterpret_cast<u32 *>(skbuf + sk_bufs<SKW>() * 16 * SKK);
  if (sgs && signs)
    for (int e = threadIdx.x; e < sgs; e += blockDim.x) sksg[e] = __ldg(signs + e);
  const u32 *sgn = sgs ? sksg : signs;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      mbar_init(full + s, 1);
      mbar_init(loaded + s, TW);
      mbar_init(decoded + s, TW);
      mbar_init(empty + s, LANES);
      mbar_init(ready + s, 1);
      mbar_init(sketched + s, 1);
    }
    tagq[0] = tagq[1] = tagq[2] = tagq[3] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntasks = nbs > g0 ? (nbs - g0 + groups - 1) / groups : 0;
  const u64 q0 = (u64)g0 * nchunks + kc, gq = (u64)groups * nchunks;   // task i is the store's task q0 + i gq
  [[maybe_unused]] const int tpc = ((nbs + groups - 1) / groups + skt - 1) / skt;   // the sketch's tiles a CTA

  if (warp == decoders + MW) {
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
  } else if (warp >= decoders && warp < decoders + MW) {
    const int g = lane >> 2, tq = lane & 3;
    [[maybe_unused]] const Sketch sk{skbuf, sgn, sk_out, tags, tag_base, (u64)blockIdx.x * tpc, skt, tpc, ntasks};
    [[maybe_unused]] SkJob J;
    J.blk = -1;
    int hashed = 0;   // the tiles whose hash has started
    for (int i = 0; i < ntasks; ++i) {
      const int s = i % stages;
      if constexpr (SKK > 0 && SKW == 0)
        while (J.blk >= 0 && !mbar_test(decoded + s, (i / stages) & 1)) sk_hash_block<SKK>(J, lane, sk);
      // Every MMA warp waits out every task's phase in turn, so none waits on a stage's parity a phase early.
      mbar_wait(decoded + s, (i / stages) & 1);
      if (MW > 1 && i % MW != warp - decoders) continue;
      float acc[NB8][4];
#pragma unroll
      for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
      const uint4 *sp = reinterpret_cast<const uint4 *>(smem + s * SB) + lane;
      constexpr int WH = NB8 == 4 ? TASK_TILES / 2 : TASK_TILES;   // the A fragments held at once (fused_warps)
#pragma unroll
      for (int h = 0; h < TASK_TILES; h += WH) {
        uint4 wv[WH];
#pragma unroll
        for (int tt = 0; tt < WH; ++tt) wv[tt] = sp[(h + tt) * LANES];
#ifdef WT_DEBUG_DUMP
        if (g_dbg) {
          uint4 *o = reinterpret_cast<uint4 *>(g_dbg + (q0 + (u64)i * gq) * TASK_BYTES) + lane;
#pragma unroll
          for (int tt = 0; tt < WH; ++tt) o[(h + tt) * LANES] = wv[tt];
        }
#endif
        if (h + WH == TASK_TILES) {
          // The stage's next task may be in the tile whose buffer the job still hashes.
          if constexpr (SKK > 0 && SKW == 0)
            if (J.blk >= 0 && skt * (J.tile + SK_BUFS) <= i + stages) sk_finish<SKK>(J, lane, sk);
          // `empty` hands the stage to the next bulk copy (the async proxy) once its last fragments are loaded, ahead
          // of their MMAs, as gemm_ws_kernel's: only a proxy fence orders this warp's shared loads before it, as ptxas
          // issues the arrive straight after the last load. The decoders' and the sketch warps' reads precede the
          // decoders' stores, which the decoders fence.
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          mbar_arrive(empty + s);
        }
#pragma unroll
        for (int tt = 0; tt < WH; ++tt) {
          const u32 a[4] = {wv[tt].x, wv[tt].y, wv[tt].z, wv[tt].w};
          mma_tile<NB8>(acc, a, xs, h + tt, g, tq);
        }
      }
      store_task<NB8, PARTIALS>(acc, y, M, N, kc, g0 + i * groups, g, tq);
      if constexpr (SKK > 0 && SKW == 0)
        while (hashed * skt < ntasks && i == min(skt * (hashed + 1), ntasks - 1)) {
          sk_finish<SKK>(J, lane, sk);
          sk_start<SKK>(J, hashed++, lane, sk);
        }
    }
    if constexpr (SKK > 0 && SKW == 0) {
      sk_finish<SKK>(J, lane, sk);
      for (; hashed < tpc; ++hashed) {   // the CTA's tiles past its tasks: zero sketches
        sk_start<SKK>(J, hashed, lane, sk);
        sk_finish<SKK>(J, lane, sk);
      }
    }
  } else if (warp < decoders) {
    const Sketch sk{skbuf, sgn, sk_out, tags, tag_base, (u64)blockIdx.x * tpc, skt, tpc, ntasks};
    decode_tasks<K, SKK, SKW, fresh_decode<K, WARPS>()>(smem, ready, loaded, decoded, sketched, stages, ntasks, q0,
                                                        gq, base, decoders, warp, lane, tsteps, sk);
  } else if constexpr (SKW > 0) {
    if (warp < decoders + 1 + MW + SKW) {
      const Sketch sk{skbuf, sgn, sk_out, tags, tag_base, (u64)blockIdx.x * tpc, skt, tpc, ntasks};
      sketch_warp<SKK, SKW, SignB>(smem, ready, sketched, stages, warp - decoders - 1 - MW, lane, sk);
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

#define WT_K_SWITCH(k, F) switch (k) { case 1: F(1); break; case 2: F(2); break; case 3: F(3); break; default: return 1; }

extern "C" {

int wt_constants(int *out) {
  out[0] = NBITS; out[1] = NL; out[2] = S; out[3] = PB; out[4] = TW; out[5] = U; out[6] = (int)CF; out[7] = WPB;
  out[8] = TC_OK ? 2 * (U + 1) : 0; out[9] = TC_WORD_BYTES; out[10] = MAXK; out[11] = SB; out[12] = FUSED_WARPS;
  out[13] = FUSED_SPAN; out[14] = TAG_BLOCK; out[15] = WT_SK_DIAG; out[16] = fused_warps<1, 0, 0>();
  out[17] = WT_RECEIPT; out[18] = WT_MASK; out[19] = WT_FOLD6;
  return 0;
}

// K, NL limbs in host memory (WT_MASK=1).
int wt_set_mask(const u32 *mask) {
  if (!MASKED) return 1;
  if (cudaMemcpyToSymbol(kMask, mask, NL * 4)) return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

// c (k elements of NL limbs), tau and p in host memory.
int wt_set_key(const u32 *c, int k, const u32 *tau, const u32 *p) {
  if (k < 1 || k > MAXK) return 1;
  u32 cc[MAXK][NL] = {}, dd[MAXK][NL] = {};
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < NL; ++i) cc[j][i] = c[j * NL + i];
  for (int j = 1; j < k; ++j) {
    u64 br = 0, cy = 0;
    for (int i = 0; i < NL; ++i) {
      const u64 v = (u64)cc[0][i] - cc[j][i] - br;
      dd[j][i] = (u32)v;
      br = (v >> 32) & 1;
    }
    for (int i = 0; br && i < NL; ++i) {
      cy += (u64)dd[j][i] + p[i];
      dd[j][i] = (u32)cy;
      cy >>= 32;
    }
  }
  if (cudaMemcpyToSymbol(kC, cc, sizeof cc) || cudaMemcpyToSymbol(kD, dd, sizeof dd) ||
      cudaMemcpyToSymbol(kTau, tau, NL * 4) || cudaMemcpyToSymbol(kP, p, NL * 4))
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

int wt_prim(int op, const u32 *a, const u32 *b, u32 *out, int n, cudaStream_t s) {
  if (op < 0 || op > 6 || (op >= 5 && !FULL_SLOT) || n < 1) return 1;
  prim_kernel<<<(n + WPB - 1) / WPB, WPB, 0, s>>>(op, a, b, out, n);
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
    if (MASKED && k > 1) return 5;   // tc_decode has no σ_K
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
// squarings (ilp words a warp); 4 b3_chain_kernel; 5 pipe_kernel; 6: as 0 with sqr_fold, a word a thread (full slot).
// steps a thread (or warp).
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
  } else if (what == 6) {
    if (!FULL_SLOT || ilp != 1) return 1;
    chain_kernel<3, 1, 1><<<grid, 128, 0, s>>>(steps, xr);
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

// The fused decode GEMM at k rounds: `decoders` decoder warps, the MMA warps (fused_mma_warps), a producer and `taggers`
// tag warps (block jl's ro-blake3/v1 tag, at index tag_base + jl, into tags[8 jl ..]), `stages` stored stages; tsteps
// as fused_kernel's.
// wt_fused_sketch with sketch_k > 0 (16 at any k, 64 at k = 1; no tag warps) computes ro-sketch-blake3/v0 at
// k_s = sketch_k over tiles of skt tasks (stages < 2 skt), the signs as sketching.sign_fragments, each vector v's tag at
// tag_base + v into tags[8 v ..] and, if sk_out, its sketch at sk_out[k_s v ..]; tags and sk_out hold
// sketch_vectors(...) vectors. sketch_warps = 0: the decoders sketch and the MMA warp hashes; 2 (k_s = 16, full slot):
// two sketch warps do both (sketch_warp), beside `decoders` decoders, and signs holds the words' B fragments
// (wt_sign_frags), 4 u32 a word. The warps a launch may take are fused_warps'.
int wt_fused_sketch(int k, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, u64 base, int partials,
                    int decoders, int stages, int groups, int taggers, const u32 *tsteps, u32 *tags, u64 tag_base,
                    int sketch_k, const u32 *signs, int *sk_out, int skt, int sketch_warps, int *launched,
                    cudaStream_t s) {
  const int nb8 = M <= 8 ? 1 : (M <= 16 ? 2 : 4);
  const int cap = sketch_warps ? fused_cap<16, 2>(nb8) : (sketch_k ? fused_cap<16, 0>(nb8) : fused_cap<0, 0>(nb8));
  const int mw =
      sketch_warps ? fused_mma<16, 2>(k, nb8) : (sketch_k ? fused_mma<16, 0>(k, nb8) : fused_mma<0, 0>(k, nb8));
  const int warps = decoders + mw + 1 + taggers + sketch_warps;
  if (Kd % TASK_COLS || N % 16 || M < 1 || M > 32 || k < 1 || k > MAXK || decoders < FUSED_SPAN || taggers < 0 ||
      warps > cap || stages < (32 * decoders + TW - 1) / TW || (w && taggers && !tags))
    return 1;
  if (sketch_k && ((sketch_k != 16 && sketch_k != 64) || (sketch_k == 64 && k != 1) || taggers ||
                   stages >= (SK_BUFS - 1) * skt || (w && (!signs || !(tags || sk_out)))))
    return 1;
  if (sketch_warps && (sketch_warps != 2 || sketch_k != 16 || PAY_E != 0)) return 1;
  if (WT_RECEIPT && sketch_k && !sketch_warps) return 1;   // only the sketch warps hash sketch-sha512
  const int nchunks = Kd / TASK_COLS, nbs = N / 16, threads = warps * 32;
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const int signs16 = sketch_k == 16 ? skt * ((SB + 511) / 512) * 32 : 0;
  auto go = [&](auto kern, int smem, int sgs) -> int {
    return launch_gemm(kern, smem, threads, groups, nchunks, nbs, w == nullptr, launched, [&](int gr) {
      kern<<<gr * nchunks, threads, smem, s>>>(static_cast<const uint8_t *>(w), x, y, M, N, Kd, base, gr, decoders,
                                               stages, tsteps, tags, tag_base, signs, sk_out, skt, sgs);
    });
  };
  // A launch of at most 12 warps at k >= 2 takes the 12-warp kernel (fresh_decode), where fused_warps allows more.
#define G(KK, NB, SK, SW)                                                                                            \
  do {                                                                                                               \
    const int sg = SW == 0 && fused_smem_bytes<NB, SK, SW>(stages, signs16) <= optin ? signs16 : 0;                  \
    const int sm = fused_smem_bytes<NB, SK, SW>(stages, sg);                                                         \
    constexpr int W12 = FRESH_DECODE && KK >= 2 ? 12 : fused_warps<NB, SK, SW>();                                    \
    if (warps <= W12)                                                                                                \
      return partials ? go(fused_kernel<KK, NB, true, SK, SW, W12>, sm, sg)                                          \
                      : go(fused_kernel<KK, NB, false, SK, SW, W12>, sm, sg);                                        \
    return partials ? go(fused_kernel<KK, NB, true, SK, SW>, sm, sg) : go(fused_kernel<KK, NB, false, SK, SW>, sm, sg); \
  } while (0)
#define GN(KK, SK, SW) \
  do { if (nb8 == 1) G(KK, 1, SK, SW); else if (nb8 == 2) G(KK, 2, SK, SW); else G(KK, 4, SK, SW); } while (0)
  if (sketch_warps && k == 1) GN(1, 16, 2);
  if (sketch_warps && k == 2) GN(2, 16, 2);
  if (sketch_warps) GN(3, 16, 2);
  if (sketch_k == 16 && k == 1) GN(1, 16, 0);
  if (sketch_k == 16 && k == 2) GN(2, 16, 0);
  if (sketch_k == 16) GN(3, 16, 0);
  if (sketch_k == 64) GN(1, 64, 0);
  if (k == 1) GN(1, 0, 0);
  if (k == 2) GN(2, 0, 0);
  GN(3, 0, 0);
#undef GN
#undef G
}

int wt_fused(int k, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, u64 base, int partials,
             int decoders, int stages, int groups, int taggers, const u32 *tsteps, u32 *tags, u64 tag_base,
             int *launched, cudaStream_t s) {
  return wt_fused_sketch(k, w, x, y, M, N, Kd, base, partials, decoders, stages, groups, taggers, tsteps, tags, tag_base,
                         0, nullptr, nullptr, 1, 0, launched, s);
}

// The vectors a sketched launch tags: 16 a tile, ceil(ceil(nbs / groups) / skt) tiles a CTA.
u64 wt_sketch_vectors(int N, int Kd, int groups, int skt) {
  const u64 nbs = N / 16, per = (nbs + groups - 1) / groups;
  return (u64)groups * (Kd / TASK_COLS) * ((per + skt - 1) / skt) * 16;
}

int wt_set_dbg(uint8_t *p) {
#ifdef WT_DEBUG_DUMP
  return cudaMemcpyToSymbol(g_dbg, &p, sizeof p) ? 2 : 0;
#else
  return p ? 5 : 0;
#endif
}

// One window's tag (ro-blake3/v1 or ro-sketch-blake3/v0): K as BLAKE3's 8 little-endian key words, and the leaves'
// prefix block Z(LEAF_DOMAIN ‖ frame(name) ‖ frame(σ)) (128 bytes), whose midstate the leaves start from. A
// WT_RECEIPT build's sketch warps start from the same midstate: there leaf_prefix is sketch_receipt.prefix(seed).
int wt_set_tag(const u32 *key, const uint8_t *leaf_prefix) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int b = 0; b < 8; ++b) v = (v << 8) | leaf_prefix[8 * i + b];
    w[i] = v;
  }
  Mid mid;
  for (int i = 0; i < 8; ++i) mid.h[i] = SHA512_IV_HOST[i];
  sha512_compress(mid.h, w);
  const u32 one = 1;
  if (cudaMemcpyToSymbol(kTagKey, key, 8 * 4) || cudaMemcpyToSymbol(kOne, &one, 4) ||
      cudaMemcpyToSymbol(g_leafmid, &mid, sizeof mid) || cudaMemcpyToSymbol(kRcptMid, mid.h, sizeof mid.h))
    return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

// The fixed bytes of a pass's messages (pass_kernel) at σ (32 bytes), k_s and n.
static int g_pass_n = 0;
int wt_set_pass(const uint8_t *sigma, int ks, int n) {
  if (ks != 16 || n < 8 || n % 8 || KPASS_ROWS * (n / 8) > PASS_BITS_BYTES || (WT_RECEIPT && n != 4096)) return 1;
  PassMsgs P{};
  auto put = [](uint8_t *m, int &o, const void *src, int len) {
    memcpy(m + o, src, len);
    o += len;
  };
  auto frame = [&](uint8_t *m, int &o, char tag, const void *body, int len) {
    m[o++] = (uint8_t)tag;
    for (int b = 7; b >= 0; --b) m[o++] = (uint8_t)((u64)len >> (8 * b));
    put(m, o, body, len);
  };
  auto frame_int = [&](uint8_t *m, int &o, u64 v) {
    uint8_t be[8];
    int len = 0;
    for (int b = 7; b >= 0; --b)
      if (len || (v >> (8 * b)) & 0xFF || b == 0) be[len++] = (uint8_t)(v >> (8 * b));
    frame(m, o, 'i', be, len);
  };
  const uint8_t zero[4 + 64] = {};
  const char *name = WT_RECEIPT ? "sketch-sha512" : "ro-sketch-blake3/v0";
  const char *sign_domain = WT_RECEIPT ? "verity/remote-bandwidth/sketch-signs/v0" : "verity/bandwidth/sketch-signs/v0";
  const int nl = (int)strlen(name);
  int o = 0;   // SHA512(PASS_DOMAIN ‖ σ ‖ u32be j ‖ root)
  put(P.seed, o, "verity/remote-bandwidth/seed/v0", 32);
  put(P.seed, o, sigma, 32);
  put(P.seed, o, zero, 4 + 64);
  P.seed_len = o;
  o = 0;   // derive(s, SIGN_DOMAIN, {"k": ks, "n": n, "tag": name})
  put(P.derive, o, "verity/randomness/derive/v2", 28);
  P.derive_s = o + 9;
  frame(P.derive, o, 'b', zero, SEED_BYTES);
  frame(P.derive, o, 's', sign_domain, (int)strlen(sign_domain));
  uint8_t ctx[128];
  int c = 0;
  frame(ctx, c, 's', "k", 1);
  frame_int(ctx, c, (u64)ks);
  frame(ctx, c, 's', "n", 1);
  frame_int(ctx, c, (u64)n);
  frame(ctx, c, 's', "tag", 3);
  frame(ctx, c, 's', name, nl);
  frame(P.derive, o, 'b', ctx, c);
  P.derive_len = o;
  o = 0;   // Z(TAG_DOMAIN ‖ frame(name) ‖ frame(s)): the key
  put(P.tag, o, "verity/bandwidth/tag/v1", 24);
  frame(P.tag, o, 's', name, nl);
  P.tag_s = o + 9;
  frame(P.tag, o, 'b', zero, 32);
  o = 0;   // Z(LEAF_DOMAIN ‖ frame(name) ‖ frame(s)): the leaves' prefix; sketch-sha512's receipt prefix
#if WT_RECEIPT
  put(P.leaf, o, "verity/remote-bandwidth/sketch-receipt/v0", 42);
#else
  put(P.leaf, o, "verity/bandwidth/tag-leaf/v1", 29);
  frame(P.leaf, o, 's', name, nl);
#endif
  P.leaf_s = o + 9;
  frame(P.leaf, o, 'b', zero, SEED_BYTES);
  o = 0;   // a row's stream block: STREAM ‖ K_S ‖ frame(frames(("row", c))) ‖ u64be block
  put(P.row, o, "verity/randomness/stream/v2", 28);
  P.row_key = o;
  put(P.row, o, zero, 32);
  uint8_t idx[32];
  c = 0;
  frame(idx, c, 's', "row", 3);
  frame_int(idx, c, 0);
  frame(P.row, o, 'b', idx, c);
  P.row_c = o - 1;
  P.row_ctr = o;
  put(P.row, o, zero, 8);
  P.row_len = o;
  P.ks = ks;
  P.n = n;
  if (cudaMemcpyToSymbol(kPass, &P, sizeof P)) return 2;
  g_pass_n = n;
  return cudaDeviceSynchronize() ? 3 : 0;
}

// Pass j's receipt state from the previous pass's root (8 SHA-512 state words on the device, as wt_tag_tree's
// nodes): its seed, its signs (sketching.sign_fragments at T tasks a tile, signs[]) and, in stream order, its key
// (kTagKey) and leaves' prefix, for every sketched launch and commit after it on the stream.
int wt_pass(unsigned j, const u64 *root, u32 *signs, int t, cudaStream_t s) {
  if (!root || !signs || t < 1 || 32 * ((SB + 511) / 512) * t != g_pass_n) return 1;
  pass_kernel<<<1, PASS_THREADS, 0, s>>>(j, root, signs, t);
  void *gp = nullptr;
  if (cudaGetLastError() || cudaGetSymbolAddress(&gp, g_pass)) return 4;
  uint8_t *g = static_cast<uint8_t *>(gp);
  if (WT_RECEIPT && cudaMemcpyToSymbolAsync(kRcptMid, g + offsetof(PassState, leafmid), 64, 0,
                                            cudaMemcpyDeviceToDevice, s))
    return 4;
  return cudaMemcpyToSymbolAsync(kTagKey, g + offsetof(PassState, key), 32, 0, cudaMemcpyDeviceToDevice, s) ? 4 : 0;
}

// The last pass's seed (the whole SHA-512, 64 bytes; ro-sketch-blake3/v0's s_j is its first 32), K_S (32), key (32,
// kTagKey's) and leaves' (sketch-sha512: receipts') midstate (8 u64, kRcptMid's) into out (192 bytes).
int wt_pass_state(uint8_t *out) {
  PassState p;
  if (cudaDeviceSynchronize() || cudaMemcpyFromSymbol(&p, g_pass, sizeof p) ||
      cudaMemcpyFromSymbol(out + 96, kTagKey, 32) || cudaMemcpyFromSymbol(out + 128, kRcptMid, 64))
    return 2;
  memcpy(out, p.seed, 64);
  memcpy(out + 64, p.ks, 32);
  if (!WT_RECEIPT) memcpy(out + 128, p.leafmid, 64);
  return 0;
}

// The sketch warps' B fragments (frags, 4 u32 a word) from the k_s = 16 sign words of t tasks a tile (signs).
int wt_sign_frags(const u32 *signs, u32 *frags, int t, cudaStream_t s) {
  if (!signs || !frags || t < 1) return 1;
  const int n = t * SK_M * LANES;
  sign_frags_kernel<<<(n + 255) / 256, 256, 0, s>>>(signs, reinterpret_cast<uint4 *>(frags), n);
  return cudaGetLastError() ? 4 : 0;
}

// leaves (ceil(nblk / 16) digests of 8 u64) from the tags of blocks [0, nblk).
int wt_tag_leaves(const u32 *tags, u64 nblk, u64 *leaves, cudaStream_t s) {
  const u64 n = (nblk + 15) / 16;
  tag_leaves_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(tags, nblk, leaves);
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
