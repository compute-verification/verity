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

// x += a (a < 2^32) into NL limbs; returns the carry out of 2^n.
__device__ __forceinline__ u32 add_small(u32 (&x)[NL], u32 a) {
  x[0] = add_cc(x[0], a);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = addc_cc(x[k], 0u);
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

// Column m >= NL of a product, v, folded into the low limbs y as c v, the carry word f running up them.
__device__ __forceinline__ void fold_column(u32 (&y)[NL], int j, u32 v, u32 &f) {
  const u64 s = (u64)y[j] + (u64)v * CF + f;
  y[j] = (u32)s;
  f = (u32)(s >> 32);
}

__device__ __forceinline__ void sqr(const u32 (&a)[NL], u32 (&x)[NL]) {
  if constexpr (STREAM) {
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

__device__ __forceinline__ void copy(u32 (&x)[NL], const u32 (&a)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = a[k];
}

// x mod p for x < 2^n: x - p = x + c - 2^n when that carries out.
__device__ __forceinline__ void canon(u32 (&x)[NL]) {
  u32 z[NL];
  z[0] = add_cc(x[0], CF);
#pragma unroll
  for (int k = 1; k < NL; ++k) z[k] = addc_cc(x[k], 0u);
  const bool ge = addc(0u, 0u) != 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? z[k] : x[k];
}

// x + b mod p, both canonical: s = x + b < 2p, and s - p = s + c - 2^n when s carried out of 2^n or s + c does.
__device__ __forceinline__ void add_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  u32 z[NL];
  x[0] = add_cc(x[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = addc_cc(x[k], b[k]);
  const u32 co = addc(0u, 0u);
  z[0] = add_cc(x[0], CF);
#pragma unroll
  for (int k = 1; k < NL; ++k) z[k] = addc_cc(x[k], 0u);
  const bool ge = (co | addc(0u, 0u)) != 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? z[k] : x[k];
}

// x - b mod p, both canonical: a borrow adds p (limbs 2^32 - c, then all ones), masked by the borrow.
__device__ __forceinline__ void sub_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  x[0] = sub_cc(x[0], b[0]);
#pragma unroll
  for (int k = 1; k < NL; ++k) x[k] = subc_cc(x[k], b[k]);
  const u32 m = subc(0u, 0u);
  x[0] = add_cc(x[0], (0u - CF) & m);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) x[k] = addc_cc(x[k], m);
  x[NL - 1] = addc(x[NL - 1], m);
}

// s = p - s when on, s canonical (0 becomes p).
__device__ __forceinline__ void neg_if(u32 (&s)[NL], bool on) {
  u32 z[NL];
  z[0] = sub_cc(0u - CF, s[0]);
#pragma unroll
  for (int k = 1; k < NL - 1; ++k) z[k] = subc_cc(0xffffffffu, s[k]);
  z[NL - 1] = subc(0xffffffffu, s[NL - 1]);
#pragma unroll
  for (int k = 0; k < NL; ++k) s[k] = on ? z[k] : s[k];
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

__device__ __forceinline__ void decode_word(u32 (&x)[NL], u64 i, int k) {
  u32 t[NL];
  tweak(i, t);
#pragma unroll 1
  for (int j = k - 1; j >= 0; --j) {
    u32 c[NL];
    round_constant(j, c);
    rho_inv(x);
    sub_mod(x, c);
    sub_mod(x, t);
  }
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

// out null: the decoded words XORed into xr (the read-and-decode rate).
__global__ void __launch_bounds__(WPB) decode_kernel(const u32 *__restrict__ in, u32 *__restrict__ out, u64 n, u64 base,
                                                     int k, unsigned long long *xr) {
  u32 acc = 0;
  for (u64 w = (u64)blockIdx.x * WPB + threadIdx.x; w < n; w += (u64)gridDim.x * WPB) {
    u32 x[NL];
    load(in + w * NL, x);
    decode_word(x, base + w, k);
    if (out) {
      store(out + w * NL, x);
    } else {
#pragma unroll
      for (int l = 0; l < NL; ++l) acc ^= x[l];
    }
  }
  if (!out) xor_out(acc, xr);
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
template <int ILP>
__global__ void __launch_bounds__(WPB) chain_kernel(int what, int k, u64 steps, unsigned long long *xr) {
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
  if (what == 0) {
#pragma unroll 1
    for (u64 s = 0; s < steps; ++s) {
#pragma unroll
      for (int w = 0; w < ILP; ++w) sqr(x[w], x[w]);
    }
  } else if (what == 1) {
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
// CTAs fit an SM; past 768 bits (STREAM) the decode's registers allow one.
constexpr int HEADS = 64, KV_HEADS = 8, GROUP = HEADS / KV_HEADS, HEAD_DIM = 128;
constexpr int TL_BYTES = 2 * KV_HEADS * HEAD_DIM * 2;
constexpr int WB = NBITS / 8;
constexpr int TLW = (TL_BYTES + WB - 1) / WB;
constexpr int TLE = TLW * WB;
constexpr int TT = 4;
constexpr int AT = 32 * KV_HEADS;
constexpr int PART = HEADS * (HEAD_DIM + 2);
static_assert(GROUP * TT == 32 && TLE % 16 == 0 && WB % 16 == 0, "a warp's scores, one a lane; 16-byte copies");

template <bool ENC>
constexpr int attn_smem() {
  return 2 * TT * (ENC ? TLE : TL_BYTES) + HEADS * HEAD_DIM * 2;
}
static_assert(attn_smem<true>() + 1024 <= 50 * 1024, "two CTAs an SM");

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
      s[h * TT + t] = qf[h][0] * kk[0] + qf[h][1] * kk[1] + qf[h][2] * kk[2] + qf[h][3] * kk[3];
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
  float gm = fmaxf(sc, __shfl_xor_sync(0xffffffffu, sc, 1));
  gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, 2));
  const float m_new = fmaxf(m_own, gm);
  const float scale = exp2f(m_own - m_new), p = exp2f(sc - m_new);
  float ps = p + __shfl_xor_sync(0xffffffffu, p, 1);
  ps += __shfl_xor_sync(0xffffffffu, ps, 2);
  l_own = l_own * scale + ps;
  m_own = m_new;
#pragma unroll
  for (int h = 0; h < GROUP; ++h) {
    const float sh = __shfl_sync(0xffffffffu, scale, h * TT);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[h][i] *= sh;
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
        for (int i = 0; i < 4; ++i) acc[h][i] += ph * vv[i];
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
__global__ void __launch_bounds__(AT, STREAM ? 1 : 2) attn_kernel(const uint8_t *__restrict__ kv, const uint16_t *__restrict__ q,
                                                     float *__restrict__ part, int ctx, int layers, int layer,
                                                     int per_split, int splits, int k) {
  constexpr int RAW = ENC ? TLE : TL_BYTES;
  extern __shared__ __align__(16) uint8_t sm[];
  uint8_t *raw = sm;
  uint8_t *qs = sm + 2 * TT * RAW;
  const int seq = blockIdx.x / splits, split = blockIdx.x % splits;
  const int t0 = split * per_split, t1 = min(ctx, t0 + per_split);
  const int g = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < HEADS * HEAD_DIM / 8; i += AT)
    reinterpret_cast<uint4 *>(qs)[i] = reinterpret_cast<const uint4 *>(q + (size_t)seq * HEADS * HEAD_DIM)[i];
  float acc[GROUP][4] = {};
  float m_own = -INFINITY, l_own = 0.f;
  const int ntiles = t1 > t0 ? (t1 - t0 + TT - 1) / TT : 0;
  const auto issue = [&](int tile) {
    const int bt = t0 + tile * TT, nt = min(TT, t1 - bt);
    uint8_t *dst = raw + (tile & 1) * TT * RAW;
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
    const int bt = t0 + tile * TT, nt = min(TT, t1 - bt);
    uint8_t *src = raw + (tile & 1) * TT * RAW;
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
    attn_tile<RAW>(src, qs, g, lane, nt, acc, m_own, l_own);
    __syncthreads();
  }
  attn_out(part + ((size_t)seq * splits + split) * PART, g, lane, acc, m_own, l_own);
}

// The warp-specialized encoded form: AT attention threads as in attn_kernel, and DW decoder warps that copy each tile
// into a ring of NBUF buffers PRE tiles ahead (cp.async, each thread its own words' chunks), decode their words in place
// and hand the buffer over (full); the attention warps hand it back (empty). The decode of later tiles runs while the
// attention warps work, and p768's 172 words a tile fill 6 warps.
constexpr int NBUF = 4, PRE = 2;
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

int kv_decode(const void *in, void *out, uint64_t n, uint64_t base, int k, int grid, unsigned long long *xr,
              cudaStream_t s) {
  if (k < 1 || k > kv::KMAX || (!out && !xr)) return -1;
  kv::decode_kernel<<<blocks(n, grid), kv::WPB, 0, s>>>((const uint32_t *)in, (uint32_t *)out, n, base, k, xr);
  return launched();
}

int kv_read(const void *store, uint64_t bytes, unsigned long long *xr, int grid, cudaStream_t s) {
  kv::read_kernel<<<grid, 256, 0, s>>>((const uint4 *)store, bytes / 16, xr);
  return launched();
}

int kv_chain(int what, int k, int ilp, uint64_t steps, int grid, unsigned long long *xr, cudaStream_t s) {
  if (what < 0 || what > 2 || ((what > 0) && (k < 1 || k > kv::KMAX))) return -1;
  if (ilp == 1) kv::chain_kernel<1><<<grid, kv::WPB, 0, s>>>(what, k, steps, xr);
  else if (ilp == 2) kv::chain_kernel<2><<<grid, kv::WPB, 0, s>>>(what, k, steps, xr);
  else return -1;
  return launched();
}

int kv_spin(long long cycles, cudaStream_t s) {
  kv::spin_kernel<<<1, 1, 0, s>>>(cycles);
  return launched();
}

// heads, KV heads, head dim, words a token-layer, its stored bytes, tokens a tile, floats a split's partials.
void kv_attn_shape(int *out) {
  out[0] = kv::HEADS; out[1] = kv::KV_HEADS; out[2] = kv::HEAD_DIM; out[3] = kv::TLW; out[4] = kv::TLE; out[5] = kv::TT;
  out[6] = kv::PART;
}

// One layer's decode attention for `batch` sequences of `ctx` tokens (attn_kernel's layout): kv plain (encoded = 0) or
// its words (encoded = 1, k rounds; encoded = 2 the warp-specialized form, the same output), q bf16 [batch][64][128]
// pre-scaled by log2(e)/sqrt(128), out fp32 [batch][64][128], part at least batch * splits * kv_attn_shape's out[6]
// floats. The tokens are split into at most `splits` ranges of whole tiles.
int kv_attn(const void *kv_, int encoded, const uint16_t *q, float *out, float *part, int batch, int ctx, int layers,
            int layer, int splits, int k, cudaStream_t s) {
  if (batch < 1 || ctx < 1 || splits < 1 || layer < 0 || layer >= layers || encoded < 0 || encoded > 2 ||
      (encoded && (k < 1 || k > kv::KMAX)))
    return -1;
  const int per = (ctx + splits - 1) / splits, per_split = (per + kv::TT - 1) / kv::TT * kv::TT;
  const int sp = (ctx + per_split - 1) / per_split;
  const auto *p = static_cast<const uint8_t *>(kv_);
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
