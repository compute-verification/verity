// The KV cache's codec on the RTX PRO 6000 (sm_120): Sloth over one wide word, `kv.py` the reference. One build per
// field, KV_FIELD = 512 or 768 (p = 2^n - c, n = 32 NL), any k up to KMAX at run time.
//
// A thread a word, NL 32-bit limbs: the square column by column, each unordered pair once and doubled
// (`benchmarks/codec_tensor/wt.cu`'s), then 2^n = c folded twice, a lazy value below 2^n; canonical (< p) only where a
// comparison, the sign rule or an addition reads it. The root is `kv.py`'s chain: x_M = a^(2^M - 1) by M's bits from the
// top, then 8 squarings with a multiply by a at LOW's set bits.
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
#else
#error "KV_FIELD is one of kv.FIELDS"
#endif
constexpr int NL = NBITS / 32;
constexpr int KMAX = 8;
constexpr int LOW_BITS = 8;
constexpr int CHAIN_M = NBITS - 2 - LOW_BITS;
constexpr u32 LOW = (1u << LOW_BITS) - (CF - 1) / 4;
constexpr int bitlen(int x) { return x ? 1 + bitlen(x >> 1) : 0; }
constexpr int MBITS = bitlen(CHAIN_M);
constexpr int WPB = 128;
static_assert(NBITS % 32 == 0 && NL % 4 == 0 && CF % 4 == 1 && (CF - 1) / 4 < (1u << LOW_BITS), "p = 2^n - c");

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

// x = t mod p lazily for t < 2^(2n): lo + c hi, whose carry word h <= c; then h c (< 2^20) into the low limbs, and c once
// more if that carried out (the limbs are then below 2^20). x < 2^n.
__device__ __forceinline__ void fold(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
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

__device__ __forceinline__ void copy(u32 (&x)[NL], const u32 (&a)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = a[k];
}

// x mod p for x < 2^n: x - p = x + c - 2^n when that carries out.
__device__ __forceinline__ void canon(u32 (&x)[NL]) {
  u32 z[NL];
  u64 v = CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += x[k];
    z[k] = (u32)v;
    v >>= 32;
  }
  const bool ge = v != 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) x[k] = ge ? z[k] : x[k];
}

// x + b mod p, both canonical: a carry out of 2^n is c (a + b - 2^n + c < 2^n), then canonical.
__device__ __forceinline__ void add_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  u64 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += (u64)x[k] + b[k];
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
  canon(x);
}

// x - b mod p, both canonical: a borrow adds p, i.e. takes c from the wrapped difference (which is then above c).
__device__ __forceinline__ void sub_mod(u32 (&x)[NL], const u32 (&b)[NL]) {
  u32 br = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    const u64 d = (u64)x[k] - b[k] - br;
    x[k] = (u32)d;
    br = (u32)(d >> 63);
  }
  if (br) {
    u32 bc = CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      const u64 d = (u64)x[k] - bc;
      x[k] = (u32)d;
      bc = (u32)(d >> 63);
    }
  }
}

// p - s for canonical s: ~s - (c - 1).
__device__ __forceinline__ void neg(u32 (&s)[NL]) {
  u32 br = CF - 1;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    const u64 d = (u64)(~s[k]) - br;
    s[k] = (u32)d;
    br = (u32)(d >> 63);
  }
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
  if ((s[0] & 1) != (square ? 0u : 1u)) neg(s);
  copy(x, s);
}

// y odd is nonzero, so its square is too.
__device__ __forceinline__ void rho_inv(u32 (&y)[NL]) {
  const bool odd = y[0] & 1;
  u32 q[NL];
  sqr(y, q);
  canon(q);
  if (odd) neg(q);
  copy(y, q);
}

// t(i) = tau i mod p: tau i < 2^(n + 64), its two top limbs h folded in as c h, canonical.
__device__ __forceinline__ void tweak(u64 i, u32 (&t)[NL]) {
  const u32 ilo = (u32)i, ihi = (u32)(i >> 32);
  u32 r[NL + 2];
  u64 v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += (u64)kTau[k] * ilo;
    r[k] = (u32)v;
    v >>= 32;
  }
  r[NL] = (u32)v;
  v = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    v += (u64)kTau[k] * ihi + r[k + 1];
    r[k + 1] = (u32)v;
    v >>= 32;
  }
  r[NL + 1] = (u32)v;
  v = (u64)r[NL] * CF + r[0];
  t[0] = (u32)v;
  v >>= 32;
  v += (u64)r[NL + 1] * CF + r[1];
  t[1] = (u32)v;
  v >>= 32;
#pragma unroll
  for (int k = 2; k < NL; ++k) {
    v += r[k];
    t[k] = (u32)v;
    v >>= 32;
  }
  if (v) {
    u64 z = CF;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      z += t[k];
      t[k] = (u32)z;
      z >>= 32;
    }
  }
  canon(t);
}

__device__ __forceinline__ void round_constant(int j, u32 (&c)[NL]) {
#pragma unroll
  for (int k = 0; k < NL; ++k) c[k] = kC[j][k];
}

__device__ __forceinline__ void encode_word(u32 (&x)[NL], u64 i, int k) {
  u32 t[NL];
  tweak(i, t);
#pragma unroll 1
  for (int j = 0; j < k; ++j) {
    u32 c[NL];
    round_constant(j, c);
    add_mod(x, c);
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
}
