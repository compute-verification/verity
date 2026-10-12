// Sloth over one wide word on the RTX PRO 6000 (sm_120): `wide.py` is the reference. One build per field, WT_FIELD = n
// (255, 511, 521, 767, 1023 or 1279; p = 2^n - c as in `wide.FIELDS`).
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
#else
#error "WT_FIELD is one of wide.FIELDS"
#endif
constexpr int NL = (NBITS + 31) / 32, S = (NBITS + 7) / 8, PB = (NBITS - 1) / 8, U = (S + 31) / 32;
constexpr int Q = NBITS / 32, R = NBITS % 32;
constexpr u32 TOPMASK = (1u << R) - 1;
constexpr bool TC_OK = S == 32 * U;
static_assert(R > 0 && Q == NL - 1, "the top limb holds bits 32Q .. n - 1");
static_assert(CF % 4 == 1 && CF < 4096, "p = 3 mod 4, c small");
static_assert(TW * PB >= 8192 && (TW * S) % 16 == 0, "a task's words");
constexpr int MAXK = 2;
constexpr int TASK_BYTES = 8192;
constexpr int SB = TW * S;                              // a task's stored bytes

__constant__ u32 kC[MAXK][NL];
__constant__ u32 kTau[NL];
__constant__ u32 kP[NL];

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

// x = t mod p, lazily: (t mod 2^n) + c (t >> n), twice. t < 2^(2n + 2) gives x < 2^n + 2^25.
__device__ __forceinline__ void fold(const u32 (&t)[2 * NL], u32 (&x)[NL]) {
  u32 y[NL + 1];
  u64 carry = 0;
#pragma unroll
  for (int k = 0; k <= NL; ++k) {
    const u32 nxt = Q + k + 1 < 2 * NL ? t[(Q + k + 1) % (2 * NL)] : 0u;
    const u32 hi = __funnelshift_r(t[Q + k], nxt, R);
    const u32 lo = k < Q ? t[k] : (k == Q ? (t[Q] & TOPMASK) : 0u);
    const u64 v = (u64)hi * CF + lo + carry;
    y[k] = (u32)v;
    carry = v >> 32;
  }
  u64 c = (u64)__funnelshift_r(y[Q], y[Q + 1], R) * CF;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += k == Q ? (y[Q] & TOPMASK) : y[k];
    x[k] = (u32)c;
    c >>= 32;
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
  const bool ge = (y[Q] >> R) != 0;
  y[Q] &= TOPMASK;
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

// x = a + b mod p, a and b canonical.
__device__ __forceinline__ void addmod(const u32 (&a)[NL], const u32 (&b)[NL], u32 (&x)[NL]) {
  u64 c = 0;
#pragma unroll
  for (int k = 0; k < NL; ++k) {
    c += (u64)a[k] + b[k];
    x[k] = (u32)c;
    c >>= 32;
  }
  canon(x);
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
  for (u64 first = (u64)blockIdx.x * WPB; first < n; first += (u64)gridDim.x * WPB) {
    __syncthreads();
    stage_words(store, first, n, sm);
    __syncthreads();
    const u64 i = first + threadIdx.x;
    if (i < n) {
      u32 y[NL];
      load_word_smem(sm + threadIdx.x * S, y);
      decode_word<K>(y, t);
      if (MODE == 2) {
#pragma unroll
        for (int k = 0; k < NL; ++k) out[i * NL + k] = y[k];
      }
      acc ^= fold64(y);
    }
    u32 t2[NL];
    addmod(t, step, t2);
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
  x[Q] &= TOPMASK >> 1;
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
      decode_word<K>(x[0], t);
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

// --- the fused decode GEMM --------------------------------------------------------------------------------------------
//
// The last warp's lane 0 bulk-copies task i's TW stored words (SB bytes) into stage i mod stages; the D decoder warps
// decode them in place; warp D, the MMA warp, takes the tasks in order and runs the plain kernel's 16 tiles on each.
// The CTA's tasks are one sequence of words (task i's word w is word TW i + w), cut into units of 32: decoder warp d
// takes units d, d + D, ..., lane l a unit's word l. A lane loads its word, decodes it, and writes its payload (stage
// bytes PB w ..) only once every word of the task is loaded, since a payload overwrites the stored bytes of the words
// before it. Barriers a stage: full (the copy), loaded (TW lanes), decoded (TW lanes), empty (the MMA warp's lanes).
//
// A parity wait is right only while its barrier is at most one phase behind, which the host's bounds give: D at least
// the units a task spans (no warp holds two units of one task, which would deadlock on `loaded`), and at least
// ceil(32 D / TW) stages (a decoder's tasks advance by at most that from one unit to its next, whose stage it then
// finds refilled or being refilled). One MMA warp keeps `decoded` and `empty` in task order.
//
// An SM's registers fit FUSED_WARPS warps at the decode's register count (a sub-partition holds 16K: four warps at
// 128 registers, three at 168, two at 255).
constexpr int FUSED_WARPS = NL <= 17 ? 16 : (NL <= 24 ? 12 : 8);
constexpr int FUSED_SPAN = (TW + 30) / 32 + 1;            // the most units a task's words touch

template <int NB8>
__host__ __device__ constexpr int fused_smem_bytes(int stages) {
  return stages * SB + NB8 * 8 * XS_STRIDE + 4 * stages * 8;
}

// Bytes [o, o + PB) of the stage get y's low PB bytes: whole 32-bit words inside, single bytes at the two ends (they
// share a word with the neighbours' payloads).
__device__ __forceinline__ void put_payload(uint8_t *st, int o, const u32 (&y)[NL]) {
  const int sh = 8 * (o & 3), w0 = o >> 2, e = o + PB;
  u32 *sw = reinterpret_cast<u32 *>(st);
#pragma unroll
  for (int k = 0; k <= NL; ++k) {
    const u32 lo = k > 0 ? y[k - 1] : 0u, hi = k < NL ? y[k] : 0u;
    const u32 v = __funnelshift_l(lo, hi, sh);
    const int a = 4 * (w0 + k);
    if (a >= o && a + 4 <= e) {
      sw[w0 + k] = v;
    } else {
#pragma unroll
      for (int b = 0; b < 4; ++b)
        if (a + b >= o && a + b < e) st[a + b] = (uint8_t)(v >> (8 * b));
    }
  }
}

// tsteps: tau (32 D mod TW + A G TW) and tau (32 D mod TW - TW + (A + 1) G TW) mod p, A = floor(32 D / TW), G the
// store's tasks between the CTA's consecutive ones: the tweak's step to a lane's next unit, its task advancing A or
// A + 1.
template <int K, int NB8, bool PARTIALS>
__global__ void __launch_bounds__(FUSED_WARPS * 32, 1) fused_kernel(
    const uint8_t *__restrict__ wstore, const uint8_t *__restrict__ x, float *__restrict__ y, int M, int N, int Kd,
    u64 base, int groups, int decoders, int stages, const u32 *__restrict__ tsteps) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t (*xs)[XS_STRIDE] = reinterpret_cast<uint8_t (*)[XS_STRIDE]>(smem + stages * SB);
  u64 *full = reinterpret_cast<u64 *>(smem + stages * SB + NB8 * 8 * XS_STRIDE);
  u64 *loaded = full + stages, *decoded = loaded + stages, *empty = decoded + stages;
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks, g0 = blockIdx.x / nchunks, nbs = N / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  load_xs<NB8>(xs, x, M, Kd, kc);
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      mbar_init(full + s, 1);
      mbar_init(loaded + s, TW);
      mbar_init(decoded + s, TW);
      mbar_init(empty + s, LANES);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntasks = nbs > g0 ? (nbs - g0 + groups - 1) / groups : 0;
  const u64 q0 = (u64)g0 * nchunks + kc, gq = (u64)groups * nchunks;   // task i is the store's task q0 + i gq

  if (warp == decoders + 1) {
    if (lane == 0) {
      for (int i = 0; i < ntasks; ++i) {
        const int s = i % stages;
        if (i >= stages) mbar_wait(empty + s, (i / stages - 1) & 1);
        mbar_expect_tx(full + s, SB);
        bulk_load(smem + s * SB, wstore + (q0 + i * gq) * SB, SB, full + s);
      }
    }
    return;
  }

  if (warp == decoders) {
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
      mbar_arrive(empty + s);
#pragma unroll
      for (int tt = 0; tt < TASK_TILES; ++tt) {
        const u32 a[4] = {wv[tt].x, wv[tt].y, wv[tt].z, wv[tt].w};
        mma_tile<NB8>(acc, a, xs, tt, g, tq);
      }
      store_task<NB8, PARTIALS>(acc, y, M, N, kc, g0 + i * groups, g, tq);
    }
    return;
  }

  const int adv = 32 * decoders / TW, rem = 32 * decoders % TW;
  int i = (32 * warp + lane) / TW, w = (32 * warp + lane) % TW;
  u32 t[NL];
  tweak(base + (q0 + (u64)i * gq) * TW + w, t);
  for (;;) {
    const int lo = __shfl_sync(~0u, i, 0);
    if (lo >= ntasks) break;
    const int hi = min(__shfl_sync(~0u, i, LANES - 1), ntasks - 1);
    const bool on = i < ntasks;
    const int s = i % stages;
    uint8_t *st = smem + s * SB;
    mbar_wait(full + lo % stages, (lo / stages) & 1);
    if (hi != lo) mbar_wait(full + hi % stages, (hi / stages) & 1);
    u32 yw[NL];
    if (on) {
      load_word_smem(st + w * S, yw);
      mbar_arrive(loaded + s);
      decode_word<K>(yw, t);
    }
    mbar_wait(loaded + lo % stages, (lo / stages) & 1);
    if (hi != lo) mbar_wait(loaded + hi % stages, (hi / stages) & 1);
    if (on) {
      put_payload(st, PB * w, yw);
      mbar_arrive(decoded + s);
    }
    const bool wrap = w + rem >= TW;
    u32 step[NL], t2[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) step[k] = __ldg(tsteps + (wrap ? NL : 0) + k);
    addmod(t, step, t2);
    copy(t, t2);
    w += wrap ? rem - TW : rem;
    i += wrap ? adv + 1 : adv;
  }
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
  out[13] = FUSED_SPAN;
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

// The fused decode GEMM at k rounds: `decoders` decoder warps, an MMA warp and a producer, `stages` stored stages;
// tsteps as fused_kernel's.
int wt_fused(int k, const void *w, const uint8_t *x, float *y, int M, int N, int Kd, u64 base, int partials,
             int decoders, int stages, int groups, const u32 *tsteps, int *launched, cudaStream_t s) {
  if (Kd % TASK_COLS || N % 16 || M < 1 || M > 32 || k < 1 || k > MAXK || decoders < FUSED_SPAN ||
      decoders + 2 > FUSED_WARPS || stages < (32 * decoders + TW - 1) / TW)
    return 1;
  const int nb8 = M <= 8 ? 1 : (M <= 16 ? 2 : 4);
  const int nchunks = Kd / TASK_COLS, nbs = N / 16, threads = (decoders + 2) * 32;
  auto go = [&](auto kern, int smem) -> int {
    return launch_gemm(kern, smem, threads, groups, nchunks, nbs, w == nullptr, launched, [&](int gr) {
      kern<<<gr * nchunks, threads, smem, s>>>(static_cast<const uint8_t *>(w), x, y, M, N, Kd, base, gr, decoders,
                                               stages, tsteps);
    });
  };
#define G(KK, NB) \
  return partials ? go(fused_kernel<KK, NB, true>, fused_smem_bytes<NB>(stages)) \
                  : go(fused_kernel<KK, NB, false>, fused_smem_bytes<NB>(stages))
#define GN(KK) do { if (nb8 == 1) G(KK, 1); else if (nb8 == 2) G(KK, 2); else G(KK, 4); } while (0)
  if (k == 1) GN(1);
  GN(2);
#undef GN
#undef G
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
