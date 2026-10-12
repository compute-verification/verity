// The wide word codecs on the RTX PRO 6000 (sm_120): `wide.py` is the reference. One build per field (WW_BITS = 127,
// 255 or 521); WW_SQR picks the squaring (0: operand scanning on 64-bit products, 1: PTX carry chains, lo and hi passes).
//
// A field element is N little-endian u32 limbs. p = 2^M − C (C = 1 for the Mersenne fields, 19 for p255). Lazy values:
// `sqr` and `mul` take inputs below 2^M + 2^8 (Mersenne) or 2^256 (p255) and return values in the same range; `canon`
// takes such a value to [0, p).
//
// The store E holds word i's limb e at u32 index 32·N·⌊i/32⌋ + 32·e + (i mod 32): a warp's load of one limb of 32
// consecutive words reads 128 contiguous bytes, and a task's record (L words a lane) is the 32·L·N u32 from 32·L·N·q.
//
// Kernels (extern "C" launchers at the end):
//   encode / decode   a flat store of payloads / words, word i at index base + i: decode mode 0 XORs the payloads,
//                     mode 2 writes them
//   chain_*           compute only, U independent words a thread, each step's input the previous step's output, the
//                     index (and so the tweak) advancing by the grid's stride: square (one squaring a step), encode,
//                     decode, and the tag (one BLAKE3 compression a step)
//   gemm              y[b][n] = Σ_k x[b][k] W[n][k], FP8 E4M3 mma.m16n8k32, W the A operand decoded from the store in the
//                     fragment layout (`wide.fragment_payloads`), one task (16 rows × 512 columns) a warp; TAG also hashes
//                     each lane's encoded words (its block) with ro-blake3/v1 into `tags`
//   commit            the tags' leaves (16 a leaf) and their RFC 6962 tree: benchmarks/bandwidth/tagger.cu's kernels
#include <cstdint>
#include <cuda_runtime.h>

#include "tagger.cu"

#ifndef WW_BITS
#define WW_BITS 521
#endif
#ifndef WW_SQR
#define WW_SQR 0
#endif
#ifndef WW_KMAX
#define WW_KMAX 4
#endif

namespace ww {

template <int BITS>
struct Fd;
template <>
struct Fd<127> {
  static constexpr int N = 4, M = 127, PB = 126;
  static constexpr uint32_t C = 1;
};
template <>
struct Fd<255> {
  static constexpr int N = 8, M = 255, PB = 254;
  static constexpr uint32_t C = 19;
};
template <>
struct Fd<521> {
  static constexpr int N = 17, M = 521, PB = 520;
  static constexpr uint32_t C = 1;
};

using F = Fd<WW_BITS>;
constexpr int N = F::N, M = F::M, PB = F::PB;
constexpr uint32_t C = F::C;
constexpr bool MERS = C == 1;
constexpr int MB = M / 32, MS = M % 32;  // bit M is bit MS of limb MB
constexpr int KMAX = WW_KMAX;
constexpr int LANES = 32, TASK_TILES = 16, LANE_BYTES = 16 * TASK_TILES, TASK_COLS = 32 * TASK_TILES;
constexpr int L = (8 * LANE_BYTES + PB - 1) / PB;  // words a lane a task
constexpr int GEMM_WARPS = 4;
constexpr int XS_STRIDE = TASK_COLS + 16;
constexpr int MSG = 2 + L * N;                     // a block's message in u32: u64be j, then the words
constexpr int MSG_STRIDE = MSG | 1;
constexpr int MSG_BLOCKS = (4 * MSG + 63) / 64;
static_assert(32 * N > M, "a spare bit above p");

__host__ __device__ constexpr uint32_t plimb(int e) {
  return e < MB ? (e == 0 ? 0xFFFFFFFFu - (C - 1) : 0xFFFFFFFFu) : e == MB ? (MS ? (1u << MS) - 1 : 0) : 0;
}

__constant__ uint32_t kC[KMAX][N];  // c_j
__constant__ uint32_t kTau[N];      // τ
__constant__ uint32_t kI[N];        // p255: √−1, c1, c1·√−1
__constant__ uint32_t kC1[N];
__constant__ uint32_t kC1I[N];

// --- limbs ------------------------------------------------------------------------------------------------------------

template <int K>
__device__ __forceinline__ uint32_t add_k(uint32_t (&r)[K], const uint32_t (&a)[K], const uint32_t (&b)[K]) {
  uint64_t c = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) {
    c += (uint64_t)a[e] + b[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
  return (uint32_t)c;
}

template <int K>
__device__ __forceinline__ uint32_t sub_k(uint32_t (&r)[K], const uint32_t (&a)[K], const uint32_t (&b)[K]) {
  int64_t c = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) {
    c += (int64_t)a[e] - b[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
  return (uint32_t)(c & 1);  // borrow
}

template <int K>
__device__ __forceinline__ void copy_k(uint32_t (&r)[K], const uint32_t (&a)[K]) {
#pragma unroll
  for (int e = 0; e < K; ++e) r[e] = a[e];
}

template <int K>
__device__ __forceinline__ void select_k(uint32_t (&r)[K], bool s, const uint32_t (&a)[K], const uint32_t (&b)[K]) {
#pragma unroll
  for (int e = 0; e < K; ++e) r[e] = s ? a[e] : b[e];
}

template <int K>
__device__ __forceinline__ bool eq_k(const uint32_t (&a)[K], const uint32_t (&b)[K]) {
  uint32_t d = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) d |= a[e] ^ b[e];
  return d == 0;
}

__device__ __forceinline__ void pconst(uint32_t (&p)[N]) {
#pragma unroll
  for (int e = 0; e < N; ++e) p[e] = plimb(e);
}

// --- products ---------------------------------------------------------------------------------------------------------

#if WW_SQR == 1
__device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t madclo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t madchi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t addc0(uint32_t a) {
  uint32_t r;
  asm volatile("addc.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
  return r;
}
__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
#endif

// t = a·b, 2N limbs.
__device__ __forceinline__ void mul_full(uint32_t (&t)[2 * N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
#pragma unroll
  for (int i = 0; i < 2 * N; ++i) t[i] = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    uint32_t c = 0;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      uint64_t s = (uint64_t)a[i] * b[j] + t[i + j] + c;
      t[i + j] = (uint32_t)s;
      c = (uint32_t)(s >> 32);
    }
    t[i + N] = c;
  }
}

// t = a², 2N limbs: the products a_i a_j (i < j) once, doubled, then the squares.
__device__ __forceinline__ void sqr_full(uint32_t (&t)[2 * N], const uint32_t (&a)[N]) {
#pragma unroll
  for (int i = 0; i < 2 * N; ++i) t[i] = 0;
#if WW_SQR == 1
#pragma unroll
  for (int i = 0; i < N - 1; ++i) {
    // lo pass: t[i + j] += lo(a_i a_j), j = i + 1 … N − 1, carry into t[i + N]
    t[2 * i + 1] = madlo_cc(a[i], a[i + 1], t[2 * i + 1]);
#pragma unroll
    for (int j = i + 2; j < N; ++j) t[i + j] = madclo_cc(a[i], a[j], t[i + j]);
    t[i + N] = addc0(t[i + N]);
    // hi pass: t[i + j + 1] += hi(a_i a_j), carry into t[i + N + 1]
    t[2 * i + 2] = madhi_cc(a[i], a[i + 1], t[2 * i + 2]);
#pragma unroll
    for (int j = i + 2; j < N; ++j) t[i + j + 1] = madchi_cc(a[i], a[j], t[i + j + 1]);
    if (i + N + 1 < 2 * N) t[i + N + 1] = addc0(t[i + N + 1]);
  }
#else
#pragma unroll
  for (int i = 0; i < N - 1; ++i) {
    uint32_t c = 0;
#pragma unroll
    for (int j = i + 1; j < N; ++j) {
      uint64_t s = (uint64_t)a[i] * a[j] + t[i + j] + c;
      t[i + j] = (uint32_t)s;
      c = (uint32_t)(s >> 32);
    }
    t[i + N] = c;
  }
#endif
  t[2 * N - 1] = t[2 * N - 2] >> 31;
#pragma unroll
  for (int i = 2 * N - 2; i > 0; --i) t[i] = __funnelshift_l(t[i - 1], t[i], 1);
  t[0] <<= 1;
#if WW_SQR == 1
  uint32_t lo = (uint32_t)((uint64_t)a[0] * a[0]);
  t[0] = add_cc(t[0], lo);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const uint32_t hi = __umulhi(a[i], a[i]);
    if (i > 0) t[2 * i] = addc_cc(t[2 * i], a[i] * a[i]);
    t[2 * i + 1] = addc_cc(t[2 * i + 1], hi);
  }
#else
  uint32_t c = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    uint64_t s = (uint64_t)a[i] * a[i] + t[2 * i] + c;
    t[2 * i] = (uint32_t)s;
    uint64_t s2 = (uint64_t)t[2 * i + 1] + (s >> 32);
    t[2 * i + 1] = (uint32_t)s2;
    c = (uint32_t)(s2 >> 32);
  }
#endif
}

// --- reduction mod p ----------------------------------------------------------------------------------------------------

// r ≡ t (mod p), t of T limbs below 2^(64·N), r lazy.
template <int T>
__device__ __forceinline__ void reduce(uint32_t (&r)[N], const uint32_t (&t)[T]) {
  if constexpr (MERS) {
    // t = lo + 2^M hi ≡ lo + hi; hi below 2^(32T − M) ≤ 2^(32N + 32)
    uint32_t lo[N], hi[N];
#pragma unroll
    for (int e = 0; e < N; ++e) {
      lo[e] = e < MB ? t[e] : e == MB ? t[e] & ((1u << MS) - 1) : 0;
      const uint32_t h0 = MB + e < T ? t[MB + e] : 0, h1 = MB + e + 1 < T ? t[MB + e + 1] : 0;
      hi[e] = MS ? __funnelshift_r(h0, h1, MS) : h0;
    }
    const uint32_t cy = add_k<N>(r, lo, hi);
    // r + 2^(32N) cy, below 2^(M + 3): fold bit M and above once more
    const uint32_t top = (MS ? (r[MB] >> MS) : 0) | (MB + 1 < N ? 0 : 0) | (cy << (32 * N - M));
    uint32_t topl[N];
#pragma unroll
    for (int e = 0; e < N; ++e) topl[e] = e == 0 ? top : 0;
    if (MB + 1 < N) {
      // bits of limbs above MB (P127 has none; P521 has none) are zero here
    }
    r[MB] &= (1u << MS) - 1;
    add_k<N>(r, r, topl);
  } else {
    // p255: 2^256 ≡ 2C = 38. t = lo + 2^256 hi ≡ lo + 38 hi, then the ninth limb once more, then a carry once more.
    static_assert(T <= 2 * N, "p255 reduces at most 2N limbs");
    uint64_t c = 0;
#pragma unroll
    for (int e = 0; e < N; ++e) {
      const uint32_t h = N + e < T ? t[N + e] : 0;
      c += (uint64_t)h * (2 * C) + t[e];
      r[e] = (uint32_t)c;
      c >>= 32;
    }
    uint64_t d = (uint64_t)(uint32_t)c * (2 * C);
#pragma unroll
    for (int e = 0; e < N; ++e) {
      d += r[e];
      r[e] = (uint32_t)d;
      d >>= 32;
    }
    r[0] += (uint32_t)d * (2 * C);  // a carry leaves r[0] below 38·39, so this one does not carry
  }
}

__device__ __forceinline__ void sqr(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  uint32_t t[2 * N];
  sqr_full(t, a);
  reduce<2 * N>(r, t);
}

__device__ __forceinline__ void mul(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  uint32_t t[2 * N];
  mul_full(t, a, b);
  reduce<2 * N>(r, t);
}

// [0, p) from a lazy value.
__device__ __forceinline__ void canon(uint32_t (&r)[N]) {
  uint32_t p[N], s[N];
  pconst(p);
  if constexpr (MERS) {
    // r < 2^M + 2^8 < 2p: r ≥ p iff r + 1 has bit M
    uint32_t one[N];
#pragma unroll
    for (int e = 0; e < N; ++e) one[e] = e == 0;
    add_k<N>(s, r, one);
    const bool ge = (s[MB] >> MS) & 1;
    s[MB] &= (1u << MS) - 1;
    select_k<N>(r, ge, s, r);
  } else {
    // r < 2^256 < 3p: subtract p while r ≥ p, twice
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const uint32_t borrow = sub_k<N>(s, r, p);
      select_k<N>(r, !borrow, s, r);
    }
  }
}

__device__ __forceinline__ void addmod(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  // a, b canonical: a + b < 2p fits N limbs
  add_k<N>(r, a, b);
  uint32_t p[N], s[N];
  pconst(p);
  const uint32_t borrow = sub_k<N>(s, r, p);
  select_k<N>(r, !borrow, s, r);
}

__device__ __forceinline__ void submod(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  // a, b canonical
  uint32_t p[N], s[N];
  pconst(p);
  const uint32_t borrow = sub_k<N>(r, a, b);
  add_k<N>(s, r, p);
  select_k<N>(r, borrow, s, r);
}

// r = a·s mod p, a canonical, s < 2^64; canonical.
__device__ __forceinline__ void mul_u64(uint32_t (&r)[N], const uint32_t (&a)[N], uint64_t s) {
  uint32_t t[N + 2];
  const uint32_t s0 = (uint32_t)s, s1 = (uint32_t)(s >> 32);
  uint32_t c = 0;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    uint64_t v = (uint64_t)a[e] * s0 + c;
    t[e] = (uint32_t)v;
    c = (uint32_t)(v >> 32);
  }
  t[N] = c;
  t[N + 1] = 0;
  c = 0;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    uint64_t v = (uint64_t)a[e] * s1 + t[e + 1] + c;
    t[e + 1] = (uint32_t)v;
    c = (uint32_t)(v >> 32);
  }
  t[N + 1] = c;
  reduce<N + 2>(r, t);
  canon(r);
}

// --- ρ and ρ⁻¹ --------------------------------------------------------------------------------------------------------

__device__ __forceinline__ bool is_odd(const uint32_t (&y)[N]) { return y[0] & 1; }

// y ← ρ⁻¹(y) − u, y and u canonical.
__device__ __forceinline__ void dec_round(uint32_t (&y)[N], const uint32_t (&u)[N]) {
  const bool odd = is_odd(y);
  uint32_t v[N];
  sqr(v, y);
  canon(v);
  uint32_t w[N];
  if constexpr (MERS) {
    uint32_t p[N];
    pconst(p);
    sub_k<N>(w, p, v);  // odd y has y² ≠ 0, so p − y² is canonical
  } else {
    uint32_t v2[N];
    add_k<N>(v2, v, v);  // 2y² < 2p < 2^256
    canon(v2);
    copy_k<N>(w, v2);
  }
  select_k<N>(v, odd, w, v);
  submod(y, v, u);
}

template <int K>
__device__ __forceinline__ void decode_word(uint32_t (&y)[N], const uint32_t (&t)[N]) {
#pragma unroll
  for (int j = K - 1; j >= 0; --j) {
    uint32_t u[N];
    addmod(u, kC[j], t);
    dec_round(y, u);
  }
}

// a^(2^n) for lazy a, n squarings.
template <int NSQ>
__device__ __forceinline__ void sqr_n(uint32_t (&a)[N]) {
#pragma unroll 1
  for (int n = 0; n < NSQ; ++n) sqr(a, a);
}

// ρ(a), a canonical.
__device__ __forceinline__ void root(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  uint32_t p[N];
  pconst(p);
  if constexpr (MERS) {
    uint32_t s[N], q[N], ns[N];
    copy_k<N>(s, a);
    sqr_n<M - 2>(s);
    canon(s);
    sqr(q, s);
    canon(q);
    const bool square = eq_k<N>(q, a), odd = is_odd(s);
    sub_k<N>(ns, p, s);
    const bool zero = eq_k<N>(s, ns) || s[0] == 0 && eq_k<N>(ns, p);
    if (zero) copy_k<N>(ns, s);
    select_k<N>(r, odd != square, s, ns);
  } else {
    // β = a^(2^252 − 2) = (a^(2^250 − 1)² · a)², the ed25519 chain
    uint32_t z2[N], z9[N], z11[N], x5[N], x10[N], x20[N], x40[N], x50[N], x100[N], t[N];
    sqr(z2, a);                 // a^2
    sqr(t, z2);
    sqr(t, t);                  // a^8
    mul(z9, t, a);              // a^9
    mul(z11, z9, z2);           // a^11
    sqr(t, z11);                // a^22
    mul(x5, t, z9);             // a^(2^5 − 1)
    copy_k<N>(t, x5);
    sqr_n<5>(t);
    mul(x10, t, x5);            // 2^10 − 1
    copy_k<N>(t, x10);
    sqr_n<10>(t);
    mul(x20, t, x10);           // 2^20 − 1
    copy_k<N>(t, x20);
    sqr_n<20>(t);
    mul(x40, t, x20);           // 2^40 − 1
    copy_k<N>(t, x40);
    sqr_n<10>(t);
    mul(x50, t, x10);           // 2^50 − 1
    copy_k<N>(t, x50);
    sqr_n<50>(t);
    mul(x100, t, x50);          // 2^100 − 1
    copy_k<N>(t, x100);
    sqr_n<100>(t);
    mul(t, t, x100);            // 2^200 − 1
    sqr_n<50>(t);
    mul(t, t, x50);             // 2^250 − 1
    sqr(t, t);
    mul(t, t, a);               // 2^251 − 1
    sqr(t, t);                  // β = a^(2^252 − 2)
    uint32_t q[N], na[N], ia[N], nia[N], sel[N], s[N], ns[N];
    sqr(q, t);
    canon(q);
    sub_k<N>(na, p, a);
    mul(ia, kI, a);
    canon(ia);
    sub_k<N>(nia, p, ia);
    const bool e_a = eq_k<N>(q, a), e_na = eq_k<N>(q, na), e_ia = eq_k<N>(q, ia);
    uint32_t one[N];
#pragma unroll
    for (int e = 0; e < N; ++e) one[e] = e == 0;
    select_k<N>(sel, e_ia, kC1, kC1I);
    select_k<N>(sel, e_na, kI, sel);
    select_k<N>(sel, e_a, one, sel);
    canon(t);
    mul(s, t, sel);
    canon(s);
    const bool want_odd = !(e_a || e_na), odd = is_odd(s);
    sub_k<N>(ns, p, s);
    if (s[0] == 0 && eq_k<N>(ns, p)) copy_k<N>(ns, s);
    select_k<N>(r, odd == want_odd, s, ns);
    (void)nia;
  }
}

template <int K>
__device__ __forceinline__ void encode_word(uint32_t (&x)[N], const uint32_t (&t)[N]) {
#pragma unroll 1
  for (int j = 0; j < K; ++j) {
    uint32_t u[N], a[N];
    addmod(u, kC[j], t);
    addmod(a, x, u);
    root(x, a);
  }
}

// --- the store ----------------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint64_t limb_index(uint64_t i, int e) { return 32ull * N * (i >> 5) + 32ull * e + (i & 31); }

__device__ __forceinline__ void load_word(uint32_t (&y)[N], const uint32_t* __restrict__ s, uint64_t i) {
#pragma unroll
  for (int e = 0; e < N; ++e) y[e] = __ldcs(s + limb_index(i, e));
}

__device__ __forceinline__ void store_word(uint32_t* __restrict__ s, uint64_t i, const uint32_t (&y)[N]) {
#pragma unroll
  for (int e = 0; e < N; ++e) s[limb_index(i, e)] = y[e];
}

template <int K, int MODE>
__global__ void __launch_bounds__(256) decode_kernel(const uint32_t* __restrict__ store, uint32_t* __restrict__ out,
                                                     uint64_t n, uint64_t base, unsigned long long* xr) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t t[N], ts[N];
  mul_u64(t, kTau, base + i);
  mul_u64(ts, kTau, stride);
  uint64_t acc = 0;
  for (; i < n; i += stride) {
    uint32_t y[N];
    load_word(y, store, i);
    decode_word<K>(y, t);
    if (MODE == 2) store_word(out, i, y);
#pragma unroll
    for (int e = 0; e < N; ++e) acc ^= (uint64_t)y[e] << (e & 1) * 32;
    addmod(t, t, ts);
  }
  atomicXor(xr, (unsigned long long)acc);
}

template <int K>
__global__ void __launch_bounds__(256) encode_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint64_t n,
                                                     uint64_t base) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t t[N], ts[N];
  mul_u64(t, kTau, base + i);
  mul_u64(ts, kTau, stride);
  for (; i < n; i += stride) {
    uint32_t x[N];
#pragma unroll
    for (int e = 0; e < N; ++e) x[e] = in[limb_index(i, e)];
    encode_word<K>(x, t);
    store_word(out, i, x);
    addmod(t, t, ts);
  }
}

// --- compute only -------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void seed_word(uint32_t (&y)[N], uint64_t id) {
  uint64_t h = id * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    y[e] = (uint32_t)(h >> 32);
  }
  y[N - 1] &= 0x3FFFFFFFu >> (32 * N - M > 2 ? 32 * N - M - 2 : 0);
  canon(y);
}

template <int U>
__global__ void __launch_bounds__(256) chain_square(uint64_t steps, unsigned long long* xr) {
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y[U][N];
#pragma unroll
  for (int u = 0; u < U; ++u) seed_word(y[u], id * U + u);
#pragma unroll 1
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) sqr(y[u], y[u]);
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int e = 0; e < N; ++e) acc ^= y[u][e];
  atomicXor(xr, (unsigned long long)acc);
}

template <int K, int U>
__global__ void __launch_bounds__(256) chain_encode(uint64_t steps, unsigned long long* xr) {
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x * U;
  uint32_t y[U][N], t[U][N], ts[N];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    seed_word(y[u], id * U + u);
    mul_u64(t[u], kTau, id * U + u);
  }
  mul_u64(ts, kTau, stride);
#pragma unroll 1
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      encode_word<K>(y[u], t[u]);
      addmod(t[u], t[u], ts);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int e = 0; e < N; ++e) acc ^= y[u][e];
  atomicXor(xr, (unsigned long long)acc);
}

template <int K, int U>
__global__ void __launch_bounds__(256) chain_decode(uint64_t steps, unsigned long long* xr) {
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x * U;
  uint32_t y[U][N], t[U][N], ts[N];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    seed_word(y[u], id * U + u);
    mul_u64(t[u], kTau, id * U + u);
  }
  mul_u64(ts, kTau, stride);
#pragma unroll 1
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      decode_word<K>(y[u], t[u]);
      addmod(t[u], t[u], ts);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int e = 0; e < N; ++e) acc ^= y[u][e];
  atomicXor(xr, (unsigned long long)acc);
}

// One BLAKE3 compression a step for each of U chains, the output fed back as the next message's first half.
template <int U>
__global__ void __launch_bounds__(256) chain_tag(uint64_t steps, bw::Key3 key, unsigned long long* xr) {
  const uint64_t id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cv[U][8];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[u][i] = key.k[i] ^ (uint32_t)(id * U + u) * 0x9E3779B9u;
#pragma unroll 1
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = cv[u][i & 7] + i;
      bw::b3_compress(key.k, m, (uint32_t)s, 64, bw::B3_KEYED, cv[u]);
    }
  }
  uint64_t acc = 0;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc ^= (uint64_t)cv[u][i] << (i & 1) * 32;
  atomicXor(xr, (unsigned long long)acc);
}

// --- the fused GEMM -----------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Stream bits bit … bit + 31 of the lane's payloads (word w carries bits w·PB … w·PB + PB − 1); `bit` is a constant.
__device__ __forceinline__ uint32_t take4(const uint32_t (&x)[L][N], int bit) {
  const int w = bit / PB, sh = bit % PB, e = sh / 32, s = sh % 32;
  uint32_t v = x[w][e] >> s;
  if (s && e + 1 < N) v |= x[w][e + 1] << (32 - s);
  if (sh + 32 > PB && w + 1 < L) v |= x[w + 1][0] << (PB - sh);
  return v;
}

// One task (16 rows × 512 columns) a warp, four warps a CTA sharing x's chunk. TAG: lane l's block (its L encoded
// words) is hashed from shared memory before it is decoded, and its tag written to tags[8 (j − jbase) …].
template <int KR, int NB8, bool PARTIALS, bool TAG>
__global__ void __launch_bounds__(GEMM_WARPS * 32) gemm_kernel(const uint32_t* __restrict__ E, const uint8_t* __restrict__ x,
                                                               float* __restrict__ y, int Mb, int Nn, int Kd,
                                                               uint64_t wbase, uint64_t jbase, bw::Key3 key,
                                                               uint32_t* __restrict__ tags) {
  __shared__ __align__(16) uint8_t xs[NB8 * 8][XS_STRIDE];
  __shared__ uint32_t msg[TAG ? GEMM_WARPS * 32 * MSG_STRIDE : 1];
  const int nchunks = Kd / TASK_COLS;
  const int kc = blockIdx.x % nchunks;
  const int nb = (blockIdx.x / nchunks) * GEMM_WARPS + threadIdx.x / 32;
  const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const uint64_t q = (uint64_t)nb * nchunks + kc;
  for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += GEMM_WARPS * 32) {
    int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < Mb) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
  }
  __syncthreads();
  if (nb * 16 >= Nn) return;
  uint32_t w[L][N];
  const uint32_t* rec = E + q * (32ull * L * N) + lane;
#pragma unroll
  for (int i = 0; i < L; ++i)
#pragma unroll
    for (int e = 0; e < N; ++e) w[i][e] = __ldcs(rec + 32 * (i * N + e));

  if constexpr (TAG) {
    uint32_t* my = msg + (threadIdx.x) * MSG_STRIDE;
    const uint64_t j = jbase + q * 32 + lane;
    my[0] = bw::bswap32((uint32_t)(j >> 32));
    my[1] = bw::bswap32((uint32_t)j);
#pragma unroll
    for (int i = 0; i < L; ++i)
#pragma unroll
      for (int e = 0; e < N; ++e) my[2 + i * N + e] = w[i][e];
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
#pragma unroll 1
    for (int b = 0; b < MSG_BLOCKS; ++b) {
      uint32_t m[16];
      const int len = b + 1 < MSG_BLOCKS ? 64 : 4 * MSG - 64 * (MSG_BLOCKS - 1);
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = 16 * b + i < MSG ? my[16 * b + i] : 0;
      const uint32_t flags = bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) |
                             (b + 1 == MSG_BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
      bw::b3_compress(cv, m, 0, len, flags, cv);
    }
    uint4* o = reinterpret_cast<uint4*>(tags + 8 * (q * 32 + lane));
    o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
    o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
  }

  if constexpr (KR > 0) {
    uint32_t t[N], t32[N];
    mul_u64(t, kTau, wbase + q * (32ull * L) + lane);
    mul_u64(t32, kTau, LANES);
#pragma unroll
    for (int i = 0; i < L; ++i) {
      decode_word<KR>(w[i], t);
      addmod(t, t, t32);
    }
  }
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
  for (int tt = 0; tt < TASK_TILES; ++tt) {
    const uint32_t a[4] = {take4(w, 128 * tt), take4(w, 128 * tt + 32), take4(w, 128 * tt + 64), take4(w, 128 * tt + 96)};
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
      mma_e4m3(acc[j], a, b0, b1);
    }
  }
#pragma unroll
  for (int j = 0; j < NB8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
      if (b < Mb) {
        if constexpr (PARTIALS) y[((size_t)kc * Mb + b) * Nn + n] = acc[j][e];
        else atomicAdd(y + (size_t)b * Nn + n, acc[j][e]);
      }
    }
  }
}

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

}  // namespace ww

using namespace ww;

static_assert(WW_KMAX >= 1 && WW_KMAX <= 8, "WW_KMAX");
#define WW_CASE(n, F) \
  case n:             \
    F(n);             \
    break;
#if WW_KMAX >= 4
#define WW_CASES(F) WW_CASE(1, F) WW_CASE(2, F) WW_CASE(3, F) WW_CASE(4, F)
#elif WW_KMAX >= 2
#define WW_CASES(F) WW_CASE(1, F) WW_CASE(2, F)
#else
#define WW_CASES(F) WW_CASE(1, F)
#endif
#define WW_K_SWITCH(k, F) \
  switch (k) {            \
    WW_CASES(F)           \
    default:              \
      return 1;           \
  }

static bw::Key3 key_of(const uint8_t* seed) { return bw::b3_key(seed); }

extern "C" {

int ww_constants(int* out) {
  out[0] = WW_BITS; out[1] = N; out[2] = PB; out[3] = L; out[4] = LANES; out[5] = TASK_TILES; out[6] = GEMM_WARPS;
  out[7] = KMAX; out[8] = WW_SQR; out[9] = MSG_BLOCKS;
  return 0;
}

// c: k × N limbs; tau, i_, c1, c1i: N limbs each (the last three p255's, ignored otherwise).
int ww_set_key(const uint32_t* c, int k, const uint32_t* tau, const uint32_t* i_, const uint32_t* c1, const uint32_t* c1i) {
  if (k < 1 || k > KMAX) return 1;
  uint32_t cc[KMAX][N] = {};
  for (int j = 0; j < k; ++j)
    for (int e = 0; e < N; ++e) cc[j][e] = c[j * N + e];
  if (cudaMemcpyToSymbol(kC, cc, sizeof cc) || cudaMemcpyToSymbol(kTau, tau, 4 * N)) return 2;
  if (i_ && (cudaMemcpyToSymbol(kI, i_, 4 * N) || cudaMemcpyToSymbol(kC1, c1, 4 * N) || cudaMemcpyToSymbol(kC1I, c1i, 4 * N)))
    return 2;
  return cudaDeviceSynchronize() ? 3 : 0;
}

int ww_decode(const uint32_t* store, uint32_t* out, uint64_t n, uint64_t base, int k, unsigned long long* xr, int grid,
              cudaStream_t s) {
#define F(K)                                                                    \
  do {                                                                          \
    if (out) decode_kernel<K, 2><<<grid, 256, 0, s>>>(store, out, n, base, xr); \
    else decode_kernel<K, 0><<<grid, 256, 0, s>>>(store, out, n, base, xr);     \
  } while (0)
  WW_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

int ww_encode(const uint32_t* in, uint32_t* out, uint64_t n, uint64_t base, int k, int grid, cudaStream_t s) {
#define F(K) encode_kernel<K><<<grid, 256, 0, s>>>(in, out, n, base)
  WW_K_SWITCH(k, F)
#undef F
  return cudaGetLastError() ? 4 : 0;
}

// what: 0 decode, 1 encode (k rounds a step), 2 the squaring (one a step), 3 the tag (one BLAKE3 compression a step);
// u: independent chains a thread (1 or 2; 1, 2 or 4 for the squaring and the tag).
int ww_chain(int what, int k, int u, uint64_t steps, unsigned long long* xr, int grid, cudaStream_t s) {
  if (what == 0) {
#define F(K) do { if (u == 2) chain_decode<K, 2><<<grid, 256, 0, s>>>(steps, xr); else chain_decode<K, 1><<<grid, 256, 0, s>>>(steps, xr); } while (0)
    WW_K_SWITCH(k, F)
#undef F
  } else if (what == 1) {
#define F(K) do { if (u == 2) chain_encode<K, 2><<<grid, 256, 0, s>>>(steps, xr); else chain_encode<K, 1><<<grid, 256, 0, s>>>(steps, xr); } while (0)
    WW_K_SWITCH(k, F)
#undef F
  } else if (what == 2) {
    if (u == 1) chain_square<1><<<grid, 256, 0, s>>>(steps, xr);
    else if (u == 2) chain_square<2><<<grid, 256, 0, s>>>(steps, xr);
    else if (u == 4) chain_square<4><<<grid, 256, 0, s>>>(steps, xr);
    else return 1;
  } else if (what == 3) {
    uint8_t seed[32] = {1};
    bw::Key3 key = key_of(seed);
    if (u == 1) chain_tag<1><<<grid, 256, 0, s>>>(steps, key, xr);
    else if (u == 2) chain_tag<2><<<grid, 256, 0, s>>>(steps, key, xr);
    else if (u == 4) chain_tag<4><<<grid, 256, 0, s>>>(steps, key, xr);
    else return 1;
  } else {
    return 1;
  }
  return cudaGetLastError() ? 4 : 0;
}

int ww_spin(long long cycles, cudaStream_t s) {
  spin_kernel<<<1, 1, 0, s>>>(cycles);
  return cudaGetLastError() ? 4 : 0;
}

// The fused GEMM: k rounds (0: the store holds payloads, no decode), tag_seed (32 bytes) or null for no tag; tags is
// the commit workspace's tag area (`ww_workspace_tags`).
int ww_gemm(int k, const uint32_t* E, const uint8_t* x, float* y, int Mb, int Nn, int Kd, uint64_t wbase, int partials,
            const uint8_t* tag_seed, uint32_t* tags, cudaStream_t s) {
  if (Mb < 1 || Mb > 32 || Nn % (16 * GEMM_WARPS) || Kd % TASK_COLS || k < 0 || k > KMAX) return 1;
  if (tag_seed && !tags) return 1;
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  const int grid = (Nn / 16 / GEMM_WARPS) * (Kd / TASK_COLS);
  bw::Key3 key = {};
  if (tag_seed) key = key_of(tag_seed);
#define G(KR, NB, T)                                                                                                 \
  do {                                                                                                               \
    if (partials) gemm_kernel<KR, NB, true, T><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, 0, key, tags); \
    else gemm_kernel<KR, NB, false, T><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, 0, key, tags);        \
  } while (0)
#define GT(KR, NB) do { if (tag_seed) G(KR, NB, true); else G(KR, NB, false); } while (0)
#define GK(KR) do { if (nb8 == 1) GT(KR, 1); else if (nb8 == 2) GT(KR, 2); else GT(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WW_CASES(GK)
    default: return 1;
  }
#undef GK
#undef GT
#undef G
  return cudaGetLastError() ? 4 : 0;
}

size_t ww_workspace_bytes(uint64_t blocks) { return bw::workspace_bytes(bw::TAG_B3, blocks); }

uint32_t* ww_workspace_tags(void* workspace, uint64_t blocks) { return bw::make_pass(bw::TAG_B3, blocks, workspace).tags; }

// The tags' leaves and tree over `blocks` blocks; with root non-null, the root's 64 bytes (and a sync).
int ww_commit(void* workspace, uint64_t blocks, const uint8_t* seed, uint8_t* root, cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  bw::Mid lm = bw::prefix_midstate(bw::LEAF_DOMAIN, bw::TAG_NAMES[bw::TAG_B3], seed);
  const unsigned leaf_grid = (unsigned)(((blocks + 15) / 16 + 127) / 128);
  bw::b3_leaves<<<leaf_grid, 128, 0, s>>>(p.tags, p.blocks, blocks, lm, p.nodes);
  bw::launch_tree(p, s);
  if (cudaGetLastError()) return 4;
  if (root) return bw::read_root(p, root, s);
  return 0;
}

// The commit workspace's leaves (64 bytes each, big-endian u64 words), for the gate.
const uint64_t* ww_workspace_leaves(void* workspace, uint64_t blocks) { return bw::make_pass(bw::TAG_B3, blocks, workspace).nodes; }

}  // extern "C"
