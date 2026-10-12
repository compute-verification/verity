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

#ifdef WW_HOST
// The field arithmetic and the codec built for the host (`tests/wide_host.sh`): the same source, CUDA's qualifiers
// emptied and its integer intrinsics written out.
#define __device__
#define __host__
#define __forceinline__ inline
#define __constant__ static
static inline uint32_t __funnelshift_l(uint32_t lo, uint32_t hi, uint32_t s) {
  s &= 31;
  return s ? (hi << s) | (lo >> (32 - s)) : hi;
}
static inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t s) {
  s &= 31;
  return s ? (lo >> s) | (hi << (32 - s)) : lo;
}
static inline uint32_t __umulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
#else
#include <cuda_runtime.h>

#include "tagger.cu"
#endif

#ifndef WW_BITS
#define WW_BITS 521
#endif
#ifndef WW_SQR
#define WW_SQR 0
#endif
#ifndef WW_KMAX
#define WW_KMAX 4
#endif
// WW_ROLL = 1: the fused GEMM's code kept small enough for the instruction cache (BLAKE3's rounds a loop, and for p521
// the words decoded in a loop), the tag's compressions before the decode rather than interleaved with it. WW_OCC: the
// CTAs an SM the fused GEMM is compiled for (`__launch_bounds__`; 1 lets the compiler use up to 255 registers).
#ifndef WW_ROLL
#define WW_ROLL 0
#endif
#ifndef WW_OCC
#define WW_OCC 1
#endif
// WW_CC = 1: the limb additions and subtractions (and the Mersenne decode's fold) as PTX carry chains, one IADD3.X a
// limb, rather than 64-bit accumulators (IADD.64 and the MOVs that build their register pairs). Device builds only.
#ifndef WW_CC
#define WW_CC 0
#endif
#if WW_CC && defined(WW_HOST)
#error "WW_CC is PTX: device builds only"
#endif
// WW_B3U = 1: the rolled BLAKE3 (WW_ROLL) two rounds an iteration, so the message permutation's moves run three times a
// compression rather than seven.
#ifndef WW_B3U
#define WW_B3U 0
#endif
// WW_TB = 2: one tag over a lane's records in two adjacent K-chunks of the same rows (`wide.block`'s tb), which one warp
// of the fused GEMM steps through: half the tags to commit, and a compression fewer a record. Rolled builds only.
#ifndef WW_TB
#define WW_TB 1
#endif
#if WW_TB != 1 && (WW_TB != 2 || !WW_ROLL)
#error "WW_TB is 1, or 2 with WW_ROLL"
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
constexpr int MSG = 2 + L * N;                     // a record's part of a message in u32: u64be j, then the words
constexpr int TB = WW_TB;
constexpr int TAG_MSG = 2 + TB * L * N;            // a block's message: u64be j, then its TB records' words
constexpr int TAG_BLOCKS = (4 * TAG_MSG + 63) / 64;
constexpr int SPLIT = MSG / 16;                    // TB = 2: compressions 0 … SPLIT − 1 read only the first record,
constexpr int TAIL = MSG - 16 * SPLIT;             // and compression SPLIT the first record's last TAIL words
static_assert(TAIL > 0, "a record's message ends inside a compression");
static_assert(32 * N > M, "a spare bit above p");
static_assert(C != 1 || (MB == N - 1 && MS > 0), "a Mersenne p's top bit is inside the last limb");

__host__ __device__ constexpr uint32_t plimb(int e) {
  return e < MB ? (e == 0 ? 0xFFFFFFFFu - (C - 1) : 0xFFFFFFFFu) : e == MB ? (MS ? (1u << MS) - 1 : 0) : 0;
}

__constant__ uint32_t kC[KMAX][N];   // c_j
__constant__ uint32_t kNC[KMAX][N];  // p − c_j (Mersenne decode)
__constant__ uint32_t kTau[N];       // τ
__constant__ uint32_t kDTw[N];       // the tweak state's step over LANES words (`tw_init`'s dv for s = LANES)
__constant__ uint32_t kI[N];        // p255: √−1, c1, c1·√−1
__constant__ uint32_t kC1[N];
__constant__ uint32_t kC1I[N];

// --- carry chains (PTX) -------------------------------------------------------------------------------------------------

#if WW_SQR == 1 || WW_CC
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
__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t subc0() {
  uint32_t r;
  asm volatile("subc.u32 %0, 0, 0;" : "=r"(r));
  return r;
}
#endif

// --- limbs ------------------------------------------------------------------------------------------------------------

template <int K>
__device__ __forceinline__ uint32_t add_k(uint32_t (&r)[K], const uint32_t (&a)[K], const uint32_t (&b)[K]) {
#if WW_CC
  r[0] = add_cc(a[0], b[0]);
#pragma unroll
  for (int e = 1; e < K; ++e) r[e] = addc_cc(a[e], b[e]);
  return addc0(0);
#else
  uint64_t c = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) {
    c += (uint64_t)a[e] + b[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
  return (uint32_t)c;
#endif
}

template <int K>
__device__ __forceinline__ uint32_t sub_k(uint32_t (&r)[K], const uint32_t (&a)[K], const uint32_t (&b)[K]) {
#if WW_CC
  r[0] = sub_cc(a[0], b[0]);
#pragma unroll
  for (int e = 1; e < K; ++e) r[e] = subc_cc(a[e], b[e]);
  return subc0() & 1;  // borrow
#else
  int64_t c = 0;
#pragma unroll
  for (int e = 0; e < K; ++e) {
    c += (int64_t)a[e] - b[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
  return (uint32_t)(c & 1);  // borrow
#endif
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

// p − a for canonical a (a = 0 gives p, which every lazy input takes).
__host__ __device__ __forceinline__ void neg_el(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  int64_t c = 0;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    c += (int64_t)plimb(e) - a[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
}

// --- products ---------------------------------------------------------------------------------------------------------

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
    // r + 2^(32N) cy, below 2^(M + 2): fold bit M and above once more
    const uint32_t top = (r[MB] >> MS) | (cy << (32 * N - M));
    uint32_t topl[N];
#pragma unroll
    for (int e = 0; e < N; ++e) topl[e] = e == 0 ? top : 0;
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

// r = a + b folded once mod a Mersenne p: a ≤ p + 1 and b ≤ p give r ≤ p + 1.
__device__ __forceinline__ void add_fold(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  add_k<N>(r, a, b);
#if WW_CC
  uint32_t top[N];
#pragma unroll
  for (int e = 0; e < N; ++e) top[e] = e == 0 ? r[MB] >> MS : 0;
  r[MB] &= (1u << MS) - 1;
  add_k<N>(r, r, top);
#else
  uint64_t c = r[MB] >> MS;
  r[MB] &= (1u << MS) - 1;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    c += r[e];
    r[e] = (uint32_t)c;
    c >>= 32;
  }
#endif
}

// y ← ρ⁻¹(y) − u for a Mersenne p, given nu ≡ −u below 2^(M+1). y² = lo + 2^M·hi ≡ lo + hi with lo, hi < 2^M, and
// −x ≡ x ⊕ p for x < 2^M, so ρ⁻¹(y) − u ≡ (lo ⊕ m) + (hi ⊕ m) + nu, m = p for odd y and 0 for even: a sum below
// 2^(M+2), folded once and made canonical.
__device__ __forceinline__ void dec_round_mers(uint32_t (&y)[N], const uint32_t (&nu)[N]) {
  static_assert(MB == N - 1, "the top limb holds bit M − 1");
  const uint32_t m = 0u - (y[0] & 1);
  uint32_t t[2 * N];
  sqr_full(t, y);
#if WW_CC
  uint32_t a[N], b[N];
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const uint32_t pm = plimb(e) & m;
    a[e] = (e < MB ? t[e] : t[e] & ((1u << MS) - 1)) ^ pm;
    b[e] = __funnelshift_r(t[MB + e], t[MB + e + 1], MS) ^ pm;
  }
  const uint32_t c1 = add_k<N>(a, a, b);
  const uint32_t c2 = add_k<N>(y, a, nu);
  uint32_t top[N];
#pragma unroll
  for (int e = 0; e < N; ++e) top[e] = e == 0 ? (y[MB] >> MS) | ((c1 + c2) << (32 * N - M)) : 0;
  y[MB] &= (1u << MS) - 1;
  add_k<N>(y, y, top);
#else
  uint64_t c = 0;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const uint32_t lo = e < MB ? t[e] : t[e] & ((1u << MS) - 1);
    const uint32_t hi = __funnelshift_r(t[MB + e], t[MB + e + 1], MS);
    const uint32_t pm = plimb(e) & m;
    c += (uint64_t)(lo ^ pm) + (hi ^ pm) + nu[e];
    y[e] = (uint32_t)c;
    c >>= 32;
  }
  uint64_t top = (y[MB] >> MS) | (c << (32 * N - M));
  y[MB] &= (1u << MS) - 1;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    top += y[e];
    y[e] = (uint32_t)top;
    top >>= 32;
  }
#endif
  canon(y);
}

// The decode's tweak over words i, i + s, i + 2s, …: t(i) = τ·i canonical, or for a Mersenne p its negation −t(i)
// (≤ p + 1, lazy), which the decode adds where it would subtract t(i).
__device__ __forceinline__ void tw_init(uint32_t (&v)[N], uint32_t (&dv)[N], uint64_t i, uint64_t s) {
  mul_u64(v, kTau, i);
  mul_u64(dv, kTau, s);
  if constexpr (MERS) {
    neg_el(v, v);
    neg_el(dv, dv);
  }
}

__device__ __forceinline__ void tw_step(uint32_t (&v)[N], const uint32_t (&dv)[N]) {
  if constexpr (MERS) add_fold(v, v, dv);
  else addmod(v, v, dv);
}

// k inverse rounds of word y under the tweak state v (`tw_init`).
template <int K>
__device__ __forceinline__ void decode_tw(uint32_t (&y)[N], const uint32_t (&v)[N]) {
  if constexpr (MERS) {
#pragma unroll
    for (int j = K - 1; j >= 0; --j) {
      uint32_t nu[N];
      add_k<N>(nu, v, kNC[j]);
      dec_round_mers(y, nu);
    }
  } else {
    decode_word<K>(y, v);
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
    select_k<N>(r, odd != square, s, ns);  // a = 0: s = 0 is even and a square, so r = s
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
    uint32_t q[N], na[N], ia[N], sel[N], s[N], ns[N];
    sqr(q, t);
    canon(q);
    sub_k<N>(na, p, a);
    mul(ia, kI, a);
    canon(ia);
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
    select_k<N>(r, odd == want_odd, s, ns);  // a = 0: β = 0 and e_a, so r = s = 0
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

#ifndef WW_HOST
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
  tw_init(t, ts, base + i, stride);
  uint64_t acc = 0;
  for (; i < n; i += stride) {
    uint32_t y[N];
    load_word(y, store, i);
    decode_tw<K>(y, t);
    if (MODE == 2) store_word(out, i, y);
#pragma unroll
    for (int e = 0; e < N; ++e) acc ^= (uint64_t)y[e] << (e & 1) * 32;
    tw_step(t, ts);
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
    tw_init(t[u], ts, id * U + u, stride);
  }
#pragma unroll 1
  for (uint64_t s = 0; s < steps; ++s) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      decode_tw<K>(y[u], t[u]);
      tw_step(t[u], ts);
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

// Word i of block j's message (u64be j ‖ the L words), as BLAKE3 reads it: little-endian u32; i a constant.
__device__ __forceinline__ uint32_t msg_word(const uint32_t (&w)[L][N], uint64_t j, int i) {
  if (i == 0) return bw::bswap32((uint32_t)(j >> 32));
  if (i == 1) return bw::bswap32((uint32_t)j);
  if (i < MSG) return w[(i - 2) / N][(i - 2) % N];
  return 0;
}

// `bw::b3_compress` with its seven rounds a loop, the message permuted in registers after each: a seventh of the code.
__device__ __forceinline__ void b3_compress_rolled(uint32_t (&cv)[8], uint32_t (&m)[16], uint32_t len, uint32_t flags) {
  using bw::rotr32;
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = B3_IV0, v9 = B3_IV1, v10 = B3_IV2, v11 = B3_IV3, v12 = 0, v13 = 0, v14 = len, v15 = flags;
#define B3_ROUND(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) \
  B3_G(v0, v4, v8, v12, a0, a1);                                                        \
  B3_G(v1, v5, v9, v13, a2, a3);                                                        \
  B3_G(v2, v6, v10, v14, a4, a5);                                                       \
  B3_G(v3, v7, v11, v15, a6, a7);                                                       \
  B3_G(v0, v5, v10, v15, a8, a9);                                                       \
  B3_G(v1, v6, v11, v12, a10, a11);                                                     \
  B3_G(v2, v7, v8, v13, a12, a13);                                                      \
  B3_G(v3, v4, v9, v14, a14, a15)
#if WW_B3U
  // two rounds an iteration, the second on σ(m) by renaming, then m ← σ²(m); the seventh round on σ⁶(m) unpermuted
#pragma unroll 1
  for (int r = 0; r < 3; ++r) {
    B3_ROUND(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    B3_ROUND(m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]);
    const uint32_t t[16] = {m[3], m[4], m[10], m[12], m[13], m[2], m[7], m[14], m[6], m[5], m[9], m[0], m[11], m[15], m[8], m[1]};
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
  B3_ROUND(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
#else
#pragma unroll 1
  for (int r = 0; r < 7; ++r) {
    B3_ROUND(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    const uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
#endif
#undef B3_ROUND
  cv[0] = v0 ^ v8, cv[1] = v1 ^ v9, cv[2] = v2 ^ v10, cv[3] = v3 ^ v11;
  cv[4] = v4 ^ v12, cv[5] = v5 ^ v13, cv[6] = v6 ^ v14, cv[7] = v7 ^ v15;
}

// Compression b of ro-blake3/v1 over a block (one chunk: the message is under 1,024 bytes) on its message words m;
// cv starts as the key, and after the last compression it is the tag.
__device__ __forceinline__ void tag_compress(uint32_t (&cv)[8], uint32_t (&m)[16], int b) {
  static_assert(4 * TAG_MSG <= 1024, "a block is one BLAKE3 chunk");
  const uint32_t len = b + 1 < TAG_BLOCKS ? 64 : 4 * TAG_MSG - 64 * (TAG_BLOCKS - 1);
  const uint32_t flags = bw::B3_KEYED | (b == 0 ? bw::B3_CHUNK_START : 0) |
                         (b + 1 == TAG_BLOCKS ? bw::B3_CHUNK_END | bw::B3_ROOT : 0);
  if constexpr (WW_ROLL) b3_compress_rolled(cv, m, len, flags);
  else bw::b3_compress(cv, m, 0, len, flags, cv);
}

// Compression b of block j, b reading only header and the first record (b < SPLIT when TB = 2), from its registers.
__device__ __forceinline__ void tag_step(uint32_t (&cv)[8], const uint32_t (&w)[L][N], uint64_t j, int b) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg_word(w, j, 16 * b + i);
  tag_compress(cv, m, b);
}

// TB = 2: compression b ≥ SPLIT, from the first record's last TAIL words (kept from its task) and the second record.
__device__ __forceinline__ void tag_step_second(uint32_t (&cv)[8], const uint32_t (&tail)[TAIL], const uint32_t (&w)[L][N],
                                                int b) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int x = 16 * b + i;
    m[i] = x < MSG ? tail[x - 16 * SPLIT] : x < TAG_MSG ? w[(x - MSG) / N][(x - MSG) % N] : 0;
  }
  tag_compress(cv, m, b);
}

// The rolled decode (WW_ROLL, p521's few wide words): word 0 decoded, then the words rotated down one, L times.
constexpr bool ROLL_DECODE = WW_ROLL && L <= 4;

// One task (16 rows × 512 columns) a warp, four warps a CTA sharing x's chunk; each warp steps through TB tasks, its
// rows' K-chunks kt·TB … kt·TB + TB − 1. TAG: lane l's block j = jbase + 32qt + l (its L encoded words in each of the
// TB tasks, `wide.block`) is hashed before they are decoded, and its tag written to tags[8 (32qt + l) …]. ahead > 0: once
// its own record is in registers, each warp prefetches into the L2 the record of the same warp `ahead` CTAs on (a wave
// later when ahead is the resident CTAs), so the next wave's HBM reads overlap this wave's arithmetic.
template <int KR, int NB8, bool PARTIALS, bool TAG>
__global__ void __launch_bounds__(GEMM_WARPS * 32, WW_OCC) gemm_kernel(const uint32_t* __restrict__ E, const uint8_t* __restrict__ x,
                                                               float* __restrict__ y, int Mb, int Nn, int Kd,
                                                               uint64_t wbase, uint64_t jbase, bw::Key3 key,
                                                               uint32_t* __restrict__ tags, int ahead) {
  __shared__ __align__(16) uint8_t xs[NB8 * 8][XS_STRIDE];
  const int nchunks = Kd / TASK_COLS, nk = nchunks / TB;
  const int kt = blockIdx.x % nk;
  const int nb = (blockIdx.x / nk) * GEMM_WARPS + threadIdx.x / 32;  // < Nn / 16: `ww_gemm`'s grid
  const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const uint64_t qt = (uint64_t)nb * nk + kt;
  const uint64_t j = jbase + qt * 32 + lane;
  uint32_t cv[8], tail[TAIL];
  if constexpr (TAG) {
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
  }
  float acc[NB8][4];
#pragma unroll
  for (int j = 0; j < NB8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll 1
  for (int s = 0; s < TB; ++s) {
    const int kc = kt * TB + s;
    const uint64_t q = (uint64_t)nb * nchunks + kc;
    if (s) __syncthreads();
    for (int e = threadIdx.x; e < NB8 * 8 * TASK_COLS / 16; e += GEMM_WARPS * 32) {
      int b = e / (TASK_COLS / 16), c = e % (TASK_COLS / 16);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (b < Mb) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kc * TASK_COLS) + c);
      *reinterpret_cast<uint4*>(&xs[b][16 * c]) = v;
    }
    __syncthreads();
    uint32_t w[L][N];
    const uint32_t* rec = E + q * (32ull * L * N) + lane;
#pragma unroll
    for (int i = 0; i < L; ++i)
#pragma unroll
      for (int e = 0; e < N; ++e) w[i][e] = __ldcs(rec + 32 * (i * N + e));
    const uint32_t* nxt = nullptr;
    if (ahead > 0) {
      const unsigned bx = blockIdx.x + (unsigned)ahead;
      const int nb2 = (int)(bx / nk) * GEMM_WARPS + threadIdx.x / 32;
      if (nb2 * 16 < Nn) nxt = E + ((uint64_t)nb2 * nchunks + (bx % nk) * TB + s) * (32ull * L * N);
    }

    // The tag's compressions in order, each word decoded in place once the last compression that reads it is done, so
    // the decode's multiplies and the hash's ALU work are independent instructions in one stream.
    uint32_t t[N], t32[N];
    if constexpr (KR > 0) tw_init(t, t32, wbase + q * (32ull * L) + lane, LANES);
    // a record is L·N rows of 128 bytes, one L2 line each
    auto prefetch = [&] {
      if (nxt)
        for (int r = lane; r < L * N; r += 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(nxt + 32 * r));
    };
    if constexpr (WW_ROLL) {
      if constexpr (TAG) {
        if (TB == 1 || s == 0) {
#pragma unroll
          for (int b = 0; b < (TB == 1 ? TAG_BLOCKS : SPLIT); ++b) tag_step(cv, w, j, b);
          if constexpr (TB > 1) {
#pragma unroll
            for (int i = 0; i < TAIL; ++i) tail[i] = msg_word(w, j, 16 * SPLIT + i);
          }
        }
        if constexpr (TB > 1) {
          if (s == 1) {
#pragma unroll
            for (int b = SPLIT; b < TAG_BLOCKS; ++b) tag_step_second(cv, tail, w, b);
          }
        }
      }
      prefetch();
      if constexpr (KR > 0 && ROLL_DECODE) {
#pragma unroll 1
        for (int i = 0; i < L; ++i) {
          decode_tw<KR>(w[0], t);
          tw_step(t, kDTw);
          uint32_t w0[N];
          copy_k<N>(w0, w[0]);
#pragma unroll
          for (int ii = 0; ii + 1 < L; ++ii) copy_k<N>(w[ii], w[ii + 1]);
          copy_k<N>(w[L - 1], w0);
        }
      } else if constexpr (KR > 0) {
#pragma unroll
        for (int i = 0; i < L; ++i) {
          decode_tw<KR>(w[i], t);
          tw_step(t, kDTw);
        }
      }
    } else {
#pragma unroll
      for (int b = 0; b < TAG_BLOCKS; ++b) {
        if constexpr (TAG) tag_step(cv, w, j, b);
        if (b == 0) prefetch();
        if constexpr (KR > 0) {
#pragma unroll
          for (int i = 0; i < L; ++i)
            if ((1 + (i + 1) * N) / 16 == b) {
              decode_tw<KR>(w[i], t);
              tw_step(t, t32);
            }
        }
      }
    }
    if constexpr (TAG) {
      if (s + 1 == TB) {
        uint4* o = reinterpret_cast<uint4*>(tags + 8 * (qt * 32 + lane));
        o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
        o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
      }
    }
#pragma unroll
    for (int tt = 0; tt < TASK_TILES; ++tt) {
      const uint32_t a[4] = {take4(w, 128 * tt), take4(w, 128 * tt + 32), take4(w, 128 * tt + 64),
                             take4(w, 128 * tt + 96)};
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 4 * tq]);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[8 * j + g][32 * tt + 16 + 4 * tq]);
        mma_e4m3(acc[j], a, b0, b1);
      }
    }
    if constexpr (PARTIALS) {
#pragma unroll
      for (int j = 0; j < NB8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
          if (b < Mb) y[((size_t)kc * Mb + b) * Nn + n] = acc[j][e];
          acc[j][e] = 0.f;
        }
      }
    }
  }
  if constexpr (!PARTIALS) {
#pragma unroll
    for (int j = 0; j < NB8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int n = nb * 16 + g + 8 * (e >> 1), b = 8 * j + 2 * tq + (e & 1);
        if (b < Mb) atomicAdd(y + (size_t)b * Nn + n, acc[j][e]);
      }
    }
  }
}

__global__ void dtw_kernel(uint32_t* out) {
  uint32_t v[N], dv[N];
  tw_init(v, dv, 0, LANES);
#pragma unroll
  for (int e = 0; e < N; ++e) out[e] = dv[e];
}

__global__ void spin_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}
#endif  // WW_HOST

}  // namespace ww

using namespace ww;


static_assert(WW_KMAX >= 1 && WW_KMAX <= 8, "WW_KMAX");
#define WW_CASE(n, F) \
  case n:             \
    F(n);             \
    break;
#if WW_KMAX >= 4
#define WW_CASES(F) WW_CASE(1, F) WW_CASE(2, F) WW_CASE(3, F) WW_CASE(4, F)
#elif WW_KMAX >= 3
#define WW_CASES(F) WW_CASE(1, F) WW_CASE(2, F) WW_CASE(3, F)
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

#ifdef WW_HOST
#include <cstdio>

// stdin: k, then the k c_j, τ, √−1, c1, c1·√−1 (each N hex limbs, least significant first), then lines "i m y": stdout
// gets, a line each, encode(m) at index i and the k inverse rounds of y at index i (`wide.decode_raw`), as N hex limbs.
static bool read_el(uint32_t (&a)[N]) {
  for (int e = 0; e < N; ++e)
    if (scanf("%x", &a[e]) != 1) return false;
  return true;
}
static void write_el(const uint32_t (&a)[N]) {
  for (int e = 0; e < N; ++e) printf(e ? " %08x" : "%08x", a[e]);
}

int main() {
  int k;
  if (scanf("%d", &k) != 1 || k < 1 || k > KMAX) return 1;
  for (int j = 0; j < k; ++j) {
    if (!read_el(kC[j])) return 1;
    neg_el(kNC[j], kC[j]);
  }
  if (!read_el(kTau) || !read_el(kI) || !read_el(kC1) || !read_el(kC1I)) return 1;
  unsigned long long i;
  while (scanf("%llu", &i) == 1) {
    uint32_t m[N], y[N], t[N], v[N], dv[N];
    if (!read_el(m) || !read_el(y)) return 1;
    mul_u64(t, kTau, i);
    // the decode's tweak stepped to i in up to 50 lazy steps of 3, as the kernels advance it
    const unsigned long long r = i / 3 < 50 ? i / 3 : 50;
    tw_init(v, dv, i - 3 * r, 3);
    for (unsigned long long s = 0; s < r; ++s) tw_step(v, dv);
#define F(K)                 \
  do {                       \
    encode_word<K>(m, t);    \
    decode_tw<K>(y, v);      \
  } while (0)
    WW_K_SWITCH(k, F)
#undef F
    write_el(m);
    printf(" ");
    write_el(y);
    printf("\n");
  }
  return 0;
}
#else

static bw::Key3 key_of(const uint8_t* seed) { return bw::b3_key(seed); }

extern "C" {

int ww_constants(int* out) {
  out[0] = WW_BITS; out[1] = N; out[2] = PB; out[3] = L; out[4] = LANES; out[5] = TASK_TILES; out[6] = GEMM_WARPS;
  out[7] = KMAX; out[8] = WW_SQR; out[9] = TAG_BLOCKS; out[10] = WW_ROLL; out[11] = WW_OCC; out[12] = WW_CC;
  out[13] = WW_B3U; out[14] = TB;
  return 0;
}

// c: k × N limbs; tau, i_, c1, c1i: N limbs each (the last three p255's, ignored otherwise).
int ww_set_key(const uint32_t* c, int k, const uint32_t* tau, const uint32_t* i_, const uint32_t* c1, const uint32_t* c1i) {
  if (k < 1 || k > KMAX) return 1;
  uint32_t cc[KMAX][N] = {}, nc[KMAX][N] = {};
  for (int j = 0; j < k; ++j) {
    for (int e = 0; e < N; ++e) cc[j][e] = c[j * N + e];
    neg_el(nc[j], cc[j]);
  }
  if (cudaMemcpyToSymbol(kC, cc, sizeof cc) || cudaMemcpyToSymbol(kNC, nc, sizeof nc) ||
      cudaMemcpyToSymbol(kTau, tau, 4 * N))
    return 2;
  if (i_ && (cudaMemcpyToSymbol(kI, i_, 4 * N) || cudaMemcpyToSymbol(kC1, c1, 4 * N) || cudaMemcpyToSymbol(kC1I, c1i, 4 * N)))
    return 2;
  uint32_t* dtw = nullptr;
  if (cudaMalloc(&dtw, 4 * N)) return 2;
  dtw_kernel<<<1, 1>>>(dtw);
  const bool bad = cudaGetLastError() || cudaMemcpyToSymbol(kDTw, dtw, 4 * N, 0, cudaMemcpyDeviceToDevice);
  cudaFree(dtw);
  if (bad) return 2;
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

// R: XOR of n16 16-byte words, streamed (benchmarks/bandwidth/tagger.cu's read kernel).
int ww_read(const void* buf, uint64_t n16, uint32_t* out, int grid, cudaStream_t s) {
  bw::read_kernel<<<grid, 256, 0, s>>>((const uint4*)buf, n16, out);
  return cudaGetLastError() ? 4 : 0;
}

int ww_spin(long long cycles, cudaStream_t s) {
  spin_kernel<<<1, 1, 0, s>>>(cycles);
  return cudaGetLastError() ? 4 : 0;
}

// The fused GEMM: k rounds (0: the store holds payloads, no decode), tag_seed (32 bytes) or null for no tag; tags is
// the commit workspace's tag area (`ww_workspace_tags`); ahead: the L2 prefetch's distance in CTAs (0: none; a wave is
// `ww_gemm_occupancy` × the SMs).
int ww_gemm(int k, const uint32_t* E, const uint8_t* x, float* y, int Mb, int Nn, int Kd, uint64_t wbase, int partials,
            const uint8_t* tag_seed, uint32_t* tags, int ahead, cudaStream_t s) {
  if (Mb < 1 || Mb > 32 || Nn % (16 * GEMM_WARPS) || Kd % (TB * TASK_COLS) || k < 0 || k > KMAX || ahead < 0) return 1;
  if (tag_seed && !tags) return 1;
  if (wbase % (32ull * L * TB)) return 1;
  const uint64_t jbase = wbase / (L * TB);
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  const int grid = (Nn / 16 / GEMM_WARPS) * (Kd / TASK_COLS / TB);
  bw::Key3 key = {};
  if (tag_seed) key = key_of(tag_seed);
#define G(KR, NB, T)                                                                                             \
  do {                                                                                                           \
    if (partials)                                                                                                \
      gemm_kernel<KR, NB, true, T><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key, tags, \
                                                                    ahead);                                      \
    else                                                                                                         \
      gemm_kernel<KR, NB, false, T><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key, tags, \
                                                                     ahead);                                     \
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

// The fused GEMM's resident CTAs an SM for k, Mb and tag (as `ww_gemm` would launch it), or −1.
int ww_gemm_occupancy(int k, int Mb, int tag, int partials) {
  if (Mb < 1 || Mb > 32 || k < 0 || k > KMAX) return -1;
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  int r = -1;
#define G(KR, NB, T)                                                                                             \
  do {                                                                                                           \
    if (partials) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gemm_kernel<KR, NB, true, T>, GEMM_WARPS * 32, 0); \
    else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gemm_kernel<KR, NB, false, T>, GEMM_WARPS * 32, 0);    \
  } while (0)
#define GT(KR, NB) do { if (tag) G(KR, NB, true); else G(KR, NB, false); } while (0)
#define GK(KR) do { if (nb8 == 1) GT(KR, 1); else if (nb8 == 2) GT(KR, 2); else GT(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WW_CASES(GK)
    default: return -1;
  }
#undef GK
#undef GT
#undef G
  return r;
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
#endif  // WW_HOST
