// The wide word codecs on the RTX PRO 6000 (sm_120): `wide.py` is the reference. One build per field (WW_BITS = 127,
// 255, 521, 256 or 512); WW_SQR picks the squaring (0: operand scanning on 64-bit products, 1: PTX carry chains, lo and
// hi passes).
//
// A field element is N little-endian u32 limbs. p = 2^M − C (C = 1 for the Mersenne fields, 19 for p255, 189 and 569
// for the full-width p256 and p512, M = 32N). Lazy values: `sqr` and `mul` take inputs below 2^M + 2^8 (Mersenne) or
// 2^(32N) (the others) and return values in the same range; `canon` takes such a value to [0, p).
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
// WW_TB = 2: one tag over a lane's records in two adjacent K-chunks of the same rows (`wide.block`'s tb), which a pair
// of warps of the fused GEMM takes: half the tags to commit, and a compression fewer a record. Rolled builds only.
#ifndef WW_TB
#define WW_TB 1
#endif
#if WW_TB != 1 && (WW_TB != 2 || !WW_ROLL)
#error "WW_TB is 1, or 2 with WW_ROLL"
#endif
// WW_KW: the adjacent K-chunks a CTA of the fused GEMM covers (GEMM_WARPS / WW_KW row groups); WW_TB = 2 needs 2.
#ifndef WW_KW
#define WW_KW WW_TB
#endif
#if (WW_KW != 1 && WW_KW != 2) || WW_KW < WW_TB
#error "WW_KW is 1 or 2, and at least WW_TB"
#endif
// WW_SKEND = 1: the fused GEMM sums its warps' sketches (and hashes them) after the GEMM rather than before the decode,
// so no warp waits at a barrier for the others' records before it decodes its own.
#ifndef WW_SKEND
#define WW_SKEND 0
#endif
// WW_SKU = 1: the sketch's two compressions a vector unrolled in a rolled build: one warp runs them at the CTA's end
// (WW_SKEND), where their latency rather than the code's size counts.
#ifndef WW_SKU
#define WW_SKU 0
#endif
// WW_MAXREG: the fused GEMM's registers a thread (`__maxnreg__`, in place of WW_OCC's bounds), such as 160, at which
// three CTAs an SM leave 4096 registers to the commit's kernels beside them.
#ifndef WW_MAXREG
#define WW_MAXREG 0
#endif
#if WW_MAXREG
#define WW_GEMM_BOUNDS __maxnreg__(WW_MAXREG)
#else
#define WW_GEMM_BOUNDS __launch_bounds__(GEMM_WARPS * 32, WW_OCC)
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
template <>
struct Fd<256> {
  static constexpr int N = 8, M = 256, PB = 256;
  static constexpr uint32_t C = 189;
};
template <>
struct Fd<512> {
  static constexpr int N = 16, M = 512, PB = 512;
  static constexpr uint32_t C = 569;
};

using F = Fd<WW_BITS>;
constexpr int N = F::N, M = F::M, PB = F::PB;
constexpr uint32_t C = F::C;
constexpr bool MERS = C == 1;
constexpr bool FULL = M == 32 * N;      // a word is all its limbs' bits, p = 3 mod 4
constexpr int MB = M / 32, MS = M % 32;  // bit M is bit MS of limb MB
constexpr uint32_t FOLD = C << (32 * N - M);  // 2^(32N) mod p, not Mersenne
// the full-width root: (p + 1)/4 = (2^RN − 1)·2^RS + RE, RE = 2^RS − (C − 1)/4 (`wide.root_split`)
constexpr int RD = (int)(C - 1) / 4;
constexpr int rs_of(int d) { return d == 0 ? 0 : 1 + rs_of(d / 2); }
constexpr int RS = rs_of(RD), RN = M - 2 - RS, RE = (1 << RS) - RD;
constexpr int KMAX = WW_KMAX;
constexpr int LANES = 32, TASK_TILES = 16, LANE_BYTES = 16 * TASK_TILES, TASK_COLS = 32 * TASK_TILES;
constexpr int L = (8 * LANE_BYTES + PB - 1) / PB;  // words a lane a task
constexpr int GEMM_WARPS = 4;
constexpr int MSG = 2 + L * N;                     // a record's part of a message in u32: u64be j, then the words
constexpr int TB = WW_TB, KW = WW_KW;
constexpr int TAG_MSG = 2 + TB * L * N;            // a block's message: u64be j, then its TB records' words
constexpr int TAG_BLOCKS = (4 * TAG_MSG + 63) / 64;
constexpr int SPLIT = MSG / 16;                    // TB = 2: compressions 0 … SPLIT − 1 read only the first record,
constexpr int TAIL = MSG - 16 * SPLIT;             // and compression SPLIT the first record's last TAIL words
static_assert(TAIL > 0, "a record's message ends inside a compression");
static_assert(GEMM_WARPS % KW == 0 && (TB == 1 || TB == KW), "a CTA holds whole pairs");
static_assert(32 * N > M || FULL, "a spare bit above p, or a full-width word");
static_assert(C != 1 || (MB == N - 1 && MS > 0), "a Mersenne p's top bit is inside the last limb");
static_assert(!FULL || (C % 4 == 1 && PB == M), "a full-width p is 3 mod 4, its payload the whole word");
static_assert(MERS || FOLD < 65536, "`reduce`'s last carry fits limb 0");

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
    // 2^(32N) ≡ FOLD (38 for p255, C at full width). t = lo + 2^(32N) hi ≡ lo + FOLD hi, then the carry limb (≤ FOLD)
    // once more, then a carry once more.
    static_assert(T <= 2 * N, "reduce takes at most 2N limbs");
    uint64_t c = 0;
#pragma unroll
    for (int e = 0; e < N; ++e) {
      const uint32_t h = N + e < T ? t[N + e] : 0;
      c += (uint64_t)h * FOLD + t[e];
      r[e] = (uint32_t)c;
      c >>= 32;
    }
    uint64_t d = (uint64_t)(uint32_t)c * FOLD;
#pragma unroll
    for (int e = 0; e < N; ++e) {
      d += r[e];
      r[e] = (uint32_t)d;
      d >>= 32;
    }
    r[0] += (uint32_t)d * FOLD;  // a carry leaves r below FOLD², in limb 0, so this one does not carry
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
    // r < 2^(32N), below 3p for p255 and 2p at full width: subtract p while r ≥ p, twice or once
#pragma unroll
    for (int pass = 0; pass < (FULL ? 1 : 2); ++pass) {
      const uint32_t borrow = sub_k<N>(s, r, p);
      select_k<N>(r, !borrow, s, r);
    }
  }
}

__device__ __forceinline__ void addmod(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  // a, b canonical: a + b < 2p, at full width with a carry out of the N limbs
  const uint32_t cy = add_k<N>(r, a, b);
  uint32_t p[N], s[N];
  pconst(p);
  const uint32_t borrow = sub_k<N>(s, r, p);
  select_k<N>(r, cy || !borrow, s, r);
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
  if constexpr (MERS || FULL) {
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

// r = a + b folded once mod a Mersenne p: a ≤ p + 1 and b ≤ p give r ≤ p + 1. (A template, as `dec_round_mers`, so
// that the full-width builds, whose MB is N, never instantiate it.)
template <int B = MB>
__device__ __forceinline__ void add_fold(uint32_t (&r)[N], const uint32_t (&a)[N], const uint32_t (&b)[N]) {
  static_assert(MERS && B == MB, "Mersenne only");
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
template <int B = MB>
__device__ __forceinline__ void dec_round_mers(uint32_t (&y)[N], const uint32_t (&nu)[N]) {
  static_assert(MERS && B == N - 1, "the top limb holds bit M − 1");
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

// r = a^(2^n − 1), a lazy: x_2j = x_j^(2^j)·x_j, x_(j+1) = x_j²·a (`wide._pow2m1_chain`).
template <int NN>
__device__ __forceinline__ void pow2m1(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  static_assert(NN >= 1, "n ≥ 1");
  if constexpr (NN == 1) {
    copy_k<N>(r, a);
  } else if constexpr (NN % 2) {
    pow2m1<NN - 1>(r, a);
    sqr(r, r);
    mul(r, r, a);
  } else {
    uint32_t h[N];
    pow2m1<NN / 2>(h, a);
    copy_k<N>(r, h);
    sqr_n<NN / 2>(r);
    mul(r, r, h);
  }
}

// r = a^E for a small E ≥ 1, left to right.
template <int E>
__device__ __forceinline__ void pow_small(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  static_assert(E >= 1, "E ≥ 1");
  if constexpr (E == 1) {
    copy_k<N>(r, a);
  } else if constexpr (E % 2) {
    pow_small<E - 1>(r, a);
    mul(r, r, a);
  } else {
    pow_small<E / 2>(r, a);
    sqr(r, r);
  }
}

// ρ(a), a canonical.
__device__ __forceinline__ void root(uint32_t (&r)[N], const uint32_t (&a)[N]) {
  uint32_t p[N];
  pconst(p);
  if constexpr (MERS || FULL) {
    // s = a^((p+1)/4): 2^(M − 2) for a Mersenne p, (2^RN − 1)·2^RS + RE at full width
    uint32_t s[N], q[N], ns[N];
    if constexpr (MERS) {
      copy_k<N>(s, a);
      sqr_n<M - 2>(s);
    } else {
      uint32_t e[N];
      pow2m1<RN>(s, a);
      sqr_n<RS>(s);
      pow_small<RE>(e, a);
      mul(s, s, e);
    }
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

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The sketch-then-hash tag (`wide_sketch`): a lane's record is SK_M steps of 16 bytes, its limbs in (word, limb) order.
constexpr int SK_K = 16, SK_M = (L * N + 3) / 4;
constexpr int SK_ROW = SK_K + 1;  // a vector's row in shared memory, padded against bank conflicts

// Limb x of the lane's record in (word, limb) order, 0 past its end; x a constant.
__device__ __forceinline__ uint32_t rec_limb(const uint32_t (&w)[L][N], int x) { return x < L * N ? w[x / N][x % N] : 0; }

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

__device__ __forceinline__ void sk_compress(uint32_t (&cv)[8], uint32_t (&m)[16], uint32_t len, uint32_t flags) {
  if constexpr (WW_ROLL && !WW_SKU) b3_compress_rolled(cv, m, len, flags);
  else bw::b3_compress(cv, m, 0, len, flags, cv);
}

// ro-sketch-blake3/v0's tag of one vector: BLAKE3-keyed(key, u64be j ‖ its 16 int32 words), two compressions.
__device__ __forceinline__ void sketch_tag(const bw::Key3& key, uint64_t j, const int32_t* s, uint32_t (&cv)[8]) {
  uint32_t m[16];
  m[0] = bw::bswap32((uint32_t)(j >> 32));
  m[1] = bw::bswap32((uint32_t)j);
#pragma unroll
  for (int i = 0; i < 14; ++i) m[2 + i] = (uint32_t)s[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
  sk_compress(cv, m, 64, bw::B3_KEYED | bw::B3_CHUNK_START);
  m[0] = (uint32_t)s[14];
  m[1] = (uint32_t)s[15];
#pragma unroll
  for (int i = 2; i < 16; ++i) m[i] = 0;
  sk_compress(cv, m, 8, bw::B3_KEYED | bw::B3_CHUNK_END | bw::B3_ROOT);
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

// A warp pair's named barrier, its id an immediate: with a register id ptxas reserves all 16, one CTA an SM.
template <int ID>
__device__ __forceinline__ void pair_arrive() {
  asm volatile("bar.arrive %0, 64;" ::"n"(ID) : "memory");
}
template <int ID>
__device__ __forceinline__ void pair_sync() {
  asm volatile("bar.sync %0, 64;" ::"n"(ID) : "memory");
}

// One task (16 rows × 512 columns) a warp, four warps a CTA: GEMM_WARPS / KW row groups × KW adjacent K-chunks, the
// CTA's chunks of x in shared memory. TAG: lane l's block j = jbase + 32qt + l (its L encoded words in each of the TB
// tasks of its row group, `wide.block`) is hashed before they are decoded, and its tag written to tags[8 (32qt + l) …].
// TB = 2: the even warp of a pair runs the first SPLIT compressions on its record and leaves the chaining value in the
// tag's slot (named barrier 1 + pair); the odd warp reads the even record's last TAIL words from the store and finishes
// the tag on its own record. ahead > 0: once its own record is in registers, each warp prefetches into the L2 the
// record of the same warp `ahead` CTAs on (a wave later when ahead is the resident CTAs), so the next wave's HBM reads
// overlap this wave's arithmetic.
// SK > 0 (KW = 1, no TAG): the sketch of the encoded words (`wide_sketch`), the CTA being tile blockIdx.x and warp i its
// task i: each warp's mma.m16n8k32.s8 over its record against `signs` (`wide_sketch.sign_fragments`) before the
// decode, the four accumulators summed in shared memory, and the tile's 16 vectors at sk[256 · blockIdx.x …] when sk is
// non-null. SK = 2: warp 0 also hashes them (ro-sketch-blake3/v0 under key, vector v of the call at index jbase + v)
// into tags[8 v …] before its decode, its upper 16 lanes repeating the lower's so the warp never diverges. WW_SKEND:
// the sum and the hash after the GEMM instead.
template <int KR, int NB8, bool PARTIALS, bool TAG, int SK>
__global__ void WW_GEMM_BOUNDS gemm_kernel(const uint32_t* __restrict__ E, const uint8_t* __restrict__ x,
                                                               float* __restrict__ y, int Mb, int Nn, int Kd,
                                                               uint64_t wbase, uint64_t jbase, bw::Key3 key,
                                                               uint32_t* __restrict__ tags, int ahead,
                                                               const uint4* __restrict__ signs, int32_t* __restrict__ sk) {
  constexpr int RG = GEMM_WARPS / KW, UNITS = TASK_COLS / 16;
  static_assert(TB == 1 || RG == 2, "two warp pairs a CTA, named barriers 1 and 2");
  static_assert(SK == 0 || (KW == 1 && !TAG), "the sketch's tile is four row blocks at one chunk, and replaces the tag");
  __shared__ __align__(16) uint8_t xs[KW][NB8 * 8][TASK_COLS];  // row r's 16-byte unit c at c ^ (r mod 8): no conflicts
  const int nchunks = Kd / TASK_COLS, nk = nchunks / KW;
  const int warp = threadIdx.x / 32, h = warp % KW, pair = warp / KW;
  const int kt = blockIdx.x % nk, kc = kt * KW + h;
  const int nb = (blockIdx.x / nk) * RG + pair;  // < Nn / 16: `ww_gemm`'s grid
  const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
  const uint64_t q = (uint64_t)nb * nchunks + kc, qt = q / TB;
  for (int e = threadIdx.x; e < NB8 * 8 * KW * UNITS; e += GEMM_WARPS * 32) {
    const int b = e / (KW * UNITS), c = e % (KW * UNITS);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < Mb) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * Kd + (size_t)kt * KW * TASK_COLS) + c);
    *reinterpret_cast<uint4*>(&xs[c / UNITS][b][16 * ((c % UNITS) ^ (b & 7))]) = v;
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
    const int nb2 = (int)(bx / nk) * RG + pair;
    if (nb2 * 16 < Nn) nxt = E + ((uint64_t)nb2 * nchunks + (bx % nk) * KW + h) * (32ull * L * N);
  }
  __shared__ int32_t skp[SK > 0 ? GEMM_WARPS : 1][16][SK_ROW];
  auto sketch_sum = [&] {
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * SK_K; i += GEMM_WARPS * 32) {
      const int r = i / SK_K, c = i % SK_K;
      int v = 0;
#pragma unroll
      for (int u = 0; u < GEMM_WARPS; ++u) v += skp[u][r][c];
      skp[0][r][c] = v;
      if (sk) sk[(size_t)blockIdx.x * (16 * SK_K) + i] = v;
    }
    if constexpr (SK == 2) {
      __syncthreads();
      if (warp == 0) {
        const int r = lane & 15;
        uint32_t cv[8];
        sketch_tag(key, jbase + 16ull * blockIdx.x + r, skp[0][r], cv);
        if (lane < 16) {
          uint4* o = reinterpret_cast<uint4*>(tags + 8 * (16ull * blockIdx.x + r));
          o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
          o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
        }
      }
    }
  };
  if constexpr (SK > 0) {
    int d[2][4] = {};
    const uint4* sg = signs + (size_t)warp * SK_M * 32 + lane;
#pragma unroll
    for (int s = 0; s < SK_M; ++s) {
      const uint32_t a[4] = {rec_limb(w, 4 * s), rec_limb(w, 4 * s + 1), rec_limb(w, 4 * s + 2), rec_limb(w, 4 * s + 3)};
      const uint4 b = __ldg(sg + 32 * s);
      mma_s8(d[0], a, b.x, b.y);
      mma_s8(d[1], a, b.z, b.w);
    }
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8)
#pragma unroll
      for (int e = 0; e < 4; ++e) skp[warp][g + 8 * (e >> 1)][8 * c8 + 2 * tq + (e & 1)] = d[c8][e];
    if constexpr (!WW_SKEND) sketch_sum();
  }

  // The tag's compressions in order, each word decoded in place once the last compression that reads it is done, so
  // the decode's multiplies and the hash's ALU work are independent instructions in one stream.
  uint32_t cv[8], t[N], t32[N];
  const uint64_t j = jbase + qt * 32 + lane;
  uint4* const slot = reinterpret_cast<uint4*>(tags + 8 * (qt * 32 + lane));
  if constexpr (TAG) {
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key.k[i];
  }
  if constexpr (KR > 0) tw_init(t, t32, wbase + q * (32ull * L) + lane, LANES);
  // a record is L·N rows of 128 bytes, one L2 line each
  auto prefetch = [&] {
    if (nxt)
      for (int r = lane; r < L * N; r += 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(nxt + 32 * r));
  };
  if constexpr (WW_ROLL) {
    if constexpr (TAG && TB == 1) {
#pragma unroll
      for (int b = 0; b < TAG_BLOCKS; ++b) tag_step(cv, w, j, b);
    } else if constexpr (TAG) {
      if (h == 0) {
#pragma unroll
        for (int b = 0; b < SPLIT; ++b) tag_step(cv, w, j, b);
        slot[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
        slot[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
        if (pair == 0) pair_arrive<1>();
        else pair_arrive<2>();
      } else {
        // message word 16·SPLIT + i is the even record's limb 16·SPLIT − 2 + i
        uint32_t tail[TAIL];
        const uint32_t* prev = rec - 32ull * L * N;
#pragma unroll
        for (int i = 0; i < TAIL; ++i) tail[i] = __ldcg(prev + 32 * (16 * SPLIT - 2 + i));
        if (pair == 0) pair_sync<1>();
        else pair_sync<2>();
        const uint4 c0 = __ldcg(slot), c1 = __ldcg(slot + 1);
        cv[0] = c0.x, cv[1] = c0.y, cv[2] = c0.z, cv[3] = c0.w, cv[4] = c1.x, cv[5] = c1.y, cv[6] = c1.z, cv[7] = c1.w;
#pragma unroll
        for (int b = SPLIT; b < TAG_BLOCKS; ++b) tag_step_second(cv, tail, w, b);
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
    if (TB == 1 || h == 1) {
      slot[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
      slot[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
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
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(&xs[h][8 * j + g][16 * ((2 * tt) ^ g) + 4 * tq]);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(&xs[h][8 * j + g][16 * ((2 * tt + 1) ^ g) + 4 * tq]);
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
  if constexpr (SK > 0 && WW_SKEND) sketch_sum();
}

// ro-sketch-blake3/v0's tags apart from the GEMM: a thread a vector of sk (16 int32 each), vector v's tag (index
// jbase + v) at tags[8 v …].
__global__ void __launch_bounds__(128) sketch_tags_kernel(const int32_t* __restrict__ sk, uint64_t n, uint64_t jbase,
                                                          bw::Key3 key, uint32_t* __restrict__ tags) {
  const uint64_t v = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (v >= n) return;
  int32_t s[SK_K];
  const int4* p = reinterpret_cast<const int4*>(sk + SK_K * v);
#pragma unroll
  for (int i = 0; i < SK_K / 4; ++i) {
    const int4 q = __ldcs(p + i);
    s[4 * i] = q.x, s[4 * i + 1] = q.y, s[4 * i + 2] = q.z, s[4 * i + 3] = q.w;
  }
  uint32_t cv[8];
  sketch_tag(key, jbase + v, s, cv);
  uint4* o = reinterpret_cast<uint4*>(tags + 8 * v);
  o[0] = make_uint4(cv[0], cv[1], cv[2], cv[3]);
  o[1] = make_uint4(cv[4], cv[5], cv[6], cv[7]);
}

// `bw::b3_leaves`'s leaf `leaf` of a pass of B tags, over its tags at T.
__device__ __forceinline__ void sketch_leaf(const uint32_t* __restrict__ T, uint64_t B, uint64_t leaf,
                                            const bw::Mid& leafmid, uint64_t s[8]) {
  const uint64_t first = leaf * 16;
  const int c = (int)(B - first < 16 ? B - first : 16);
  const int data = 1 + 4 * c, nb = (8 * data + 17 + 127) / 128;
#pragma unroll
  for (int e = 0; e < 8; ++e) s[e] = leafmid.h[e];
#pragma unroll 1
  for (int blk = 0; blk < nb; ++blk) {
    uint64_t w[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int q = 16 * blk + e;
      uint64_t v = 0;
      if (q == 0)
        v = leaf;
      else if (q < data)
        v = ((uint64_t)bw::bswap32(T[2 * (q - 1)]) << 32) | bw::bswap32(T[2 * (q - 1) + 1]);
      else if (q == data)
        v = 0x80ULL << 56;
      if (q == 16 * nb - 1) v = (uint64_t)(128 + 8 + 32 * c) * 8;
      w[e] = v;
    }
    bw::compress(s, w);
  }
}

// The leaves over the n tags [16 leaf0, 16 leaf0 + n) of a pass of B tags, held at `tags` (one linear's, read while
// L2 still has them), each leaf at its index in the pass.
__global__ void __launch_bounds__(128) sketch_leaves(const uint32_t* __restrict__ tags, uint64_t B, uint64_t leaf0,
                                                     uint64_t n, bw::Mid leafmid, uint64_t* __restrict__ leaves) {
  const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, leaf = leaf0 + i;
  if (16 * i >= n) return;
  uint64_t s[8];
  sketch_leaf(tags + 16 * i * 8, B, leaf, leafmid, s);
#pragma unroll
  for (int e = 0; e < 8; ++e) leaves[leaf * 8 + e] = s[e];
}

// The tree levels a linear's leaves fold into (`ww_sketch_fold`) before the pass's tree goes on from level SUB.
constexpr int SUB = 6;

struct Levels {
  uint64_t off[SUB + 1];  // in nodes: level t's node for the range's first, t = 0..SUB
};

// `ww_sketch_fold`'s levels with one warp a block (2^SUB leaves, 2^(SUB−1) threads at level 1), so that a block's
// registers fit in what three fused-GEMM CTAs an SM leave (4096 at 160 registers) and it never waits for a CTA's slot.
static_assert(SUB == 6, "one warp a block at level 1");
__global__ void __launch_bounds__(32) fold_warp(uint64_t* nodes, Levels lv, uint64_t c) {
#pragma unroll 1
  for (int t = 0; t < SUB; ++t) {
    const uint64_t w = 32 >> t;
    if (threadIdx.x < w) bw::level_step(nodes + lv.off[t] * 8, c >> t, nodes + lv.off[t + 1] * 8, blockIdx.x * w + threadIdx.x);
    __syncwarp();
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

static const char SKETCH_NAME[] = "ro-sketch-blake3/v0";

// ro-sketch-blake3/v0's key: `bw::b3_key` under its name.
static bw::Key3 sketch_key(const uint8_t* seed) {
  std::vector<uint8_t> p = bw::prefix_block(bw::TAG_DOMAIN, SKETCH_NAME, seed);
  uint64_t d[8];
  bw::sha512_host(p.data(), p.size(), d);
  bw::Key3 k;
  for (int i = 0; i < 8; ++i) {
    const uint32_t be = (uint32_t)(i & 1 ? d[i / 2] : d[i / 2] >> 32);
    k.k[i] = (be >> 24) | ((be >> 8) & 0xff00u) | ((be << 8) & 0xff0000u) | (be << 24);
  }
  return k;
}

// The leaves (16 tags each) and tree over `blocks` tags of the workspace under the tag's name.
static int commit_named(void* workspace, uint64_t blocks, const char* name, const uint8_t* seed, uint8_t* root,
                        cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  bw::Mid lm = bw::prefix_midstate(bw::LEAF_DOMAIN, name, seed);
  const unsigned leaf_grid = (unsigned)(((blocks + 15) / 16 + 127) / 128);
  bw::b3_leaves<<<leaf_grid, 128, 0, s>>>(p.tags, p.blocks, blocks, lm, p.nodes);
  bw::launch_tree(p, s);
  if (cudaGetLastError()) return 4;
  if (root) return bw::read_root(p, root, s);
  return 0;
}

extern "C" {

int ww_constants(int* out) {
  out[0] = WW_BITS; out[1] = N; out[2] = PB; out[3] = L; out[4] = LANES; out[5] = TASK_TILES; out[6] = GEMM_WARPS;
  out[7] = KMAX; out[8] = WW_SQR; out[9] = TAG_BLOCKS; out[10] = WW_ROLL; out[11] = WW_OCC; out[12] = WW_CC;
  out[13] = WW_B3U; out[14] = TB; out[15] = KW; out[16] = SK_M; out[17] = WW_SKEND; out[18] = SUB; out[19] = WW_MAXREG;
  out[20] = WW_SKU;
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
  if (Mb < 1 || Mb > 32 || Nn % (16 * GEMM_WARPS) || Kd % (KW * TASK_COLS) || k < 0 || k > KMAX || ahead < 0) return 1;
  if (tag_seed && !tags) return 1;
  if (wbase % (32ull * L * TB)) return 1;
  const uint64_t jbase = wbase / (L * TB);
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  const int grid = (Nn / 16 / (GEMM_WARPS / KW)) * (Kd / TASK_COLS / KW);
  bw::Key3 key = {};
  if (tag_seed) key = key_of(tag_seed);
#define G(KR, NB, T)                                                                                             \
  do {                                                                                                           \
    if (partials)                                                                                                \
      gemm_kernel<KR, NB, true, T, 0><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key,    \
                                                                       tags, ahead, nullptr, nullptr);           \
    else                                                                                                         \
      gemm_kernel<KR, NB, false, T, 0><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key,   \
                                                                        tags, ahead, nullptr, nullptr);          \
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

// The fused GEMM with the sketch-then-hash tag in place of ro-blake3/v1 (`wide_sketch`; KW = 1 builds): signs is the
// window's `wide_sketch.sign_fragments` (T · SK_M · 512 bytes), sk null or 16 · SK_K int32 a tile (the grid's CTAs,
// `wide_sketch.tiles`); with seed and tags non-null the kernel hashes each tile's vectors too, vector v of this call
// (index jbase + v) at tags[8 v …], else ww_sketch_tags hashes sk afterwards.
int ww_gemm_sketch(int k, const uint32_t* E, const uint8_t* x, float* y, int Mb, int Nn, int Kd, uint64_t wbase,
                   int partials, const void* signs, int32_t* sk, uint64_t jbase, const uint8_t* seed, uint32_t* tags,
                   int ahead, cudaStream_t s) {
  if (KW != 1 || !signs || (seed != nullptr) != (tags != nullptr)) return 1;
  if (Mb < 1 || Mb > 32 || Nn % (16 * GEMM_WARPS) || Kd % TASK_COLS || k < 0 || k > KMAX || ahead < 0) return 1;
  if (wbase % (32ull * L)) return 1;
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  const int grid = (Nn / 16 / GEMM_WARPS) * (Kd / TASK_COLS);
  bw::Key3 key = {};
  if (seed) key = sketch_key(seed);
#if WW_KW == 1
  const uint4* sg = reinterpret_cast<const uint4*>(signs);
#define G(KR, NB, S)                                                                                             \
  do {                                                                                                           \
    if (partials)                                                                                                \
      gemm_kernel<KR, NB, true, false, S><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase, key, \
                                                                           tags, ahead, sg, sk);                 \
    else                                                                                                         \
      gemm_kernel<KR, NB, false, false, S><<<grid, GEMM_WARPS * 32, 0, s>>>(E, x, y, Mb, Nn, Kd, wbase, jbase,     \
                                                                            key, tags, ahead, sg, sk);           \
  } while (0)
#define GT(KR, NB) do { if (tags) G(KR, NB, 2); else G(KR, NB, 1); } while (0)
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
#else
  return 1;
#endif
}

// ro-sketch-blake3/v0's tags of n vectors of sk (16 int32 each), vector v's (index jbase + v) at tags[8 v …].
int ww_sketch_tags(const int32_t* sk, uint64_t n, uint64_t jbase, const uint8_t* seed, uint32_t* tags, cudaStream_t s) {
  if (!n) return 0;
  sketch_tags_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(sk, n, jbase, sketch_key(seed), tags);
  return cudaGetLastError() ? 4 : 0;
}

// The fused GEMM's resident CTAs an SM for k, Mb and tag (0 none, 1 ro-blake3/v1, 2 the sketch, 3 the sketch hashed in
// the kernel), as `ww_gemm` or `ww_gemm_sketch` would launch it, or −1.
int ww_gemm_occupancy(int k, int Mb, int tag, int partials) {
  if (Mb < 1 || Mb > 32 || k < 0 || k > KMAX || tag < 0 || tag > 3 || (tag >= 2 && KW != 1)) return -1;
  const int nb8 = Mb <= 8 ? 1 : Mb <= 16 ? 2 : 4;
  int r = -1;
#define G(KR, NB, T, S)                                                                                          \
  do {                                                                                                           \
    if (partials) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gemm_kernel<KR, NB, true, T, S>, GEMM_WARPS * 32, 0); \
    else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, gemm_kernel<KR, NB, false, T, S>, GEMM_WARPS * 32, 0);    \
  } while (0)
#if WW_KW == 1
#define GS(KR, NB, S) G(KR, NB, false, S)
#else
#define GS(KR, NB, S) r = -1
#endif
#define GT(KR, NB)                       \
  do {                                   \
    if (tag == 1) G(KR, NB, true, 0);    \
    else if (tag == 2) GS(KR, NB, 1);    \
    else if (tag == 3) GS(KR, NB, 2);    \
    else G(KR, NB, false, 0);            \
  } while (0)
#define GK(KR) do { if (nb8 == 1) GT(KR, 1); else if (nb8 == 2) GT(KR, 2); else GT(KR, 4); } while (0)
  switch (k) {
    case 0: GK(0); break;
    WW_CASES(GK)
    default: return -1;
  }
#undef GK
#undef GT
#undef GS
#undef G
  return r;
}

size_t ww_workspace_bytes(uint64_t blocks) { return bw::workspace_bytes(bw::TAG_B3, blocks); }

uint32_t* ww_workspace_tags(void* workspace, uint64_t blocks) { return bw::make_pass(bw::TAG_B3, blocks, workspace).tags; }

// The tags' leaves and tree over `blocks` blocks; with root non-null, the root's 64 bytes (and a sync).
int ww_commit(void* workspace, uint64_t blocks, const uint8_t* seed, uint8_t* root, cudaStream_t s) {
  return commit_named(workspace, blocks, bw::TAG_NAMES[bw::TAG_B3], seed, root, s);
}

// `ww_commit` over n ro-sketch-blake3/v0 tags (a leaf a tile).
int ww_sketch_commit(void* workspace, uint64_t n, const uint8_t* seed, uint8_t* root, cudaStream_t s) {
  return commit_named(workspace, n, SKETCH_NAME, seed, root, s);
}

// `ww_sketch_commit`'s leaves of the pass's tags [jbase, jbase + n) (jbase a multiple of 16), read from `tags`, into
// the leaves of the commit workspace of a pass of `blocks` tags; its own tag area is not read.
int ww_sketch_leaves(void* workspace, uint64_t blocks, const uint32_t* tags, uint64_t jbase, uint64_t n,
                     const uint8_t* seed, cudaStream_t s) {
  if (jbase % 16 || jbase + n > blocks) return 2;
  if (!n) return 0;
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  bw::Mid lm = bw::prefix_midstate(bw::LEAF_DOMAIN, SKETCH_NAME, seed);
  sketch_leaves<<<(unsigned)(((n + 15) / 16 + 127) / 128), 128, 0, s>>>(tags, blocks, jbase / 16, n, lm, p.nodes);
  return cudaGetLastError() ? 4 : 0;
}

// The tree's levels 1 … SUB over the leaves [jbase / 16, (jbase + n) / 16) that `ww_sketch_leaves` wrote, while L2
// still holds them (jbase and n multiples of 16 · 2^SUB, so no level of the range carries a node), in one launch
// (`node_span`, or with warp `fold_warp`): `ww_sketch_tree` from level SUB then reads back 64 B a 2^SUB leaves rather
// than every leaf.
int ww_sketch_fold(void* workspace, uint64_t blocks, uint64_t jbase, uint64_t n, int warp, cudaStream_t s) {
  const uint64_t span = 16ull << SUB;
  if (jbase % span || n % span || jbase + n > blocks) return 2;
  if (!n) return 0;
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  if (p.lv.n.size() <= (size_t)SUB) return 2;
  const uint64_t l0 = jbase / 16, c = n / 16;
  if (warp) {
    Levels lv;
    for (int t = 0; t <= SUB; ++t) lv.off[t] = p.lv.off[t] + (l0 >> t);
    fold_warp<<<(unsigned)(c >> SUB), 32, 0, s>>>(p.nodes, lv, c);
    return cudaGetLastError() ? 4 : 0;
  }
  static_assert(SUB <= bw::SPAN, "one node_span launch");
  bw::Span sp;
  sp.levels = SUB;
  for (int t = 0; t <= SUB; ++t) {
    sp.off[t] = p.lv.off[t] + (l0 >> t);
    sp.n[t] = c >> t;
  }
  bw::node_span<<<(unsigned)((c + 2 * bw::SPAN_THREADS - 1) / (2 * bw::SPAN_THREADS)), bw::SPAN_THREADS, 0, s>>>(p.nodes, sp);
  return cudaGetLastError() ? 4 : 0;
}

// The tree from level `from` (0: the leaves `ww_sketch_leaves` wrote; SUB: the nodes `ww_sketch_fold` wrote); with
// root non-null, the root's 64 bytes (and a sync).
int ww_sketch_tree(void* workspace, uint64_t blocks, int from, uint8_t* root, cudaStream_t s) {
  bw::Pass p = bw::make_pass(bw::TAG_B3, blocks, workspace);
  if (from < 0 || (size_t)from >= p.lv.n.size()) return 2;
  bw::launch_tree(p, s, (size_t)from);
  if (cudaGetLastError()) return 4;
  return root ? bw::read_root(p, root, s) : 0;
}

// The commit workspace's leaves (64 bytes each, big-endian u64 words), for the gate.
const uint64_t* ww_workspace_leaves(void* workspace, uint64_t blocks) { return bw::make_pass(bw::TAG_B3, blocks, workspace).nodes; }

}  // extern "C"
#endif  // WW_HOST
