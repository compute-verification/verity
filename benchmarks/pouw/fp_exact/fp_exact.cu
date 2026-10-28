// Is an FP8 or FP4 MMA chain exact integer arithmetic on sm_120? Each warp chains K / k_step MMAs of one format into
// one m16n8 FP32 accumulator, from an initial word C0, with operands sampled from an exact grid (fp_exact.py's GRIDS) by a
// counter hash, and compares every output word with the exact sum, computed in int64 by the same lane in the grid's
// units. The same tile runs in three accumulation orders (the K chunks of 8 permuted), so a value that comes out as two
// words is counted. fp_exact.py writes the configs, rechecks sample tiles' references and words on the host, and
// reads the counts.
//
//   fp_exact CONFIGS OUT_JSONL RECORDS_BIN [--control]
//   fp_exact CONFIGS --dump OUT_BIN        # the first config's tile 0 operands from the host's generator (no GPU)
//
// CONFIGS: one line a config, "name fmt grid dist c0_word K tiles seed nrec" (fmt, grid, dist: the enums below).
// --control runs nothing after poisoning (gpu_kit's no-write control): every config must then be refused.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../gpu_kit/bench.cuh"

enum Fmt : int { E4M3 = 0, E4M3_KIND = 1, MXF4 = 2, MXF4NV = 3, NVF4 = 4, SP_MXF4 = 5, SP_NVF4 = 6, NFMT };
enum Dist : int { D_U = 0, D_MAX = 1, D_MAXSIGN = 2, D_RAMP = 3, D_CANCEL = 4, D_EDGE = 5 };
// Per-block scales. UNIT: 1.0 on every block. LO/HI: one fixed scale a block (UE4M3 3 / 15; UE8M0 2^-3 / 2^3).
// MIX: a scale a block, drawn per block and per operand (UE4M3 the odd integers 1..15; UE8M0 2^k, k in [-3, 3]).
// BIND / BIND26: UE8M0 blocks alternate 2^7 and 2^-7 (spread 28, one past the 27-bit anchor) / 2^7 and 2^-6 (spread 26).
enum Scale : int { SC_UNIT = 0, SC_LO = 1, SC_HI = 2, SC_MIX = 3, SC_BIND = 4, SC_BIND26 = 5 };
constexpr int NORD = 3;
constexpr int NSAMPLE = 8;
constexpr uint32_t TAG = 0x5A000000u;

struct Grid {
  int n;
  uint8_t code[40];
  int32_t val[40];  // the value in units of 2^-s (the grid's resolution)
  uint8_t neg[40];  // the index of -val
};

struct ScaleSet {
  int n;              // 1 for a fixed scale; the set size for MIX; 2 for BIND
  int mode;           // 0 fixed (index 0); 1 drawn per block; 2 alternating (block & 1)
  uint8_t code[8];    // the UE4M3 / UE8M0 byte
  int num[8], exp[8]; // the scale value is num * 2^exp (UE4M3 odd: num odd, exp 0; UE8M0 2^k: num 1, exp k)
};

struct Params {
  Grid ga1, ga2, gb1, gb2;  // k < ksplit draws from *1, else *2
  int K, tiles, ksplit, cancel, s_tot, nrec;
  int sh;                   // the reference's unit is 2^-(s_tot + sh): sh carries the blocks' most negative scale
  int scblk;                // elements a scale block (16 UE4M3, 32 UE8M0; K when there is no block scale)
  int sblk_chunks;          // chunks of 8 a scale block (scblk / 8): the orders permute whole scale blocks
  ScaleSet sa, sb;          // the scale set of each operand
  uint32_t c0;
  long long c0_units;  // C0 in units of 2^-(s_tot + sh)
  unsigned long long seed;
  uint32_t pmul[NORD], padd[NORD];  // order o: physical block p holds logical block (pmul p + padd) mod (K / scblk)
};

struct Sample {
  int tile, row, col, rep_all;
  uint32_t d[NORD], expect;
  long long ref;
};

struct Counts {
  // rz_final, rn_final: of the outputs with an unrepresentable partial, those that are RZ / RN of the exact final sum
  unsigned long long out, exact, rep_all, bad_rep, unrep, bad_unrep, rz_final, rn_final, zero, negzero, order_diff;
  unsigned long long exact_ord[NORD];
  unsigned long long max_partial;  // the largest |partial sum| at an MMA boundary, in units
  unsigned int nsample, tag;
  Sample sample[NSAMPLE];
};

struct Rec {
  uint32_t d[NORD], flags;  // flags: TAG | bit0 every partial representable | bit1 final representable
  long long ref;
};

__host__ __device__ inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__host__ __device__ inline uint64_t hkey(const Params& P, int tile, int mat, int rc, int k) {
  return mix64(P.seed ^ (((uint64_t)tile << 32) | ((uint64_t)mat << 30) | ((uint64_t)rc << 24) | (uint64_t)k));
}

__host__ __device__ inline int pick(int n, uint64_t h) { return (int)(((h >> 32) * (uint64_t)n) >> 32); }

// 2:4 sparse A: each logical chunk of 8 keeps two of its four element pairs, p0 < p1 (patterns 0..5: 01 02 03 12 13 23).
__host__ __device__ inline int pat0(int p) { return (0x211000 >> (4 * p)) & 0xF; }
__host__ __device__ inline int pat1(int p) { return (0x332321 >> (4 * p)) & 0xF; }

struct El {
  int code, val;
};

__host__ __device__ inline int pattern(const Params& P, int tile, int row, int lchunk) {
  return pick(6, hkey(P, tile, 2, row, lchunk));
}

template <bool SPARSE>
__host__ __device__ inline El elA(const Params& P, int tile, int row, int k) {
  if (SPARSE) {
    const int pat = pattern(P, tile, row, k >> 3), pair = (k & 7) >> 1;
    if (pair != pat0(pat) && pair != pat1(pat)) return {0, 0};
  }
  const Grid& g = k < P.ksplit ? P.ga1 : P.ga2;
  const int i = pick(g.n, hkey(P, tile, 0, row, P.cancel ? (k & ~1) : k));
  return {g.code[i], g.val[i]};
}

__host__ __device__ inline El elB(const Params& P, int tile, int col, int k) {
  const Grid& g = k < P.ksplit ? P.gb1 : P.gb2;
  int i = pick(g.n, hkey(P, tile, 1, col, P.cancel ? (k & ~1) : k));
  if (P.cancel && (k & 1)) i = g.neg[i];
  return {g.code[i], g.val[i]};
}

// Order o permutes whole scale blocks (sblk_chunks chunks each), so a block's scale travels with its elements, and only
// the block order changes. Within a block the chunk offset is kept.
__host__ __device__ inline int lchunk(const Params& P, int o, int pchunk) {
  const int u = P.sblk_chunks, nblk = (P.K >> 3) / u;
  const int m = pchunk / u, r = pchunk % u;
  const int mm = (int)((P.pmul[o] * (uint32_t)m + P.padd[o]) & (uint32_t)(nblk - 1));
  return mm * u + r;
}
__host__ __device__ inline int logical(const Params& P, int o, int p) { return (lchunk(P, o, p >> 3) << 3) | (p & 7); }

// The scale of a logical block: op 0 is A (rc = row), op 1 is B (rc = col).
__host__ __device__ inline uint64_t skey(const Params& P, int tile, int op, int rc, int block) {
  return mix64(P.seed ^ 0x5CA1E5CA1E000000ull ^ ((uint64_t)tile << 40) ^ ((uint64_t)op << 38) ^ ((uint64_t)rc << 30) ^
               (uint64_t)block);
}
__host__ __device__ inline int scale_idx(const ScaleSet& s, const Params& P, int tile, int op, int rc, int block) {
  return s.mode == 0 ? 0 : s.mode == 2 ? (block & 1) : pick(s.n, skey(P, tile, op, rc, block));
}

// ---- the MMAs (mnemonics and fragment layouts as benchmarks/tc_probe_fp4/mma_fp4.cu's) ----

__device__ __forceinline__ void mma_e4m3(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_e4m3_kind(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_mxf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_mxf4nv(float* d, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_nvf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_sp_mxf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t e, uint32_t sa,
                                            uint32_t sb) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1."
      "f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%14}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e), "r"(sa),
        "r"(sb));
}
__device__ __forceinline__ void mma_sp_nvf4(float* d, const uint32_t* a, const uint32_t* b, uint32_t e, uint32_t sa,
                                            uint32_t sb) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1."
      "f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0, {%13}, {0, 0}, {%14}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e), "r"(sa),
        "r"(sb));
}

template <int F>
__host__ __device__ constexpr int kstep() {
  return F == E4M3 || F == E4M3_KIND ? 32 : F == SP_MXF4 || F == SP_NVF4 ? 128 : 64;
}

// Unit scales on every block: UE8M0 2^0 is 0x7F, UE4M3 1.0 is 0x38.
template <int F>
__host__ __device__ constexpr uint32_t unit_scale() {
  return F == NVF4 || F == SP_NVF4 ? 0x38383838u : 0x7F7F7F7Fu;
}

// The scale fragment of one firing (tc_probe_fp4's layout): byte b is K-block b of this lane's row / column. SFA's lane
// supplies row 8(lane&1)+(lane>>2) (thread-id selector 0 broadcasts lane bit 1); SFB's supplies column lane>>2.
template <int F>
__device__ __forceinline__ uint32_t pack_sf(const Params& P, const ScaleSet& s, int tile, int op, int rc, int o, int pb) {
  constexpr int KS = kstep<F>();
  const int nbf = KS / (F == NVF4 || F == SP_NVF4 ? 16 : 32);  // blocks a firing: 4 UE4M3, 2 UE8M0
  uint32_t v = 0;
  for (int b = 0; b < nbf; b++) {
    const int lb = lchunk(P, o, (pb >> 3) + P.sblk_chunks * b) / P.sblk_chunks;
    v |= (uint32_t)s.code[scale_idx(s, P, tile, op, rc, lb)] << (8 * b);
  }
  return v;
}

// One MMA of the chain, at physical K offset pb in order o.
template <int F>
__device__ __forceinline__ void step(const Params& P, int tile, int o, int pb, int lane, float* d) {
  const int g = lane >> 2, t = lane & 3;
  if constexpr (F == E4M3 || F == E4M3_KIND) {
    uint32_t a[4], b[2];
#pragma unroll
    for (int r = 0; r < 4; r++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 4; j++)
        v |= (uint32_t)elA<false>(P, tile, g + 8 * (r & 1), logical(P, o, pb + 4 * t + j + 16 * (r >> 1))).code << (8 * j);
      a[r] = v;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) v |= (uint32_t)elB(P, tile, g, logical(P, o, pb + 4 * t + j + 16 * r)).code << (8 * j);
      b[r] = v;
    }
    if constexpr (F == E4M3) mma_e4m3(d, a, b);
    else mma_e4m3_kind(d, a, b);
  } else if constexpr (F == MXF4 || F == MXF4NV || F == NVF4) {
    uint32_t a[4], b[2];
#pragma unroll
    for (int r = 0; r < 4; r++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 8; j++)
        v |= (uint32_t)(elA<false>(P, tile, g + 8 * (r & 1), logical(P, o, pb + 8 * t + j + 32 * (r >> 1))).code & 0xF)
             << (4 * j);
      a[r] = v;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 8; j++)
        v |= (uint32_t)(elB(P, tile, g, logical(P, o, pb + 8 * t + j + 32 * r)).code & 0xF) << (4 * j);
      b[r] = v;
    }
    const uint32_t sa = pack_sf<F>(P, P.sa, tile, 0, 8 * (lane & 1) + (lane >> 2), o, pb);
    const uint32_t sb = pack_sf<F>(P, P.sb, tile, 1, lane >> 2, o, pb);
    if constexpr (F == MXF4) mma_mxf4(d, a, b, sa, sb);
    else if constexpr (F == MXF4NV) mma_mxf4nv(d, a, b, sa, sb);
    else mma_nvf4(d, a, b, sa, sb);
  } else {
    // Stored A: the dense K = 64 layout over stored columns; stored column sc is logical chunk lc's (sc & 3)-th kept
    // element, pair p0 first. B: reg r, nibble j -> col g, k = 8t + j + 32r. Metadata: lane 4g+q holds row g + 8(q&1),
    // nibble c = physical chunk 8(q>>1) + c's (p1 << 2) | p0.
    const int pc0 = pb >> 3;
    uint32_t a[4], b[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int row = g + 8 * (r & 1);
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int sc = 8 * t + j + 32 * (r >> 1), lc = lchunk(P, o, pc0 + (sc >> 2)), pat = pattern(P, tile, row, lc);
        const int pair = (sc & 3) < 2 ? pat0(pat) : pat1(pat);
        v |= (uint32_t)(elA<true>(P, tile, row, 8 * lc + 2 * pair + (sc & 1)).code & 0xF) << (4 * j);
      }
      a[r] = v;
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 8; j++)
        v |= (uint32_t)(elB(P, tile, g, logical(P, o, pb + 8 * t + j + 32 * r)).code & 0xF) << (4 * j);
      b[r] = v;
    }
    const int q = t, mrow = g + 8 * (q & 1);
    uint32_t e = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int pat = pattern(P, tile, mrow, lchunk(P, o, pc0 + 8 * (q >> 1) + c));
      e |= (uint32_t)((pat1(pat) << 2) | pat0(pat)) << (4 * c);
    }
    constexpr uint32_t s = unit_scale<F>();
    if constexpr (F == SP_MXF4) mma_sp_mxf4(d, a, b, e, s, s);
    else mma_sp_nvf4(d, a, b, e, s, s);
  }
}

__device__ __forceinline__ bool representable(long long v) {
  if (v == 0) return true;
  unsigned long long u = v < 0 ? (unsigned long long)(-v) : (unsigned long long)v;
  u >>= __ffsll((long long)u) - 1;
  return u < (1ull << 24);
}

__device__ __forceinline__ uint32_t scaled_word(float f, int s) { return __float_as_uint(f * __int_as_float((127 - s) << 23)); }

template <int F>
__global__ void __launch_bounds__(128) exact_kernel(Params P, Counts* C, Rec* R) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= P.tiles) return;
  const int tile = warp, g = lane >> 2, t = lane & 3;
  constexpr int KS = kstep<F>();
  constexpr bool SPARSE = F == SP_MXF4 || F == SP_NVF4;
  float d[NORD][4];
  for (int o = 0; o < NORD; o++) {
    float acc[4];
    for (int i = 0; i < 4; i++) acc[i] = __uint_as_float(P.c0);
#pragma unroll 1
    for (int pb = 0; pb < P.K; pb += KS) step<F>(P, tile, o, pb, lane, acc);
    for (int i = 0; i < 4; i++) d[o][i] = acc[i];
  }
  // The exact sums, in order 0, with every partial at an MMA boundary.
  const int rows[2] = {g, g + 8}, cols[2] = {2 * t, 2 * t + 1};
  long long ref[4];
  bool rep[4];
  unsigned long long mx = (unsigned long long)llabs(P.c0_units);
  for (int i = 0; i < 4; i++) ref[i] = P.c0_units, rep[i] = true;
#pragma unroll 1
  for (int p = 0; p < P.K; p++) {
    const int k = logical(P, 0, p), blk = k / P.scblk;
    const int ia0 = scale_idx(P.sa, P, tile, 0, rows[0], blk), ia1 = scale_idx(P.sa, P, tile, 0, rows[1], blk);
    const int ib0 = scale_idx(P.sb, P, tile, 1, cols[0], blk), ib1 = scale_idx(P.sb, P, tile, 1, cols[1], blk);
    const El a0 = elA<SPARSE>(P, tile, rows[0], k), a1 = elA<SPARSE>(P, tile, rows[1], k);
    const El b0 = elB(P, tile, cols[0], k), b1 = elB(P, tile, cols[1], k);
    ref[0] += ((long long)a0.val * b0.val * P.sa.num[ia0] * P.sb.num[ib0]) << (P.sa.exp[ia0] + P.sb.exp[ib0] + P.sh);
    ref[1] += ((long long)a0.val * b1.val * P.sa.num[ia0] * P.sb.num[ib1]) << (P.sa.exp[ia0] + P.sb.exp[ib1] + P.sh);
    ref[2] += ((long long)a1.val * b0.val * P.sa.num[ia1] * P.sb.num[ib0]) << (P.sa.exp[ia1] + P.sb.exp[ib0] + P.sh);
    ref[3] += ((long long)a1.val * b1.val * P.sa.num[ia1] * P.sb.num[ib1]) << (P.sa.exp[ia1] + P.sb.exp[ib1] + P.sh);
    if ((p + 1) % KS == 0)
      for (int i = 0; i < 4; i++) {
        rep[i] = rep[i] && representable(ref[i]);
        const unsigned long long m = (unsigned long long)llabs(ref[i]);
        mx = m > mx ? m : mx;
      }
  }
  unsigned long long c[11 + NORD] = {};
  for (int i = 0; i < 4; i++) {
    const int row = rows[i >> 1], col = cols[i & 1];
    const bool frep = representable(ref[i]);
    const uint32_t want = ref[i] == 0 ? 0u : scaled_word(__ll2float_rn(ref[i]), P.s_tot + P.sh);
    const uint32_t rz = ref[i] == 0 ? 0u : scaled_word(__ll2float_rz(ref[i]), P.s_tot + P.sh);
    const uint32_t w0 = __float_as_uint(d[0][i]);
    const bool ex = frep && w0 == want;
    c[0]++;
    c[1] += ex;
    c[2] += rep[i];
    c[3] += rep[i] && !ex;
    c[4] += !rep[i];
    c[5] += !rep[i] && !ex;
    c[6] += !rep[i] && w0 == rz;
    c[7] += !rep[i] && w0 == want;
    c[8] += ref[i] == 0;
    c[9] += ref[i] == 0 && w0 == 0x80000000u;
    bool diff = false;
    for (int o = 0; o < NORD; o++) {
      const uint32_t w = __float_as_uint(d[o][i]);
      c[11 + o] += frep && w == want;
      diff = diff || w != w0;
    }
    c[10] += diff;
    if (!ex || diff) {
      const unsigned slot = atomicAdd(&C->nsample, 1u);
      if (slot < NSAMPLE) {
        Sample& s = C->sample[slot];
        s.tile = tile, s.row = row, s.col = col, s.rep_all = rep[i], s.expect = want, s.ref = ref[i];
        for (int o = 0; o < NORD; o++) s.d[o] = __float_as_uint(d[o][i]);
      }
    }
    if (tile < P.nrec) {
      Rec& r = R[(size_t)tile * 128 + row * 8 + col];
      for (int o = 0; o < NORD; o++) r.d[o] = __float_as_uint(d[o][i]);
      r.flags = TAG | (rep[i] ? 1u : 0u) | (frep ? 2u : 0u);
      r.ref = ref[i];
    }
  }
  unsigned long long* cp = &C->out;
  for (int j = 0; j < 11; j++) atomicAdd(cp + j, c[j]);
  for (int o = 0; o < NORD; o++) atomicAdd(&C->exact_ord[o], c[11 + o]);
  atomicMax(&C->max_partial, mx);
  if (warp == 0 && lane == 0) C->tag = TAG | (unsigned)F;
}

// ---- host ----

struct GridDef {
  const char* name;
  bool e2m1;
  int s;                  // resolution 2^-s
  std::vector<int> vals;  // in units of 2^-s
};

// Must match fp_exact.py's GRIDS.
static std::vector<GridDef> grids() {
  std::vector<int> i16, h8, q4;
  for (int v = -16; v <= 16; v++) i16.push_back(v), h8.push_back(v), q4.push_back(v);
  return {{"e4m3_int16", false, 0, i16},
          {"e4m3_half8", false, 1, h8},
          {"e4m3_quarter4", false, 2, q4},
          {"e2m1_x2", true, 1, {0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12}},
          {"e2m1_int", true, 1, {0, 2, -2, 4, -4, 6, -6, 8, -8, 12, -12}}};
}

static double e4m3_value(int c) {
  const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  if (e == 15 && m == 7) return NAN;
  const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
  return s ? -v : v;
}

static int e4m3_code(double v) {
  if (v == 0) return 0;
  for (int c = 0; c < 256; c++)
    if (e4m3_value(c) == v) return c;
  std::fprintf(stderr, "fp_exact: %g is not an E4M3 value\n", v);
  std::exit(2);
}

static int e2m1_code(int halves) {
  static const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  const int a = std::abs(halves);
  for (int i = 0; i < 8; i++)
    if (mag[i] == a) return i | (halves < 0 && a ? 8 : 0);
  std::fprintf(stderr, "fp_exact: %d halves is not an E2M1 value\n", halves);
  std::exit(2);
}

static Grid make_grid(const GridDef& gd, const std::vector<int>& vals) {
  Grid g{};
  g.n = (int)vals.size();
  for (int i = 0; i < g.n; i++) {
    g.val[i] = vals[i];
    g.code[i] = (uint8_t)(gd.e2m1 ? e2m1_code(vals[i]) : e4m3_code(std::ldexp((double)vals[i], -gd.s)));
    g.neg[i] = 0xFF;
    for (int j = 0; j < g.n; j++)
      if (vals[j] == -vals[i]) g.neg[i] = (uint8_t)j;
  }
  return g;
}

struct Config {
  std::string name;
  int fmt, grid, dist, scale, K, tiles, nrec;
  uint32_t c0;
  unsigned long long seed;
};

static int ue8m0_code(int k) { return 127 + k; }  // 2^k, k in [-127, 127]

// The scale set of one format and Scale enum (both operands draw from it); the UE4M3 odd integers 1..15 and the UE8M0
// powers 2^k are exact. Returns the set; sh (the extra fraction bits) is -2 * min exponent, clamped at 0.
static ScaleSet make_scale(int fmt, int scale) {
  const bool ue4m3 = fmt == NVF4 || fmt == SP_NVF4, ue8m0 = fmt == MXF4 || fmt == MXF4NV || fmt == SP_MXF4;
  ScaleSet s{};
  auto add = [&](int code, int num, int exp) { s.code[s.n] = (uint8_t)code; s.num[s.n] = num; s.exp[s.n] = exp; s.n++; };
  const int unit = ue8m0 ? 0x7F : 0x38;  // 2^0 / 1.0 (unused but set for E4M3)
  switch (scale) {
    case SC_UNIT: s.mode = 0; add(unit, 1, 0); break;
    case SC_LO: s.mode = 0; ue4m3 ? add(e4m3_code(3), 3, 0) : add(ue8m0_code(-3), 1, -3); break;
    case SC_HI: s.mode = 0; ue4m3 ? add(e4m3_code(15), 15, 0) : add(ue8m0_code(3), 1, 3); break;
    case SC_MIX:
      s.mode = 1;
      if (ue4m3) for (int v = 1; v <= 15; v += 2) add(e4m3_code(v), v, 0);
      else for (int k = -3; k <= 3; k++) add(ue8m0_code(k), 1, k);
      break;
    case SC_BIND: s.mode = 2; add(ue8m0_code(7), 1, 7); add(ue8m0_code(-7), 1, -7); break;
    case SC_BIND26: s.mode = 2; add(ue8m0_code(7), 1, 7); add(ue8m0_code(-6), 1, -6); break;
    default: std::fprintf(stderr, "fp_exact: bad scale %d\n", scale); std::exit(2);
  }
  return s;
}

static Params make_params(const Config& c) {
  const GridDef gd = grids().at(c.grid);
  int mx = 0, mn = 1 << 30;
  for (int v : gd.vals) mx = std::max(mx, std::abs(v)), mn = v ? std::min(mn, std::abs(v)) : mn;
  Params P{};
  const Grid all = make_grid(gd, gd.vals), top = make_grid(gd, {mx}), topsign = make_grid(gd, {mx, -mx}),
             low = make_grid(gd, {mn}), lowsign = make_grid(gd, {mn, -mn});
  P.ga1 = P.ga2 = P.gb1 = P.gb2 = all;
  P.ksplit = c.K;
  if (c.dist == D_MAX) P.ga1 = P.gb1 = top;
  if (c.dist == D_MAXSIGN) P.ga1 = top, P.gb1 = topsign;
  if (c.dist == D_RAMP) P.ga1 = P.gb1 = top, P.ga2 = lowsign, P.gb2 = low, P.ksplit = c.K / 2;
  // RAMP's +-min products sum to an even number a firing; EDGE's 0 or min products make a firing's sum odd half the time.
  if (c.dist == D_EDGE) P.ga1 = P.gb1 = top, P.ga2 = make_grid(gd, {mn, 0}), P.gb2 = low, P.ksplit = c.K / 2;
  P.cancel = c.dist == D_CANCEL;
  P.K = c.K, P.tiles = c.tiles, P.nrec = c.nrec, P.s_tot = 2 * gd.s, P.c0 = c.c0, P.seed = c.seed;
  P.sa = P.sb = make_scale(c.fmt, c.scale);
  int minexp = 0;
  for (int i = 0; i < P.sa.n; i++) minexp = std::min(minexp, P.sa.exp[i]);
  P.sh = minexp < 0 ? -2 * minexp : 0;
  const bool ue4m3 = c.fmt == NVF4 || c.fmt == SP_NVF4, ue8m0 = c.fmt == MXF4 || c.fmt == MXF4NV || c.fmt == SP_MXF4;
  P.scblk = ue4m3 ? 16 : ue8m0 ? 32 : c.K;              // elements a scale block
  P.sblk_chunks = (c.fmt <= E4M3_KIND) ? 1 : P.scblk / 8;  // E4M3 keeps the fine 8-chunk permutation
  float f;
  std::memcpy(&f, &c.c0, 4);
  const double u = std::ldexp((double)f, P.s_tot + P.sh);
  if (u != std::floor(u) || std::fabs(u) > 9e15) {
    std::fprintf(stderr, "fp_exact: %s: C0 0x%08x is not a whole number of units\n", c.name.c_str(), c.c0);
    std::exit(2);
  }
  P.c0_units = (long long)u;
  const uint32_t nblk = (uint32_t)(c.K >> 3) / (uint32_t)P.sblk_chunks;  // scale blocks the orders permute
  P.pmul[0] = 1, P.padd[0] = 0;
  P.pmul[1] = nblk - 1, P.padd[1] = nblk - 1;
  P.pmul[2] = 5, P.padd[2] = nblk / 2 + 3;
  return P;
}

typedef void (*KernelFn)(Params, Counts*, Rec*);
static const KernelFn KERNELS[NFMT] = {exact_kernel<E4M3>,   exact_kernel<E4M3_KIND>, exact_kernel<MXF4>,
                                       exact_kernel<MXF4NV>, exact_kernel<NVF4>,      exact_kernel<SP_MXF4>,
                                       exact_kernel<SP_NVF4>};
static const int KSTEP[NFMT] = {32, 32, 64, 64, 64, 128, 128};

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: fp_exact CONFIGS OUT_JSONL RECORDS_BIN [--control]\n");
    return 2;
  }
  const bool control = argc > 4 && std::string(argv[4]) == "--control";
  std::vector<Config> cfgs;
  {
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      Config c;
      std::string c0;
      ss >> c.name >> c.fmt >> c.grid >> c.dist >> c.scale >> c0 >> c.K >> c.tiles >> c.seed >> c.nrec;
      c.c0 = (uint32_t)std::stoul(c0, nullptr, 16);
      if (!ss || c.fmt < 0 || c.fmt >= NFMT || c.K % KSTEP[c.fmt] || (c.K & (c.K - 1)) || c.K < 32) {
        std::fprintf(stderr, "fp_exact: bad config line: %s\n", line.c_str());
        return 2;
      }
      cfgs.push_back(c);
    }
  }
  if (std::string(argv[2]) == "--dump") {
    // The first config's tile 0 from the host's copy of the generator, no GPU: int32 A[16][K], then B[8][K], logical k.
    const Config& c = cfgs.at(0);
    const Params P = make_params(c);
    const bool sp = c.fmt == SP_MXF4 || c.fmt == SP_NVF4;
    std::vector<int32_t> v;
    for (int r = 0; r < 16; r++)
      for (int k = 0; k < c.K; k++) v.push_back(sp ? elA<true>(P, 0, r, k).val : elA<false>(P, 0, r, k).val);
    for (int n = 0; n < 8; n++)
      for (int k = 0; k < c.K; k++) v.push_back(elB(P, 0, n, k).val);
    FILE* f = std::fopen(argv[3], "wb");
    if (!f) return 2;
    std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
    return 0;
  }
  cudaStream_t s;
  GK_CK(cudaStreamCreate(&s));
  cudaDeviceProp prop;
  GK_CK(cudaGetDeviceProperties(&prop, 0));
  FILE* out = std::fopen(argv[2], "w");
  FILE* rec = std::fopen(argv[3], "wb");
  if (!out || !rec) return 2;
  std::fprintf(out, "{\"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"control\": %s}\n", prop.name, prop.major,
               prop.minor, prop.multiProcessorCount, control ? "true" : "false");
  Counts* dC;
  GK_CK(cudaMalloc(&dC, sizeof(Counts)));
  for (const Config& c : cfgs) {
    const Params P = make_params(c);
    const size_t rbytes = (size_t)std::max(c.nrec, 1) * 128 * sizeof(Rec);
    Rec* dR;
    GK_CK(cudaMalloc(&dR, rbytes));
    gk::poison(dR, rbytes, s);
    GK_CK(cudaMemsetAsync(dC, 0, sizeof(Counts), s));
    gk::poison(&dC->tag, sizeof(unsigned), s);
    cudaEvent_t e0, e1;
    GK_CK(cudaEventCreate(&e0));
    GK_CK(cudaEventCreate(&e1));
    GK_CK(cudaEventRecord(e0, s));
    if (!control) KERNELS[c.fmt]<<<(c.tiles + 3) / 4, 128, 0, s>>>(P, dC, dR);
    GK_CK(cudaGetLastError());
    GK_CK(cudaEventRecord(e1, s));
    Counts h;
    std::vector<Rec> hr(rbytes / sizeof(Rec));
    GK_CK(cudaMemcpyAsync(&h, dC, sizeof(Counts), cudaMemcpyDeviceToHost, s));
    GK_CK(cudaMemcpyAsync(hr.data(), dR, rbytes, cudaMemcpyDeviceToHost, s));
    GK_CK(cudaStreamSynchronize(s));
    float ms = 0;
    GK_CK(cudaEventElapsedTime(&ms, e0, e1));
    std::fwrite(hr.data(), sizeof(Rec), (size_t)c.nrec * 128, rec);
    std::fprintf(out,
                 "{\"name\": \"%s\", \"tag\": %u, \"ms\": %.3f, \"out\": %llu, \"exact\": %llu, \"rep_all\": %llu, "
                 "\"bad_rep\": %llu, \"unrep\": %llu, \"bad_unrep\": %llu, \"rz_final\": %llu, \"rn_final\": %llu, "
                 "\"zero\": %llu, \"negzero\": %llu, \"order_diff\": %llu, \"exact_ord\": [%llu, %llu, %llu], "
                 "\"max_partial\": %llu, \"c0_units\": %lld, \"samples\": [",
                 c.name.c_str(), h.tag, ms, h.out, h.exact, h.rep_all, h.bad_rep, h.unrep, h.bad_unrep, h.rz_final,
                 h.rn_final, h.zero, h.negzero, h.order_diff, h.exact_ord[0], h.exact_ord[1], h.exact_ord[2],
                 h.max_partial, P.c0_units);
    const unsigned ns = std::min<unsigned>(h.tag == (TAG | (unsigned)c.fmt) ? h.nsample : 0, NSAMPLE);
    for (unsigned i = 0; i < ns; i++) {
      const Sample& x = h.sample[i];
      std::fprintf(out,
                   "%s{\"tile\": %d, \"row\": %d, \"col\": %d, \"rep_all\": %d, \"d\": [%u, %u, %u], \"expect\": %u, "
                   "\"ref\": %lld}",
                   i ? ", " : "", x.tile, x.row, x.col, x.rep_all, x.d[0], x.d[1], x.d[2], x.expect, x.ref);
    }
    std::fprintf(out, "], \"nsample\": %u}\n", h.tag == (TAG | (unsigned)c.fmt) ? h.nsample : 0);
    std::fflush(out);
    GK_CK(cudaFree(dR));
    GK_CK(cudaEventDestroy(e0));
    GK_CK(cudaEventDestroy(e1));
  }
  std::fclose(out);
  std::fclose(rec);
  return 0;
}
