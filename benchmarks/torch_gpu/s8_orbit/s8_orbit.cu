// One Strassen level of the exact int8 GEMM O = X W^T on the H200 (sm_90a), fused: orbit.py's scheme on the tensor cores.
// X (M x K) is A, W (N x K) is B^T, both row-major int8, depth contiguous; M, N and K are split in halves (M, N even, N / 2
// even, K a multiple of 64).  B's seven forms (orbit.B_FORMS) are made once a weight, wrapped to int8, as one [7, N/2, K/2]
// tensor (orbit_gpu.b_forms).  A's seven forms are made a stage at a time in shared memory: TMA brings X's four blocks
// (A11, A12, A21, A22) of 64 rows x 32 bytes of depth with SWIZZLE_32B, and each consumer warpgroup forms the two-term
// forms its products read (orbit.A_FORMS) with byte-wise wrapped adds, OR-ing each byte's signed overflow into a carry word.
// The same swizzle on every 64 x 32 block makes the forms elementwise in the physical layout.  The seven products are
// wgmma m64nBNk32 .s32.s8.s8 on the wrapped forms, each into its own s32 accumulator over all of K/2; the epilogue
// assembles O's four quadrants (orbit.C_SUMS) and stores them as int32 (y32) or as the exact linear's bf16 words (y).
//
// This kernel makes no carry corrections: its O is X W^T exactly when no form of A or B carries (|x|, |w| <= 63 suffice),
// and X W^T mod 2^8 always.  `flags` counts the carry bits A's forms set, so a launch says whether its O is exact.
// s8orbit_bfix, after it, adds B's carries' corrections (exact when A's forms don't carry).
//
// Persistent, one CTA an SM: CTA c takes tiles c, c + gridDim.x, ... of the (M/2 / 64) x (N/2 / BN) quadrant tiles
// (`raster`).  Warpgroup 0 holds products M1, M2, M3, M6 and warpgroup 1 holds M4, M5, M7 (BN / 2 accumulators a product a
// thread).  Warpgroup 1's first thread also issues every TMA (one 4-D box of X's four blocks, one 3-D box of B's seven forms
// a stage), up to STAGES stages ahead of the consumers, blocking only when the stage it is about to consume isn't issued.
// In the epilogue warpgroup 0 hands M1, M2, M3 to warpgroup 1 through shared memory, XCH accumulators a thread at a time:
// warpgroup 0 stores C22 = M1 - M2 + M3 + M6, warpgroup 1 C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4.
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "s8_orbit_acc.h"

#define DEV __device__ __forceinline__

struct alignas(128) Params {
  CUtensorMap a;              // X as [2 halves of M][M/2 rows][2 halves of K][K/2 bytes], boxes of 32 x 64 x 2 x 2
  CUtensorMap b;              // B's forms [7][N/2][K/2], boxes of 32 x BN x 7
  const float* sa;            // s_a [M] (bf16 output)
  const float* sw;            // s_w [N]
  uint16_t* y;                // [M, N] bf16 words, or null
  int32_t* y32;               // [M, N] int32, or null
  unsigned long long* flags;  // carry bits set by A's forms, added to
  unsigned long long* clk;    // [gridDim.x][4]: clock64 and globaltimer at the CTA's start and end, or null
  int M, N, K, group;
};
static_assert(offsetof(Params, sa) == 256 && offsetof(Params, M) == 304 && sizeof(Params) == 384, "orbit_gpu.py's Params");

DEV uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok;
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load4(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %5}], [%2];" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(0)
      : "memory");
}
DEV void tma_load3(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(0)
      : "memory");
}
DEV void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEV unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

// K-major operands with 32-byte swizzling (TMA's SWIZZLE_32B boxes, 32 bytes of depth a row): 8-row groups 256 bytes apart
DEV uint64_t desc_sw32(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(256 >> 4) << 32) | (3ull << 62);
}

// d = A B + (sd ? d : 0): one s8 atom (k = 32) of a 64 x N warpgroup tile.  Accumulator i of thread x is row
// 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  static_assert(N == 96 || N == 80 || N == 64, "n96, n80 or n64");
  if constexpr (N == 96)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %50, 0;\n wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8 " ACC48_S
                 ", %48, %49, p;\n}" : ACC48_R : "l"(da), "l"(db), "r"(sd));
  else if constexpr (N == 80)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %42, 0;\n wgmma.mma_async.sync.aligned.m64n80k32.s32.s8.s8 " ACC40_S
                 ", %40, %41, p;\n}" : ACC40_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " ACC32_S
                 ", %32, %33, p;\n}" : ACC32_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// byte-wise wrapped a + b and a - b of four int8 lanes; ov's sign bits collect the lanes whose true value leaves int8
constexpr uint32_t HI = 0x80808080u, LO = 0x7f7f7f7fu;
DEV uint32_t addb(uint32_t a, uint32_t b, uint32_t& ov) {
  const uint32_t s = ((a & LO) + (b & LO)) ^ ((a ^ b) & HI);
  ov |= (a ^ s) & ~(a ^ b);
  return s;
}
DEV uint32_t subb(uint32_t a, uint32_t b, uint32_t& ov) {
  const uint32_t s = ((a | HI) - (b & LO)) ^ ((a ^ ~b) & HI);
  ov |= (a ^ b) & (a ^ s);
  return s;
}
DEV uint4 add4(uint4 a, uint4 b, uint32_t& ov) {
  return make_uint4(addb(a.x, b.x, ov), addb(a.y, b.y, ov), addb(a.z, b.z, ov), addb(a.w, b.w, ov));
}
DEV uint4 sub4(uint4 a, uint4 b, uint32_t& ov) {
  return make_uint4(subb(a.x, b.x, ov), subb(a.y, b.y, ov), subb(a.z, b.z, ov), subb(a.w, b.w, ov));
}

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

DEV uint32_t words(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}

// ---------------------------------------------------------------- the orbit
constexpr int BK = 32, TM = 64, XCH = 8, MAX_STAGES = 8, SMEM_LIMIT = 232448, NT = 256;
constexpr uint32_t BLK = TM * BK;  // one 64 x 32 block of A

template <int BN> struct Orbit {
  static constexpr int NA = BN / 2;
  static constexpr uint32_t SRC = 4 * BLK, BF = 7 * BN * BK, FORMS = 5 * BLK, STAGE = SRC + BF + FORMS, TX = SRC + BF;
  static constexpr uint32_t XBYTES = 3 * XCH * 128 * 4;
  static constexpr int STAGES_FIT = (SMEM_LIMIT - 1024 - (int)XBYTES) / ((int)STAGE + 16);
  static constexpr int STAGES = STAGES_FIT < MAX_STAGES ? STAGES_FIT : MAX_STAGES;
  static constexpr int SMEM = 1024 + STAGES * STAGE + XBYTES + 16 * STAGES;
  static_assert(STAGE % 256 == 0 && SMEM <= SMEM_LIMIT && NA % XCH == 0, "stages 256-byte aligned for SWIZZLE_32B");
};
// a stage: X's blocks A11 A12 A21 A22, then B's forms 1..7 (BN rows each), then A's two-term forms F1 F2 F5 F6 F7
enum { A11 = 0, A12 = 1, A21 = 2, A22 = 3 };
enum { F1 = 0, F2 = 1, F5 = 2, F6 = 3, F7 = 4 };

// warpgroup WG's products in accumulator order: WG 0 M1 M2 M3 M6, WG 1 M4 M5 M7 (orbit.py's 0-based j)
template <int WG> constexpr int NPROD = WG == 0 ? 4 : 3;
template <int WG> DEV constexpr int jof(int i) {
  return WG == 0 ? (i == 0 ? 0 : i == 1 ? 1 : i == 2 ? 2 : 5) : (i == 0 ? 3 : i == 1 ? 4 : 6);
}

// A's operand of product j in a stage: a form, or the block itself for M3 (A11) and M4 (A22)
template <int BN> DEV uint32_t a_operand(uint32_t st, int j) {
  using O = Orbit<BN>;
  const uint32_t f = st + O::SRC + O::BF;
  switch (j) {
    case 0: return f + F1 * BLK;
    case 1: return f + F2 * BLK;
    case 2: return st + A11 * BLK;
    case 3: return st + A22 * BLK;
    case 4: return f + F5 * BLK;
    case 5: return f + F6 * BLK;
    default: return f + F7 * BLK;
  }
}

// warpgroup WG's forms of one stage, one 16-byte piece of each 2,048-byte block a thread
template <int BN, int WG> DEV void form(uint32_t st, int x, uint32_t& ov) {
  using O = Orbit<BN>;
  const uint32_t f = st + O::SRC + O::BF, o = 16 * x;
  if constexpr (WG == 0) {
    const uint4 a11 = lds128(st + A11 * BLK + o), a21 = lds128(st + A21 * BLK + o), a22 = lds128(st + A22 * BLK + o);
    sts128(f + F1 * BLK + o, add4(a11, a22, ov));
    sts128(f + F2 * BLK + o, add4(a21, a22, ov));
    sts128(f + F6 * BLK + o, sub4(a21, a11, ov));
  } else {
    const uint4 a11 = lds128(st + A11 * BLK + o), a12 = lds128(st + A12 * BLK + o), a22 = lds128(st + A22 * BLK + o);
    sts128(f + F5 * BLK + o, add4(a11, a12, ov));
    sts128(f + F7 * BLK + o, sub4(a12, a22, ov));
  }
}

template <int BN> DEV void store_pair(const Params& p, int qm, int qn, int r, int c, int32_t v0, int32_t v1) {
  const int hm = p.M / 2, hn = p.N / 2;
  const int row = qm * hm + r, col = qn * hn + c;
  if (p.y32) {
    *reinterpret_cast<int2*>(p.y32 + (size_t)row * p.N + col) = make_int2(v0, v1);
  } else {
    const float2 s = *reinterpret_cast<const float2*>(p.sw + col);
    reinterpret_cast<uint32_t*>(p.y + (size_t)row * p.N)[col / 2] = words(v0, v1, p.sa[row], s.x, s.y);
  }
}

template <int BN, int WG> DEV void consumer(const Params& p, uint32_t base, uint32_t bars, int32_t* xs) {
  using O = Orbit<BN>;
  constexpr int NA = O::NA, NP = NPROD<WG>, STAGES = O::STAGES;
  const int hm = p.M / 2, hn = p.N / 2, KB = (p.K / 2 + BK - 1) / BK;
  const int tiles_m = (hm + TM - 1) / TM, tiles_n = (hn + BN - 1) / BN, total = tiles_m * tiles_n;
  const int x = threadIdx.x % 128, lane = x % 32;
  int s = 0;
  uint32_t ph = 0, ov = 0;
  int32_t d[NP][NA];

  // the loads, in the consumers' order of stages; `load` issues the next one once its slot is empty
  const bool loads = WG == 1 && x == 0;
  int lt = blockIdx.x, lkb = 0, ls = 0, issued = 0, consumed = 0;
  uint32_t lph = 0;
  auto load = [&](bool block) {
    if (lt >= total) return false;
    const uint32_t full = bars + 8 * ls, empty = bars + 8 * (STAGES + ls);
    if (block)
      mbar_wait(empty, lph ^ 1);
    else if (!mbar_test(empty, lph ^ 1))
      return false;
    int ltm, ltn;
    raster(lt, tiles_m, tiles_n, p.group, ltm, ltn);
    const uint32_t st = base + ls * O::STAGE;
    mbar_expect_tx(full, O::TX);
    tma_load4(st, &p.a, full, lkb * BK, ltm * TM);
    tma_load3(st + O::SRC, &p.b, full, lkb * BK, ltn * BN);
    if (++ls == STAGES) { ls = 0; lph ^= 1; }
    if (++lkb == KB) { lkb = 0; lt += gridDim.x; }
    issued++;
    return true;
  };
  if (loads)
    while (load(false)) {}

  for (int t = blockIdx.x; t < total; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, p.group, tm, tn);
    for (int kb = 0; kb < KB; kb++) {
      const uint32_t st = base + s * O::STAGE;
      if (loads)
        while (issued <= consumed) load(true);
      mbar_wait(bars + 8 * s, ph);
      form<BN, WG>(st, x, ov);
      fence_async();
      bar_sync(1 + WG, 128);
      WG_FENCE;
#pragma unroll
      for (int i = 0; i < NP; i++)
        imma<BN>(d[i], desc_sw32(a_operand<BN>(st, jof<WG>(i))), desc_sw32(st + O::SRC + jof<WG>(i) * BN * BK), kb > 0);
      WG_COMMIT;
      WG_WAIT(1);
      if (kb > 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
      if (++s == STAGES) { s = 0; ph ^= 1; }
      consumed++;
      if (loads)
        while (load(false)) {}
    }
    WG_WAIT(0);
#pragma unroll
    for (int i = 0; i < NP; i++) acc_fence<NA>(d[i]);
    if (lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));

    const int r0 = tm * TM + 16 * (x / 32) + (x % 32) / 4, c0 = tn * BN + 2 * (x % 4);
#pragma unroll
    for (int e0 = 0; e0 < NA; e0 += XCH) {
      if constexpr (WG == 0) {
#pragma unroll
        for (int e = 0; e < XCH; e++) {
          xs[(0 * XCH + e) * 128 + x] = d[0][e0 + e];
          xs[(1 * XCH + e) * 128 + x] = d[1][e0 + e];
          xs[(2 * XCH + e) * 128 + x] = d[2][e0 + e];
        }
      }
      bar_sync(3, 256);
      if (loads && e0 == 0)
        while (load(false)) {}
#pragma unroll
      for (int e = 0; e < XCH; e += 2) {
        const int i = e0 + e, r = r0 + 8 * ((i / 2) % 2), c = c0 + 8 * (i / 4);
        if (r < hm && c < hn) {
          if constexpr (WG == 0) {
            store_pair<BN>(p, 1, 1, r, c, d[0][i] - d[1][i] + d[2][i] + d[3][i], d[0][i + 1] - d[1][i + 1] + d[2][i + 1] + d[3][i + 1]);
          } else {
            const int32_t m1a = xs[(0 * XCH + e) * 128 + x], m1b = xs[(0 * XCH + e + 1) * 128 + x];
            const int32_t m2a = xs[(1 * XCH + e) * 128 + x], m2b = xs[(1 * XCH + e + 1) * 128 + x];
            const int32_t m3a = xs[(2 * XCH + e) * 128 + x], m3b = xs[(2 * XCH + e + 1) * 128 + x];
            store_pair<BN>(p, 0, 0, r, c, m1a + d[0][i] - d[1][i] + d[2][i], m1b + d[0][i + 1] - d[1][i + 1] + d[2][i + 1]);
            store_pair<BN>(p, 0, 1, r, c, m3a + d[1][i], m3b + d[1][i + 1]);
            store_pair<BN>(p, 1, 0, r, c, m2a + d[0][i], m2b + d[0][i + 1]);
          }
        }
      }
      bar_sync(4, 256);
    }
  }
  ov &= HI;
  if (ov) atomicAdd(p.flags, (unsigned long long)__popc(ov));
}

template <int BN> DEV void body(const Params& p) {
  using O = Orbit<BN>;
  constexpr int STAGES = O::STAGES;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, xb = base + STAGES * O::STAGE, bars = xb + O::XBYTES;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (p.clk) { p.clk[4 * blockIdx.x] = clock64(); p.clk[4 * blockIdx.x + 1] = globaltimer(); }
  }
  if (tid == 128) {
    tma_prefetch(&p.a);
    tma_prefetch(&p.b);
  }
  __syncthreads();
  int32_t* xs = reinterpret_cast<int32_t*>(smem + (xb - smem_at(smem)));
  if (tid < 128)
    consumer<BN, 0>(p, base, bars, xs);
  else
    consumer<BN, 1>(p, base, bars, xs);
  if (tid == 0 && p.clk) { p.clk[4 * blockIdx.x + 2] = clock64(); p.clk[4 * blockIdx.x + 3] = globaltimer(); }
}

// 256 threads and no producer warp: the register file is split over the SM's four sub-partitions, so a ninth warp would put
// three on one of them and cap every thread at 168 registers
#define ORBIT(BN) \
  extern "C" __global__ void __launch_bounds__(NT, 1) s8orbit_##BN(const __grid_constant__ Params p) { body<BN>(p); }
ORBIT(96)
ORBIT(80)
ORBIT(64)

// ---------------------------------------------------------------- B's carries, after the orbit
// With A's forms exact, O = (the orbit's output) + 256 X~ G^T: X~[r] = X[r] ‖ X[r + M/2] is local row r's pair row (2K
// bytes) and G, made once a weight (orbit_gpu.fix_lists), sums each product's B carries through its A form's blocks and its
// quadrants (orbit.C_SUMS).  List l = q (N/2) + c (quadrant q = 2 qm + qn, column c) holds G's entries (o, g) of output
// column c of quadrant q, o ascending, each o << 8 | (uint8) g; those in chunk h (pair-row bytes [2048 h, 2048 h + 2048))
// are ent[ptr[l (H + 1) + h] .. ptr[l (H + 1) + h + 1]), H = K / 1024.  A CTA takes FR local rows (a lane each) and FC
// columns of every quadrant (FC / 2 lists a warp), stages the rows' pair rows a chunk at a time in shared memory, and adds
// its 4 x FR x FC sums into y32 through shared memory, a row at a time.
struct FixParams {
  const int8_t* x;
  int32_t* y32;
  const int32_t* ptr;
  const uint32_t* ent;
  int M, N, K, pad;
};
static_assert(sizeof(FixParams) == 48, "orbit_gpu.py's FixParams");
constexpr int FR = 32, FC = 128, FCHUNK = 2048, FROW = FCHUNK + 4, FLISTS = 4 * FC / 8;
constexpr int FIX_SMEM = FR * FROW > 4 * FR * (FC + 1) * 4 ? FR * FROW : 4 * FR * (FC + 1) * 4;

extern "C" __global__ void __launch_bounds__(256, 1) s8orbit_bfix(const __grid_constant__ FixParams p) {
  extern __shared__ __align__(16) uint8_t fs[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int hm = p.M / 2, hn = p.N / 2, H = p.K / 1024;
  const int r0 = blockIdx.y * FR, c0 = blockIdx.x * FC;
  int32_t acc[FLISTS];
#pragma unroll
  for (int i = 0; i < FLISTS; i++) acc[i] = 0;
  for (int h = 0; h < H; h++) {
    __syncthreads();
    for (int t = threadIdx.x; t < FR * (FCHUNK / 16); t += 256) {
      const int rr = t / (FCHUNK / 16), o = FCHUNK * h + 16 * (t % (FCHUNK / 16)), r = r0 + rr;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < hm) v = *reinterpret_cast<const uint4*>(o < p.K ? p.x + (size_t)r * p.K + o : p.x + (size_t)(r + hm) * p.K + (o - p.K));
      uint32_t* d = reinterpret_cast<uint32_t*>(fs + rr * FROW + 16 * (t % (FCHUNK / 16)));
      d[0] = v.x, d[1] = v.y, d[2] = v.z, d[3] = v.w;
    }
    __syncthreads();
    const int8_t* row = reinterpret_cast<const int8_t*>(fs + lane * FROW) - FCHUNK * h;
#pragma unroll
    for (int i = 0; i < FLISTS; i++) {
      const int l = warp + 8 * i, c = c0 + l % FC;
      if (c < hn) {
        const int32_t* q = p.ptr + ((l / FC) * hn + c) * (H + 1) + h;
        const int b = __ldg(q), e = __ldg(q + 1);
        int a = acc[i];
        for (int t = b; t < e; t++) {
          const uint32_t v = __ldg(p.ent + t);
          a += (int)(int8_t)(v & 0xff) * (int)row[v >> 8];
        }
        acc[i] = a;
      }
    }
  }
  __syncthreads();
  int32_t* res = reinterpret_cast<int32_t*>(fs);
#pragma unroll
  for (int i = 0; i < FLISTS; i++) {
    const int l = warp + 8 * i;
    res[((l / FC) * FR + lane) * (FC + 1) + l % FC] = acc[i];
  }
  __syncthreads();
  for (int t = threadIdx.x; t < 4 * FR * FC; t += 256) {
    const int cc = t % FC, rr = (t / FC) % FR, q = t / (FC * FR), r = r0 + rr, c = c0 + cc;
    if (r < hm && c < hn) p.y32[(size_t)((q >> 1) * hm + r) * p.N + (q & 1) * hn + c] += 256 * res[(q * FR + rr) * (FC + 1) + cc];
  }
}

// what orbit_gpu.py takes from this build, checked when it loads the cubin: Params' size and offsets, then each tile's
// shared memory and stages in ORBIT's order, then FixParams' size and the correction pass's shared memory
extern "C" __global__ void s8orbit_layout(int* out) {
  out[0] = sizeof(Params);
  out[1] = offsetof(Params, sa);
  out[2] = offsetof(Params, M);
  out[3] = Orbit<96>::SMEM;
  out[4] = Orbit<96>::STAGES;
  out[5] = Orbit<80>::SMEM;
  out[6] = Orbit<80>::STAGES;
  out[7] = Orbit<64>::SMEM;
  out[8] = Orbit<64>::STAGES;
  out[9] = sizeof(FixParams);
  out[10] = FIX_SMEM;
}
