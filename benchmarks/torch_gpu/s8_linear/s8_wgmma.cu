// The exact W8A8 linear on the H100 and H200 (sm_90a): y = bf16(RN(RN(f32(X W^T) s_a) s_w)), `ScaledMmS8Coordinate_v1`'s word
// on GEMM_SM90_S8.  X (M x K) and W (N x K) are row-major int8, depth contiguous, read by TMA in boxes of 128 bytes of depth
// with SWIZZLE_128B (the K-major layout a wgmma descriptor with 128-byte swizzling reads) and multiplied by warpgroup MMAs
// (wgmma m64nNk32 .s32.s8.s8) into one s32 accumulator over all of K: exact, as |sum| <= 128 * 128 * K < 2^31 for K < 2^17.
// The epilogue is the Definition's, in its order, on each accumulator: cvt.rn.f32.s32, mul.rn by s_a[row], mul.rn by
// s_w[col], cvt.rn.bf16.f32 (built with -fmad=false and no FTZ flag; nothing is added, so nothing could fuse).
//
// pearl_c_sm90.cu's plain GEMM (benchmarks/pouw/pearl_c_sm90, gemm_fast) with the s8 atom for the E4M3 one.  Persistent and
// warp-specialized: CTA c takes tiles c, c + gridDim.x, ... of the TM x BN tiles (`raster`: groups of `group` tile rows,
// column by column), one CTA an SM (its shared memory), so the grid is the set of SMs the GEMM holds and the rest are free.
// TM / 64 consumer warpgroups, each a 64 x BN wgmma tile (rows 64 wg ..), then a producer warpgroup whose first thread issues
// every TMA into STAGES stages of 128 bytes of depth (X's TM rows, then W's BN), each stage's arrival counted on an mbarrier
// (full) and its release on another (empty: lane 0 of every consumer warp once its warpgroup's MMAs on it are done).  The
// producer runs up to STAGES stages ahead across tiles, so the next tile's operands load under a tile's epilogue; it holds
// 40 registers, the consumers the rest, and it issues nothing on the integer pipe in the main loop but its own counters.
// M, N and K need not be multiples of the tile: a box past the matrix reads zeros (TMA's out-of-bounds fill), which add
// nothing to a sum, and the epilogue stores rows < M and columns < N alone (N even, K a multiple of 16: wgmma.py checks).
// s8mm2_<TM>x<BN> is the same GEMM on clusters of two CTAs (`body`'s CL = 2): the pair shares each W box through TMA
// multicast, a third less traffic from L2 at 128 x 256; nothing in the arithmetic changes.
//
// s8mr_128x<BN>_r<RR> (and s8mrb_*) is s8mm_128x<BN> with a reduction of O = X W^T in its epilogue: Z [M, RR] s32, Z[m, i] =
// sum over j < N and bytes b < 4 of s8(byte b of O[m, j]) R[i, 4 j + b], O's cells read as their four little-endian bytes,
// each a signed int8, times a public R [RR, 4N] int8 (O viewed as int8 [M, 4N] times R^T).  y is the s8mm kernel's, unchanged.
// Each consumer warpgroup multiplies its 64 x BN accumulators, still in registers, by R's 4 BN columns of the tile on the
// tensor cores (wgmma m64n<RR>k32 with A from registers: a thread's s32 accumulator is the four s8 bytes A's register fragment
// holds, so nothing is split), R's columns streamed through the stage ring after the tile's K, and adds the 64 x RR sums into Z
// (s8mr: red.global.add per accumulator; s8mrb: through shared memory, one bulk reduce-add a warpgroup).  Integer addition, so
// Z is exact in any order (mod 2^32).  R is read in the kernel's order (wgmma.py's `kernel_order`): within each 32 bytes of
// depth, the column the register fragment puts each accumulator byte in.  Z is zero at launch.
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#include "s8_acc.h"

#define DEV __device__ __forceinline__

// one launch's arguments, a __grid_constant__ parameter (wgmma.py's `Params`): the tensor maps live in parameter space.
// Aligned to 128 bytes, CUDA 13's CUtensorMap alignment (CUDA 12's is 64), so the layout is the same under either
struct alignas(128) Params {
  CUtensorMap a, b;    // X: M rows of K bytes, boxes of TM rows; W: N rows, boxes of BN rows (wgmma.py's `tmap`)
  const float* sa;     // s_a [M]
  const float* sw;     // s_w [N]
  uint16_t* y;         // y [M, N], bf16 words, row-major
  int M, N, K, group;  // group: tile rows a raster group
  int splits, pad0;    // splits of K (1: none)
  int32_t* ws;         // with splits > 1: [M, N] s32 partial sums, zero between launches
  uint32_t* cnt;       // with splits > 1: a counter a consumer warp of each tile, zero between launches
  uint64_t* clk;       // null, or [4 x CTAs]: thread 0's clock64 and %globaltimer at its CTA's start, then at its end
};
static_assert(offsetof(Params, sa) == 256 && offsetof(Params, M) == 280 && offsetof(Params, ws) == 304 && sizeof(Params) == 384,
              "wgmma.py's Params");
static_assert(offsetof(Params, clk) == 320, "wgmma.py's Params");
// an s8mr launch's: the GEMM's Params, then R's tensor map and Z (wgmma.py's `RParams`)
struct alignas(128) RParams {
  Params p;
  CUtensorMap r;  // R in the kernel's order: RR rows of 4N bytes, boxes of RR rows (wgmma.py's `tmap`)
  int32_t* z;     // Z [M, RR] s32, row-major, zero at launch
};
static_assert(offsetof(RParams, r) == 384 && offsetof(RParams, z) == 512 && sizeof(RParams) == 640, "wgmma.py's RParams");

// ---------------------------------------------------------------- mbarriers, TMA, wgmma (pearl_c_sm90.cu's)
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
// a consumer warp's wait: try_wait's predicate is per thread, so the warp is reconverged before the .aligned wgmma that follow
DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
// box (c0 = byte offset along the depth, c1 = row) of `map` into dst, counted on bar
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
// where a launch reads each W box once (one tile row), W's stream goes first out of L2
DEV uint64_t l2_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DEV void tma_load_hint(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(policy)
               : "memory");
}
// a pair cluster's (s8mm2_*): W's box in halves, each CTA's half multicast into both CTAs' stage at the same offset and
// counted on both CTAs' full barrier there
DEV void tma_load_mc(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask)
               : "memory");
}
DEV void tma_load_mc_hint(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint16_t mask, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5, %6;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask), "l"(policy)
               : "memory");
}
DEV uint32_t cta_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// p.clk's words i, i + 1: this SM's cycle count and the global nanosecond timer
DEV void clock_mark(uint64_t* clk, int i) {
  uint64_t t;
  const uint64_t c = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  clk[4 * blockIdx.x + i] = c;
  clk[4 * blockIdx.x + i + 1] = t;
}
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// the mbarrier at shared offset `bar` in CTA `rank` of the cluster, arrived on with release at cluster scope
DEV void mbar_arrive_at(uint32_t bar, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(r) : "memory");
}
// the producer's wait on a stage both CTAs' consumers release
DEV void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAITC:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra WAITC;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

// registers per thread for the rest of the warpgroup's life (all 4 warps of the warpgroup execute it)
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

// K-major operands with 128-byte swizzling (TMA's SWIZZLE_128B boxes): 8-row groups 1,024 bytes apart, the start advanced 32
// bytes per k32 atom within the 128-byte rows
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}

// d = A B + (sd ? d : 0): one s8 atom (k = 32) of a 64 x N warpgroup tile, s32 accumulators.  Accumulator i of thread x is
// row 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  static_assert(N == 256 || N == 128 || N == 64, "n256, n128 or n64");
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else if constexpr (N == 128)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 " ACC32_S
                 ", %32, %33, p;\n}" : ACC32_R : "l"(da), "l"(db), "r"(sd));
}
// after WG_WAIT: the accumulators are read only from here on (the compiler would otherwise move their readers above the wait)
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// d = A B + (sd ? d : 0): one s8 atom (k = 32) of a 64 x N tile with A from registers, the register fragment of
// mma.m16n8k32's A on each warp's 16 rows: a0 row g, a1 row g + 8, bytes 4 t .. 4 t + 3 of the 32; a2, a3 the same rows,
// bytes 16 + 4 t ..  (g = lane / 4, t = lane % 4)
#define IMMR(NS, ACCS, ACCR, A0, A1, A2, A3, DB, SD)                                                                      \
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %" #SD ", 0;\n wgmma.mma_async.sync.aligned.m64n" #NS "k32.s32.s8.s8 " \
               ACCS ", {%" #A0 ", %" #A1 ", %" #A2 ", %" #A3 "}, %" #DB ", p;\n}"                                       \
               : ACCR : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd))
template <int N> DEV void immr(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd);
template <> DEV void immr<8>(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd) {
  IMMR(8, ACC4_S, ACC4_R, 4, 5, 6, 7, 8, 9);
}
template <> DEV void immr<16>(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd) {
  IMMR(16, ACC8_S, ACC8_R, 8, 9, 10, 11, 12, 13);
}
template <> DEV void immr<32>(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd) {
  IMMR(32, ACC16_S, ACC16_R, 16, 17, 18, 19, 20, 21);
}
template <> DEV void immr<64>(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd) {
  IMMR(64, ACC32_S, ACC32_R, 32, 33, 34, 35, 36, 37);
}
template <> DEV void immr<128>(int32_t* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, uint32_t sd) {
  IMMR(128, ACC64_S, ACC64_R, 64, 65, 66, 67, 68, 69);
}

// the named barrier `id` over `n` threads, and an arrival counting `n` on an mbarrier
DEV void bar_sync(uint32_t id, uint32_t n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV void mbar_arrive_n(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
// `bytes` of shared memory at src added into global memory at dst as s32 words, in L2, by the async proxy
DEV void bulk_add_s32(int32_t* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32 [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// ---------------------------------------------------------------- the GEMM
constexpr int BK = 128, SMEM_BUDGET = 220 * 1024, MAX_STAGES = 12;

template <int TM, int BN> struct Tile {
  static_assert((TM == 64 || TM == 128) && (BN == 64 || BN == 128 || BN == 256), "64 or 128 rows by 64, 128 or 256 columns");
  static constexpr int NWG = TM / 64, NT = 128 * (NWG + 1), NA = BN / 2;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int STAGES = SMEM_BUDGET / STAGE < MAX_STAGES ? SMEM_BUDGET / STAGE : MAX_STAGES;
  // the stages from a 1,024-byte-aligned base (SWIZZLE_128B's period), then full[s] and empty[s]
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES;
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0, "every box 1,024-byte aligned for SWIZZLE_128B");
};

// the reduction's R in the stage ring: a tile's 4 BN bytes of depth in CH chunks of 128 bytes by RR rows (four k32 atoms
// each), Q chunks a stage, RST stages a tile after its K.  s8mrb stages warpgroup w's 64 x RR sums in R stage w, or both
// warpgroups' in the one
template <int TM, int BN, int RR> struct Red {
  using T = Tile<TM, BN>;
  static constexpr int CH = BN / 32, CHUNK = RR * 128;
  static constexpr int Q = (int)T::STAGE / CHUNK < CH ? (int)T::STAGE / CHUNK : CH, RST = (CH + Q - 1) / Q;
  static constexpr bool OWN = RST >= T::NWG;
  static_assert(RR % 8 == 0 && RR <= 128 && CHUNK % 1024 == 0 && Q >= 1, "RR a multiple of 8 up to 128, chunks swizzle-aligned");
  static_assert(RST < T::STAGES, "a tile's R and a stage of the next tile's operands fit the ring");
  static_assert((OWN ? 1 : T::NWG) * 64 * RR * 4 <= (int)T::STAGE, "s8mrb's sums fit the R stages");
};

DEV void st_shared_v2(uint32_t addr, int32_t a, int32_t b) {
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::"r"(addr), "r"(a), "r"(b) : "memory");
}

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// two adjacent columns' words of one row, each the Definition's: cvt.rn.f32.s32, mul.rn by s_a, mul.rn by s_w, cvt.rn.bf16
DEV uint32_t words(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}

// the epilogue's y words of a thread's accumulators in 8-column groups J0 .. J1 - 1, rows `row` and `row` + 8 (SW: groups
// in A's order, `a_order`'s)
template <int J0, int J1, bool SW = false>
DEV void y_store(const int32_t* d, const Params& p, int row, int col0, bool live0, bool live1, float r0, float r1) {
  constexpr int I1 = SW ? 2 : 1, I2 = SW ? 1 : 2;
  uint32_t* y0 = reinterpret_cast<uint32_t*>(p.y + (size_t)row * p.N);
  uint32_t* y1 = reinterpret_cast<uint32_t*>(p.y + (size_t)(row + 8) * p.N);
#pragma unroll
  for (int j = J0; j < J1; j++) {
    const int col = col0 + 8 * j;
    if (col < p.N) {
      const float2 c = *reinterpret_cast<const float2*>(p.sw + col);
      if (live0) y0[col / 2] = words(d[4 * j], d[4 * j + I1], r0, c.x, c.y);
      if (live1) y1[col / 2] = words(d[4 * j + I2], d[4 * j + 3], r1, c.x, c.y);
    }
  }
}

// 8-column groups J0 .. J1 - 1 of a thread's accumulators put in A's order: the group's words 1 (row `row`, column 2 (x % 4)
// + 1) and 2 (row `row` + 8, column 2 (x % 4)) exchanged in their own registers, so each group's four consecutive registers
// are A's fragment for depth 32 (rows g, g + 8, g, g + 8) with no copy
template <int J0, int J1> DEV void a_order(int32_t* d) {
#pragma unroll
  for (int j = J0; j < J1; j++)
    asm volatile("{\n\txor.b32 %0, %0, %1;\n\txor.b32 %1, %1, %0;\n\txor.b32 %0, %0, %1;\n}" : "+r"(d[4 * j + 1]), "+r"(d[4 * j + 2]));
}

// a consumer warp's release of stage `bar`: to its own CTA's producer, and with CL = 2 to both CTAs', each of which writes
// half of W's box into this CTA's stage
template <int CL> DEV void release(uint32_t bar) {
  if constexpr (CL == 1) {
    mbar_arrive(bar);
  } else {
    mbar_arrive_at(bar, 0);
    mbar_arrive_at(bar, 1);
  }
}

// CL = 1: one CTA a tile.  CL = 2 (s8mm2_*, a cluster of two CTAs on M-adjacent tiles of one column of tiles): the unit of
// work is a pair of tile rows; CTA r takes tile row 2 u + r (all zeros past M, from TMA's out-of-bounds fill, and stores
// nothing) and loads half r of W's box, multicast to both, so the pair reads each W box from L2 once.  RR > 0 (s8mr*, one CTA a
// tile, K unsplit): the reduction into Z after each tile's K, its R chunks from rmap (FL 0: red.global.add; 1: bulk add)
template <int TM, int BN, int CL, int RR = 0, int FL = 0>
DEV void body(const Params& p, const CUtensorMap* rmap = nullptr, int32_t* zout = nullptr) {
  using T = Tile<TM, BN>;
  constexpr int NWG = T::NWG, STAGES = T::STAGES, NA = T::NA;
  static_assert(CL == 1 || CL == 2, "a CTA or a pair");
  static_assert(RR == 0 || (CL == 1 && TM == 128), "the reduction runs on 128-row tiles of one CTA");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  const int M = p.M, N = p.N, KB = (p.K + BK - 1) / BK;
  const int tiles_m = (M + TM - 1) / TM, tiles_n = (N + BN - 1) / BN, rows_u = (tiles_m + CL - 1) / CL, units = rows_u * tiles_n;
  const int splits = p.splits, total = units * splits;
  const int tid = threadIdx.x, wg = tid / 128, rank = CL == 1 ? 0 : (int)cta_rank();
  const int first_u = blockIdx.x / CL, step_u = gridDim.x / CL;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), CL * NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // with CL = 2, no multicast or remote arrival reaches a CTA before its barriers are initialized
  if constexpr (CL == 1)
    __syncthreads();
  else
    cluster_sync();
  if (p.clk != nullptr && tid == 0) clock_mark(p.clk, 0);

  if (wg == NWG) {
    // the producer exits rather than returns: a path back would join its register budget with the consumers'
    if constexpr (NWG == 2) regs_dec<40>();
    if (tid == NWG * 128) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      if constexpr (RR > 0) tma_prefetch(rmap);
      const bool once = rows_u == 1;
      const uint64_t first = l2_evict_first();
      constexpr uint32_t B_HALF = BN / CL * BK;
      int s = 0;
      uint32_t ph = 0;
      for (int t = first_u; t < total; t += step_u) {
        int tm, tn;
        raster(t % units, rows_u, tiles_n, p.group, tm, tn);
        tm = tm * CL + rank;
        const int z = t / units, kb1 = (z + 1) * KB / splits;
        for (int kb = z * KB / splits; kb < kb1; kb++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          if constexpr (CL == 1)
            mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          else
            mbar_wait_cluster(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          tma_load(st, &p.a, full, kb * BK, tm * TM);
          const uint32_t bdst = st + T::A_BYTES + rank * B_HALF;
          const int brow = tn * BN + rank * (BN / CL);
          if constexpr (CL == 1) {
            if (once)
              tma_load_hint(bdst, &p.b, full, kb * BK, brow, first);
            else
              tma_load(bdst, &p.b, full, kb * BK, brow);
          } else {
            if (once)
              tma_load_mc_hint(bdst, &p.b, full, kb * BK, brow, 0x3, first);
            else
              tma_load_mc(bdst, &p.b, full, kb * BK, brow, 0x3);
          }
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
        if constexpr (RR > 0) {
          using G = Red<TM, BN, RR>;
#pragma unroll
          for (int i = 0; i < G::RST; i++) {
            const int nq = G::CH - i * G::Q < G::Q ? G::CH - i * G::Q : G::Q;
            const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
            mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
            mbar_expect_tx(full, nq * G::CHUNK);
            for (int q = 0; q < nq; q++) tma_load(st + q * G::CHUNK, rmap, full, tn * 4 * BN + (i * G::Q + q) * 128, 0);
            if (++s == STAGES) { s = 0; ph ^= 1; }
          }
        }
      }
      // the peer's consumers arrive on this CTA's barriers: it stays until every stage it loaded is released by both
      if constexpr (CL == 2)
        for (int i = 0; i < STAGES; i++) {
          mbar_wait_cluster(bars + 8 * (STAGES + s), ph ^ 1);
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
    }
    asm volatile("exit;");
  }
  if constexpr (NWG == 2) regs_inc<232>();

  const int x = tid % 128, lane = tid % 32;
  int s = 0;
  uint32_t ph = 0;
  int32_t d[NA];
  // the reduction's sums, carried across tiles like d: a fresh array would be undefined at each tile's first atom, and the
  // compiler's definitions of it could land between the fence and the atom, serializing the wgmma pipeline
  int32_t zs[RR > 0 ? RR / 2 : 1];
  for (int t = first_u; t < total; t += step_u) {
    int tm, tn;
    const int u = t % units;
    raster(u, rows_u, tiles_n, p.group, tm, tn);
    tm = tm * CL + rank;
    const int z = t / units, kb0 = z * KB / splits, kb1 = (z + 1) * KB / splits;
    // one accumulator over the unit's K (all of it, or split z's segments): each stage's four atoms issued before the stage
    // before it is waited on, which is then released; none in flight past the unit
    for (int kb = kb0; kb < kb1; kb++) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
      WG_FENCE;
      imma<BN>(d, desc_sw128(a0), desc_sw128(b0), kb > kb0);
      imma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
      imma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
      imma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
      WG_COMMIT;
      WG_WAIT(1);
      if (kb > kb0 && lane == 0) release<CL>(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
      if (++s == STAGES) { s = 0; ph ^= 1; }
    }
    WG_WAIT(0);
    acc_fence<NA>(d);
    if (lane == 0) release<CL>(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));

    const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
    const bool live0 = row < M, live1 = row + 8 < M;
    if (RR == 0 && splits > 1) {
      // split K: each split adds its s32 sums into ws (integer addition, so the total is exact in any order); the warp
      // whose split arrives last for its 16 rows reads the whole sum back, leaves ws and its counter zero, and runs the
      // epilogue on it
      int32_t* w0 = p.ws + (size_t)row * N;
      int32_t* w1 = p.ws + (size_t)(row + 8) * N;
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        if (col < N) {
          if (live0) { atomicAdd(w0 + col, d[4 * j]); atomicAdd(w0 + col + 1, d[4 * j + 1]); }
          if (live1) { atomicAdd(w1 + col, d[4 * j + 2]); atomicAdd(w1 + col + 1, d[4 * j + 3]); }
        }
      }
      __threadfence();
      __syncwarp();
      uint32_t* c = p.cnt + (size_t)u * (NWG * 4) + wg * 4 + x / 32;
      uint32_t arrived = 0;
      if (lane == 0) arrived = atomicAdd(c, 1u);
      arrived = __shfl_sync(0xffffffffu, arrived, 0);
      if (arrived != (uint32_t)(splits - 1)) continue;
      __threadfence();
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        if (col < N) {
          if (live0) { d[4 * j] = __ldcg(w0 + col); d[4 * j + 1] = __ldcg(w0 + col + 1); w0[col] = 0; w0[col + 1] = 0; }
          if (live1) { d[4 * j + 2] = __ldcg(w1 + col); d[4 * j + 3] = __ldcg(w1 + col + 1); w1[col] = 0; w1[col + 1] = 0; }
        }
      }
      if (lane == 0) *c = 0;
    }
    const float r0 = live0 ? p.sa[row] : 0.0f, r1 = live1 ? p.sa[row + 8] : 0.0f;
    if constexpr (RR == 0) {
      y_store<0, NA / 4>(d, p, row, col0, live0, live1, r0, r1);
    } else {
      // the reduction: chunk c's four atoms take accumulator groups 4 c .. 4 c + 3, in A's order, as A, then the y words.
      // Between the fence and the wait only atoms and their addresses: a barrier's wait loop there, or a y word computed
      // under the atoms, makes ptxas serialize the wgmma pipeline.  The groups are put in A's order ahead of the R stages'
      // wait loops, whose branches keep ptxas from sinking the exchanges past the fence (it would, within one block, and an
      // exchange there writes the main atoms' accumulators).  Each R stage is released once no atom reads it (s8mrb: once
      // the bulk add has read the sums over it)
      using G = Red<TM, BN, RR>;
      constexpr int CH = G::CH, Q = G::Q, RST = G::RST;
      a_order<0, NA / 4>(d);
      int rs[RST];
#pragma unroll
      for (int i = 0; i < RST; i++) {
        mbar_wait_warp(bars + 8 * s, ph);
        rs[i] = s;
        if (++s == STAGES) { s = 0; ph ^= 1; }
      }
      WG_FENCE;
#pragma unroll
      for (int c = 0; c < CH; c++) {
        const uint32_t a = base + rs[c / Q] * T::STAGE + (c % Q) * G::CHUNK;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int j = 4 * c + jj;
          immr<RR>(zs, d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3], desc_sw128(a + 32 * jj), c + jj > 0);
        }
      }
      WG_COMMIT;
      WG_WAIT(0);
      acc_fence<RR / 2>(zs);
      acc_fence<NA>(d);
      if constexpr (FL == 0) {
#pragma unroll
        for (int i = 0; i < RST; i++)
          if (lane == 0) release<CL>(bars + 8 * (STAGES + rs[i]));
      }
      y_store<0, NA / 4, true>(d, p, row, col0, live0, live1, r0, r1);
      // accumulator i of the 64 x RR sums: row `row` + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
      if constexpr (FL == 0) {
#pragma unroll
        for (int i = 0; i < RR / 2; i++) {
          const int zr = row + 8 * ((i / 2) % 2), zc = 8 * (i / 4) + 2 * (x % 4) + i % 2;
          if (zr < M) atomicAdd(zout + (size_t)zr * RR + zc, zs[i]);
        }
      } else {
        // both warpgroups' atoms are done with R before either writes its sums over it; warpgroup w's 64 rows of Z are
        // contiguous, so one bulk add takes them
        bar_sync(1, NWG * 128);
        const int own = G::OWN ? (wg == 0 ? rs[0] : rs[RST > 1 ? 1 : 0]) : rs[0];
        const uint32_t stg = base + own * T::STAGE + (G::OWN ? 0 : wg * 64 * RR * 4);
        const int lr = 16 * (x / 32) + (x % 32) / 4;
#pragma unroll
        for (int i = 0; i < RR / 2; i += 2) {
          const int zr = lr + 8 * ((i / 2) % 2), zc = 8 * (i / 4) + 2 * (x % 4);
          st_shared_v2(stg + (zr * RR + zc) * 4, zs[i], zs[i + 1]);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        bar_sync(2 + wg, 128);
        if (x == 0) {
          const int r0w = tm * TM + wg * 64, rows = M - r0w < 64 ? M - r0w : 64;
          if (rows > 0) {
            bulk_add_s32(zout + (size_t)r0w * RR, stg, rows * RR * 4);
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          }
#pragma unroll
          for (int i = 0; i < RST; i++) mbar_arrive_n(bars + 8 * (STAGES + rs[i]), 4);
        }
      }
    }
  }
  // s8mrb: the last bulk adds complete before the CTA ends
  if constexpr (RR > 0 && FL == 1)
    if (tid % 128 == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  if (p.clk != nullptr && tid == 0) clock_mark(p.clk, 2);
}

#define S8MM(TM, BN)                                                                                                     \
  extern "C" __global__ void __launch_bounds__(Tile<TM, BN>::NT, 1) s8mm_##TM##x##BN(const __grid_constant__ Params p) { \
    body<TM, BN, 1>(p);                                                                                                  \
  }                                                                                                                      \
  extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(Tile<TM, BN>::NT, 1)                            \
      s8mm2_##TM##x##BN(const __grid_constant__ Params p) {                                                              \
    body<TM, BN, 2>(p);                                                                                                  \
  }
S8MM(128, 256)
S8MM(128, 128)
S8MM(128, 64)
S8MM(64, 256)
S8MM(64, 128)
S8MM(64, 64)

#define S8MR1(BN, RR, FL, NAME)                                                                                          \
  extern "C" __global__ void __launch_bounds__(Tile<128, BN>::NT, 1) NAME##_128x##BN##_r##RR(const __grid_constant__ RParams q) { \
    body<128, BN, 1, RR, FL>(q.p, &q.r, q.z);                                                                            \
  }
#define S8MR(BN, RR) S8MR1(BN, RR, 0, s8mr) S8MR1(BN, RR, 1, s8mrb)
// no s8mr_128x256_r8 or _r16: ptxas puts their groups' exchanges between the fence and the atoms, wherever the source puts
// them, and serializes every wgmma of the kernel (wgmma.py's UNBUILT)
S8MR1(256, 8, 1, s8mrb)
S8MR1(256, 16, 1, s8mrb)
S8MR(256, 32)
S8MR(256, 64)
S8MR(256, 128)
S8MR(128, 8)
S8MR(128, 16)
S8MR(128, 32)
S8MR(128, 64)
S8MR(128, 128)

// RParams' size and offsets, then (Q, RST) of each s8mr tile in S8MR's order (wgmma.py's `red_geometry`)
extern "C" __global__ void s8mr_layout(int* out) {
  out[0] = sizeof(RParams);
  out[1] = offsetof(RParams, r);
  out[2] = offsetof(RParams, z);
  const int geo[] = {Red<128, 256, 8>::Q,  Red<128, 256, 8>::RST,  Red<128, 256, 16>::Q,  Red<128, 256, 16>::RST,
                     Red<128, 256, 32>::Q, Red<128, 256, 32>::RST, Red<128, 256, 64>::Q,  Red<128, 256, 64>::RST,
                     Red<128, 256, 128>::Q, Red<128, 256, 128>::RST, Red<128, 128, 8>::Q, Red<128, 128, 8>::RST,
                     Red<128, 128, 16>::Q, Red<128, 128, 16>::RST, Red<128, 128, 32>::Q, Red<128, 128, 32>::RST,
                     Red<128, 128, 64>::Q, Red<128, 128, 64>::RST, Red<128, 128, 128>::Q, Red<128, 128, 128>::RST};
  for (int i = 0; i < 20; i++) out[3 + i] = geo[i];
}

// what wgmma.py takes from this build, checked when it loads the cubin: Params' size and offsets, then each tile's shared
// memory and stages in S8MM's order
extern "C" __global__ void s8mm_layout(int* out) {
  out[0] = sizeof(Params);
  out[1] = offsetof(Params, sa);
  out[2] = offsetof(Params, M);
  const int smem[] = {Tile<128, 256>::SMEM, Tile<128, 128>::SMEM, Tile<128, 64>::SMEM,
                      Tile<64, 256>::SMEM,  Tile<64, 128>::SMEM,  Tile<64, 64>::SMEM};
  const int stages[] = {Tile<128, 256>::STAGES, Tile<128, 128>::STAGES, Tile<128, 64>::STAGES,
                        Tile<64, 256>::STAGES,  Tile<64, 128>::STAGES,  Tile<64, 64>::STAGES};
  for (int i = 0; i < 6; i++) {
    out[3 + 2 * i] = smem[i];
    out[4 + 2 * i] = stages[i];
  }
}
