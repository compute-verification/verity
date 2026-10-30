// Pearl-C on the H100 and H200 (sm_90a): the kernels of `pearl-c-h100-v1-h3` (verity/ml/protocols/pouw/schemes/
// pearl_c.py on pearl_c_device.H100, hashing -h3) and the plain FP8 GEMMs, for run.py.  Every kernel takes one Args by value:
// 16 pointers p[] and 16 ints i[].  The products are warpgroup MMAs (wgmma m64nNk32 E4M3 for every FP8 atom, m64nNk16 BF16
// for the peel) on operands in shared memory that TMA fills, within 227 KB of shared memory per block.
//
// Layout (fixture.py's `rowmajor`): every operand is a row-major byte matrix with its depth contiguous, byte k of row r at
// r * depth + k, in natural order along k.  So are A' (m x k) and B~ (n x k), the model's own FP8 layout; E and E' (rows x 32);
// the F lines (k x 32, line t is row t) and their transposes F^T (32 x k, and 128 x k with rows 32-127 zero for G); and the
// peel factors P_A (m x 128 bytes) and P_B (n x 128), 64 BF16 words a row.  TMA loads boxes of 128 bytes of depth with
// SWIZZLE_128B (chunk c of row r at chunk c ^ (r & 7) of its 128-byte row, the pattern repeating every 1,024 bytes), which is
// the K-major layout a wgmma descriptor with 128-byte swizzling reads.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "../pearl_c/hash_h3.cuh"
#include "../pearl_c_sm120/hash_fast.cuh"
// decode's A commitment and tile leaves (run.py's PD_SERVED): the variants the decode shapes take, then their PD_STAMP twins
// (run.py's PD_STAMPED, the profile's stamps); PD_PDL: pd_commit waits at its entry for the launch before it and lets the
// call's forming launch behind it (pdl_entry); PD_SM90: hash_decode.cuh's PD_SM90_FORMS
#define PD_VARIANT_LIST(X) X(851) X(979) X(883) X(1011)
#define PD_PDL
#define PD_SM90
#include "../pearl_c_sm120/hash_decode.cuh"
// A's commitment past decode's rows (run.py's chain "pl", PL_VARIANT)
#include "hash_prefill.cuh"
PL_INSTANCE(851)
#include "acc.h"

// the grouped calls' commitment, A's tree past pd_commit's mode 1, and tile leaves (grouped.py), hash_decode.cuh's bodies
// unchanged: unit u's launch is the Args at p0 + u i0 (bytes), and CTA (x, u) is unit u's CTA x when x is inside its launch
// (pd_commit's rows, pd_tree's one, pd_tiles' tiles or PD_GROUP's groups of them); pd_commit_g and pd_commit_gm take a
// unit's filler rows' digest from its record (pd_commit_t's Z: they are zero rows).  The Args are read through the read-only
// path, which ptxas reloads after PD_LOOP's called producer (pd_pserve_call); a plain copy stays live across that call and
// spills (56 bytes of stack in pd_commit_g_851)
DEV Args pd_unit_at(const Args& g, uint32_t u) {
  const Args* __restrict__ s = reinterpret_cast<const Args*>(g.p[0] + (uint64_t)g.i[0] * u);
  Args a;
  for (int j = 0; j < 16; j++) a.p[j] = __ldg(s->p + j);
  for (int j = 0; j < 16; j++) a.i[j] = __ldg(s->i + j);
  return a;
}
DEV Args pd_unit(const Args& g) { return pd_unit_at(g, blockIdx.y); }
#define PD_GROUPED(V)                                                                                                    \
  extern "C" __global__ void __launch_bounds__(192) pd_commit_g_##V(const Args g) {                                    \
    const Args a = pd_unit(g);                                                                                           \
    if ((int)blockIdx.x < a.i[0]) pd_commit_t<V, true>(a, blockIdx.x);                                                   \
  }                                                                                                                      \
  extern "C" __global__ void __launch_bounds__(128) pd_tree_g_##V(const Args g) { pd_tree_t<V>(pd_unit(g)); }          \
  extern "C" __global__ void __launch_bounds__(32) pd_tiles_g_##V(const Args g) {                                      \
    const Args a = pd_unit(g);                                                                                           \
    static_assert((V & PD_GROUP) == 0, "a CTA a tile");                                                                  \
    if ((int)blockIdx.x < (a.i[0] + 63) / 64 * (a.i[1] / 64)) pd_tiles_t<V>(a);                                          \
  }
PD_GROUPED(851)
// pd_tiles_g with PD_GROUP: CTA (x, u) unit u's CTA x of G tiles of a row block (pd_tiles_V's grid at G, cdiv(tiles a row,
// G) a row block), their folds side by side on one warp: pd_tiles_g at PD_G, pd_tiles_gw at PD_GW, a quarter of the CTAs,
// so a decode call's tile leaves hold fewer of the SMs the next call's pd_commit_gm takes whole
constexpr int PD_GW = 16;
#define PD_GROUPED_TILES(V, G, NAME)                                                                                     \
  extern "C" __global__ void __launch_bounds__(32 * G) NAME##_##V(const Args g) {                                      \
    const Args a = pd_unit(g);                                                                                           \
    static_assert((V & PD_GROUP) != 0, "G tiles a CTA");                                                                 \
    if ((int)blockIdx.x < (a.i[0] + 63) / 64 * ((a.i[1] / 64 + G - 1) / G)) pd_tiles_t<V, G>(a);                         \
  }
PD_GROUPED_TILES(979, PD_G, pd_tiles_g)
PD_GROUPED_TILES(979, PD_GW, pd_tiles_gw)
// pd_commit_g on a 1-D grid of the call's rows: CTA x is row map[x] & 0xFFFF of unit map[x] >> 16, the map (u32s) at p1, in
// the order grouped.py gives (the units' rows apart on the SMs, their filler rows on the CTAs past the SMs)
#define PD_GROUPED_ROWS(V)                                                                                               \
  extern "C" __global__ void __launch_bounds__(192) pd_commit_gm_##V(const Args g) {                                   \
    const uint32_t m = __ldg(reinterpret_cast<const uint32_t*>(g.p[1]) + blockIdx.x);                                    \
    pd_commit_t<V, true>(pd_unit_at(g, m >> 16), (int)(m & 0xFFFFu));                                                    \
  }
PD_GROUPED_ROWS(851)

// ---------------------------------------------------------------- formats (IEEE, nearest even, no FTZ)
DEV float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
DEV uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
DEV uint16_t bmul(uint16_t a, uint16_t b) { return f2bf(__fmul_rn(bf2f(a), bf2f(b))); }
DEV uint16_t bdiv(uint16_t a, uint16_t b) { return f2bf(__fdiv_rn(bf2f(a), bf2f(b))); }
DEV uint8_t e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
// two codes from one cvt.rn.satfinite.e4m3x2.f32 (both operands live): lo in the low byte
DEV uint16_t e4m3x2(float lo, float hi) { return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3); }
DEV float e4m3f(uint8_t c) { return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3))); }
DEV uint16_t e4m3bf(uint8_t c) { return f2bf(e4m3f(c)); }

DEV uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------- mbarriers, TMA, wgmma
DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
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
// the maps are written before the launch (run.py's encode_maps), so a prefetch is all they need
// a launch behind the launch before it on its stream (run.py's PDL_KERNELS): every thread waits for that launch to complete
// before it touches global memory that a launch since the weights' jobs writes, then lets the next launch start behind this
// one; both are no-ops in a plain launch
DEV void pdl_entry() { asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory"); }
// its two halves, for a kernel that reads the launch before it's output late (gemm_body's LATE)
DEV void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEV void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// run.py launches PDL_KERNELS programmatically only from a cubin with this kernel (PDL_MARK), whose PDL_KERNELS wait
extern "C" __global__ void pdl_waits() {}
// and PDL_ENTRY_KERNELS (widen_a, and hash_decode.cuh's pd_commit under PD_PDL) only from one with this (PDL_ENTRY_MARK)
extern "C" __global__ void pdl_waits_entry() {}
// run.py gives pd_commit a call's BF16 rows (p15, i5) only from a cubin with this (BF16_ROWS_MARK), whose pd_commit reads them
extern "C" __global__ void pd_reads_bf16() {}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
// box (c0 = byte offset along the depth, c1 = row) of `map` into dst, counted on bar
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
// the same box under an L2 policy (createpolicy's): where a launch reads each B box once, its B stream goes first out of L2,
// so a decode call's weights (25 to 117 MB at Llama-3.1-8B's linears) don't evict the next call's hashing code and tables
DEV uint64_t l2_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DEV uint64_t l2_evict_normal() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DEV void tma_load_hint(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint64_t policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(policy)
               : "memory");
}
// the box into L2 alone: nothing reaches the CTA, and L2 is where every write lands, so it may run before griddepcontrol.wait
DEV void tma_prefetch_l2(const void* map, int c0, int c1, uint64_t policy) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0, {%1, %2}], %3;" ::"l"(map), "r"(c0),
               "r"(c1), "l"(policy)
               : "memory");
}
// registers per thread for the rest of the warpgroup's life (all 4 warps of the warpgroup execute it)
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

// K-major operands with 128-byte swizzling (TMA's SWIZZLE_128B boxes): 8-row groups 1,024 bytes apart, the start advanced 32
// bytes per k32 atom (k16 BF16 step) within the 128-byte rows
DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// K-major without swizzle: core matrices of 8 rows x 16 bytes, 128 bytes apart along k and 256 between 8-row groups (form_s5's
// k = 32 operands: byte j of row r at (r >> 3) * 256 + (j >> 4) * 128 + (r & 7) * 16 + (j & 15))
DEV uint64_t desc_plain(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
}
DEV uint32_t core(int r, int j) { return (r >> 3) * 256 + (j >> 4) * 128 + (r & 7) * 16 + (j & 15); }

// d = A B + (sd ? d : 0): one E4M3 atom (k = 32) of a 64 x N warpgroup tile.  Accumulator i of thread x is row
// 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> DEV void qmma(float* d, uint64_t da, uint64_t db, uint32_t sd) {
  static_assert(N == 128 || N == 64 || N == 32, "n128, n64 or n32");
  if constexpr (N == 128)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " ACC64_S
                 ", %64, %65, p, 1, 1;\n}" : ACC64_F : "l"(da), "l"(db), "r"(sd));
  else if constexpr (N == 64)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " ACC32_S
                 ", %32, %33, p, 1, 1;\n}" : ACC32_F : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %18, 0;\n wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 " ACC16_S
                 ", %16, %17, p, 1, 1;\n}" : ACC16_F : "l"(da), "l"(db), "r"(sd));
}
// d += A B: one BF16 step (k = 16), the accumulators as qmma's
template <int N> DEV void hmma(float* d, uint64_t da, uint64_t db) {
  static_assert(N == 128 || N == 64, "n128 or n64");
  if constexpr (N == 128)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " ACC64_S
                 ", %64, %65, p, 1, 1, 0, 0;\n}" : ACC64_F : "l"(da), "l"(db), "r"(1));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %34, 0;\n wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " ACC32_S
                 ", %32, %33, p, 1, 1, 0, 0;\n}" : ACC32_F : "l"(da), "l"(db), "r"(1));
}
// after WG_WAIT: the accumulators are read only from here on (the compiler would otherwise move their readers above the wait)
template <int NA> DEV void acc_fence(float* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+f"(d[i])::"memory");
}

// ---------------------------------------------------------------- the GEMM
// A (M x K) and B (N x K) row-major E4M3, by tensor map (p10: Maps, each UINT8, rank 2, boxes of 128 bytes x TM rows for A and
// P_A and x BN rows for B and P_B, SWIZZLE_128B).  One CTA per TM x BN tile, persistent over tiles in groups of `i[3]` tile
// rows: TM / 64 consumer warpgroups (rows 64 wg ..), each a 64 x BN wgmma tile, then a producer warpgroup whose first thread
// issues every TMA into STAGES stages of 128 bytes of depth (A's TM rows, then B's BN), each stage's arrival counted on an
// mbarrier (full) and its release on another (empty: lane 0 of every consumer warp after its warpgroup's wait).  G is the
// promotion period: with G = 4, each stage's four atoms (k = 128, one window) run from +0 (scale-d 0 on the first) and their
// sum is added into the FP32 total by one FADD (RN), windows in ascending k, the total starting at +0; G = 0 is one
// accumulator over all of K (no promotion; the next stage's atoms issued before the last stage's are waited on).  With the
// peel (PEARLC, PEARLC_S), the producer loads P_A's TM and P_B's BN rows (128 bytes each) into the stage after each tile's
// last, which the consumers take for the four k16 BF16 steps of P_A P_B^T into the total.  Modes:
//   PLAIN     y = bf16(total sa sb)
//   CT        stores C~ (FP32)
//   PEARLC    stores C~, then the peel into the total, stores U and y = bf16(U ia ib)
//   PEARLC_S  PEARLC without storing C~ or U (only y): the call without hashing
// GR (the grouped calls, grouped.py): A's rows are many units' one after another, and p12 holds each 64-row block's expert e
// (int32); a tile's B and P_B rows are its expert's, rows e N + tn BN of the experts' stacked B side, and its columns' scales
// cs + e N.  C~, U and y are A's rows by N, as without it.  i4 (live, 0 for all M): rows from live on are filler, whose
// C~, U and y are not stored (their products are still computed, as a wgmma tile's 64 rows are one instruction's).  On
// 64-row tiles with G = 4, each window's atoms are issued, into a second set of accumulators, before the window before it
// is waited on, so its MMAs run while that one's sum is added; the same sums go into the total in the same order.
// WIN (G = 4): a CTA a tile's window kb, its sum from +0 stored to slab kb of p2 (K / 128 slabs of M x N FP32), which
// pa_w adds up as the other modes' total does.
enum { M_PLAIN, M_CT, M_PEARLC, M_PEARLC_S, M_WIN };
// a 64-row tile (one consumer warpgroup) keeps up to MAX_STAGES_64 stages in flight, the others MAX_STAGES
constexpr int BK = 128, PEEL_ROW = 128, SMEM_BUDGET = 200 * 1024, MAX_STAGES = 8, MAX_STAGES_64 = 12;
struct Maps { uint8_t a[128], b[128], pa[128], pb[128]; };

template <int TM, int BN> struct Tile {
  static_assert((TM == 64 || TM == 128) && (BN == 32 || BN == 64 || BN == 128), "64 or 128 rows by 32, 64 or 128 columns");
  static constexpr int NWG = TM / 64, NT = 128 * (NWG + 1), CAP = NWG == 1 ? MAX_STAGES_64 : MAX_STAGES;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int STAGES = SMEM_BUDGET / STAGE < CAP ? SMEM_BUDGET / STAGE : CAP;
  // the stages from a 1,024-byte-aligned base (SWIZZLE_128B's period), then full[s] and empty[s]
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES;
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0, "every box 1,024-byte aligned for SWIZZLE_128B");
};

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

DEV void store2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
DEV void store_y(uint16_t* p, float a, float b, float sr, float sc0, float sc1) {
  *reinterpret_cast<uint32_t*>(p) = f2bf(__fmul_rn(__fmul_rn(a, sr), sc0)) | (uint32_t)f2bf(__fmul_rn(__fmul_rn(b, sr), sc1)) << 16;
}

template <int MODE, int TM, int BN, int G, bool GR = false> DEV void gemm_body(const Args& a) {
  using T = Tile<TM, BN>;
  constexpr int NWG = T::NWG, STAGES = T::STAGES, NA = BN / 2;
  constexpr bool PEEL = MODE == M_PEARLC || MODE == M_PEARLC_S;
  constexpr bool STORE_C = MODE == M_CT || MODE == M_PEARLC, STORE_U = MODE == M_PEARLC;
  static_assert(G == 0 || G == BK / 32, "a promotion window is one stage");
  static_assert(!PEEL || BN >= 64, "the peel's wgmma is n64 or n128");
  static_assert(MODE != M_WIN || G != 0, "a window's sum is a promotion window's");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (sa(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  const int M = a.i[0], N = a.i[1], K = a.i[2], group = a.i[3], live = a.i[4] ? a.i[4] : M;
  const int KB = K / BK, wins = MODE == M_WIN ? KB : 1;
  const int tiles_m = M / TM, tiles_n = N / BN, tiles = tiles_m * tiles_n * wins;
  const int tid = threadIdx.x, wg = tid / 128;
  const int* ex = reinterpret_cast<const int*>(a.p[12]);
  const auto b_row = [&](int tm) { return GR ? ex[tm * (TM / 64)] * N : 0; };
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // one row block: every B box is read once, by one tile
  const bool once = tiles_m == 1;
  const uint64_t first = l2_evict_first();
  // under PDL the launch before this one may still be running: the first tile's first STAGES B boxes start toward L2 meanwhile
  // (B is the weights, and the maps are written before the launch; GR's rows wait, as they read p12)
  if (!GR && tid == NWG * 128 && (int)blockIdx.x < tiles) {
    const Maps* maps = reinterpret_cast<const Maps*>(a.p[10]);
    int tm, tn;
    raster(blockIdx.x / wins, tiles_m, tiles_n, group, tm, tn);
    const int k0 = MODE == M_WIN ? blockIdx.x % wins : 0, k1 = MODE == M_WIN ? k0 + 1 : min(KB, STAGES);
    for (int kb = k0; kb < k1; kb++) tma_prefetch_l2(maps->b, kb * BK, tn * BN, once ? first : l2_evict_normal());
  }
  // LATE: of the launch before it (pa_w, or pa), a Pearl-C GEMM reads only P_A, which its producer alone loads, for each
  // tile's peel; that thread waits there, so the windows run while that launch does.  Everything else it reads (A', B, the
  // scales) launches before that one wrote, and they had completed when it passed its own entry wait (pa_w's pdl_entry; pa
  // lets nothing start early)
  constexpr bool LATE = PEEL && !GR;
  __syncthreads();
  if constexpr (LATE)
    pdl_launch();
  else
    pdl_entry();

  if (wg == NWG) {
    // the producer exits rather than returns: a path back would join its register budget with the consumers'
    if constexpr (NWG == 2) regs_dec<40>();
    if (tid == NWG * 128) {
      const Maps* maps = reinterpret_cast<const Maps*>(a.p[10]);
      tma_prefetch(maps->a);
      tma_prefetch(maps->b);
      if (PEEL) { tma_prefetch(maps->pa); tma_prefetch(maps->pb); }
      int s = 0;
      uint32_t ph = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int tm, tn;
        raster(t / wins, tiles_m, tiles_n, group, tm, tn);
        const int br = b_row(tm) + tn * BN, k0 = MODE == M_WIN ? t % wins : 0, k1 = MODE == M_WIN ? k0 + 1 : KB;
        for (int kb = k0; kb < k1 + PEEL; kb++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          if (kb < k1) {
            tma_load(st, maps->a, full, kb * BK, tm * TM);
            if (once)
              tma_load_hint(st + T::A_BYTES, maps->b, full, kb * BK, br, first);
            else
              tma_load(st + T::A_BYTES, maps->b, full, kb * BK, br);
          } else {
            if constexpr (LATE) pdl_wait();
            tma_load(st, maps->pa, full, 0, tm * TM);
            tma_load(st + T::A_BYTES, maps->pb, full, 0, br);
          }
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
    }
    asm volatile("exit;");
  }
  if constexpr (NWG == 2) regs_inc<232>();

  const int x = tid % 128, lane = tid % 32;
  int s = 0;
  uint32_t ph = 0;
  float d[NA], total[NA];
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    raster(t / wins, tiles_m, tiles_n, group, tm, tn);
    const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
    // y's scales, read before the windows so their latency is under them
    float r0 = 0.0f, r1 = 0.0f, sc[NA / 2];
    if constexpr (MODE == M_PLAIN || PEEL) {
      const float* rs = reinterpret_cast<const float*>(a.p[7]);
      const float* cs = reinterpret_cast<const float*>(a.p[8]) + b_row(tm);
      r0 = rs[row];
      r1 = rs[row + 8];
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        sc[2 * j] = cs[col0 + 8 * j];
        sc[2 * j + 1] = cs[col0 + 8 * j + 1];
      }
    }
    if constexpr (MODE == M_WIN) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
      WG_FENCE;
      qmma<BN>(d, desc_sw128(a0), desc_sw128(b0), 0);
      qmma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
      qmma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
      qmma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
      WG_COMMIT;
      WG_WAIT(0);
      acc_fence<NA>(d);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + s));
      if (++s == STAGES) { s = 0; ph ^= 1; }
      float* W = reinterpret_cast<float*>(a.p[2]) + (size_t)(t % wins) * M * N;
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        store2(W + (size_t)row * N + col, d[4 * j], d[4 * j + 1]);
        store2(W + (size_t)(row + 8) * N + col, d[4 * j + 2], d[4 * j + 3]);
      }
      continue;
    }
    if constexpr (G == 0) {
      for (int kb = 0; kb < KB; kb++) {
        mbar_wait_warp(bars + 8 * s, ph);
        const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
        WG_FENCE;
        qmma<BN>(d, desc_sw128(a0), desc_sw128(b0), kb > 0);
        qmma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
        qmma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
        qmma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
        WG_COMMIT;
        WG_WAIT(1);
        if (kb > 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
        if (++s == STAGES) { s = 0; ph ^= 1; }
      }
      WG_WAIT(0);
      acc_fence<NA>(d);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
#pragma unroll
      for (int i = 0; i < NA; i++) total[i] = d[i];
    } else if constexpr (NWG == 1) {
      // windows kb (into d) and kb + 1 (into e) issued together, d's sum added while e's atoms run; each window's sum still
      // goes into the total alone, in ascending k.  K / 128 is even (K is a multiple of 256).  No wgmma is in flight across
      // the loop's back edge or issued or waited on in a branch, either of which ptxas serializes (C7514, C7518)
      float e[NA];
      const auto issue = [&](float* acc) {
        mbar_wait_warp(bars + 8 * s, ph);
        const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
        acc_fence<NA>(acc);
        WG_FENCE;
        qmma<BN>(acc, desc_sw128(a0), desc_sw128(b0), 0);
        qmma<BN>(acc, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
        qmma<BN>(acc, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
        qmma<BN>(acc, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
        WG_COMMIT;
        acc_fence<NA>(acc);
        if (++s == STAGES) { s = 0; ph ^= 1; }
      };
      const auto take = [&](float* acc, int st) {
        acc_fence<NA>(acc);
        if (lane == 0) mbar_arrive(bars + 8 * (STAGES + st));
#pragma unroll
        for (int i = 0; i < NA; i++) total[i] = __fadd_rn(total[i], acc[i]);
      };
      if (KB % 2) __trap();
#pragma unroll
      for (int i = 0; i < NA; i++) total[i] = 0.0f;
      for (int kb = 0; kb < KB; kb += 2) {
        const int sd = s;
        issue(d);
        const int se = s;
        issue(e);
        WG_WAIT(1);
        take(d, sd);
        WG_WAIT(0);
        take(e, se);
      }
    } else {
#pragma unroll
      for (int i = 0; i < NA; i++) total[i] = 0.0f;
      for (int kb = 0; kb < KB; kb++) {
        mbar_wait_warp(bars + 8 * s, ph);
        const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
        WG_FENCE;
        qmma<BN>(d, desc_sw128(a0), desc_sw128(b0), 0);
        qmma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
        qmma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
        qmma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
        WG_COMMIT;
        WG_WAIT(0);
        acc_fence<NA>(d);
        if (lane == 0) mbar_arrive(bars + 8 * (STAGES + s));
#pragma unroll
        for (int i = 0; i < NA; i++) total[i] = __fadd_rn(total[i], d[i]);
        if (++s == STAGES) { s = 0; ph ^= 1; }
      }
    }

    const bool live0 = row < live, live1 = row + 8 < live;
    float* C = reinterpret_cast<float*>(a.p[2]);
    float* U = reinterpret_cast<float*>(a.p[3]);
    uint16_t* Y = reinterpret_cast<uint16_t*>(a.p[4]);
    if constexpr (MODE == M_PLAIN) {
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        const float c0 = sc[2 * j], c1 = sc[2 * j + 1];
        if (live0) store_y(Y + (size_t)row * N + col, total[4 * j], total[4 * j + 1], r0, c0, c1);
        if (live1) store_y(Y + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3], r1, c0, c1);
      }
    } else if constexpr (STORE_C) {
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        if (live0) store2(C + (size_t)row * N + col, total[4 * j], total[4 * j + 1]);
        if (live1) store2(C + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3]);
      }
    }
    if constexpr (PEEL) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t pa0 = base + s * T::STAGE + wg * 64 * PEEL_ROW, pb0 = base + s * T::STAGE + T::A_BYTES;
      WG_FENCE;
      hmma<BN>(total, desc_sw128(pa0), desc_sw128(pb0));
      hmma<BN>(total, desc_sw128(pa0 + 32), desc_sw128(pb0 + 32));
      hmma<BN>(total, desc_sw128(pa0 + 64), desc_sw128(pb0 + 64));
      hmma<BN>(total, desc_sw128(pa0 + 96), desc_sw128(pb0 + 96));
      WG_COMMIT;
      WG_WAIT(0);
      acc_fence<NA>(total);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + s));
      if (++s == STAGES) { s = 0; ph ^= 1; }
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const int col = col0 + 8 * j;
        const float c0 = sc[2 * j], c1 = sc[2 * j + 1];
        if constexpr (STORE_U) {
          if (live0) store2(U + (size_t)row * N + col, total[4 * j], total[4 * j + 1]);
          if (live1) store2(U + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3]);
        }
        if (live0) store_y(Y + (size_t)row * N + col, total[4 * j], total[4 * j + 1], r0, c0, c1);
        if (live1) store_y(Y + (size_t)(row + 8) * N + col, total[4 * j + 2], total[4 * j + 3], r1, c0, c1);
      }
    }
  }
}

// G4: pearl-c-h100-v1 (pearl_c_device.H100); gemm_fast is the plain GEMM unpromoted.  The 128 x 128 tile; _n64: 128 x 64
// (n a multiple of 64, not 128); _d: 64 x 64 (decode's few rows, m a multiple of 64, not 128); gemm_words: 128 x 32 and
// gemm_words_d: 64 x 32, the 32-column words (A'F_B, B~F_A, G); gemm_words_win_d: gemm_words_d's windows a CTA each (WIN)
#define GEMM(NAME, MODE, TM, BN, G)                                                                                       \
  extern "C" __global__ void __launch_bounds__(Tile<TM, BN>::NT, 1) NAME(const Args a) { gemm_body<MODE, TM, BN, G>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = Tile<TM, BN>::SMEM; }
GEMM(gemm_plain, M_PLAIN, 128, 128, 4)
GEMM(gemm_fast, M_PLAIN, 128, 128, 0)
GEMM(gemm_ct, M_CT, 128, 128, 4)
GEMM(gemm_pearlc, M_PEARLC, 128, 128, 4)
GEMM(gemm_pearlc_s, M_PEARLC_S, 128, 128, 4)
GEMM(gemm_plain_n64, M_PLAIN, 128, 64, 4)
GEMM(gemm_fast_n64, M_PLAIN, 128, 64, 0)
GEMM(gemm_ct_n64, M_CT, 128, 64, 4)
GEMM(gemm_pearlc_n64, M_PEARLC, 128, 64, 4)
GEMM(gemm_pearlc_s_n64, M_PEARLC_S, 128, 64, 4)
GEMM(gemm_plain_d, M_PLAIN, 64, 64, 4)
GEMM(gemm_fast_d, M_PLAIN, 64, 64, 0)
GEMM(gemm_ct_d, M_CT, 64, 64, 4)
GEMM(gemm_pearlc_d, M_PEARLC, 64, 64, 4)
GEMM(gemm_pearlc_s_d, M_PEARLC_S, 64, 64, 4)
GEMM(gemm_words, M_CT, 128, 32, 4)
GEMM(gemm_words_d, M_CT, 64, 32, 4)
GEMM(gemm_words_win_d, M_WIN, 64, 32, 4)
// the grouped calls' chain and A'F_B (GR), on 64-row tiles: each unit's segment of A is a multiple of 64 rows
#define GEMM_G(NAME, MODE, TM, BN, G)                                                                                     \
  extern "C" __global__ void __launch_bounds__(Tile<TM, BN>::NT, 1) NAME(const Args a) { gemm_body<MODE, TM, BN, G, true>(a); } \
  extern "C" __global__ void NAME##_smem(int* out) { *out = Tile<TM, BN>::SMEM; }
GEMM_G(gemm_pearlc_g_d, M_PEARLC, 64, 64, 4)
GEMM_G(gemm_words_g_d, M_CT, 64, 32, 4)

// ---------------------------------------------------------------- noise lines (Pearl's line rule on -h3's H_line)
DEV uint32_t isqrt(uint32_t v) {
  uint32_t r = (uint32_t)sqrt((double)v);
  while ((uint64_t)r * r > v) r--;
  while ((uint64_t)(r + 1) * (r + 1) <= v) r++;
  return r;
}

// p0 the line key (8 words), p1 codes (count x 32, row-major), p2 transposed (32 x count) or 0, p3 transposed into 128 x count
// (rows 32-127 left as they are) or 0 (the F lines, count = k); i0 side, i1 factor, i2 count, i3 bf16(32 x the line's L2
// norm): 0x4600 for Pearl's 256, 0x4400 for 16.  i4 lines keyed per row (E_A, `pearl_c.PearlC.encode`): line t < i4 is line 0
// under its own line key, p4 + 32 t (s512_seeds' p2); the rest, line t under p0.  The raw bytes are H_line's (hash_h3.cuh's
// s512_line_raw, `pearl_c.sample_line` on an S512Seed).
extern "C" __global__ void __launch_bounds__(128) lines(const Args a) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x, count = a.i[2];
  if (line >= count) return;
  const bool own = line < a.i[4];
  uint32_t kw[8], h[8];
  load_key(own ? a.p[4] + (uint64_t)line * 32 : a.p[0], kw);
  s512_line_raw(kw, (uint32_t)a.i[0], (uint32_t)a.i[1], own ? 0u : (uint32_t)line, h);
  int xs[32];
  uint32_t sumsq = 0;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    const uint32_t b = (h[j / 4] >> (8 * (j % 4))) & 0xFF;
    xs[j] = (b >> 7 ? -1 : 1) * (int)((b & 0x7F) + 1);
    sumsq += xs[j] * xs[j];
  }
  const uint16_t scale = bdiv((uint16_t)a.i[3], f2bf((float)isqrt(sumsq * 1024)));
  uint8_t c[32];
#pragma unroll
  for (int j = 0; j < 32; j++) c[j] = e4m3(bf2f(bmul(f2bf((float)xs[j]), scale)));
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; i++) w[i] = c[4 * i] | c[4 * i + 1] << 8 | c[4 * i + 2] << 16 | (uint32_t)c[4 * i + 3] << 24;
  uint4* out = reinterpret_cast<uint4*>(a.p[1] + (size_t)line * 32);
  out[0] = make_uint4(w[0], w[1], w[2], w[3]);
  out[1] = make_uint4(w[4], w[5], w[6], w[7]);
#pragma unroll
  for (int s = 2; s < 4; s++)
    if (a.p[s]) {
      uint8_t* t = reinterpret_cast<uint8_t*>(a.p[s]);
#pragma unroll
      for (int j = 0; j < 32; j++) t[(size_t)j * count + line] = c[j];
    }
}

// ---------------------------------------------------------------- forming v1 (cheap-binding §5)
constexpr uint32_t S5_LINE = 0x3AB504F3;   // f32(sqrt(32) / (256 * 16))
constexpr float S5_FLOOR = 0x1p-32f;

// one warp per FP32 row, one pass: s = max|x|, rho (32 partial FMA chains, chain c over positions 8c + 256i in order of i,
// then the xor butterfly), alpha, beta_row.  Lane l reads x[256i + 4l ..] and x[256i + 128 + 4l ..], so chain c's position is
// the first (c < 16) or second (c >= 16) vector's x of lane 2 (c mod 16).  p0 rows (rows x k words), p1 alpha, p2 beta,
// p3 1/alpha (glue), p4 (s, rho) per row; i0 rows, i1 k (a multiple of 256)
extern "C" __global__ void __launch_bounds__(256) stats_s5(const Args a) {
  const int row = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32, rows = a.i[0], k = a.i[1];
  if (row >= rows) return;
  const float4* x = reinterpret_cast<const float4*>(a.p[0]) + (size_t)row * (k / 4);
  const int src = 2 * (lane & 15);
  float s = 0.0f, part = 0.0f;
#pragma unroll 4
  for (int j = lane; j < k / 4; j += 64) {
    const float4 v = x[j], w = x[j + 32];
    s = fmaxf(s, fmaxf(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))),
                       fmaxf(fmaxf(fabsf(w.x), fabsf(w.y)), fmaxf(fabsf(w.z), fabsf(w.w)))));
    const float lo = __shfl_sync(~0u, v.x, src), hi = __shfl_sync(~0u, w.x, src), e = lane < 16 ? lo : hi;
    part = __fmaf_rn(e, e, part);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s = fmaxf(s, __shfl_xor_sync(~0u, s, o)); part = __fadd_rn(part, __shfl_xor_sync(~0u, part, o)); }
  if (lane) return;
  const float rho = __fsqrt_rn(__fdiv_rn(part, (float)((k + 7) / 8)));
  const float alpha = __fdiv_rn(448.0f, fmaxf(__fadd_rn(s, __fmul_rn(4.0f, rho)), S5_FLOOR));
  reinterpret_cast<float*>(a.p[1])[row] = alpha;
  reinterpret_cast<float*>(a.p[2])[row] = __fmul_rn(__fmul_rn(rho, alpha), __uint_as_float(S5_LINE));
  reinterpret_cast<float*>(a.p[3])[row] = __frcp_rn(alpha);
  reinterpret_cast<float2*>(a.p[4])[row] = make_float2(s, rho);
}

// E' = e4m3(beta_row E) per row.  p0 E codes (rows x 32), p1 beta (f32), p2 out E' codes (rows x 32); i0 rows
extern "C" __global__ void eprime(const Args a) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.i[0]) return;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * 32;
  const float beta = reinterpret_cast<const float*>(a.p[1])[row];
  uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < 32; j++) w[j / 4] |= (uint32_t)e4m3(__fmul_rn(beta, e4m3f(E[j]))) << (8 * (j % 4));
  uint4* out = reinterpret_cast<uint4*>(a.p[2] + (size_t)row * 32);
  out[0] = make_uint4(w[0], w[1], w[2], w[3]);
  out[1] = make_uint4(w[4], w[5], w[6], w[7]);
}

// X~ = e4m3(D), D = the atom with alpha x as its accumulator (scale-d 1) over E' and the F lines: per block 64 rows x 128 of
// depth, one m64n128k32 wgmma (A = the rows' E', B = 128 consecutive F lines, D = alpha x).  p0 rows (FP32), p1 alpha, p2 E'
// codes (rows x 32), p3 F lines (k x 32), p4 out codes (rows x k); i0 rows, i1 k.  grid (k / 128, rows / 64), 128 threads.
// E' and the lines are staged in the descriptor's core matrices (`core`); thread x's accumulators are the alpha x of its
// wgmma positions (qmma's), read as float2 pairs.  The codes are staged at a 144-byte row stride, so a warp's 16-bit stores
// (8 rows by 4 column pairs) hit distinct banks, and leave as whole 128-byte rows.  MODE 0 is form_s5's cast (two codes per
// cvt), 1 form_s5_scalar's (one code per cvt), its gate: both write the same codes.  GR (form_s5_g, the grouped calls): p3
// holds the experts' F lines one after another (k x 32 each) and p5 each 64-row block's expert, whose lines the block takes.
// EP (form_s5_ep, eprime in the same launch): p2 holds the rows' E codes, p5 beta and p6 the E' codes out; each block makes
// its rows' E' as eprime does while staging them, and the blocks at depth 0 store them.  WORDS (form_s5_epw, with EP): each
// block also makes gemm_words_win_d's tile for its rows and window kb = blockIdx.x, A'F_B's: its codes (staged again in core
// matrices, a k32 atom each 2,048 bytes) by the 128 bytes of window kb of p7's 32 rows (the F_B^T lines, 32 x k; a k32 atom
// each 1,024 bytes), four m64n32k32 atoms from +0 (scale-d 0 on the first), stored to slab kb of p8 (k / 128 slabs of rows x
// 32 FP32) as gemm_words_win_d stores it: the same atoms over the same operands, in the same order
constexpr int FORM_ROW = 144;
template <int MODE, bool GR = false, bool EP = false, bool WORDS = false> DEV void form_body(const Args& a) {
  static_assert(!(GR && EP), "p5 is GR's experts or EP's beta");
  static_assert(!WORDS || EP, "the windows are form_s5_ep's");
  __shared__ __align__(128) uint8_t sE[64 * 32], sF[128 * 32];
  __shared__ __align__(16) uint8_t sO[64 * FORM_ROW];
  const int tid = threadIdx.x, t0 = blockIdx.x * 128, r0 = blockIdx.y * 64, k = a.i[1];
  auto stage_f = [&] {
    const uint8_t* F = reinterpret_cast<const uint8_t*>(a.p[3]) +
                       (GR ? (size_t)reinterpret_cast<const int*>(a.p[5])[blockIdx.y] * k * 32 : 0);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int idx = tid + 128 * j, r = idx / 2, c = idx % 2;
      *reinterpret_cast<uint4*>(sF + core(r, 16 * c)) = *reinterpret_cast<const uint4*>(F + (size_t)(t0 + r) * 32 + 16 * c);
    }
  };
  // The F lines (p3) and the windows' F_B^T bytes (p7) are a weight's, written by its job's launches when it is registered and
  // by no launch since, so they are read before the wait, their latency under the launch before (GR's expert map, p5, after).
  // The F_B^T bytes: line n's 16-byte chunk c, a warp's quarter on eight lines (n & 7) of one chunk, so its staging stores hit
  // distinct banks
  uint4 fb[2];
  if constexpr (WORDS) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int idx = tid + 128 * j, n = 8 * (idx / 64) + idx % 8, c = (idx / 8) % 8;
      fb[j] = *reinterpret_cast<const uint4*>(a.p[7] + (size_t)n * k + t0 + 16 * c);
    }
  }
  if constexpr (!GR) stage_f();
  pdl_entry();
  if constexpr (GR) stage_f();
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[2]);
  const size_t eo = (size_t)(r0 + tid / 2) * 32 + 16 * (tid % 2);
  uint4 ev = *reinterpret_cast<const uint4*>(E + eo);
  if constexpr (EP) {
    const float beta = reinterpret_cast<const float*>(a.p[5])[r0 + tid / 2];
    uint32_t w[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t o = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) o |= (uint32_t)e4m3(__fmul_rn(beta, e4m3f((uint8_t)(w[i] >> (8 * j))))) << (8 * j);
      w[i] = o;
    }
    ev = make_uint4(w[0], w[1], w[2], w[3]);
    if (blockIdx.x == 0) *reinterpret_cast<uint4*>(a.p[6] + eo) = ev;
  }
  *reinterpret_cast<uint4*>(sE + core(tid / 2, 16 * (tid % 2))) = ev;
  const float* x = reinterpret_cast<const float*>(a.p[0]);
  const float* alpha = reinterpret_cast<const float*>(a.p[1]);
  const int rA = 16 * (tid / 32) + (tid % 32) / 4, cA = 2 * (tid % 4);
  const float al0 = alpha[r0 + rA], al1 = alpha[r0 + rA + 8];
  const float* x0 = x + (size_t)(r0 + rA) * k + t0 + cA;
  const float* x1 = x0 + (size_t)8 * k;
  float d[64];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const float2 v0 = *reinterpret_cast<const float2*>(x0 + 8 * j), v1 = *reinterpret_cast<const float2*>(x1 + 8 * j);
    d[4 * j] = __fmul_rn(al0, v0.x); d[4 * j + 1] = __fmul_rn(al0, v0.y);
    d[4 * j + 2] = __fmul_rn(al1, v1.x); d[4 * j + 3] = __fmul_rn(al1, v1.y);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the staged operands, written by the generic proxy, to wgmma
  __syncthreads();
  WG_FENCE;
  qmma<128>(d, desc_plain(sa(sE)), desc_plain(sa(sF)), 1);
  WG_COMMIT;
  WG_WAIT(0);
  acc_fence<64>(d);
  uint8_t* o0 = sO + rA * FORM_ROW + cA;
  uint8_t* o1 = o0 + 8 * FORM_ROW;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    if (MODE == 0) {
      *reinterpret_cast<uint16_t*>(o0 + 8 * j) = e4m3x2(d[4 * j], d[4 * j + 1]);
      *reinterpret_cast<uint16_t*>(o1 + 8 * j) = e4m3x2(d[4 * j + 2], d[4 * j + 3]);
    } else {
      *reinterpret_cast<uint16_t*>(o0 + 8 * j) = e4m3(d[4 * j]) | (uint16_t)e4m3(d[4 * j + 1]) << 8;
      *reinterpret_cast<uint16_t*>(o1 + 8 * j) = e4m3(d[4 * j + 2]) | (uint16_t)e4m3(d[4 * j + 3]) << 8;
    }
  }
  __syncthreads();
  float w[16];
  if constexpr (WORDS) {
    // row r's 16-byte chunk c of the codes into atom c / 2's core matrices, a warp's quarter on eight rows (r & 7) of one
    // chunk: distinct banks on both sides.  The atoms run while the codes leave
    __shared__ __align__(128) uint8_t sA[4 * 64 * 32], sB[4 * 32 * 32];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int idx = tid + 128 * j, r = 8 * (idx / 64) + idx % 8, c = (idx / 8) % 8;
      *reinterpret_cast<uint4*>(sA + (c / 2) * 2048 + core(r, 16 * (c % 2))) = *reinterpret_cast<const uint4*>(sO + r * FORM_ROW + 16 * c);
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int idx = tid + 128 * j, n = 8 * (idx / 64) + idx % 8, c = (idx / 8) % 8;
      *reinterpret_cast<uint4*>(sB + (c / 2) * 1024 + core(n, 16 * (c % 2))) = fb[j];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    WG_FENCE;
    qmma<32>(w, desc_plain(sa(sA)), desc_plain(sa(sB)), 0);
    qmma<32>(w, desc_plain(sa(sA + 2048)), desc_plain(sa(sB + 1024)), 1);
    qmma<32>(w, desc_plain(sa(sA + 4096)), desc_plain(sa(sB + 2048)), 1);
    qmma<32>(w, desc_plain(sa(sA + 6144)), desc_plain(sa(sB + 3072)), 1);
    WG_COMMIT;
  }
  uint8_t* out = reinterpret_cast<uint8_t*>(a.p[4]);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = tid + 128 * j, r = idx / 8, c = idx % 8;
    *reinterpret_cast<uint4*>(out + (size_t)(r0 + r) * k + t0 + 16 * c) = *reinterpret_cast<const uint4*>(sO + r * FORM_ROW + 16 * c);
  }
  if constexpr (WORDS) {
    WG_WAIT(0);
    acc_fence<16>(w);
    float* W = reinterpret_cast<float*>(a.p[8]) + (size_t)blockIdx.x * a.i[0] * 32 + (size_t)(r0 + rA) * 32;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      store2(W + cA + 8 * j, w[4 * j], w[4 * j + 1]);
      store2(W + 8 * 32 + cA + 8 * j, w[4 * j + 2], w[4 * j + 3]);
    }
  }
}
extern "C" __global__ void __launch_bounds__(128) form_s5(const Args a) { form_body<0>(a); }
extern "C" __global__ void __launch_bounds__(128) form_s5_scalar(const Args a) { form_body<1>(a); }
extern "C" __global__ void __launch_bounds__(128) form_s5_g(const Args a) { form_body<0, true>(a); }
extern "C" __global__ void __launch_bounds__(128) form_s5_ep(const Args a) { form_body<0, false, true>(a); }
extern "C" __global__ void __launch_bounds__(128) form_s5_epw(const Args a) { form_body<0, false, true, true>(a); }

// the adopted quantizer (the model's FP8 path with PoUW off): codes = e4m3(alpha x) row-major, alpha from stats_s5.
// p0 rows (FP32), p1 alpha, p2 out codes; i0 rows, i1 k.  One thread per 16 elements
extern "C" __global__ void quant_s5(const Args a) {
  const int k = a.i[1];
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (size_t)a.i[0] * (k / 16)) return;
  const int row = (int)(g / (k / 16)), t = (int)(g % (k / 16)) * 16;
  const float al = reinterpret_cast<const float*>(a.p[1])[row];
  const float4* x = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(a.p[0]) + (size_t)row * k + t);
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 v = x[i];
    o[i] = e4m3(__fmul_rn(al, v.x)) | e4m3(__fmul_rn(al, v.y)) << 8 | e4m3(__fmul_rn(al, v.z)) << 16 | (uint32_t)e4m3(__fmul_rn(al, v.w)) << 24;
  }
  *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.p[2]) + (size_t)row * k + t) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ---------------------------------------------------------------- the widening
// The one step from B's stored form to the FP32 rows the epoch commits (run.py's xb); everything after it reads those rows.
// widen_bf16: BF16 words (rows x k) to FP32, exact.  widen_fp8_block: E4M3 codes (rows x k) times their 128 x 128 block's
// FP32 scale (a checkpoint's weight_scale_inv: ceil(rows / 128) x ceil(k / 128), row-major), each product RN to FP32
// (`pearl_c_sm90.run.widen`'s choice "fp8_block": a defined B, not the exact real one when code x scale doesn't fit FP32).
// p0 source, p1 scales (widen_fp8_block), p2 out (rows x k FP32); i0 rows, i1 k (a multiple of 8).  A thread per 8 elements.
// widen_a: a served call's A words (pouw_pearl_c_device's xa), widen_bf16's widening of its i0 rows, then rows i0 to i2 - 1
// zero, in one launch; p2 out (i2 x k FP32).  A PDL_ENTRY_KERNELS launch (pdl_entry), so the call's pd_commit, launched
// behind it, is resident when it completes.
DEV void widen8(const uint4* src, uint4* out, size_t g) {
  const uint4 v = src[g];
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  out[2 * g] = make_uint4(w[0] << 16, w[0] & 0xFFFF0000u, w[1] << 16, w[1] & 0xFFFF0000u);
  out[2 * g + 1] = make_uint4(w[2] << 16, w[2] & 0xFFFF0000u, w[3] << 16, w[3] & 0xFFFF0000u);
}
extern "C" __global__ void widen_bf16(const Args a) {
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g < (size_t)a.i[0] * a.i[1] / 8) widen8(reinterpret_cast<const uint4*>(a.p[0]), reinterpret_cast<uint4*>(a.p[2]), g);
}
extern "C" __global__ void widen_a(const Args a) {
  pdl_entry();
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x, real = (size_t)a.i[0] * a.i[1] / 8;
  uint4* out = reinterpret_cast<uint4*>(a.p[2]);
  if (g < real)
    widen8(reinterpret_cast<const uint4*>(a.p[0]), out, g);
  else if (g < (size_t)a.i[2] * a.i[1] / 8)
    out[2 * g] = out[2 * g + 1] = make_uint4(0, 0, 0, 0);
}
extern "C" __global__ void widen_fp8_block(const Args a) {
  const int rows = a.i[0], k = a.i[1];
  const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= (size_t)rows * k / 8) return;
  const int row = (int)(g / (k / 8)), col = (int)(g % (k / 8)) * 8;
  const float s = reinterpret_cast<const float*>(a.p[1])[(size_t)(row / 128) * ((k + 127) / 128) + col / 128];
  const uint2 v = reinterpret_cast<const uint2*>(a.p[0])[g];
  float o[8];
#pragma unroll
  for (int j = 0; j < 8; j++) o[j] = __fmul_rn(e4m3f((uint8_t)((j < 4 ? v.x : v.y) >> (8 * (j % 4)))), s);
  float4* out = reinterpret_cast<float4*>(a.p[2]) + 2 * g;
  out[0] = make_float4(o[0], o[1], o[2], o[3]);
  out[1] = make_float4(o[4], o[5], o[6], o[7]);
}

// ---------------------------------------------------------------- the plain baseline's quantize and inputs
// one CTA per FP32 row: s = 448 / amax, codes = fp8(x s) row-major, glue scale amax / 448.  p0 X (rows x k), p1 out codes,
// p2 out glue (f32); i0 rows, i1 k
extern "C" __global__ void __launch_bounds__(256) plainq(const Args a) {
  __shared__ float red[8];
  const int row = blockIdx.x, k = a.i[1], tid = threadIdx.x;
  const float4* X = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(a.p[0]) + (size_t)row * k);
  float amax = 0.0f;
  for (int g = tid; g < k / 4; g += 256) {
    const float4 v = X[g];
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
  if (tid % 32 == 0) red[tid / 32] = amax;
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int i = 1; i < 8; i++) amax = fmaxf(amax, red[i]);
  const float s = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
  if (tid == 0) reinterpret_cast<float*>(a.p[2])[row] = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
  uint32_t* out = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(a.p[1]) + (size_t)row * k);
  for (int g = tid; g < k / 4; g += 256) {
    const float4 v = X[g];
    out[g] = e4m3(__fmul_rn(v.x, s)) | e4m3(__fmul_rn(v.y, s)) << 8 | e4m3(__fmul_rn(v.z, s)) << 16 | (uint32_t)e4m3(__fmul_rn(v.w, s)) << 24;
  }
}

// timing inputs for v1: FP32 rows, x = (mix32 byte, -128 -> -127) 2^(-4 - h mod 5) with 16 random low mantissa bits when nonzero.
// p0 out; i0 rows, i1 k, i2 seed
extern "C" __global__ void gen_f32(const Args a) {
  const size_t w = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= (size_t)a.i[0] * a.i[1]) return;
  const uint32_t h = mix32((uint32_t)a.i[2] ^ mix32((uint32_t)w * 2u + 1u)), b = h & 0xFF;
  const int code = (int)(int8_t)(b == 0x80 ? 0x81 : b);
  const float x = __fmul_rn((float)code, __uint_as_float((123u - (h >> 8) % 5) << 23));
  reinterpret_cast<uint32_t*>(a.p[0])[w] = code ? __float_as_uint(x) | (h >> 16) : 0;
}

// ---------------------------------------------------------------- the clean-up's factors (row-major, 128 bytes a row)
// P_A rows: [bf16(E'_A) || RNE_bf16(A'F_B)].  p0 E'_A codes (rows x 32), p2 AF words (rows x 32), p3 out; i0 rows
extern "C" __global__ void pa(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 64) return;
  const int row = t / 64, e = t % 64;
  const uint16_t v = e < 32 ? e4m3bf(reinterpret_cast<const uint8_t*>(a.p[0])[row * 32 + e])
                            : f2bf(reinterpret_cast<const float*>(a.p[2])[row * 32 + e - 32]);
  reinterpret_cast<uint16_t*>(a.p[3])[(size_t)row * 64 + e] = v;
}

// pa from A'F_B's windows (gemm_words_win_d's): each AF word their sum from +0, one FADD (RN) a window in ascending k, as
// gemm_words_d's total, stored to p2 and into P_A as pa's.  p0 E'_A codes, p1 the windows (i1 slabs of rows x 32), p2 AF
// words out, p3 out; i0 rows, i1 windows
extern "C" __global__ void pa_w(const Args a) {
  pdl_entry();
  const int t = blockIdx.x * blockDim.x + threadIdx.x, rows = a.i[0], wins = a.i[1];
  if (t >= rows * 64) return;
  const int row = t / 64, e = t % 64;
  uint16_t v;
  if (e < 32) {
    v = e4m3bf(reinterpret_cast<const uint8_t*>(a.p[0])[row * 32 + e]);
  } else {
    const float* w = reinterpret_cast<const float*>(a.p[1]) + row * 32 + e - 32;
    float total = 0.0f;
#pragma unroll 8
    for (int kb = 0; kb < wins; kb++) total = __fadd_rn(total, w[(size_t)kb * rows * 32]);
    reinterpret_cast<float*>(a.p[2])[row * 32 + e - 32] = total;
    v = f2bf(total);
  }
  reinterpret_cast<uint16_t*>(a.p[3])[(size_t)row * 64 + e] = v;
}

// P_B rows: [RNE_bf16(FP32 add(fma chain from +0 over p of be_p G_pq, -BF_jq)) || -be], be = bf16(E'_B).
// p0 E'_B codes, p2 BF words (rows x 32), p3 G words (32 x 32, row p), p4 out; i0 rows
extern "C" __global__ void pb(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.i[0] * 32) return;
  const int row = t / 32, q = t % 32;
  const uint8_t* E = reinterpret_cast<const uint8_t*>(a.p[0]) + (size_t)row * 32;
  const float* G = reinterpret_cast<const float*>(a.p[3]);
  float acc = 0.0f;
  for (int p = 0; p < 32; p++) acc = __fmaf_rn(bf2f(e4m3bf(E[p])), G[p * 32 + q], acc);
  const uint16_t mid = f2bf(__fadd_rn(acc, -reinterpret_cast<const float*>(a.p[2])[row * 32 + q]));
  uint16_t* out = reinterpret_cast<uint16_t*>(a.p[4]) + (size_t)row * 64;
  out[q] = mid;
  out[32 + q] = e4m3bf(E[q]) ^ 0x8000;
}

// words differing between two buffers: p0, p1 (u32), p2 counter (u64); i0 count (in words, < 2^31 per launch)
extern "C" __global__ void diff_words(const Args a) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const bool d = i < (size_t)a.i[0] && reinterpret_cast<const uint32_t*>(a.p[0])[i] != reinterpret_cast<const uint32_t*>(a.p[1])[i];
  const unsigned n = __syncthreads_count(d);
  if (threadIdx.x == 0 && n) atomicAdd(reinterpret_cast<unsigned long long*>(a.p[2]), (unsigned long long)n);
}
