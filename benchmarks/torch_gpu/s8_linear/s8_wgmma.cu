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
};
static_assert(offsetof(Params, sa) == 256 && offsetof(Params, M) == 280 && sizeof(Params) == 384, "wgmma.py's Params");

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

template <int TM, int BN> DEV void body(const Params& p) {
  using T = Tile<TM, BN>;
  constexpr int NWG = T::NWG, STAGES = T::STAGES, NA = T::NA;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  const int M = p.M, N = p.N, KB = (p.K + BK - 1) / BK;
  const int tiles_m = (M + TM - 1) / TM, tiles_n = (N + BN - 1) / BN, tiles = tiles_m * tiles_n;
  const int tid = threadIdx.x, wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == NWG) {
    // the producer exits rather than returns: a path back would join its register budget with the consumers'
    if constexpr (NWG == 2) regs_dec<40>();
    if (tid == NWG * 128) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      const bool once = tiles_m == 1;
      const uint64_t first = l2_evict_first();
      int s = 0;
      uint32_t ph = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        int tm, tn;
        raster(t, tiles_m, tiles_n, p.group, tm, tn);
        for (int kb = 0; kb < KB; kb++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          tma_load(st, &p.a, full, kb * BK, tm * TM);
          if (once)
            tma_load_hint(st + T::A_BYTES, &p.b, full, kb * BK, tn * BN, first);
          else
            tma_load(st + T::A_BYTES, &p.b, full, kb * BK, tn * BN);
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
  int32_t d[NA];
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, p.group, tm, tn);
    // one accumulator over all of K: each stage's four atoms issued before the stage before it is waited on, which is then
    // released; none in flight past the tile
    for (int kb = 0; kb < KB; kb++) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
      WG_FENCE;
      imma<BN>(d, desc_sw128(a0), desc_sw128(b0), kb > 0);
      imma<BN>(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
      imma<BN>(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
      imma<BN>(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
      WG_COMMIT;
      WG_WAIT(1);
      if (kb > 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
      if (++s == STAGES) { s = 0; ph ^= 1; }
    }
    WG_WAIT(0);
    acc_fence<NA>(d);
    if (lane == 0) mbar_arrive(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));

    const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
    const bool live0 = row < M, live1 = row + 8 < M;
    const float r0 = live0 ? p.sa[row] : 0.0f, r1 = live1 ? p.sa[row + 8] : 0.0f;
    uint32_t* y0 = reinterpret_cast<uint32_t*>(p.y + (size_t)row * N);
    uint32_t* y1 = reinterpret_cast<uint32_t*>(p.y + (size_t)(row + 8) * N);
#pragma unroll
    for (int j = 0; j < NA / 4; j++) {
      const int col = col0 + 8 * j;
      if (col < N) {
        const float2 c = *reinterpret_cast<const float2*>(p.sw + col);
        if (live0) y0[col / 2] = words(d[4 * j], d[4 * j + 1], r0, c.x, c.y);
        if (live1) y1[col / 2] = words(d[4 * j + 2], d[4 * j + 3], r1, c.x, c.y);
      }
    }
  }
}

#define S8MM(TM, BN)                                                                                                     \
  extern "C" __global__ void __launch_bounds__(Tile<TM, BN>::NT, 1) s8mm_##TM##x##BN(const __grid_constant__ Params p) { \
    body<TM, BN>(p);                                                                                                     \
  }
S8MM(128, 256)
S8MM(128, 128)
S8MM(128, 64)
S8MM(64, 256)
S8MM(64, 128)
S8MM(64, 64)

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
