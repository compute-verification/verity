// Pearl-C's FP8 mainloop on sm_120a: one tile's chain C~ = A' B~^T into registers, in the fragment layout pearl_c_sm120.cu's
// epilogue reads (a consumer warp per WM x 32 warp tile, total[MI][NI][4] as mma.sync m16n8 accumulators).
//
// The chain is the scheme's, word for word: atom j is one m16n8k32 E4M3 mma.sync over k in [32 j, 32 j + 32), atoms in
// ascending k.  With G = 4, every 4 atoms (one 128-byte stage) start from +0 and their sum is added into the FP32 total by
// one FADD (RN), windows in ascending k, the total starting at +0; with G = 0 there is one accumulator over all of K.
//
// The schedule is CUTLASS's sm_120 warp-specialized mainloop (collective/sm120_mma_tma_blockwise_scaling.hpp): a producer warp
// loads 128 x 128-byte boxes of A and B by TMA (SWIZZLE_128B, so chunk c of row r sits at c ^ (r & 7)) into STAGES stages of
// shared memory, each stage's arrival counted on an mbarrier; the consumer warps wait on it, ldmatrix their fragments one atom
// ahead (the next stage's first atom is loaded before this stage's last atom's MMAs), and release the stage on a second
// mbarrier.  No block-wide barrier.  A window's FADD is issued with the next window's first MMAs, which start from +0 and so
// only overwrite the accumulator once the FADD has read it.  After a tile's last stage the producer loads the peel's rows (P_A,
// P_B) into the next stage (the tail), which the consumers hold through the epilogue and release when it is done.
//
// Host side: four CUtensorMaps (A, B, P_A, P_B), each UINT8, rank 2, globalDim {depth, rows}, box {128, TM} for A and P_A and
// {128, BN} for B and P_B, SWIZZLE_128B, contiguous in global memory (64-byte aligned), passed by pointer.
#pragma once
#include <cstdint>

namespace ml {

#define ML_DEV __device__ __forceinline__

ML_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

ML_DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
ML_DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// a consumer's release of a stage.  Its ldmatrix reads (generic proxy) must be done before the TMA that refills the stage
// (async proxy) writes it, and the arrive alone doesn't order them: without the fence an arrive overtakes an ldmatrix still
// queued, and the refill lands under it (seen on the device as one 16-byte row of a fragment from the next tile).
ML_DEV void mbar_release(uint32_t bar) {
  asm volatile("fence.proxy.async.shared::cta;\n mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
ML_DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n ML_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra ML_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
// a consumer warp's wait: try_wait's predicate is per thread, so the warp is reconverged before the ldmatrix and mma.sync
// (.aligned) that follow
ML_DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
ML_DEV void tma_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(map) : "memory");
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
// box (c0 = byte offset along the depth, c1 = row) of `map` into dst, counted on bar
ML_DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
// registers per thread for the rest of the warpgroup's life (all 4 warps of the warpgroup execute it)
template <int R> ML_DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> ML_DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
ML_DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// d = a b + c, one E4M3 atom (k = 32)
ML_DEV void qmma(float* d, const uint32_t* a, const uint32_t* b, const float* c) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// d = a b + 0
ML_DEV void qmma0(float* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f));
}

// The tile order both roles walk: groups of `group` tile rows, column by column inside a group.
ML_DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

struct Maps { uint8_t a[128], b[128], pa[128], pb[128]; };

// TM x BN tile, one consumer warp per WM x 32 warp tile (NCW; WM = 64: 8 warps at 128 x 128, 4 at 128 x 64; WM = 16: 8 at
// 64 x 64 and 128 x 32), then the producer, whose first lane issues every TMA.  When the consumers are whole warpgroups (SPLIT)
// the producer is a warpgroup and `setmaxnreg` moves its registers to the consumers (REGS_P and REGS_C a thread: RP and what
// the launch leaves, 40 and 232 at 128 x 128 as in CUTLASS, 24 and 240 with RP = 24); otherwise it is one warp and registers
// stay as launched.  Launch NT threads with __launch_bounds__(NT, CTAS), CTAS blocks per SM, and SMEM bytes of dynamic shared
// memory at a 1,024-byte-aligned address (`extern __shared__ __align__(1024)`).  STAGES stages of 128 bytes of depth; G = 4
// or 0; TAIL: the producer loads P_A's TM and P_B's BN rows (128 bytes each) into a stage after each tile's last, for the
// epilogue.
template <int TM, int BN, int G, int STAGES, bool TAIL, int CTAS = 1, int RP = 40, int WM_ROWS = 64> struct Mainloop {
  static constexpr int BK = 128, WM = WM_ROWS, WN = 32, WARPS_M = TM / WM, WARPS_N = BN / WN, NCW = WARPS_M * WARPS_N, NCT = NCW * 32,
                       MI = WM / 16, NI = WN / 8, APS = BK / 32;
  static constexpr bool SPLIT = NCW % 4 == 0;
  static constexpr int NT = NCT + (SPLIT ? 128 : 32), REGS_P = RP, REGS_LAUNCH = 65536 / (CTAS * NT) / 8 * 8,
                       REGS_FIT = (NT * REGS_LAUNCH - 128 * REGS_P) / NCT / 8 * 8,
                       REGS_C = !SPLIT ? 0 : REGS_FIT < 256 ? REGS_FIT : 256;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int SMEM = STAGES * STAGE + 16 * STAGES;
  static_assert(G == 0 || G == APS, "a promotion window is one stage");
  static_assert(WM % 16 == 0 && TM % WM == 0 && BN % WN == 0 && NCW >= 1, "whole WM x 32 warp tiles");
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0, "every box 1,024-byte aligned for SWIZZLE_128B");

  uint32_t base, bars;   // the stages; full[s] at bars + 8 s, empty[s] at bars + 8 (STAGES + s)
  int s = 0;
  uint32_t ph = 0;

  ML_DEV explicit Mainloop(uint8_t* smem) {
    base = smem_u32(smem);
    if (base & 1023) __trap();
    bars = base + STAGES * STAGE;
  }
  ML_DEV uint32_t full(int i) const { return bars + 8 * i; }
  ML_DEV uint32_t empty(int i) const { return bars + 8 * (STAGES + i); }
  ML_DEV uint32_t stage(int i) const { return base + i * STAGE; }
  ML_DEV void advance() {
    if (++s == STAGES) { s = 0; ph ^= 1; }
  }
  // every thread of the block, before the roles split
  ML_DEV void init(int tid) {
    if (tid == 0) {
      for (int i = 0; i < STAGES; i++) { mbar_init(full(i), 1); mbar_init(empty(i), NCT); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
  }
  // every thread of the block: init, then the producer's warps produce every tile's stages and exit; the consumer warps take
  // their registers and fragment offsets and return, to run tile() per tile in the same raster order.  The producer exits
  // rather than returns: a path back to the caller would join the two register budgets, and ptxas would compile the
  // consumers' code under the producer's.
  ML_DEV void roles(const Maps* maps, int M, int N, int K, int group) {
    init(threadIdx.x);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= NCW) {
      if constexpr (SPLIT) regs_dec<REGS_P>();
      if (warp == NCW && lane == 0) produce(maps, M, N, K, group);
      asm volatile("exit;");
    }
    if constexpr (SPLIT) regs_inc<REGS_C>();
    consumer_init(warp, lane);
  }

  // the producer warp's lane 0: every tile's stages (and tails) in the consumers' order
  ML_DEV void produce(const Maps* maps, int M, int N, int K, int group) {
    const int tiles_m = M / TM, tiles_n = N / BN, KT = K / BK;
    tma_acquire(maps->a); tma_acquire(maps->b);
    if (TAIL) { tma_acquire(maps->pa); tma_acquire(maps->pb); }
    for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
      int tm, tn;
      raster(t, tiles_m, tiles_n, group, tm, tn);
      for (int kt = 0; kt < KT; kt++) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        tma_load(stage(s), maps->a, full(s), kt * BK, tm * TM);
        tma_load(stage(s) + A_BYTES, maps->b, full(s), kt * BK, tn * BN);
        advance();
      }
      if (TAIL) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        tma_load(stage(s), maps->pa, full(s), 0, tm * TM);
        tma_load(stage(s) + A_BYTES, maps->pb, full(s), 0, tn * BN);
        advance();
      }
    }
  }

  // ---- the consumers (warps 0-7): one tile's chain into total, acc the chain's scratch (dead after)
  uint32_t a_off[APS], b_off[APS];
  ML_DEV void consumer_init(int warp, int lane) {
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, x = lane & 7;
    const int a_row = wm * WM + (lane & 15), a_half = lane >> 4;
    const int b_row = wn * WN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
#pragma unroll
    for (int kk = 0; kk < APS; kk++) {
      a_off[kk] = a_row * BK + (((2 * kk + a_half) ^ x) << 4);
      b_off[kk] = A_BYTES + b_row * BK + (((2 * kk + b_half) ^ x) << 4);
    }
  }
  ML_DEV void load(uint32_t (&fa)[MI][4], uint32_t (&fb)[NI][2], uint32_t sb, int kk) const {
#pragma unroll
    for (int mi = 0; mi < MI; mi++) ldsm4(fa[mi], sb + a_off[kk] + mi * 16 * BK);
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, sb + b_off[kk] + nj * 16 * BK);
      fb[2 * nj][0] = r[0]; fb[2 * nj][1] = r[1]; fb[2 * nj + 1][0] = r[2]; fb[2 * nj + 1][1] = r[3];
    }
  }
  // one atom's MMAs: FROM0, the window's first atom (from +0), FADD, the previous window's promotion issued with it
  template <bool FROM0, bool FADD>
  ML_DEV static void mma(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], const uint32_t (&fa)[MI][4],
                         const uint32_t (&fb)[NI][2]) {
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        if (FADD) {
#pragma unroll
          for (int e = 0; e < 4; e++) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
        }
        if (FROM0) qmma0(acc[mi][ni], fa[mi], fb[ni]);
        else qmma(acc[mi][ni], fa[mi], fb[ni], acc[mi][ni]);
      }
  }
  // one stage: FIRST, the tile's first (no window before it); LAST, its last (nothing to load after it).  The stage is released
  // after the MMAs that read its last fragments (LATE) or right after its last ldmatrix (!LATE); mbar_release orders it after
  // them either way
  template <bool FIRST, bool LAST, bool LATE>
  ML_DEV void stage_body(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[2][NI][2]) {
    const uint32_t sb = stage(s), done = empty(s);
    load(fa[1], fb[1], sb, 1);
    mma<G != 0 || FIRST, G != 0 && !FIRST>(acc, total, fa[0], fb[0]);
    load(fa[0], fb[0], sb, 2);
    mma<false, false>(acc, total, fa[1], fb[1]);
    load(fa[1], fb[1], sb, 3);
    mma<false, false>(acc, total, fa[0], fb[0]);
    if (!LATE) mbar_release(done);
    advance();
    if (!LAST) {
      mbar_wait_warp(full(s), ph);
      load(fa[0], fb[0], stage(s), 0);
    }
    mma<false, false>(acc, total, fa[1], fb[1]);
    if (LATE) mbar_release(done);
  }
  // U: stages per K-loop iteration
  template <int U = 1, bool LATE = true> ML_DEV void tile(int K, float (&acc)[MI][NI][4], float (&total)[MI][NI][4]) {
    const int KT = K / BK;
    uint32_t fa[2][MI][4], fb[2][NI][2];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++)
#pragma unroll
        for (int e = 0; e < 4; e++) total[mi][ni][e] = 0.0f;
    mbar_wait_warp(full(s), ph);
    load(fa[0], fb[0], stage(s), 0);
    stage_body<true, false, LATE>(acc, total, fa, fb);
#pragma unroll U
    for (int kt = 1; kt < KT - 1; kt++) stage_body<false, false, LATE>(acc, total, fa, fb);
    stage_body<false, true, LATE>(acc, total, fa, fb);
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++)
#pragma unroll
        for (int e = 0; e < 4; e++) total[mi][ni][e] = G ? __fadd_rn(total[mi][ni][e], acc[mi][ni][e]) : acc[mi][ni][e];
  }
  // ---- the skewed schedule of the same chain: at step t, row block mi (16 rows) runs atom t - mi, so each step closes one row
  // block's window and its 16 FADDs sit among the other blocks' MMAs.  B's fragments of four atoms in a ring (slot = atom % 4),
  // A's double-buffered by step; a stage is released once row block MI - 1's MMAs of its last atom have issued.
  // One row block's MMAs of step t = 4 kt + J (d = J - mi: its atom relative to the stage; d = 4, the last window's promotion)
  template <int J, bool FIRST, bool LAST>
  ML_DEV static void skew_row(int mi, float (&acc)[MI][NI][4], float (&total)[MI][NI][4], const uint32_t (&fa)[2][MI][4],
                              const uint32_t (&fb)[4][NI][2]) {
    const int d = J - mi;
    if ((FIRST && d < 0) || d > 4) return;
    const bool from0 = G ? (d & 3) == 0 : FIRST && d == 0, fadd = G && from0 && !(FIRST && d == 0);
#pragma unroll
    for (int ni = 0; ni < NI; ni++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        if (fadd) total[mi][ni][e] = __fadd_rn(total[mi][ni][e], acc[mi][ni][e]);
        if (!G && d == 4) total[mi][ni][e] = acc[mi][ni][e];
      }
    if (d == 4) return;
#pragma unroll
    for (int ni = 0; ni < NI; ni++) {
      if (from0) qmma0(acc[mi][ni], fa[J & 1][mi], fb[(d + 4) & 3][ni]);
      else qmma(acc[mi][ni], fa[J & 1][mi], fb[(d + 4) & 3][ni], acc[mi][ni]);
    }
  }
  template <int J, bool FIRST, bool LAST>
  ML_DEV void skew_step(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[4][NI][2]) {
    static_assert(MI == APS, "one row block per atom of a stage");
    const int sp = s == 0 ? STAGES - 1 : s - 1, sn = s + 1 == STAGES ? 0 : s + 1;
    skew_row<J, FIRST, LAST>(MI - 1, acc, total, fa, fb);   // first: the next atom's B fragments take its slot
    if (J == 2 && !FIRST) mbar_release(empty(sp));
    if (LAST && J == 6) {
      mbar_release(empty(s));
      advance();
    }
    if (J == 3 && !LAST) mbar_wait_warp(full(sn), sn == 0 ? ph ^ 1 : ph);
    if (!(LAST && J + 1 >= APS)) {
      const uint32_t sb = J + 1 < APS ? stage(s) : stage(sn);
#pragma unroll
      for (int nj = 0; nj < NI / 2; nj++) {
        uint32_t r[4];
        ldsm4(r, sb + b_off[(J + 1) & 3] + nj * 16 * BK);
        fb[(J + 1) & 3][2 * nj][0] = r[0]; fb[(J + 1) & 3][2 * nj][1] = r[1];
        fb[(J + 1) & 3][2 * nj + 1][0] = r[2]; fb[(J + 1) & 3][2 * nj + 1][1] = r[3];
      }
    }
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      const int d = J + 1 - mi;
      if ((FIRST && d < 0) || (LAST && d >= APS)) continue;
      const uint32_t sb = d < 0 ? stage(sp) : d < APS ? stage(s) : stage(sn);
      ldsm4(fa[(J + 1) & 1][mi], sb + a_off[(d + 4) & 3] + mi * 16 * BK);
    }
#pragma unroll
    for (int mi = 0; mi < MI - 1; mi++) skew_row<J, FIRST, LAST>(mi, acc, total, fa, fb);
  }
  template <bool FIRST, bool LAST>
  ML_DEV void skew_stage(float (&acc)[MI][NI][4], float (&total)[MI][NI][4], uint32_t (&fa)[2][MI][4], uint32_t (&fb)[4][NI][2]) {
    skew_step<0, FIRST, LAST>(acc, total, fa, fb);
    skew_step<1, FIRST, LAST>(acc, total, fa, fb);
    skew_step<2, FIRST, LAST>(acc, total, fa, fb);
    skew_step<3, FIRST, LAST>(acc, total, fa, fb);
    if (LAST) {
      skew_step<4, FIRST, LAST>(acc, total, fa, fb);
      skew_step<5, FIRST, LAST>(acc, total, fa, fb);
      skew_step<6, FIRST, LAST>(acc, total, fa, fb);
      skew_step<7, FIRST, LAST>(acc, total, fa, fb);
    } else {
      advance();
    }
  }
  ML_DEV void tile_skew(int K, float (&acc)[MI][NI][4], float (&total)[MI][NI][4]) {
    const int KT = K / BK;
    uint32_t fa[2][MI][4], fb[4][NI][2];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++)
#pragma unroll
        for (int e = 0; e < 4; e++) total[mi][ni][e] = 0.0f;
    mbar_wait_warp(full(s), ph);
    ldsm4(fa[0][0], stage(s) + a_off[0]);
#pragma unroll
    for (int nj = 0; nj < NI / 2; nj++) {
      uint32_t r[4];
      ldsm4(r, stage(s) + b_off[0] + nj * 16 * BK);
      fb[0][2 * nj][0] = r[0]; fb[0][2 * nj][1] = r[1]; fb[0][2 * nj + 1][0] = r[2]; fb[0][2 * nj + 1][1] = r[3];
    }
    skew_stage<true, false>(acc, total, fa, fb);
#pragma unroll 1
    for (int kt = 1; kt < KT - 1; kt++) skew_stage<false, false>(acc, total, fa, fb);
    skew_stage<false, true>(acc, total, fa, fb);
  }
  // the tail stage (TAIL): P_A's rows at the returned address, P_B's at + A_BYTES, both 128-byte rows in the SWIZZLE_128B layout
  ML_DEV uint32_t tail_wait() {
    mbar_wait_warp(full(s), ph);
    return stage(s);
  }
  ML_DEV void tail_release() {
    mbar_release(empty(s));
    advance();
  }
};

}  // namespace ml
