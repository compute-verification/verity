// The NVFP4 block-scaled mainloop on sm_120a: one TM x 128 tile of C = A B^T into registers, each accumulator the chain of
// mma.sync kind::mxf4nvf4 scale_vec::4X m16n8k64 atoms (E2M1 codes, a UE4M3 scale per 16 along k) from +0 in ascending k,
// which is BLACKWELL_SM120_NVF4's chain (verity.ml.tc.models): the words are the model's, whatever the schedule.
//
// Operands, as CUTLASS's and cuBLASLt's NVFP4 GEMMs take them: A (M x K) and B (N x K) K-major, two codes a byte (element 2i
// in the low nibble); their scales in the 128 x 4 tiled layout, scale (r, c) (c = k / 16) at
// ((r / 128) (K / 64) + c / 4) 512 + (r % 32) 16 + ((r % 128) / 32) 4 + c % 4, so each 512-byte atom holds 128 rows' scales of
// one 64-deep step.
//
// The schedule is mainloop_sm120.cuh's (the FP8 one): a producer warpgroup whose first lane loads each 128-deep stage (A's TM
// and B's 128 rows of 64 bytes by TMA, SWIZZLE_64B, and their scale atoms by bulk copy) into STAGES stages counted on
// mbarriers; consumer warps wait on a stage, ldmatrix their fragments one 64-deep step ahead, and each release it on a second
// mbarrier after the MMAs that read its last fragments.  No block-wide barrier (CUTLASS v4.8.0's sm_120 block-scaled mainloop
// syncs its 256 MMA threads on a named barrier every stage, before the stage's last step's MMAs).
//
// Consumer warp w (8 at TM = 256, 4 at 128) owns 64 rows and 64 columns, chosen so that one 16-byte load gives a lane its
// scale words for four MMAs: rows 128 h + 16 q + 32 mi + [0, 16) (m16 block mi = 0..3; wm = w / 2 = 2 h + q) and columns
// 32 i + 8 (2 wn + jl) + [0, 8) (n8 block (i, jl), i = 0..3, jl = 0..1; wn = w % 2).  acc[mi][i][jl][e] is the mma.sync
// m16n8 accumulator of that block: element e at row + lane / 4 + 8 (e / 2), column + 2 (lane % 4) + e % 2 (Mainloop::row_of,
// col_of).
//
// Host side: two CUtensorMaps (A, B), each UINT8, rank 2, globalDim {K / 2, rows}, globalStrides {K / 2}, box {64, TM} for A
// and {64, 128} for B, SWIZZLE_64B, contiguous (64-byte aligned), passed by pointer; the scale buffers by address (16-byte
// aligned).  M a multiple of TM, N of 128, K of 128 (and K / 64 whole atoms, so of 64 too).
#pragma once
#include <cstdint>

namespace nvf4 {

#define NV_DEV __device__ __forceinline__

NV_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

NV_DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
NV_DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// a consumer's release of a stage: its ldmatrix and ld.shared reads (generic proxy) ordered before the TMA and bulk copies
// (async proxy) that refill it (mainloop_sm120.cuh: without the fence an arrive overtakes a queued ldmatrix)
NV_DEV void mbar_release(uint32_t bar) {
  asm volatile("fence.proxy.async.shared::cta;\n mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
NV_DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n NV_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra NV_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
// try_wait's predicate is per thread: the warp reconverges before the .aligned ldmatrix and mma.sync that follow
NV_DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
NV_DEV void tma_acquire(const void* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(map) : "memory");
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
// box (c0 = byte offset along the depth, c1 = row) of `map` into dst, counted on bar
NV_DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
// `bytes` contiguous bytes (a multiple of 16, both ends 16-byte aligned) into dst, counted on bar.  The .shared::cta form:
// ptxas 12.9 lowers .shared::cluster's to a CALL, and then ignores every setmaxnreg in the kernel (C7506)
NV_DEV void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(bar)
               : "memory");
}
template <int R> NV_DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> NV_DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
NV_DEV void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
NV_DEV void lds128(uint32_t* r, uint32_t addr) {
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// d = a b + d, one NVFP4 atom (k = 64): sfa's byte b scales k-block b of this lane's A row (8 (lane & 1) + lane / 4 of the
// m16 block), sfb's of its B column (lane / 4)
NV_DEV void omma(float* d, const uint32_t* a, const uint32_t* b, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}

// The tile order both roles walk: groups of `group` tile rows, column by column inside a group.
NV_DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

struct Maps { uint8_t a[128], b[128]; };

// TM x 128 tiles (TM = 128 or 256), consumer warps then a producer warpgroup, whose first lane issues every copy; setmaxnreg
// gives the producer RP registers a thread and the consumers the rest (232 at TM = 256 and RP = 40, as in CUTLASS; 240 at
// RP = 24).  Launch NT threads with __launch_bounds__(NT, 1) and SMEM bytes of dynamic shared memory at a 1,024-byte-aligned
// address (`extern __shared__ __align__(1024)`).  BORDER: the MMA order inside a 64-deep step, 0 A-major (each A fragment's
// 8 MMAs in a row), 1 B-major (each B fragment's 4, as CUTLASS).
template <int TM, int STAGES = 3, int RP = 40, int BORDER = 0> struct Mainloop {
  static constexpr int BN = 128, BK = 128, ROW = BK / 2, WARPS_M = TM / 64, NCW = 2 * WARPS_M, NCT = 32 * NCW, NT = NCT + 128;
  static constexpr int REGS_P = RP, REGS_LAUNCH = 65536 / NT / 8 * 8, REGS_FIT = (NT * REGS_LAUNCH - 128 * REGS_P) / NCT / 8 * 8,
                       REGS_C = REGS_FIT < 256 ? REGS_FIT : 256;
  static constexpr uint32_t A_BYTES = TM * ROW, B_BYTES = BN * ROW, SFA_BYTES = TM * BK / 16, SFB_BYTES = BN * BK / 16,
                            SF_OFF = A_BYTES + B_BYTES, STAGE = SF_OFF + SFA_BYTES + SFB_BYTES;
  static constexpr int SMEM = STAGES * STAGE + 16 * STAGES;
  static_assert(TM == 128 || TM == 256, "TM x 128 tiles of 64 x 64 warp tiles");
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0 && SF_OFF % 1024 == 0, "boxes on SWIZZLE_64B's 512-byte period");
  static_assert(SMEM <= 99 * 1024, "sm_120's 99 KB a block");

  uint32_t base, bars;   // the stages; full[s] at bars + 8 s, empty[s] at bars + 8 (STAGES + s)
  int s = 0;
  uint32_t ph = 0;

  NV_DEV explicit Mainloop(uint8_t* smem) {
    base = smem_u32(smem);
    if (base & 1023) __trap();
    bars = base + STAGES * STAGE;
  }
  NV_DEV uint32_t full(int i) const { return bars + 8 * i; }
  NV_DEV uint32_t empty(int i) const { return bars + 8 * (STAGES + i); }
  NV_DEV uint32_t stage(int i) const { return base + i * STAGE; }
  NV_DEV void advance() {
    if (++s == STAGES) { s = 0; ph ^= 1; }
  }
  NV_DEV void init(int tid) {
    if (tid == 0) {
      for (int i = 0; i < STAGES; i++) { mbar_init(full(i), 1); mbar_init(empty(i), NCT); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
  }
  // every thread of the block: the producer warpgroup produces every tile's stages and exits (a path back to the caller would
  // join the two register budgets); the consumers take their registers and offsets and return, to run tile() per tile in the
  // same raster order
  NV_DEV void roles(const Maps* maps, const uint8_t* sfa, const uint8_t* sfb, int M, int N, int K, int group) {
    init(threadIdx.x);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (warp >= NCW) {
      regs_dec<REGS_P>();
      if (warp == NCW && lane == 0) produce(maps, sfa, sfb, M, N, K, group);
      asm volatile("exit;");
    }
    regs_inc<REGS_C>();
    consumer_init(warp, lane);
  }

  NV_DEV void produce(const Maps* maps, const uint8_t* sfa, const uint8_t* sfb, int M, int N, int K, int group) {
    const int tiles_m = M / TM, tiles_n = N / BN, KT = K / BK, KA = K / 64;
    tma_acquire(maps->a);
    tma_acquire(maps->b);
    for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
      int tm, tn;
      raster(t, tiles_m, tiles_n, group, tm, tn);
      const uint8_t* sa = sfa + (size_t)tm * (TM / 128) * KA * 512;
      const uint8_t* sb = sfb + (size_t)tn * KA * 512;
      for (int kt = 0; kt < KT; kt++) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        const uint32_t st = stage(s);
        tma_load(st, maps->a, full(s), kt * ROW, tm * TM);
        tma_load(st + A_BYTES, maps->b, full(s), kt * ROW, tn * BN);
#pragma unroll
        for (int h = 0; h < TM / 128; h++) bulk_load(st + SF_OFF + h * 1024, sa + ((size_t)h * KA + 2 * kt) * 512, 1024, full(s));
        bulk_load(st + SF_OFF + SFA_BYTES, sb + (size_t)2 * kt * 512, 1024, full(s));
        advance();
      }
    }
  }

  // ---- the consumers
  int wm, wn, lane_;
  uint32_t a_off[2], b_off[2], sfa_off, sfb_off;
  NV_DEV void consumer_init(int warp, int lane) {
    wm = warp / 2;
    wn = warp % 2;
    lane_ = lane;
    const int h = wm >> 1, q = wm & 1;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      a_off[j] = (128 * h + 16 * q + (lane & 15)) * ROW + (((2 * j + (lane >> 4)) ^ ((lane >> 1) & 3)) << 4);
      b_off[j] = A_BYTES + (16 * wn + (lane & 7) + 8 * (lane >> 4)) * ROW + (((2 * j + ((lane >> 3) & 1)) ^ ((lane >> 1) & 3)) << 4);
    }
    sfa_off = SF_OFF + h * 1024 + (16 * q + 8 * (lane & 1) + (lane >> 2)) * 16;
    sfb_off = SF_OFF + SFA_BYTES + ((lane >> 2) + 16 * wn) * 16;
  }
  // the accumulator element e of block (mi, i, jl) sits at (row_of(mi, e), col_of(i, jl, e)) of the tile
  NV_DEV int row_of(int mi, int e) const { return 128 * (wm >> 1) + 16 * (wm & 1) + 32 * mi + (lane_ >> 2) + 8 * (e >> 1); }
  NV_DEV int col_of(int i, int jl, int e) const { return 32 * i + 8 * (2 * wn + jl) + 2 * (lane_ & 3) + (e & 1); }

  // one 64-deep step's fragments: A's 4 m16 blocks, B's 8 n8 blocks, their scale words (sa[mi], sb[jl][i])
  struct Frag { uint32_t a[4][4], b[4][2][2], sa[4], sb[2][4]; };
  // part g of step j of stage st: A's m16 block g and B's n8 blocks (g, 0), (g, 1)
  NV_DEV void load_part(Frag& f, uint32_t st, int j, int g) const {
    ldsm4(f.a[g], st + a_off[j] + g * 32 * ROW);
    uint32_t r[4];
    ldsm4(r, st + b_off[j] + g * 32 * ROW);
    f.b[g][0][0] = r[0]; f.b[g][0][1] = r[1]; f.b[g][1][0] = r[2]; f.b[g][1][1] = r[3];
  }
  NV_DEV void load_scales(Frag& f, uint32_t st, int j) const {
    lds128(f.sa, st + sfa_off + j * 512);
    lds128(f.sb[0], st + sfb_off + j * 512);
    lds128(f.sb[1], st + sfb_off + 8 * 16 + j * 512);
  }
  NV_DEV void load(Frag& f, uint32_t st, int j) const {
    load_scales(f, st, j);
#pragma unroll
    for (int g = 0; g < 4; g++) load_part(f, st, j, g);
  }
  // group g of a step's 32 MMAs, the last reads of one of its fragments: BORDER 0 A's m16 block g (its 8 n8 blocks), 1 B's
  // n8 blocks (g, 0) and (g, 1) (with the 4 m16 blocks each)
  NV_DEV static void group(float (&acc)[4][4][2][4], const Frag& f, int g) {
    if (BORDER == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jl = 0; jl < 2; jl++) omma(acc[g][i][jl], f.a[g], f.b[i][jl], f.sa[g], f.sb[jl][i]);
    } else {
#pragma unroll
      for (int jl = 0; jl < 2; jl++)
#pragma unroll
        for (int mi = 0; mi < 4; mi++) omma(acc[mi][g][jl], f.a[mi], f.b[g][jl], f.sa[mi], f.sb[jl][g]);
    }
  }
  // cur's MMAs, with nxt (step j of stage st, if `on`) loaded behind them part by part, each part after the group that
  // frees the registers it can take; `mid` runs after group 0
  template <class Mid>
  NV_DEV void step(float (&acc)[4][4][2][4], const Frag& cur, Frag& nxt, uint32_t st, int j, bool on, Mid mid) const {
    group(acc, cur, 0);
    mid();
    if (on) load_scales(nxt, st, j);
#pragma unroll
    for (int g = 0; g < 4; g++) {
      if (g) group(acc, cur, g);
      if (on) load_part(nxt, st, j, g);
    }
  }
  // one tile's chain into acc (from +0): step 1's fragments load behind step 0's MMAs and the next stage's step 0 behind
  // step 1's; a stage is released after step 1's first group (its last fragments were read behind step 0)
  NV_DEV void tile(int K, float (&acc)[4][4][2][4]) {
    const int KT = K / BK;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jl = 0; jl < 2; jl++)
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][i][jl][e] = 0.0f;
    Frag f0, f1;
    mbar_wait_warp(full(s), ph);
    load(f0, stage(s), 0);
#pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
      const uint32_t done = empty(s);
      step(acc, f0, f1, stage(s), 1, true, [] {});
      advance();
      const bool more = kt + 1 < KT;
      step(acc, f1, f0, stage(s), 0, more, [&] {
        mbar_release(done);
        if (more) mbar_wait_warp(full(s), ph);
      });
    }
  }
};

}  // namespace nvf4
