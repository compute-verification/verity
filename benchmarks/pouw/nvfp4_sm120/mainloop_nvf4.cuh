// The NVFP4 block-scaled mainloop on sm_120a: one TM x 128 tile of C = A B^T into registers, each accumulator the chain of
// mma.sync kind::mxf4nvf4 scale_vec::4X m16n8k64 atoms (E2M1 codes, a UE4M3 scale per 16 along k) from +0 in ascending k,
// which is BLACKWELL_SM120_NVF4's chain (verity.ml.circuits.silicon.models): the words are the model's, whatever the schedule.
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
// syncs its 256 MMA threads on a named barrier every stage, before the stage's last step's MMAs).  (64-deep stages, 7 of
// them with 32-byte SWIZZLE_32B boxes, measured 23% slower at 8,192^3.)
//
// Consumer warp w (8 at TM = 256, 4 at 128) owns 64 rows and 64 columns, chosen so that one 16-byte load gives a lane its
// scale words for four MMAs: rows 128 h + 16 q + 32 mi + [0, 16) (m16 block mi = 0..3; wm = w / 2 = 2 h + q) and columns
// 32 i + 8 (2 wn + jl) + [0, 8) (n8 block (i, jl), i = 0..3, jl = 0..1; wn = w % 2).  acc[mi][i][jl][e] is the mma.sync
// m16n8 accumulator of that block: element e at row + lane / 4 + 8 (e / 2), column + 2 (lane % 4) + e % 2 (Mainloop::row_of,
// col_of).  WIDE: columns 64 wn + 16 i + 8 jl + [0, 8) instead, a warp's 64 contiguous, so that store_bf16 writes rows of
// 128 bytes (a warp's 16-byte store 4 lines, not 16: the epilogue's stores made 2,048 L1 tag requests a tile and SM, on the
// pipe the other warps' ldmatrix take); its scale words then take four 8-byte loads a step, not two 16-byte ones.
//
// FP8 1: the plain E4M3 GEMM on the same schedule.  A and B one code a byte, K-major; each accumulator the chain of
// mma.sync kind::f8f6f4 m16n8k32 E4M3 atoms from +0 in ascending k, BLACKWELL_SM120_E4M3_M16N8K32's chain; no scales.  An
// E4M3 atom's fragments are an NVFP4 atom's bytes (A 16 rows and B 8 columns of 32 bytes, the same ldmatrix), and it takes
// the tensor pipe as long, so everything above holds with each stage's 64-byte rows 64 deep, not 128, and no scale loads.
//
// Host side: two CUtensorMaps (A, B), each UINT8, rank 2, globalDim {K / 2 (FP8: K), rows}, globalStrides the same, box
// {64, TM} for A and {64, 128} for B, SWIZZLE_64B, contiguous (64-byte aligned), passed by pointer; the scale buffers by
// address (16-byte aligned; FP8 none).  M a multiple of TM, N of 128, K of 128 (FP8: 64).
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
// whether the phase of parity `parity` has completed, without waiting; true for every lane or for none
NV_DEV bool mbar_ready_warp(uint32_t bar, uint32_t parity) {
  uint32_t r;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(r)
               : "r"(bar), "r"(parity)
               : "memory");
  return __all_sync(0xffffffffu, r);
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
NV_DEV void lds64(uint32_t* r, uint32_t addr) {
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
NV_DEV void stsm4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}
NV_DEV void stg128(void* p, const uint32_t* r) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
// two FP32 values to BF16 (lo in the low half), round to nearest even, as the plain GEMMs' epilogues
NV_DEV uint32_t bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// d = a b + d, one NVFP4 atom (k = 64): sfa's byte b scales k-block b of this lane's A row (8 (lane & 1) + lane / 4 of the
// m16 block), sfb's of its B column (lane / 4)
NV_DEV void omma(float* d, const uint32_t* a, const uint32_t* b, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {0, 0}, {%11}, {0, 0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
}
// d = a b + d, one E4M3 atom (k = 32)
NV_DEV void qmma(float* d, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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
// 8 MMAs in a row), 1 B-major (each B fragment's 4, as CUTLASS).  LATE 1: a warp that finds the next stage not yet full
// after its first group issues the step's other 24 MMAs before it waits, and then loads the next step's fragments in one go
// (0: it waits there, the other groups' loads behind them).  (Starting the second 128 rows' warps up to 2,048 cycles
// late, so that the two warps of an SMSP would reach their epilogues apart, gained nothing: the full barriers they share pull
// them back within a few hundred cycles.  Folding each tile's BF16 store into the next tile's first
// step, each group's 4 blocks stored just before the group overwrites them, measured 2.2% slower.)
// WIDE: each warp's 64 columns contiguous (above).  FP8: E4M3 (above).
template <int TM, int STAGES = 3, int RP = 40, int BORDER = 0, int LATE = 0, int WIDE = 0, int FP8 = 0> struct Mainloop {
  static constexpr int BN = 128, BK = FP8 ? 64 : 128, ROW = 64, WARPS_M = TM / 64, NCW = 2 * WARPS_M, NCT = 32 * NCW, NT = NCT + 128;
  static constexpr int REGS_P = RP, REGS_LAUNCH = 65536 / NT / 8 * 8, REGS_FIT = (NT * REGS_LAUNCH - 128 * REGS_P) / NCT / 8 * 8,
                       REGS_C = REGS_FIT < 256 ? REGS_FIT : 256;
  static constexpr uint32_t A_BYTES = TM * ROW, B_BYTES = BN * ROW, SFA_BYTES = FP8 ? 0 : TM * BK / 16,
                            SFB_BYTES = FP8 ? 0 : BN * BK / 16, SF_OFF = A_BYTES + B_BYTES, STAGE = SF_OFF + SFA_BYTES + SFB_BYTES;
  // store_bf16's scratch: SLOTS 512-byte slots a consumer warp (as many as fit, up to 4)
  static constexpr int SLOTS_FIT = (99 * 1024 - STAGES * STAGE - 16 * STAGES) / (512 * NCW), SLOTS = SLOTS_FIT < 4 ? SLOTS_FIT : 4;
  static constexpr uint32_t SCRATCH = STAGES * STAGE, SCR_WARP = 512 * SLOTS;
  static constexpr int SMEM = SCRATCH + SCR_WARP * NCW + 16 * STAGES;
  static_assert(SLOTS >= 1, "a 512-byte scratch slot a consumer warp");
  static_assert(!WIDE || SLOTS == 4, "WIDE's store_bf16 stages an m16 block's 16 x 64 in 2 KB a warp");
  static_assert(TM == 128 || TM == 256, "TM x 128 tiles of 64 x 64 warp tiles");
  static_assert(STAGE % 512 == 0 && A_BYTES % 512 == 0 && SF_OFF % 512 == 0, "boxes on SWIZZLE_64B's 512-byte period");
  static_assert(SMEM <= 99 * 1024, "sm_120's 99 KB a block");

  uint32_t base, bars;   // the stages; full[s] at bars + 8 s, empty[s] at bars + 8 (STAGES + s)
  int s = 0;
  uint32_t ph = 0;

  NV_DEV explicit Mainloop(uint8_t* smem) {
    base = smem_u32(smem);
    if (base & 1023) __trap();
    bars = base + SCRATCH + SCR_WARP * NCW;
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
      const uint8_t* sa = FP8 ? nullptr : sfa + (size_t)tm * (TM / 128) * KA * 512;
      const uint8_t* sb = FP8 ? nullptr : sfb + (size_t)tn * KA * 512;
      for (int kt = 0; kt < KT; kt++) {
        mbar_wait(empty(s), ph ^ 1);
        mbar_expect_tx(full(s), STAGE);
        const uint32_t st = stage(s);
        tma_load(st, maps->a, full(s), kt * ROW, tm * TM);
        tma_load(st + A_BYTES, maps->b, full(s), kt * ROW, tn * BN);
        if constexpr (!FP8) {
#pragma unroll
          for (int h = 0; h < TM / 128; h++) bulk_load(st + SF_OFF + h * 1024, sa + ((size_t)h * KA + 2 * kt) * 512, 1024, full(s));
          bulk_load(st + SF_OFF + SFA_BYTES, sb + (size_t)2 * kt * 512, 1024, full(s));
        }
        advance();
      }
    }
  }

  // ---- the consumers
  int wm, wn, lane_;
  uint32_t a_off[2], b_off[2], sfa_off, sfb_off, scr;
  NV_DEV void consumer_init(int warp, int lane) {
    wm = warp / 2;
    wn = warp % 2;
    lane_ = lane;
    const int h = wm >> 1, q = wm & 1;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      a_off[j] = (128 * h + 16 * q + (lane & 15)) * ROW + (((2 * j + (lane >> 4)) ^ ((lane >> 1) & 3)) << 4);
      b_off[j] = A_BYTES + ((WIDE ? 64 : 16) * wn + (lane & 7) + 8 * (lane >> 4)) * ROW +
                 (((2 * j + ((lane >> 3) & 1)) ^ ((lane >> 1) & 3)) << 4);
    }
    sfa_off = SF_OFF + h * 1024 + (16 * q + 8 * (lane & 1) + (lane >> 2)) * 16;
    sfb_off = WIDE ? SF_OFF + SFA_BYTES + (lane >> 2) * 16 + 8 * wn : SF_OFF + SFA_BYTES + ((lane >> 2) + 16 * wn) * 16;
    scr = base + SCRATCH + SCR_WARP * warp;
  }
  // the accumulator element e of block (mi, i, jl) sits at (row_of(mi, e), col_of(i, jl, e)) of the tile
  NV_DEV int row_of(int mi, int e) const { return 128 * (wm >> 1) + 16 * (wm & 1) + 32 * mi + (lane_ >> 2) + 8 * (e >> 1); }
  NV_DEV int col_of(int i, int jl, int e) const {
    return (WIDE ? 64 * wn + 16 * i + 8 * jl : 32 * i + 8 * (2 * wn + jl)) + 2 * (lane_ & 3) + (e & 1);
  }

  // one 64-deep step's fragments: A's 4 m16 blocks, B's 8 n8 blocks, their scale words (sa[mi], sb[jl][i])
  struct Frag { uint32_t a[4][4], b[4][2][2], sa[4], sb[2][4]; };
  // part g of step j of stage st: A's m16 block g and B's n8 blocks (g, 0), (g, 1)
  NV_DEV void load_part(Frag& f, uint32_t st, int j, int g) const {
    ldsm4(f.a[g], st + a_off[j] + g * 32 * ROW);
    uint32_t r[4];
    ldsm4(r, st + b_off[j] + g * (WIDE ? 16 : 32) * ROW);
    f.b[g][0][0] = r[0]; f.b[g][0][1] = r[1]; f.b[g][1][0] = r[2]; f.b[g][1][1] = r[3];
  }
  // WIDE: B row 64 wn + 16 i + 8 jl + lane / 4's word sits at (16 (i % 2) + 8 jl + lane / 4) 16 + 4 (2 wn + i / 2) of an atom
  NV_DEV void load_scales(Frag& f, uint32_t st, int j) const {
    lds128(f.sa, st + sfa_off + j * 512);
    if (WIDE) {
#pragma unroll
      for (int jl = 0; jl < 2; jl++)
#pragma unroll
        for (int p = 0; p < 2; p++) {
          uint32_t r[2];
          lds64(r, st + sfb_off + (16 * p + 8 * jl) * 16 + j * 512);
          f.sb[jl][p] = r[0];
          f.sb[jl][p + 2] = r[1];
        }
    } else {
      lds128(f.sb[0], st + sfb_off + j * 512);
      lds128(f.sb[1], st + sfb_off + 8 * 16 + j * 512);
    }
  }
  NV_DEV void load(Frag& f, uint32_t st, int j) const {
    if constexpr (!FP8) load_scales(f, st, j);
#pragma unroll
    for (int g = 0; g < 4; g++) load_part(f, st, j, g);
  }
  NV_DEV static void mma(float* d, const Frag& f, int mi, int i, int jl) {
    if constexpr (FP8) qmma(d, f.a[mi], f.b[i][jl]);
    else omma(d, f.a[mi], f.b[i][jl], f.sa[mi], f.sb[jl][i]);
  }
  // group g of a step's 32 MMAs, the last reads of one of its fragments: BORDER 0 A's m16 block g (its 8 n8 blocks), 1 B's
  // n8 blocks (g, 0) and (g, 1) (with the 4 m16 blocks each)
  NV_DEV static void group(float (&acc)[4][4][2][4], const Frag& f, int g) {
    if (BORDER == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jl = 0; jl < 2; jl++) mma(acc[g][i][jl], f, g, i, jl);
    } else {
#pragma unroll
      for (int jl = 0; jl < 2; jl++)
#pragma unroll
        for (int mi = 0; mi < 4; mi++) mma(acc[mi][g][jl], f, mi, g, jl);
    }
  }
  // cur's MMAs, with nxt (step j of stage st, if `on`) loaded behind them part by part, each part after the group that
  // frees the registers it can take; `mid` runs after group 0
  template <class Mid>
  NV_DEV void step(float (&acc)[4][4][2][4], const Frag& cur, Frag& nxt, uint32_t st, int j, bool on, Mid mid) const {
    group(acc, cur, 0);
    mid();
    rest(acc, cur, nxt, st, j, on);
  }
  // step() after its group 0 and mid
  NV_DEV void rest(float (&acc)[4][4][2][4], const Frag& cur, Frag& nxt, uint32_t st, int j, bool on) const {
    if (!FP8 && on) load_scales(nxt, st, j);
#pragma unroll
    for (int g = 0; g < 4; g++) {
      if (g) group(acc, cur, g);
      if (on) load_part(nxt, st, j, g);
    }
  }
  Frag f0;               // the next step's fragments: loaded ahead, from the next tile's first stage too if `next`
  bool primed = false;   // f0 holds them
  // one tile's chain into acc (from +0): step 1's fragments load behind step 0's MMAs and the next stage's step 0 behind
  // step 1's; a stage is released after step 1's first group (its last fragments were read behind step 0).  next: this warp
  // runs tile() again, so the last step loads the next tile's first fragments (behind its MMAs, not after the epilogue)
  NV_DEV void tile(int K, float (&acc)[4][4][2][4], bool next = false) {
    const int KT = K / BK;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int jl = 0; jl < 2; jl++)
#pragma unroll
          for (int e = 0; e < 4; e++) acc[mi][i][jl][e] = 0.0f;
    Frag f1;
    if (!primed) {
      mbar_wait_warp(full(s), ph);
      load(f0, stage(s), 0);
    }
#pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
      const uint32_t done = empty(s);
      step(acc, f0, f1, stage(s), 1, true, [] {});
      advance();
      const bool more = kt + 1 < KT || next;
      if (LATE) {
        group(acc, f1, 0);
        mbar_release(done);
        const bool ready = more && mbar_ready_warp(full(s), ph);
        rest(acc, f1, f0, stage(s), 0, ready);
        if (more && !ready) {
          mbar_wait_warp(full(s), ph);
          load(f0, stage(s), 0);
        }
      } else {
        step(acc, f1, f0, stage(s), 0, more, [&] {
          mbar_release(done);
          if (more) mbar_wait_warp(full(s), ph);
        });
      }
    }
    primed = next;
  }

  // ---- epilogues: tile (tm, tn) of C, row-major with ldc elements a row
  NV_DEV void store_f32(float* C, int ldc, int tm, int tn, const float (&acc)[4][4][2][4]) const {
    float* p = C + ((size_t)tm * TM + row_of(0, 0)) * ldc + (size_t)tn * BN + col_of(0, 0, 0);
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        float* r = p + (size_t)(32 * mi + 8 * hf) * ldc;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int jl = 0; jl < 2; jl++)
            *reinterpret_cast<float2*>(r + (WIDE ? 16 : 32) * i + 8 * jl) =
                make_float2(acc[mi][i][jl][2 * hf], acc[mi][i][jl][2 * hf + 1]);
      }
  }
  // BF16 (bf16x2), each 16 x 16 block (mi, i) of the warp's through a scratch slot: stmatrix writes the block's four 8 x 8
  // matrices as 16 rows of 32 bytes (chunk c of row r at c ^ (r / 4) % 2, bank-conflict free), then each lane reads 16 bytes
  // of row lane / 2 and stores them, so a warp's store is 16 rows' 32 contiguous bytes; SLOTS blocks a round, so that their
  // reads are in flight together.  WIDE: each m16 block mi's 16 x 64 through the warp's 2 KB: stmatrix writes block (mi, i)
  // into 16 rows of 128 bytes (chunk c of row r at c ^ r % 8), then each lane reads 16 bytes of row 4 u + lane / 8 and stores
  // them, so a warp's store is 4 rows' 128 contiguous bytes
  NV_DEV void store_bf16(uint16_t* C, int ldc, int tm, int tn, const float (&acc)[4][4][2][4]) const {
    if constexpr (WIDE) {
      const int l = lane_, mm = l >> 3, r = (l & 7) + 8 * (mm >> 1);
      const uint32_t st_row = scr + r * 128, c0 = (mm & 1) ^ (l & 7), ld_row = scr + (l >> 3) * 128, c1 = (l & 7) ^ (l >> 3);
      uint16_t* p = C + ((size_t)tm * TM + 128 * (wm >> 1) + 16 * (wm & 1) + (l >> 3)) * ldc + (size_t)tn * BN + 64 * wn + 8 * (l & 7);
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float(&x)[2][4] = acc[mi][i];
          stsm4(st_row + ((c0 ^ (2 * i)) << 4), bf16x2(x[0][0], x[0][1]), bf16x2(x[1][0], x[1][1]), bf16x2(x[0][2], x[0][3]),
                bf16x2(x[1][2], x[1][3]));
        }
        __syncwarp();
        uint32_t v[4][4];
#pragma unroll
        for (int u = 0; u < 4; u++) lds128(v[u], ld_row + 512 * u + ((c1 ^ (4 * (u & 1))) << 4));
#pragma unroll
        for (int u = 0; u < 4; u++) stg128(p + (size_t)(32 * mi + 4 * u) * ldc, v[u]);
        __syncwarp();
      }
    } else {
      const int l = lane_, mm = l >> 3, r = (l & 7) + 8 * (mm >> 1);
      const uint32_t st_addr = scr + r * 32 + (((mm & 1) ^ ((r >> 2) & 1)) << 4);
      const uint32_t ld_addr = scr + (l >> 1) * 32 + (((l & 1) ^ ((l >> 3) & 1)) << 4);
      uint16_t* p = C + ((size_t)tm * TM + 128 * (wm >> 1) + 16 * (wm & 1) + (l >> 1)) * ldc + (size_t)tn * BN + 16 * wn + 8 * (l & 1);
#pragma unroll
      for (int g0 = 0; g0 < 16; g0 += SLOTS) {
#pragma unroll
        for (int u = 0; u < SLOTS && g0 + u < 16; u++) {
          const float(&x)[2][4] = acc[(g0 + u) / 4][(g0 + u) % 4];
          stsm4(st_addr + 512 * u, bf16x2(x[0][0], x[0][1]), bf16x2(x[1][0], x[1][1]), bf16x2(x[0][2], x[0][3]),
                bf16x2(x[1][2], x[1][3]));
        }
        __syncwarp();
        uint32_t v[SLOTS][4];
#pragma unroll
        for (int u = 0; u < SLOTS && g0 + u < 16; u++) lds128(v[u], ld_addr + 512 * u);
#pragma unroll
        for (int u = 0; u < SLOTS && g0 + u < 16; u++) stg128(p + (size_t)32 * ((g0 + u) / 4) * ldc + 32 * ((g0 + u) % 4), v[u]);
        __syncwarp();
      }
    }
  }
};

}  // namespace nvf4
