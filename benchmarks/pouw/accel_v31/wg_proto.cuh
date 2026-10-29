// ACCEL's protocol GEMM (v31.cu's gemm_kernel<true>) on the H200's warpgroup MMAs: wgmma.linear's mainloop
// (benchmarks/torch_gpu/s8_linear/s8_wgmma.cu: persistent, warp-specialized, TMA with SWIZZLE_128B, wgmma m64nNk32
// .s32.s8.s8, 128 x BN tiles of two consumer warpgroups) with the receipt's values, bit for bit gemm_kernel<true>'s:
//   T_s     after the main k-tile that ends segment s, each cell's low 16 bits of X W'^T over the first s L inner indices,
//           to the prefix-sum buffer in v31's layout ([128 x 128 tile][s][cell], u16)
//   O       T_S + 256 dO - Z_X U^T - V Y_W^T mod 2^32, row-major s32
// The peel's tail rows P_A, P_B (form_a_kernel's and form_b_kernel's, tail segments of L') go through the same accumulator,
// ahead of the main k-tiles, in Horner order: the tail segments of the top digit D first (from zero), then acc *= 128 and
// digit D - 1's, ..., acc *= -128 after digit 1, then the shift-0 tail segments (-Z_0 U^T and -V Y_0^T), so acc holds the tail
// tau = -Z_X U^T - V Y_W^T when the main k-tiles start and tau + T_S after them. Each segment's prefix sum is the low half of
// acc - tau: tau's low halves are kept (CM 0: packed in 64 registers a thread at BN 256; CM 1: in an L2-resident scratch
// slot of the CTA's, read back at each segment end), so a tail costs (dz + dy) L' inner indices on the tensor cores and
// no second accumulator. The tail's step order and events are a table in shared memory, built once a CTA.
// RP, the register plan (below): 0 is wgmma.linear's, 3 leaves room for one of the commit's CTAs beside each GEMM CTA, 4
// and 5 hash the commit's rows in the GEMM's own CTA.
//   mode 0  the plain product X W'^T to O (the bench's `plain` arm)
//   mode 1  the protocol's GEMM (diag 1: no segment stores, 2: no tail, 4: no dO read; timing only)
//   mode 2  wgmma.linear itself on A, B: bf16(RN(RN(f32(acc) s_a) s_w)) values, the `wgl` arm, gamma's baseline
// Integer wgmma accumulates mod 2^32 (no .satfinite), as the receipt's O is taken.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

#include "../../torch_gpu/s8_linear/s8_acc.h"

namespace wgp {

#define WGD __device__ __forceinline__

// TW: v31's prefix-sum tile; MAX_TT: the tail's k-tiles a table holds ((dz + dy) L' / BK, 48 at L' 1024 and dz = dy = 3, and
// KQ more)
constexpr int BK = 128, TM = 128, SMEM_BUDGET = 220 * 1024, MAX_STAGES = 8, TW = 128, MAX_TT = 1024;
// The register plans (RP). An SM's registers are four files of 16,384, one a sub-partition, and a CTA's warps are dealt to
// them in turn, so a commit CTA (4 warps, SMALL_MINB 5: <= 104 registers a thread, 3,328 a warp) fits beside a GEMM CTA
// only if every sub-partition has that much left:
//   RP 0  384 threads at 168 (wgmma.linear's: the producer warpgroup drops to 40, the consumers rise to 232): 3 warps
//         a sub-partition, 16,128, no room
//   RP 3  256 threads at 200 (no producer warp: thread 0 issues the loads between its own steps): 2 warps a
//         sub-partition, 12,800, 3,584 left
//   RP 4  384 threads at 168, the producer warpgroup at 104 and the consumers at 200: warp 8 issues the loads, warps 9 to
//         11 (one a sub-partition 1 to 3) hash the rows of the tiles the consumers have stored (gemm_r4's H), so the
//         commit's compressions run on the ALU while the consumers wait on their MMAs. No back-pressure: when the
//         consumers are done (static schedule, every gridDim.x-th tile) a hash warp stops between two chunks, keeping
//         its rows' chaining values, and the leaf phase hashes on from there (H's flags)
//   RP 5  RP 4 with a hash warpgroup of its own: 512 threads at 128, warps 9 to 15 hashing (two a sub-partition, one on
//         sub-partition 0 beside the load warp), the producer and hash warpgroups at HR and the consumers at 256 - HR
//         (2 HR + 2 (256 - HR) = 512 a sub-partition). A 128 x 256 tile's MMAs are two m64n128k32 a k32 atom, into
//         accumulators 0 to 63 and 64 to 127 (m64n256's launch budget is 154). Mode 0, and mode 1 without a tail (a
//         served call's prefix sums alone, whose hash warps hash its tiles' rows, the prefix-sum rows as they're posted).
//         TST: the consumers store O and T_s through shared memory, a box at a time into two slots a warpgroup (TSLOT
//         bytes each, SWIZZLE_128B, the stages' 4 kept), and warp 9 stores the boxes with TMA (store_tiles), so 6 warps hash.
//   RP 6  ping-pong (CUTLASS's sm_90 KernelTmaWarpSpecializedPingpong; body_pp): RP 0's registers, but each consumer
//         warpgroup owns a whole 128 x 128 tile (two m64n128k32 atoms a k32 step, 128 accumulators a thread) and the CTA's
//         tiles go to the two in turn, so while one runs a tile's mainloop on the tensor pipe the other stores its last
//         tile and hashes O rows (H) on the ALU, in the accumulators' registers. Mode 0 only (a served call's D)
//   RP 7  RP 6 with warps 9 to 11 hashing too, at RP 4's 104 (the consumers at 200)
// (384 threads at 136 can't be built: wgmma n256 needs a launch budget of 154, ptxas C7602, whatever setmaxnreg raises it
// to. 288 threads at 168, one producer warp, puts 3 warps and 16,128 registers on sub-partition 0, and the commit's CTAs
// never land beside it.)
template <int RP> __host__ __device__ constexpr int threads() { return RP == 3 ? 2 * 128 : RP == 5 ? 4 * 128 : 3 * 128; }
// RP 5's register splits (HR, the producer and hash warpgroups'), each a build of gemm_r5. At 104 the consumers' 152
// can't keep a 128 x 256 tile's MMAs in flight (ptxas C7512 serializes them) and spill 1 KB; at 88 the hash spills.
constexpr int R5_SPLITS[] = {96, 88};
// TST's slot: a box of 64 rows of 128 bytes (32 O cells or 64 prefix sums a row)
constexpr uint32_t TSLOT = 64 * 128;
// RP 4's and 5's hasher, H: lpt() rows a 128 x 128 tile; h(ct, j, stop, budget) hashes on row j of commit tile ct (tm N /
// 128 + tn) by at most budget SHA-512 chunks, until stop() holds, and says whether the row is hashed whole; chunk, a
// visit's budget (0: a row whole at once)
struct NoHash {
  static constexpr unsigned chunk = 0;
  WGD unsigned lpt() const { return 1; }
  template <class Stop> WGD bool operator()(unsigned long long, unsigned, const Stop&, unsigned) const { return true; }
};

template <int BN> struct Tile {
  static_assert(BN == 128 || BN == 256, "128 x 128 or 128 x 256 tiles");
  static constexpr int NWG = 2, NA = BN / 2;
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int STAGES = SMEM_BUDGET / STAGE < MAX_STAGES ? SMEM_BUDGET / STAGE : MAX_STAGES;
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES + 4 * MAX_TT + 4 * STAGES + 16;
  // TST: the four slots after the stages and their 8 mbarriers, no tail table (RP 5 has no tail)
  static constexpr int SMEM_TST = 1024 + STAGES * STAGE + 4 * TSLOT + 16 * STAGES + 64 + 4 * STAGES + 16;
  static_assert(SMEM_TST <= 227 * 1024, "an H200 CTA's shared memory");
};
struct alignas(128) Args {
  CUtensorMap a, b, pa, pb;   // X or A (boxes of 128 rows), W' or B (BN rows), P_A (128 rows), P_B (BN rows)
  CUtensorMap q;              // KQ > 0: -2 w_W (BN rows), whose KQ k-tiles against X's add 256 X w_W^T into O
  CUtensorMap tso, tsw;       // TST: O's store map (s32, boxes of 64 rows by 32 cells) and words' (u16 rows of 128, a prefix-sum
                              // tile's row a row, boxes of 64 by 64), SWIZZLE_128B
  uint16_t* words;
  const int* dO;
  const int *lA, *cA;         // wrap 7: X's wraps a row (form_a's list: k << 1 | negative) and their count, for w_X B~^T
  const int16_t* bt;          // wrap 7: B~^T (K x N), form_b's
  int capA;
  int* O;                     // mode 0, 1: s32 cells; mode 2: bf16 values
  uint32_t* cbuf;             // CM 1: a slot of TM x BN / 2 u32s a CTA
  const float *sa, *sw;       // mode 2's scales
  // a CTA's first tile is blockIdx.x, its next ones gridDim.x + a claim on tctr less tbase (null: every gridDim.x-th);
  // launches that share tctr run in one stream's order, and each advances it by its tiles (v31.cu's Ring, gctr)
  unsigned* tctr;
  unsigned tbase;
  int M, N, K, L, S, lp, dz, dy, TT, group, mode, diag, KQ;
  int hall;                   // RP 4, 5: 1, the hash warps hash every row, past the consumers' end (timing their own rate);
                              // -1, none (timing the GEMM alone in the plan's CTA)
  int ppm;                    // RP 6, 7: a consumer starts no SHA-512 chunk once the other's mainloop has ppm k-tiles left
};

WGD uint32_t smem_at(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
WGD void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
WGD void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
WGD void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
WGD void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WGP_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WGP_WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
WGD void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
WGD bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t r;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(r)
               : "r"(bar), "r"(parity)
               : "memory");
  return r != 0;
}
WGD void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
WGD void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
// a SWIZZLE_128B box of shared memory to the tensor at (c0, c1), in the thread's bulk async-group
WGD void tma_store(const void* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(map), "r"(src), "r"(c0), "r"(c1)
               : "memory");
}
WGD void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> WGD void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
template <int N> WGD void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory"); }
WGD void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
WGD void fence_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
template <int R> WGD void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> WGD void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
WGD uint32_t ld_acq(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(smem_at(p)) : "memory");
  return v;
}
WGD void red_rel(uint32_t* p, uint32_t v) {
  asm volatile("red.release.cta.shared::cta.add.u32 [%0], %1;" ::"r"(smem_at(p)), "r"(v) : "memory");
}
WGD uint32_t ld_vol(const uint32_t* p) { return *reinterpret_cast<const volatile uint32_t*>(p); }
WGD void st_vol(uint32_t* p, uint32_t v) { *reinterpret_cast<volatile uint32_t*>(p) = v; }
#define WGP_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WGP_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WGP_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

WGD uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0), one k32 atom of a 64 x N tile: accumulator i of thread x is row 16 (x / 32) + (x % 32) / 4 +
// 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
template <int N> WGD void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  if constexpr (N == 256)
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
                 ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
  else
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %66, 0;\n wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 " ACC64_S
                 ", %64, %65, p;\n}" : ACC64_R : "l"(da), "l"(db), "r"(sd));
}
// one k32 atom of the warpgroup's 64 x BN tile; SPLIT: a 128 x 256 tile's as two n128 halves (B's rows 128 on start
// 128 BK bytes on), the same accumulator layout
template <int BN, bool SPLIT> WGD void imma_atom(int32_t* d, uint32_t a, uint32_t b, uint32_t sd) {
  if constexpr (SPLIT && BN == 256) {
    imma<128>(d, desc_sw128(a), desc_sw128(b), sd);
    imma<128>(d + 64, desc_sw128(a), desc_sw128(b + 128 * BK), sd);
  } else {
    imma<BN>(d, desc_sw128(a), desc_sw128(b), sd);
  }
}
template <int NA> WGD void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}
// bounds the loads an unrolled epilogue hoists (all 64 of a 128 x 256 tile's dO pairs would spill beside 128 accumulators):
// ptxas moves no load above a warp barrier, and an empty asm reaches only NVVM
WGD void ld_fence() { __syncwarp(); }

// the tail's k-tiles in Horner order: digit e of tail segment b (0 for 0 and 1, then Z's digits, then Y's), the top digit
// D = max(dz, dy) - 1 first; step i < TT is k-tile kt of P's columns, and `end` says the step closes its digit's group
WGD int digit_of(int b, int dz) { return b < 2 ? 0 : (b <= dz ? b - 1 : b - dz); }
struct TailStep { int kt; int e; bool end; };
WGD TailStep tail_step(int i, int lp, int dz, int dy) {
  const int kpb = lp / BK, nb = dz + dy, D = (dz > dy ? dz : dy) - 1;
  for (int e = D; e >= 0; --e) {
    int n = 0;
    for (int b = 0; b < nb; ++b) n += digit_of(b, dz) == e;
    const int len = n * kpb;
    if (i < len) {
      int seen = 0;
      for (int b = 0; b < nb; ++b)
        if (digit_of(b, dz) == e) {
          if (i < seen + kpb) return TailStep{b * kpb + (i - seen), e, i == len - 1};
          seen += kpb;
        }
    }
    i -= len;
  }
  return TailStep{0, 0, false};
}

WGD uint32_t bf16_pair(int32_t a, int32_t b, float sr, float c0, float c1) {
  const float ya = __fmul_rn(__fmul_rn(__int2float_rn(a), sr), c0), yb = __fmul_rn(__fmul_rn(__int2float_rn(b), sr), c1);
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(ya)) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(yb)) << 16;
}

WGD void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// the tail table's entries: k-tile kt | event << 16 (the event at the step's end) | TAIL_Q for a k-tile of X (-2 w_W)^T.
// The KQ k-tiles of X (-2 w_W)^T close digit 1's group, so its x(-128) leaves 256 X w_W^T in tau, and O = acc + 256 dO
// with dO = w_X B~^T alone; T_s = lo16(acc - tau) are untouched.
enum { TAIL_NONE = 0, TAIL_X128 = 1, TAIL_XM128 = 2, TAIL_CAPTURE = 3, TAIL_Q = 1 << 20 };

// RP 4's and 5's hash warps (warps W0 to W0 + NHW - 1): warp w takes batches w, w + NHW, ..: batch b is the CTA's rows (H's)
// 32 b .. 32 b + 31 in its tiles' order, Lg = (BN / 128) h.lpt() a tile, a row a lane, once the consumers have posted (8
// warps a tile) every tile the batch reads. h.chunk 0: each batch whole in turn; else passes over the batches posted so far, at most
// chunk SHA-512 chunks of each a visit, so the rows the leaf phase takes on are left with chains of about one length.
// Without hall, nothing starts once the consumers are done, and stop() ends the row's hash where it is.
template <int BN, int NHW, int W0, class H>
WGD void hash_warps(const Args& p, const H& h, const uint32_t* posted, int tiles_m, int tiles_n, int ntile) {
  if (p.hall < 0) return;
  const unsigned w = threadIdx.x / 32 - W0, lane = threadIdx.x % 32, lpo = h.lpt(), Lg = (BN / TW) * lpo,
                 nq = (unsigned)ntile * Lg, nb = (nq + 31) / 32, done = 8u * (unsigned)ntile;
  auto stop = [&]() { return !p.hall && ld_acq(posted) >= done; };
  auto need = [&](unsigned b) { return 8 * (min(32 * b + 31, nq - 1) / Lg + 1); };
  auto posts = [&]() { return __shfl_sync(~0u, ld_acq(posted), 0); };   // one value for the warp: its loops stay uniform
  auto visit = [&](unsigned b, unsigned budget) {   // this lane's row of batch b: whether it's hashed whole
    const unsigned q = 32 * b + lane;
    if (q >= nq) return true;
    const unsigned i = q / Lg, jj = q % Lg;
    int tm, tn;
    raster((int)(blockIdx.x + i * gridDim.x), tiles_m, tiles_n, p.group, tm, tn);
    return h((unsigned long long)tm * (p.N / TW) + (unsigned long long)tn * (BN / TW) + jj / lpo, jj % lpo, stop, budget);
  };
  if (!h.chunk) {
    for (unsigned b = w; b < nb; b += NHW) {
      unsigned v = posts();
      if (!p.hall && v >= done) break;
      while (v < need(b)) {
        __nanosleep(256);
        v = posts();
      }
      visit(b, ~0u);
    }
    return;
  }
  for (unsigned lo = w; lo < nb;) {   // lo: this warp's first batch not yet hashed whole
    unsigned v = posts();
    if (!p.hall && v >= done) break;
    if (v < need(lo)) {
      __nanosleep(256);
      continue;
    }
    for (unsigned b = lo; b < nb && v >= need(b) && (p.hall || v < done); b += NHW) {
      if (__all_sync(~0u, visit(b, h.chunk)) && b == lo) lo += NHW;
      v = posts();
    }
  }
}

// TST's store thread (warp 9's lane 0). Each consumer warpgroup fills its two slots (at stg, slots 2 wg and 2 wg + 1) a box
// at a time, in its tiles' order: each of a tile's nseg segments as BN / 64 boxes of 64 prefix sums by 64 rows (prefix-sum
// tile b / 2's columns 64 (b % 2) on), then its O as BN / 32 boxes of 32 cells by 64 rows. Slot i's full barrier is at sbar
// + 8 i (the warpgroup's 4 warps), its empty one at sbar + 32 + 8 i. The thread stores the warpgroups' boxes in turn, frees
// a slot once the next store is issued and this one has read it, and at a tile's end, once its stores are done, makes the
// 8 posts its consumer warps would.
template <int BN>
WGD void store_tiles(const Args& p, uint32_t stg, uint32_t sbar, uint32_t* posted, int tiles_m, int tiles_n, int nseg) {
  if (p.O) tma_prefetch(&p.tso);
  if (nseg) tma_prefetch(&p.tsw);
  const int items = nseg + (p.O ? 1 : 0);
  unsigned n = 0;   // the boxes each warpgroup has filled
  int prev = -1;    // the slot of the last store issued, while it's held
  for (int t = blockIdx.x; t < tiles_m * tiles_n; t += gridDim.x) {
    int tm, tn;
    raster(t, tiles_m, tiles_n, p.group, tm, tn);
    for (int it = 0; it < items; it++) {
      const bool o = it == nseg;
      for (int b = 0; b < (o ? BN / 32 : BN / 64); b++, n++)
        for (int g = 0; g < 2; g++) {
          const int sl = 2 * g + (int)(n & 1);
          mbar_wait(sbar + 8 * sl, (n >> 1) & 1);
          const uint32_t src = stg + sl * TSLOT;
          if (o) {
            tma_store(&p.tso, src, tn * BN + 32 * b, tm * TM + 64 * g);
          } else {
            const unsigned ps = (unsigned)tm * (p.N / TW) + (unsigned)tn * (BN / TW) + b / 2;
            tma_store(&p.tsw, src, 64 * (b % 2), (int)((ps * p.S + it) * TW) + 64 * g);
          }
          bulk_commit();
          bulk_wait_read<1>();
          if (prev >= 0) mbar_arrive(sbar + 32 + 8 * prev);
          prev = sl;
        }
    }
    bulk_wait<0>();
    if (prev >= 0) mbar_arrive(sbar + 32 + 8 * prev);
    prev = -1;
    fence_async_global();
    red_rel(posted, 8);
  }
}

template <int BN, int CM, int RP, class H = NoHash, int HR = 0, bool TST = false>
__device__ void body(const Args& p, const H& h = H{}) {
  using T = Tile<BN>;
  constexpr int NWG = T::NWG, STAGES = T::STAGES, NA = T::NA;
  constexpr bool HW = RP == 4 || RP == 5;   // hash warps beside the producer (static schedule, posted tiles)
  static_assert(RP != 5 || (HR % 8 == 0 && HR >= 24 && HR <= 128), "setmaxnreg takes multiples of 8 from 24");
  static_assert(!TST || RP == 5, "TST is RP 5's");
  extern __shared__ __align__(1024) uint8_t smem[];
  // TST: the slots at stg, their mbarriers at sbar (store_tiles')
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, stg = base + STAGES * T::STAGE, bars = stg + (TST ? 4 * TSLOT : 0),
                 sbar = bars + 16 * STAGES;
  int* tab = reinterpret_cast<int*>(smem + (sbar - smem_at(smem)) + (TST ? 64 : 0));
  int* const stile = tab + (TST ? 0 : MAX_TT);   // the tile a stage starts (written with its first k-tile), or -1: no more tiles
  uint32_t* const posted = reinterpret_cast<uint32_t*>(stile + STAGES);   // RP 4, 5: consumer warps' stored tiles
  const int M = p.M, N = p.N, KT = p.K / BK, tiles_m = M / TM, tiles_n = N / BN, tiles = tiles_m * tiles_n;
  const bool proto = RP != 5 && p.mode == 1;
  const int TT = proto ? p.TT + p.KQ : 0, steps = TT + KT;
  const bool segs = (proto || (RP == 5 && p.mode == 1)) && !(p.diag & 1);
  const int kpl = p.L / BK;
  const int tid = threadIdx.x, wg = tid / 128;
  if (tid == 0) {
    *posted = 0;
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), NWG * 4); }
    if constexpr (TST)
      for (int i = 0; i < 4; i++) { mbar_init(sbar + 8 * i, 4); mbar_init(sbar + 32 + 8 * i, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (TT) {
      int n = 0;
      bool qdone = p.KQ == 0;
      for (int i = 0; i < p.TT; i++) {
        const TailStep ts = tail_step(i, p.lp, p.dz, p.dy);
        if (!qdone && ts.e == 0) {
          for (int q = 0; q < p.KQ; q++) tab[n++] = q | TAIL_Q | (q == p.KQ - 1 ? TAIL_XM128 : TAIL_NONE) << 16;
          qdone = true;
        }
        int ev = !ts.end ? TAIL_NONE : ts.e > 1 ? TAIL_X128 : ts.e == 1 ? TAIL_XM128 : TAIL_CAPTURE;
        if (ev == TAIL_XM128 && p.KQ) ev = TAIL_NONE;
        tab[n++] = ts.kt | ev << 16;
      }
    }
  }
  __syncthreads();

  auto next = [&](int t) {
    const unsigned q = !HW && p.tctr ? gridDim.x + (atomicAdd(p.tctr, 1u) - p.tbase) : (unsigned)t + gridDim.x;
    return q < (unsigned)tiles ? (int)q : tiles;
  };
  auto prefetch = [&]() {
    wgp::tma_prefetch(&p.a);
    wgp::tma_prefetch(&p.b);
    if (TT) { wgp::tma_prefetch(&p.pa); wgp::tma_prefetch(&p.pb); }
    if (TT && p.KQ) wgp::tma_prefetch(&p.q);
  };
  // step i of tile (tm, tn) into the stage at st, its bytes counted on full
  auto load = [&](uint32_t st, uint32_t full, int i, int tm, int tn) {
    mbar_expect_tx(full, T::STAGE);
    if (i < TT) {
      const int e = tab[i], kt = e & 0xFFFF;
      if (e & TAIL_Q) {
        wgp::tma_load(st, &p.a, full, kt * BK, tm * TM);
        wgp::tma_load(st + T::A_BYTES, &p.q, full, kt * BK, tn * BN);
      } else {
        wgp::tma_load(st, &p.pa, full, kt * BK, tm * TM);
        wgp::tma_load(st + T::A_BYTES, &p.pb, full, kt * BK, tn * BN);
      }
    } else {
      wgp::tma_load(st, &p.a, full, (i - TT) * BK, tm * TM);
      wgp::tma_load(st + T::A_BYTES, &p.b, full, (i - TT) * BK, tn * BN);
    }
  };

  if (RP != 3 && wg >= NWG) {
    if constexpr (RP == 0) regs_dec<40>();
    if constexpr (RP == 4) regs_dec<104>();
    if constexpr (RP == 5) regs_dec<HR>();
    if constexpr (HW)
      if (tid >= NWG * 128 + (TST ? 64 : 32)) {
        const int ntile = (int)blockIdx.x < tiles ? (tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
        hash_warps<BN, threads<RP>() / 32 - 9 - TST, 9 + TST>(p, h, posted, tiles_m, tiles_n, ntile);
      }
    if constexpr (TST)
      if (tid == NWG * 128 + 32) store_tiles<BN>(p, stg, sbar, posted, tiles_m, tiles_n, segs ? (KT + kpl - 1) / kpl : 0);
    if (tid == NWG * 128) {
      prefetch();
      int s = 0;
      uint32_t ph = 0;
      for (int t = blockIdx.x; t < tiles; t = next(t)) {
        int tm, tn;
        raster(t, tiles_m, tiles_n, p.group, tm, tn);
        for (int i = 0; i < steps; i++) {
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          if (i == 0) stile[s] = t;
          load(base + s * T::STAGE, bars + 8 * s, i, tm, tn);
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
      mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
      stile[s] = -1;
      mbar_arrive(bars + 8 * s);
    }
    asm volatile("exit;");
  }
  if constexpr (RP == 0) regs_inc<232>();
  if constexpr (RP == 4) regs_inc<200>();
  if constexpr (RP == 5) regs_inc<256 - HR>();
  // RP 4, 5: the warp's stores of the tile are done (bar.warp.sync orders its lanes' stores before lane 0's release)
  auto post = [&]() {
    if constexpr (HW && !TST) {
      __syncwarp();
      if (tid % 32 == 0) red_rel(posted, 1);
    }
  };
  // TST: the warpgroup's next slot once its last store has read it, and filled: the warp's writes to the async proxy and
  // its arrival on the slot's full barrier
  uint8_t* const sbuf = smem + (stg - smem_at(smem));
  unsigned nb = 0;   // the boxes this warpgroup has filled
  auto slot = [&]() {
    const int sl = 2 * wg + (int)(nb & 1);
    mbar_wait(sbar + 32 + 8 * sl, ((nb >> 1) & 1) ^ 1);
    return sl;
  };
  auto filled = [&](int sl) {
    fence_async_smem();
    __syncwarp();
    if (tid % 32 == 0) mbar_arrive(sbar + 8 * sl);
    nb++;
  };

  // RP 3: thread 0 is the producer too, between its own steps: issue() fills the next stage (waiting for it to be free),
  // pump() every stage already free up to STAGES steps past the consumers' cg, need() whatever step cg still lacks
  int pt = blockIdx.x, pi = 0, ps = 0, pg = 0, ptm = 0, ptn = 0;
  uint32_t pph = 0;
  bool pfin = false;
  auto issue = [&]() {
    mbar_wait(bars + 8 * (STAGES + ps), pph ^ 1);
    if (pt >= tiles) {
      stile[ps] = -1;
      mbar_arrive(bars + 8 * ps);
      pfin = true;
    } else {
      if (pi == 0) { stile[ps] = pt; raster(pt, tiles_m, tiles_n, p.group, ptm, ptn); }
      load(base + ps * T::STAGE, bars + 8 * ps, pi, ptm, ptn);
      if (++pi == steps) { pi = 0; pt = next(pt); }
    }
    pg++;
    if (++ps == STAGES) { ps = 0; pph ^= 1; }
  };
  auto pump = [&](int cg) {
    if constexpr (RP == 3) {
      if (tid == 0)
        while (!pfin && pg < cg + STAGES && mbar_test(bars + 8 * (STAGES + ps), pph ^ 1)) issue();
      __syncwarp();
    }
  };
  auto need = [&](int cg) {
    if constexpr (RP == 3) {
      if (tid == 0)
        while (pg <= cg) issue();
      __syncwarp();
    }
  };
  if constexpr (RP == 3) {
    if (tid == 0) prefetch();
    pump(0);
  }

  const int x = tid % 128, lane = tid % 32;
  const int rloc = wg * 64 + 16 * (x / 32) + (x % 32) / 4, cq = 2 * (x % 4);   // the thread's first row and column in the tile
  int s = 0;
  uint32_t ph = 0;
  int32_t d[NA];
  uint32_t cp[CM == 0 ? NA / 2 : 1];
  uint32_t* const cs = CM == 1 ? p.cbuf + (size_t)blockIdx.x * (TM * BN / 2) + tid : nullptr;   // the thread's packed tau halves
  // TST: the thread's first row in its warpgroup's boxes, of 128 bytes each (its second is 8 on: 1024 bytes); a box's
  // 16-byte unit u of row r is at u ^ (r % 8), as SWIZZLE_128B
  const int br = rloc - 64 * wg, bx = br % 8;
  int cg = 0;   // the steps consumed (RP 3's producer's reference)
  for (;;) {
    need(cg);
    mbar_wait_warp(bars + 8 * s, ph);
    const int t = stile[s];
    if (t < 0) break;
    int tm, tn;
    raster(t, tiles_m, tiles_n, p.group, tm, tn);
    int pend = -1;            // the stage whose release waits on its MMAs
    int segk = 0, segn = 0;   // the main k-tiles into the open segment, and the segments stored
    for (int i = 0; i < steps; i++) {
      need(cg);
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
      WGP_FENCE;
      imma_atom<BN, RP == 5>(d, a0, b0, i > 0);
      imma_atom<BN, RP == 5>(d, a0 + 32, b0 + 32, 1);
      imma_atom<BN, RP == 5>(d, a0 + 64, b0 + 64, 1);
      imma_atom<BN, RP == 5>(d, a0 + 96, b0 + 96, 1);
      WGP_COMMIT;
      WGP_WAIT(1);
      if (pend >= 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
      pend = s;
      if (++s == STAGES) { s = 0; ph ^= 1; }
      pump(++cg);
      // the events that read the accumulators: a tail digit's end, and a segment's end
      int scale = 0;
      bool capture = false;
      int seg = 0;
      if (i < TT) {
        const int ev = (tab[i] >> 16) & 15;
        if (ev == TAIL_X128) scale = 128;
        else if (ev == TAIL_XM128) scale = -128;
        else capture = ev == TAIL_CAPTURE;
      } else if (segs && (++segk == kpl || i + 1 == steps)) {
        seg = ++segn;
        segk = 0;
      }
      if (!scale && !capture && !seg) continue;
      WGP_WAIT(0);
      acc_fence<NA>(d);
      if (lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
      pend = -1;
      pump(cg);
      if (scale) {
#pragma unroll
        for (int j = 0; j < NA; j++) d[j] = (int32_t)((uint32_t)d[j] * (uint32_t)scale);
      } else if (capture) {
#pragma unroll
        for (int j = 0; j < NA / 2; j++) {
          const uint32_t w = __byte_perm((uint32_t)d[2 * j], (uint32_t)d[2 * j + 1], 0x5410);
          if constexpr (CM == 0) cp[j] = w;
          else cs[j * (NWG * 128)] = w;
        }
      } else if constexpr (TST) {
        // T_s (no tail: acc's low halves) a box at a time: box b holds columns 64 b to 64 b + 63, of which the thread's
        // pair 8 jj + cq is unit jj
#pragma unroll
        for (int b = 0; b < BN / 64; b++) {
          const int sl = slot();
          uint8_t* const q = sbuf + sl * TSLOT + 128 * br + 4 * (x % 4);
#pragma unroll
          for (int jj = 0; jj < 8; jj++) {
            const int j = 8 * b + jj, u = 16 * (jj ^ bx);
            *reinterpret_cast<uint32_t*>(q + u) = __byte_perm((uint32_t)d[4 * j], (uint32_t)d[4 * j + 1], 0x5410);
            *reinterpret_cast<uint32_t*>(q + 1024 + u) = __byte_perm((uint32_t)d[4 * j + 2], (uint32_t)d[4 * j + 3], 0x5410);
          }
          filled(sl);
        }
      } else {
        // T_s: the low halves of acc - tau, a pair of columns a 32-bit store, into the 128 x 128 prefix-sum tiles the tile covers
        // (S segments apart). Column 8 j + cq is prefix-sum tile j / 16's column 8 (j % 16) + cq, so every address is one of two
        // bases plus a constant (cq is even, so the halving distributes).
        uint32_t* const w0 = reinterpret_cast<uint32_t*>(p.words) +
                             ((((size_t)tm * (N / TW) + (size_t)tn * (BN / TW)) * p.S + (seg - 1)) * (TW * TW) +
                              (size_t)rloc * TW + cq) / 2;
        uint32_t* const w1 = w0 + (size_t)p.S * (TW * TW / 2);
#pragma unroll
        for (int j = 0; j < NA / 4; j++) {
          uint32_t* const w = j < TW / 8 ? w0 : w1;
#pragma unroll
          for (int h = 0; h < 2; h++) {
            uint32_t c2 = 0;
            if (TT) {
              if constexpr (CM == 0) c2 = cp[2 * j + h];
              else c2 = cs[(2 * j + h) * (NWG * 128)];
            }
            const uint32_t lo = (uint32_t)d[4 * j + 2 * h] - c2, hi = (uint32_t)d[4 * j + 2 * h + 1] - (c2 >> 16);
            w[(8 * h * TW + 8 * (j % (TW / 8))) / 2] = __byte_perm(lo, hi, 0x5410);
          }
          if (j % 8 == 7) ld_fence();
        }
      }
    }
    // unconditional: ptxas injects its own waits wherever a path could read the accumulators with a group in flight
    WGP_WAIT(0);
    acc_fence<NA>(d);
    if (pend >= 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
    pump(cg);
    if (!p.O) {   // the prefix sums alone (a served call's segments 1 .. S - 1)
      post();
      continue;
    }
    if constexpr (TST) {
      // O a box at a time: box b holds columns 32 b to 32 b + 31, of which the thread's pair 8 jj + cq is unit 2 jj + cq / 4's
      // half cq / 2 % 2
#pragma unroll
      for (int b = 0; b < BN / 32; b++) {
        const int sl = slot();
        uint8_t* const q = sbuf + sl * TSLOT + 128 * br + 8 * (x % 2);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int j = 4 * b + jj, u = 16 * ((2 * jj + (x % 4) / 2) ^ bx);
          *reinterpret_cast<int2*>(q + u) = make_int2(d[4 * j], d[4 * j + 1]);
          *reinterpret_cast<int2*>(q + 1024 + u) = make_int2(d[4 * j + 2], d[4 * j + 3]);
        }
        filled(sl);
      }
      continue;
    }

    const int row = tm * TM + rloc, col0 = tn * BN + cq;
    // every address below is a base plus a constant (row and col0 are even)
    if (RP != 5 && p.mode == 2) {
      const float r0 = p.sa[row], r1 = p.sa[row + 8];
      uint32_t* const y0 = reinterpret_cast<uint32_t*>(p.O) + ((size_t)row * N + col0) / 2;
      uint32_t* const y1 = y0 + (size_t)4 * N;
      const float* const sw = p.sw + col0;
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        const float2 c = *reinterpret_cast<const float2*>(sw + 8 * j);
        y0[4 * j] = bf16_pair(d[4 * j], d[4 * j + 1], r0, c.x, c.y);
        y1[4 * j] = bf16_pair(d[4 * j + 2], d[4 * j + 3], r1, c.x, c.y);
      }
      post();
      continue;
    }
    if (proto && p.lA && !(p.diag & 4)) {
      // w_X B~^T a wrap at a time (wrap 7, in place of dO): each of the thread's two rows' wraps adds s 256 B~[col, k] to
      // its cells, B~^T's row k two columns a load. The loop runs to the warp's most wraps (ld_fence is a warp barrier),
      // a lane's loads predicated on its own row's count.
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int r = row + 8 * h, na = p.cA[r], nmax = __reduce_max_sync(0xFFFFFFFFu, na);
        const int* const la = p.lA + (size_t)r * p.capA;
        for (int q = 0; q < nmax; q++) {
          const bool on = q < na;
          const int e = on ? __ldg(la + q) : 0;
          const uint32_t* const b = reinterpret_cast<const uint32_t*>(p.bt + (size_t)(e >> 1) * N + col0);
          const uint32_t s = !on ? 0u : e & 1 ? 0u - 256u : 256u;
#pragma unroll
          for (int j = 0; j < NA / 4; j++) {
            const uint32_t v = on ? __ldg(b + 4 * j) : 0u;
            d[4 * j + 2 * h] = (int32_t)((uint32_t)d[4 * j + 2 * h] + s * (uint32_t)(int32_t)(int16_t)(v & 0xFFFF));
            d[4 * j + 2 * h + 1] = (int32_t)((uint32_t)d[4 * j + 2 * h + 1] + s * (uint32_t)(int32_t)(int16_t)(v >> 16));
            if (j % 8 == 7) ld_fence();
          }
        }
      }
    }
    pump(cg);
    const bool add = proto && !(p.diag & 4) && !p.lA;
    int* const o0 = p.O + (size_t)row * N + col0;
    int* const o1 = o0 + (size_t)8 * N;
    const int* const e0p = p.dO + (size_t)row * N + col0;
    const int* const e1p = e0p + (size_t)8 * N;
#pragma unroll
    for (int j = 0; j < NA / 4; j++) {
      int2 lo = make_int2(d[4 * j], d[4 * j + 1]), hi = make_int2(d[4 * j + 2], d[4 * j + 3]);
      if (add) {
        const int2 e0 = __ldg(reinterpret_cast<const int2*>(e0p + 8 * j)), e1 = __ldg(reinterpret_cast<const int2*>(e1p + 8 * j));
        lo.x = (int)((uint32_t)lo.x + ((uint32_t)e0.x << 8)); lo.y = (int)((uint32_t)lo.y + ((uint32_t)e0.y << 8));
        hi.x = (int)((uint32_t)hi.x + ((uint32_t)e1.x << 8)); hi.y = (int)((uint32_t)hi.y + ((uint32_t)e1.y << 8));
      }
      *reinterpret_cast<int2*>(o0 + 8 * j) = lo;
      *reinterpret_cast<int2*>(o1 + 8 * j) = hi;
      if (j % 8 == 7) ld_fence();
    }
    post();
  }
}

// RP 6 and 7's hashing: the CTA's O rows in batches of 32 (a row a lane) in its tiles' order, Lg = h.lpt() a 128 x 128
// tile, a batch claimed (ctl[2], the next one) by whichever warp comes first once every tile it reads is posted (ctl[w]:
// warpgroup w's warps' posts; tile i is warpgroup i % 2's (i / 2)-th). A warp keeps its batch until it's hashed whole,
// on from its rows' chaining values (H's flags) across its own mainloops; it starts no SHA-512 chunk once stop() holds,
// and when every batch posted is taken it leaves (wait: waits for more, until stop()).
template <class H, class Stop>
WGD void pp_hash(const Args& p, const H& h, uint32_t* ctl, int& cur, int ntile, int tiles_m, int tiles_n, const Stop& stop,
                 bool wait) {
  const unsigned lane = threadIdx.x % 32, Lg = h.lpt(), nq = (unsigned)ntile * Lg, nb = (nq + 31) / 32;
  auto posted = [&](unsigned i) { return ld_acq(ctl + (i & 1)) >= 4 * (i / 2 + 1); };
  auto claim = [&]() {
    int got = -1;
    if (lane == 0)
      for (unsigned b = ld_vol(ctl + 2); b < nb;) {
        const unsigned i1 = min(32 * b + 31, nq - 1) / Lg;   // the tiles below i1 of both parities are posted with i1, i1 - 1
        if (!posted(i1) || (i1 && !posted(i1 - 1))) break;
        const unsigned seen = atomicCAS(ctl + 2, b, b + 1);
        if (seen == b) { got = (int)b; break; }
        b = seen;
      }
    got = __shfl_sync(~0u, got, 0);
    if (got >= 0) { (void)ld_acq(ctl); (void)ld_acq(ctl + 1); }   // each lane's own acquire of the posts its rows follow
    return got;
  };
  for (;;) {
    if (__shfl_sync(~0u, (int)stop(), 0)) return;
    if (cur < 0 && (cur = claim()) < 0) {
      if (!wait || ld_vol(ctl + 2) >= nb) return;
      __nanosleep(256);
      continue;
    }
    const unsigned q = 32 * (unsigned)cur + lane;
    bool whole = true;
    if (q < nq) {
      int tm, tn;
      raster((int)(blockIdx.x + q / Lg * gridDim.x), tiles_m, tiles_n, p.group, tm, tn);
      whole = h((unsigned long long)tm * (p.N / TW) + (unsigned long long)tn, q % Lg, stop, ~0u);
    }
    if (__all_sync(~0u, whole)) cur = -1;
  }
}

// RP 6 and 7 (above): 128 x 128 tiles, the CTA's i-th (static schedule) warpgroup i % 2's, the ring's k-tiles in the
// tiles' order (k-tile g = i KT + k is stage g % STAGES's (g / STAGES)-th fill, its empty barrier one warpgroup's 4 warps).
// ctl[3 + w]: the k-tiles warpgroup w has issued (g + 1), so a consumer whose next tile starts at g0 hashes until the two
// have issued g0 - ppm and starts its mainloop at g0 (with no tile left, the CTA's last: the tensor pipe is never kept
// waiting past a chunk), and RP 7's hash warps until both mainloops end. H NoHash: the ping-pong GEMM alone.
template <int RP, class H> __device__ void body_pp(const Args& p, const H& h) {
  static_assert(RP == 6 || RP == 7, "the ping-pong plans");
  using T = Tile<128>;
  constexpr int BN = 128, STAGES = T::STAGES;
  constexpr bool HASH = !std::is_same<H, NoHash>::value;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  uint32_t* const ctl = reinterpret_cast<uint32_t*>(smem + (base - smem_at(smem)) + STAGES * T::STAGE + 16 * STAGES);
  const int KT = p.K / BK, tiles_m = p.M / TM, tiles_n = p.N / BN, tiles = tiles_m * tiles_n;
  const int tid = threadIdx.x, wg = tid / 128;
  const int ntile = (int)blockIdx.x < tiles ? (tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
  const uint32_t gend = (uint32_t)(ntile * KT);
  auto issued = [&]() { return max(ld_vol(ctl + 3), ld_vol(ctl + 4)); };
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), 4); }
    for (int c = 0; c < 5; c++) ctl[c] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == 2) {
    if constexpr (RP == 7) regs_dec<104>();
    else regs_dec<40>();
    if (tid == 256) {
      wgp::tma_prefetch(&p.a);
      wgp::tma_prefetch(&p.b);
      int s = 0;
      uint32_t ph = 0;
      for (int i = 0; i < ntile; i++) {
        int tm, tn;
        raster((int)blockIdx.x + i * (int)gridDim.x, tiles_m, tiles_n, p.group, tm, tn);
        for (int k = 0; k < KT; k++) {
          const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
          mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          mbar_expect_tx(full, T::STAGE);
          wgp::tma_load(st, &p.a, full, k * BK, tm * TM);
          wgp::tma_load(st + T::A_BYTES, &p.b, full, k * BK, tn * BN);
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
      }
    } else if constexpr (RP == 7 && HASH) {
      if (tid >= 256 + 32) {
        int cur = -1;
        pp_hash(p, h, ctl, cur, ntile, tiles_m, tiles_n, [&]() { return issued() >= gend; }, true);
      }
    }
    asm volatile("exit;");
  }
  if constexpr (RP == 7) regs_inc<200>();
  else regs_inc<232>();
  const int w = wg, x = tid % 128, lane = tid % 32;
  int32_t d[128];
  int cur = -1;
  for (int i = w;; i += 2) {
    const uint32_t g0 = (uint32_t)(min(i, ntile) * KT);
    if constexpr (HASH)
      pp_hash(p, h, ctl, cur, ntile, tiles_m, tiles_n, [&]() { return issued() + (uint32_t)p.ppm >= g0; }, i >= ntile);
    if (i >= ntile) break;
    // the mainloops take turns (CUTLASS's order barrier): until the other warpgroup has issued k-tile g0 - 1, this tile's
    // first stages' full barriers can sit a phase behind, and a parity wait passes on the other tile's fill. The poll
    // sleeps: a tight loop of shared loads takes the shared memory pipe the other warpgroup's wgmma reads its operands on
    while (issued() < g0) __nanosleep(20);
    __syncwarp();
    int tm, tn;
    raster((int)blockIdx.x + i * (int)gridDim.x, tiles_m, tiles_n, p.group, tm, tn);
    int s = (int)(g0 % STAGES), pend = -1;
    uint32_t ph = (g0 / STAGES) & 1;
    // imma's operands are "+r", so without this the last tile's accumulators would be live through pp_hash
#pragma unroll
    for (int j = 0; j < 128; j++) d[j] = 0;
    for (int k = 0; k < KT; k++) {
      mbar_wait_warp(bars + 8 * s, ph);
      const uint32_t a0 = base + s * T::STAGE, b0 = a0 + T::A_BYTES;
      WGP_FENCE;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        imma<128>(d, desc_sw128(a0 + 32 * q), desc_sw128(b0 + 32 * q), k > 0 || q > 0);
        imma<128>(d + 64, desc_sw128(a0 + 64 * BK + 32 * q), desc_sw128(b0 + 32 * q), k > 0 || q > 0);
      }
      WGP_COMMIT;
      if (x == 0) st_vol(ctl + 3 + w, g0 + k + 1);
      WGP_WAIT(1);
      if (pend >= 0 && lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
      pend = s;
      if (++s == STAGES) { s = 0; ph ^= 1; }
    }
    WGP_WAIT(0);
    acc_fence<128>(d);
    if (lane == 0) mbar_arrive(bars + 8 * (STAGES + pend));
    const int col0 = tn * BN + 2 * (x % 4);
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      int* const o0 = p.O + (size_t)(tm * TM + 64 * hh + 16 * (x / 32) + (x % 32) / 4) * p.N + col0;
      int* const o1 = o0 + (size_t)8 * p.N;
#pragma unroll
      for (int j = 0; j < 16; j++) {
        *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[64 * hh + 4 * j], d[64 * hh + 4 * j + 1]);
        *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[64 * hh + 4 * j + 2], d[64 * hh + 4 * j + 3]);
        if (j % 8 == 7) ld_fence();
      }
    }
    __syncwarp();   // the warp's stores before lane 0's release
    if (lane == 0) red_rel(ctl + w, 1);
  }
}

template <int BN, int CM>
__global__ void __launch_bounds__(threads<0>(), 1) gemm(const __grid_constant__ Args p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body<BN, CM, 0>(p);
#endif
}
// RP 3 keeps tau in L2 (CM 1: CM 0's 64 registers of it don't fit)
template <int BN>
__global__ void __maxnreg__(200) gemm_r3(const __grid_constant__ Args p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body<BN, 1, 3>(p);
#endif
}
// RP 4: the GEMM with the commit's row hashes in its producer warpgroup (CM 1, as RP 3)
template <int BN, class H>
__global__ void __launch_bounds__(threads<4>(), 1) gemm_r4(const __grid_constant__ Args p, const __grid_constant__ H h) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body<BN, 1, 4, H>(p, h);
#endif
}
// RP 5: the plain product or the prefix sums alone, with the commit's row hashes in a hash warpgroup and the producer's (HR
// registers, one of R5_SPLITS); TST: stored through shared memory with TMA (Tile's SMEM_TST)
template <int BN, class H, int HR, bool TST = false>
__global__ void __launch_bounds__(threads<5>(), 1) gemm_r5(const __grid_constant__ Args p, const __grid_constant__ H h) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body<BN, 1, 5, H, HR, TST>(p, h);
#endif
}
// RP 6, 7: the ping-pong GEMM (mode 0), 128 x 128 tiles, hashing O rows with h between its warpgroups' mainloops
template <int RP, class H>
__global__ void __launch_bounds__(threads<6>(), 1) gemm_pp(const __grid_constant__ Args p, const __grid_constant__ H h) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  body_pp<RP, H>(p, h);
#endif
}

}  // namespace wgp
