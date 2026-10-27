// The exact int8 GEMM O = A W^T on the H100 and H200 (sm_90a), s32 out, on operands the filler draws in a 2-bit range
// (bytes in {0, 1, 2, 3}, or any int8: nothing here depends on the range but the sums' size).  A (M x K) and W (N x K) are
// row-major int8, depth contiguous, streamed from global memory by TMA in boxes of 128 bytes of depth with SWIZZLE_128B and
// multiplied by warpgroup MMAs (wgmma m64n256k32 .s32.s8.s8) into one s32 accumulator over all of K, exact while
// |sum| < 2^31 (K < 2^17 for any int8, K < 2^31 / 9 for {0..3}).  O is torch._int_mm(A, W.t())'s int32 output, bit for bit.
//
// s8_linear's s8_wgmma.cu (persistent, raster groups of tile rows, one CTA an SM, a W box shared by a pair of M-adjacent CTAs
// through TMA multicast with CL = 2) with three schedules, each a 64 NWG x 256 tile:
//   u2mm_ws_m<CL>    NWG 2 consumer warpgroups and a producer warpgroup whose first thread issues every TMA (s8_wgmma.cu's)
//   u2mm_ci<NWG>_m<CL>  NWG 1, 2 or 3 consumer warpgroups and no producer: consumer thread 0 issues each stage's TMA as the
//                    stage before it is released, so the CTA holds 128 NWG threads (ci1, ci2: at most 168 registers each,
//                    which leaves an SM's registers for a 256-thread CTA of hash_k beside it; ci3 a 192 x 256 tile, the whole
//                    register file)
// M must be a multiple of 64 NWG CL, N of 256 and K of 128 (bench.py checks): the filler picks its shape.  The epilogue
// stores each accumulator pair as one 8-byte word.
//
// hash_k: SHA-512 compressions (sha512_rows' sx::compress, the default V 0, rolled), each thread chaining its own state, for
// the rate beside the GEMM (CTAs of 256 threads on the GEMM's SMs).  stamp_k: each SM's %clock64 and %globaltimer, one CTA an
// SM (its shared memory keeps a second off), so the SM clock over a span of launches is the difference of two stamps.
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "../s8_linear/s8_acc.h"
#include "../sha512_rows/sha512x.cuh"

#define DEV __device__ __forceinline__

// a CTA's span: %globaltimer (ns) and %clock64 at its start and end, its SM and the tiles it did
struct Stamp {
  unsigned long long t0, t1, c0, c1, sm, tiles;
};

// one launch's arguments, a __grid_constant__ parameter (bench.py's `Params`)
struct alignas(128) Params {
  CUtensorMap a, b;    // A: M rows of K bytes, boxes of 64 NWG rows; W: N rows, boxes of 256 / CL rows
  int32_t* o;          // O [M, N], s32, row-major
  Stamp* st;           // a stamp a CTA, or null
  int M, N, K, group;  // group: tile rows a raster group
  unsigned* started;   // counted up by each CTA as it starts, or null
};
static_assert(offsetof(Params, o) == 256 && offsetof(Params, M) == 272 && offsetof(Params, started) == 288 &&
                  sizeof(Params) == 384,
              "bench.py's Params");

DEV unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DEV uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// ---------------------------------------------------------------- mbarriers, TMA, wgmma (s8_wgmma.cu's)
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
DEV void mbar_wait_warp(uint32_t bar, uint32_t parity) {
  mbar_wait(bar, parity);
  __syncwarp();
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load(uint32_t dst, const void* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
               "l"(map), "r"(bar), "r"(c0), "r"(c1)
               : "memory");
}
DEV void tma_load_mc(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask)
               : "memory");
}
DEV uint32_t cta_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEV void mbar_arrive_at(uint32_t bar, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(r) : "memory");
}
DEV void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAITC:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra WAITC;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
template <int R> DEV void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R> DEV void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

DEV uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0): one s8 atom (k = 32) of a 64 x 256 warpgroup tile.  Accumulator i of thread x is row
// 16 (x / 32) + (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
DEV void imma(int32_t* d, uint64_t da, uint64_t db, uint32_t sd) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
               ", %128, %129, p;\n}" : ACC128_R : "l"(da), "l"(db), "r"(sd));
}
template <int NA> DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

// ---------------------------------------------------------------- the GEMM
constexpr int BN = 256, BK = 128, NA = BN / 2, SMEM_BUDGET = 224 * 1024, MAX_STAGES = 8;

template <int NWG, int PROD> struct Tile {
  static_assert(NWG >= 1 && NWG <= 3 && (PROD == 0 || NWG == 2), "1 to 3 consumer warpgroups; a producer beside 2");
  static constexpr int TM = 64 * NWG, NT = 128 * (NWG + PROD);
  static constexpr uint32_t A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
  static constexpr int STAGES = SMEM_BUDGET / STAGE < MAX_STAGES ? SMEM_BUDGET / STAGE : MAX_STAGES;
  // the stages from a 1,024-byte-aligned base (SWIZZLE_128B's period), then full[s] and empty[s]
  static constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES;
  static_assert(STAGE % 1024 == 0 && A_BYTES % 1024 == 0 && SMEM <= 227 * 1024, "boxes 1,024-byte aligned; an SM's shared memory");
};

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

// a consumer warp's release of stage `bar`: to its own CTA, and with CL = 2 to both CTAs of the pair, each of which writes
// half of W's box into this CTA's stage
template <int CL> DEV void release(uint32_t bar) {
  if constexpr (CL == 1) {
    mbar_arrive(bar);
  } else {
    mbar_arrive_at(bar, 0);
    mbar_arrive_at(bar, 1);
  }
}

// the next stage's TMA into slot s: this CTA's A box (TM rows) and its half of W's box, multicast to both CTAs with CL = 2
template <class T, int CL>
DEV void load_stage(const Params& p, uint32_t base, uint32_t bars, int s, int tm, int tn, int kb, uint32_t rank) {
  const uint32_t st = base + s * T::STAGE, full = bars + 8 * s;
  mbar_expect_tx(full, T::STAGE);
  tma_load(st, &p.a, full, kb * BK, tm * T::TM);
  const uint32_t bdst = st + T::A_BYTES + rank * (BN / CL) * BK;
  const int brow = tn * BN + rank * (BN / CL);
  if constexpr (CL == 1)
    tma_load(bdst, &p.b, full, kb * BK, brow);
  else
    tma_load_mc(bdst, &p.b, full, kb * BK, brow, 0x3);
}

template <int CL> DEV void wait_empty(uint32_t bar, uint32_t parity) {
  if constexpr (CL == 1)
    mbar_wait(bar, parity);
  else
    mbar_wait_cluster(bar, parity);
}

// a 64 x 256 accumulator of warpgroup wg into O: rows r and r + 8, each pair of adjacent columns one 8-byte store
DEV void store_tile(const Params& p, const int32_t* d, int tm, int tn, int TM, int wg, int x) {
  const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
  int32_t* o0 = p.o + (size_t)row * p.N + col0;
  int32_t* o1 = o0 + (size_t)8 * p.N;
#pragma unroll
  for (int j = 0; j < NA / 4; j++) {
    *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[4 * j], d[4 * j + 1]);
    *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[4 * j + 2], d[4 * j + 3]);
  }
}

DEV void stamp_start(const Params& p, unsigned long long& t0, unsigned long long& c0) {
  if (threadIdx.x == 0) {
    t0 = gtimer();
    c0 = clock64();
    if (p.started) atomicAdd(p.started, 1u);
  }
}
DEV void stamp_end(const Params& p, unsigned long long t0, unsigned long long c0, int tiles) {
  if (threadIdx.x == 0 && p.st) p.st[blockIdx.x] = Stamp{t0, gtimer(), c0, (unsigned long long)clock64(), smid(), (unsigned long long)tiles};
}

// one unit's main loop over K for a consumer warpgroup: each stage's four atoms issued before the stage before it is waited
// on and released; with ISSUE, thread 0 then refills the released slot (the stage STAGES ahead in this CTA's order)
template <class T, int CL, bool ISSUE, class Refill>
DEV void mainloop(int32_t* d, uint32_t base, uint32_t bars, int KB, int wg, int lane, int& s, uint32_t& ph, Refill refill) {
  constexpr int STAGES = T::STAGES;
  for (int kb = 0; kb < KB; kb++) {
    mbar_wait_warp(bars + 8 * s, ph);
    const uint32_t a0 = base + s * T::STAGE + wg * 64 * BK, b0 = base + s * T::STAGE + T::A_BYTES;
    WG_FENCE;
    imma(d, desc_sw128(a0), desc_sw128(b0), kb > 0);
    imma(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
    imma(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
    imma(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
    WG_COMMIT;
    WG_WAIT(1);
    if (kb > 0) {
      if (lane == 0) release<CL>(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
      if constexpr (ISSUE) refill();
    }
    if (++s == STAGES) { s = 0; ph ^= 1; }
  }
  WG_WAIT(0);
  acc_fence<NA>(d);
  if (lane == 0) release<CL>(bars + 8 * (STAGES + (s == 0 ? STAGES - 1 : s - 1)));
  if constexpr (ISSUE) refill();
}

template <int NWG, int PROD, int CL> DEV void body(const Params& p) {
  using T = Tile<NWG, PROD>;
  constexpr int STAGES = T::STAGES, TM = T::TM;
  static_assert(CL == 1 || CL == 2, "a CTA or a pair");
  extern __shared__ __align__(1024) uint8_t smem[];
  unsigned long long t0 = 0, c0 = 0;
  stamp_start(p, t0, c0);
  const uint32_t base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * T::STAGE;
  const int KB = p.K / BK, rows_u = p.M / TM / CL, tiles_n = p.N / BN, units = rows_u * tiles_n;
  const int tid = threadIdx.x, wg = tid / 128, x = tid % 128, lane = tid % 32;
  const uint32_t rank = CL == 1 ? 0 : cta_rank();
  const int first_u = blockIdx.x / CL, step_u = gridDim.x / CL;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(bars + 8 * s, 1); mbar_init(bars + 8 * (STAGES + s), CL * NWG * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (CL == 1)
    __syncthreads();
  else
    cluster_sync();

  if constexpr (PROD) {
    if (wg == NWG) {
      regs_dec<40>();
      if (tid == NWG * 128) {
        tma_prefetch(&p.a);
        tma_prefetch(&p.b);
        int s = 0;
        uint32_t ph = 0;
        for (int t = first_u; t < units; t += step_u) {
          int tm, tn;
          raster(t, rows_u, tiles_n, p.group, tm, tn);
          for (int kb = 0; kb < KB; kb++) {
            wait_empty<CL>(bars + 8 * (STAGES + s), ph ^ 1);
            load_stage<T, CL>(p, base, bars, s, tm * CL + rank, tn, kb, rank);
            if (++s == STAGES) { s = 0; ph ^= 1; }
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
    regs_inc<232>();
  }

  // the issuer's cursor (consumer-issued schedules): this CTA's next load, its unit, k-block, slot and phase
  int lu = first_u, lkb = 0, ltm = 0, ltn = 0, ls = 0;
  uint32_t lph = 0;
  auto refill = [&]() {
    if (tid != 0 || lu >= units) return;
    if (lkb == 0) raster(lu, rows_u, tiles_n, p.group, ltm, ltn);
    wait_empty<CL>(bars + 8 * (STAGES + ls), lph ^ 1);
    load_stage<T, CL>(p, base, bars, ls, ltm * CL + rank, ltn, lkb, rank);
    if (++ls == STAGES) { ls = 0; lph ^= 1; }
    if (++lkb == KB) { lkb = 0; lu += step_u; }
  };
  if constexpr (!PROD) {
    if (tid == 0) {
      tma_prefetch(&p.a);
      tma_prefetch(&p.b);
      for (int i = 0; i < STAGES; i++) refill();
    }
  }

  int s = 0, tiles = 0;
  uint32_t ph = 0;
  int32_t d[NA];
  for (int t = first_u; t < units; t += step_u) {
    int tm, tn;
    raster(t, rows_u, tiles_n, p.group, tm, tn);
    mainloop<T, CL, !PROD>(d, base, bars, KB, wg, lane, s, ph, refill);
    store_tile(p, d, tm * CL + rank, tn, TM, wg, x);
    tiles++;
  }
  // with CL = 2 and no producer (whose drain does it), the peer's multicasts into this CTA and its arrivals on this CTA's
  // barriers are done before either exits
  if constexpr (CL == 2 && !PROD) cluster_sync();
  // the consumers' stores are issued before the end stamp (named barrier 1: the producer has exited)
  asm volatile("bar.sync 1, %0;" ::"n"(128 * NWG) : "memory");
  stamp_end(p, t0, c0, tiles);
}

#define U2MM(NAME, NWG, PROD, CL, BOUNDS)                                                                           \
  extern "C" __global__ void BOUNDS NAME(const __grid_constant__ Params p) { body<NWG, PROD, CL>(p); }
#define LB(NT) __launch_bounds__(NT, 1)
#define LBC(NT) __cluster_dims__(2, 1, 1) __launch_bounds__(NT, 1)
// ci1 and ci2 at most 168 registers a thread: 128 NWG x 168 leaves a 256-thread hash_k CTA's 80 beside them
#define MR __maxnreg__(168)
#define MRC __cluster_dims__(2, 1, 1) __maxnreg__(168)
U2MM(u2mm_ws_m1, 2, 1, 1, LB(384))
U2MM(u2mm_ws_m2, 2, 1, 2, LBC(384))
U2MM(u2mm_ci1_m1, 1, 0, 1, MR)
U2MM(u2mm_ci1_m2, 1, 0, 2, MRC)
U2MM(u2mm_ci2_m1, 2, 0, 1, MR)
U2MM(u2mm_ci2_m2, 2, 0, 2, MRC)
U2MM(u2mm_ci3_m1, 3, 0, 1, LB(384))
U2MM(u2mm_ci3_m2, 3, 0, 2, LBC(384))

// what bench.py takes from this build, checked when it loads the cubin: Params' size and offsets, then (threads, shared
// memory, stages) of NWG 1, 2 and 3 without a producer and of 2 with one
extern "C" __global__ void u2mm_layout(int* out) {
  out[0] = sizeof(Params);
  out[1] = offsetof(Params, o);
  out[2] = offsetof(Params, M);
  out[3] = offsetof(Params, started);
  const int v[] = {Tile<1, 0>::NT, Tile<1, 0>::SMEM, Tile<1, 0>::STAGES, Tile<2, 0>::NT, Tile<2, 0>::SMEM, Tile<2, 0>::STAGES,
                   Tile<3, 0>::NT, Tile<3, 0>::SMEM, Tile<3, 0>::STAGES, Tile<2, 1>::NT, Tile<2, 1>::SMEM, Tile<2, 1>::STAGES};
  for (int i = 0; i < 12; i++) out[4 + i] = v[i];
}

// ---------------------------------------------------------------- beside the GEMM
// `reps` compressions a thread, each over a block whose first eight words are xored with the chaining value; io is written
// only for a state no input reaches, so the compressions stay live
extern "C" __global__ void __maxnreg__(80) hash_k(unsigned long long* io, int reps, Stamp* out) {
  unsigned long long t0 = 0, c0 = 0;
  if (threadIdx.x == 0) { t0 = gtimer(); c0 = clock64(); }
  const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = t * 16 + i;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] ^= st[i];
    sx::compress<0, 1>(st, w);
  }
  if (st[0] == 0x0123456789abcdefull) io[t] = st[1];
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{t0, gtimer(), c0, (unsigned long long)clock64(), smid(), (unsigned long long)reps};
}

// waits until `want` CTAs of a launch have started (Params.started), so a kernel launched after it on another stream finds
// that launch's CTAs resident
extern "C" __global__ void wait_k(volatile unsigned* started, unsigned want) {
  while (*started < want) {
  }
}

extern "C" __global__ void stamp_k(Stamp* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = Stamp{gtimer(), 0, (unsigned long long)clock64(), 0, smid(), 0};
}
