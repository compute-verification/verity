// What a SHA-512 warp costs the served GEMM's tensor pipe, taken apart: the served RP 5 CTA (wg_proto.cuh's gemm_r5: NM
// consumer warpgroups at Y registers and the rest at HR, the producer warp first among the rest, the MMAs wg_proto.cuh's
// own imma calls) with each of its parts optional, beside hash warps that run SHA-512 on registers alone
// (sx::compress<0, 1> on a chained block, as the co-issue probe's hash CTAs do), until the consumers are done.
//
// A configuration (CFGS) sets:
//   NWG    the CTA's warpgroups (gemm_r5: 4), NM of them consumers; setmaxnreg only moves registers within the launch's,
//          65,536 over the CTA's threads rounded down to 8 a thread (128 at 4 warpgroups, 96 at 5), so NM Y + (NWG - NM)
//          HR is at most NWG times that (512, 480); past it the consumers' setmaxnreg waits for ever
//   NM     consumer warpgroups: 1 (a 64 x BN tile) or 2 (128 x BN, gemm_r5's)
//   FORM   n128: one m64n128k32 a k32 atom (BN 128); alt: gemm_r5's 128 x 256 tile, two m64n128k32 an atom into
//          accumulators 0..63 and 64..127 in turn (imma_atom<256, true>); chain: the same eight MMAs a k-tile, the four
//          into 0..63 first, then the four into 64..127; n256: one m64n256k32 an atom, which takes 154 of the launch's
//          registers a thread whatever setmaxnreg gives the consumers, so at most 384 threads (168; 512 have 128)
//   flags  res: the operands resident in one stage of shared memory, the consumers free-running (no barriers);
//          bar: the producer warp hands the stages over on gemm_r5's full and empty mbarriers, loading nothing;
//          load: the producer loads each k-tile with TMA from A (M x K) and B (N x K), tiles in gemm_r5's static
//          schedule (every gridDim.x-th, rows then columns), or with RASTER = G in bands of G rows, columns within,
//          so the CTAs at work share their operands in L2 (rows then columns give each CTA a B tile of its own, BN x K
//          bytes, 2 MiB at K 8,192, and 132 of them don't fit in L2's 50 MB);
//          epi: at each tile's end (K / 128 k-tiles), wgmma.wait_group 0 and gemm_r5's direct O stores (s32, int2 a
//          pair) into an M x N buffer;
//          ns (resident only): the operands as the co-issue probe's unswizzled 8 x 16-byte core matrices in place of
//          TMA's SWIZZLE_128B rows; kg2 (resident only): two k-tiles a wgmma commit group in place of one;
//          mc2, mc4 (with load): clusters of 2 or 4 CTAs on tiles one above the other, each loading its part of B's box
//          into all of them (TMA multicast), so L2 serves B once a cluster; a stage is free once every CTA's consumers
//          release it, and the grid is the CTAs whose clusters fit on the SMs at once;
//          s5, s8 (with load): five or eight stages of shared memory in place of four
// NWG = NM: the consumers alone (the co-issue probe's 128-thread CTA), at the launch's registers, with no setmaxnreg.
// A launch's arm: hash warps 0 (the GEMM alone) or NHW of the CTA's free warps (the producer's warpgroup's other three,
// then the next warpgroups'): each count HWS lists (7, 11 and 99 by default), a count past the most it has taken as
// that most; and with resident operands, the hash alone at each (no consumers; each hash warp HREPS compressions).
//
// Per launch, from each CTA's consumer warp 0 (clock64 and %globaltimer at its first and last MMA) and its hash warps'
// compression counts: the SM clock (clock64 over %globaltimer); theta, the INT8 MACs a second over 989.4 T (4,096 a
// SM-clock at 1,830 MHz on 132 SMs), and theta_clk, a SM-clock over 4,096 (so theta_clk / theta is the clock's share of
// the loss); the hash's compressions a second over the bar (sx's 4.355 G/s), and hash_bar_clk, a SM-clock over 0.018029.
// Each counts every SM, an SM without a CTA as zero.
// The arms are medians of REPS launches. Then an energy item: the arm's launches back to back for ITEM_MS or longer, the
// board's energy over it (NVML's counter): mean watts, pJ a MAC and nJ a compression, and the most of its power sampled
// every 20 ms after its first second. The operands are bytes of one kind (OPS): s8, random int8, or u2, the 2-bit values 0..3 stored as int8,
// whose MMAs draw less power. Each line of output is JSON.
//
//   coresident [REPS] [KTILES] [HREPS] [ITEM_MS] [K]   (KTILES: a resident configuration's k-tiles a CTA; ITEM_MS 2000;
//                                                        K, the loaded operands' depth, 4096; CFGS in the environment:
//                                                        the configurations whose names start with one of its
//                                                        comma-separated prefixes, else all; OPS: the operands' kinds
//                                                        to run every configuration on, comma-separated, s8 by
//                                                        default, the hash alone with the first; HWS: the hash
//                                                        warps' counts, comma-separated)
//
// Build (sm_90a): nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -o coresident coresident.cu -lcuda
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

#include "../../torch_gpu/sha512_rows/sha512x.cuh"
#include "../gpu_kit/bench.cuh"
#include "wg_proto.cuh"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

using u64 = unsigned long long;
using u32 = uint32_t;
using wgp::BK;

enum Form { N128 = 0, ALT = 1, CHAIN = 2, N256 = 3 };
enum Flag { BAR = 1, LOAD = 2, EPI = 4, NOSW = 8, KG2 = 16, MC2 = 32, MC4 = 64 };
__host__ __device__ constexpr int cluster_of(int fl) { return fl & MC4 ? 4 : fl & MC2 ? 2 : 1; }
constexpr int ST(int n) { return n << 8; }   // a loaded form's stages, in the flags above the rest
__host__ __device__ constexpr int stages_of(int fl) { return fl >> 8 ? fl >> 8 : 4; }
constexpr double BAR_RATE = 0.018029;   // compressions a SM-clock: 4.355e9 / (132 x 1830e6)
constexpr double PEAK_MHZ = 1830;       // the clock of int8 wgmma's peak, 989.4 T MACs a second

__host__ __device__ inline u32 mix32(u64 i) {   // splitmix64's finalizer: random bytes
  u64 z = (i + 1) * 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return (u32)((z ^ (z >> 31)) >> 32);
}

struct Stamp {
  u64 t0, t1, c0, c1;   // the consumers' first and last MMA (or the hash's window, with no consumers)
  u64 comps;            // the CTA's compressions
  u64 sink;
};

struct alignas(128) Prm {
  CUtensorMap a, b, bp;   // bp: B's box in a cluster's parts, a CTA's part multicast to all of them
  int* O;
  Stamp* st;
  int M, N, K, groups, nhw, hreps;
  int raster;   // loaded tiles' order: 0, gemm_r5's (rows, then columns); G, G rows of units at a time, columns within
  u32 mask;   // each operand byte's: 0xFF random int8, 0x03 the 2-bit values 0..3
};

__device__ __forceinline__ u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int cta_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// a box into the same offset of every CTA in `mask`, counted on each one's barrier at `bar`
__device__ __forceinline__ void tma_load_mc(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask)
               : "memory");
}
// a consumer warp's release of a stage to every CTA of the cluster, whose parts of B it read. CTA scope, as CUTLASS's
// ClusterBarrier: a cluster-scope release puts a GPU-wide fence before each arrive.
template <int CL> __device__ __forceinline__ void release(uint32_t bar) {
  if constexpr (CL == 1) {
    wgp::mbar_arrive(bar);
  } else {
#pragma unroll
    for (int r = 0; r < CL; r++) {
      u32 a;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(bar), "r"(r));
      asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
    }
  }
}

template <int NM, int FORM> struct Shape {
  static constexpr int BN = FORM == N128 ? 128 : 256, TMC = 64 * NM, NA = BN / 2;
  static constexpr int A_BYTES = TMC * BK, STAGE = (TMC + BN) * BK;
  static constexpr double MACS = (double)TMC * BN * BK;   // a k-tile's
};
template <int NM, int FORM, int FL> __host__ __device__ constexpr int smem_bytes() {
  return 1024 + (FL & LOAD ? stages_of(FL) : 1) * Shape<NM, FORM>::STAGE + 16 * stages_of(FL) + 16;
}

template <int NM, int FORM, int FL, int Y, int HR, int NWG>
__global__ void __launch_bounds__(NWG * 128, 1) cta(const __grid_constant__ Prm p) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using S = Shape<NM, FORM>;
  constexpr int NT = NWG * 128, BASE = 65536 / NT / 8 * 8;   // the launch's registers a thread, as setmaxnreg counts them
  static_assert(NM * Y + (NWG - NM) * HR <= NWG * BASE, "the warpgroups' registers exceed the launch's");
  static_assert(NWG == NM || (HR <= BASE && Y >= BASE), "the consumers take registers, the rest give them");
  static_assert(FORM != N128 || Y >= 96, "n128's 64 accumulators");
  static_assert(FORM != N256 || BASE >= 160, "ptxas sizes a m64n256k32 by the launch's registers a thread, and it takes 154");
  static_assert(!(FL & (NOSW | KG2)) || !(FL & (BAR | LOAD | EPI)), "ns and kg2 are resident operands' forms");
  static_assert(smem_bytes<NM, FORM, FL>() <= 232448, "the stages exceed a CTA's 227 KiB of shared memory");
  constexpr int STAGES = stages_of(FL);
  constexpr bool bar = FL & (BAR | LOAD);
  constexpr int KG = FL & KG2 ? 2 : 1;   // k-tiles a commit group
  constexpr int CL = cluster_of(FL);      // the cluster's CTAs: M-neighbours on one column of tiles, sharing B's box
  static_assert(CL == 1 || (FL & LOAD), "multicast is a loaded form");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t base = (wgp::smem_at(smem) + 1023) & ~1023u;
  const uint32_t bars = base + (FL & LOAD ? STAGES : 1) * S::STAGE;   // STAGES full, STAGES empty, then done
  u32* const done = reinterpret_cast<u32*>(smem + (bars - wgp::smem_at(smem)) + 16 * STAGES);
  const int tid = threadIdx.x, wg = tid / 128, warp = tid / 32, lane = tid % 32;
  const int tiles_m = p.M / S::TMC, tiles_n = p.N / S::BN, TK = p.K / BK, rank = CL == 1 ? 0 : cta_rank();
  // the unit of work: CL tiles one above the other, CTA r of the cluster the r-th
  const int units = tiles_m / CL * tiles_n, u0 = (int)blockIdx.x / CL, ustep = (int)gridDim.x / CL;
  const int my_tiles = u0 < units ? (units - 1 - u0) / ustep + 1 : 0;
  const int groups = FL & LOAD ? my_tiles * TK : p.groups;
  auto tile = [&](int g, int& tm, int& tn) {
    const int u = (u0 + (g / TK) * ustep) % units;
    int ur = u / tiles_n;
    tn = u % tiles_n;
    if (p.raster > 0) {
      const int band = p.raster * tiles_n, w = u % band;
      ur = u / band * p.raster + w % p.raster;
      tn = w / p.raster;
    }
    tm = ur * CL + rank;
  };
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      wgp::mbar_init(bars + 8 * s, 1);
      wgp::mbar_init(bars + 8 * (STAGES + s), CL * NM * 4);
    }
    *done = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (!(FL & LOAD))   // resident operands
    for (int i = tid; i < S::STAGE / 4; i += NT) reinterpret_cast<u32*>(smem + (base - wgp::smem_at(smem)))[i] = mix32(i) & p.mask * 0x01010101u;
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if constexpr (CL > 1) cluster_sync();   // no multicast or remote arrive reaches a CTA before its barriers are set
  Stamp* const out = p.st + blockIdx.x;

  if constexpr (NWG > NM) if (wg >= NM) {
    if constexpr (HR < BASE) wgp::regs_dec<HR>();
    const int w = warp - 4 * NM;   // 0: the producer warp; 1 ..: hash warps
    if (w == 0) {
      if (lane == 0 && bar && groups > 0) {
        if (FL & LOAD) { wgp::tma_prefetch(&p.a); wgp::tma_prefetch(CL == 1 ? &p.b : &p.bp); }
        int s = 0;
        uint32_t ph = 0;
        for (int g = 0; g < groups; g++) {
          wgp::mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
          if constexpr (FL & LOAD) {
            int tm, tn;
            tile(g, tm, tn);
            const int kt = g % TK;
            const uint32_t st = base + s * S::STAGE, full = bars + 8 * s;
            wgp::mbar_expect_tx(full, S::STAGE);
            wgp::tma_load(st, &p.a, full, kt * BK, tm * S::TMC);
            if constexpr (CL == 1)
              wgp::tma_load(st + S::A_BYTES, &p.b, full, kt * BK, tn * S::BN);
            else
              tma_load_mc(st + S::A_BYTES + rank * (S::BN / CL) * BK, &p.bp, full, kt * BK, tn * S::BN + rank * (S::BN / CL),
                          (1 << CL) - 1);
          } else {
            wgp::mbar_arrive(bars + 8 * s);
          }
          if (++s == STAGES) { s = 0; ph ^= 1; }
        }
        // the other CTAs' consumers arrive on this one's barriers: it stays until all of them released every stage
        if constexpr (CL > 1)
          for (int i = 0; i < STAGES; i++) {
            wgp::mbar_wait(bars + 8 * (STAGES + s), ph ^ 1);
            if (++s == STAGES) { s = 0; ph ^= 1; }
          }
      }
      asm volatile("exit;");
    }
    if (w > p.nhw) asm volatile("exit;");
    // a hash warp: SHA-512 over a chained block, in registers, until the consumers are done (or HREPS with none)
    const u64 g = (u64)blockIdx.x * NT + tid;
    u64 st[8], wv[16];
    sx::iv(st);
#pragma unroll
    for (int i = 0; i < 16; ++i) wv[i] = g * 16 + i;
    u64 n = 0, t0 = 0, c0 = 0;
    if (w == 1 && lane == 0) { t0 = gtimer(); c0 = clock64(); }
    const bool alone = groups == 0;
    for (;;) {
      if (alone ? n >= (u64)p.hreps : *reinterpret_cast<volatile u32*>(done) >= (u32)(NM * 4)) break;
#pragma unroll 1
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < 8; ++i) wv[i] ^= st[i];
        sx::compress<0, 1>(st, wv);
      }
      n += 4;
    }
    if (lane == 0) atomicAdd(&out->comps, 32 * n);
    if (st[0] == 0x0123456789abcdefull) out->sink = st[1];
    if (alone && w == 1 && lane == 0) { out->t0 = t0; out->c0 = c0; out->t1 = gtimer(); out->c1 = clock64(); }
    asm volatile("exit;");
  }

  if constexpr (NWG > NM && Y > BASE) wgp::regs_inc<Y>();
  const int x = tid % 128;
  int32_t d[S::NA];
#pragma unroll
  for (int i = 0; i < S::NA; i++) d[i] = 0;
  int s = 0, pend = -1;
  uint32_t ph = 0;
  u64 t0 = 0, c0 = 0;
  if (tid == 0) { t0 = gtimer(); c0 = clock64(); }
  // k32 slice i's descriptors. SW128: a row's 128 bytes of K, slice i 32 bytes on, B's rows 128 on (h 1) 128 BK bytes on.
  // ns: slice i of the warpgroup's 64 rows 2 KiB at (i NM + wg) 2 KiB, of B's BN rows BN x 32 bytes, rows 128 on 4 KiB on;
  // 8-row core matrices 256 bytes apart, K's two 16-byte halves 128.
  auto dsc = [](uint32_t addr) -> uint64_t {
    if constexpr (FL & NOSW) return (uint64_t)((addr & 0x3FFFF) >> 4) | (uint64_t)(128 >> 4) << 16 | (uint64_t)(256 >> 4) << 32;
    else return wgp::desc_sw128(addr);
  };
  for (int g = 0; g < groups; g += KG) {
    if (bar) wgp::mbar_wait_warp(bars + 8 * s, ph);
    const uint32_t stg = base + (FL & LOAD ? s : 0) * S::STAGE, b0 = stg + S::A_BYTES;
    auto da = [&](int i) { return dsc(FL & NOSW ? stg + (i * NM + wg) * 2048 : stg + wg * 64 * BK + 32 * i); };
    auto db = [&](int i, int h) { return dsc(FL & NOSW ? b0 + i * S::BN * 32 + h * 4096 : b0 + h * 128 * BK + 32 * i); };
    WGP_FENCE;
#pragma unroll
    for (int q = 0; q < KG; q++) {
      const uint32_t sd = (g + q) % TK != 0;
      if constexpr (FORM == N128 || FORM == N256) {
#pragma unroll
        for (int i = 0; i < 4; i++) wgp::imma<S::BN>(d, da(i), db(i, 0), i > 0 || sd);
      } else if constexpr (FORM == ALT) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          wgp::imma<128>(d, da(i), db(i, 0), i > 0 || sd);
          wgp::imma<128>(d + 64, da(i), db(i, 1), i > 0 || sd);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; i++) wgp::imma<128>(d, da(i), db(i, 0), i > 0 || sd);
#pragma unroll
        for (int i = 0; i < 4; i++) wgp::imma<128>(d + 64, da(i), db(i, 1), i > 0 || sd);
      }
    }
    WGP_COMMIT;
    WGP_WAIT(1);
    if (bar && pend >= 0 && lane == 0) release<CL>(bars + 8 * (STAGES + pend));
    pend = s;
    if (++s == STAGES) { s = 0; ph ^= 1; }
    if constexpr (FL & EPI) {
      if (g % TK == TK - 1) {
        WGP_WAIT(0);
        wgp::acc_fence<S::NA>(d);
        if (bar && lane == 0) release<CL>(bars + 8 * (STAGES + pend));
        pend = -1;
        int tm, tn;
        tile(g, tm, tn);
        const int rloc = wg * 64 + 16 * (x / 32) + (x % 32) / 4, cq = 2 * (x % 4);
        int* const o0 = p.O + (size_t)(tm * S::TMC + rloc) * p.N + tn * S::BN + cq;
        int* const o1 = o0 + (size_t)8 * p.N;
#pragma unroll
        for (int j = 0; j < S::NA / 4; j++) {
          *reinterpret_cast<int2*>(o0 + 8 * j) = make_int2(d[4 * j], d[4 * j + 1]);
          *reinterpret_cast<int2*>(o1 + 8 * j) = make_int2(d[4 * j + 2], d[4 * j + 3]);
          if (j % 8 == 7) wgp::ld_fence();
        }
      }
    }
  }
  WGP_WAIT(0);
  wgp::acc_fence<S::NA>(d);
  if (bar && pend >= 0 && lane == 0) release<CL>(bars + 8 * (STAGES + pend));
  if (tid == 0 && groups > 0) { out->t0 = t0; out->c0 = c0; out->t1 = gtimer(); out->c1 = clock64(); }
  u32 xr = 0;
#pragma unroll
  for (int i = 0; i < S::NA; i++) xr ^= (u32)d[i];
  if (xr == 0x9e3779b9u) out->sink = xr;
  __syncwarp();
  if (lane == 0) atomicAdd(done, 1u);
#endif
}

// ---------------------------------------------------------------------------------------------------------- host
struct Cfg {
  const char* name;
  int nm, form, fl, y, hr, nwg, smem;
  double macs;   // a k-tile's, a CTA
  void (*k)(Prm);
};
#define CFG(NAME, NM, FORM, FL, Y, HR, NWG) \
  {NAME, NM, FORM, FL, Y, HR, NWG, smem_bytes<NM, FORM, FL>(), Shape<NM, FORM>::MACS, cta<NM, FORM, FL, Y, HR, NWG>}
// The consumers' register budget Y and, for each, the tiles and CTA sizes it fits, the hash warps at the registers left:
// res, the operands resident (the ceiling), and le, TMA loads and the epilogue (the served call's parts). gemm_r5 is
// y160_m2_alt_w4 (HR 96, 7 hash warps beside the producer). At most 5 warpgroups: ptxas sizes each MMA by the launch's
// registers a thread, and a m64n128k32 takes 90, which 6 warpgroups' 85 doesn't hold. Left out, under SHA-512's 48
// registers of state: Y 232 at 5 warpgroups (HR 56) and two consumers at Y 160 at 5 (HR 48).
#define BOTH(NAME, NM, FORM, Y, HR, NWG) CFG(NAME "_res", NM, FORM, 0, Y, HR, NWG), CFG(NAME "_le", NM, FORM, LOAD | EPI, Y, HR, NWG)
// One consumer warpgroup beside up to 15 hash warps (640 threads), each part apart: resident operands in TMA's layout
// (res), in the co-issue probe's (ns), two k-tiles a commit group (kg2), TMA loads alone (l), and with the epilogue (le)
#define ONE(NAME, FORM, Y, HR)                                                                                       \
  CFG(NAME "_res", 1, FORM, 0, Y, HR, 5), CFG(NAME "_ns", 1, FORM, NOSW, Y, HR, 5), CFG(NAME "_kg2", 1, FORM, KG2, Y, HR, 5), \
      CFG(NAME "_l", 1, FORM, LOAD, Y, HR, 5), CFG(NAME "_le", 1, FORM, LOAD | EPI, Y, HR, 5)
static const Cfg CFGS[] = {
    BOTH("y232_m1_alt_w4", 1, ALT, 232, 88, 4),   BOTH("y200_m1_alt_w4", 1, ALT, 200, 104, 4),
    BOTH("y176_m2_alt_w4", 2, ALT, 176, 80, 4),   BOTH("y160_m2_alt_w4", 2, ALT, 160, 96, 4),
    BOTH("y128_m2_n128_w4", 2, N128, 128, 128, 4), BOTH("y128_m2_n128_w5", 2, N128, 128, 72, 5),
    ONE("y200_m1_alt_w5", ALT, 200, 64),          ONE("y176_m1_alt_w5", ALT, 176, 72),
    ONE("y160_m1_n128_w5", N128, 160, 80),
    CFG("y176_m1_chain_w5_res", 1, CHAIN, 0, 176, 72, 5), CFG("y176_m1_chain_w5_l", 1, CHAIN, LOAD, 176, 72, 5),
    // TMA loads with B's box multicast over a cluster of 2 or 4 (lmc2, lmc4; each CTA loads its part of B to all), and
    // with the epilogue (lemc2); gemm_r5's two consumers on the loads alone (l), as the streamed GEMMs' 128 x 256 tile
    CFG("y176_m1_alt_w5_lmc2", 1, ALT, LOAD | MC2, 176, 72, 5), CFG("y176_m1_alt_w5_lmc4", 1, ALT, LOAD | MC4, 176, 72, 5),
    CFG("y176_m1_alt_w5_lemc2", 1, ALT, LOAD | EPI | MC2, 176, 72, 5),
    CFG("y200_m1_alt_w5_lmc2", 1, ALT, LOAD | MC2, 200, 64, 5), CFG("y160_m1_n128_w5_lmc2", 1, N128, LOAD | MC2, 160, 80, 5),
    CFG("y160_m2_alt_w4_l", 2, ALT, LOAD, 160, 96, 4),     CFG("y160_m2_alt_w4_lmc2", 2, ALT, LOAD | MC2, 160, 96, 4),
    // five stages in place of four (s5; eight for n128's smaller stage), and the hash warps at 80 registers (y160 alt)
    CFG("y176_m1_alt_w5_l_s5", 1, ALT, LOAD | ST(5), 176, 72, 5),
    CFG("y176_m1_alt_w5_lmc2_s5", 1, ALT, LOAD | MC2 | ST(5), 176, 72, 5),
    CFG("y160_m1_n128_w5_l_s8", 1, N128, LOAD | ST(8), 160, 80, 5),
    CFG("y160_m1_alt_w5_res", 1, ALT, 0, 160, 80, 5),      CFG("y160_m1_alt_w5_l", 1, ALT, LOAD, 160, 80, 5),
    // 384 threads: one consumer warpgroup and 7 hash warps beside the producer, the launch's 168 registers a thread
    // enough for a m64n256k32 (n256: one an atom, 80 bytes of shared memory a SM-clock to alt's 96)
    CFG("y232_m1_n256_w3_res", 1, N256, 0, 232, 136, 3),   CFG("y232_m1_n256_w3_l", 1, N256, LOAD, 232, 136, 3),
    CFG("y232_m1_n256_w3_le", 1, N256, LOAD | EPI, 232, 136, 3),
    CFG("y232_m1_n256_w3_lmc2", 1, N256, LOAD | MC2, 232, 136, 3),
    CFG("y232_m1_n256_w3_l_s5", 1, N256, LOAD | ST(5), 232, 136, 3),
    CFG("y232_m1_n256_w3_lmc2_s5", 1, N256, LOAD | MC2 | ST(5), 232, 136, 3),
    CFG("y232_m1_alt_w3_res", 1, ALT, 0, 232, 136, 3),     CFG("y232_m1_alt_w3_l", 1, ALT, LOAD, 232, 136, 3),
    // the consumers alone, as the co-issue probe's CTA (Y: the launch's registers)
    CFG("y255_m1_n128_w1_res", 1, N128, 0, 255, 0, 1), CFG("y255_m1_n128_w1_ns", 1, N128, NOSW, 255, 0, 1),
    CFG("y255_m1_alt_w1_res", 1, ALT, 0, 255, 0, 1),   CFG("y255_m1_alt_w1_ns", 1, ALT, NOSW, 255, 0, 1),
    CFG("y255_m1_alt_w1_kg2", 1, ALT, KG2, 255, 0, 1),     CFG("y255_m1_n256_w1_res", 1, N256, 0, 255, 0, 1),
};

static CUtensorMap tmap(const void* ptr, int depth, int rows, int box_rows) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {(cuuint64_t)depth, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)depth};
  const cuuint32_t box[2] = {128, (cuuint32_t)box_rows}, es[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled: %d\n", (int)r); exit(3); }
  return m;
}

__global__ void fill_k(uint8_t* p, size_t n, u64 seed, u32 mask) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint8_t)(mix32(seed * 0x100000000ull + i) & mask);
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static std::string jnum(double x, int digits) {   // a JSON number, or null
  char b[48];
  if (!std::isfinite(x)) return "null";
  snprintf(b, sizeof b, "%.*f", digits, x);
  return b;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5, groups = argc > 2 ? atoi(argv[2]) : 4096, hreps = argc > 3 ? atoi(argv[3]) : 2048;
  const double item_ms = argc > 4 ? atof(argv[4]) : 2000;
  const int M = 16384, N = 28672, K = argc > 5 ? atoi(argv[5]) : 4096;   // Llama 3 8B's gate-up at M 16,384
  if (K <= 0 || K % (4 * BK) || groups <= 0 || groups % 2) {
    fprintf(stderr, "K is a positive multiple of %d, and KTILES a positive even number\n", 4 * BK);
    return 2;
  }
  const std::string sel = getenv("CFGS") ? getenv("CFGS") : "", opl = getenv("OPS") ? getenv("OPS") : "s8";
  const int raster = getenv("RASTER") ? atoi(getenv("RASTER")) : 0;
  std::vector<int> hwl;   // HWS: the hash warps' counts, past a CTA's most taken as its most
  for (const char* h = getenv("HWS") ? getenv("HWS") : "7,11,99"; *h;) {
    char* e;
    const long v = strtol(h, &e, 10);
    if (e == h || v <= 0 || (*e && *e != ',')) { fprintf(stderr, "HWS lists positive hash-warp counts, comma-separated\n"); return 2; }
    hwl.push_back((int)v);
    h = *e ? e + 1 : e;
  }
  std::vector<std::pair<std::string, u32>> kinds;   // the operands' kinds, each a byte mask
  for (size_t i = 0; i <= opl.size();) {
    const size_t j = std::min(opl.find(',', i), opl.size());
    const std::string k = opl.substr(i, j - i);
    if (k == "s8" || k == "u2") kinds.push_back({k, k == "s8" ? 0xFFu : 0x03u});
    else if (!k.empty()) { fprintf(stderr, "OPS lists s8 (random int8) and u2 (the 2-bit values 0..3)\n"); return 2; }
    i = j + 1;
  }
  auto chosen = [&](const Cfg& c) {
    if (sel.empty()) return true;
    for (size_t i = 0; i <= sel.size();) {
      const size_t j = std::min(sel.find(',', i), sel.size());
      if (j > i && std::string(c.name).rfind(sel.substr(i, j - i), 0) == 0) return true;
      i = j + 1;
    }
    return false;
  };
  cudaDeviceProp dp;
  CK(cudaGetDeviceProperties(&dp, 0));
  const int nsm = dp.multiProcessorCount;
  const double peak = nsm * 4096.0 * PEAK_MHZ * 1e6;
  gk::Nvml nv(0);
  printf("{\"device\": \"%s\", \"sms\": %d, \"M\": %d, \"N\": %d, \"K\": %d, \"groups\": %d, \"hreps\": %d, \"item_ms\": %.0f, "
         "\"ops\": \"%s\", \"raster\": %d, \"nvml\": %s, \"energy\": %s}\n",
         dp.name, nsm, M, N, K, groups, hreps, item_ms, opl.c_str(), raster, nv.ok() ? "true" : "false",
         nv.energy_mj() ? "true" : "false");
  uint8_t *A, *B;
  int* O;
  Stamp* st;
  CK(cudaMalloc(&A, (size_t)M * K));
  CK(cudaMalloc(&B, (size_t)N * K));
  CK(cudaMalloc(&O, (size_t)M * N * 4));
  CK(cudaMalloc(&st, nsm * sizeof(Stamp)));
  // a launch: one CTA an SM, in clusters of cluster_of(fl); its grid, the CTAs whose clusters can all be resident at once
  cudaLaunchAttribute at{};   // the cluster's shape, which every config below points to
  auto config = [&](const Cfg& c, int gx) {
    const int cl = cluster_of(c.fl);
    at.id = cudaLaunchAttributeClusterDimension;
    at.val.clusterDim.x = cl;
    at.val.clusterDim.y = 1;
    at.val.clusterDim.z = 1;
    cudaLaunchConfig_t lc{};
    lc.gridDim = dim3(gx);
    lc.blockDim = dim3(128 * c.nwg);
    lc.dynamicSmemBytes = c.smem;
    lc.attrs = &at;
    lc.numAttrs = cl > 1;
    return lc;
  };
  std::vector<int> grid(sizeof CFGS / sizeof CFGS[0], nsm);
  for (const Cfg& c : CFGS) {
    if (!chosen(c)) continue;
    CK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, c.k));
    const int cl = cluster_of(c.fl);
    if (raster < 0 || (raster > 0 && (c.fl & LOAD) && M / (64 * c.nm) / cl % raster)) {
      fprintf(stderr, "%s: RASTER is 0 or divides the %d rows of units\n", c.name, M / (64 * c.nm) / cl);
      return 2;
    }
    if (cl > 1) {
      const cudaLaunchConfig_t lc = config(c, nsm);
      int nc = 0;
      CK(cudaOccupancyMaxActiveClusters(&nc, c.k, &lc));
      grid[&c - CFGS] = std::min(nsm / cl, nc) * cl;
    }
    printf("{\"cfg\": \"%s\", \"nm\": %d, \"bn\": %d, \"y\": %d, \"hr\": %d, \"threads\": %d, \"regs\": %d, \"local\": %zu, "
           "\"smem\": %d, \"fl\": %d, \"stages\": %d, \"cluster\": %d, \"ctas\": %d}\n",
           c.name, c.nm, c.form == N128 ? 128 : 256, c.y, c.hr, 128 * c.nwg, fa.numRegs, fa.localSizeBytes, c.smem, c.fl,
           c.fl & LOAD ? stages_of(c.fl) : 1, cl, grid[&c - CFGS]);
  }
  auto go = [&](const Cfg& c, const Prm& p) {
    const cudaLaunchConfig_t lc = config(c, grid[&c - CFGS]);
    CK(cudaLaunchKernelEx(&lc, c.k, p));
  };
  std::string ops;
  u32 mask = 0xFF;
  auto prm = [&](const Cfg& c, int g, int nhw) {
    Prm p{};
    const int bn = c.form == N128 ? 128 : 256;
    p.a = tmap(A, K, M, 64 * c.nm);
    p.b = tmap(B, K, N, bn);
    p.bp = tmap(B, K, N, bn / cluster_of(c.fl));
    p.O = O;
    p.st = st;
    p.M = M; p.N = N; p.K = K; p.groups = g; p.nhw = nhw; p.hreps = hreps; p.mask = mask; p.raster = raster;
    return p;
  };
  auto launch = [&](const Cfg& c, int g, int nhw) {
    const Prm p = prm(c, g, nhw);
    CK(cudaMemset(st, 0, nsm * sizeof(Stamp)));
    go(c, p);
    CK(cudaDeviceSynchronize());
    std::vector<Stamp> s(nsm);
    CK(cudaMemcpy(s.data(), st, nsm * sizeof(Stamp), cudaMemcpyDeviceToHost));
    return s;
  };
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  // one arm: first the energy item (so the stamped launches after it run at the board's steady power), then REPS
  // launches: theta and the hash's rate, over the CTAs' own windows, a second and a SM-clock; the SM clock; the wall
  auto arm = [&](const Cfg& c, int g, int nhw, const char* what) {
    // an arm takes its item and a few launches: one that hangs ends the process, the arms printed so far kept
    alarm(60 + (unsigned)(3 * item_ms / 1000));
    std::vector<double> th, thc, hb, hbc, mhz, ms, mc, cm;
    launch(c, g, nhw);
    auto stamped = [&] {
      auto s = launch(c, g, nhw);
      const int bn = c.form == N128 ? 128 : 256, cl = cluster_of(c.fl), gx = grid[&c - CFGS], TK = K / BK;
      const int units = M / (64 * c.nm) / cl * (N / bn);
      double cyc = 0, ns = 0, macs = 0, comps = 0;
      u64 a = ~0ull, b = 0;
      for (int i = 0; i < gx; i++) {
        cyc += (double)(s[i].c1 - s[i].c0);
        ns += (double)(s[i].t1 - s[i].t0);
        a = std::min(a, s[i].t0);
        b = std::max(b, s[i].t1);
        comps += (double)s[i].comps;
        const int myt = i / cl < units ? (units - 1 - i / cl) / (gx / cl) + 1 : 0;
        macs += g == 0 ? 0 : c.macs * ((c.fl & LOAD) ? (double)myt * TK : (double)g);
      }
      // a CTA's MACs a clock, over every SM (gx of them busy)
      const double f = cyc / ns * 1e3, sms = (double)gx / nsm;
      thc.push_back(macs / cyc / 4096 * sms);
      th.push_back(macs / cyc / 4096 * sms * f / PEAK_MHZ);
      hbc.push_back(comps / cyc / BAR_RATE * sms);
      hb.push_back(comps / cyc / BAR_RATE * sms * f / PEAK_MHZ);
      mhz.push_back(f);
      ms.push_back((double)(b - a) * 1e-6);
      mc.push_back(macs);
      cm.push_back(comps);
    };
    stamped();
    const int chunk = std::max(1, (int)std::ceil(0.25 * item_ms / std::max(ms[0], 1e-3)));
    const Prm p = prm(c, g, nhw);
    gk::PowerTrace pt(nv, 20);
    const unsigned long long j0 = nv.energy_mj();
    CK(cudaEventRecord(e0));
    int n = 0;
    float ims = 0;
    while (ims < item_ms) {
      for (int i = 0; i < chunk; i++) go(c, p);
      n += chunk;
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      CK(cudaEventElapsedTime(&ims, e0, e1));
    }
    const unsigned long long j1 = nv.energy_mj();
    pt.stop();
    CK(cudaGetLastError());
    th.clear(); thc.clear(); hb.clear(); hbc.clear(); mhz.clear(); ms.clear(); mc.clear(); cm.clear();
    for (int r = 0; r < reps; r++) stamped();
    const double mj = j0 && j1 >= j0 ? (double)(j1 - j0) : NAN, macs = median(mc), comps = median(cm);
    printf("{\"cfg\": \"%s\", \"ops\": \"%s\", \"arm\": \"%s\", \"nhw\": %d, \"theta\": %.4f, \"theta_clk\": %.4f, \"hash_bar\": %.4f, "
           "\"hash_bar_clk\": %.4f, \"mhz\": %.0f, \"ms\": %.4f, \"item_ms\": %.1f, \"launches\": %d, \"theta_item\": %s, "
           "\"w\": %s, \"wmax\": %.1f, \"wsampled\": %.1f, \"pj_mac\": %s, \"nj_comp\": %s}\n",
           c.name, ops.c_str(), what, nhw, median(th), median(thc), median(hb), median(hbc), median(mhz), median(ms), ims, n,
           jnum(macs > 0 ? n * macs / (ims * 1e-3) / peak : NAN, 4).c_str(), jnum(mj / ims, 1).c_str(), pt.max_mw / 1e3,
           pt.mean_mw() / 1e3, jnum(macs > 0 ? mj * 1e9 / (n * macs) : NAN, 3).c_str(),
           jnum(comps > 0 ? mj * 1e6 / (n * comps) : NAN, 3).c_str());
    fflush(stdout);
    alarm(0);
  };
  // the hash alone reads no operands: once, with the first kind
  for (size_t ki = 0; ki < kinds.size(); ki++) {
    ops = kinds[ki].first;
    mask = kinds[ki].second;
    fill_k<<<1024, 256>>>(A, (size_t)M * K, 1, mask);
    fill_k<<<1024, 256>>>(B, (size_t)N * K, 2, mask);
    CK(cudaDeviceSynchronize());
    for (const Cfg& c : CFGS) {
      if (!chosen(c)) continue;
      const int maxhw = (c.nwg - c.nm) * 4 - 1;
      arm(c, groups, 0, "gemm");
      std::vector<int> hws;
      for (int h : hwl)
        if (maxhw > 0) hws.push_back(std::min(h, maxhw));
      std::sort(hws.begin(), hws.end());
      hws.erase(std::unique(hws.begin(), hws.end()), hws.end());
      for (int nhw : hws) {
        arm(c, groups, nhw, "gemm+hash");
        if (ki == 0 && !(c.fl & (LOAD | NOSW | KG2))) arm(c, 0, nhw, "hash");
      }
    }
  }
  return 0;
}
