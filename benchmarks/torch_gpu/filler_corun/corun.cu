// The filler's exact int8 GEMM on the H200 (sm_90a) with SHA-512 in separate CTAs on the same SMs.
//
// gemm_<CL>: O = A B^T, O [M, N] s32, A [M, K] and B [N, K] row-major int8 (depth contiguous), every sum exact (s32 over K,
// the operands' range bounding it: u2 bytes in [0, 3] give at most 9 K). s8_linear's sm_90a GEMM (s8_wgmma.cu) with
// three changes: the epilogue stores the s32 sums themselves (torch._int_mm's words), there is no producer warpgroup
// (thread 0 issues every TMA between its own wgmmas), and the kernel is capped at GEMM_REGS registers a thread with no
// setmaxnreg, so that one GEMM CTA an SM leaves registers for a hash CTA beside it. ptxas sizes each wgmma by the
// launch's registers a thread, so the cap must cover m64n256k32's (154). A CTA is a 128 x 256 tile: two warpgroups
// (warps 0..7), each a 64 x 256 wgmma tile (m64n256k32 .s32.s8.s8, 128 accumulators a thread). The k-blocks of the CTA's
// tiles are one stream through STAGES stages of 128 bytes of depth with SWIZZLE_128B, each stage's arrival counted on an
// mbarrier (full) and its release on another (empty: lane 0 of each warp); when thread 0 has released element g it
// loads element g + STAGES into that stage, across tile boundaries, so the next tile's loads overlap this one's
// epilogue. Persistent: CTA c takes tiles c, c + gridDim.x, ... of `passes` passes over the tiles (raster: groups of
// `group` tile rows, column by column), so a long item has no wave tail inside it. CL = 2 runs clusters of two CTAs on
// M-adjacent tiles that share each B box by TMA multicast.
//
// An SM's registers are four sub-partitions of 16,384, and a warp's come from its own sub-partition. Eight GEMM warps
// at 160 take 10,240 of each, leaving room for two hash warps at up to 96 on every sub-partition; a ninth (producer)
// warp would fill one sub-partition and leave a hash CTA of more than six warps no place.
//
// hash_rows: sx0000 (sha512x.cuh's compress<0, 1>) over 1 KiB messages read from memory: message i is bytes
// [1024 i, 1024 i + 1024) of `msg` (the GEMM's O), its digest SHA-512 of those bytes (nine compressions), written to
// dig[8 i ..] as big-endian words. Thread t takes messages t, t + T, ... (mod nmsg) until `stop` is set or its CTA has
// run `dur_ns`. hash_chain: the bar's own form, each thread chaining compressions in registers.
//
// Every CTA stamps %globaltimer and clock64 at its start and end and its %smid; GEMM CTAs log %globaltimer every
// `log_every` tiles. gate_k holds the hash stream until every GEMM CTA has started, so hash CTAs take only what the GEMM
// CTAs leave on each SM.
#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "../s8_linear/s8_acc.h"
#include "../sha512_rows/sha512x.cuh"

#define DEV __device__ __forceinline__

#ifndef GEMM_REGS
#define GEMM_REGS 160
#endif
#ifndef HASH_REGS
#define HASH_REGS 80
#endif

using u32 = uint32_t;
using u64 = unsigned long long;

struct Stamp {
  u64 t0, t1, c0, c1;
  u32 sm, n;  // n: tiles (GEMM) or messages (hash) done
  u64 work;   // hash: compressions
};
static_assert(sizeof(Stamp) == 48, "corun.py's STAMP");

constexpr int TM = 128, BN = 256, BK = 128, NWG = 2, NA = BN / 2, STAGES = 4, THREADS = 128 * NWG;
constexpr u32 A_BYTES = TM * BK, STAGE = (TM + BN) * BK;
constexpr int SMEM = 1024 + STAGES * STAGE + 16 * STAGES;
constexpr int LOGMAX = 4096;

struct alignas(128) Params {
  CUtensorMap a, b;  // A: M rows of K bytes, boxes of TM rows; B: N rows, boxes of BN / CL rows
  int32_t* o;        // O [M, N] s32, row-major
  int M, N, K, group;
  int passes, log_every;
  u64* log;          // [gridDim.x][LOGMAX]
  Stamp* stamps;     // [gridDim.x]
  int* started;      // CTAs started
  int* done;         // CTAs finished
  int* stop;         // set by the last CTA to finish
};
static_assert(offsetof(Params, o) == 256 && offsetof(Params, M) == 264 && offsetof(Params, log) == 288 &&
                  sizeof(Params) == 384,
              "corun.py's Params");

DEV u64 gtimer() {
  u64 t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DEV u32 smid() {
  u32 s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// ---------------------------------------------------------------- mbarriers, TMA, wgmma (s8_wgmma.cu's)
DEV u32 smem_at(const void* p) { return (u32)__cvta_generic_to_shared(p); }
DEV void mbar_init(u32 bar, u32 count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEV void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }
DEV void mbar_wait(u32 bar, u32 parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV void mbar_wait_cluster(u32 bar, u32 parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAITC:\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra "
      "WAITC;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
DEV void mbar_arrive_at(u32 bar, u32 rank) {
  u32 r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(r) : "memory");
}
DEV void tma_prefetch(const void* map) { asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory"); }
DEV void tma_load(u32 dst, const void* map, u32 bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1)
      : "memory");
}
DEV void tma_load_mc(u32 dst, const void* map, u32 bar, int c0, int c1, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}
DEV u32 cta_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

#define WG_FENCE asm volatile("wgmma.fence.sync.aligned;" ::: "memory")
#define WG_COMMIT asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory")
#define WG_WAIT(n) asm volatile("wgmma.wait_group.sync.aligned " #n ";" ::: "memory")

DEV u64 desc_sw128(u32 addr) {
  return (u64)((addr & 0x3FFFF) >> 4) | (1ull << 16) | ((u64)(1024 >> 4) << 32) | (1ull << 62);
}
// d = A B + (sd ? d : 0), one s8 atom (k = 32) of a 64 x 256 tile. Accumulator i of thread x is row 16 (x / 32) +
// (x % 32) / 4 + 8 ((i / 2) % 2), column 8 (i / 4) + 2 (x % 4) + i % 2
DEV void imma(int32_t* d, u64 da, u64 db, u32 sd) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %130, 0;\n wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 " ACC128_S
               ", %128, %129, p;\n}"
               : ACC128_R
               : "l"(da), "l"(db), "r"(sd));
}
DEV void acc_fence(int32_t* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+r"(d[i])::"memory");
}

DEV void raster(int t, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
  const int per = group * tiles_n, first = (t / per) * group, size = min(tiles_m - first, group), r = t % per;
  tm = first + r % size;
  tn = r / size;
}

template <int CL> DEV void release(u32 bar) {
  if constexpr (CL == 1) {
    mbar_arrive(bar);
  } else {
    mbar_arrive_at(bar, 0);
    mbar_arrive_at(bar, 1);
  }
}

// ---------------------------------------------------------------- the GEMM
struct Stream {
  int KB, rows_u, tiles_n, units, first_u, step_u, rank, group;
  long n;  // k-blocks in this CTA's stream
};

// element e of the CTA's k-block stream into stage e % STAGES, once every warp (both CTAs' with CL = 2) released it
template <int CL> DEV void produce(const Params& p, const Stream& q, u32 base, u32 bars, long e) {
  const int s = (int)(e % STAGES);
  const u32 round = (u32)(e / STAGES), st = base + s * STAGE, full = bars + 8 * s;
  if constexpr (CL == 1)
    mbar_wait(bars + 8 * (STAGES + s), (round & 1) ^ 1);
  else
    mbar_wait_cluster(bars + 8 * (STAGES + s), (round & 1) ^ 1);
  const long it = e / q.KB;
  const int kb = (int)(e - it * q.KB);
  int tm, tn;
  raster((int)((q.first_u + it * q.step_u) % q.units), q.rows_u, q.tiles_n, q.group, tm, tn);
  tm = tm * CL + q.rank;
  constexpr u32 B_PART = BN / CL * BK;
  mbar_expect_tx(full, STAGE);
  tma_load(st, &p.a, full, kb * BK, tm * TM);
  const u32 bdst = st + A_BYTES + q.rank * B_PART;
  const int brow = tn * BN + q.rank * (BN / CL);
  if constexpr (CL == 1)
    tma_load(bdst, &p.b, full, kb * BK, brow);
  else
    tma_load_mc(bdst, &p.b, full, kb * BK, brow, 0x3);
}

template <int CL> DEV void gemm_body(const Params& p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const u32 base = (smem_at(smem) + 1023) & ~1023u, bars = base + STAGES * STAGE;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, wg = warp / 4;
  const int KB = p.K / BK, tiles_m = p.M / TM, tiles_n = p.N / BN, rows_u = tiles_m / CL, units = rows_u * tiles_n;
  const long total = (long)units * p.passes;
  const int rank = CL == 1 ? 0 : (int)cta_rank(), first_u = blockIdx.x / CL, step_u = gridDim.x / CL;
  const Stream q{KB, rows_u, tiles_n, units, first_u, step_u, rank, p.group,
                 first_u < total ? (long)KB * ((total - first_u + step_u - 1) / step_u) : 0};
  u64 t0 = 0, c0 = 0;
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * (STAGES + s), CL * NWG * 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    atomicAdd(p.started, 1);
    t0 = gtimer();
    c0 = clock64();
  }
  if constexpr (CL == 1)
    __syncthreads();
  else
    cluster_sync();

  if (tid == 0) {
    tma_prefetch(&p.a);
    tma_prefetch(&p.b);
    for (long e = 0; e < STAGES && e < q.n; e++) produce<CL>(p, q, base, bars, e);
  }
  __syncwarp();
  {
    const int x = tid % 128;
    int n = 0, li = 0;
    long g = 0;  // the stream's next element
    int32_t d[NA];
    // release element g - 1 (its wgmma has retired) and load element g - 1 + STAGES into its stage
    auto retire = [&]() {
      const int s = (int)((g - 1) % STAGES);
      if (lane == 0) release<CL>(bars + 8 * (STAGES + s));
      if (tid == 0 && g - 1 + STAGES < q.n) produce<CL>(p, q, base, bars, g - 1 + STAGES);
      __syncwarp();
    };
    for (long t = first_u; t < total; t += step_u) {
      int tm, tn;
      raster((int)(t % units), rows_u, tiles_n, p.group, tm, tn);
      tm = tm * CL + rank;
      for (int kb = 0; kb < KB; kb++, g++) {
        const int s = (int)(g % STAGES);
        mbar_wait(bars + 8 * s, (u32)(g / STAGES) & 1);
        __syncwarp();
        const u32 a0 = base + s * STAGE + wg * 64 * BK, b0 = base + s * STAGE + A_BYTES;
        WG_FENCE;
        imma(d, desc_sw128(a0), desc_sw128(b0), kb > 0);
        imma(d, desc_sw128(a0 + 32), desc_sw128(b0 + 32), 1);
        imma(d, desc_sw128(a0 + 64), desc_sw128(b0 + 64), 1);
        imma(d, desc_sw128(a0 + 96), desc_sw128(b0 + 96), 1);
        WG_COMMIT;
        WG_WAIT(1);
        if (kb > 0) retire();
      }
      WG_WAIT(0);
      acc_fence(d);
      retire();
      const int row = tm * TM + wg * 64 + 16 * (x / 32) + (x % 32) / 4, col0 = tn * BN + 2 * (x % 4);
      int2* o0 = reinterpret_cast<int2*>(p.o + (size_t)row * p.N + col0);
      int2* o1 = reinterpret_cast<int2*>(p.o + (size_t)(row + 8) * p.N + col0);
#pragma unroll
      for (int j = 0; j < NA / 4; j++) {
        o0[4 * j] = make_int2(d[4 * j], d[4 * j + 1]);
        o1[4 * j] = make_int2(d[4 * j + 2], d[4 * j + 3]);
      }
      ++n;
      if (tid == 0 && p.log_every > 0 && n % p.log_every == 0 && li < LOGMAX)
        p.log[(size_t)blockIdx.x * LOGMAX + li++] = gtimer();
    }
    if (tid == 0) p.stamps[blockIdx.x].n = n;
  }
  if constexpr (CL == 2) cluster_sync();
  __syncthreads();
  if (tid == 0) {
    Stamp& o = p.stamps[blockIdx.x];
    o.t0 = t0;
    o.c0 = c0;
    o.t1 = gtimer();
    o.c1 = clock64();
    o.sm = smid();
    o.work = 0;
    __threadfence();
    if (atomicAdd(p.done, 1) == (int)gridDim.x - 1) atomicExch(p.stop, 1);
  }
}

extern "C" __global__ void __maxnreg__(GEMM_REGS) gemm_1(const __grid_constant__ Params p) { gemm_body<1>(p); }
extern "C" __global__ void __cluster_dims__(2, 1, 1) __maxnreg__(GEMM_REGS) gemm_2(const __grid_constant__ Params p) {
  gemm_body<2>(p);
}

// ---------------------------------------------------------------- the hash
struct HashArgs {
  const uint4* msg;  // nmsg messages of 1 KiB
  u64* dig;          // [nmsg][8]
  const int* stop;
  Stamp* stamps;     // [gridDim.x]
  u64 nmsg, dur_ns;
  u64 most;          // messages (hash_rows) or 16-compression rounds (hash_chain) a thread at most; 0: no limit
};

DEV u64 be64(u32 lo, u32 hi) { return sx::be64(lo, hi); }

DEV void block_stamp(Stamp* stamps, u64 t0, u64 c0, u32 n, u64 work) {
  __shared__ u64 tot_n, tot_w;
  if (threadIdx.x == 0) { tot_n = 0; tot_w = 0; }
  __syncthreads();
  atomicAdd(&tot_n, (u64)n);
  atomicAdd(&tot_w, work);
  __syncthreads();
  if (threadIdx.x == 0) {
    Stamp& o = stamps[blockIdx.x];
    o.t0 = t0;
    o.c0 = c0;
    o.t1 = gtimer();
    o.c1 = clock64();
    o.sm = smid();
    o.n = (u32)tot_n;
    o.work = tot_w;
  }
}

// beside the GEMM CTA the hash CTA has 80 registers a thread: compress has one call site (the padding chunk is the block
// loop's ninth turn; a second inlined compress spills), the loop state is 32-bit and the start time stays in shared memory
extern "C" __global__ void __maxnreg__(HASH_REGS) hash_rows(const HashArgs a) {
  __shared__ u64 start[2];
  if (threadIdx.x == 0) { start[0] = gtimer(); start[1] = clock64(); }
  __syncthreads();
  const volatile u64* t0p = start;
  const u32 T = gridDim.x * blockDim.x, nmsg = (u32)a.nmsg;
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  const volatile int* stop = a.stop;
  u32 n = 0;
  while (i < nmsg && (a.most == 0 || n < a.most)) {
    if (*stop || gtimer() - *t0p >= a.dur_ns) break;
    const uint4* m = a.msg + (u64)i * 64;
    u64 st[8], w[16];
    sx::iv(st);
#pragma unroll 1
    for (int b = 0; b < 9; b++) {
      if (b < 8) {
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const uint4 v = __ldcg(m + 8 * b + q);
          w[2 * q] = be64(v.x, v.y);
          w[2 * q + 1] = be64(v.z, v.w);
        }
      } else {
        w[0] = 0x8000000000000000ull;
#pragma unroll
        for (int q = 1; q < 15; q++) w[q] = 0;
        w[15] = 1024 * 8;
      }
      sx::compress<0, 1>(st, w);
    }
    u64* o = a.dig + (u64)i * 8;
#pragma unroll
    for (int q = 0; q < 8; q++) o[q] = st[q];
    ++n;
    i += T;
    if (i >= nmsg) i -= nmsg;
  }
  block_stamp(a.stamps, start[0], start[1], n, 9ull * n);
}

// the bar's form: each thread chains compressions of a block it varies by its state, in registers
extern "C" __global__ void __maxnreg__(HASH_REGS) hash_chain(const HashArgs a) {
  __shared__ u64 start[2];
  if (threadIdx.x == 0) { start[0] = gtimer(); start[1] = clock64(); }
  __syncthreads();
  const u64 t0 = start[0], c0 = start[1];
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const volatile int* stop = a.stop;
  u64 st[8], w[16];
  sx::iv(st);
#pragma unroll
  for (int q = 0; q < 16; q++) w[q] = t * 16 + q;
  u32 n = 0;
  while (a.most == 0 || n < a.most) {
    if (*stop || gtimer() - t0 >= a.dur_ns) break;
#pragma unroll 1
    for (int r = 0; r < 16; r++) {
#pragma unroll
      for (int q = 0; q < 8; q++) w[q] ^= st[q];
      sx::compress<0, 1>(st, w);
    }
    ++n;
  }
  if (st[0] == 0x0123456789abcdefull) a.dig[t] = st[1];
  block_stamp(a.stamps, t0, c0, n, 16ull * n);
}

extern "C" __global__ void gate_k(const volatile int* started, int want) {
  while (*started < want) {
  }
}

// what corun.py takes from this build, checked when it loads the cubin
extern "C" __global__ void corun_layout(int* out) {
  out[0] = sizeof(Params);
  out[1] = offsetof(Params, o);
  out[2] = offsetof(Params, M);
  out[3] = offsetof(Params, log);
  out[4] = sizeof(Stamp);
  out[5] = SMEM;
  out[6] = STAGES;
  out[7] = THREADS;
  out[8] = LOGMAX;
  out[9] = sizeof(HashArgs);
}
