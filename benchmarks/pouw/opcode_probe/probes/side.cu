// Side units beside the honest kernel (EPS1's eps_side, note:20261017T0310Z-report-eps1-program): the words a cycle a unit
// that works off the SM's vector pipes (L2 reductions and atomics, cp.async, TMA bulk copies and reductions, DSMEM) adds
// beside a saturated int8 IMMA kernel that folds every output word with duo (receipt_folds.cuh, the served fold), and
// whether the MMAs keep their rate. Joint points, rows.jsonl only.
//   honest        16 warps: 8 IMMA an iteration, then rf_duo_step on each accumulator's 4 words (side warps idle)
//   hs_<s>        the same 16 warps (OP_LOOP), and the side role (OP_UNTIL) running unit s only while they do
//   sa_<s>        the side role alone, the same warps (OP_LOOP)
// The side role is 4 warps (one a SMSP) issuing 8 ops a trip, but a bulk unit's role is one warp whose lane 0 issues: a
// TMA bulk op takes uniform operands, so its addresses come from blockIdx and the op's slot only, never the lane. Vector
// reductions are 128 bits a lane at most (v4.f32, v4.bf16x2).
#include "../probe.cuh"
#include "honest.cuh"

constexpr int HW = 16, SW = 4, R = 8, SMEM = 64 * 1024;
typedef unsigned long long u64;

enum { S_RED32, S_RED64, S_REDV4, S_REDV4H, S_ATOMG32, S_LDGSTS, S_BLKRED512, S_BLKRED4K, S_BLKCP4K, S_DSM_ST4, S_DSM_RED };

constexpr bool bulk(int s) { return s == S_BLKRED512 || s == S_BLKRED4K || s == S_BLKCP4K; }
constexpr unsigned bulk_bytes(int s) { return s == S_BLKRED512 ? 512u : 4096u; }

__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// One side lane: its pool cursor (each lane its own words, coalesced across the warp) and its shared-memory slots.
struct Side {
  unsigned char* g;
  uint32_t gtid, total, acc, lane, sw;
  unsigned sbase, peer;
  __device__ Side(const OpArgs& a, int side_warp, unsigned char* sb, bool cluster) {
    g = a.g;
    lane = threadIdx.x & 31;
    sw = side_warp;
    gtid = (blockIdx.x * SW + sw) * 32 + lane;
    total = gridDim.x * SW * 32;
    acc = gtid;
    sbase = (unsigned)__cvta_generic_to_shared(sb);
    peer = 0;
    if (cluster) {
      unsigned rank;
      asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
      peer = mapa(sbase, rank ^ 1u);
    }
  }
  template <int S>
  __device__ __forceinline__ void op(int q, uint32_t trip) {
    const uint32_t seq = trip * R + q, v = gtid * 2654435761u ^ seq;
    if constexpr (S == S_RED32) {
      uint32_t* p = reinterpret_cast<uint32_t*>(g) + ((gtid + seq * total) & ((64u << 20) / 4 - 1));
      asm volatile("red.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    }
    if constexpr (S == S_RED64) {
      u64* p = reinterpret_cast<u64*>(g) + ((gtid + seq * total) & ((64u << 20) / 8 - 1));
      asm volatile("red.global.add.u64 [%0], %1;" ::"l"(p), "l"((u64)v) : "memory");
    }
    if constexpr (S == S_REDV4) {
      float* p = reinterpret_cast<float*>(g) + 4 * ((gtid + seq * total) & ((64u << 20) / 16 - 1));
      const float f = (float)(v & 0xff);
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(f), "f"(f + 1.f), "f"(f + 2.f),
                   "f"(f + 3.f) : "memory");
    }
    if constexpr (S == S_REDV4H) {
      uint32_t* p = reinterpret_cast<uint32_t*>(g) + 4 * ((gtid + seq * total) & ((64u << 20) / 16 - 1));
      const uint32_t h = 0x3f803f80u ^ (v & 0x00070007u);
      asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(h), "r"(h ^ 1u),
                   "r"(h ^ 2u), "r"(h ^ 3u) : "memory");
    }
    if constexpr (S == S_ATOMG32) {
      uint32_t* p = reinterpret_cast<uint32_t*>(g) + ((gtid + seq * total) & ((64u << 20) / 4 - 1));
      uint32_t r;
      asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory");
      acc ^= r;
    }
    if constexpr (S == S_LDGSTS) {
      const unsigned char* p = g + 16u * ((gtid + seq * total) & ((64u << 20) / 16 - 1));
      const unsigned s = sbase + ((sw * R + q) * 32u + lane) * 16u;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(p) : "memory");
    }
    if constexpr (S == S_DSM_ST4) {
      const unsigned s = peer + ((sw * R + q) * 32u + lane) * 16u;
      asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(s), "r"(v), "r"(v ^ 1u), "r"(v ^ 2u),
                   "r"(v ^ 3u) : "memory");
    }
    if constexpr (S == S_DSM_RED) {
      const unsigned s = peer + ((sw * R + q) * 32u + lane) * 4u;
      asm volatile("red.shared::cluster.add.u32 [%0], %1;" ::"r"(s), "r"(v) : "memory");
    }
  }
  // A bulk trip: R ops from one thread, each to its own slot of the block's region, then a group bound so the bulk
  // unit's queue stays full without unbounded reads of the source slots in flight.
  template <int S>
  __device__ __forceinline__ void bulk_trip() {
    constexpr unsigned B = bulk_bytes(S);
    unsigned char* region = g + (size_t)blockIdx.x * (R * B);
#pragma unroll
    for (int q = 0; q < R; q++) {
      unsigned char* p = region + q * B;
      const unsigned s = sbase + q * B % SMEM;
      if constexpr (S == S_BLKCP4K)
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(p), "r"(s), "n"(B) : "memory");
      else
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(p), "r"(s),
                     "n"(B) : "memory");
    }
    asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 2;" ::: "memory");
  }
  template <int S>
  __device__ __forceinline__ void trip(uint32_t t) {
    if constexpr (bulk(S)) {
      if (lane == 0) bulk_trip<S>();
    } else {
#pragma unroll
      for (int q = 0; q < R; q++) op<S>(q, t);
      if constexpr (S == S_LDGSTS) asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 2;" ::: "memory");
    }
  }
  template <int S>
  __device__ __forceinline__ void finish() {
    if constexpr (bulk(S)) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    if constexpr (S == S_LDGSTS) asm volatile("cp.async.wait_all;" ::: "memory");
  }
};

constexpr bool clustered(int s) { return s == S_DSM_ST4 || s == S_DSM_RED; }

template <int S, bool SIDE_ON>
__device__ __forceinline__ void honest_side(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  const int w = threadIdx.x >> 5;
  Honest h(a.mode);
  Side s(a, w - HW, sb, clustered(S));
  if constexpr (clustered(S)) cluster_sync();
  OP_BEGIN_STOP
  if (w < HW) {
    OP_LOOP(it) { h.step(); }
    OP_DONE
  } else if (SIDE_ON && (!bulk(S) || w == HW)) {
    OP_UNTIL(n) { s.template trip<S>((uint32_t)n); }
    OP_TRIPS(n)
    s.template finish<S>();
  }
  OP_END_STOP
  if constexpr (clustered(S)) cluster_sync();
  OP_SINK(h.fold() ^ s.acc)
}

template <int S>
__device__ __forceinline__ void side_alone(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  Side s(a, threadIdx.x >> 5, sb, clustered(S));
  if constexpr (clustered(S)) cluster_sync();
  OP_BEGIN
  OP_LOOP(it) { s.template trip<S>((uint32_t)it); }
  s.template finish<S>();
  OP_END
  if constexpr (clustered(S)) cluster_sync();
  OP_SINK(s.acc)
}

#define KHS(nm, S, ON, CL)                                                                                        \
  extern "C" __global__ void CL __launch_bounds__((HW + SW) * 32, 1) nm(OpArgs a) { honest_side<S, ON>(a); }
#define KSA(nm, S, CL) \
  extern "C" __global__ void CL __launch_bounds__(bulk(S) ? 32 : SW * 32, 1) nm(OpArgs a) { side_alone<S>(a); }
#define NOCL
#define CL2 __cluster_dims__(2, 1, 1)
KHS(honest, S_RED32, false, NOCL)
KHS(honest_cl, S_DSM_ST4, false, CL2)
#define SIDE(s, S, CL) KHS(hs_##s, S, true, CL) KSA(sa_##s, S, CL)
SIDE(red32, S_RED32, NOCL)
SIDE(red64, S_RED64, NOCL)
SIDE(redv4, S_REDV4, NOCL)
SIDE(redv4h, S_REDV4H, NOCL)
SIDE(atomg32, S_ATOMG32, NOCL)
SIDE(ldgsts, S_LDGSTS, NOCL)
SIDE(blkred512, S_BLKRED512, NOCL)
SIDE(blkred4k, S_BLKRED4K, NOCL)
SIDE(blkcp4k, S_BLKCP4K, NOCL)
SIDE(dsm_st4, S_DSM_ST4, CL2)
SIDE(dsm_red, S_DSM_RED, CL2)

#define IMMA "IMMA.16832.S8.S8"
#define HTPB ((HW + SW) * 32)
// expect: the honest loop's IMMAs; the side loop's ops (W words a warp instruction of the counted op). A bulk side role
// is one warp (BW 1), every other 4.
#define P_SIDE(s, OP, N, W, CLN, IT, BW)                                                                          \
  {.name = "hs_" #s, .kernel = (const void*)hs_##s, .tpb = HTPB, .expect = IMMA "=8;" OP "=" #N,                  \
   .counted = OP, .warps = "16;" #BW, .words = W, .dyn_smem = SMEM, .cluster = CLN, .iters = 2048, .until = 2},   \
  {.name = "sa_" #s, .kernel = (const void*)sa_##s, .tpb = BW * 32, .expect = OP "=" #N, .words = W,               \
   .dyn_smem = SMEM, .cluster = CLN, .iters = IT}

static const OpProbe PROBES[] = {
    {.name = "honest", .kernel = (const void*)honest, .tpb = HTPB, .expect = IMMA "=8", .warps = "16",
     .dyn_smem = SMEM, .iters = 2048},
    {.name = "honest_cl", .kernel = (const void*)honest_cl, .tpb = HTPB, .expect = IMMA "=8", .warps = "16",
     .dyn_smem = SMEM, .cluster = 2, .iters = 2048},
    P_SIDE(red32, "REDG*", 8, 32, 1, 1024, 4),
    P_SIDE(red64, "REDG*", 8, 64, 1, 1024, 4),
    P_SIDE(redv4, "REDG*", 8, 128, 1, 512, 4),
    P_SIDE(redv4h, "REDG*", 8, 128, 1, 512, 4),
    P_SIDE(atomg32, "ATOMG*", 8, 32, 1, 1024, 4),
    P_SIDE(ldgsts, "LDGSTS*", 8, 128, 1, 1024, 4),
    P_SIDE(blkred512, "UBLKRED*", 8, 128, 1, 512, 1),
    P_SIDE(blkred4k, "UBLKRED*", 8, 1024, 1, 128, 1),
    P_SIDE(blkcp4k, "UBLKCP*", 8, 1024, 1, 128, 1),
    P_SIDE(dsm_st4, "ST*", 8, 128, 2, 512, 4),
    P_SIDE(dsm_red, "ATOM*", 8, 32, 2, 32, 4),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
