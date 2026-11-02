// Side units beside the honest kernel (eps_side): the words a cycle a unit
// that works off the SM's vector pipes (L2 reductions and atomics, cp.async, TMA bulk copies and reductions, DSMEM,
// texture point fetch and bilinear filtering over 32 MB, and tld1c/texc over a window the texture cache holds; tld1d,
// texd, atomgd, tld1cd: tld1, texbil, atomg32, tld1c with 4x, 2x, 4x, 4x the fetches in flight; texr: texd on two rows
// (32 KB); tld1w, texw, tld1cw: tld1d, texr and tld1cd with 12 side warps; redblk: REDG.32 and a 1 KB bulk reduce a warp in one trip, both L2 paths) adds
// beside a saturated int8 IMMA kernel that folds every output word with duo (receipt_folds.cuh, the served fold), and
// whether the MMAs keep their rate. Joint points, rows.jsonl only.
//   honest        16 warps: 8 IMMA an iteration, then rf_duo_step on each accumulator's 4 words (side warps idle)
//   honest16      the same in a block of only its 16 warps
//   hs_<s>        the same 16 warps (OP_LOOP), and the side role (OP_UNTIL) running unit s only while they do
//   sa_<s>        the side role alone, the same warps (OP_LOOP)
// The side role is 4 warps (one a SMSP; 12 for the wide roles) issuing 8 ops a trip (ops()), but a bulk unit's role is one
// warp whose lane 0 issues: a
// TMA bulk op takes uniform operands, so its addresses come from blockIdx and the op's slot only, never the lane. Vector
// reductions are 128 bits a lane at most (v4.f32, v4.bf16x2).
#include "../probe.cuh"
#include "honest.cuh"

constexpr int HW = 16, SW = 4, R = 8, SMEM = 64 * 1024;
typedef unsigned long long u64;

enum {
  S_RED32, S_RED64, S_REDV4, S_REDV4H, S_ATOMG32, S_LDGSTS, S_BLKRED512, S_BLKRED4K, S_BLKCP4K, S_DSM_ST4, S_DSM_RED,
  S_TLD1, S_TEXBIL, S_REDBLK, S_TLD1C, S_TEXC, S_TLD1D, S_TEXD, S_ATOMGD, S_TEXR, S_TLD1CD
};

__host__ __device__ constexpr bool bulk(int s) { return s == S_BLKRED512 || s == S_BLKRED4K || s == S_BLKCP4K; }
// ops a trip: R, but the deep roles (TLD, TEX, ATOMG with a result) keep 4x or 2x as many fetches in flight a warp
__host__ __device__ constexpr int ops(int s) {
  return s == S_TLD1D || s == S_ATOMGD || s == S_TLD1CD ? 4 * R : s == S_TEXD || s == S_TEXR ? 2 * R : R;
}
__host__ __device__ constexpr unsigned bulk_bytes(int s) { return s == S_BLKRED512 ? 512u : 4096u; }

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
  cudaTextureObject_t tex1, tex2;
  uint32_t gtid, total, acc, lane, sw;
  unsigned sbase, peer;
  __device__ Side(const OpArgs& a, int side_warp, int nsw, unsigned char* sb, bool cluster) {
    g = a.g;
    tex1 = a.tex1;
    tex2 = a.tex2;
    lane = threadIdx.x & 31;
    sw = side_warp;
    gtid = (blockIdx.x * nsw + sw) * 32 + lane;
    total = gridDim.x * nsw * 32;
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
    const uint32_t seq = trip * ops(S) + q, v = gtid * 2654435761u ^ seq;
    if constexpr (S == S_RED32 || S == S_REDBLK) {
      constexpr uint32_t pool = (S == S_REDBLK ? 32u << 20 : 64u << 20) / 4;
      uint32_t* p = reinterpret_cast<uint32_t*>(g) + ((gtid + seq * total) & (pool - 1));
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
    if constexpr (S == S_ATOMG32 || S == S_ATOMGD) {
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
    if constexpr (S == S_TLD1 || S == S_TLD1D)
      acc ^= tex1Dfetch<unsigned>(tex1, (int)((gtid + seq * total) & ((64u << 20) / 4 - 1)));
    if constexpr (S == S_TEXBIL || S == S_TEXD) {
      // off-texel coordinates, so every fetch blends 4 texels with nonzero weights
      const float x = (float)((gtid + seq * 37u) & 1023u) + 0.25f, y = (float)((gtid >> 10) + seq) + 0.75f;
      const float4 f = tex2D<float4>(tex2, x, y);
      acc ^= __float_as_uint(f.x) ^ __float_as_uint(f.y) ^ __float_as_uint(f.z) ^ __float_as_uint(f.w);
    }
    // the same two fetches inside a 32 KB (TLD) or 20 KB (TEX) window every block shares, so they hit the texture cache
    if constexpr (S == S_TLD1C)
      acc ^= tex1Dfetch<unsigned>(tex1, (int)(((sw * R + q) * 32u + lane + (trip & 7u) * 1024u) & 8191u));
    if constexpr (S == S_TLD1CD)
      acc ^= tex1Dfetch<unsigned>(tex1, (int)(((sw * 4u * R + q) * 32u + lane + (trip & 7u) * 1024u) & 8191u));
    if constexpr (S == S_TEXC) {
      const float x = (float)((((sw * R + q) * 8u + (lane & 7u)) + (trip & 7u) * 32u) & 255u) + 0.25f,
                  y = (float)(lane >> 3) + 0.75f;
      const float4 f = tex2D<float4>(tex2, x, y);
      acc ^= __float_as_uint(f.x) ^ __float_as_uint(f.y) ^ __float_as_uint(f.z) ^ __float_as_uint(f.w);
    }
    // texbil's x walk on rows 0 and 1 only: 32 KB, and only two rows at the 16 KB pitch (texc's five alias in the cache)
    if constexpr (S == S_TEXR) {
      const float x = (float)((gtid + seq * 37u) & 1023u) + 0.25f;
      const float4 f = tex2D<float4>(tex2, x, 0.75f);
      acc ^= __float_as_uint(f.x) ^ __float_as_uint(f.y) ^ __float_as_uint(f.z) ^ __float_as_uint(f.w);
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
      for (int q = 0; q < ops(S); q++) op<S>(q, t);
      if constexpr (S == S_LDGSTS) asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 2;" ::: "memory");
      if constexpr (S == S_REDBLK) {
        // and one 1 KB bulk reduce a warp, from lane 0, into the warp's own region past the REDG pool
        if (lane == 0) {
          unsigned char* p = g + (32u << 20) + ((size_t)(blockIdx.x * SW + sw) << 10);
          asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], 1024;\n\t"
                       "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 2;" ::"l"(p),
                       "r"(sbase + sw * 1024u) : "memory");
        }
      }
    }
  }
  template <int S>
  __device__ __forceinline__ void finish() {
    if constexpr (bulk(S) || S == S_REDBLK) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    if constexpr (S == S_LDGSTS) asm volatile("cp.async.wait_all;" ::: "memory");
  }
};

__host__ __device__ constexpr bool clustered(int s) { return s == S_DSM_ST4 || s == S_DSM_RED; }

template <int S, bool SIDE_ON>
__device__ __forceinline__ void honest_side(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  const int w = threadIdx.x >> 5;
  Honest h(a.mode);
  Side s(a, w - HW, blockDim.x / 32 - HW, sb, clustered(S));
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
  Side s(a, threadIdx.x >> 5, blockDim.x / 32, sb, clustered(S));
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
extern "C" __global__ void __launch_bounds__(HW * 32, 1) honest16(OpArgs a) { honest_side<S_RED32, false>(a); }
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
SIDE(tld1, S_TLD1, NOCL)
SIDE(texbil, S_TEXBIL, NOCL)
SIDE(redblk, S_REDBLK, NOCL)
SIDE(tld1c, S_TLD1C, NOCL)
SIDE(texc, S_TEXC, NOCL)
SIDE(tld1d, S_TLD1D, NOCL)
SIDE(texd, S_TEXD, NOCL)
SIDE(atomgd, S_ATOMGD, NOCL)
SIDE(texr, S_TEXR, NOCL)
SIDE(tld1cd, S_TLD1CD, NOCL)
// the wide side roles: SWW warps beside the same 16 honest warps (the honest block's registers fit (HW + SWW) * 32)
constexpr int SWW = 12;
#define SIDEW(s, S)                                                                                                \
  extern "C" __global__ void __launch_bounds__((HW + SWW) * 32, 1) hs_##s(OpArgs a) { honest_side<S, true>(a); } \
  extern "C" __global__ void __launch_bounds__(SWW * 32, 1) sa_##s(OpArgs a) { side_alone<S>(a); }
SIDEW(tld1w, S_TLD1D)
SIDEW(texw, S_TEXR)
SIDEW(tld1cw, S_TLD1CD)

#define IMMA "IMMA.16832.S8.S8"
#define HTPB ((HW + SW) * 32)
// expect: the honest loop's IMMAs; the side loop's ops (W words a warp instruction of the counted op). A bulk side role
// is one warp (BW 1), every other 4.
#define P_SIDE_X(s, OP, N, W, CLN, IT, BW, MORE)                                                                  \
  {.name = "hs_" #s, .kernel = (const void*)hs_##s, .tpb = HTPB, .expect = IMMA "=8;" OP "=" #N MORE,             \
   .counted = OP, .warps = "16;" #BW, .words = W, .dyn_smem = SMEM, .cluster = CLN, .iters = 2048, .until = 2},   \
  {.name = "sa_" #s, .kernel = (const void*)sa_##s, .tpb = BW * 32, .expect = OP "=" #N MORE, .words = W,          \
   .dyn_smem = SMEM, .cluster = CLN, .iters = IT}
#define P_SIDE(s, OP, N, W, CLN, IT, BW) P_SIDE_X(s, OP, N, W, CLN, IT, BW, "")
#define P_SIDE_W(s, OP, N, W, IT)                                                                                  \
  {.name = "hs_" #s, .kernel = (const void*)hs_##s, .tpb = (HW + SWW) * 32, .expect = IMMA "=8;" OP "=" #N,       \
   .counted = OP, .warps = "16;12", .words = W, .dyn_smem = SMEM, .iters = 2048, .until = 2},                     \
  {.name = "sa_" #s, .kernel = (const void*)sa_##s, .tpb = SWW * 32, .expect = OP "=" #N, .words = W,              \
   .dyn_smem = SMEM, .iters = IT}

static const OpProbe PROBES[] = {
    {.name = "honest", .kernel = (const void*)honest, .tpb = HTPB, .expect = IMMA "=8", .warps = "16",
     .dyn_smem = SMEM, .iters = 2048},
    {.name = "honest_cl", .kernel = (const void*)honest_cl, .tpb = HTPB, .expect = IMMA "=8", .warps = "16",
     .dyn_smem = SMEM, .cluster = 2, .iters = 2048},
    {.name = "honest16", .kernel = (const void*)honest16, .tpb = HW * 32, .expect = IMMA "=8", .warps = "16",
     .dyn_smem = SMEM, .iters = 2048, .note = "honest alone in a block of its 16 warps (no side warps)"},
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
    P_SIDE(tld1, "TLD*", 8, 32, 1, 1024, 4),
    P_SIDE(texbil, "TEX*", 8, 128, 1, 512, 4),
    P_SIDE_X(redblk, "REDG*", 8, 32, 1, 1024, 4, ",UBLKRED*=1"),
    P_SIDE(tld1c, "TLD*", 8, 32, 1, 1024, 4),
    P_SIDE(texc, "TEX*", 8, 128, 1, 512, 4),
    P_SIDE(tld1d, "TLD*", 32, 32, 1, 256, 4),
    P_SIDE(texd, "TEX*", 16, 128, 1, 256, 4),
    P_SIDE(atomgd, "ATOMG*", 32, 32, 1, 256, 4),
    P_SIDE(texr, "TEX*", 16, 128, 1, 256, 4),
    P_SIDE_W(tld1w, "TLD*", 32, 32, 256),
    P_SIDE_W(texw, "TEX*", 16, 128, 256),
    P_SIDE(tld1cd, "TLD*", 32, 32, 1, 256, 4),
    P_SIDE_W(tld1cw, "TLD*", 32, 32, 256),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
