// The copy routes a prover could use to place words it has already computed (copy-floor): each role copies tiles
// whose coordinate it computes from a register, one coordinate a tile, as an adaptive prover's would come from the
// coins. Each role alone (`sa_*`: its words an SM-clock, so 4096 / (rate x words) units a word) and beside the honest
// kernel (`hs_*`: ../opcode_probe/probes/honest.cuh's 16 IMMA warps, the role an until loop, as side.cu), so a copy
// unit that works off the SM's issue slots shows what it adds beside honest's MMAs and folds. `hs_*z` sleeps
// (nanosleep) after each pass of its poll, so the poll's own issue slots leave honest's and only the copies' cost stays.
//   bulk (one lane of one warp; the source pool 8 MB, L2-resident, or 256 MB for _hbm):
//     g2s<B>    cp.async.bulk global -> shared (TMA, UBLKCP), B-byte tiles, one mbarrier a trip
//     s2g<B>    cp.async.bulk shared -> global, bulk_group, at most 5 trips in flight
//     dsm<B>    cp.async.bulk shared -> the peer CTA's shared memory (a 2-CTA cluster), the peer's mbarrier
//   lane (SW warps, 8 copies a trip, one coordinate a trip for the warp's 8 rows):
//     ldgsts4 / ldgsts16   cp.async global -> shared, 4 or 16 bytes a lane (LDGSTS)
//     ldssts4 / ldssts16   LDS then STS, 4 or 16 bytes a lane (shared -> register -> shared)
//     ldgstg16             LDG.128 then STG.128 (global -> register -> global)
// An mbarrier role retires a trip when its barrier's phase has completed (mbarrier.test_wait, no spin loop, so the gate
// sees one flat loop) and keeps at most NB trips in flight; alone it runs until every trip it issued has landed, beside
// honest it counts the trips that landed by the honest warps' stop.
#include "../opcode_probe/probe.cuh"
#include "../opcode_probe/probes/honest.cuh"

constexpr int HW = 16, SW = 8, R = 8, NB = 16, STAGE = 32 * 1024;
constexpr uint32_t L2POOL = 8u << 20, HBMPOOL = 256u << 20, LSRC = 64 * 1024;

enum { G2S, G2S_HBM, S2G, DSM, LDGSTS4, LDGSTS16, LDSSTS4, LDSSTS16, LDGSTG16 };
__host__ __device__ constexpr bool is_bulk(int k) { return k <= DSM; }
// a bulk trip's copies: R, or as many B-byte tiles as one STAGE holds
__host__ __device__ constexpr int nr(int b) { return b * R > STAGE ? STAGE / b : R; }

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ bool test_wait(unsigned bar, unsigned parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
               : "=r"(ok)
               : "r"(bar), "r"(parity)
               : "memory");
  return ok;
}

template <int K, int B>
struct Bulk {
  static constexpr int N = nr(B);
  static constexpr uint32_t POOL = K == G2S_HBM ? HBMPOOL : L2POOL;
  unsigned char* g;
  unsigned st, bars, pst, pbars;  // the stage and barriers, own and (dsm) the peer's
  uint32_t issued, done, salt;
  __device__ Bulk(const OpArgs& a, unsigned char* sb) : g(a.g), issued(0), done(0) {
    st = (unsigned)__cvta_generic_to_shared(sb);
    bars = st + STAGE;
    pst = st;
    pbars = bars;
    salt = blockIdx.x * 0x10001u;
    if constexpr (K == DSM) {
      unsigned r;
      asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
      pst = mapa(st, r ^ 1u);
      pbars = mapa(bars, r ^ 1u);
    }
  }
  __device__ void init() {
    for (int b = 0; b < NB; b++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bars + 8 * b) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __device__ __forceinline__ void issue(uint32_t t) {
    const unsigned slot = 8 * (t % NB);
    if constexpr (K != S2G)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bars + slot), "r"(N * B) : "memory");
#pragma unroll
    for (int q = 0; q < N; q++) {
      unsigned char* p = g + (size_t)(coin(salt + t * N + q) & (POOL / B - 1)) * B;
      const unsigned s = st + q * B;
      if constexpr (K == G2S || K == G2S_HBM)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s),
                     "l"(p), "n"(B), "r"(bars + slot) : "memory");
      if constexpr (K == S2G)
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(p), "r"(s), "n"(B) : "memory");
      if constexpr (K == DSM)
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         pst + q * B),
                     "r"(s), "n"(B), "r"(pbars + slot) : "memory");
    }
    if constexpr (K == S2G) asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 4;" ::: "memory");
  }
  // retire the oldest trip if its copies landed, else issue one while fewer than NB are in flight
  __device__ __forceinline__ void pass(uint32_t limit) {
    if constexpr (K == S2G) {
      issue(issued);
      done = ++issued;
    } else {
      if (done < issued && test_wait(bars + 8 * (done % NB), (done / NB) & 1u)) done++;
      else if (issued < limit && issued - done < NB) issue(issued++);
    }
  }
  // outside the timed region: every issued copy lands before the block exits
  __device__ void drain() {
    if constexpr (K == S2G) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    else {
#pragma unroll 1
      while (done < issued)
        if (test_wait(bars + 8 * (done % NB), (done / NB) & 1u)) done++;
    }
  }
};

template <int K>
struct Lane {
  static constexpr uint32_t W = K == LDGSTS4 || K == LDSSTS4 ? 4 : 16, ROW = 32 * W, TILE = R * ROW;
  unsigned char* g;
  unsigned src, dst;
  uint32_t lane, gw, acc;
  __device__ Lane(const OpArgs& a, int sw, unsigned char* sb) : g(a.g) {
    lane = threadIdx.x & 31;
    gw = blockIdx.x * SW + sw;
    src = (unsigned)__cvta_generic_to_shared(sb);
    dst = src + LSRC + sw * TILE;
    acc = 0;
  }
  __device__ __forceinline__ void trip(uint32_t t) {
    const uint32_t c = coin(gw * 0x10000u + t);
    if constexpr (K == LDGSTS4 || K == LDGSTS16) {
      const unsigned char* p = g + (size_t)(c & (L2POOL / TILE - 1)) * TILE + lane * W;
#pragma unroll
      for (int q = 0; q < R; q++) {
        if constexpr (K == LDGSTS4)
          asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst + q * ROW + lane * W), "l"(p + q * ROW)
                       : "memory");
        else
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + q * ROW + lane * W), "l"(p + q * ROW)
                       : "memory");
      }
      asm volatile("cp.async.commit_group;\n\tcp.async.wait_group 2;" ::: "memory");
    }
    if constexpr (K == LDSSTS4 || K == LDSSTS16 || K == LDGSTG16) {
      // R loads, then R stores, so the loads are in flight together
      uint32_t v[R][W / 4];
      if constexpr (K == LDGSTG16) {
        const unsigned char* p = g + (size_t)(c & (L2POOL / TILE - 1)) * TILE + lane * W;
#pragma unroll
        for (int q = 0; q < R; q++)
          asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(v[q][0]), "=r"(v[q][1]), "=r"(v[q][2]), "=r"(v[q][3])
                       : "l"(p + q * ROW)
                       : "memory");
        unsigned char* d = g + L2POOL + (size_t)gw * TILE + lane * W;
#pragma unroll
        for (int q = 0; q < R; q++)
          asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(d + q * ROW), "r"(v[q][0]), "r"(v[q][1]),
                       "r"(v[q][2]), "r"(v[q][3]) : "memory");
      } else {
        const unsigned s = src + (c & (LSRC / TILE - 1)) * TILE + lane * W;
#pragma unroll
        for (int q = 0; q < R; q++) {
          if constexpr (W == 4)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v[q][0]) : "r"(s + q * ROW) : "memory");
          else
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v[q][0]), "=r"(v[q][1]), "=r"(v[q][2]), "=r"(v[q][3])
                         : "r"(s + q * ROW)
                         : "memory");
        }
#pragma unroll
        for (int q = 0; q < R; q++) {
          if constexpr (W == 4)
            asm volatile("st.shared.u32 [%0], %1;" ::"r"(dst + q * ROW + lane * W), "r"(v[q][0]) : "memory");
          else
            asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(dst + q * ROW + lane * W), "r"(v[q][0]),
                         "r"(v[q][1]), "r"(v[q][2]), "r"(v[q][3]) : "memory");
        }
      }
    }
  }
  __device__ __forceinline__ void finish() {
    if constexpr (K == LDGSTS4 || K == LDGSTS16) asm volatile("cp.async.wait_all;" ::: "memory");
  }
};

template <int K, int B>
__device__ __forceinline__ void bulk_alone(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  Bulk<K, B> b(a, sb);
  const bool l0 = (threadIdx.x & 31) == 0;
  if (l0 && K != S2G) b.init();
  __syncwarp();
  if constexpr (K == DSM) cluster_sync();
  OP_BEGIN
  if (l0) {
#pragma unroll 1
    while (b.done < (uint32_t)a.iters) b.pass((uint32_t)a.iters);
    if constexpr (K == S2G) b.drain();
  }
  OP_END
  if constexpr (K == DSM) cluster_sync();
  OP_SINK(b.done)
}

template <int K>
__device__ __forceinline__ void lane_alone(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  Lane<K> l(a, threadIdx.x >> 5, sb);
  OP_BEGIN
  OP_LOOP(it) { l.trip((uint32_t)it); }
  l.finish();
  OP_END
  OP_SINK(l.acc)
}

// the honest warps, and the bulk role in warp HW (one lane) or the lane role in warps HW.. (SIDE false: idle); NS > 0:
// the bulk role sleeps NS ns after each pass
template <int K, int B, bool SIDE, int NS = 0>
__device__ __forceinline__ void beside(const OpArgs& a) {
  extern __shared__ __align__(128) unsigned char sb[];
  const int w = threadIdx.x >> 5;
  const bool l0 = (threadIdx.x & 31) == 0;
  Honest h(a.mode);
  uint32_t side = 0;
  if constexpr (is_bulk(K)) {
    Bulk<K, B> b(a, sb);
    if (SIDE && w == HW && l0 && K != S2G) b.init();
    OP_BEGIN_STOP
    if (w < HW) {
      OP_LOOP(it) { h.step(); }
      OP_DONE
    } else if (SIDE && w == HW && l0) {
#pragma unroll 1
      while (!*(volatile int*)&op_stop_) {
        b.pass(~0u);
        if constexpr (NS > 0) __nanosleep(NS);
      }
      atomicAdd(&op_trips_, b.done);
    }
    OP_END_STOP
    if (SIDE && w == HW && l0) b.drain();
    side = b.done;
  } else {
    Lane<K> l(a, w - HW, sb);
    OP_BEGIN_STOP
    if (w < HW) {
      OP_LOOP(it) { h.step(); }
      OP_DONE
    } else if (SIDE) {
      OP_UNTIL(n) { l.trip((uint32_t)n); }
      OP_TRIPS(n)
      l.finish();
    }
    OP_END_STOP
    side = l.acc;
  }
  OP_SINK(h.fold() ^ side)
}

#define TB_BULK ((HW + 1) * 32)
#define TB_LANE ((HW + SW) * 32)
#define SA_BULK(nm, K, B, CL) \
  extern "C" __global__ void CL __launch_bounds__(32, 1) sa_##nm(OpArgs a) { bulk_alone<K, B>(a); }
#define HS_BULK(nm, K, B) \
  extern "C" __global__ void __launch_bounds__(TB_BULK, 1) hs_##nm(OpArgs a) { beside<K, B, true>(a); }
#define HS_BULKZ(nm, K, B, NS) \
  extern "C" __global__ void __launch_bounds__(TB_BULK, 1) hs_##nm(OpArgs a) { beside<K, B, true, NS>(a); }
#define SA_LANE(nm, K) \
  extern "C" __global__ void __launch_bounds__(SW * 32, 1) sa_##nm(OpArgs a) { lane_alone<K>(a); }
#define HS_LANE(nm, K) \
  extern "C" __global__ void __launch_bounds__(TB_LANE, 1) hs_##nm(OpArgs a) { beside<K, 0, true>(a); }
#define NOCL
SA_BULK(g2s256, G2S, 256, NOCL)
SA_BULK(g2s1k, G2S, 1024, NOCL)
SA_BULK(g2s4k, G2S, 4096, NOCL)
SA_BULK(g2s16k, G2S, 16384, NOCL)
SA_BULK(g2s4kh, G2S_HBM, 4096, NOCL)
SA_BULK(s2g1k, S2G, 1024, NOCL)
SA_BULK(s2g4k, S2G, 4096, NOCL)
SA_BULK(dsm1k, DSM, 1024, __cluster_dims__(2, 1, 1))
SA_BULK(dsm4k, DSM, 4096, __cluster_dims__(2, 1, 1))
HS_BULK(g2s1k, G2S, 1024)
HS_BULK(g2s4k, G2S, 4096)
HS_BULK(g2s4kh, G2S_HBM, 4096)
HS_BULK(s2g4k, S2G, 4096)
HS_BULK(g2s16k, G2S, 16384)
HS_BULKZ(g2s4kz, G2S, 4096, 256)
HS_BULKZ(g2s16kz, G2S, 16384, 256)
HS_BULKZ(s2g4kz, S2G, 4096, 256)
SA_LANE(ldgsts4, LDGSTS4)
SA_LANE(ldgsts16, LDGSTS16)
SA_LANE(ldssts4, LDSSTS4)
SA_LANE(ldssts16, LDSSTS16)
SA_LANE(ldgstg16, LDGSTG16)
HS_LANE(ldgsts4, LDGSTS4)
HS_LANE(ldgsts16, LDGSTS16)
HS_LANE(ldssts4, LDSSTS4)
HS_LANE(ldssts16, LDSSTS16)
extern "C" __global__ void __launch_bounds__(TB_BULK, 1) honest_b(OpArgs a) { beside<G2S, 4096, false>(a); }
extern "C" __global__ void __launch_bounds__(TB_LANE, 1) honest_l(OpArgs a) { beside<LDGSTS16, 0, false>(a); }

#define IMMA "IMMA.16832.S8.S8=8"
constexpr int BULK_SMEM = STAGE + NB * 8, LANE_SMEM = LSRC + SW * R * 512;
// bulk: one warp, its copies an iteration of the issue-and-retire loop (a pass issues at most one trip)
#define P_SA_BULK(nm, N, W, IT, CLN)                                                                             \
  {.name = "sa_" #nm, .kernel = (const void*)sa_##nm, .tpb = 32, .expect = "UBLKCP*=" #N, .words = W,             \
   .dyn_smem = BULK_SMEM, .cluster = CLN, .iters = IT}
#define P_HS_BULK(nm, N, W)                                                                                        \
  {.name = "hs_" #nm, .kernel = (const void*)hs_##nm, .tpb = TB_BULK, .expect = IMMA ";UBLKCP*=" #N,              \
   .counted = "UBLKCP*", .warps = "16;1", .words = W, .dyn_smem = BULK_SMEM, .iters = 2048, .until = 2}
#define P_SA_LANE(nm, OP, W, IT)                                                                                   \
  {.name = "sa_" #nm, .kernel = (const void*)sa_##nm, .tpb = SW * 32, .expect = OP, .words = W,                    \
   .dyn_smem = LANE_SMEM, .iters = IT}
// beside honest: the until loop's poll of the stop flag is one more LDS a trip
#define P_HS_LANE(nm, OP, CNT, W)                                                                                  \
  {.name = "hs_" #nm, .kernel = (const void*)hs_##nm, .tpb = TB_LANE, .expect = IMMA ";" OP, .counted = CNT,     \
   .warps = "16;8", .words = W, .dyn_smem = LANE_SMEM, .iters = 2048, .until = 2}

static const OpProbe PROBES[] = {
    {.name = "honest_b", .kernel = (const void*)honest_b, .tpb = TB_BULK, .expect = IMMA, .warps = "16",
     .dyn_smem = BULK_SMEM, .iters = 2048, .note = "honest beside an idle bulk warp"},
    {.name = "honest_l", .kernel = (const void*)honest_l, .tpb = TB_LANE, .expect = IMMA, .warps = "16",
     .dyn_smem = LANE_SMEM, .iters = 2048, .note = "honest beside 8 idle lane warps"},
    P_SA_BULK(g2s256, 8, 64, 4096, 1),
    P_SA_BULK(g2s1k, 8, 256, 2048, 1),
    P_SA_BULK(g2s4k, 8, 1024, 512, 1),
    P_SA_BULK(g2s16k, 2, 4096, 512, 1),
    P_SA_BULK(g2s4kh, 8, 1024, 512, 1),
    P_SA_BULK(s2g1k, 8, 256, 2048, 1),
    P_SA_BULK(s2g4k, 8, 1024, 512, 1),
    P_SA_BULK(dsm1k, 8, 256, 2048, 2),
    P_SA_BULK(dsm4k, 8, 1024, 512, 2),
    P_HS_BULK(g2s1k, 8, 256),
    P_HS_BULK(g2s4k, 8, 1024),
    P_HS_BULK(g2s4kh, 8, 1024),
    P_HS_BULK(s2g4k, 8, 1024),
    P_HS_BULK(g2s16k, 2, 4096),
    P_HS_BULK(g2s4kz, 8, 1024),
    P_HS_BULK(g2s16kz, 2, 4096),
    P_HS_BULK(s2g4kz, 8, 1024),
    P_SA_LANE(ldgsts4, "LDGSTS*=8", 32, 1024),
    P_SA_LANE(ldgsts16, "LDGSTS*=8", 128, 512),
    P_SA_LANE(ldssts4, "LDS*=8,STS*=8", 32, 2048),
    P_SA_LANE(ldssts16, "LDS*=8,STS*=8", 128, 1024),
    P_SA_LANE(ldgstg16, "LDG*=8,STG*=8", 128, 256),
    P_HS_LANE(ldgsts4, "LDGSTS*=8", "LDGSTS*", 32),
    P_HS_LANE(ldgsts16, "LDGSTS*=8", "LDGSTS*", 128),
    P_HS_LANE(ldssts4, "LDS*=9,STS*=8", "STS*", 32),
    P_HS_LANE(ldssts16, "LDS*=9,STS*=8", "STS*", 128),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
