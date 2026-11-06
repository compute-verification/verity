// Hopper (sm_90a, H200): the rows the H200 design reads (JI90_0..2, JPA90_0/1, JPI90_0/1, issue, M) against
// facet_break's sm_120 break (few accumulators, one-read lanes by an immediate, 2 warps a SMSP) moved beside wgmma, and
// the wgmma kinds the design's menu clause reads (f16, tf32, s8 RS) in honest's shape. Built and gated like
// sm90_wgmma.cu (run.sh sm90jb).
//   jb_<form>_s<NS>u<U>p<P>_<lane>c<C>_r<R>_w<W>
//       W warpgroups (W warps a SMSP), each: U x NS commit groups an iteration, group g 4 wgmma of <form> into accumulator
//       set g mod NS, then R lane ops (P 0) or R/4 after each wgmma (P 1), then wait_group NS-1 (NS = 1: honest's shape)
//   jbU_<lane>c<C>_r<R>_m<MW>l<LW>
//       MW warpgroups wgmma s8 m64n256k32 (group4, wait_group 1) for fixed iterations; LW lane warps a SMSP run R lane
//       ops a trip until they finish (OP_UNTIL), so neither side counts idle time
//   jba_<lane>c<C>_w<W>      the lane alone, W warps a SMSP, 256 ops a trip
//   jbk_<kind>_mix_r<R>      honest's shape (wg_mix) with another wgmma kind; jbkU_<kind>_lif_r<R> its exact point
// Lanes act on C chains of their own registers. One-read lanes: fmuli, faddi (FMUL/FADD by an immediate), axi (add and
// xor of an immediate in turn on each chain: IADD3 + LOP3, one alone would fold), imli (multiply and xor by an
// immediate in turn: IMAD + LOP3, LI's pipes with one register read). Two-read lanes: ffma, li (LOP3 + IMAD, LI's mix).
#include "wg_common.cuh"

enum { J_FMULI, J_FADDI, J_AXI, J_IMLI, J_FFMA, J_LI, J_MIX, J_LIF };

template <int C>
struct LaneC {
  uint32_t r[C];
  float f[C];
  DC_DEV LaneC(uint32_t tid, int mode) {
#pragma unroll
    for (int i = 0; i < C; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu + (uint32_t)mode) | 1u;
      f[i] = 1.0f + (float)(r[i] & 0xffu) * 0x1p-12f;
    }
  }
  template <int X>
  DC_DEV void op(int q) {
    const int s = q % C, t = (q + 1) % C, k = q / C;
    if constexpr (X == J_FMULI) asm volatile("mul.rn.f32 %0, %0, 0f3F800010;" : "+f"(f[s]));
    if constexpr (X == J_FADDI) asm volatile("add.rn.f32 %0, %0, 0f33800000;" : "+f"(f[s]));
    if constexpr (X == J_AXI) {
      if (k & 1) asm volatile("xor.b32 %0, %0, 0x5A5A5A5A;" : "+r"(r[s]));
      else asm volatile("add.u32 %0, %0, 0x9E3779B9;" : "+r"(r[s]));
    }
    if constexpr (X == J_IMLI) {
      if (k & 1) asm volatile("xor.b32 %0, %0, 0x5A5A5A5A;" : "+r"(r[s]));
      else asm volatile("mul.lo.u32 %0, %0, 0x9E3779B1;" : "+r"(r[s]));
    }
    if constexpr (X == J_FFMA) asm volatile("fma.rn.f32 %0, %0, %1, 0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == J_LI) {
      if (q & 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
      else asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    }
    if constexpr (X == J_MIX) {  // lanes.cuh's mix: the pair's chain advances, so no add sees a constant neighbour
      const int p = (q >> 1) % C, pn = ((q >> 1) + 1) % C;
      if (q & 1) asm volatile("fma.rn.f32 %0, %0, %1, 0f3F800000;" : "+f"(f[p]) : "f"(f[pn]));
      else asm volatile("add.u32 %0, %0, %1;" : "+r"(r[p]) : "r"(r[pn]));
    }
    if constexpr (X == J_LIF) {  // LI with one FFMA in eight (lanes.cuh's lif)
      if (q % 8 == 7) asm volatile("fma.rn.f32 %0, %0, %1, 0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
      else if ((q % 8 + q / 8) % 2 == 0) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
      else asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    }
  }
  // ops q0 .. q0 + N - 1 of the stream: a group continues its predecessor's, so ptxas can't merge two immediates
  template <int X, int N>
  DC_DEV void ops(int q0) {
#pragma unroll
    for (int q = 0; q < N; q++) op<X>(q0 + q);
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < C; i++) x += r[i] ^ __float_as_uint(f[i]);
    return x;
  }
};

template <class F, int NS, int U, int P, int X, int C, int R>
DC_DEV void jb_mix(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  typename F::Acc d[NS][F::NA] = {};
  LaneC<C> l(threadIdx.x, a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int g = 0; g < U * NS; g++) {
      auto& acc = d[g % NS];
      wgmma_fence();
      if constexpr (P == 0) {
        F::mma(acc, t, g < NS ? 0 : 1);
        F::mma(acc, t, 1);
        F::mma(acc, t, 1);
        F::mma(acc, t, 1);
        wgmma_commit();
        l.template ops<X, R>(g * R);
      } else {
        F::mma(acc, t, g < NS ? 0 : 1);
        l.template ops<X, R / 4>(g * R);
        F::mma(acc, t, 1);
        l.template ops<X, R / 4>(g * R + R / 4);
        F::mma(acc, t, 1);
        l.template ops<X, R / 4>(g * R + R / 2);
        F::mma(acc, t, 1);
        wgmma_commit();
        l.template ops<X, R / 4>(g * R + 3 * R / 4);
      }
      wgmma_wait<NS - 1>();
    }
  }
  wgmma_wait<0>();
#pragma unroll
  for (int j = 0; j < NS; j++) wgmma_hold(d[j]);
  OP_END
  uint32_t s = l.fold() ^ threadIdx.x;
#pragma unroll
  for (int j = 0; j < NS; j++) s ^= fold_acc(d[j]);
  OP_SINK(s)
}

template <int X, int C, int R, int MW, class F = F_s8_n256>
DC_DEV void jb_until(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  const bool mma_role = threadIdx.x < 128 * MW;
  typename F::Acc d[F::NA] = {};
  LaneC<C> l(threadIdx.x, a.mode);
  OP_BEGIN_STOP
  if (mma_role) {
    OP_LOOP(it) {
      group4<F>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    OP_DONE
    wgmma_hold(d);
  } else {
    OP_UNTIL(n) { l.template ops<X, R>(0); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((mma_role ? fold_acc(d) : l.fold()) ^ threadIdx.x)
}

template <int X, int C>
DC_DEV void jb_alone(const OpArgs& a) {
  LaneC<C> l(threadIdx.x, a.mode);
  OP_BEGIN
  OP_LOOP(it) { l.template ops<X, 256>(0); }
  OP_END
  OP_SINK(l.fold() ^ threadIdx.x)
}

// honest's shape (wg_mix) with any form, lanes from LaneC<8>
template <class F, int X, int R>
DC_DEV void jb_kmix(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  typename F::Acc d[F::NA] = {};
  LaneC<8> l(threadIdx.x, a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t);
    l.template ops<X, R>(0);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

// ---- kernels ----------------------------------------------------------------------------------------------------------
#define JB(NAME, F, NS, U, P, X, C, R, W, PAT) \
  OP_KERNEL(jb_##NAME, 128 * W) { jb_mix<F_##F, NS, U, P, X, C, R>(a); }
#define JBU(NAME, X, C, R, MW, LW, PAT, WARPS) \
  OP_KERNEL(jbU_##NAME, 128 * (MW + LW)) { jb_until<X, C, R, MW>(a); }
#define JBA(NAME, X, C, W, PAT) \
  OP_KERNEL(jba_##NAME, 128 * W) { jb_alone<X, C>(a); }
#define JBK(NAME, F, X, R, PAT) \
  OP_KERNEL(jbk_##NAME, 256) { jb_kmix<F_##F, X, R>(a); }
#define JBKU(NAME, F, X, R, PAT) \
  OP_KERNEL(jbkU_##NAME, 384) { jb_until<X, 8, R, 2, F_##F>(a); }
#include "sm90_jbreak.inc"
#undef JB
#undef JBU
#undef JBA
#undef JBK
#undef JBKU

#define JB(NAME, F, NS, U, P, X, C, R, W, PAT) \
  {.name = "jb_" #NAME, .kernel = (const void*)jb_##NAME, .tpb = 128 * W, .expect = PAT, .iters = 2048},
#define JBU(NAME, X, C, R, MW, LW, PAT, WARPS)                                                                       \
  {.name = "jbU_" #NAME, .kernel = (const void*)jbU_##NAME, .tpb = 128 * (MW + LW), .expect = PAT, .warps = WARPS, \
   .iters = 4096, .until = 2},
#define JBA(NAME, X, C, W, PAT) \
  {.name = "jba_" #NAME, .kernel = (const void*)jba_##NAME, .tpb = 128 * W, .expect = PAT, .iters = 2048},
#define JBK(NAME, F, X, R, PAT) \
  {.name = "jbk_" #NAME, .kernel = (const void*)jbk_##NAME, .tpb = 256, .expect = PAT, .iters = 4096},
#define JBKU(NAME, F, X, R, PAT) \
  {.name = "jbkU_" #NAME, .kernel = (const void*)jbkU_##NAME, .tpb = 384, .expect = PAT, .warps = "8;4", \
   .iters = 4096, .until = 2},
static const OpProbe PROBES[] = {
#include "sm90_jbreak.inc"
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
