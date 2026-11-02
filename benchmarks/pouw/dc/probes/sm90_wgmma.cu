// The tensor measurement M3 on Hopper (sm_90a, H100/H200) as opcode probes.
// Build with ../opcode_probe/probe.py build --arch sm_90a; every probe passes the SASS gate, the poisoned sink and its
// no-write control before it is timed (probe.cuh). Accumulators stay in registers; A and B come from one shared-memory
// tile through no-swizzle K-major descriptors (dc.cuh). Each iteration's first wgmma overwrites D (scale-d 0), so FP
// accumulators stay finite on random operands.
//   wg_<form>_a0 / _a1    2 warpgroups a CTA, 4 wgmma a commit group, wait_group 0 / 1 after it (the form's rate alone)
//   wgz_<form>_a1         the same on zero operands (third-party rates differ by data, [LUO])
//   wgspec_<lane>_r<R>    warpgroups 0-1 wgmma s8 m64n256k32 (as wg_s8_n256_a1), warpgroup 2 R lane ops an iteration
//   wgmix_<lane>_r<R>     each of 2 warpgroups issues its 4 wgmma, then R lane ops, then waits (the honest kernel's
//                         shape: the fold runs while the group is in flight)
//   imma_s8_k32 / _k16    mma.sync on sm_90 (TENSOR's table: k16 full rate, k32 at 2/3 per [LUO])
//   sync_<form>           the other mma.sync shapes. CUDA 13.0 lowers e4m3 m16n8k32 to 2 HMMA.16816.F32 + 4 FADD (no
//                         native FP8 mma.sync on sm_90) and s4 m16n8k64 to a CALL into an emulation routine (not timed)
// Lanes (lanes.cuh): mix, imad, lop3, shfl (interleaved only: warp-specialized, ptxas pulls the clock read into a loop),
// duo (the fold's 3 ops a word), imadw (IMAD.WIDE.U32), and the memory lanes lde, atome, redg, atoms and lds128
// (wgU_ and wgmix_ only: LD.E, ATOM.E, REDG, ATOMS and LDS.128 beside the IGMMA, so the census's issue-only classes get
// their joint issue lines). Their alone rates and pipe pairs are scalar.cu's.
// dc/analyze.py turns rows.jsonl into words a tensor-pipe cycle and the issue cost a beside each lane.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "lanes.cuh"
#include "wg_common.cuh"
#include "../ffill.cuh"

// ---- kernels ----------------------------------------------------------------------------------------------------------
template <class F, int W>
DC_DEV void wg_alone(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  typename F::Acc d[F::NA] = {};
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t);
    wgmma_wait<W>();
  }
  wgmma_wait<0>();
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ threadIdx.x)
}

template <int X, int R>
DC_DEV void wg_spec(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  int d[128] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  if (mma_role) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    wgmma_hold(d);
  } else {
    OP_LOOP(it) { lane_ops<X, R>(l); }
  }
  OP_END
  OP_SINK((mma_role ? fold_acc(d) : l.fold()) ^ threadIdx.x)
}

// wg_spec exact on both sides: the two wgmma warpgroups fixed, each SMSP's lane warp running only while they do
// (OP_UNTIL), so neither rate counts time at the barrier. F: the wgmma form (a deeper kind's coupling).
template <int X, int R, class F = F_s8_n256>
DC_DEV void wg_until(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  typename F::Acc d[F::NA] = {};
  Lane l(threadIdx.x);
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
    OP_UNTIL(n) { lane_ops<X, R>(l); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((mma_role ? fold_acc(d) : l.fold()) ^ threadIdx.x)
}

// wg_until with a memory lane (lanes.cuh's mem_ops, M): the lane warps' R accesses a trip.
template <int M, int R>
DC_DEV void wg_until_mem(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  constexpr int SX = 32 * MEM_SX;  // 12 warps' scratch, and 64 LDS.128 lines
  __shared__ __align__(16) unsigned char sx[SX];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  int d[128] = {};
  Lane l(threadIdx.x);
  unsigned char* gp = mem_global(a.g);
  const uint32_t sb = mem_shared<SX>(sx);
  OP_BEGIN_STOP
  if (mma_role) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    OP_DONE
    wgmma_hold(d);
  } else {
    OP_UNTIL(n) { mem_ops<M, R, SX>(l, gp, sb, n); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((mma_role ? fold_acc(d) : l.fold()) ^ threadIdx.x)
}

// wg_mix with a memory lane: each warpgroup's 4 wgmma, then R accesses a warp, then the wait (the honest kernel's shape).
template <int M, int R>
DC_DEV void wg_mix_mem(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  constexpr int SX = 16 * MEM_SX;
  __shared__ __align__(16) unsigned char sx[SX];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  unsigned char* gp = mem_global(a.g);
  const uint32_t sb = mem_shared<SX>(sx);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F_s8_n256>(d, t);
    mem_ops<M, R, SX>(l, gp, sb, it);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

template <int X, int R, class F = F_s8_n256>
DC_DEV void wg_mix(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  typename F::Acc d[F::NA] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F>(d, t);
    lane_ops<X, R>(l);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}

// The honest LI-CP-W loop, static-trip (OP_LOOP, so a warp-collective redux gates on sm_90, unlike the dynamic-trip
// OP_UNTIL loop the gate refuses): each warpgroup issues its 4 wgmma, then folds the four segments' 512
// committed words with LI (LOP3 to absorb a word, IMAD to multiply by an odd a) and, for CP > 0, LI-CP-W's per-segment
// checkpoint (4 LOP3 to XOR a lane's 8 chains, then the warp reduces once by `redux.sync.xor.b32` (CP 1) or the 5-shuffle
// butterfly fallback (CP 2)), then waits. No filler (that is the design's). The fold reads the Lane's registers, not
// the wgmma's accumulators, so it overlaps its own wgmma, which the receipt can't: dc90-licpw-redux-l32 is timed from
// rt_hopper's wgli_db_w2, which folds the accumulators.
template <int CP>
DC_DEV void li_fold(Lane& l) {
#pragma unroll
  for (int c = 0; c < 256; c++) {
    l.op<L_LOP3>(c);
    l.op<L_IMAD>(c);
  }
  if constexpr (CP > 0) {
#pragma unroll
    for (int s = 0; s < 4; s++) {
      l.op<L_LOP3>(0);
      l.op<L_LOP3>(2);
      l.op<L_LOP3>(4);
      l.op<L_LOP3>(6);
      if constexpr (CP == 1) l.op<L_REDUX>(0);
      else l.op<L_REDUCE>(0);
    }
  }
}
template <int CP>
DC_DEV void wg_li(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F_s8_n256>(d, t);
    li_fold<CP>(l);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
}
// wgli_fill: wgli_redux with ffill's filler beside it (../ffill.cuh, dither-2s-licpw-redux-l32's [filler]): three steps a
// segment on three of its words, an IMAD and two FFMA a step, so 12 IMAD and 24 FFMA an iteration. Its tensor fraction
// over wgli_redux's is what the filler costs the GEMM. The words are the fold's own lane state (ptxas can't hoist them),
// the coins come from the block and thread, and `one` is 1 at run time (mode 0 or 1), so the IMAD stays an IMAD.
DC_DEV void wg_li_fill(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  uint32_t c[4];
  ff_coins(blockIdx.x, threadIdx.x, a.iters, c);
  FfChain ff(c, 1u | (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F_s8_n256>(d, t);
    li_fold<1>(l);
    ff.mma<0>(l.r[0], l.r[1], l.r[2]);
    ff.mma<1>(l.r[3], l.r[4], l.r[5]);
    ff.mma<0>(l.r[6], l.r[7], l.r[0]);
    ff.mma<1>(l.r[1], l.r[2], l.r[3]);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ __float_as_uint(ff.x) ^ __float_as_uint(ff.y) ^ threadIdx.x)
}

// TMA bulk-async copies beside the saturated wgmma: one side warp's lane 0 issues cp.async.bulk (SASS
// UBLKCP, a move) or cp.reduce.async.bulk.add.u32 (UBLKRED, a combine), 4 KB (1024 words) an op, DC_BULK_R a trip, into
// the block's own global region, with a group bound (wait_group.read 2) so the queue stays full. The two wgmma
// warpgroups run flat out (OP_LOOP), the bulk warp only while they do (OP_UNTIL), so neither rate counts barrier time. A
// TMA bulk op takes uniform operands (its addresses come from blockIdx and the op's slot, not the lane), so lane 0 issues
// one warp instruction an op; the measured tensor fraction says whether that steals the wgmma's issue (the facet
// question), and the bulk's rate is the TMA engine's words a SM-cycle. All ops read the same 4 KB of the shared tile; the
// source content does not matter for throughput.
enum { BULK_CP, BULK_RED };
constexpr int DC_BULK_R = 8;
template <int S>
DC_DEV void bulk_trip(const OpArgs& a, unsigned sbase) {
  constexpr unsigned B = 4096;
  unsigned char* region = a.g + (size_t)blockIdx.x * (DC_BULK_R * B);
#pragma unroll
  for (int q = 0; q < DC_BULK_R; q++) {
    unsigned char* p = region + q * B;
    if constexpr (S == BULK_CP)
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(p), "r"(sbase), "n"(B) : "memory");
    else
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(p), "r"(sbase),
                   "n"(B) : "memory");
  }
  asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 2;" ::: "memory");
}
template <int S>
DC_DEV void wg_bulk(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  int d[128] = {};
  OP_BEGIN_STOP
  if (mma_role) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    OP_DONE
    wgmma_hold(d);
  } else if (w == 8) {
    const unsigned sbase = smem_u32(sm);
    OP_UNTIL(n) {
      if (lane == 0) bulk_trip<S>(a, sbase);
    }
    OP_TRIPS(n)
    if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  OP_END_STOP
  OP_SINK((mma_role ? fold_acc(d) : threadIdx.x))
}
template <int S>
DC_DEV void bulk_solo(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  fill_tiles<0>(sm, a.mode);
  const unsigned sbase = smem_u32(sm);
  const int lane = threadIdx.x & 31;
  OP_BEGIN
  OP_LOOP(it) {
    if (lane == 0) bulk_trip<S>(a, sbase);
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  OP_END
  OP_SINK(threadIdx.x)
}

DC_DEV void mma_s8k32(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DC_DEV void mma_s8k16(int32_t* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}
// The other mma.sync shapes with 4 A, 2 B and 4 D registers a thread (TENSOR's M3: every mma.sync shape on sm_90).
#define SYNC_FORM(NAME, PTX, ACC, CON, TY)                                                                   \
  struct S_##NAME {                                                                                          \
    using Acc = ACC;                                                                                         \
    static constexpr int TYPE = TY;                                                                          \
    DC_DEV static void mma(Acc* d, const uint32_t* a, const uint32_t* b) {                                   \
      asm volatile("mma.sync.aligned." PTX " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"          \
                   : "+" CON(d[0]), "+" CON(d[1]), "+" CON(d[2]), "+" CON(d[3])                              \
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));                      \
    }                                                                                                        \
  };
SYNC_FORM(b1_k256, "m16n8k256.row.col.s32.b1.b1.s32.and.popc", int, "r", 0)
SYNC_FORM(e4m3_k32, "m16n8k32.row.col.f32.e4m3.e4m3.f32", float, "f", 1)
SYNC_FORM(f16_k16, "m16n8k16.row.col.f32.f16.f16.f32", float, "f", 4)
SYNC_FORM(bf16_k16, "m16n8k16.row.col.f32.bf16.bf16.f32", float, "f", 2)
SYNC_FORM(tf32_k8, "m16n8k8.row.col.f32.tf32.tf32.f32", float, "f", 3)
#undef SYNC_FORM
template <class S>
DC_DEV void sync_alone(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4], y[2];
#pragma unroll
  for (int i = 0; i < 4; i++) x[i] = operand_word<S::TYPE>(lane + 32 * i, a.mode);
#pragma unroll
  for (int i = 0; i < 2; i++) y[i] = operand_word<S::TYPE>(lane + 128 + 32 * i, a.mode);
  typename S::Acc d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) S::mma(d[c], x, y);
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int e = 0; e < 4; e++) s ^= *reinterpret_cast<const uint32_t*>(&d[c][e]);
  OP_SINK(s)
}

template <int K>
DC_DEV void imma_alone(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t x[4] = {dc_mix32(lane), dc_mix32(lane + 32), dc_mix32(lane + 64), dc_mix32(lane + 96)};
  uint32_t y[2] = {dc_mix32(lane + 128), dc_mix32(lane + 160)};
  int32_t d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      if constexpr (K == 32) mma_s8k32(d[c], x, y);
      else mma_s8k16(d[c], x, y);
    }
  }
  OP_END
  uint32_t s = lane;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)(d[c][0] ^ d[c][1] ^ d[c][2] ^ d[c][3]);
  OP_SINK(s)
}

// mma.sync beside wgmma, exact on both sides (as wg_until): the two wgmma warpgroups fixed, each SMSP's mma.sync warp
// until. (The other way round, the wgmma warps until, spills.)
template <int K>
DC_DEV void wg_sync_until(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  const uint32_t lane = threadIdx.x & 31;
  int d[128] = {};
  uint32_t x[4] = {dc_mix32(lane), dc_mix32(lane + 32), dc_mix32(lane + 64), dc_mix32(lane + 96)};
  uint32_t y[2] = {dc_mix32(lane + 128), dc_mix32(lane + 160)};
  int32_t e[8][4] = {};
  auto imma8 = [&] {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      if constexpr (K == 32) mma_s8k32(e[c], x, y);
      else mma_s8k16(e[c], x, y);
    }
  };
  OP_BEGIN_STOP
  if (mma_role) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    OP_DONE
    wgmma_hold(d);
  } else {
    OP_UNTIL(n) { imma8(); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  uint32_t s = lane;
  if (mma_role) {
    s ^= fold_acc(d);
  } else {
#pragma unroll
    for (int c = 0; c < 8; c++) s ^= (uint32_t)(e[c][0] ^ e[c][1] ^ e[c][2] ^ e[c][3]);
  }
  OP_SINK(s ^ threadIdx.x)
}

#define ALONE(NAME)                                                         \
  OP_KERNEL(wg_##NAME##_a0, 256) { wg_alone<F_##NAME, 0>(a); }               \
  OP_KERNEL(wg_##NAME##_a1, 256) { wg_alone<F_##NAME, 1>(a); }
ALONE(s8_n8)
ALONE(s8_n16)
ALONE(s8_n32)
ALONE(s8_n64)
ALONE(s8_n128)
ALONE(s8_n192)
ALONE(s8_n256)
ALONE(s8_n256_rs)
ALONE(u8s8_n256)
ALONE(e4m3_n128)
ALONE(e4m3_n256)
ALONE(e5m2_n256)
ALONE(f16_n256)
ALONE(bf16_n256)
ALONE(bf16_n128)
ALONE(tf32_n256)
ALONE(b1_n256)
#undef ALONE
#define JOINT(LN, X, R)                                                       \
  OP_KERNEL(wgspec_##LN##_r##R, 384) { wg_spec<X, R>(a); }                    \
  OP_KERNEL(wgmix_##LN##_r##R, 256) { wg_mix<X, R>(a); }
JOINT(mix, L_MIX, 64)
JOINT(mix, L_MIX, 256)
JOINT(mix, L_MIX, 512)
JOINT(imad, L_IMAD, 64)
JOINT(imad, L_IMAD, 256)
JOINT(lop3, L_LOP3, 64)
JOINT(lop3, L_LOP3, 256)
OP_KERNEL(wgmix_shfl_r32, 256) { wg_mix<L_SHFL, 32>(a); }
OP_KERNEL(wgmix_shfl_r128, 256) { wg_mix<L_SHFL, 128>(a); }
OP_KERNEL(wgmix_redux_r64, 256) { wg_mix<L_REDUX, 64>(a); }
OP_KERNEL(wgmix_reduce_r8, 256) { wg_mix<L_REDUCE, 8>(a); }
OP_KERNEL(wgli_base, 256) { wg_li<0>(a); }
OP_KERNEL(wgli_redux, 256) { wg_li<1>(a); }
OP_KERNEL(wgli_reduce, 256) { wg_li<2>(a); }
OP_KERNEL(wgli_fill, 256) { wg_li_fill(a); }
OP_KERNEL(wgbulk_cp, 384) { wg_bulk<BULK_CP>(a); }
OP_KERNEL(wgbulk_red, 384) { wg_bulk<BULK_RED>(a); }
OP_KERNEL(bulkcp, 32) { bulk_solo<BULK_CP>(a); }
OP_KERNEL(bulkred, 32) { bulk_solo<BULK_RED>(a); }
JOINT(duo, L_DUO, 96)
JOINT(duo, L_DUO, 384)
JOINT(imadw, L_IMADW, 64)
JOINT(imadw, L_IMADW, 256)
#undef JOINT
OP_KERNEL(wgU_imadw, 384) { wg_until<L_IMADW, 64>(a); }
OP_KERNEL(wgU_imadw_r256, 384) { wg_until<L_IMADW, 256>(a); }
#define MEM(LN, M)                                                   \
  OP_KERNEL(wgU_##LN, 384) { wg_until_mem<M, 16>(a); }               \
  OP_KERNEL(wgU_##LN##_r32, 384) { wg_until_mem<M, 32>(a); }         \
  OP_KERNEL(wgmix_##LN##_r16, 256) { wg_mix_mem<M, 16>(a); }
MEM(lde, M_LDE)
MEM(atome, M_ATOME)
MEM(redg, M_REDG)
MEM(atoms, M_ATOMS)
MEM(lds128, M_LDS128)
#undef MEM
OP_KERNEL(wgU_mix, 384) { wg_until<L_MIX, 64>(a); }
OP_KERNEL(wgU_imad, 384) { wg_until<L_IMAD, 64>(a); }
OP_KERNEL(wgU_lop3, 384) { wg_until<L_LOP3, 64>(a); }
OP_KERNEL(wgU_duo, 384) { wg_until<L_DUO, 48>(a); }
OP_KERNEL(wgU_li, 384) { wg_until<L_LI, 64>(a); }
OP_KERNEL(wgU_lif, 384) { wg_until<L_LIF, 64>(a); }
OP_KERNEL(wgU_li_r512, 384) { wg_until<L_LI, 512>(a); }
OP_KERNEL(wgU_mix_r512, 384) { wg_until<L_MIX, 512>(a); }
OP_KERNEL(wgU_lop3_r256, 384) { wg_until<L_LOP3, 256>(a); }
OP_KERNEL(wgU_lif_r256, 384) { wg_until<L_LIF, 256>(a); }
OP_KERNEL(wgU_lif_r512, 384) { wg_until<L_LIF, 512>(a); }
OP_KERNEL(wgUb1_lif_r256, 384) { wg_until<L_LIF, 256, F_b1_n256>(a); }
OP_KERNEL(wgUb1_lif_r512, 384) { wg_until<L_LIF, 512, F_b1_n256>(a); }
OP_KERNEL(wgUe4_lif_r512, 384) { wg_until<L_LIF, 512, F_e4m3_n256>(a); }
OP_KERNEL(wgmixb1_mix_r256, 256) { wg_mix<L_MIX, 256, F_b1_n256>(a); }
OP_KERNEL(wgmixb1_mix_r512, 256) { wg_mix<L_MIX, 512, F_b1_n256>(a); }
OP_KERNEL(wgmixe4_mix_r256, 256) { wg_mix<L_MIX, 256, F_e4m3_n256>(a); }
OP_KERNEL(wgmixe4_mix_r512, 256) { wg_mix<L_MIX, 512, F_e4m3_n256>(a); }
OP_KERNEL(wgU_sync_k16, 384) { wg_sync_until<16>(a); }
OP_KERNEL(wgU_sync_k32, 384) { wg_sync_until<32>(a); }
OP_KERNEL(imma_s8_k32, 512) { imma_alone<32>(a); }
OP_KERNEL(imma_s8_k16, 512) { imma_alone<16>(a); }
#define SYNC(NAME) OP_KERNEL(sync_##NAME, 512) { sync_alone<S_##NAME>(a); }
SYNC(b1_k256)
SYNC(e4m3_k32)
SYNC(f16_k16)
SYNC(bf16_k16)
SYNC(tf32_k8)
#undef SYNC

// SASS opcodes (CUDA 13.0, sm_90a): IGMMA for s8, QGMMA for fp8, HGMMA for bf16/tf32.
#define P_ALONE(NAME, PAT, ...) \
  {.name = "wg_" #NAME "_a0", .kernel = (const void*)wg_##NAME##_a0, .tpb = 256, .expect = PAT, .iters = 4096, ##__VA_ARGS__}, \
  {.name = "wg_" #NAME "_a1", .kernel = (const void*)wg_##NAME##_a1, .tpb = 256, .expect = PAT, .iters = 4096, ##__VA_ARGS__}
static const OpProbe PROBES[] = {
    P_ALONE(s8_n8, "IGMMA.64x8x32.S8.S8*=4"),
    P_ALONE(s8_n16, "IGMMA.64x16x32.S8.S8*=4"),
    P_ALONE(s8_n32, "IGMMA.64x32x32.S8.S8*=4"),
    P_ALONE(s8_n64, "IGMMA.64x64x32.S8.S8*=4"),
    P_ALONE(s8_n128, "IGMMA.64x128x32.S8.S8*=4"),
    P_ALONE(s8_n192, "IGMMA.64x192x32.S8.S8*=4"),
    P_ALONE(s8_n256, "IGMMA.64x256x32.S8.S8*=4"),
    P_ALONE(s8_n256_rs, "IGMMA.64x256x32.S8.S8*=4"),
    P_ALONE(u8s8_n256, "IGMMA.64x256x32.U8.S8*=4"),
    P_ALONE(e4m3_n128, "QGMMA.64x128x32.F32.E4M3.E4M3*=4"),
    P_ALONE(e4m3_n256, "QGMMA.64x256x32.F32.E4M3.E4M3*=4"),
    P_ALONE(e5m2_n256, "QGMMA.64x256x32.F32.E5M2.E5M2*=4"),
    P_ALONE(f16_n256, "HGMMA.64x256x16.F32*=4"),
    P_ALONE(bf16_n256, "HGMMA.64x256x16.F32.BF16*=4"),
    P_ALONE(bf16_n128, "HGMMA.64x128x16.F32.BF16*=4"),
    P_ALONE(tf32_n256, "HGMMA.64x256x8.F32.TF32*=4"),
    P_ALONE(b1_n256, "BGMMA.64x256x256.AND.POPC*=4"),
    {.name = "wgz_s8_n256_a1", .kernel = (const void*)wg_s8_n256_a1, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4", .mode = 1, .iters = 4096, .fn = "wg_s8_n256_a1"},
    {.name = "wgz_e4m3_n256_a1", .kernel = (const void*)wg_e4m3_n256_a1, .tpb = 256,
     .expect = "QGMMA.64x256x32.F32.E4M3.E4M3*=4", .mode = 1, .iters = 4096, .fn = "wg_e4m3_n256_a1"},
#define P_JOINT(LN, R, LPAT)                                                                                     \
  {.name = "wgspec_" #LN "_r" #R, .kernel = (const void*)wgspec_##LN##_r##R, .tpb = 384,                          \
   .expect = "IGMMA.64x256x32.S8.S8*=4;" LPAT, .warps = "8;4", .iters = 4096},                                    \
  {.name = "wgmix_" #LN "_r" #R, .kernel = (const void*)wgmix_##LN##_r##R, .tpb = 256,                            \
   .expect = "IGMMA.64x256x32.S8.S8*=4," LPAT, .iters = 4096}
    P_JOINT(mix, 64, "I*ADD*=32,FFMA=32"),
    P_JOINT(mix, 256, "I*ADD*=128,FFMA=128"),
    P_JOINT(mix, 512, "I*ADD*=256,FFMA=256"),
    P_JOINT(imad, 64, "IMAD=64"),
    P_JOINT(imad, 256, "IMAD=256"),
    P_JOINT(lop3, 64, "LOP3.LUT=64"),
    P_JOINT(lop3, 256, "LOP3.LUT=256"),
    {.name = "wgmix_shfl_r32", .kernel = (const void*)wgmix_shfl_r32, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,SHFL.BFLY=32", .iters = 4096},
    {.name = "wgmix_shfl_r128", .kernel = (const void*)wgmix_shfl_r128, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,SHFL.BFLY=128", .iters = 4096},
    {.name = "wgmix_redux_r64", .kernel = (const void*)wgmix_redux_r64, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,REDUX*=64", .iters = 4096},
    {.name = "wgmix_reduce_r8", .kernel = (const void*)wgmix_reduce_r8, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,SHFL.BFLY=40,LOP3.LUT=40", .iters = 4096},
    {.name = "wgli_base", .kernel = (const void*)wgli_base, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,LOP3.LUT=256,IMAD=256", .iters = 4096},
    {.name = "wgli_redux", .kernel = (const void*)wgli_redux, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,LOP3.LUT=272,IMAD=256,REDUX*=4", .iters = 4096},
    {.name = "wgli_reduce", .kernel = (const void*)wgli_reduce, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,LOP3.LUT=292,IMAD=256,SHFL.BFLY=20", .iters = 4096},
    {.name = "wgli_fill", .kernel = (const void*)wgli_fill, .tpb = 256,
     .expect = "IGMMA.64x256x32.S8.S8*=4,LOP3.LUT=272,IMAD*=268,FFMA=24,REDUX*=4", .iters = 4096},
    P_JOINT(duo, 96, "LOP3.LUT=32,IMAD=32"),
    P_JOINT(duo, 384, "LOP3.LUT=128,IMAD=128"),
    P_JOINT(imadw, 64, "IMAD.WIDE.U32=64"),
    P_JOINT(imadw, 256, "IMAD.WIDE.U32=256"),
#undef P_JOINT
#define WGI "IGMMA.64x256x32.S8.S8*=4"
#define P_UNTIL(NAME, LPAT) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 384, .expect = WGI ";" LPAT, .warps = "8;4", .iters = 4096, .until = 2}
    P_UNTIL(wgU_mix, "I*ADD*=32,FFMA=32"),
    P_UNTIL(wgU_imad, "IMAD=64"),
    P_UNTIL(wgU_lop3, "LOP3.LUT=64"),
    P_UNTIL(wgU_duo, "LOP3.LUT=16,IMAD=16"),
    P_UNTIL(wgU_li, "LOP3.LUT=32,IMAD=32"),
    P_UNTIL(wgU_lif, "LOP3.LUT=28,IMAD=28,FFMA=8"),
    P_UNTIL(wgU_li_r512, "LOP3.LUT=256,IMAD=256"),
    P_UNTIL(wgU_mix_r512, "I*ADD*=256,FFMA=256"),
    P_UNTIL(wgU_lop3_r256, "LOP3.LUT=256"),
    P_UNTIL(wgU_sync_k16, "IMMA.16816.S8.S8=8"),
    P_UNTIL(wgU_sync_k32, "IMMA.16832.S8.S8=8"),
    P_UNTIL(wgU_lif_r256, "LOP3.LUT=112,IMAD=112,FFMA=32"),
    P_UNTIL(wgU_lif_r512, "LOP3.LUT=224,IMAD=224,FFMA=64"),
    P_UNTIL(wgU_imadw, "IMAD.WIDE.U32=64"),
    P_UNTIL(wgU_imadw_r256, "IMAD.WIDE.U32=256"),
#define P_MEM(LN, PAT)                                                                                    \
  P_UNTIL(wgU_##LN, PAT "=16"), P_UNTIL(wgU_##LN##_r32, PAT "=32"),                                       \
      {.name = "wgmix_" #LN "_r16", .kernel = (const void*)wgmix_##LN##_r16, .tpb = 256, .expect = WGI "," PAT "=16", \
       .iters = 4096}
    P_MEM(lde, "LD.E*"),
    P_MEM(atome, "ATOM.E*"),
    P_MEM(redg, "REDG*"),
    P_MEM(atoms, "ATOMS*"),
    P_MEM(lds128, "LDS.128"),
#undef P_MEM
#undef P_UNTIL
#define P_UNTILF(NAME, WPAT, LPAT) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 384, .expect = WPAT ";" LPAT, .warps = "8;4", .iters = 4096, .until = 2}
    P_UNTILF(wgUb1_lif_r256, "BGMMA.64x256x256.AND.POPC*=4", "LOP3.LUT=112,IMAD=112,FFMA=32"),
    P_UNTILF(wgUb1_lif_r512, "BGMMA.64x256x256.AND.POPC*=4", "LOP3.LUT=224,IMAD=224,FFMA=64"),
    P_UNTILF(wgUe4_lif_r512, "QGMMA.64x256x32.F32.E4M3.E4M3*=4", "LOP3.LUT=224,IMAD=224,FFMA=64"),
#undef P_UNTILF
#define P_MIXF(NAME, WPAT, LPAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 256, .expect = WPAT "," LPAT, .iters = 4096}
    P_MIXF(wgmixb1_mix_r256, "BGMMA.64x256x256.AND.POPC*=4", "I*ADD*=128,FFMA=128"),
    P_MIXF(wgmixb1_mix_r512, "BGMMA.64x256x256.AND.POPC*=4", "I*ADD*=256,FFMA=256"),
    P_MIXF(wgmixe4_mix_r256, "QGMMA.64x256x32.F32.E4M3.E4M3*=4", "I*ADD*=128,FFMA=128"),
    P_MIXF(wgmixe4_mix_r512, "QGMMA.64x256x32.F32.E4M3.E4M3*=4", "I*ADD*=256,FFMA=256"),
#undef P_MIXF
#undef WGI
    {.name = "wgbulk_cp", .kernel = (const void*)wgbulk_cp, .tpb = 384,
     .expect = "IGMMA.64x256x32.S8.S8*=4;UBLKCP*=8", .counted = "UBLKCP*", .warps = "8;1", .words = 1024,
     .iters = 4096, .until = 2},
    {.name = "wgbulk_red", .kernel = (const void*)wgbulk_red, .tpb = 384,
     .expect = "IGMMA.64x256x32.S8.S8*=4;UBLKRED*=8", .counted = "UBLKRED*", .warps = "8;1", .words = 1024,
     .iters = 4096, .until = 2},
    {.name = "bulkcp", .kernel = (const void*)bulkcp, .tpb = 32, .expect = "UBLKCP*=8", .words = 1024, .iters = 1024},
    {.name = "bulkred", .kernel = (const void*)bulkred, .tpb = 32, .expect = "UBLKRED*=8", .words = 1024,
     .iters = 1024},
    {.name = "imma_s8_k32", .kernel = (const void*)imma_s8_k32, .tpb = 512, .expect = "IMMA.16832.S8.S8=8", .iters = 4096},
    {.name = "imma_s8_k16", .kernel = (const void*)imma_s8_k16, .tpb = 512, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
#define P_SYNC(NAME, PAT) \
  {.name = "sync_" #NAME, .kernel = (const void*)sync_##NAME, .tpb = 512, .expect = PAT, .iters = 4096}
    P_SYNC(b1_k256, "BMMA.168256.AND.POPC=8"),
    P_SYNC(e4m3_k32, "HMMA.16816.F32=16,FADD=32"),
    P_SYNC(f16_k16, "HMMA.16816.F32=8"),
    P_SYNC(bf16_k16, "HMMA.16816.F32.BF16=8"),
    P_SYNC(tf32_k8, "HMMA.1688.F32.TF32=8"),
#undef P_SYNC
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
