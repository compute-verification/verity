// EPS1 DC on datacenter Blackwell (sm_100a, B200): TENSOR's M4 (note:20261017T0321Z-draft-eps1-tensor §5) as opcode probes.
// Build with ../opcode_probe/probe.py build --arch sm_100a; gated and poisoned as every opcode probe (probe.cuh).
// tcgen05.mma (cta_group::1) is issued by one thread into tensor memory (TMEM, 512 columns allocated a CTA) from one
// shared-memory tile (no-swizzle K-major descriptors, dc.cuh); its completion is one mbarrier phase after the loop.
//   tc_<kind>_m<M>_n<N>   4 tcgen05.mma an iteration into two accumulators (M4a: slots written a tensor-pipe cycle;
//                         cta_group::2 needs a cluster pair and is not probed yet)
//   ld_<shape>            8 warps, 4 groups an iteration of 4 tcgen05.ld (x32: 2) and a tcgen05.wait::ld (M4b alone):
//                         32x32b.x{1,4,16,32,64,128}, 16x64b.x16, 16x128b.x8, 16x256b.x4; ld16w_*: 16 warps
//   ldmma_w<LW>[_r<R>]    warp 0: tcgen05.mma i8 128x256x32 into columns 0-255; LW warps: R tcgen05.ld.32x32b.x32 of
//                         columns 256-511 an iteration (M4b joint facet: 2 LW R words a clock asked; block-timed, so a
//                         saturated point's faster role is a lower bound)
//   ldU_w<LW>, mmaU_ld8   the same roles, exact on both sides: MMA fixed and LW loaders running only while it does
//                         (OP_UNTIL), or 8 loaders fixed and the MMA until
//   syncU_k*, mmaU_sync_k32  syncmma's roles, exact on both sides in the same way
//   imma_s8_k32 / _k16    mma.sync alone on sm_100 (M4c)
//   syncmma_k32 / _k16    thread 0 tcgen05.mma beside 8 warps of mma.sync (M4c: does mma.sync share the tcgen05 pipe?)
//   tclane_<lane>_r<R>    thread 0 tcgen05.mma beside 12 warps of lane ops (the scalar pipes beside a saturated MMA)
//   tclaneU_<lane>        the same roles, exact on both sides (the lanes OP_UNTIL): a tclane_ point's tensor rate is
//                         the MMA's fixed work over the lanes' longer block time, a lower bound, not a joint's edge
// dc/analyze.py turns rows.jsonl into words a cycle and the joint facets.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "lanes.cuh"

constexpr int KB = 32;  // 32 bytes of K a row: i8/e4m3 K = 32, bf16 K = 16
constexpr int AROWS = 128, BROWS = 256, TMEM_COLS = 512;

struct Smem {
  unsigned char tile[(AROWS + BROWS) * KB];
  uint64_t bar;
  uint32_t tmem;
};

// The warp index as ptxas sees a uniform value (CUTLASS's canonical_warp_idx_sync): no divergence fallback paths.
DC_DEV uint32_t warp_u() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }

// Every thread: fill the operand tile with random codes valid for the type, init the barrier, allocate TMEM (warp 0).
template <int TY>
DC_DEV uint32_t setup(Smem& s, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(s.tile);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) {
    const uint32_t x = mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
    w[i] = TY == 0 ? x : TY == 1 ? (x & 0x77777777u) : ((x & 0x807F807Fu) | 0x3F003F00u);
  }
  if (threadIdx.x == 0) {
    mbar_init(&s.bar, 1);
    fence_mbar_init();
  }
  if (threadIdx.x < 32) tmem_alloc(&s.tmem, TMEM_COLS);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  return s.tmem;
}
DC_DEV void teardown(uint32_t tmem) {
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(tmem, TMEM_COLS);
}
DC_DEV uint64_t desc_a(Smem& s) { return umma_desc(smem_u32(s.tile), kmaj_lbo(), kmaj_sbo(KB)); }
DC_DEV uint64_t desc_b(Smem& s) { return umma_desc(smem_u32(s.tile) + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB)); }

// The sink: each warp's lane quarter, 16 columns of D0 (any thread can read its own row after the MMAs completed).
DC_DEV uint32_t sink_ld(uint32_t tmem, uint32_t col) {
  uint32_t r[16];
  tmem_ld16(tmem + ((32u * ((threadIdx.x >> 5) & 3u)) << 16) + col, r);
  tmem_wait_ld();
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) x ^= r[i];
  return x;
}

// KIND 0 i8 (s8 x s8 -> s32), 1 f8f6f4 (e4m3 -> f32), 2 f16 (bf16 -> f32)
template <int KIND>
DC_DEV void mma_k(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
  if constexpr (KIND == 0) umma_i8(d, da, db, idesc, acc);
  else if constexpr (KIND == 1) umma_f8f6f4(d, da, db, idesc, acc);
  else umma_f16(d, da, db, idesc, acc);
}
template <int KIND>
DC_HD constexpr uint32_t idesc_of(int m, int n) {
  return KIND == 0 ? umma_idesc(2, 1, 1, m, n) : KIND == 1 ? umma_idesc(1, 0, 0, m, n) : umma_idesc(1, 1, 1, m, n);
}
// Four MMAs an iteration: D0 and D1 overwritten, then accumulated (FP sums stay finite).
template <int KIND, int M, int N>
DC_DEV void mma4(uint32_t tmem, uint64_t da, uint64_t db) {
  constexpr uint32_t idesc = idesc_of<KIND>(M, N), d1 = N <= 256 ? 256u : 0u;
  mma_k<KIND>(tmem, da, db, idesc, 0);
  mma_k<KIND>(tmem + d1, da, db, idesc, 0);
  mma_k<KIND>(tmem, da, db, idesc, 1);
  mma_k<KIND>(tmem + d1, da, db, idesc, 1);
}

template <int KIND, int M, int N>
DC_DEV void tc_alone(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<KIND>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<KIND, M, N>(tmem, da, db);
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem, 0);
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

// WARPS warps; warp w reads its lane quarter (w % 4) in a span of 2048 / WARPS columns (256 at 8 warps, 128 at 16). The
// 16-lane shapes' x16/x8/x4 read 32 columns each, so they step by 32 within the span (c32), never past the 512 allocated.
template <int SHAPE, int WARPS = 8>
DC_DEV void ld_alone(const OpArgs& a) {
  constexpr uint32_t SPAN = 2048 / WARPS;
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint32_t w = warp_u(), base = tmem + ((32u * (w & 3u)) << 16) + (w >> 2) * SPAN;
  auto c32 = [&](int g, int j) { return base + ((128u * g + 32u * j) & (SPAN - 1)); };
  auto col = [&](uint32_t c) { return base + (c & (SPAN - 1)); };
  uint32_t r0[32], r1[32];
  uint32_t q0[16], q1[16], q2[16], q3[16];
  uint32_t v0[4], v1[4], v2[4], v3[4];
  uint32_t h[64], z[128];
  uint32_t acc = 0;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int g = 0; g < 4; g++) {
      if constexpr (SHAPE == 0) {
        tmem_ld32(col(64 * g), r0); tmem_ld32(col(64 * g + 32), r1);
        tmem_wait_ld();
        acc += r0[0] ^ r1[1];
      } else if constexpr (SHAPE == 1) {
        tmem_ld16(col(64 * g), q0); tmem_ld16(col(64 * g + 16), q1); tmem_ld16(col(64 * g + 32), q2);
        tmem_ld16(col(64 * g + 48), q3);
        tmem_wait_ld();
        acc += q0[0] ^ q1[1] ^ q2[2] ^ q3[3];
      } else if constexpr (SHAPE == 2) {
        tmem_ld16x256(c32(g, 0), q0); tmem_ld16x256(c32(g, 1), q1); tmem_ld16x256(c32(g, 2), q2);
        tmem_ld16x256(c32(g, 3), q3);
        tmem_wait_ld();
        acc += q0[0] ^ q1[1] ^ q2[2] ^ q3[3];
      } else if constexpr (SHAPE == 3) {
        tmem_ld16x64(c32(g, 0), q0); tmem_ld16x64(c32(g, 1), q1); tmem_ld16x64(c32(g, 2), q2);
        tmem_ld16x64(c32(g, 3), q3);
        tmem_wait_ld();
        acc += q0[0] ^ q1[1] ^ q2[2] ^ q3[3];
      } else if constexpr (SHAPE == 4) {
        tmem_ld16x128(c32(g, 0), q0); tmem_ld16x128(c32(g, 1), q1); tmem_ld16x128(c32(g, 2), q2);
        tmem_ld16x128(c32(g, 3), q3);
        tmem_wait_ld();
        acc += q0[0] ^ q1[1] ^ q2[2] ^ q3[3];
      } else if constexpr (SHAPE == 5) {
        tmem_ld4(col(64 * g), v0); tmem_ld4(col(64 * g + 16), v1); tmem_ld4(col(64 * g + 32), v2);
        tmem_ld4(col(64 * g + 48), v3);
        tmem_wait_ld();
        acc += v0[0] ^ v1[1] ^ v2[2] ^ v3[3];
      } else if constexpr (SHAPE == 6) {
        tmem_ld1(col(64 * g), v0[0]); tmem_ld1(col(64 * g + 16), v1[0]); tmem_ld1(col(64 * g + 32), v2[0]);
        tmem_ld1(col(64 * g + 48), v3[0]);
        tmem_wait_ld();
        acc += v0[0] ^ v1[0] ^ v2[0] ^ v3[0];
      } else if constexpr (SHAPE == 7) {
        tmem_ld64(col(64 * g), h);  // one in flight: two (128 registers) spill
        tmem_wait_ld();
        acc += h[0] ^ h[1];
      } else {
        tmem_ld128(col(128 * g), z);
        tmem_wait_ld();
        acc += z[0] ^ z[1];
      }
    }
  }
  OP_END
  // acc alone is the sink: reading every register of the last loads after the loop keeps them live across the back
  // edge, and ptxas then holds a copy for each unrolled step (235 registers at 8 warps, spills at 16).
  const uint32_t x = threadIdx.x + acc;
  teardown(tmem);
  OP_SINK(x)
}

// warp 0: tcgen05.mma i8 128x256x32 into D0 (columns 0-255); warps 4 .. 4 + LW - 1: R loads of 32 columns of 256-511 an
// iteration (warps of one lane quarter offset by 64 columns). An iteration's MMAs take 512 cycles at 256 slots a clock,
// so the loaders ask 2 LW R words a clock. probe.py times every loop over the block, so a loader that asks less than it
// can load reads as what it asked (with the tensor at 1): only a point that asks more than the MMA writes bounds ρ_ld.
template <int LW, int R = 8>
DC_DEV void ld_beside_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  const uint32_t w = warp_u(), sub = (w - 4u) >> 2, base = tmem + ((32u * (w & 3u)) << 16) + 256u;
  uint32_t r0[32], r1[32];
  uint32_t x = threadIdx.x;
  OP_BEGIN
  if (w == 0) {
    OP_LOOP(it) {
      if (elect_one()) {
        umma_i8(tmem, da, db, idesc, 0);
        umma_i8(tmem, da, db, idesc, 1);
        umma_i8(tmem, da, db, idesc, 1);
        umma_i8(tmem, da, db, idesc, 1);
      }
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  } else if (w >= 4 && w < 4 + LW) {
    OP_LOOP(it) {
#pragma unroll
      for (int j = 0; j < R / 2; j++) {
        tmem_ld32(base + (64u * (j + sub)) % 256u, r0);
        tmem_ld32(base + (64u * (j + sub) + 32u) % 256u, r1);
        tmem_wait_ld();
        x += r0[0] ^ r1[1];
      }
    }
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  x ^= sink_ld(tmem, 0);
  teardown(tmem);
  OP_SINK(x)
}

// ld_beside_mma's roles with exact rates on both sides: the fixed role runs a.iters (OP_LOOP), the other runs only while
// it does (OP_UNTIL), flat out, so neither rate includes time it sat at the barrier. MMA_FIXED: the MMA warp fixed and LW
// loaders until (the most a loader gets while the MMA runs at whatever rate it keeps); else the loaders fixed, MMA until.
template <int LW, bool MMA_FIXED>
DC_DEV void ld_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  const uint32_t w = warp_u(), sub = (w - 4u) >> 2, base = tmem + ((32u * (w & 3u)) << 16) + 256u;
  const bool loader = w >= 4 && w < 4 + LW;
  uint32_t r0[32], r1[32];
  uint32_t x = threadIdx.x;
  auto mma = [&] {
    if (elect_one()) {
      umma_i8(tmem, da, db, idesc, 0);
      umma_i8(tmem, da, db, idesc, 1);
      umma_i8(tmem, da, db, idesc, 1);
      umma_i8(tmem, da, db, idesc, 1);
    }
    __syncwarp();
  };
  auto loads = [&] {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      tmem_ld32(base + (64u * (j + sub)) % 256u, r0);
      tmem_ld32(base + (64u * (j + sub) + 32u) % 256u, r1);
      tmem_wait_ld();
      x += r0[0] ^ r1[1];
    }
  };
  OP_BEGIN_STOP
  if constexpr (MMA_FIXED) {
    if (w == 0) {
      OP_LOOP(it) { mma(); }
      OP_DONE
    } else if (loader) {
      OP_UNTIL(n) { loads(); }
      OP_TRIPS(n)
    }
  } else {
    if (loader) {
      OP_LOOP(it) { loads(); }
      OP_DONE
    } else if (w == 0) {
      OP_UNTIL(n) { mma(); }
      OP_TRIPS(n)
    }
  }
  OP_END_STOP
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  x ^= sink_ld(tmem, 0);
  teardown(tmem);
  OP_SINK(x)
}

// The honest LI loop beside the saturated MMA, exact by OP_UNTIL (as ld_until_mma): warp 0 fixed on tcgen05.mma, LW
// loader warps reading the accumulator from TMEM and folding it with LI (LOP3 to absorb a word, IMAD to multiply by an
// odd a), a segment of 128 words a trip (4 tcgen05.ld.32x32b.x32). CP 0 is the bare fold (LI); CP 1 adds LI-CP-W's
// checkpoint with `redux.sync.xor.b32`, CP 2 with the 5-shuffle fallback: each lane XORs its 8 chains (4 LOP3) and the
// warp reduces once a segment. The three throughputs' ratio is what the checkpoint costs the latency-bound loop, which
// the honest profile scales dc100-li-l32's measured cycles by (note:LI-ALT asked this of DC).
template <int CP, int LW>
DC_DEV void tcli_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  const uint32_t w = warp_u(), sub = (w - 4u) >> 2, base = tmem + ((32u * (w & 3u)) << 16) + 256u;
  const bool loader = w >= 4 && w < 4 + LW;
  Lane l(threadIdx.x);
  auto mma = [&] {
    if (elect_one()) {
      umma_i8(tmem, da, db, idesc, 0);
      umma_i8(tmem, da, db, idesc, 1);
      umma_i8(tmem, da, db, idesc, 1);
      umma_i8(tmem, da, db, idesc, 1);
    }
    __syncwarp();
  };
  auto seg = [&] {
    uint32_t r0[32], r1[32];
#pragma unroll
    for (int g = 0; g < 2; g++) {
      tmem_ld32(base + (64u * (g + sub)) % 256u, r0);
      tmem_ld32(base + (64u * (g + sub) + 32u) % 256u, r1);
      tmem_wait_ld();
#pragma unroll
      for (int c = 0; c < 32; c++) {
        asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[c & 7]) : "r"(r0[c]), "r"(0x85EBCA6Bu));
        asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[c & 7]) : "r"(0x9E3779B9u), "r"(r1[c]));
      }
    }
    if constexpr (CP > 0) {
      l.op<L_LOP3>(0); l.op<L_LOP3>(2); l.op<L_LOP3>(4); l.op<L_LOP3>(6);
      if constexpr (CP == 1) l.op<L_REDUX>(0);
      else l.op<L_REDUCE>(0);
    }
  };
  OP_BEGIN_STOP
  if (w == 0) {
    OP_LOOP(it) { mma(); }
    OP_DONE
  } else if (loader) {
    OP_UNTIL(n) { seg(); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
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
template <int K>
DC_DEV void imma8(int32_t (&d)[8][4], const uint32_t* x, const uint32_t* y) {
#pragma unroll
  for (int c = 0; c < 8; c++) {
    if constexpr (K == 32) mma_s8k32(d[c], x, y);
    else mma_s8k16(d[c], x, y);
  }
}
DC_DEV void imma_ops(uint32_t lane, uint32_t (&x)[4], uint32_t (&y)[2]) {
#pragma unroll
  for (int i = 0; i < 4; i++) x[i] = dc_mix32(lane + 32 * i);
  y[0] = dc_mix32(lane + 128);
  y[1] = dc_mix32(lane + 160);
}
DC_DEV uint32_t fold8(const int32_t (&d)[8][4]) {
  uint32_t s = 0;
#pragma unroll
  for (int c = 0; c < 8; c++) s ^= (uint32_t)(d[c][0] ^ d[c][1] ^ d[c][2] ^ d[c][3]);
  return s;
}
template <int K>
DC_DEV void imma_alone(const OpArgs& a) {
  uint32_t x[4], y[2];
  imma_ops(threadIdx.x & 31, x, y);
  int32_t d[8][4] = {};
  OP_BEGIN
  OP_LOOP(it) { imma8<K>(d, x, y); }
  OP_END
  OP_SINK(fold8(d) ^ threadIdx.x)
}
// thread 0: tcgen05.mma i8 128x256x32; warps 8-15 (2 a SMSP): 8 mma.sync an iteration.
template <int K>
DC_DEV void sync_beside_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  uint32_t x[4], y[2];
  imma_ops(threadIdx.x & 31, x, y);
  int32_t d[8][4] = {};
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<0, 128, 256>(tmem, da, db);
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  } else if (warp_u() >= 8) {
    OP_LOOP(it) { imma8<K>(d, x, y); }
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t v = sink_ld(tmem, 0) ^ fold8(d);
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}
// sync_beside_mma with exact rates on both sides (as ld_until_mma): MMA_FIXED, the tcgen05 warp fixed and the mma.sync
// warps until; else the mma.sync warps fixed and the tcgen05 warp until.
template <int K, bool MMA_FIXED>
DC_DEV void sync_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  uint32_t x[4], y[2];
  imma_ops(threadIdx.x & 31, x, y);
  int32_t d[8][4] = {};
  const uint32_t w = warp_u();
  auto mma = [&] {
    if (elect_one()) mma4<0, 128, 256>(tmem, da, db);
    __syncwarp();
  };
  OP_BEGIN_STOP
  if constexpr (MMA_FIXED) {
    if (w == 0) {
      OP_LOOP(it) { mma(); }
      OP_DONE
    } else if (w >= 8) {
      OP_UNTIL(n) { imma8<K>(d, x, y); }
      OP_TRIPS(n)
    }
  } else {
    if (w >= 8) {
      OP_LOOP(it) { imma8<K>(d, x, y); }
      OP_DONE
    } else if (w == 0) {
      OP_UNTIL(n) { mma(); }
      OP_TRIPS(n)
    }
  }
  OP_END_STOP
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t v = sink_ld(tmem, 0) ^ fold8(d);
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}

// ---- lane ops beside tcgen05.mma ---------------------------------------------------------------------------------------
// thread 0: tcgen05.mma; warps 4-15 (3 a SMSP): R lane ops an iteration.
template <int X, int R>
DC_DEV void lane_beside_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  Lane l(threadIdx.x);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<0, 128, 256>(tmem, da, db);
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  } else if (warp_u() >= 4) {
    OP_LOOP(it) { lane_ops<X, R>(l); }
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t v = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}
// TMA bulk-async copies beside the saturated tcgen05.mma (DC EPS1 r10): one side warp's lane 0 issues cp.async.bulk (SASS
// UBLKCP, a move) or cp.reduce.async.bulk.add.u32 (UBLKRED, a combine), 4 KB (1024 words) an op, DC_BULK_R a trip, into
// the block's own global region, with a group bound so the queue stays full. Warp 0 runs tcgen05.mma flat out (OP_LOOP),
// the bulk warp only while it does (OP_UNTIL). A TMA bulk op takes uniform operands (addresses from blockIdx and the op's
// slot, not the lane), so lane 0 issues one warp instruction an op; the measured tensor fraction says whether that steals
// the MMA's issue (the facet question), and the bulk's rate is the TMA engine's words a SM-clock.
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
DC_DEV void bulk_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  const unsigned sbase = smem_u32(s.tile);
  const int lane = threadIdx.x & 31;
  OP_BEGIN_STOP
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<0, 128, 256>(tmem, da, db);
      __syncwarp();
    }
    OP_DONE
  } else if (warp_u() == 4) {
    OP_UNTIL(n) {
      if (lane == 0) bulk_trip<S>(a, sbase);
    }
    OP_TRIPS(n)
    if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  OP_END_STOP
  if (warp_u() == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t v = sink_ld(tmem, 0);
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}
template <int S>
DC_DEV void bulk_solo_tc(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[4096];
  for (uint32_t i = threadIdx.x; i < 1024; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = i;
  __syncthreads();
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

// lane_beside_mma exact on both sides: the MMA warp fixed, the lane warps running only while it does (OP_UNTIL).
template <int X, int R>
DC_DEV void lane_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  Lane l(threadIdx.x);
  OP_BEGIN_STOP
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<0, 128, 256>(tmem, da, db);
      __syncwarp();
    }
    OP_DONE
  } else if (warp_u() >= 4) {
    OP_UNTIL(n) { lane_ops<X, R>(l); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  if (warp_u() == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t v = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}

OP_KERNEL(tc_i8_m128_n32, 128) { tc_alone<0, 128, 32>(a); }
OP_KERNEL(tc_i8_m128_n64, 128) { tc_alone<0, 128, 64>(a); }
OP_KERNEL(tc_i8_m128_n128, 128) { tc_alone<0, 128, 128>(a); }
OP_KERNEL(tc_i8_m128_n256, 128) { tc_alone<0, 128, 256>(a); }
OP_KERNEL(tc_i8_m64_n64, 128) { tc_alone<0, 64, 64>(a); }
OP_KERNEL(tc_i8_m64_n128, 128) { tc_alone<0, 64, 128>(a); }
OP_KERNEL(tc_i8_m64_n256, 128) { tc_alone<0, 64, 256>(a); }
OP_KERNEL(tc_f8_m128_n128, 128) { tc_alone<1, 128, 128>(a); }
OP_KERNEL(tc_f8_m128_n256, 128) { tc_alone<1, 128, 256>(a); }
OP_KERNEL(tc_f16_m128_n256, 128) { tc_alone<2, 128, 256>(a); }
OP_KERNEL(ld_32x32b_x32, 256) { ld_alone<0>(a); }
OP_KERNEL(ld_32x32b_x16, 256) { ld_alone<1>(a); }
OP_KERNEL(ld_16x256b_x4, 256) { ld_alone<2>(a); }
OP_KERNEL(ld_16x64b_x16, 256) { ld_alone<3>(a); }
OP_KERNEL(ld_16x128b_x8, 256) { ld_alone<4>(a); }
OP_KERNEL(ld_32x32b_x4, 256) { ld_alone<5>(a); }
OP_KERNEL(ld_32x32b_x1, 256) { ld_alone<6>(a); }
OP_KERNEL(ld16w_32x32b_x32, 512) { ld_alone<0, 16>(a); }
OP_KERNEL(ld16w_32x32b_x16, 512) { ld_alone<1, 16>(a); }
OP_KERNEL(ld_32x32b_x64, 256) { ld_alone<7>(a); }
OP_KERNEL(ld_32x32b_x128, 256) { ld_alone<8>(a); }
OP_KERNEL(ldU_w4, 256) { ld_until_mma<4, true>(a); }
OP_KERNEL(ldU_w8, 384) { ld_until_mma<8, true>(a); }
OP_KERNEL(ldU_w12, 512) { ld_until_mma<12, true>(a); }
OP_KERNEL(mmaU_ld8, 384) { ld_until_mma<8, false>(a); }
OP_KERNEL(syncU_k32, 512) { sync_until_mma<32, true>(a); }
OP_KERNEL(syncU_k16, 512) { sync_until_mma<16, true>(a); }
OP_KERNEL(mmaU_sync_k32, 512) { sync_until_mma<32, false>(a); }
OP_KERNEL(ldmma_w4, 256) { ld_beside_mma<4>(a); }
OP_KERNEL(ldmma_w8, 384) { ld_beside_mma<8>(a); }
OP_KERNEL(ldmma_w12, 512) { ld_beside_mma<12>(a); }
OP_KERNEL(ldmma_w8_r16, 384) { ld_beside_mma<8, 16>(a); }
OP_KERNEL(ldmma_w12_r16, 512) { ld_beside_mma<12, 16>(a); }
OP_KERNEL(imma_s8_k32, 512) { imma_alone<32>(a); }
OP_KERNEL(imma_s8_k16, 512) { imma_alone<16>(a); }
OP_KERNEL(syncmma_k32, 512) { sync_beside_mma<32>(a); }
OP_KERNEL(syncmma_k16, 512) { sync_beside_mma<16>(a); }
OP_KERNEL(tclane_mix_r64, 512) { lane_beside_mma<L_MIX, 64>(a); }
OP_KERNEL(tclane_mix_r256, 512) { lane_beside_mma<L_MIX, 256>(a); }
OP_KERNEL(tclane_imad_r128, 512) { lane_beside_mma<L_IMAD, 128>(a); }
OP_KERNEL(tclane_lop3_r128, 512) { lane_beside_mma<L_LOP3, 128>(a); }
OP_KERNEL(tclane_duo_r192, 512) { lane_beside_mma<L_DUO, 192>(a); }
OP_KERNEL(tclane_redux_r64, 512) { lane_beside_mma<L_REDUX, 64>(a); }
OP_KERNEL(tclane_reduce_r8, 512) { lane_beside_mma<L_REDUCE, 8>(a); }
OP_KERNEL(tclaneU_mix, 512) { lane_until_mma<L_MIX, 64>(a); }
OP_KERNEL(tclaneU_imad, 512) { lane_until_mma<L_IMAD, 64>(a); }
OP_KERNEL(tclaneU_redux, 512) { lane_until_mma<L_REDUX, 64>(a); }
OP_KERNEL(tclaneU_reduce, 512) { lane_until_mma<L_REDUCE, 8>(a); }
OP_KERNEL(tclaneU_duo, 512) { lane_until_mma<L_DUO, 48>(a); }
OP_KERNEL(tclaneU_lop3, 512) { lane_until_mma<L_LOP3, 128>(a); }
OP_KERNEL(tcliU_base, 384) { tcli_until_mma<0, 8>(a); }
OP_KERNEL(tcliU_redux, 384) { tcli_until_mma<1, 8>(a); }
OP_KERNEL(tcliU_reduce, 384) { tcli_until_mma<2, 8>(a); }
OP_KERNEL(tcbulk_cp, 256) { bulk_until_mma<BULK_CP>(a); }
OP_KERNEL(tcbulk_red, 256) { bulk_until_mma<BULK_RED>(a); }
OP_KERNEL(bulkcp, 32) { bulk_solo_tc<BULK_CP>(a); }
OP_KERNEL(bulkred, 32) { bulk_solo_tc<BULK_RED>(a); }

#define UI "UTCIMMA*=4"
#define P_TC(NAME, PAT) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 128, .expect = PAT, .warps = "1", .iters = 4096}
#define P_LD(NAME, PAT) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 256, .expect = PAT, .iters = 4096}
static const OpProbe PROBES[] = {
    P_TC(tc_i8_m128_n32, UI),
    P_TC(tc_i8_m128_n64, UI),
    P_TC(tc_i8_m128_n128, UI),
    P_TC(tc_i8_m128_n256, UI),
    {.name = "tcz_i8_m128_n256", .kernel = (const void*)tc_i8_m128_n256, .tpb = 128, .expect = UI, .warps = "1", .mode = 1, .iters = 4096, .fn = "tc_i8_m128_n256"},
    P_TC(tc_i8_m64_n64, UI),
    P_TC(tc_i8_m64_n128, UI),
    P_TC(tc_i8_m64_n256, UI),
    P_TC(tc_f8_m128_n128, "UTCQMMA*=4"),
    P_TC(tc_f8_m128_n256, "UTCQMMA*=4"),
    P_TC(tc_f16_m128_n256, "UTCHMMA*=4"),
    P_LD(ld_32x32b_x32, "LDTM.x32=8"),
    P_LD(ld_32x32b_x16, "LDTM.x16=16"),
    P_LD(ld_16x256b_x4, "LDTM.16dp256bit.x4=16"),
    P_LD(ld_16x64b_x16, "LDTM.16dp64bit.x16=16"),
    P_LD(ld_16x128b_x8, "LDTM.16dp128bit.x8=16"),
    P_LD(ld_32x32b_x4, "LDTM.x4=16"),
    P_LD(ld_32x32b_x1, "LDTM=16"),
    {.name = "ld16w_32x32b_x32", .kernel = (const void*)ld16w_32x32b_x32, .tpb = 512, .expect = "LDTM.x32=8", .iters = 4096},
    {.name = "ld16w_32x32b_x16", .kernel = (const void*)ld16w_32x32b_x16, .tpb = 512, .expect = "LDTM.x16=16", .iters = 4096},
    P_LD(ld_32x32b_x64, "LDTM.x64=4"),
    P_LD(ld_32x32b_x128, "LDTM.x128=4"),
#undef P_TC
#undef P_LD
    {.name = "ldmma_w4", .kernel = (const void*)ldmma_w4, .tpb = 256, .expect = UI ";LDTM.x32=8", .warps = "1;4", .iters = 2048},
    {.name = "ldmma_w8", .kernel = (const void*)ldmma_w8, .tpb = 384, .expect = UI ";LDTM.x32=8", .warps = "1;8", .iters = 2048},
    {.name = "ldmma_w12", .kernel = (const void*)ldmma_w12, .tpb = 512, .expect = UI ";LDTM.x32=8", .warps = "1;12", .iters = 2048},
    {.name = "ldmma_w8_r16", .kernel = (const void*)ldmma_w8_r16, .tpb = 384, .expect = UI ";LDTM.x32=16", .warps = "1;8", .iters = 2048},
    {.name = "ldmma_w12_r16", .kernel = (const void*)ldmma_w12_r16, .tpb = 512, .expect = UI ";LDTM.x32=16", .warps = "1;12", .iters = 2048},
    {.name = "ldU_w4", .kernel = (const void*)ldU_w4, .tpb = 256, .expect = UI ";LDTM.x32=8", .warps = "1;4", .iters = 2048, .until = 2},
    {.name = "ldU_w8", .kernel = (const void*)ldU_w8, .tpb = 384, .expect = UI ";LDTM.x32=8", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "ldU_w12", .kernel = (const void*)ldU_w12, .tpb = 512, .expect = UI ";LDTM.x32=8", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "mmaU_ld8", .kernel = (const void*)mmaU_ld8, .tpb = 384, .expect = "LDTM.x32=8;" UI, .warps = "8;1", .iters = 2048, .until = 2},
    {.name = "syncU_k32", .kernel = (const void*)syncU_k32, .tpb = 512, .expect = UI ";IMMA.16832.S8.S8=8", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "syncU_k16", .kernel = (const void*)syncU_k16, .tpb = 512, .expect = UI ";IMMA.16816.S8.S8=8", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "mmaU_sync_k32", .kernel = (const void*)mmaU_sync_k32, .tpb = 512, .expect = "IMMA.16832.S8.S8=8;" UI, .warps = "8;1", .iters = 2048, .until = 2},
    {.name = "imma_s8_k32", .kernel = (const void*)imma_s8_k32, .tpb = 512, .expect = "IMMA.16832.S8.S8=8", .iters = 4096},
    {.name = "imma_s8_k16", .kernel = (const void*)imma_s8_k16, .tpb = 512, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
    {.name = "syncmma_k32", .kernel = (const void*)syncmma_k32, .tpb = 512, .expect = UI ";IMMA.16832.S8.S8=8", .warps = "1;8", .iters = 2048},
    {.name = "syncmma_k16", .kernel = (const void*)syncmma_k16, .tpb = 512, .expect = UI ";IMMA.16816.S8.S8=8", .warps = "1;8", .iters = 2048},
    {.name = "tclane_mix_r64", .kernel = (const void*)tclane_mix_r64, .tpb = 512, .expect = UI ";FFMA=32", .warps = "1;12", .iters = 2048},
    {.name = "tclane_mix_r256", .kernel = (const void*)tclane_mix_r256, .tpb = 512, .expect = UI ";FFMA=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_imad_r128", .kernel = (const void*)tclane_imad_r128, .tpb = 512, .expect = UI ";IMAD=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_lop3_r128", .kernel = (const void*)tclane_lop3_r128, .tpb = 512, .expect = UI ";LOP3.LUT=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_duo_r192", .kernel = (const void*)tclane_duo_r192, .tpb = 512, .expect = UI ";LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048},
    {.name = "tclane_redux_r64", .kernel = (const void*)tclane_redux_r64, .tpb = 512, .expect = UI ";REDUX*=64", .warps = "1;12", .iters = 2048},
    {.name = "tclane_reduce_r8", .kernel = (const void*)tclane_reduce_r8, .tpb = 512, .expect = UI ";SHFL.BFLY=40,LOP3.LUT=40", .warps = "1;12", .iters = 2048},
    {.name = "tclaneU_mix", .kernel = (const void*)tclaneU_mix, .tpb = 512, .expect = UI ";FFMA=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_imad", .kernel = (const void*)tclaneU_imad, .tpb = 512, .expect = UI ";IMAD=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_redux", .kernel = (const void*)tclaneU_redux, .tpb = 512, .expect = UI ";REDUX*=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_reduce", .kernel = (const void*)tclaneU_reduce, .tpb = 512, .expect = UI ";SHFL.BFLY=40,LOP3.LUT=40", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_lop3", .kernel = (const void*)tclaneU_lop3, .tpb = 512, .expect = UI ";LOP3.LUT=128", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcbulk_cp", .kernel = (const void*)tcbulk_cp, .tpb = 256, .expect = UI ";UBLKCP*=8", .counted = "UBLKCP*", .warps = "1;1", .words = 1024, .iters = 4096, .until = 2},
    {.name = "tcbulk_red", .kernel = (const void*)tcbulk_red, .tpb = 256, .expect = UI ";UBLKRED*=8", .counted = "UBLKRED*", .warps = "1;1", .words = 1024, .iters = 4096, .until = 2},
    {.name = "bulkcp", .kernel = (const void*)bulkcp, .tpb = 32, .expect = "UBLKCP*=8", .words = 1024, .iters = 1024},
    {.name = "bulkred", .kernel = (const void*)bulkred, .tpb = 32, .expect = "UBLKRED*=8", .words = 1024, .iters = 1024},
    {.name = "tcliU_base", .kernel = (const void*)tcliU_base, .tpb = 384, .expect = UI ";LDTM.x32=4,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_redux", .kernel = (const void*)tcliU_redux, .tpb = 384, .expect = UI ";LDTM.x32=4,LOP3.LUT=68,IMAD=64,REDUX*=1", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_reduce", .kernel = (const void*)tcliU_reduce, .tpb = 384, .expect = UI ";LDTM.x32=4,LOP3.LUT=73,IMAD=64,SHFL.BFLY=5", .warps = "1;8", .iters = 2048, .until = 2},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
