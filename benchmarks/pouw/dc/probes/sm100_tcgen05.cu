// The tensor measurement M4 on datacenter Blackwell (sm_100a, B200) as opcode probes.
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
//   tcliU_x<C>, _p<C>[f]  the LI fold beside the MMA, exact: lane warps load C columns of TMEM and fold each word
//                         (p: two buffers, a load in flight while the other folds; f: one FFMA filler op in 8 words)
//   tc_mxf4_m128_n256     kind::mxf4 128x256x64 alone; tcfU_<lane>: lanes beside it, exact (a deep form's coupling)
//   tck_<mma>_n<NF>[_t<P>] honest's x128 LI lanes with NF FFMA filler a 128 words (46: L = 96's 0.36 a word), P loads a
//                         trip, beside warp 0's i8 MMA, kind::mxf4 MMA or no MMA (a sleep), all exact (OP_UNTIL);
//                         tck_d<S>_n<NF>: beside 4 kind::mxf4 MMAs and an S ns sleep a trip (a part-busy deep form);
//                         tck_e2_* / tck_e4_*: beside kind::f8f6f4 on E2M1 / E4M3 (a same-depth kind for bit noises)
//   tclane_<lane>_r<R>    thread 0 tcgen05.mma beside 12 warps of lane ops (the scalar pipes beside a saturated MMA)
//   tclaneU_<lane>        the same roles, exact on both sides (the lanes OP_UNTIL): a tclane_ point's tensor rate is
//                         the MMA's fixed work over the lanes' longer block time, a lower bound, not a joint's edge
//   tcliU_base/redux/reduce  the honest LI loop (8 loader warps, 4 tcgen05.ld a segment) beside the MMA, exact
//   tcliU_floor           ldtm0's floor program beside the MMA, exact: the LI-CP-W redux fold on words in registers,
//                         12 lane warps, so the lanes' issue is the most the floor's mix gets beside the tensor
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
// odd a), a segment of 128 words a trip (4 tcgen05.ld.32x32b.x32). CP 0 is the bare fold (LI); CP 1 adds the LI
// checkpoint with `redux.sync.xor.b32`, CP 2 with the 5-shuffle fallback: each lane XORs its 8 chains (4 LOP3) and the
// warp reduces once a segment. The three throughputs' ratio is what the checkpoint costs the latency-bound loop, which
// the honest profile scales dc100-li-l32's measured cycles by.
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

// The floor program of dither-2s-licpw-redux-ldtm0-l32 beside the saturated MMA, exact by OP_UNTIL: tcli_until_mma's
// CP 1 segment with its words already in registers (♭ credits no tcgen05.ld) and ♭'s SHFL class (the REDUX and 2
// SHFL), so a segment is 68 LOP3, 64 IMAD and 3 SHFL-class for 128 words, ♭'s 544 : 512 : 24 a UTCIMMA. SEGS segments a
// trip and 12 lane warps (3 a SMSP) leave issue, not latency, binding: the lanes' issue is the most the floor's own mix
// gets beside the tensor.
template <int SEGS>
DC_DEV void tcfloor_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  const uint32_t w = warp_u();
  Lane l(threadIdx.x);
  uint32_t r0[16], r1[16];
#pragma unroll
  for (int c = 0; c < 16; c++) {
    r0[c] = dc_mix32(threadIdx.x * 32u + c);
    r1[c] = dc_mix32(threadIdx.x * 32u + 16u + c);
  }
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
#pragma unroll
    for (int c = 0; c < 64; c++) {
      asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[c & 7]) : "r"(r0[c & 15]), "r"(0x85EBCA6Bu));
      asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[c & 7]) : "r"(0x9E3779B9u), "r"(r1[c & 15]));
    }
    l.op<L_LOP3>(0); l.op<L_LOP3>(2); l.op<L_LOP3>(4); l.op<L_LOP3>(6);
    l.op<L_REDUX>(0); l.op<L_SHFL>(1); l.op<L_SHFL>(3);
  };
  OP_BEGIN_STOP
  if (w == 0) {
    OP_LOOP(it) { mma(); }
    OP_DONE
  } else if (w >= 4) {
    OP_UNTIL(n) {
#pragma unroll
      for (int k = 0; k < SEGS; k++) seg();
    }
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

// ---- kind::mxf4: E2M1 K = 64 a MMA at twice i8's MAC rate (lowbit_dc/fpx.cu's call and descriptor) --------------------
// The operand tile's 32 bytes a row are 64 packed E2M1 (every code valid). UE8M0 scales of 1 (0x7F) fill columns 256-383
// (uniform, so their layout doesn't matter), so the MMAs write D0 only.
DC_DEV void umma_mxf4(uint32_t d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d),
      "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
      : "memory");
}
// A and B E2M1 (1) at [7,10) and [10,13), N >> 3 at [17,23), UE8M0 scales (1) at 23, M >> 4 at [24,29)
DC_HD constexpr uint32_t idesc_mxf4(uint32_t m, uint32_t n) {
  return (1u << 7) | (1u << 10) | ((n >> 3) << 17) | (1u << 23) | ((m >> 4) << 24);
}
DC_DEV void sf_fill(uint32_t tmem) {
  if (warp_u() < 4) {
    uint32_t r[16];
#pragma unroll
    for (int i = 0; i < 16; i++) r[i] = 0x7F7F7F7Fu;
    for (int u = 0; u < 8; u++) tmem_st16(tmem + ((32u * warp_u()) << 16) + 256u + 16u * u, r);
    tmem_wait_st();
  }
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
}
DC_DEV void mxf4x4(uint32_t tmem, uint64_t da, uint64_t db) {
  constexpr uint32_t idesc = idesc_mxf4(128, 256);
  umma_mxf4(tmem, da, db, idesc, tmem + 256, tmem + 320, 0);
  umma_mxf4(tmem, da, db, idesc, tmem + 256, tmem + 320, 1);
  umma_mxf4(tmem, da, db, idesc, tmem + 256, tmem + 320, 1);
  umma_mxf4(tmem, da, db, idesc, tmem + 256, tmem + 320, 1);
}
DC_DEV void mxf4_alone(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  sf_fill(tmem);
  const uint64_t da = desc_a(s), db = desc_b(s);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mxf4x4(tmem, da, db);
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

// ---- lane ops beside tcgen05.mma ---------------------------------------------------------------------------------------
// thread 0: tcgen05.mma; warps 4-15 (3 a SMSP): R lane ops an iteration.
template <int X, int R, int N = 256>
DC_DEV void lane_beside_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  Lane l(threadIdx.x);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4<0, 128, N>(tmem, da, db);
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
// TMA bulk-async copies beside the saturated tcgen05.mma: one side warp's lane 0 issues cp.async.bulk (SASS
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
// MXF4: the MMA is kind::mxf4 (mxf4x4) instead of i8.
template <int X, int R, bool MXF4 = false>
DC_DEV void lane_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  if constexpr (MXF4) sf_fill(tmem);
  const uint64_t da = desc_a(s), db = desc_b(s);
  Lane l(threadIdx.x);
  OP_BEGIN_STOP
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) {
        if constexpr (MXF4) mxf4x4(tmem, da, db);
        else mma4<0, 128, 256>(tmem, da, db);
      }
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
// The LI fold as the honest kernel runs it: warp 0 the MMA, fixed (ld_until_mma's 4 into D0); warps 4 .. 4 + LW - 1 load
// C columns of their lane quarter of D1 (tcgen05.ld.32x32b.x<C>) and fold each word with one LI op (LOP3 and IMAD
// alternating) until it is done. At L = 64 the MMA commits one slot in two, so the lanes keep up when they fold
// UTCIMMA's rate x 128 x 256 / 2 words a clock. PIPE: two buffers, each load in flight while the other buffer folds.
// FILL: one FFMA filler op every 8 words. PER: loads and folds a trip (the stop is polled once a trip). Each thread
// holds C (PIPE: 2C) loaded registers, so 128 of them run 8 lane warps (384 threads).
template <int C>
DC_DEV void tmem_ldc(uint32_t taddr, uint32_t (&z)[C]) {
  if constexpr (C == 32) tmem_ld32(taddr, z);
  else if constexpr (C == 64) tmem_ld64(taddr, z);
  else tmem_ld128(taddr, z);
}
template <int C, bool FILL>
DC_DEV void li_fold(Lane& l, const uint32_t (&z)[C]) {
#pragma unroll
  for (int i = 0; i < C; i += 2) {
    asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[i & 7]) : "r"(z[i]), "r"(0x85EBCA6Bu));
    asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[(i + 1) & 7]) : "r"(z[i + 1]), "r"(0x9E3779B9u));
    if constexpr (FILL) {
      if (i % 8 == 6) l.op<L_FFMA>(i / 8);
    }
  }
}
template <int C, int LW, bool PIPE = false, bool FILL = false, int PER = 1>
DC_DEV void li_until_mma(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256), SPAN = PIPE ? 2 * C : C;
  const uint32_t w = warp_u(), col = tmem + ((32u * (w & 3u)) << 16) + 256u + (((w - 4u) >> 2) * SPAN) % 256u;
  Lane l(threadIdx.x);
  uint32_t z[C], y[PIPE ? C : 1];
  OP_BEGIN_STOP
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
    OP_DONE
  } else if (w >= 4 && w < 4 + LW) {
    if constexpr (PIPE) {
      tmem_ldc<C>(col, z);
      tmem_wait_ld();
      OP_UNTIL(n) {
        tmem_ldc<C>(col + C, y);
        li_fold<C, FILL>(l, z);
        tmem_wait_ld();
        tmem_ldc<C>(col, z);
        li_fold<C, FILL>(l, y);
        tmem_wait_ld();
      }
      OP_TRIPS(n)
    } else {
      OP_UNTIL(n) {
#pragma unroll
        for (int k = 0; k < PER; k++) {
          tmem_ldc<C>(col, z);
          tmem_wait_ld();
          li_fold<C, FILL>(l, z);
        }
      }
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
  const uint32_t v = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}
// li_until_mma's lane warps with no MMA (warps 0-3 idle), fixed iterations: the most these lanes reach when the MMA's
// work is removed, which caps what any MMA schedule beside them can give the lanes.
template <int C, int LW, bool FILL = false>
DC_DEV void li_alone(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  const uint32_t w = warp_u(), col = tmem + ((32u * (w & 3u)) << 16) + 256u + (((w - 4u) >> 2) * C) % 256u;
  Lane l(threadIdx.x);
  uint32_t z[C];
  OP_BEGIN
  if (w >= 4 && w < 4 + LW) {
    OP_LOOP(it) {
      tmem_ldc<C>(col, z);
      tmem_wait_ld();
      li_fold<C, FILL>(l, z);
    }
  }
  OP_END
  const uint32_t v = sink_ld(tmem, 0) ^ l.fold();
  teardown(tmem);
  OP_SINK(v ^ threadIdx.x)
}
// li_fold with NF FFMA filler ops spread over the C words (L = 96's filler, 0.36 a word, is 46 a 128)
template <int C, int NF>
DC_DEV void li_foldn(Lane& l, const uint32_t (&z)[C]) {
#pragma unroll
  for (int i = 0; i < C; i += 2) {
    asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(l.r[i & 7]) : "r"(z[i]), "r"(0x85EBCA6Bu));
    asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(l.r[(i + 1) & 7]) : "r"(z[i + 1]), "r"(0x9E3779B9u));
    if constexpr (NF > 0) {
      if ((i + 2) * NF / C > i * NF / C) l.op<L_FFMA>(i * NF / C);
    }
  }
}
// li_until_mma's x<C> lanes (LW warps, PER loads and folds a trip, NF filler a load) beside warp 0 running, for fixed
// iterations, MMA = 0: 4 kind::i8 MMAs (honest's), 1: 4 kind::mxf4 MMAs (the deep form), 2: no MMA (a sleep), 3: 4
// kind::mxf4 MMAs and an SL ns sleep (the deep form part-busy), 4: 4 kind::f8f6f4 MMAs on E2M1 (K = 32, the same depth
// as i8, holding the bit noises' 0 and ±1), 5: the same on E4M3. The lane rate is OP_UNTIL trips over the stop in all.
template <int C, int LW, int NF, int PER, int MMA, int SL = 256>
DC_DEV void li_until_k(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup<0>(s, a.mode);
  if constexpr (MMA == 1 || MMA == 3) sf_fill(tmem);
  const uint64_t da = desc_a(s), db = desc_b(s);
  constexpr uint32_t idesc = idesc_of<0>(128, 256);
  const uint32_t w = warp_u(), col = tmem + ((32u * (w & 3u)) << 16) + 256u + (((w - 4u) >> 2) * C) % 256u;
  Lane l(threadIdx.x);
  uint32_t z[C];
  OP_BEGIN_STOP
  if (w == 0) {
    OP_LOOP(it) {
      if constexpr (MMA == 2) {
        __nanosleep(256);
      } else {
        if (elect_one()) {
          if constexpr (MMA == 1 || MMA == 3) {
            mxf4x4(tmem, da, db);
          } else if constexpr (MMA == 4 || MMA == 5) {
            constexpr uint32_t fdesc = MMA == 4 ? umma_idesc(1, 5, 5, 128, 256) : idesc_of<1>(128, 256);
            umma_f8f6f4(tmem, da, db, fdesc, 0);
            umma_f8f6f4(tmem, da, db, fdesc, 1);
            umma_f8f6f4(tmem, da, db, fdesc, 1);
            umma_f8f6f4(tmem, da, db, fdesc, 1);
          } else {
            umma_i8(tmem, da, db, idesc, 0);
            umma_i8(tmem, da, db, idesc, 1);
            umma_i8(tmem, da, db, idesc, 1);
            umma_i8(tmem, da, db, idesc, 1);
          }
        }
        __syncwarp();
        if constexpr (MMA == 3) __nanosleep(SL);
      }
    }
    OP_DONE
  } else if (w >= 4 && w < 4 + LW) {
    OP_UNTIL(n) {
#pragma unroll
      for (int k = 0; k < PER; k++) {
        tmem_ldc<C>(col, z);
        tmem_wait_ld();
        li_foldn<C, NF>(l, z);
      }
    }
    OP_TRIPS(n)
  }
  OP_END_STOP
  if (w == 0) {
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
OP_KERNEL(tclane_mix_r256_n128, 512) { lane_beside_mma<L_MIX, 256, 128>(a); }
OP_KERNEL(tclane_imad_r128_n128, 512) { lane_beside_mma<L_IMAD, 128, 128>(a); }
OP_KERNEL(tclane_lop3_r128_n128, 512) { lane_beside_mma<L_LOP3, 128, 128>(a); }
OP_KERNEL(tclane_duo_r192_n128, 512) { lane_beside_mma<L_DUO, 192, 128>(a); }
OP_KERNEL(tclane_imad_r128_n64, 512) { lane_beside_mma<L_IMAD, 128, 64>(a); }
OP_KERNEL(tclane_lop3_r128_n64, 512) { lane_beside_mma<L_LOP3, 128, 64>(a); }
OP_KERNEL(tclane_duo_r192_n64, 512) { lane_beside_mma<L_DUO, 192, 64>(a); }
OP_KERNEL(tclaneU_mix, 512) { lane_until_mma<L_MIX, 64>(a); }
OP_KERNEL(tclaneU_imad, 512) { lane_until_mma<L_IMAD, 64>(a); }
OP_KERNEL(tclaneU_redux, 512) { lane_until_mma<L_REDUX, 64>(a); }
OP_KERNEL(tclaneU_reduce, 512) { lane_until_mma<L_REDUCE, 8>(a); }
OP_KERNEL(tclaneU_duo, 512) { lane_until_mma<L_DUO, 48>(a); }
OP_KERNEL(tclaneU_lop3, 512) { lane_until_mma<L_LOP3, 128>(a); }
OP_KERNEL(tcliU_base, 384) { tcli_until_mma<0, 8>(a); }
OP_KERNEL(tcliU_redux, 384) { tcli_until_mma<1, 8>(a); }
OP_KERNEL(tcliU_reduce, 384) { tcli_until_mma<2, 8>(a); }
OP_KERNEL(tcliU_floor, 512) { tcfloor_until_mma<4>(a); }
OP_KERNEL(tcbulk_cp, 256) { bulk_until_mma<BULK_CP>(a); }
OP_KERNEL(tcbulk_red, 256) { bulk_until_mma<BULK_RED>(a); }
OP_KERNEL(bulkcp, 32) { bulk_solo_tc<BULK_CP>(a); }
OP_KERNEL(bulkred, 32) { bulk_solo_tc<BULK_RED>(a); }
OP_KERNEL(tclaneU_lop3_r64, 512) { lane_until_mma<L_LOP3, 64>(a); }
OP_KERNEL(tclaneU_li, 512) { lane_until_mma<L_LI, 64>(a); }
OP_KERNEL(tcliU_x32, 512) { li_until_mma<32, 12>(a); }
OP_KERNEL(tcliU_x64, 512) { li_until_mma<64, 12>(a); }
OP_KERNEL(tcliU_x128, 384) { li_until_mma<128, 8>(a); }
OP_KERNEL(tcliU_p32, 512) { li_until_mma<32, 12, true>(a); }
OP_KERNEL(tcliU_p64, 384) { li_until_mma<64, 8, true>(a); }
OP_KERNEL(tcliU_p32f, 512) { li_until_mma<32, 12, true, true>(a); }
OP_KERNEL(tclaneU_lif, 512) { lane_until_mma<L_LIF, 64>(a); }
OP_KERNEL(tclaneU_li_r512, 512) { lane_until_mma<L_LI, 512>(a); }
OP_KERNEL(tcliU_x64_t2, 384) { li_until_mma<64, 8, false, false, 2>(a); }
OP_KERNEL(tcliU_x32_t8, 512) { li_until_mma<32, 12, false, false, 8>(a); }
OP_KERNEL(tc_mxf4_m128_n256, 128) { mxf4_alone(a); }
OP_KERNEL(tcfU_li, 512) { lane_until_mma<L_LI, 64, true>(a); }
OP_KERNEL(tcfU_mix, 512) { lane_until_mma<L_MIX, 64, true>(a); }
OP_KERNEL(tcfU_li_r512, 512) { lane_until_mma<L_LI, 512, true>(a); }
OP_KERNEL(tcfU_mix_r512, 512) { lane_until_mma<L_MIX, 512, true>(a); }
OP_KERNEL(tclaneU_li_r128, 512) { lane_until_mma<L_LI, 128>(a); }
OP_KERNEL(tclaneU_li_r256, 512) { lane_until_mma<L_LI, 256>(a); }
OP_KERNEL(tcfU_li_r128, 512) { lane_until_mma<L_LI, 128, true>(a); }
OP_KERNEL(tcfU_li_r256, 512) { lane_until_mma<L_LI, 256, true>(a); }
OP_KERNEL(tcfU_li_r1024, 512) { lane_until_mma<L_LI, 1024, true>(a); }
OP_KERNEL(tclaneU_lif_r128, 512) { lane_until_mma<L_LIF, 128>(a); }
OP_KERNEL(tclaneU_lif_r512, 512) { lane_until_mma<L_LIF, 512>(a); }
OP_KERNEL(tcfU_lif, 512) { lane_until_mma<L_LIF, 64, true>(a); }
OP_KERNEL(tcfU_lif_r128, 512) { lane_until_mma<L_LIF, 128, true>(a); }
OP_KERNEL(tcfU_lif_r512, 512) { lane_until_mma<L_LIF, 512, true>(a); }
OP_KERNEL(tcliU_x128f, 384) { li_until_mma<128, 8, false, true>(a); }
OP_KERNEL(tclia_x128, 384) { li_alone<128, 8>(a); }
OP_KERNEL(tclia_x128f, 384) { li_alone<128, 8, true>(a); }
OP_KERNEL(tck_i8_n46, 384) { li_until_k<128, 8, 46, 1, 0>(a); }
OP_KERNEL(tck_i8_n46_t2, 384) { li_until_k<128, 8, 46, 2, 0>(a); }
OP_KERNEL(tck_f4_n46, 384) { li_until_k<128, 8, 46, 1, 1>(a); }
OP_KERNEL(tck_f4_n46_t2, 384) { li_until_k<128, 8, 46, 2, 1>(a); }
OP_KERNEL(tck_no_n46, 384) { li_until_k<128, 8, 46, 1, 2>(a); }
OP_KERNEL(tck_no_n46_t2, 384) { li_until_k<128, 8, 46, 2, 2>(a); }
OP_KERNEL(tck_i8_n16, 384) { li_until_k<128, 8, 16, 1, 0>(a); }
OP_KERNEL(tck_i8_n16_t2, 384) { li_until_k<128, 8, 16, 2, 0>(a); }
OP_KERNEL(tck_f4_n16, 384) { li_until_k<128, 8, 16, 1, 1>(a); }
OP_KERNEL(tck_f4_n16_t2, 384) { li_until_k<128, 8, 16, 2, 1>(a); }
OP_KERNEL(tck_no_n16, 384) { li_until_k<128, 8, 16, 1, 2>(a); }
OP_KERNEL(tck_no_n16_t2, 384) { li_until_k<128, 8, 16, 2, 2>(a); }
OP_KERNEL(tck_i8_n0, 384) { li_until_k<128, 8, 0, 1, 0>(a); }
OP_KERNEL(tck_f4_n0, 384) { li_until_k<128, 8, 0, 1, 1>(a); }
OP_KERNEL(tck_no_n0, 384) { li_until_k<128, 8, 0, 1, 2>(a); }
OP_KERNEL(tck_i8_n32, 384) { li_until_k<128, 8, 32, 1, 0>(a); }
OP_KERNEL(tck_f4_n32, 384) { li_until_k<128, 8, 32, 1, 1>(a); }
OP_KERNEL(tck_no_n32, 384) { li_until_k<128, 8, 32, 1, 2>(a); }
OP_KERNEL(tck_no_n32_t2, 384) { li_until_k<128, 8, 32, 2, 2>(a); }
OP_KERNEL(tck_i8_n36, 384) { li_until_k<128, 8, 36, 1, 0>(a); }
OP_KERNEL(tck_f4_n36, 384) { li_until_k<128, 8, 36, 1, 1>(a); }
OP_KERNEL(tck_no_n36, 384) { li_until_k<128, 8, 36, 1, 2>(a); }
OP_KERNEL(tck_no_n36_t2, 384) { li_until_k<128, 8, 36, 2, 2>(a); }
OP_KERNEL(tck_i8_n40, 384) { li_until_k<128, 8, 40, 1, 0>(a); }
OP_KERNEL(tck_f4_n40, 384) { li_until_k<128, 8, 40, 1, 1>(a); }
OP_KERNEL(tck_no_n40, 384) { li_until_k<128, 8, 40, 1, 2>(a); }
OP_KERNEL(tck_no_n40_t2, 384) { li_until_k<128, 8, 40, 2, 2>(a); }
OP_KERNEL(tck_d128_n46, 384) { li_until_k<128, 8, 46, 1, 3, 128>(a); }
OP_KERNEL(tck_d512_n46, 384) { li_until_k<128, 8, 46, 1, 3, 512>(a); }
OP_KERNEL(tck_d128_n36, 384) { li_until_k<128, 8, 36, 1, 3, 128>(a); }
OP_KERNEL(tck_d512_n36, 384) { li_until_k<128, 8, 36, 1, 3, 512>(a); }
OP_KERNEL(tck_e2_n0, 384) { li_until_k<128, 8, 0, 1, 4>(a); }
OP_KERNEL(tck_e2_n0_t2, 384) { li_until_k<128, 8, 0, 2, 4>(a); }
OP_KERNEL(tck_e4_n0, 384) { li_until_k<128, 8, 0, 1, 5>(a); }
OP_KERNEL(tck_i8_n0_t2, 384) { li_until_k<128, 8, 0, 2, 0>(a); }
OP_KERNEL(tck_no_n0_t2, 384) { li_until_k<128, 8, 0, 2, 2>(a); }

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
    {.name = "tclane_mix_r256_n128", .kernel = (const void*)tclane_mix_r256_n128, .tpb = 512, .expect = UI ";FFMA=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_imad_r128_n128", .kernel = (const void*)tclane_imad_r128_n128, .tpb = 512, .expect = UI ";IMAD=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_lop3_r128_n128", .kernel = (const void*)tclane_lop3_r128_n128, .tpb = 512, .expect = UI ";LOP3.LUT=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_duo_r192_n128", .kernel = (const void*)tclane_duo_r192_n128, .tpb = 512, .expect = UI ";LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048},
    {.name = "tclane_imad_r128_n64", .kernel = (const void*)tclane_imad_r128_n64, .tpb = 512, .expect = UI ";IMAD=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_lop3_r128_n64", .kernel = (const void*)tclane_lop3_r128_n64, .tpb = 512, .expect = UI ";LOP3.LUT=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_duo_r192_n64", .kernel = (const void*)tclane_duo_r192_n64, .tpb = 512, .expect = UI ";LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048},
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
    {.name = "tcliU_floor", .kernel = (const void*)tcliU_floor, .tpb = 512, .expect = UI ";LOP3.LUT=272,IMAD=256,REDUX*=4,SHFL.BFLY=8", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_redux", .kernel = (const void*)tcliU_redux, .tpb = 384, .expect = UI ";LDTM.x32=4,LOP3.LUT=68,IMAD=64,REDUX*=1", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_reduce", .kernel = (const void*)tcliU_reduce, .tpb = 384, .expect = UI ";LDTM.x32=4,LOP3.LUT=73,IMAD=64,SHFL.BFLY=5", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tclaneU_lop3_r64", .kernel = (const void*)tclaneU_lop3_r64, .tpb = 512, .expect = UI ";LOP3.LUT=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_li", .kernel = (const void*)tclaneU_li, .tpb = 512, .expect = UI ";LOP3.LUT=32,IMAD=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_x32", .kernel = (const void*)tcliU_x32, .tpb = 512, .expect = UI ";LDTM.x32=1,LOP3.LUT=16,IMAD=16", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_x64", .kernel = (const void*)tcliU_x64, .tpb = 512, .expect = UI ";LDTM.x64=1,LOP3.LUT=32,IMAD=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_x128", .kernel = (const void*)tcliU_x128, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_p32", .kernel = (const void*)tcliU_p32, .tpb = 512, .expect = UI ";LDTM.x32=2,LOP3.LUT=32,IMAD=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_p64", .kernel = (const void*)tcliU_p64, .tpb = 384, .expect = UI ";LDTM.x64=2,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_p32f", .kernel = (const void*)tcliU_p32f, .tpb = 512, .expect = UI ";LDTM.x32=2,LOP3.LUT=32,IMAD=32,FFMA=8", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_lif", .kernel = (const void*)tclaneU_lif, .tpb = 512, .expect = UI ";LOP3.LUT=28,IMAD=28,FFMA=8", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_li_r512", .kernel = (const void*)tclaneU_li_r512, .tpb = 512, .expect = UI ";LOP3.LUT=256,IMAD=256", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_x64_t2", .kernel = (const void*)tcliU_x64_t2, .tpb = 384, .expect = UI ";LDTM.x64=2,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tcliU_x32_t8", .kernel = (const void*)tcliU_x32_t8, .tpb = 512, .expect = UI ";LDTM.x32=8,LOP3.LUT=128,IMAD=128", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tc_mxf4_m128_n256", .kernel = (const void*)tc_mxf4_m128_n256, .tpb = 128, .expect = "UTC*MMA*=4", .warps = "1", .iters = 4096},
    {.name = "tcfU_li", .kernel = (const void*)tcfU_li, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=32,IMAD=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_mix", .kernel = (const void*)tcfU_mix, .tpb = 512, .expect = "UTC*MMA*=4;FFMA=32", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_li_r512", .kernel = (const void*)tcfU_li_r512, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=256,IMAD=256", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_mix_r512", .kernel = (const void*)tcfU_mix_r512, .tpb = 512, .expect = "UTC*MMA*=4;FFMA=256", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_li_r128", .kernel = (const void*)tclaneU_li_r128, .tpb = 512, .expect = UI ";LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_li_r256", .kernel = (const void*)tclaneU_li_r256, .tpb = 512, .expect = UI ";LOP3.LUT=128,IMAD=128", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_li_r128", .kernel = (const void*)tcfU_li_r128, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_li_r256", .kernel = (const void*)tcfU_li_r256, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=128,IMAD=128", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_li_r1024", .kernel = (const void*)tcfU_li_r1024, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=512,IMAD=512", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_lif_r128", .kernel = (const void*)tclaneU_lif_r128, .tpb = 512, .expect = UI ";LOP3.LUT=56,IMAD=56,FFMA=16", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tclaneU_lif_r512", .kernel = (const void*)tclaneU_lif_r512, .tpb = 512, .expect = UI ";LOP3.LUT=224,IMAD=224,FFMA=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_lif", .kernel = (const void*)tcfU_lif, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=28,IMAD=28,FFMA=8", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_lif_r128", .kernel = (const void*)tcfU_lif_r128, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=56,IMAD=56,FFMA=16", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcfU_lif_r512", .kernel = (const void*)tcfU_lif_r512, .tpb = 512, .expect = "UTC*MMA*=4;LOP3.LUT=224,IMAD=224,FFMA=64", .warps = "1;12", .iters = 2048, .until = 2},
    {.name = "tcliU_x128f", .kernel = (const void*)tcliU_x128f, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=16", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tclia_x128", .kernel = (const void*)tclia_x128, .tpb = 384, .expect = "LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "8", .iters = 2048},
    {.name = "tclia_x128f", .kernel = (const void*)tclia_x128f, .tpb = 384, .expect = "LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=16", .warps = "8", .iters = 2048},
    {.name = "tck_i8_n46", .kernel = (const void*)tck_i8_n46, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=46", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n46_t2", .kernel = (const void*)tck_i8_n46_t2, .tpb = 384, .expect = UI ";LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=92", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n46", .kernel = (const void*)tck_f4_n46, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=46", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n46_t2", .kernel = (const void*)tck_f4_n46_t2, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=92", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n46", .kernel = (const void*)tck_no_n46, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=46", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n46_t2", .kernel = (const void*)tck_no_n46_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=92", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n16", .kernel = (const void*)tck_i8_n16, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=16", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n16_t2", .kernel = (const void*)tck_i8_n16_t2, .tpb = 384, .expect = UI ";LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n16", .kernel = (const void*)tck_f4_n16, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=16", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n16_t2", .kernel = (const void*)tck_f4_n16_t2, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n16", .kernel = (const void*)tck_no_n16, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=16", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n16_t2", .kernel = (const void*)tck_no_n16_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n0", .kernel = (const void*)tck_i8_n0, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n0", .kernel = (const void*)tck_f4_n0, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n0", .kernel = (const void*)tck_no_n0, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n32", .kernel = (const void*)tck_i8_n32, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n32", .kernel = (const void*)tck_f4_n32, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n32", .kernel = (const void*)tck_no_n32, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=32", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n32_t2", .kernel = (const void*)tck_no_n32_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n36", .kernel = (const void*)tck_i8_n36, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=36", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n36", .kernel = (const void*)tck_f4_n36, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=36", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n36", .kernel = (const void*)tck_no_n36, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=36", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n36_t2", .kernel = (const void*)tck_no_n36_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=72", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n40", .kernel = (const void*)tck_i8_n40, .tpb = 384, .expect = UI ";LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=40", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_f4_n40", .kernel = (const void*)tck_f4_n40, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=40", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n40", .kernel = (const void*)tck_no_n40, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=40", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n40_t2", .kernel = (const void*)tck_no_n40_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128,FFMA=80", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_d128_n46", .kernel = (const void*)tck_d128_n46, .tpb = 384, .expect = "UTC*MMA*=4,NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=46", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_d512_n46", .kernel = (const void*)tck_d512_n46, .tpb = 384, .expect = "UTC*MMA*=4,NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=46", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_d128_n36", .kernel = (const void*)tck_d128_n36, .tpb = 384, .expect = "UTC*MMA*=4,NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=36", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_d512_n36", .kernel = (const void*)tck_d512_n36, .tpb = 384, .expect = "UTC*MMA*=4,NANOSLEEP=1;LDTM.x128=1,LOP3.LUT=64,IMAD=64,FFMA=36", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_e2_n0", .kernel = (const void*)tck_e2_n0, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_e2_n0_t2", .kernel = (const void*)tck_e2_n0_t2, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=2,LOP3.LUT=128,IMAD=128", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_e4_n0", .kernel = (const void*)tck_e4_n0, .tpb = 384, .expect = "UTC*MMA*=4;LDTM.x128=1,LOP3.LUT=64,IMAD=64", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_i8_n0_t2", .kernel = (const void*)tck_i8_n0_t2, .tpb = 384, .expect = UI ";LDTM.x128=2,LOP3.LUT=128,IMAD=128", .warps = "1;8", .iters = 2048, .until = 2},
    {.name = "tck_no_n0_t2", .kernel = (const void*)tck_no_n0_t2, .tpb = 384, .expect = "NANOSLEEP=1;LDTM.x128=2,LOP3.LUT=128,IMAD=128", .warps = "1;8", .iters = 2048, .until = 2},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
