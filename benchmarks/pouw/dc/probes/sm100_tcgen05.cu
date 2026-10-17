// EPS1 DC on datacenter Blackwell (sm_100a, B200): TENSOR's M4 (note:20261017T0321Z-draft-eps1-tensor §5) as opcode probes.
// Build with ../opcode_probe/probe.py build --arch sm_100a; gated and poisoned as every opcode probe (probe.cuh).
// tcgen05.mma (cta_group::1) is issued by one thread into tensor memory (TMEM, 512 columns allocated a CTA) from one
// shared-memory tile (no-swizzle K-major descriptors, dc.cuh); its completion is one mbarrier phase after the loop.
//   tc_<kind>_m<M>_n<N>   4 tcgen05.mma an iteration into two accumulators (M4a: slots written a tensor-pipe cycle;
//                         cta_group::2 needs a cluster pair and is not probed yet)
//   ld_<shape>            8 warps, 4 groups an iteration of 4 tcgen05.ld (x32: 2) and a tcgen05.wait::ld (M4b alone):
//                         32x32b.x{1,4,16,32}, 16x64b.x16, 16x128b.x8, 16x256b.x4 (x64 and x128 are not probed yet)
//   ldmma_w<LW>           warp 0: tcgen05.mma i8 128x256x32 into columns 0-255; LW warps: 8 tcgen05.ld.32x32b.x32 of
//                         columns 256-511 an iteration (M4b joint facet: 64, 128 words a clock asked)
//   imma_s8_k32 / _k16    mma.sync alone on sm_100 (M4c)
//   syncmma_k32 / _k16    thread 0 tcgen05.mma beside 8 warps of mma.sync (M4c: does mma.sync share the tcgen05 pipe?)
//   tclane_<lane>_r<R>    thread 0 tcgen05.mma beside 12 warps of lane ops (the scalar pipes beside a saturated MMA)
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
      } else {
        tmem_ld1(col(64 * g), v0[0]); tmem_ld1(col(64 * g + 16), v1[0]); tmem_ld1(col(64 * g + 32), v2[0]);
        tmem_ld1(col(64 * g + 48), v3[0]);
        tmem_wait_ld();
        acc += v0[0] ^ v1[0] ^ v2[0] ^ v3[0];
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
#undef P_TC
#undef P_LD
    {.name = "ldmma_w4", .kernel = (const void*)ldmma_w4, .tpb = 256, .expect = UI ";LDTM.x32=8", .warps = "1;4", .iters = 2048},
    {.name = "ldmma_w8", .kernel = (const void*)ldmma_w8, .tpb = 384, .expect = UI ";LDTM.x32=8", .warps = "1;8", .iters = 2048},
    {.name = "ldmma_w12", .kernel = (const void*)ldmma_w12, .tpb = 512, .expect = UI ";LDTM.x32=8", .warps = "1;12", .iters = 2048},
    {.name = "ldmma_w8_r16", .kernel = (const void*)ldmma_w8_r16, .tpb = 384, .expect = UI ";LDTM.x32=16", .warps = "1;8", .iters = 2048},
    {.name = "ldmma_w12_r16", .kernel = (const void*)ldmma_w12_r16, .tpb = 512, .expect = UI ";LDTM.x32=16", .warps = "1;12", .iters = 2048},
    {.name = "imma_s8_k32", .kernel = (const void*)imma_s8_k32, .tpb = 512, .expect = "IMMA.16832.S8.S8=8", .iters = 4096},
    {.name = "imma_s8_k16", .kernel = (const void*)imma_s8_k16, .tpb = 512, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
    {.name = "syncmma_k32", .kernel = (const void*)syncmma_k32, .tpb = 512, .expect = UI ";IMMA.16832.S8.S8=8", .warps = "1;8", .iters = 2048},
    {.name = "syncmma_k16", .kernel = (const void*)syncmma_k16, .tpb = 512, .expect = UI ";IMMA.16816.S8.S8=8", .warps = "1;8", .iters = 2048},
    {.name = "tclane_mix_r64", .kernel = (const void*)tclane_mix_r64, .tpb = 512, .expect = UI ";FFMA=32", .warps = "1;12", .iters = 2048},
    {.name = "tclane_mix_r256", .kernel = (const void*)tclane_mix_r256, .tpb = 512, .expect = UI ";FFMA=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_imad_r128", .kernel = (const void*)tclane_imad_r128, .tpb = 512, .expect = UI ";IMAD=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_lop3_r128", .kernel = (const void*)tclane_lop3_r128, .tpb = 512, .expect = UI ";LOP3.LUT=128", .warps = "1;12", .iters = 2048},
    {.name = "tclane_duo_r192", .kernel = (const void*)tclane_duo_r192, .tpb = 512, .expect = UI ";LOP3.LUT=64,IMAD=64", .warps = "1;12", .iters = 2048},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
