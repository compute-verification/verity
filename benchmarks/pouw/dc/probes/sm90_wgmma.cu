// EPS1 DC on Hopper (sm_90a, H100/H200): TENSOR's M3 (note:20261017T0321Z-draft-eps1-tensor §5) as opcode probes.
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
// Lanes: mix (add.u32 and FFMA alternating; ptxas on sm_90 splits the adds between IADD3 and IMAD.IADD, the FMA pipe),
// imad, lop3, shfl (interleaved only: warp-specialized, ptxas pulls the clock read into a loop), duo (the fold's 3 ops a word).
// dc/analyze.py turns rows.jsonl into words a tensor-pipe cycle and the issue cost a beside each lane.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "../wgmma.cuh"

constexpr int KB = 32;  // every form's K tile is 32 bytes of a row: s8/e4m3 k32, bf16 k16, tf32 k8
constexpr int AROWS = 64, BROWS = 256;

// Operand bytes: random codes valid for the type (mode 0), or zeros (mode 1).
template <int TY>
DC_DEV uint32_t operand_word(uint32_t i, int mode) {
  if (mode == 1) return 0u;
  const uint32_t x = dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  if constexpr (TY == 0) return x;                                   // s8/u8: every code
  if constexpr (TY == 1) return x & 0x77777777u;                     // e4m3/e5m2: no NaN or inf code
  if constexpr (TY == 2) return (x & 0x807F807Fu) | 0x3F003F00u;     // bf16 in ±[0.5, 1)
  return (x & 0x007FFFFFu) | 0x3F800000u;                            // tf32 in [1, 2)
}

struct Tiles {
  uint64_t da, db;
};
// Fills A (64 rows) and B (256 rows) in the K-major layout; returns the descriptors. All threads call it.
template <int TY>
DC_DEV Tiles fill_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) w[i] = operand_word<TY>(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}

#define FORM(NAME, ACC, NACC, TY)                                                                  \
  struct F_##NAME {                                                                               \
    using Acc = ACC;                                                                              \
    static constexpr int NA = NACC, TYPE = TY;                                                    \
    DC_DEV static void mma(Acc (&d)[NACC], const Tiles& t, int sd) { wgmma_##NAME(d, t.da, t.db, sd); } \
  };
FORM(s8_n64, int, 32, 0)
FORM(s8_n128, int, 64, 0)
FORM(s8_n256, int, 128, 0)
FORM(u8s8_n256, int, 128, 0)
FORM(e4m3_n256, float, 128, 1)
FORM(e5m2_n256, float, 128, 1)
FORM(bf16_n256, float, 128, 2)
FORM(bf16_n128, float, 64, 2)
FORM(tf32_n256, float, 128, 3)
#undef FORM
struct F_s8_n256_rs {  // A from registers: each thread's 4 words of the 64 x 32 A fragment
  using Acc = int;
  static constexpr int NA = 128, TYPE = 0;
  DC_DEV static void mma(int (&d)[128], const Tiles& t, int sd) {
    const uint32_t a[4] = {(uint32_t)t.da ^ threadIdx.x, (uint32_t)(t.da >> 32), (uint32_t)t.db, 0x01020304u};
    wgmma_s8_n256_rs(d, a, t.db, sd);
  }
};

template <class Acc, int N>
DC_DEV uint32_t fold_acc(const Acc (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= *reinterpret_cast<const uint32_t*>(&d[i]);
  return x;
}

// Four wgmma into one accumulator: the first overwrites, the rest accumulate.
template <class F>
DC_DEV void group4(typename F::Acc (&d)[F::NA], const Tiles& t) {
  wgmma_fence();
  F::mma(d, t, 0);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  F::mma(d, t, 1);
  wgmma_commit();
}

// ---- lane ops ---------------------------------------------------------------------------------------------------------
enum { L_MIX, L_IMAD, L_LOP3, L_SHFL, L_DUO };
struct Lane {
  uint32_t r[8];
  float f[8];
  DC_DEV explicit Lane(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      r[i] = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      f[i] = 1.0f + (float)(r[i] & 0xffu) * 0x1p-12f;
    }
  }
  template <int X>
  DC_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7;
    if constexpr (X == L_MIX) {
      if (i & 1) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[(i >> 1) & 7]) : "f"(f[((i >> 1) + 1) & 7]));
      else asm volatile("add.u32 %0,%0,%1;" : "+r"(r[(i >> 1) & 7]) : "r"(r[((i >> 1) + 1) & 7]));
    }
    if constexpr (X == L_IMAD) asm volatile("mad.lo.u32 %0,%0,%1,%2;" : "+r"(r[s]) : "r"(r[t]), "r"(0x9E3779B9u));
    if constexpr (X == L_LOP3) asm volatile("lop3.b32 %0,%0,%1,%2,0x96;" : "+r"(r[s]) : "r"(r[t]), "r"(0x85EBCA6Bu));
    if constexpr (X == L_SHFL) asm volatile("shfl.sync.bfly.b32 %0,%0,1,0x1f,0xffffffff;" : "+r"(r[s]));
  }
  // The duo fold, three ops a word: p is a word of the chain's own state, so ptxas can't fold it to a constant.
  template <int T>
  DC_DEV void duo_op(int i) {
    DuoSt st{r[i & 7], r[(i + 4) & 7]};
    duo<T>(st, r[(i + 1) & 7]);
    r[i & 7] = st.y;
    r[(i + 4) & 7] = st.z;
  }
  DC_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x += r[i] ^ __float_as_uint(f[i]);
    return x;
  }
};
template <int Q, int N>
DC_DEV void duo_ops(Lane& l) {
  if constexpr (Q < N) {
    l.duo_op<Q % DUO_NPOS>(Q);
    duo_ops<Q + 1, N>(l);
  }
}
// R lane ops (the duo lane: R/3 words of three ops each)
template <int X, int R>
DC_DEV void lane_ops(Lane& l) {
  if constexpr (X == L_DUO) {
    duo_ops<0, R / 3>(l);
  } else {
#pragma unroll
    for (int q = 0; q < R; q++) l.op<X>(q);
  }
}

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

template <int X, int R>
DC_DEV void wg_mix(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<0>(sm, a.mode);
  int d[128] = {};
  Lane l(threadIdx.x);
  OP_BEGIN
  OP_LOOP(it) {
    group4<F_s8_n256>(d, t);
    lane_ops<X, R>(l);
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ l.fold() ^ threadIdx.x)
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

#define ALONE(NAME)                                                         \
  OP_KERNEL(wg_##NAME##_a0, 256) { wg_alone<F_##NAME, 0>(a); }               \
  OP_KERNEL(wg_##NAME##_a1, 256) { wg_alone<F_##NAME, 1>(a); }
ALONE(s8_n64)
ALONE(s8_n128)
ALONE(s8_n256)
ALONE(s8_n256_rs)
ALONE(u8s8_n256)
ALONE(e4m3_n256)
ALONE(e5m2_n256)
ALONE(bf16_n256)
ALONE(bf16_n128)
ALONE(tf32_n256)
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
JOINT(duo, L_DUO, 96)
JOINT(duo, L_DUO, 384)
#undef JOINT
OP_KERNEL(imma_s8_k32, 512) { imma_alone<32>(a); }
OP_KERNEL(imma_s8_k16, 512) { imma_alone<16>(a); }

// SASS opcodes (CUDA 13.0, sm_90a): IGMMA for s8, QGMMA for fp8, HGMMA for bf16/tf32.
#define P_ALONE(NAME, PAT, ...) \
  {.name = "wg_" #NAME "_a0", .kernel = (const void*)wg_##NAME##_a0, .tpb = 256, .expect = PAT, .iters = 4096, ##__VA_ARGS__}, \
  {.name = "wg_" #NAME "_a1", .kernel = (const void*)wg_##NAME##_a1, .tpb = 256, .expect = PAT, .iters = 4096, ##__VA_ARGS__}
static const OpProbe PROBES[] = {
    P_ALONE(s8_n64, "IGMMA.64x64x32.S8.S8*=4"),
    P_ALONE(s8_n128, "IGMMA.64x128x32.S8.S8*=4"),
    P_ALONE(s8_n256, "IGMMA.64x256x32.S8.S8*=4"),
    P_ALONE(s8_n256_rs, "IGMMA.64x256x32.S8.S8*=4"),
    P_ALONE(u8s8_n256, "IGMMA.64x256x32.U8.S8*=4"),
    P_ALONE(e4m3_n256, "QGMMA.64x256x32.F32.E4M3.E4M3*=4"),
    P_ALONE(e5m2_n256, "QGMMA.64x256x32.F32.E5M2.E5M2*=4"),
    P_ALONE(bf16_n256, "HGMMA.64x256x16.F32.BF16*=4"),
    P_ALONE(bf16_n128, "HGMMA.64x128x16.F32.BF16*=4"),
    P_ALONE(tf32_n256, "HGMMA.64x256x8.F32.TF32*=4"),
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
    P_JOINT(duo, 96, "LOP3.LUT=32,IMAD=32"),
    P_JOINT(duo, 384, "LOP3.LUT=128,IMAD=128"),
#undef P_JOINT
    {.name = "imma_s8_k32", .kernel = (const void*)imma_s8_k32, .tpb = 512, .expect = "IMMA.16832.S8.S8=8", .iters = 4096},
    {.name = "imma_s8_k16", .kernel = (const void*)imma_s8_k16, .tpb = 512, .expect = "IMMA.16816.S8.S8=8", .iters = 4096},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
