// H32 on the H200 (sm_90a), the tensor-core register writers: wgmma of every kind at small and large N and with FP16
// accumulate, legacy mma.sync of every shape the PTX ISA gives sm_90 at 128 MMAs a trip, DMMA, and the tensor pipes
// co-issued with h32.cuh's scalar lanes. Notes as sm90_h32.cu's (c=CLASS w=PATTERN:REGS); ../h32.py prices them.
//   wa_<form>         2 warpgroups, 4 wgmma a commit group, wait_group 1 (wg_common.cuh's wg_alone)
//   ms_<form>         512 threads, 8 accumulators a warp, 128 mma.sync a trip (Round 8's loop-free reading)
//   mm_<form>_<mix>   the same, each mma.sync followed by one round of the scalar mix
//   wu_<mix>          2 warpgroups of s8 m64n128k32 fixed; 2 lane warps a SMSP run the mix only while they do
//   wu256_<mix>       the same at m64n256k32, 1 lane warp a SMSP (the register file holds no more)
// CUDA 13.0 lowers m8n8k4 to a CALL into an emulation routine (ms_*_m8k4 time the call, per D register), min.f64 to
// DSETP and two selects, and m16n8k32 e4m3 to 2 HMMA.16816.F32 on a zero accumulator and 4 FADD (sm_90 has no FP8
// mma.sync of its own), whose HMMAs ptxas hoists out of a loop with fixed operands: it is not timed.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "h32_mma.cuh"

// ---- wgmma alone ------------------------------------------------------------------------------------------------------
#define WA(NAME) OP_KERNEL(wa_##NAME, 256) { wg_alone<F_##NAME, 1>(a); }
WA(s8_n8)
WA(s8_n64)
WA(s8_n256)
WA(e4m3_n8)
WA(e4m3_n16)
WA(e4m3_n32)
WA(e4m3_n64)
WA(e4m3_n256)
WA(f16_n8)
WA(f16_n64)
WA(f16_n128)
WA(f16_n256)
WA(bf16_n8)
WA(bf16_n64)
WA(tf32_n8)
WA(tf32_n64)
WA(tf32_n128)
WA(b1_n8)
WA(b1_n64)
WA(b1_n128)
WA(b1_n256)
WA(f16f16_n8)
WA(f16f16_n256)
WA(e4m3f16_n8)
WA(e4m3f16_n256)
WA(e5m2f16_n256)
#undef WA

// ---- legacy mma.sync ---------------------------------------------------------------------------------------------------
MS(f16_k8, 421, "m16n8k8.row.col.f32.f16.f16.f32", float, "f", uint32_t, "r", 4, 2, 1, 4)
MS(bf16_k8, 421, "m16n8k8.row.col.f32.bf16.bf16.f32", float, "f", uint32_t, "r", 4, 2, 1, 2)
MS(tf32_k4, 421, "m16n8k4.row.col.f32.tf32.tf32.f32", float, "f", uint32_t, "r", 4, 2, 1, 3)
MS(s8_k16, 421, "m16n8k16.row.col.s32.s8.s8.s32", int, "r", uint32_t, "r", 4, 2, 1, 0)
MS(b1_k128, 421, "m16n8k128.row.col.s32.b1.b1.s32.and.popc", int, "r", uint32_t, "r", 4, 2, 1, 0)
MS(f16_k16, 442, "m16n8k16.row.col.f32.f16.f16.f32", float, "f", uint32_t, "r", 4, 4, 2, 4)
MS(bf16_k16, 442, "m16n8k16.row.col.f32.bf16.bf16.f32", float, "f", uint32_t, "r", 4, 4, 2, 2)
MS(tf32_k8, 442, "m16n8k8.row.col.f32.tf32.tf32.f32", float, "f", uint32_t, "r", 4, 4, 2, 3)
MS(s8_k32, 442, "m16n8k32.row.col.s32.s8.s8.s32", int, "r", uint32_t, "r", 4, 4, 2, 0)
MS(b1_k256, 442, "m16n8k256.row.col.s32.b1.b1.s32.and.popc", int, "r", uint32_t, "r", 4, 4, 2, 0)
MS(f16f16_k8, 221, "m16n8k8.row.col.f16.f16.f16.f16", uint32_t, "r", uint32_t, "r", 2, 2, 1, 4)
MS(f16f16_k16, 242, "m16n8k16.row.col.f16.f16.f16.f16", uint32_t, "r", uint32_t, "r", 2, 4, 2, 4)
MS(s8_m8k16, 211, "m8n8k16.row.col.s32.s8.s8.s32", int, "r", uint32_t, "r", 2, 1, 1, 0)
MS(b1_m8k128, 211, "m8n8k128.row.col.s32.b1.b1.s32.and.popc", int, "r", uint32_t, "r", 2, 1, 1, 0)
MS(f16_m8k4, 822, "m8n8k4.row.col.f32.f16.f16.f32", float, "f", uint32_t, "r", 8, 2, 2, 4)
MS(f16f16_m8k4, 422, "m8n8k4.row.col.f16.f16.f16.f16", uint32_t, "r", uint32_t, "r", 4, 2, 2, 4)
MS(f64_m8k4, 211, "m8n8k4.row.col.f64.f64.f64.f64", double, "d", double, "d", 2, 1, 1, 0)
MS(f64_k4, 421, "m16n8k4.row.col.f64.f64.f64.f64", double, "d", double, "d", 4, 2, 1, 0)
MS(f64_k8, 442, "m16n8k8.row.col.f64.f64.f64.f64", double, "d", double, "d", 4, 4, 2, 0)
#undef MS
#undef MSP

#define MSK(NAME) OP_KERNEL(ms_##NAME, 512) { ms_run<M_##NAME, 128>(a); }
MSK(f16_k8)
MSK(bf16_k8)
MSK(tf32_k4)
MSK(s8_k16)
MSK(b1_k128)
MSK(f16_k16)
MSK(bf16_k16)
MSK(tf32_k8)
MSK(s8_k32)
MSK(b1_k256)
MSK(f16f16_k8)
MSK(f16f16_k16)
MSK(s8_m8k16)
MSK(b1_m8k128)
MSK(f16_m8k4)
MSK(f16f16_m8k4)
MSK(f64_m8k4)
MSK(f64_k4)
MSK(f64_k8)
#undef MSK
OP_KERNEL(mm_f16k8_dadd2_fadd, 512) { mm_run<M_f16_k8, 64, O_DADD_I, O_DADD_I, O_FADD_I>(a); }
OP_KERNEL(mm_f16k8_dadd, 512) { mm_run<M_f16_k8, 64, O_DADD_I>(a); }
OP_KERNEL(mm_f16k8_fadd3, 512) { mm_run<M_f16_k8, 64, O_FADD_I, O_FADD_I, O_FADD_I>(a); }
OP_KERNEL(mm_f16k8_dadd_imad, 512) { mm_run<M_f16_k8, 64, O_DADD_I, O_IMAD_R>(a); }
OP_KERNEL(mm_s8k16_dadd2_fadd, 512) { mm_run<M_s8_k16, 64, O_DADD_I, O_DADD_I, O_FADD_I>(a); }
OP_KERNEL(mm_b1k128_dadd2_fadd, 512) { mm_run<M_b1_k128, 64, O_DADD_I, O_DADD_I, O_FADD_I>(a); }
OP_KERNEL(mm_f64m8k4_fadd3, 512) { mm_run<M_f64_m8k4, 64, O_FADD_I, O_FADD_I, O_FADD_I>(a); }

// ---- wgmma beside the scalar pipes ---------------------------------------------------------------------------------
template <class F, int R, int... X>
DC_DEV void wu_run(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F::TYPE>(sm, a.mode);
  const bool mma_role = threadIdx.x < 256;
  typename F::Acc d[F::NA] = {};
  HLane l(threadIdx.x ^ (uint32_t)a.mode);
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
    OP_UNTIL(n) { h32_mix<R, X...>(l); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK((mma_role ? fold_acc(d) : h32_fold<X...>(l)) ^ threadIdx.x)
}
#define WU(NAME, R, ...)                                                                  \
  OP_KERNEL(wu_##NAME, 512) { wu_run<F_s8_n128, R, __VA_ARGS__>(a); }                     \
  OP_KERNEL(wu256_##NAME, 384) { wu_run<F_s8_n256, R, __VA_ARGS__>(a); }
WU(dadd, 64, O_DADD_I)
WU(fadd, 64, O_FADD_I)
WU(dadd_fadd, 64, O_DADD_I, O_FADD_I)
WU(dadd2_imadw_fadd, 64, O_DADD_I, O_IMADW, O_DADD_I, O_FADD_I)
WU(imad_ffma, 64, O_IMAD_R, O_FFMA_RI)
WU(idp4a, 64, O_IDP4A)
WU(imadw, 64, O_IMADW)
WU(hfma2_lop3, 64, O_HFMA2_RI, O_LOP3_R)
#undef WU

// ---- the table ---------------------------------------------------------------------------------------------------------
#define P_WA(NAME, PAT, REGS) \
  {.name = "wa_" #NAME, .kernel = (const void*)wa_##NAME, .tpb = 256, .expect = PAT "=4", .iters = 4096, \
   .note = "c=wgmma w=" PAT ":" #REGS}
#define P_MS(NAME, PAT, N, NOTE) \
  {.name = "ms_" #NAME, .kernel = (const void*)ms_##NAME, .tpb = 512, .expect = PAT "=" #N, .iters = 512, .note = NOTE}
#define P_MM(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 512, .expect = PAT, .iters = 512, .note = NOTE}
#define WG128 "IGMMA.64x128x32.S8.S8*"
#define WG256 "IGMMA.64x256x32.S8.S8*"
#define P_WU(NAME, LPAT, LNOTE)                                                                                     \
  {.name = "wu_" #NAME, .kernel = (const void*)wu_##NAME, .tpb = 512, .expect = WG128 "=4;" LPAT, .warps = "8;8", \
   .iters = 4096, .note = "c=mix w=" WG128 ":64," LNOTE, .until = 2},                                              \
  {.name = "wu256_" #NAME, .kernel = (const void*)wu256_##NAME, .tpb = 384, .expect = WG256 "=4;" LPAT,           \
   .warps = "8;4", .iters = 4096, .note = "c=mix w=" WG256 ":128," LNOTE, .until = 2}
static const OpProbe PROBES[] = {
    P_WA(s8_n8, "IGMMA.64x8x32.S8.S8*", 4),
    P_WA(s8_n64, "IGMMA.64x64x32.S8.S8*", 32),
    P_WA(s8_n256, "IGMMA.64x256x32.S8.S8*", 128),
    P_WA(e4m3_n8, "QGMMA.64x8x32.F32.E4M3.E4M3*", 4),
    P_WA(e4m3_n16, "QGMMA.64x16x32.F32.E4M3.E4M3*", 8),
    P_WA(e4m3_n32, "QGMMA.64x32x32.F32.E4M3.E4M3*", 16),
    P_WA(e4m3_n64, "QGMMA.64x64x32.F32.E4M3.E4M3*", 32),
    P_WA(e4m3_n256, "QGMMA.64x256x32.F32.E4M3.E4M3*", 128),
    P_WA(f16_n8, "HGMMA.64x8x16.F32*", 4),
    P_WA(f16_n64, "HGMMA.64x64x16.F32*", 32),
    P_WA(f16_n128, "HGMMA.64x128x16.F32*", 64),
    P_WA(f16_n256, "HGMMA.64x256x16.F32*", 128),
    P_WA(bf16_n8, "HGMMA.64x8x16.F32.BF16*", 4),
    P_WA(bf16_n64, "HGMMA.64x64x16.F32.BF16*", 32),
    P_WA(tf32_n8, "HGMMA.64x8x8.F32.TF32*", 4),
    P_WA(tf32_n64, "HGMMA.64x64x8.F32.TF32*", 32),
    P_WA(tf32_n128, "HGMMA.64x128x8.F32.TF32*", 64),
    P_WA(b1_n8, "BGMMA.64x8x256.AND.POPC*", 4),
    P_WA(b1_n64, "BGMMA.64x64x256.AND.POPC*", 32),
    P_WA(b1_n128, "BGMMA.64x128x256.AND.POPC*", 64),
    P_WA(b1_n256, "BGMMA.64x256x256.AND.POPC*", 128),
    P_WA(f16f16_n8, "HGMMA.64x8x16.F16*", 2),
    P_WA(f16f16_n256, "HGMMA.64x256x16.F16*", 64),
    P_WA(e4m3f16_n8, "QGMMA.64x8x32.F16.E4M3.E4M3*", 2),
    P_WA(e4m3f16_n256, "QGMMA.64x256x32.F16.E4M3.E4M3*", 64),
    P_WA(e5m2f16_n256, "QGMMA.64x256x32.F16.E5M2.E5M2*", 64),
    P_MS(f16_k8, "HMMA.1688.F32", 128, "c=tc w=HMMA.1688.F32:4"),
    P_MS(bf16_k8, "HMMA.1688.F32.BF16", 128, "c=tc w=HMMA.1688.F32.BF16:4"),
    P_MS(tf32_k4, "HMMA.1684.F32.TF32", 128, "c=tc w=HMMA.1684.F32.TF32:4"),
    P_MS(s8_k16, "IMMA.16816.S8.S8", 128, "c=tc w=IMMA.16816.S8.S8:4"),
    P_MS(b1_k128, "BMMA.168128.AND.POPC", 128, "c=tc w=BMMA.168128.AND.POPC:4"),
    P_MS(f16_k16, "HMMA.16816.F32", 128, "c=tc w=HMMA.16816.F32:4"),
    P_MS(bf16_k16, "HMMA.16816.F32.BF16", 128, "c=tc w=HMMA.16816.F32.BF16:4"),
    P_MS(tf32_k8, "HMMA.1688.F32.TF32", 128, "c=tc w=HMMA.1688.F32.TF32:4"),
    P_MS(s8_k32, "IMMA.16832.S8.S8", 128, "c=tc w=IMMA.16832.S8.S8:4"),
    P_MS(b1_k256, "BMMA.168256.AND.POPC", 128, "c=tc w=BMMA.168256.AND.POPC:4"),
    P_MS(f16f16_k8, "HMMA.1688.F16", 128, "c=tc w=HMMA.1688.F16:2"),
    P_MS(f16f16_k16, "HMMA.16816.F16", 128, "c=tc w=HMMA.16816.F16:2"),
    P_MS(s8_m8k16, "IMMA.8816.S8.S8", 128, "c=tc w=IMMA.8816.S8.S8:2"),
    P_MS(b1_m8k128, "BMMA.88128.AND.POPC", 128, "c=tc w=BMMA.88128.AND.POPC:2"),
    P_MS(f16_m8k4, "CALL.REL.NOINC", 128, "c=tc w=CALL.REL.NOINC:8"),
    P_MS(f16f16_m8k4, "CALL.REL.NOINC", 128, "c=tc w=CALL.REL.NOINC:4"),
    P_MS(f64_m8k4, "DMMA.8x8x4*", 128, "c=tc w=DMMA.8x8x4*:4"),
    P_MS(f64_k4, "DMMA.16x8x4*", 128, "c=tc w=DMMA.16x8x4*:8"),
    P_MS(f64_k8, "DMMA.16x8x8*", 128, "c=tc w=DMMA.16x8x8*:8"),
    P_MM(mm_f16k8_dadd2_fadd, "HMMA.1688.F32=64,DADD=128,FADD=64", "c=mix w=HMMA.1688.F32:4,DADD:2,FADD:1"),
    P_MM(mm_f16k8_dadd, "HMMA.1688.F32=64,DADD=64", "c=mix w=HMMA.1688.F32:4,DADD:2"),
    P_MM(mm_f16k8_fadd3, "HMMA.1688.F32=64,FADD=192", "c=mix w=HMMA.1688.F32:4,FADD:1"),
    P_MM(mm_f16k8_dadd_imad, "HMMA.1688.F32=64,DADD=64,IMAD=64", "c=mix w=HMMA.1688.F32:4,DADD:2,IMAD:1"),
    P_MM(mm_s8k16_dadd2_fadd, "IMMA.16816.S8.S8=64,DADD=128,FADD=64", "c=mix w=IMMA.16816.S8.S8:4,DADD:2,FADD:1"),
    P_MM(mm_b1k128_dadd2_fadd, "BMMA.168128.AND.POPC=64,DADD=128,FADD=64",
         "c=mix w=BMMA.168128.AND.POPC:4,DADD:2,FADD:1"),
    P_MM(mm_f64m8k4_fadd3, "DMMA.8x8x4*=64,FADD=192", "c=mix w=DMMA.8x8x4*:4,FADD:1"),
    P_WU(dadd, "DADD=64", "DADD:2"),
    P_WU(fadd, "FADD=64", "FADD:1"),
    P_WU(dadd_fadd, "DADD=32,FADD=32", "DADD:2,FADD:1"),
    P_WU(dadd2_imadw_fadd, "DADD=32,IMAD.WIDE.U32=16,FADD=16", "DADD:2,IMAD.WIDE.U32:2,FADD:1"),
    P_WU(imad_ffma, "IMAD=32,FFMA=32", "IMAD:1,FFMA:1"),
    P_WU(idp4a, "IDP.4A*=64", "IDP.4A*:1"),
    P_WU(imadw, "IMAD.WIDE.U32=64", "IMAD.WIDE.U32:2"),
    P_WU(hfma2_lop3, "HFMA2=32,LOP3.LUT=32", "HFMA2:1,LOP3.LUT:1"),
};
#undef P_WA
#undef P_MS
#undef P_MM
#undef P_WU
#undef WG128
#undef WG256

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
