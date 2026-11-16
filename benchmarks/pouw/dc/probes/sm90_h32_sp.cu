// H32 on the H200 (sm_90a), sparse mma.sync (mma.sp::ordered_metadata, h32_mma.cuh's 2:4 metadata) at 128 MMAs a trip,
// as sm90_h32_tc.cu's ms_ probes. Its own binary, so a shape ptxas refuses for sm_90 takes only these probes with it.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "h32_mma.cuh"

MSP(f16_k16, S422, "m16n8k16.row.col.f32.f16.f16.f32", float, "f", uint32_t, "r", 4, 2, 2, 4)
MSP(f16_k32, S444, "m16n8k32.row.col.f32.f16.f16.f32", float, "f", uint32_t, "r", 4, 4, 4, 4)
MSP(bf16_k32, S444, "m16n8k32.row.col.f32.bf16.bf16.f32", float, "f", uint32_t, "r", 4, 4, 4, 2)
MSP(f16f16_k32, S244, "m16n8k32.row.col.f16.f16.f16.f16", uint32_t, "r", uint32_t, "r", 2, 4, 4, 4)
MSP(s8_k32, S422, "m16n8k32.row.col.s32.s8.s8.s32", int, "r", uint32_t, "r", 4, 2, 2, 0)
MSP(s8_k64, S444, "m16n8k64.row.col.s32.s8.s8.s32", int, "r", uint32_t, "r", 4, 4, 4, 0)
MSP(tf32_k8, S422, "m16n8k8.row.col.f32.tf32.tf32.f32", float, "f", uint32_t, "r", 4, 2, 2, 3)
MSP(tf32_k16, S444, "m16n8k16.row.col.f32.tf32.tf32.f32", float, "f", uint32_t, "r", 4, 4, 4, 3)
#undef MS
#undef MSP

#define SPK(NAME) OP_KERNEL(sp_##NAME, 512) { ms_run<M_##NAME, 128>(a); }
SPK(f16_k16)
SPK(f16_k32)
SPK(bf16_k32)
SPK(f16f16_k32)
SPK(s8_k32)
SPK(s8_k64)
SPK(tf32_k8)
SPK(tf32_k16)
#undef SPK
OP_KERNEL(mm_spf16k32_dadd2_fadd, 512) { mm_run<M_f16_k32, 64, O_DADD_I, O_DADD_I, O_FADD_I>(a); }

#define P_SP(NAME, PAT, REGS)                                                                                     \
  {.name = "sp_" #NAME, .kernel = (const void*)sp_##NAME, .tpb = 512, .expect = PAT "=128", .iters = 512,         \
   .note = "c=tc w=" PAT ":" #REGS}
static const OpProbe PROBES[] = {
    P_SP(f16_k16, "HMMA.SP*", 4),
    P_SP(f16_k32, "HMMA.SP*", 4),
    P_SP(bf16_k32, "HMMA.SP*", 4),
    P_SP(f16f16_k32, "HMMA.SP*", 2),
    P_SP(s8_k32, "IMMA.SP*", 4),
    P_SP(s8_k64, "IMMA.SP*", 4),
    P_SP(tf32_k8, "HMMA.SP*", 4),
    P_SP(tf32_k16, "HMMA.SP*", 4),
    {.name = "mm_spf16k32_dadd2_fadd", .kernel = (const void*)mm_spf16k32_dadd2_fadd, .tpb = 512,
     .expect = "HMMA.SP*=64,DADD=128,FADD=64", .iters = 512, .note = "c=mix w=HMMA.SP*:4,DADD:2,FADD:1"},
};
#undef P_SP

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
