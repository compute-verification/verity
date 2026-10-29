// H32 on the H200 (sm_90a), the scalar register writers: every form of MH100's ffma, int32, fp64 and other classes
// alone, and the co-issued mixes across pipes. Build with ../opcode_probe/probe.py build --arch sm_90a; every probe
// passes the SASS gate, the poisoned sink and its no-write control before it is timed (probe.cuh). ../h32.py turns
// rows.jsonl into MH100's unit, 4,096 over the registers a SM writes a clock.
//   x_<form>        1024 threads (8 warps a SMSP), 512 of the form an iteration: the loop's own 3 instructions are 0.6%
//                   of it (Round 8's loop-free reading); _r64: 64 an iteration (Round 7's reading, the loop's share)
//   m_<a>_<b>...    a mix in one warp, its forms round-robin on independent chains (h32.cuh's h32_mix), 512 an iteration
// Each probe's note is what h32.py prices: c=CLASS (ffma, int32, fp64, other; mix for a mix across classes) and
// w=PATTERN:REGS,..., the 32-bit registers one warp instruction of each counted pattern writes a lane.
#include "../../opcode_probe/probe.cuh"
#include "h32.cuh"

template <int R, int... X>
DC_DEV void h_run(const OpArgs& a) {
  HLane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) { h32_mix<R, X...>(l); }
  OP_END
  OP_SINK(h32_fold<X...>(l) ^ threadIdx.x)
}

#define X1(NAME, ...) OP_KERNEL(x_##NAME, 1024) { h_run<512, __VA_ARGS__>(a); }
#define X64(NAME, ...) OP_KERNEL(x_##NAME##_r64, 1024) { h_run<64, __VA_ARGS__>(a); }
#define MX(NAME, ...) OP_KERNEL(m_##NAME, 1024) { h_run<512, __VA_ARGS__>(a); }
X1(fadd_i, O_FADD_I)
X1(fadd_r, O_FADD_R)
X1(fmul_i, O_FMUL_I)
X1(fmul_r, O_FMUL_R)
X1(ffma_ri, O_FFMA_RI)
X1(ffma_rr, O_FFMA_RR)
X1(ffma_ii, O_FFMA_II)
X1(iadd_r, O_IADD_R)
X1(iadd3_rr, O_IADD3_RR)
X1(addxor_i, O_ADDXOR_I)
X1(mulxor_i, O_MULXOR_I)
X1(lop3_r, O_LOP3_R)
X1(lop3_rr, O_LOP3_RR)
X1(shf_r, O_SHF_R)
X1(lea, O_LEA)
X1(imad_r, O_IMAD_R)
X1(imad_rr, O_IMAD_RR)
X1(imadw, O_IMADW)
X1(imadhi, O_IMADHI)
X1(idp4a, O_IDP4A)
X1(idp2a, O_IDP2A)
X1(imnmx, O_IMNMX)
X1(popc, O_POPC)
X1(flo, O_FLO)
X1(dadd_i, O_DADD_I)
X1(dadd_r, O_DADD_R)
X1(dmul_i, O_DMUL_I)
X1(dmul_r, O_DMUL_R)
X1(dfma_ri, O_DFMA_RI)
X1(dfma_rr, O_DFMA_RR)
X1(dmnmx, O_DMNMX)
X1(f2f64, O_F2F64)
X1(hadd2_i, O_HADD2_I)
X1(hadd2_r, O_HADD2_R)
X1(hmul2_i, O_HMUL2_I)
X1(hmul2_r, O_HMUL2_R)
X1(hfma2_ri, O_HFMA2_RI)
X1(hfma2_rr, O_HFMA2_RR)
X1(badd2_r, O_BADD2_R)
X1(bfma2_ri, O_BFMA2_RI)
X1(hmnmx2, O_HMNMX2)
X1(f2fp_b, O_F2FP_B)
OP_KERNEL(x_e4m3, 1024) { h_run<256, O_E4M3>(a); }
X1(fmnmx, O_FMNMX)
X1(fsel, O_FSEL)
X1(sel, O_SEL)
X1(prmt, O_PRMT)
X1(ex2, O_EX2)
X1(rsq, O_RSQ)
X1(f2i, O_F2I)
X1(frnd, O_FRND)
X1(shfl, O_SHFL)
X1(vote, O_VOTE)
X64(fadd_i, O_FADD_I)
X64(ffma_rr, O_FFMA_RR)
X64(iadd_r, O_IADD_R)
X64(dadd_i, O_DADD_I)
X64(dfma_rr, O_DFMA_RR)
X64(hadd2_r, O_HADD2_R)
MX(imad_lop3, O_IMAD_R, O_LOP3_R)
MX(imad_iadd, O_IMAD_R, O_IADD_R)
MX(lop3_iadd, O_LOP3_R, O_IADD_R)
MX(imadw_lop3, O_IMADW, O_LOP3_R)
MX(hadd2_prmt, O_HADD2_I, O_PRMT)
MX(hfma2_lop3, O_HFMA2_RI, O_LOP3_R)
MX(ffma_lop3, O_FFMA_RI, O_LOP3_R)
MX(dadd_fadd, O_DADD_I, O_FADD_I)
MX(daddr_faddr, O_DADD_R, O_FADD_R)
MX(dfma_fadd, O_DFMA_RI, O_FADD_I)
MX(dadd_lop3, O_DADD_I, O_LOP3_R)
MX(dadd_imad, O_DADD_I, O_IMAD_R)
MX(dadd_hadd2, O_DADD_I, O_HADD2_I)
MX(dadd_idp4a, O_DADD_I, O_IDP4A)
MX(dadd2_imadw_fadd, O_DADD_I, O_IMADW, O_DADD_I, O_FADD_I)
MX(dadd2_imadw_lop3, O_DADD_I, O_IMADW, O_DADD_I, O_LOP3_R)
MX(imadw_fadd3, O_IMADW, O_FADD_I, O_FADD_I, O_FADD_I)
MX(imadw_lop3_fadd, O_IMADW, O_LOP3_R, O_FADD_I, O_LOP3_R)
MX(imad_ffma, O_IMAD_R, O_FFMA_RI)
MX(idp4a_ffma, O_IDP4A, O_FFMA_RI)
MX(dadd2_imad_fadd, O_DADD_I, O_IMAD_R, O_DADD_I, O_FADD_I)
#undef X1
#undef X64
#undef MX

#define P(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .iters = 512, .note = NOTE}
#define P64(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .iters = 4096, .note = NOTE}
static const OpProbe PROBES[] = {
    P(x_fadd_i, "FADD=512", "c=ffma w=FADD:1"),
    P(x_fadd_r, "FADD=512", "c=ffma w=FADD:1"),
    P(x_fmul_i, "FMUL=512", "c=ffma w=FMUL:1"),
    P(x_fmul_r, "FMUL=512", "c=ffma w=FMUL:1"),
    P(x_ffma_ri, "FFMA=512", "c=ffma w=FFMA:1"),
    P(x_ffma_rr, "FFMA=512", "c=ffma w=FFMA:1"),
    P(x_ffma_ii, "FFMA=512", "c=ffma w=FFMA:1"),
    P(x_iadd_r, "IMAD.IADD=261,IADD3=251", "c=int32 w=IMAD.IADD:1,IADD3:1"),
    P(x_iadd3_rr, "IADD3=512", "c=int32 w=IADD3:1"),
    P(x_addxor_i, "VIADD=256,LOP3.LUT=256", "c=int32 w=VIADD:1,LOP3.LUT:1"),
    P(x_mulxor_i, "IMAD=256,LOP3.LUT=256", "c=int32 w=IMAD:1,LOP3.LUT:1"),
    P(x_lop3_r, "LOP3.LUT=512", "c=int32 w=LOP3.LUT:1"),
    P(x_lop3_rr, "LOP3.LUT=512", "c=int32 w=LOP3.LUT:1"),
    P(x_shf_r, "SHF.L.W.U32.HI=512", "c=int32 w=SHF.L.W.U32.HI:1"),
    P(x_lea, "LEA.HI=512", "c=int32 w=LEA.HI:1"),
    P(x_imad_r, "IMAD=512", "c=int32 w=IMAD:1"),
    P(x_imad_rr, "IMAD=512", "c=int32 w=IMAD:1"),
    P(x_imadw, "IMAD.WIDE.U32=512", "c=int32 w=IMAD.WIDE.U32:2"),
    P(x_imadhi, "IMAD.HI.U32=512", "c=int32 w=IMAD.HI.U32:1"),
    P(x_idp4a, "IDP.4A*=512", "c=int32 w=IDP.4A*:1"),
    P(x_idp2a, "IDP.2A*=512", "c=int32 w=IDP.2A*:1"),
    P(x_imnmx, "*IMNMX*=512", "c=int32 w=*IMNMX*:1"),
    P(x_popc, "POPC=512", "c=int32 w=POPC:1"),
    P(x_flo, "FLO*=512", "c=int32 w=FLO*:1"),
    P(x_dadd_i, "DADD=512", "c=fp64 w=DADD:2"),
    P(x_dadd_r, "DADD=512", "c=fp64 w=DADD:2"),
    P(x_dmul_i, "DMUL=512", "c=fp64 w=DMUL:2"),
    P(x_dmul_r, "DMUL=512", "c=fp64 w=DMUL:2"),
    P(x_dfma_ri, "DFMA=512", "c=fp64 w=DFMA:2"),
    P(x_dfma_rr, "DFMA=512", "c=fp64 w=DFMA:2"),
    P(x_dmnmx, "DSETP.MIN.AND=512,SEL=512,FSEL=512", "c=fp64 w=SEL:1,FSEL:1"),
    P(x_f2f64, "F2F.F64.F32=256,F2F.F32.F64=256", "c=fp64 w=F2F.F64.F32:2,F2F.F32.F64:1"),
    P(x_hadd2_i, "HADD2=512", "c=other w=HADD2:1"),
    P(x_hadd2_r, "HFMA2.MMA=256,HADD2=256", "c=other w=HFMA2.MMA:1,HADD2:1"),
    P(x_hmul2_i, "HFMA2.MMA=256,HMUL2=256", "c=other w=HFMA2.MMA:1,HMUL2:1"),
    P(x_hmul2_r, "HFMA2.MMA=256,HMUL2=256", "c=other w=HFMA2.MMA:1,HMUL2:1"),
    P(x_hfma2_ri, "HFMA2.MMA=256,HFMA2=256", "c=other w=HFMA2.MMA:1,HFMA2:1"),
    P(x_hfma2_rr, "HFMA2.MMA=256,HFMA2=256", "c=other w=HFMA2.MMA:1,HFMA2:1"),
    P(x_badd2_r, "*BF16*=512", "c=other w=*BF16*:1"),
    P(x_bfma2_ri, "HFMA2.MMA.BF16_V2=256,HFMA2.BF16_V2=256", "c=other w=HFMA2.MMA.BF16_V2:1,HFMA2.BF16_V2:1"),
    P(x_hmnmx2, "HMNMX2=512", "c=other w=HMNMX2:1"),
    P(x_f2fp_b, "F2FP.BF16*=256", "c=other w=F2FP.BF16*:1"),
    P(x_e4m3, "F2FP*=512", "c=other w=F2FP*:1"),
    P(x_fmnmx, "FMNMX=512", "c=other w=FMNMX:1"),
    P(x_fsel, "FSEL=512", "c=other w=FSEL:1"),
    P(x_sel, "SEL=512", "c=other w=SEL:1"),
    P(x_prmt, "PRMT=512", "c=other w=PRMT:1"),
    P(x_ex2, "MUFU.EX2=512", "c=other w=MUFU.EX2:1"),
    P(x_rsq, "MUFU.RSQ=512", "c=other w=MUFU.RSQ:1"),
    P(x_f2i, "F2I*=256,I2F*=256", "c=other w=F2I*:1,I2F*:1"),
    P(x_frnd, "FRND=256,FADD=256", "c=mix w=FRND:1,FADD:1"),
    P(x_shfl, "SHFL.BFLY=512", "c=other w=SHFL.BFLY:1"),
    P(x_vote, "VOTE*=512", "c=other w=VOTE*:1"),
    P64(x_fadd_i_r64, "FADD=64", "c=ffma w=FADD:1"),
    P64(x_ffma_rr_r64, "FFMA=64", "c=ffma w=FFMA:1"),
    P64(x_iadd_r_r64, "IMAD.IADD=37,IADD3=27", "c=int32 w=IMAD.IADD:1,IADD3:1"),
    P64(x_dadd_i_r64, "DADD=64", "c=fp64 w=DADD:2"),
    P64(x_dfma_rr_r64, "DFMA=64", "c=fp64 w=DFMA:2"),
    P64(x_hadd2_r_r64, "HFMA2.MMA=32,HADD2=32", "c=other w=HFMA2.MMA:1,HADD2:1"),
    P(m_imad_lop3, "IMAD=256,LOP3.LUT=256", "c=int32 w=IMAD:1,LOP3.LUT:1"),
    P(m_imad_iadd, "IMAD=256,IADD3=244,IMAD.IADD=12", "c=int32 w=IMAD:1,IADD3:1,IMAD.IADD:1"),
    P(m_lop3_iadd, "LOP3.LUT=256,IMAD.IADD=256", "c=int32 w=LOP3.LUT:1,IMAD.IADD:1"),
    P(m_imadw_lop3, "IMAD.WIDE.U32=256,LOP3.LUT=256", "c=int32 w=IMAD.WIDE.U32:2,LOP3.LUT:1"),
    P(m_hadd2_prmt, "HADD2=256,PRMT=256", "c=other w=HADD2:1,PRMT:1"),
    P(m_hfma2_lop3, "HFMA2.MMA=128,HFMA2=128,LOP3.LUT=256", "c=mix w=HFMA2.MMA:1,HFMA2:1,LOP3.LUT:1"),
    P(m_ffma_lop3, "FFMA=256,LOP3.LUT=256", "c=mix w=FFMA:1,LOP3.LUT:1"),
    P(m_dadd_fadd, "DADD=256,FADD=256", "c=mix w=DADD:2,FADD:1"),
    P(m_daddr_faddr, "DADD=256,FADD=256", "c=mix w=DADD:2,FADD:1"),
    P(m_dfma_fadd, "DFMA=256,FADD=256", "c=mix w=DFMA:2,FADD:1"),
    P(m_dadd_lop3, "DADD=256,LOP3.LUT=256", "c=mix w=DADD:2,LOP3.LUT:1"),
    P(m_dadd_imad, "DADD=256,IMAD=256", "c=mix w=DADD:2,IMAD:1"),
    P(m_dadd_hadd2, "DADD=256,HADD2=256", "c=mix w=DADD:2,HADD2:1"),
    P(m_dadd_idp4a, "DADD=256,IDP.4A*=256", "c=mix w=DADD:2,IDP.4A*:1"),
    P(m_dadd2_imadw_fadd, "DADD=256,IMAD.WIDE.U32=128,FADD=128", "c=mix w=DADD:2,IMAD.WIDE.U32:2,FADD:1"),
    P(m_dadd2_imadw_lop3, "DADD=256,IMAD.WIDE.U32=128,LOP3.LUT=128", "c=mix w=DADD:2,IMAD.WIDE.U32:2,LOP3.LUT:1"),
    P(m_imadw_fadd3, "IMAD.WIDE.U32=128,FADD=384", "c=mix w=IMAD.WIDE.U32:2,FADD:1"),
    P(m_imadw_lop3_fadd, "IMAD.WIDE.U32=128,LOP3.LUT=256,FADD=128", "c=mix w=IMAD.WIDE.U32:2,LOP3.LUT:1,FADD:1"),
    P(m_imad_ffma, "IMAD=256,FFMA=256", "c=mix w=IMAD:1,FFMA:1"),
    P(m_idp4a_ffma, "IDP.4A*=256,FFMA=256", "c=mix w=IDP.4A*:1,FFMA:1"),
    P(m_dadd2_imad_fadd, "DADD=256,IMAD=128,FADD=128", "c=mix w=DADD:2,IMAD:1,FADD:1"),
};
#undef P
#undef P64

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
