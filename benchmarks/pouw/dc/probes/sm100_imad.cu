// sm_100a's table for imad_forms.cuh: every loop's opcodes as ptxas 13.0.88 emits them (the eight most frequent,
// the branch aside).
#include "imad_forms.cuh"

static const OpProbe PROBES[] = {
    P_I(s_imadhi, "IMAD.HI.U32=64,MOV=7,HFMA2=4,UIADD3=1,UISETP.NE.AND=1", "IMAD.HI.U32"),
    P_I(d_imadhi, "LOP3.LUT=32,IMAD.HI.U32=32,IMAD=32,LEA.HI=32,IMAD.MOV.U32=17,HFMA2=7,MOV=6,UIADD3=1", "IMAD.HI.U32"),
    P_I(d_imad, "IMAD=64,LOP3.LUT=32,LEA.HI=32,UIADD3=1,UISETP.NE.AND=1", "IMAD"),
    P_I(d_lop3, "LOP3.LUT=64,IMAD=32,LEA.HI=32,UIADD3=1,UISETP.NE.AND=1", "LOP3.LUT"),
    P_I(f_duo, "LOP3.LUT=24,IMAD=24,LEA.HI=24,UIADD3=1,UISETP.NE.AND=1", "LOP3.LUT"),
    P_I(f_duoi, "IMAD=48,LOP3.LUT=24,IMAD.HI.U32=23,MOV=11,HFMA2=8,IMAD.MOV.U32=3,UIADD3=1,UISETP.NE.AND=1", "LOP3.LUT"),
    P_I(f_duom, "IMAD=32,LOP3.LUT=24,LEA.HI=17,IMAD.HI.U32=7,IMAD.MOV.U32=5,HFMA2=2,UIADD3=1,UISETP.NE.AND=1", "LOP3.LUT"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
