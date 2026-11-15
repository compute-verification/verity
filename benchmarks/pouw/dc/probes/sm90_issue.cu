// sm_90a's table for uniform.cuh's issue-saturated stream (b: the honest step + 256 FFMA): every loop's opcodes as
// ptxas 13.0.88 emits them (the eight most frequent, the branch aside), so a chain on lanes or a merged instruction refuses.
#include "uniform.cuh"

static const OpProbe PROBES[] = {
    P_U(ub_0, "FFMA=256,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,ISETP.LE.AND=1", "IMMA.16832.S8.S8"),
    P_U(ub_64i, "FFMA=256,UIMAD=32,ULOP3.LUT=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(ub_64x, "FFMA=256,ULOP3.LUT=96,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,ISETP.LE.AND=1", "IMMA.16832.S8.S8"),
    P_U(ub_64a, "FFMA=256,UIADD3=33,ULOP3.LUT=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,ISETP.LE.AND=1", "IMMA.16832.S8.S8"),
    P_U(ub_64f, "FFMA=256,USHF.L.W.U32.HI=32,ULOP3.LUT=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(ub_64p, "FFMA=256,UPRMT=32,ULOP3.LUT=32,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(ub_128i, "FFMA=256,UIMAD=64,ULOP3.LUT=64,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1", "IMMA.16832.S8.S8"),
    P_U(ub_128x, "FFMA=256,ULOP3.LUT=192,LOP3.LUT=32,IMAD=32,LEA.HI=32,IMMA.16832.S8.S8=8,UIADD3=1,ISETP.LE.AND=1", "IMMA.16832.S8.S8"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
