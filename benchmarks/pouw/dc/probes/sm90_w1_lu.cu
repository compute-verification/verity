// LDL.LU.128 and LDL.LU.64 on the H200 (sm_90a), the two census gaps PTX can't reach: ptxas drops `.lu` on ld.local
// and emits LDL.LU only for its own spill reloads. These are sm90_w1_gaps.cu's local loads (ldl.cuh), which ptxas
// lowers to LDL.128 and LDL.64; ldl_lu.sh then sets the last-use hint on them (ldl_lu.py), and the gate checks the
// patched binary against the counts below. Unpatched, the gate refuses both.
#include "ldl.cuh"

constexpr int TPB = 1024, U = 4;

OP_KERNEL(l_ldllu4, TPB) { ldl_body<U, 4>(a); }
OP_KERNEL(l_ldllu2, TPB) { ldl_body<U, 2>(a); }

#define L(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 512, .note = NOTE, .allow = "LDL,STL"}
static const OpProbe PROBES[] = {
    L(l_ldllu4, "LDL.LU.128=4,LOP3.LUT=8", "c=other w=LDL.LU.128:4,LOP3.LUT:1"),
    L(l_ldllu2, "LDL.LU.64=4,LOP3.LUT=4", "c=other w=LDL.LU.64:2,LOP3.LUT:1"),
};
#undef L

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
