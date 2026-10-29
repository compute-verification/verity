// W1's census gaps on the H200 (sm_90a) that the opcode-probe harness can time from PTX: the local loads LDL.128 and
// LDL.64 (ldl.cuh), which the gate refuses as spills unless a probe allows them. Built, gated and timed as sm90_w1.cu's
// (emp.sh); notes as h32.cuh's. LDL.LU is sm90_w1_lu.cu's. PTX doesn't reach LEPC, which ptxas emits only beside a
// CALL.ABS to an external function, so no probe here times it alone.
#include "ldl.cuh"

constexpr int TPB = 1024, U = 4;

OP_KERNEL(l_ldl4, TPB) { ldl_body<U, 4>(a); }
OP_KERNEL(l_ldl2, TPB) { ldl_body<U, 2>(a); }

#define L(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 512, .note = NOTE, .allow = "LDL,STL"}
static const OpProbe PROBES[] = {
    L(l_ldl4, "LDL.128=4,LOP3.LUT=8", "c=other w=LDL.128:4,LOP3.LUT:1"),
    L(l_ldl2, "LDL.64=4,LOP3.LUT=4", "c=other w=LDL.64:2,LOP3.LUT:1"),
};
#undef L

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
