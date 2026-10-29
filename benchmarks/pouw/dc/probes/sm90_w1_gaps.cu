// W1's census gaps on the H200 (sm_90a) that the opcode-probe harness can time: the local loads, which the gate refuses
// as spills unless a probe allows them. Built, gated and timed as sm90_w1.cu's (emp.sh); notes as h32.cuh's.
//   l_ldl<V>      ld.local.v4 and .v2 (LDL.128, LDL.64): each thread's own 2·U cells of V words in local memory, written
//                 once before the timed region (STL) and loaded at fixed offsets, alternating between two sets a trip so
//                 ptxas can't hoist them; each load folded into its own chain by LOP3, which the note counts too, so the
//                 row prices every register the loop writes
// PTX reaches neither LDL.LU (ptxas drops .lu on ld.local and emits it only for its own spill reloads) nor LEPC (only
// beside a CALL.ABS to an external function), so no probe here times them alone.
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"

constexpr int TPB = 1024, U = 4;

template <int V>
DC_DEV void ldl_body(const OpArgs& a) {
  __align__(16) unsigned buf[2 * U * V];
  unsigned long long lp;
  asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(lp) : "l"(buf) : "memory");
#pragma unroll
  for (int c = 0; c < 2 * U; c++) {
    const unsigned s = threadIdx.x * 0x9E3779B9u + c * 0x85EBCA6Bu + a.mode;
    if constexpr (V == 4)
      asm volatile("st.local.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(lp + c * 16), "r"(s), "r"(s ^ 1u), "r"(s ^ 2u),
                   "r"(s ^ 3u) : "memory");
    else
      asm volatile("st.local.v2.u32 [%0], {%1,%2};" ::"l"(lp + c * 8), "r"(s), "r"(s ^ 1u) : "memory");
  }
  unsigned acc[U];
#pragma unroll
  for (int u = 0; u < U; u++) acc[u] = threadIdx.x * 0x27D4EB2Fu + u;
  OP_BEGIN
  OP_LOOP(it) {
    const unsigned long long set = ((unsigned long long)it & 1ull) * (U * V * 4);
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned long long p = lp + set + u * V * 4;
      if constexpr (V == 4) {
        unsigned w, x, y, z;
        asm volatile("ld.local.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w), "=r"(x), "=r"(y), "=r"(z) : "l"(p) : "memory");
        acc[u] ^= w ^ x ^ y ^ z;
      } else {
        unsigned w, x;
        asm volatile("ld.local.v2.u32 {%0,%1}, [%2];" : "=r"(w), "=r"(x) : "l"(p) : "memory");
        acc[u] ^= w ^ x;
      }
    }
  }
  OP_END
  unsigned t = 0;
#pragma unroll
  for (int u = 0; u < U; u++) t ^= acc[u];
  OP_SINK(t)
}
OP_KERNEL(l_ldl4, TPB) { ldl_body<4>(a); }
OP_KERNEL(l_ldl2, TPB) { ldl_body<2>(a); }

#define L(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 512, .note = NOTE, .allow = "LDL,STL"}
static const OpProbe PROBES[] = {
    L(l_ldl4, "LDL.128=4,LOP3.LUT=8", "c=other w=LDL.128:4,LOP3.LUT:1"),
    L(l_ldl2, "LDL.64=4,LOP3.LUT=4", "c=other w=LDL.64:2,LOP3.LUT:1"),
};
#undef L

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
