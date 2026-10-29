// W1's census gaps on the H200 (sm_90a) that the opcode-probe harness can time: the local loads, which the gate refuses
// as spills unless a probe allows them, and LEPC. Built, gated and timed as sm90_w1.cu's (emp.sh); notes as h32.cuh's.
//   l_ldl<V>, l_ldllu<V>  ld.local.v4 and .v2 (LDL.128, LDL.64), and the same with .lu (LDL.LU.128, LDL.LU.64): each
//                 thread's own 2·U cells of V words in local memory, written once before the timed region (STL) and
//                 loaded at fixed offsets, alternating between two sets a trip so ptxas can't hoist them; each load
//                 folded into its own chain by LOP3, which the note counts too, so the row prices every register the
//                 loop writes
//   c_lepc        an indirect call a trip through a two-entry table of device functions: ptxas sets the return address
//                 with LEPC (2 registers a lane). The loop holds the call, the callee and its return, so its price is
//                 an upper bound of LEPC's
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"

constexpr int TPB = 1024, U = 4;

template <int V, bool LU>
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
        if constexpr (LU)
          asm volatile("ld.local.lu.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w), "=r"(x), "=r"(y), "=r"(z) : "l"(p) : "memory");
        else
          asm volatile("ld.local.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(w), "=r"(x), "=r"(y), "=r"(z) : "l"(p) : "memory");
        acc[u] ^= w ^ x ^ y ^ z;
      } else {
        unsigned w, x;
        if constexpr (LU)
          asm volatile("ld.local.lu.v2.u32 {%0,%1}, [%2];" : "=r"(w), "=r"(x) : "l"(p) : "memory");
        else
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
OP_KERNEL(l_ldl4, TPB) { ldl_body<4, false>(a); }
OP_KERNEL(l_ldl2, TPB) { ldl_body<2, false>(a); }
OP_KERNEL(l_ldllu4, TPB) { ldl_body<4, true>(a); }
OP_KERNEL(l_ldllu2, TPB) { ldl_body<2, true>(a); }

__device__ __noinline__ unsigned lepc_a(unsigned x) { return x * 0x9E3779B9u + 1u; }
__device__ __noinline__ unsigned lepc_b(unsigned x) { return (x ^ 0x7F4A7C15u) + 3u; }
__device__ unsigned (*lepc_table[2])(unsigned) = {lepc_a, lepc_b};

OP_KERNEL(c_lepc, TPB) {
  unsigned x = threadIdx.x ^ (unsigned)a.mode;
  OP_BEGIN
  OP_LOOP(it) { x = lepc_table[(it ^ a.mode) & 1](x); }
  OP_END
  OP_SINK(x)
}

#define L(NAME, PAT, NOTE) \
  {.name = #NAME, .kernel = (const void*)NAME, .tpb = TPB, .expect = PAT, .iters = 512, .note = NOTE, .allow = "LDL,STL"}
static const OpProbe PROBES[] = {
    L(l_ldl4, "LDL.128=4,LOP3.LUT=8", "c=other w=LDL.128:4,LOP3.LUT:1"),
    L(l_ldl2, "LDL.64=4,LOP3.LUT=4", "c=other w=LDL.64:2,LOP3.LUT:1"),
    L(l_ldllu4, "LDL.LU.128=4,LOP3.LUT=8", "c=other w=LDL.LU.128:4,LOP3.LUT:1"),
    L(l_ldllu2, "LDL.LU.64=4,LOP3.LUT=4", "c=other w=LDL.LU.64:2,LOP3.LUT:1"),
    L(c_lepc, "LEPC=1", "c=other w=LEPC:2"),
};
#undef L

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
