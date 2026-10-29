// The local-load body sm90_w1_gaps.cu and sm90_w1_lu.cu time: ld.local.v4 or .v2 (LDL.128, LDL.64), each thread's own
// 2·U cells of V words in local memory, written once before the timed region (STL) and loaded at fixed offsets,
// alternating between two sets a trip so ptxas can't hoist them; each load folded into its own chain by LOP3, which the
// note counts too, so the row prices every register the loop writes.
#pragma once
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"

template <int U, int V>
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
