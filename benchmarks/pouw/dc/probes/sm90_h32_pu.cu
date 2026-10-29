// H32 on the H200 (sm_90a), the writers beside a lane's general registers: carry writers and the uniform datapath.
// Notes as sm90_h32.cu's (c=CLASS w=PATTERN:REGS); ../h32.py prices them per general register a lane writes, so a
// note counts no carry, which is a predicate a lane. A uniform instruction writes one word a warp: its note has no
// class, and its price is 4,096 over its rate (rows.jsonl). One 1024-thread block an SM, 8 independent chains a
// thread, each written from itself and its neighbour so ptxas can neither fold nor hoist.
//   c_add64     add.cc + addc: a 64-bit add (IADD3 writes its word and the carry, IADD3.X the high word)
//   c_add128    add.cc + 2 addc.cc + addc: a 128-bit add (IADD3.X with a carry in and out writes a word and a carry)
//   u_imad, u_lop3, u_iadd   mad.lo, lop3 and add on warp-uniform words (UIMAD, ULOP3, UIADD3: one word a warp)
#include "../../opcode_probe/probe.cuh"

constexpr int TPB = 1024, U = 8;

OP_KERNEL(c_add64, TPB) {
  unsigned lo[U], hi[U];
#pragma unroll
  for (int u = 0; u < U; u++) {
    lo[u] = threadIdx.x * 0x9E3779B9u + u * 0x85EBCA6Bu;
    hi[u] = lo[u] ^ 0x27D4EB2Fu;
  }
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int v = (u + 1) % U;
      asm volatile("add.cc.u32 %0,%0,%2;\n\taddc.u32 %1,%1,%3;" : "+r"(lo[u]), "+r"(hi[u]) : "r"(lo[v]), "r"(hi[v]));
    }
  }
  OP_END
  unsigned s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= lo[u] + hi[u];
  OP_SINK(s)
}

OP_KERNEL(c_add128, TPB) {
  unsigned w[U][4];
#pragma unroll
  for (int u = 0; u < U; u++)
#pragma unroll
    for (int j = 0; j < 4; j++) w[u][j] = threadIdx.x * 0x9E3779B9u + (4 * u + j) * 0x85EBCA6Bu;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int v = (u + 1) % U;
      asm volatile("add.cc.u32 %0,%0,%4;\n\taddc.cc.u32 %1,%1,%5;\n\taddc.cc.u32 %2,%2,%6;\n\taddc.u32 %3,%3,%7;"
                   : "+r"(w[u][0]), "+r"(w[u][1]), "+r"(w[u][2]), "+r"(w[u][3])
                   : "r"(w[v][0]), "r"(w[v][1]), "r"(w[v][2]), "r"(w[v][3]));
    }
  }
  OP_END
  unsigned s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= w[u][0] + w[u][1] + w[u][2] + w[u][3];
  OP_SINK(s)
}

// A chain of warp-uniform words: blockIdx and the loop's trip, which ptxas keeps on the uniform datapath.
template <int X>
__device__ __forceinline__ void uni_body(const OpArgs& a) {
  unsigned v[U];
#pragma unroll
  for (int u = 0; u < U; u++) v[u] = blockIdx.x * 0x9E3779B9u + u * 0x85EBCA6Bu;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int t = (u + 1) % U;
      if constexpr (X == 0) asm volatile("mad.lo.u32 %0,%0,%1,0x9E3779B9;" : "+r"(v[u]) : "r"(v[t]));
      if constexpr (X == 1) asm volatile("lop3.b32 %0,%0,%1,0x85EBCA6B,0x96;" : "+r"(v[u]) : "r"(v[t]));
      if constexpr (X == 2) asm volatile("add.u32 %0,%0,%1;" : "+r"(v[u]) : "r"(v[t]));
    }
  }
  OP_END
  unsigned s = threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++) s ^= v[u];
  OP_SINK(s)
}
OP_KERNEL(u_imad, TPB) { uni_body<0>(a); }
OP_KERNEL(u_lop3, TPB) { uni_body<1>(a); }
OP_KERNEL(u_iadd, TPB) { uni_body<2>(a); }

static const OpProbe PROBES[] = {
    {.name = "c_add64", .kernel = (const void*)c_add64, .tpb = TPB, .expect = "IADD3=8,IADD3.X=8", .iters = 512,
     .note = "c=int32 w=IADD3:1,IADD3.X:1"},
    {.name = "c_add128", .kernel = (const void*)c_add128, .tpb = TPB, .expect = "IADD3=8,IADD3.X=24", .iters = 512,
     .note = "c=int32 w=IADD3:1,IADD3.X:1"},
    {.name = "u_imad", .kernel = (const void*)u_imad, .tpb = TPB, .expect = "UIMAD=8", .iters = 512,
     .note = "uniform UIMAD:1"},
    {.name = "u_lop3", .kernel = (const void*)u_lop3, .tpb = TPB, .expect = "ULOP3.LUT=8", .iters = 512,
     .note = "uniform ULOP3.LUT:1"},
    {.name = "u_iadd", .kernel = (const void*)u_iadd, .tpb = TPB, .expect = "UIADD3=8", .iters = 512,
     .note = "uniform UIADD3:1"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
