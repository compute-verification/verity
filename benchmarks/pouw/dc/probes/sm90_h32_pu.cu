// H32 on the H200 (sm_90a), the carry writers: add.cc and addc chains. Notes as sm90_h32.cu's (c=CLASS w=PATTERN:REGS);
// ../h32.py prices them per general register a lane writes, so a note counts no carry, which is a predicate a lane.
// ptxas 13.0.88 splits addc across IMAD.X and IADD3.X, so each note names both. One 1024-thread block an SM, 8
// independent chains a thread, each written from itself and its neighbour so ptxas can neither fold nor hoist. The
// uniform datapath's probes are ../../opcode_probe/probes/uniform.cu's: ptxas puts a warp-uniform chain there only in a
// loop with MMAs.
//   c_add64     add.cc + addc: a 64-bit add (IADD3 writes its word and the carry; IMAD.X or IADD3.X the high word)
//   c_add128    add.cc + 2 addc.cc + addc: a 128-bit add (IADD3.X with a carry in and out writes a word and a carry)
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

static const OpProbe PROBES[] = {
    {.name = "c_add64", .kernel = (const void*)c_add64, .tpb = TPB, .expect = "IADD3=8,IMAD.X=7,IADD3.X=1",
     .iters = 512, .note = "c=int32 w=IADD3:1,IMAD.X:1,IADD3.X:1"},
    {.name = "c_add128", .kernel = (const void*)c_add128, .tpb = TPB, .expect = "IADD3=8,IMAD.X=6,IADD3.X=18",
     .iters = 512, .note = "c=int32 w=IADD3:1,IMAD.X:1,IADD3.X:1"},
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
