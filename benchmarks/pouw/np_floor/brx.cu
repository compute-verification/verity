// A coin-indexed jump (adaptive-dw): BRX through a 16-entry table, 2 a trip, each reading 4 bits of a coin word that
// changes every trip, each entry one add of its own constant. As branch.cu (whose `arm` is the work without the
// decisions): 32 warps an SM, and the price of a decision is 4096 x the block's cycles / (iters x 32 warps x 2) units.
//   brxu   `brx.idx.uni` on a coin ptxas can see is warp-uniform
//   brxv   `brx.idx.uni` on a vector-register coin, uniform at run time
//   brxw   `brx.idx` on the same coin: ptxas must allow divergence, the warp doesn't diverge
//   brxd   `brx.idx` on each lane's own 4 bits: up to 16 targets a warp
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024;
enum { U, V, W, D };

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}

#define ARMX(n, k) "L" #n ":\n\tadd.u32 %0, %0, " #k ";\n\tbra.uni J;\n"
#define BRX(UNI, SH)                                                                                              \
  "{\n\t.reg .b32 i;\n\tshr.b32 i, %1, " #SH ";\n\tand.b32 i, i, 15;\n\t"                                          \
  "ts: .branchtargets L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15;\n\t"                  \
  "brx.idx" UNI " i, ts;\n" ARMX(0, 0x3c6ef372) ARMX(1, 0x0a54ff53) ARMX(2, 0x510e527f) ARMX(3, 0x1b05688c)        \
      ARMX(4, 0x1f83d9ab) ARMX(5, 0x5be0cd19) ARMX(6, 0x6a09e667) ARMX(7, 0x3b67ae85) ARMX(8, 0x428a2f98)          \
          ARMX(9, 0x71374491) ARMX(10, 0x35c0fbcf) ARMX(11, 0x69b5dba5) ARMX(12, 0x3956c25b) ARMX(13, 0x59f111f1)  \
              ARMX(14, 0x123f82a4) ARMX(15, 0x2b1c5ed5) "J:\n\t}"

template <int F>
__device__ __forceinline__ void body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31, salt = blockIdx.x * 0x10001u;
  const uint32_t vary = threadIdx.x >> 10;  // 0 in a 1024-thread block, which ptxas can't assume
  uint32_t x = threadIdx.x;
  OP_BEGIN
  OP_LOOP(it) {
    const uint32_t u = (uint32_t)it;
    const uint32_t c = F == D ? coin(u * 32u + lane + salt) : F == U ? coin(u + salt) : coin(u + salt + vary);
    if constexpr (F == U || F == V) {
      asm volatile(BRX(".uni", 3) : "+r"(x) : "r"(c));
      asm volatile(BRX(".uni", 17) : "+r"(x) : "r"(c));
    } else {
      asm volatile(BRX("", 3) : "+r"(x) : "r"(c));
      asm volatile(BRX("", 17) : "+r"(x) : "r"(c));
    }
  }
  OP_END
  OP_SINK(x)
}

OP_KERNEL(brxu, TPB) { body<U>(a); }
OP_KERNEL(brxv, TPB) { body<V>(a); }
OP_KERNEL(brxw, TPB) { body<W>(a); }
OP_KERNEL(brxd, TPB) { body<D>(a); }

#define P(NM, NOTE) {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = "BRX*=2", .iters = 4096, .note = NOTE}
static const OpProbe PROBES[] = {
    P(brxu, "brx.idx.uni, coin ptxas sees uniform"),
    P(brxv, "brx.idx.uni on a vector coin"),
    P(brxw, "brx.idx on a vector coin, uniform at run time"),
    P(brxd, "brx.idx on each lane's 4 bits: divergent"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
