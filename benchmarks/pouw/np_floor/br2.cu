// Two-way decisions on coin bits that jump (adaptive-dw), beside branch.cu's if-converted ones. As branch.cu: 32 warps
// an SM, a coin word that changes every trip, and the price of a decision is 4096 x the block's cycles / (iters x 32
// warps x decisions a trip) units a warp decision, a lane-bit that over 32.
//   x2u, x2w, x2d   8 decisions a trip, each a 2-entry `brx.idx` on one bit (arms: add the coin, or XOR a constant): on
//                   a coin ptxas sees uniform (.uni), a vector coin uniform at run time, each lane's own bit
//   lu, lw, ld      4 decisions a trip, each `@p bra` between two 8-instruction arms (long enough that ptxas keeps the
//                   branch): coins as above
//   larm            the 4 long arms unconditionally: lu's work without the decisions
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024;
enum { X2U, X2W, X2D, LU, LW, LD, LARM };

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}

#define X2(UNI, SH, K)                                                                                             \
  "{\n\t.reg .b32 i;\n\tshr.b32 i, %1, " #SH ";\n\tand.b32 i, i, 1;\n\tts: .branchtargets L0, L1;\n\t"              \
  "brx.idx" UNI " i, ts;\nL0:\n\txor.b32 %0, %0, " #K ";\n\tbra.uni J;\nL1:\n\tadd.u32 %0, %0, %1;\nJ:\n\t}"
// an 8-instruction arm: add the coin and XOR a constant, 4 times
#define ARM8(A, B, C, D) \
  "add.u32 %0, %0, %1;\n\txor.b32 %0, %0, " #A ";\n\tadd.u32 %0, %0, %1;\n\txor.b32 %0, %0, " #B ";\n\t"      \
  "add.u32 %0, %0, %1;\n\txor.b32 %0, %0, " #C ";\n\tadd.u32 %0, %0, %1;\n\txor.b32 %0, %0, " #D ";\n\t"
#define LONG(UNI, BIT, A, B, C, D, E, F, G, H)                                                                     \
  "{\n\t.reg .pred p;\n\t.reg .b32 t;\n\tand.b32 t, %1, " #BIT ";\n\tsetp.ne.b32 p, t, 0;\n\t@p bra" UNI " A;\n\t" \
  ARM8(A, B, C, D) "bra.uni J;\nA:\n\t" ARM8(E, F, G, H) "J:\n\t}"

template <int F>
__device__ __forceinline__ void body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31, salt = blockIdx.x * 0x10001u;
  const uint32_t vary = threadIdx.x >> 10;  // 0 in a 1024-thread block, which ptxas can't assume
  uint32_t x = threadIdx.x;
  OP_BEGIN
  OP_LOOP(it) {
    const uint32_t u = (uint32_t)it;
    const uint32_t c = F == X2D || F == LD ? coin(u * 32u + lane + salt)
                       : F == X2U || F == LU || F == LARM ? coin(u + salt)
                                                          : coin(u + salt + vary);
    if constexpr (F == X2U) {
      asm volatile(X2(".uni", 0, 0x3c6ef372) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 4, 0x0a54ff53) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 8, 0x510e527f) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 12, 0x1b05688c) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 16, 0x1f83d9ab) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 20, 0x5be0cd19) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 24, 0x6a09e667) : "+r"(x) : "r"(c));
      asm volatile(X2(".uni", 28, 0x3b67ae85) : "+r"(x) : "r"(c));
    }
    if constexpr (F == X2W || F == X2D) {
      asm volatile(X2("", 0, 0x3c6ef372) : "+r"(x) : "r"(c));
      asm volatile(X2("", 4, 0x0a54ff53) : "+r"(x) : "r"(c));
      asm volatile(X2("", 8, 0x510e527f) : "+r"(x) : "r"(c));
      asm volatile(X2("", 12, 0x1b05688c) : "+r"(x) : "r"(c));
      asm volatile(X2("", 16, 0x1f83d9ab) : "+r"(x) : "r"(c));
      asm volatile(X2("", 20, 0x5be0cd19) : "+r"(x) : "r"(c));
      asm volatile(X2("", 24, 0x6a09e667) : "+r"(x) : "r"(c));
      asm volatile(X2("", 28, 0x3b67ae85) : "+r"(x) : "r"(c));
    }
    if constexpr (F == LU) {
      asm volatile(LONG(".uni", 0x1, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18) : "+r"(x) : "r"(c));
      asm volatile(LONG(".uni", 0x100, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28) : "+r"(x) : "r"(c));
      asm volatile(LONG(".uni", 0x10000, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38) : "+r"(x) : "r"(c));
      asm volatile(LONG(".uni", 0x1000000, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48) : "+r"(x) : "r"(c));
    }
    if constexpr (F == LW || F == LD) {
      asm volatile(LONG("", 0x1, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18) : "+r"(x) : "r"(c));
      asm volatile(LONG("", 0x100, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28) : "+r"(x) : "r"(c));
      asm volatile(LONG("", 0x10000, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38) : "+r"(x) : "r"(c));
      asm volatile(LONG("", 0x1000000, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48) : "+r"(x) : "r"(c));
    }
    if constexpr (F == LARM) {
      asm volatile(ARM8(0x11, 0x12, 0x13, 0x14) : "+r"(x) : "r"(c));
      asm volatile(ARM8(0x25, 0x26, 0x27, 0x28) : "+r"(x) : "r"(c));
      asm volatile(ARM8(0x31, 0x32, 0x33, 0x34) : "+r"(x) : "r"(c));
      asm volatile(ARM8(0x45, 0x46, 0x47, 0x48) : "+r"(x) : "r"(c));
    }
  }
  OP_END
  OP_SINK(x)
}

OP_KERNEL(x2u, TPB) { body<X2U>(a); }
OP_KERNEL(x2w, TPB) { body<X2W>(a); }
OP_KERNEL(x2d, TPB) { body<X2D>(a); }
OP_KERNEL(lu, TPB) { body<LU>(a); }
OP_KERNEL(lw, TPB) { body<LW>(a); }
OP_KERNEL(ld, TPB) { body<LD>(a); }
OP_KERNEL(larm, TPB) { body<LARM>(a); }

#define P(NM, EXP, NOTE) \
  {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = EXP, .iters = 4096, .note = NOTE}
static const OpProbe PROBES[] = {
    P(x2u, "BRX=8", "2-entry brx.idx.uni, coin ptxas sees uniform"),
    P(x2w, "BRX=8", "2-entry brx.idx on a vector coin, uniform at run time"),
    P(x2d, "BRX=8", "2-entry brx.idx on each lane's bit: divergent"),
    P(lu, "BRA=9", "bra.uni between 8-instruction arms, coin ptxas sees uniform"),
    P(lw, "BRA=9", "bra between 8-instruction arms, vector coin uniform at run time"),
    P(ld, "BRA=9", "bra between 8-instruction arms, each lane's bit: divergent"),
    P(larm, "BRA=1", "the long arms alone"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
