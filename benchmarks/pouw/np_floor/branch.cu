// What a decision on a coin bit costs on the H200 (adaptive-dw): a loop of 8 decisions a trip, each reading one bit of
// a coin word that changes every trip, each choosing between two one-instruction arms (add the coin, or XOR a
// constant), in 32 warps an SM so latency hides. Every form computes the same coin word a trip, so the difference from
// `arm` is the decision's own cost. probe.py's rate isn't the price here: the price of a decision is 4096 x the block's
// cycles / (iters x 32 warps x 8) units a warp decision, and a lane-bit is that over 32.
//   arm    the 8 arms unconditionally (alternately add and XOR): the work without the decisions
//   bru    `@p bra.uni` on a coin ptxas can see is warp-uniform (computed from the trip and the block)
//   brv    `@p bra.uni` on a coin in a vector register, uniform at run time
//   brw    `@p bra` on the same coin: ptxas must allow divergence, the warp doesn't diverge
//   brd    `@p bra` on each lane's own coin bit: the warp diverges and runs both arms
//   sel    both arms, then `selp`
//   lutb   4 shared-memory lookups a trip, each indexed by 8 bits of a warp-uniform coin (a broadcast LDS)
//   lutl   the same, each lane its own coin and its own column of a 256 x 32 table (no bank conflict)
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024;
enum { ARM, BRU, BRV, BRW, BRD, SEL, LUTB, LUTL };

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}

#define BR(UNI)                                                                                                     \
  "{\n\t.reg .pred p;\n\t.reg .b32 t;\n\tand.b32 t, %1, %2;\n\tsetp.ne.b32 p, t, 0;\n\t@p bra" UNI " A;\n\t"         \
  "xor.b32 %0, %0, %3;\n\tbra.uni J;\nA:\n\tadd.u32 %0, %0, %1;\nJ:\n\t}"
#define SL                                                                                                          \
  "{\n\t.reg .pred p;\n\t.reg .b32 t, u, v;\n\tand.b32 t, %1, %2;\n\tsetp.ne.b32 p, t, 0;\n\tadd.u32 u, %0, %1;\n\t" \
  "xor.b32 v, %0, %3;\n\tselp.b32 %0, u, v, p;\n\t}"

template <int V, int Q, uint32_t BIT, uint32_t K>
__device__ __forceinline__ void decide(uint32_t& x, uint32_t c) {
  if constexpr (V == ARM && (Q & 1)) asm volatile("xor.b32 %0, %0, %3;" : "+r"(x) : "r"(c), "n"(BIT), "n"(K));
  if constexpr (V == ARM && !(Q & 1)) asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(c), "n"(BIT), "n"(K));
  if constexpr (V == BRU || V == BRV) asm volatile(BR(".uni") : "+r"(x) : "r"(c), "n"(BIT), "n"(K));
  if constexpr (V == BRW || V == BRD) asm volatile(BR("") : "+r"(x) : "r"(c), "n"(BIT), "n"(K));
  if constexpr (V == SEL) asm volatile(SL : "+r"(x) : "r"(c), "n"(BIT), "n"(K));
}

template <int V>
__device__ __forceinline__ void body(const OpArgs& a) {
  __shared__ uint32_t lut[256 * 32];
  const uint32_t lane = threadIdx.x & 31, salt = blockIdx.x * 0x10001u;
  const uint32_t vary = threadIdx.x >> 10;  // 0 in a 1024-thread block, which ptxas can't assume
  if constexpr (V == LUTB || V == LUTL) {
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) lut[i] = coin((uint32_t)i);
  }
  uint32_t x = threadIdx.x;
  OP_BEGIN
  OP_LOOP(it) {
    const uint32_t u = (uint32_t)it;
    const uint32_t c = V == BRD || V == LUTL ? coin(u * 32u + lane + salt)
                       : V == BRU || V == LUTB ? coin(u + salt)
                                               : coin(u + salt + vary);
    if constexpr (V == LUTB || V == LUTL) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const uint32_t i = (c >> (8 * k)) & 255u;
        x ^= lut[V == LUTL ? i * 32u + lane : i];
      }
    } else {
      decide<V, 0, 0x00000001u, 0x3c6ef372u>(x, c);
      decide<V, 1, 0x00000010u, 0xa54ff53au>(x, c);
      decide<V, 2, 0x00000100u, 0x510e527fu>(x, c);
      decide<V, 3, 0x00001000u, 0x9b05688cu>(x, c);
      decide<V, 4, 0x00010000u, 0x1f83d9abu>(x, c);
      decide<V, 5, 0x00100000u, 0x5be0cd19u>(x, c);
      decide<V, 6, 0x01000000u, 0x6a09e667u>(x, c);
      decide<V, 7, 0x10000000u, 0xbb67ae85u>(x, c);
    }
  }
  OP_END
  OP_SINK(x)
}

OP_KERNEL(arm, TPB) { body<ARM>(a); }
OP_KERNEL(bru, TPB) { body<BRU>(a); }
OP_KERNEL(brv, TPB) { body<BRV>(a); }
OP_KERNEL(brw, TPB) { body<BRW>(a); }
OP_KERNEL(brd, TPB) { body<BRD>(a); }
OP_KERNEL(sel, TPB) { body<SEL>(a); }
OP_KERNEL(lutb, TPB) { body<LUTB>(a); }
OP_KERNEL(lutl, TPB) { body<LUTL>(a); }

#define P(NM, EXP, NOTE) \
  {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = EXP, .iters = 4096, .note = NOTE}
// ptxas 13.0 if-converts every two-way form here: a LOP3 that writes a predicate (bru, brv) or R2P, two coin bits to
// two predicates (brw, brd, sel), then both arms under the predicate and its complement. No BRA is left but the loop's,
// so these price a predicated decision; brx.cu's BRX and br2.cu's long arms are the forms that jump.
static const OpProbe PROBES[] = {
    P(arm, "IMAD.IADD=4,LOP3*=5", "the arms alone"),
    P(bru, "IMAD.IADD=8,LOP3*=17", "bra.uni, coin ptxas sees uniform: if-converted, LOP3 to P"),
    P(brv, "IMAD.IADD=8,LOP3*=17", "bra.uni on a vector coin: if-converted, LOP3 to P"),
    P(brw, "R2P=3,IMAD.IADD=8,LOP3*=11", "bra on a vector coin: if-converted, R2P"),
    P(brd, "R2P=3,IMAD.IADD=8,LOP3*=11", "bra on each lane's coin bit: if-converted, R2P"),
    P(sel, "R2P=4,IMAD.IADD=8,LOP3*=9", "both arms and selp: predicated, R2P"),
    P(lutb, "LDS*=4,BRA=1", "4 broadcast lookups, 8 coin bits each"),
    P(lutl, "LDS*=4,BRA=1", "4 lane-column lookups, 8 coin bits each"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
