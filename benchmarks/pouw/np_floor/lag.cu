// What a committed word copied into the hash input costs on the H200, against computing it (copy-floor; N-RED's models
// P, C and W). A lane makes 8 entries a trip: its position steps by a 4-bit digit of its own coin, the entry's word is
// one IMAD of the position, and every entry folds its word into a Duo-style accumulator (xor, rotl, mul). A copy keeps a
// word the lane already holds, and the IMAD is predicated off. 32 warps an SM, as branch.cu; the price is 4096 x the
// block's cycles / (iters x 32 warps x 8) units a warp entry, and a lane's is that over 32.
//   cmp    every word computed: the step, the IMAD and the fold
//   cpy    every word held: the step and the fold, no IMAD (what a copy could save, when the whole warp copies)
//   l1     a lane's lag-1 copy decided by predicates alone (model P's free copy): the step's digit is 0, so the
//          register keeps the previous word
//   l1w    the same on a digit uniform at run time
//   lagD   a lane holds its last D positions and words (D = 1, 2, 4, 8); ISETP compares the position with each, and a
//          hit copies that word: a revisit's detection at lag D
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024;
enum { CMP, CPY, L1, L1W, LAG };

__device__ __forceinline__ uint32_t coin(uint32_t x) {
  x *= 0x9E3779B1u;
  return x ^ (x >> 15);
}

__device__ __forceinline__ uint32_t fold(uint32_t h, uint32_t w) {
  h ^= w;
  h = __funnelshift_l(h, h, 5);
  return h * 0x9E3779B9u;
}

template <int V, int D = 0>
__device__ __forceinline__ void body(const OpArgs& a) {
  const uint32_t lane = threadIdx.x & 31, salt = blockIdx.x * 0x10001u;
  const uint32_t vary = threadIdx.x >> 10;  // 0 in a 1024-thread block, which ptxas can't assume
  uint32_t h = threadIdx.x, pos = lane, w = lane;
  uint32_t P[8], W[8];
#pragma unroll
  for (int k = 0; k < 8; k++) P[k] = 0xFFFFFFF0u + k, W[k] = k;
  OP_BEGIN
  OP_LOOP(it) {
    const uint32_t c = coin((uint32_t)it * 32u + lane + salt);
    const uint32_t b = V == L1W ? coin((uint32_t)it + salt + vary) : c;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      pos += (c >> (4 * k)) & 15u;
      if constexpr (V == CMP) w = pos * 0x01000193u + 0x811C9DC5u;
      if constexpr (V == L1 || V == L1W)
        if ((b >> (4 * k)) & 15u) w = pos * 0x01000193u + 0x811C9DC5u;
      if constexpr (V == LAG) {
        bool hit = false;
#pragma unroll
        for (int d = D; d >= 1; d--)
          if (pos == P[(k - d) & 7]) w = W[(k - d) & 7], hit = true;
        if (!hit) w = pos * 0x01000193u + 0x811C9DC5u;
        P[k] = pos, W[k] = w;
      }
      h = fold(h, w);
    }
  }
  OP_END
  OP_SINK(h ^ w ^ pos)
}

OP_KERNEL(cmp, TPB) { body<CMP>(a); }
OP_KERNEL(cpy, TPB) { body<CPY>(a); }
OP_KERNEL(l1, TPB) { body<L1>(a); }
OP_KERNEL(l1w, TPB) { body<L1W>(a); }
OP_KERNEL(lag1, TPB) { body<LAG, 1>(a); }
OP_KERNEL(lag2, TPB) { body<LAG, 2>(a); }
OP_KERNEL(lag4, TPB) { body<LAG, 4>(a); }
OP_KERNEL(lag8, TPB) { body<LAG, 8>(a); }

#define P(NM, EXP, NOTE) \
  {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = EXP, .iters = 4096, .note = NOTE}
static const OpProbe PROBES[] = {
    P(cmp, "BRA=1", "every word computed and folded"),
    P(cpy, "BRA=1", "every word held and folded"),
    P(l1, "BRA=1", "a lane's lag-1 copy on its own digit: predicated IMAD"),
    P(l1w, "BRA=1", "lag-1 copy on a run-time-uniform digit: predicated IMAD"),
    P(lag1, "BRA=1", "lag 1 by comparing held positions"),
    P(lag2, "BRA=1", "lag 2 by comparing held positions"),
    P(lag4, "BRA=1", "lag 4 by comparing held positions"),
    P(lag8, "BRA=1", "lag 8 by comparing held positions"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
