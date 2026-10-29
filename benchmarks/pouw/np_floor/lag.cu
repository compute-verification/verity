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
// `hw_*` runs a form in WL side warps beside the lead's tensor kernel (race.cu's wa_s8_n256, as copy.cu's hw_*), an until
// loop, so what the selection and the copy cost there is the wgmma's lost time a lane-entry; `honest_we` is the wgmma
// beside WL idle warps.
#include "../dc/probes/wg_common.cuh"
#include "../opcode_probe/probe.cuh"

constexpr int TPB = 1024, WG = 8, WL = 4;
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
struct Entries {
  uint32_t lane, salt, vary, h, pos, w;
  uint32_t P[8], W[8];
  __device__ __forceinline__ Entries() {
    lane = threadIdx.x & 31;
    salt = blockIdx.x * 0x10001u;
    vary = threadIdx.x >> 10;  // 0 in a block of at most 1024 threads, which ptxas can't assume
    h = threadIdx.x, pos = lane, w = lane;
#pragma unroll
    for (int k = 0; k < 8; k++) P[k] = 0xFFFFFFF0u + k, W[k] = k;
  }
  __device__ __forceinline__ void trip(uint32_t it) {
    const uint32_t c = coin(it * 32u + lane + salt);
    const uint32_t b = V == L1W ? coin(it + salt + vary) : c;
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
  __device__ __forceinline__ uint32_t sink() const { return h ^ w ^ pos; }
};

template <int V, int D = 0>
__device__ __forceinline__ void body(const OpArgs& a) {
  Entries<V, D> e;
  OP_BEGIN
  OP_LOOP(it) { e.trip((uint32_t)it); }
  OP_END
  OP_SINK(e.sink())
}

// the wgmma warpgroups (warps 0..WG-1), and the form in warps WG..WG+WL-1 (SIDE false: idle)
template <int V, int D, bool SIDE>
__device__ __forceinline__ void beside_wg(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles<F_s8_n256::TYPE>(sm, a.mode);
  const int wp = threadIdx.x >> 5;
  F_s8_n256::Acc d[F_s8_n256::NA] = {};
  Entries<V, D> e;
  OP_BEGIN_STOP
  if (wp < WG) {
    OP_LOOP(it) {
      group4<F_s8_n256>(d, t);
      wgmma_wait<1>();
    }
    wgmma_wait<0>();
    wgmma_hold(d);
    OP_DONE
  } else if (SIDE) {
    OP_UNTIL(n) { e.trip((uint32_t)n); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK(fold_acc(d) ^ threadIdx.x ^ e.sink())
}

OP_KERNEL(cmp, TPB) { body<CMP>(a); }
OP_KERNEL(cpy, TPB) { body<CPY>(a); }
OP_KERNEL(l1, TPB) { body<L1>(a); }
OP_KERNEL(l1w, TPB) { body<L1W>(a); }
OP_KERNEL(lag1, TPB) { body<LAG, 1>(a); }
OP_KERNEL(lag2, TPB) { body<LAG, 2>(a); }
OP_KERNEL(lag4, TPB) { body<LAG, 4>(a); }
OP_KERNEL(lag8, TPB) { body<LAG, 8>(a); }
#define TB_WE ((WG + WL) * 32)
OP_KERNEL(hw_cmp, TB_WE) { beside_wg<CMP, 0, true>(a); }
OP_KERNEL(hw_l1, TB_WE) { beside_wg<L1, 0, true>(a); }
OP_KERNEL(hw_lag1, TB_WE) { beside_wg<LAG, 1, true>(a); }
OP_KERNEL(hw_lag2, TB_WE) { beside_wg<LAG, 2, true>(a); }
OP_KERNEL(honest_we, TB_WE) { beside_wg<CMP, 0, false>(a); }

#define P(NM, EXP, NOTE) \
  {.name = #NM, .kernel = (const void*)NM, .tpb = TPB, .expect = EXP, .iters = 4096, .note = NOTE}
#define IGMMA "IGMMA.64x256x32.S8.S8*=4"
// beside the wgmma: a warp's entries an SM-clock are its SHF.L.W (one a fold) an SM-clock, 32 lane-entries each
#define PW(NM, EXP, NOTE)                                                                                         \
  {.name = "hw_" #NM, .kernel = (const void*)hw_##NM, .tpb = TB_WE, .expect = IGMMA ";" EXP,                       \
   .counted = "SHF.L.W.U32.HI", .warps = "8;4", .words = 32, .iters = 2048, .note = NOTE, .until = 2}
// ptxas 13.0: the fold is LOP3, SHF.L.W and IMAD; the word is one IMAD. l1 and l1w predicate the word's IMAD from a LOP3
// on the digit. lag1 reduces to l1 (ptxas sees the held position is the previous one), and lag2 to lag8 select the
// held word with SEL, one a lag an entry, after about two ISETP a lag.
static const OpProbe PROBES[] = {
    P(cmp, "SHF.L.W.U32.HI=8,IMAD=16,LOP3.LUT=16", "every word computed and folded"),
    P(cpy, "SHF.L.W.U32.HI=8,IMAD=8,LOP3.LUT=16", "every word held and folded"),
    P(l1, "SHF.L.W.U32.HI=8,IMAD=16,LOP3.LUT=23", "a lane's lag-1 copy on its own digit: predicated IMAD"),
    P(l1w, "SHF.L.W.U32.HI=8,IMAD=16,LOP3.LUT=25", "lag-1 copy on a run-time-uniform digit: predicated IMAD"),
    P(lag1, "SHF.L.W.U32.HI=8,IMAD=16,ISETP*=3,LOP3.LUT=16", "lag 1 by comparing held positions"),
    P(lag2, "SHF.L.W.U32.HI=8,IMAD=16,ISETP*=19,SEL=9", "lag 2 by comparing held positions"),
    P(lag4, "SHF.L.W.U32.HI=8,IMAD=18,ISETP*=45,SEL=32", "lag 4 by comparing held positions"),
    P(lag8, "SHF.L.W.U32.HI=8,IMAD=18,ISETP*=109,SEL=64", "lag 8 by comparing held positions"),
    {.name = "honest_we", .kernel = (const void*)honest_we, .tpb = TB_WE, .expect = IGMMA, .warps = "8", .iters = 2048,
     .note = "the lead's wgmma beside 4 idle warps"},
    PW(cmp, "SHF.L.W.U32.HI=8,IMAD=16,LOP3.LUT=16", "cmp beside the lead's wgmma"),
    PW(l1, "SHF.L.W.U32.HI=8,IMAD=16,LOP3.LUT=23", "l1, model P's copy, beside the lead's wgmma"),
    PW(lag1, "SHF.L.W.U32.HI=8,IMAD=16,ISETP*=3,LOP3.LUT=16", "lag1 beside the lead's wgmma"),
    PW(lag2, "SHF.L.W.U32.HI=8,IMAD=16,ISETP*=19,SEL=9", "lag2's ISETP and SEL selection beside the lead's wgmma"),
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
