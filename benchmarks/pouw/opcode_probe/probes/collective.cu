// The warp collectives' rates on sm_120 and sm_90, for the coverage table's REDUX and VOTE rows (machine_model/data/
// cover/rows.toml). Joint points: rows.jsonl only, no census line. Each warp runs 8 independent chains, every
// collective's input the last one's output mixed with the lane id: a REDUX's chain is a move out of its uniform register
// and an add (CHAIN_*), a ballot's a LOP3 (so its mixes hold Q + 8).
//   redux_add, redux_min, redux_xor   REDUX.SUM, .MIN, .XOR of 32 words into a uniform register
//   vote_ballot, vote_all             VOTE.ANY (the ballot word) and VOTE.ALL (a predicate)
//   mix_redux_lop3_Q, mix_vote_lop3_Q every warp 8 of the collective and Q LOP3 on chains of their own
//   mix_redux_match                   8 REDUX.SUM and 8 MATCH.ANY (1 class, its fastest): whether they share a unit
// A fold of 32 words with a 3-input add is 16 alu lanes, so a collective is dominated by `lane`s on `alu` if the card
// spends at least that: alone, LOP3 runs at 2 warp instructions a SM-clock (64 alu lanes), and beside the collective
// at r_C the row is dominated if the alu instructions' rate plus r_C / 2 is at most 2 at every point. ELECT isn't here:
// with a mask that doesn't depend on the chain ptxas merges ELECTs, and with one that does it adds a REDUX.OR and a
// branch.
#include "../probe.cuh"

constexpr int TPB = 1024;

template <int OP>
__device__ __forceinline__ unsigned redux(unsigned v) {
  if constexpr (OP == 0) return __reduce_add_sync(~0u, v);
  else if constexpr (OP == 1) return __reduce_min_sync(~0u, v);
  else return __reduce_xor_sync(~0u, v);
}

template <int OP, int Q>
__device__ __forceinline__ void reduxes(const OpArgs& a) {
  const unsigned lane = threadIdx.x & 31;
  unsigned v[8];
  uint32_t w[8];
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
#pragma unroll
  for (int u = 0; u < 8; u++) {
    v[u] = (unsigned)a.iters * 977u + lane * 8u + (unsigned)u;
    w[u] = threadIdx.x * 8u + u + 1u;
  }
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      v[u] = redux<OP>(v[u]) + lane;
#pragma unroll
      for (int j = 0; j < Q / 8; j++)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[(u * Q / 8 + j) & 7]) : "r"(k1), "r"(k2));
    }
  }
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= v[u] ^ w[u];
  OP_SINK(s)
}

template <int Q>
__device__ __forceinline__ void ballots(const OpArgs& a) {
  const unsigned lane = threadIdx.x & 31;
  unsigned v[8];
  uint32_t w[8];
  const uint32_t k1 = 0x3C013C01u ^ (uint32_t)a.mode, k2 = 0x00010001u ^ (uint32_t)(a.mode << 3);
#pragma unroll
  for (int u = 0; u < 8; u++) {
    v[u] = (unsigned)a.iters * 977u + lane * 8u + (unsigned)u;
    w[u] = threadIdx.x * 8u + u + 1u;
  }
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      v[u] = __ballot_sync(~0u, (v[u] >> lane) & 1u);
#pragma unroll
      for (int j = 0; j < Q / 8; j++)
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(w[(u * Q / 8 + j) & 7]) : "r"(k1), "r"(k2));
    }
  }
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= v[u] ^ w[u];
  OP_SINK(s)
}

__device__ __forceinline__ void redux_match(const OpArgs& a) {
  const unsigned lane = threadIdx.x & 31;
  unsigned v[8], m[8];
#pragma unroll
  for (int u = 0; u < 8; u++) {
    v[u] = (unsigned)a.iters * 977u + lane * 8u + (unsigned)u;
    m[u] = (unsigned)a.iters * 977u + (unsigned)u;
  }
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      v[u] = __reduce_add_sync(~0u, v[u]) + lane;
      m[u] = __match_any_sync(~0u, m[u]);
    }
  }
  OP_END
  uint32_t s = 0;
#pragma unroll
  for (int u = 0; u < 8; u++) s ^= v[u] ^ m[u];
  OP_SINK(s)
}

__device__ __forceinline__ void alls(const OpArgs& a) {
  const unsigned lane = threadIdx.x & 31;
  unsigned v[8];
#pragma unroll
  for (int u = 0; u < 8; u++) v[u] = (unsigned)a.iters * 977u + lane * 8u + (unsigned)u;
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int u = 0; u < 8; u++) v[u] += (unsigned)__all_sync(~0u, v[u] & 1u);
  }
  OP_END
  OP_SINK(v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[4] ^ v[5] ^ v[6] ^ v[7])
}

OP_KERNEL(redux_add, TPB) { reduxes<0, 0>(a); }
OP_KERNEL(redux_min, TPB) { reduxes<1, 0>(a); }
OP_KERNEL(redux_xor, TPB) { reduxes<2, 0>(a); }
OP_KERNEL(vote_ballot, TPB) { ballots<0>(a); }
OP_KERNEL(vote_all, TPB) { alls(a); }
OP_KERNEL(mix_redux_match, TPB) { redux_match(a); }
#define MIXR(Q) OP_KERNEL(mix_redux_lop3_##Q, TPB) { reduxes<0, Q>(a); }
#define MIXV(Q) OP_KERNEL(mix_vote_lop3_##Q, TPB) { ballots<Q>(a); }
MIXR(16) MIXR(32) MIXR(64) MIXV(16) MIXV(32) MIXV(64)
// A REDUX chain's add and move a step: sm_120's ptxas emits IADD3 and MOV (and an IADD3 for the loop's counter); sm_90's
// splits them between IADD3 and MOV on alu and IMAD.IADD and IMAD.U32 on fma, putting more on IMAD the more LOP3 are
// beside them, and counts the loop in a UIADD3.
#if PROBE_ARCH == 90
#define CHAIN_0 "IMAD.IADD=5,IADD3=3,IMAD.U32=5,MOV=3"
#define CHAIN_MATCH CHAIN_0
#define CHAIN_16 "IMAD.IADD=7,IADD3=1,IMAD.U32=7,MOV=1"
#define CHAIN_32 "IMAD.IADD=8,IMAD.U32=8"
#define CHAIN_64 CHAIN_32
#else
#define CHAIN_0 "IADD3=9,MOV=8"
#define CHAIN_MATCH "IADD3=8,MOV=8"
#define CHAIN_16 CHAIN_0
#define CHAIN_32 CHAIN_0
#define CHAIN_64 CHAIN_0
#endif
#define P_MIXR(Q)                                                                       \
  {.name = "mix_redux_lop3_" #Q, .kernel = (const void*)mix_redux_lop3_##Q, .tpb = TPB, \
   .expect = "REDUX.SUM=8," CHAIN_##Q ",LOP3.LUT=" #Q}

static const OpProbe PROBES[] = {
    {.name = "redux_add", .kernel = (const void*)redux_add, .tpb = TPB, .expect = "REDUX.SUM=8," CHAIN_0},
    {.name = "redux_min", .kernel = (const void*)redux_min, .tpb = TPB, .expect = "REDUX.MIN=8," CHAIN_0},
    {.name = "redux_xor", .kernel = (const void*)redux_xor, .tpb = TPB, .expect = "REDUX.XOR=8," CHAIN_0},
    {.name = "vote_ballot", .kernel = (const void*)vote_ballot, .tpb = TPB, .expect = "VOTE.ANY=8"},
    {.name = "vote_all", .kernel = (const void*)vote_all, .tpb = TPB, .expect = "VOTE.ALL=8"},
    {.name = "mix_redux_match", .kernel = (const void*)mix_redux_match, .tpb = TPB,
     .expect = "REDUX.SUM=8,MATCH.ANY=8," CHAIN_MATCH},
    P_MIXR(16), P_MIXR(32), P_MIXR(64),
    {.name = "mix_vote_lop3_16", .kernel = (const void*)mix_vote_lop3_16, .tpb = TPB, .expect = "VOTE.ANY=8,LOP3.LUT=24"},
    {.name = "mix_vote_lop3_32", .kernel = (const void*)mix_vote_lop3_32, .tpb = TPB, .expect = "VOTE.ANY=8,LOP3.LUT=40"},
    {.name = "mix_vote_lop3_64", .kernel = (const void*)mix_vote_lop3_64, .tpb = TPB, .expect = "VOTE.ANY=8,LOP3.LUT=72"},
};
int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
