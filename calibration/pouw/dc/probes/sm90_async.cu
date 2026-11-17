// Red-team probe: the uniform datapath beside a *saturated asynchronous* wgmma on sm_90a (H100/H200).
// The committed-word model prices a uniform instruction at an issue slot (sm_90 census UNI,issue 0.924,
// UNI,JI90_0 0.965), measured beside the synchronous honest step (mma.sync, uniform.cuh's F_M) and beside issue-saturated
// lane streams (uniform.cuh's us_/uv_ families). It has never been measured beside a saturated async wgmma, whose issue
// slot is one instruction but whose tensor pipe runs for ~128 clocks after it, so the warp scheduler's issue slots sit
// mostly idle: the question is whether ptxas/hardware dual-issues a uniform instruction
// into that shadow at less than an issue slot.
//
//   uwg_<S>_<K>   bare: each of 2 warpgroups issues 4 wgmma s8 m64n256k32 a commit group (saturating the tensor pipe, as
//                 wg_mix), waits, and carries K warp-uniform chain instructions a iteration (uchain.cuh). Nothing reads
//                 the accumulators inside the loop, so -- the facet_break finding -- ptxas 13.0 lowers the chain to the
//                 vector lanes (IMAD/LOP3/IADD3), not the uniform datapath: a co-located uniform chain gets no free ride.
//                 The gate pins the lane counts; the K-slope is the lane cost beside a saturated async wgmma.
//   uwgf_<S>_<K>  folded: as uwg, but each iteration folds the 8 leading wgmma accumulators by duo (honest.cuh's fold,
//                 reading the accumulators), the one structure where ptxas keeps the chain on the uniform datapath
//                 (UIMAD/ULOP3, as uniform.cuh's F_M does beside mma.sync). The gate pins the uniform counts; the K-slope
//                 is the uniform instruction's cost beside a saturated async wgmma -- the dual-issue price.
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"
#include "../dc.cuh"
#include "../wgmma.cuh"
#include "uchain.cuh"

constexpr int ATPB = 256, AITERS = 2048, KB = 32, AROWS = 64, BROWS = 256;

DC_DEV uint32_t operand_word(uint32_t i, int mode) {
  return mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
}
struct Tiles {
  uint64_t da, db;
};
DC_DEV Tiles fill_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) w[i] = operand_word(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}
// Four wgmma into one accumulator: the first overwrites (scale_d 0), the rest accumulate; saturates the tensor pipe.
DC_DEV void group4(int (&d)[128], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n256(d, t.da, t.db, 0);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_commit();
}
template <int N>
DC_DEV uint32_t fold_acc(const int (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= (uint32_t)d[i];
  return x;
}

template <int S, int K>
DC_DEV void uwg(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles(sm, a.mode);
  int d[128] = {};
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    group4(d, t);
    u.step();
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ u.fold() ^ threadIdx.x)
}

// Folding the accumulators in the loop (duo, reading every word) is the one place ptxas keeps the chain uniform.
template <int S, int K>
DC_DEV void uwgf(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles(sm, a.mode);
  int d[128] = {};
  Honest h(a.mode);  // its duo-fold state (ys, zs, a0, a1, k0, k1); no mma.sync is issued, only the wgmma above
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    group4(d, t);
    wgmma_wait<0>();
#pragma unroll
    for (int c = 0; c < 8; c++)
      rf_duo_step(h.ys[c], h.zs[c], (uint32_t)d[c * 4], (uint32_t)d[c * 4 + 1], (uint32_t)d[c * 4 + 2],
                  (uint32_t)d[c * 4 + 3], h.a0, h.a1, h.k0, h.k1);
    u.step();
  }
  OP_END
  OP_SINK(h.fold() ^ u.fold() ^ threadIdx.x)
}

// The honest step (8 mma.sync + its duo fold) in the loop is the one thing that makes ptxas keep the chain on the uniform
// datapath (UIMAD/ULOP3); the fold alone (uwgf) does not. mma.sync and wgmma share the tensor pipe, so the chain sits
// beside a saturated tensor pipe here too; the K-slope is the forced-uniform instruction's cost beside it.
template <int S, int K>
DC_DEV void uwgh(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = fill_tiles(sm, a.mode);
  int d[128] = {};
  Honest h(a.mode);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    group4(d, t);
    h.step<8, true>();
    u.step();
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(fold_acc(d) ^ h.fold() ^ u.fold() ^ threadIdx.x)
}

#define UWG(S, N, K) \
  OP_KERNEL(uwg_##N##_##K, ATPB) { uwg<S, K>(a); }
#define UWGF(S, N, K) \
  OP_KERNEL(uwgf_##N##_##K, ATPB) { uwgf<S, K>(a); }
#define UWGH(S, N, K) \
  OP_KERNEL(uwgh_##N##_##K, ATPB) { uwgh<S, K>(a); }
UWG(S_I, i, 0)
UWG(S_I, i, 32)
UWG(S_I, i, 64)
UWG(S_I, i, 128)
UWG(S_X, x, 64)
UWG(S_X, x, 128)
UWG(S_A, a, 64)
UWGF(S_I, i, 0)
UWGF(S_I, i, 32)
UWGF(S_I, i, 64)
UWGF(S_I, i, 128)
UWGF(S_X, x, 64)
UWGF(S_A, a, 64)
UWGH(S_I, i, 0)
UWGH(S_I, i, 32)
UWGH(S_I, i, 64)
UWGH(S_I, i, 128)
UWGH(S_X, x, 64)
UWGH(S_A, a, 64)

#define P(nm, EX, CNT) {.name = #nm, .kernel = (const void*)nm, .tpb = ATPB, .expect = EX, .counted = CNT, .iters = AITERS}

// Every loop's full opcode histogram (ptxas 13.0, CUDA 13.0), so a chain that moves off its datapath -- the WATCH --
// refuses. uwg/uwgf carry the chain on the vector lanes (IMAD/LOP3/IADD3) beside the 4 IGMMA; only uwgh, with the honest
// mma.sync (IMMA.16832) in the loop, keeps it on the uniform datapath (UIMAD/ULOP3/UIADD3). counted is the chain op (the
// K-slope's instruction), or the IGMMA at K=0.
static const OpProbe PROBES[] = {
    P(uwg_i_0, "IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IGMMA.64x256x32.S8.S8"),
    P(uwg_i_32, "IMAD=16,LOP3.LUT=16,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwg_i_64, "IMAD=32,LOP3.LUT=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwg_i_128, "IMAD=64,LOP3.LUT=64,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwg_x_64, "LOP3.LUT=96,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "LOP3.LUT"),
    P(uwg_x_128, "LOP3.LUT=192,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "LOP3.LUT"),
    P(uwg_a_64, "IADD3=32,LOP3.LUT=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IADD3"),
    P(uwgf_i_0, "LOP3.LUT=64,IMAD=32,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IGMMA.64x256x32.S8.S8"),
    P(uwgf_i_32, "LOP3.LUT=80,IMAD=48,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwgf_i_64, "LOP3.LUT=96,IMAD=64,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwgf_i_128, "LOP3.LUT=128,IMAD=96,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IMAD"),
    P(uwgf_x_64, "LOP3.LUT=160,IMAD=32,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "LOP3.LUT"),
    P(uwgf_a_64, "LOP3.LUT=96,IADD3=32,IMAD=32,LEA.HI=32,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IADD3"),
    P(uwgh_i_0, "IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,ISETP.LE.AND=1,UIADD3=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "IGMMA.64x256x32.S8.S8"),
    P(uwgh_i_32, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=16,ULOP3.LUT=16,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,PLOP3.LUT=1,UIADD3=1,UISETP.GE.AND=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "UIMAD"),
    P(uwgh_i_64, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=32,ULOP3.LUT=32,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,PLOP3.LUT=1,UIADD3=1,UISETP.GE.AND=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "UIMAD"),
    P(uwgh_i_128, "UIMAD=64,ULOP3.LUT=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,PLOP3.LUT=1,UIADD3=1,UISETP.GE.AND=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "UIMAD"),
    P(uwgh_x_64, "ULOP3.LUT=96,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,PLOP3.LUT=1,UIADD3=1,UISETP.GE.AND=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "ULOP3.LUT"),
    P(uwgh_a_64, "UIADD3=33,IMAD=32,LEA.HI=32,LOP3.LUT=32,ULOP3.LUT=32,IMMA.16832.S8.S8=8,IGMMA.64x256x32.S8.S8=4,PLOP3.LUT=1,UISETP.GE.AND=1,WARPGROUP.ARRIVE=1,WARPGROUP.DEPBAR.LE=1", "UIADD3"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
