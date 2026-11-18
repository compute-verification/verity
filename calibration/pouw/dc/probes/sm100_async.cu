// Red-team probe: the uniform datapath beside a *saturated asynchronous* tcgen05.mma on sm_100a (B200).
// The committed-word model prices a uniform instruction at an issue slot (sm_100 census UNI,issue 0.924,
// UNI,JI100_0 0.906), measured beside the synchronous mma.sync (uniform.cuh's F_M, um_*). It has never been measured
// beside a saturated async tcgen05.mma, which one thread issues for a whole 128x256 tile, leaving the warp scheduler's
// issue slots even freer than wgmma does: the question is whether ptxas/hardware
// dual-issues a uniform instruction into that shadow at less than an issue slot.
//
//   utc_<S>_<K>   bare: warp 0 issues 4 tcgen05.mma i8 128x256x32 a commit group (UTCIMMA, saturating the tensor pipe, as
//                 tc_i8_m128_n256) and carries K warp-uniform chain instructions a iteration (uchain.cuh). Nothing reads
//                 the TMEM accumulators inside the loop, so -- the facet_break finding, as on sm_90 -- ptxas 13.0 lowers
//                 the chain to the vector lanes (IMAD/LOP3/IADD3), not the uniform datapath. The gate pins the lane
//                 counts; the K-slope is the lane cost beside a saturated async tcgen05.mma.
//   utch_<S>_<K>  honest: as utc, plus the honest step (8 mma.sync + its duo fold) in warp 0's loop, the one thing that
//                 makes ptxas keep the chain on the uniform datapath (UIMAD/ULOP3/UIADD3, as uniform.cuh's um_ does). The
//                 gate pins the uniform counts; the K-slope is the forced-uniform instruction's cost beside the saturated
//                 tensor pipe mma.sync and tcgen05.mma share -- the dual-issue price.
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"
#include "../dc.cuh"
#include "uchain.cuh"

// The TMEM setup, descriptors and the i8 128x256 commit group mirror sm100_tcgen05.cu's tc_alone (its gate pins the same
// 4 UTCIMMA a group); kept here rather than shared so that file's gate-pinned SASS is untouched.
constexpr int ATPB = 128, AITERS = 4096, KB = 32, AROWS = 128, BROWS = 256, TMEM_COLS = 512;

struct Smem {
  unsigned char tile[(AROWS + BROWS) * KB];
  uint64_t bar;
  uint32_t tmem;
};
DC_DEV uint32_t warp_u() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }
DC_DEV uint32_t setup(Smem& s, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(s.tile);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x)
    w[i] = mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
  if (threadIdx.x == 0) {
    mbar_init(&s.bar, 1);
    fence_mbar_init();
  }
  if (threadIdx.x < 32) tmem_alloc(&s.tmem, TMEM_COLS);
  fence_async_smem();
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  return s.tmem;
}
DC_DEV void teardown(uint32_t tmem) {
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(tmem, TMEM_COLS);
}
DC_DEV uint64_t desc_a(Smem& s) { return umma_desc(smem_u32(s.tile), kmaj_lbo(), kmaj_sbo(KB)); }
DC_DEV uint64_t desc_b(Smem& s) { return umma_desc(smem_u32(s.tile) + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB)); }
// Four tcgen05.mma i8 128x256x32 into two accumulators: D0, D1 overwritten then accumulated; saturates the tensor pipe.
DC_DEV void mma4(uint32_t tmem, uint64_t da, uint64_t db) {
  constexpr uint32_t idesc = umma_idesc(2, 1, 1, 128, 256);
  umma_i8(tmem, da, db, idesc, 0);
  umma_i8(tmem + 256u, da, db, idesc, 0);
  umma_i8(tmem, da, db, idesc, 1);
  umma_i8(tmem + 256u, da, db, idesc, 1);
}
DC_DEV uint32_t sink_ld(uint32_t tmem) {
  uint32_t r[16];
  tmem_ld16(tmem + ((32u * ((threadIdx.x >> 5) & 3u)) << 16), r);
  tmem_wait_ld();
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) x ^= r[i];
  return x;
}

template <int S, int K>
DC_DEV void utc(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4(tmem, da, db);
      u.step();
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem) ^ u.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

// The honest step (8 mma.sync + its duo fold) in the loop is the one thing that keeps the chain on the uniform datapath;
// mma.sync and tcgen05.mma share the tensor pipe, so the chain sits beside the saturated async op here too.
template <int S, int K>
DC_DEV void utch(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  Honest h(a.mode);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4(tmem, da, db);
      h.step<8, true>();
      u.step();
      __syncwarp();
    }
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  OP_END
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem) ^ h.fold() ^ u.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

#define UTC(S, N, K) \
  OP_KERNEL(utc_##N##_##K, ATPB) { utc<S, K>(a); }
#define UTCH(S, N, K) \
  OP_KERNEL(utch_##N##_##K, ATPB) { utch<S, K>(a); }
UTC(S_I, i, 0)
UTC(S_I, i, 32)
UTC(S_I, i, 64)
UTC(S_I, i, 128)
UTC(S_X, x, 64)
UTC(S_A, a, 64)
UTCH(S_I, i, 0)
UTCH(S_I, i, 32)
UTCH(S_I, i, 64)
UTCH(S_I, i, 128)
UTCH(S_X, x, 64)
UTCH(S_A, a, 64)

#define P(nm, EX, CNT) \
  {.name = #nm, .kernel = (const void*)nm, .tpb = ATPB, .expect = EX, .counted = CNT, .warps = "1", .iters = AITERS}

// Every loop's full opcode histogram (ptxas 13.0, CUDA 13.0). On sm_100a the honest tcgen05 kernel is single-warp (one
// thread issues the whole 128x256 tile), so even the bare utc keeps the chain on the uniform datapath (UIMAD/ULOP3/UIADD3)
// beside the 4 UTCIMMA -- the uniform-beside-saturated-async case is the natural one here, unlike sm_90's wgmma (lanes).
// utch adds the honest mma.sync (IMMA.16832); the chain stays uniform. counted is the chain op (the K-slope's), UTCIMMA at
// K=0. A chain that moves off the uniform datapath -- the WATCH -- refuses.
static const OpProbe PROBES[] = {
    P(utc_i_0, "UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UTCIMMA"),
    P(utc_i_32, "UIMAD=16,ULOP3.LUT=16,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utc_i_64, "UIMAD=32,ULOP3.LUT=32,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utc_i_128, "UIMAD=64,ULOP3.LUT=64,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utc_x_64, "ULOP3.LUT=94,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "ULOP3.LUT"),
    P(utc_a_64, "UIADD3=33,ULOP3.LUT=32,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UISETP.NE.AND=1", "UIADD3"),
    P(utch_i_0, "IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UTCIMMA"),
    P(utch_i_32, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=16,ULOP3.LUT=16,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utch_i_64, "IMAD=32,LEA.HI=32,LOP3.LUT=32,UIMAD=32,ULOP3.LUT=32,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utch_i_128, "UIMAD=64,ULOP3.LUT=64,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "UIMAD"),
    P(utch_x_64, "ULOP3.LUT=94,IMAD=32,LEA.HI=32,LOP3.LUT=32,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UIADD3=1,UISETP.NE.AND=1", "ULOP3.LUT"),
    P(utch_a_64, "UIADD3=33,IMAD=32,LEA.HI=32,LOP3.LUT=32,ULOP3.LUT=32,IMMA.16832.S8.S8=8,UTCIMMA=4,UMOV=2,BRA.U=1,NOP=1,UISETP.NE.AND=1", "UIADD3"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
