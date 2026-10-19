// RT-H200-EMP's combined-route attacks on sm_90a (H200), the cases the single-stream probes leave open. Built by
// ../opcode_probe/probe.py build --arch sm_90a; every probe passes the SASS gate, the poisoned sink and its no-write
// control before it is timed (probe.cuh). Accumulators stay in registers; A and B come from one shared tile through
// no-swizzle K-major descriptors (dc.cuh).
//
// The one facet the census never measured (the WATCH, isa_escape.py; note:20261017T1834Z-draft-rt-isa, decision 88): a
// uniform-datapath instruction's issue cost beside a SATURATED asynchronous wgmma. RT-ISA r2 (sm90_async.cu, uwgh) put
// the uniform chain beside a saturated wgmma and found it nearly free -- but its vector issue sat idle, so "free" there
// is only the scheduler filling empty issue slots, not a second instruction a cycle. A2/A2u say an SMSP issues at most
// one warp instruction a cycle, uniform included: the sum over EVERY class of warp instructions a SMSP-cycle is <= 1.
// The decisive test is the uniform chain beside a saturated wgmma WHILE the vector pipes already carry issue toward 1 a
// cycle: if a uniform instruction is still free there, two instructions issue in one cycle and A2/A2u break; if it costs
// an issue slot (the loop lengthens one cycle a uniform op), A2u holds where it was never measured.
//
//   emp_pack_<S>_r<R>_k<K>  2 warpgroups, each iteration: 4 wgmma s8 m64n256k32 (a commit group, saturating the tensor
//                           pipe as wg_s8_n256), the honest step (8 mma.sync m16n8k32 + its duo fold, the one structure
//                           where ptxas keeps the chain on the uniform datapath, as uwgh), R FFMA on independent vector
//                           accumulators (the issue filler: FADD/FFMA is the vector pipe, distinct from the fold's
//                           IMAD/LOP3/LEA), and a K-instruction warp-uniform chain (uchain.cuh), then waits. The gate
//                           pins UIMAD/ULOP3 (the chain stayed uniform), FFMA = R (the filler), IMMA = 8 and IGMMA = 4.
//                           counted is the chain op (K > 0) or the IGMMA (K = 0); analyze sums every class's rate a
//                           SMSP-cycle (total issue) and reads the K-slope at each R.
#include "../../opcode_probe/probe.cuh"
#include "../../opcode_probe/probes/honest.cuh"
#include "../dc.cuh"
#include "../wgmma.cuh"
#include "lanes.cuh"
#include "uchain.cuh"

constexpr int ATPB = 256, AITERS = 4096, KB = 32, AROWS = 64, BROWS = 256;

DC_DEV uint32_t emp_word(uint32_t i, int mode) {
  return mode == 1 ? 0u : dc_mix32(i * 0x9E3779B9u + 0x1234567u);
}
struct Tiles {
  uint64_t da, db;
};
DC_DEV Tiles emp_tiles(unsigned char* sm, int mode) {
  uint32_t* w = reinterpret_cast<uint32_t*>(sm);
  for (uint32_t i = threadIdx.x; i < (AROWS + BROWS) * KB / 4; i += blockDim.x) w[i] = emp_word(i, mode);
  fence_async_smem();
  __syncthreads();
  const uint32_t base = smem_u32(sm);
  return {wgmma_desc(base, kmaj_lbo(), kmaj_sbo(KB)), wgmma_desc(base + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB))};
}
// Four wgmma into one accumulator: the first overwrites (scale_d 0), the rest accumulate; saturates the tensor pipe.
DC_DEV void emp_group4(int (&d)[128], const Tiles& t) {
  wgmma_fence();
  wgmma_s8_n256(d, t.da, t.db, 0);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_s8_n256(d, t.da, t.db, 1);
  wgmma_commit();
}
template <int N>
DC_DEV uint32_t emp_fold(const int (&d)[N]) {
  uint32_t x = 0;
#pragma unroll
  for (int i = 0; i < N; i++) x ^= (uint32_t)d[i];
  return x;
}

template <int S, int R, int K>
DC_DEV void emp_pack(const OpArgs& a) {
  __shared__ __align__(1024) unsigned char sm[(AROWS + BROWS) * KB];
  const Tiles t = emp_tiles(sm, a.mode);
  int d[128] = {};
  Honest h(a.mode);  // 8 mma.sync + duo fold: forces ptxas to keep the uniform chain on the uniform datapath
  Lane lv((uint32_t)threadIdx.x);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
    emp_group4(d, t);
    h.step<8, true>();
    lane_ops<L_FFMA, R>(lv);
    u.step();
    wgmma_wait<0>();
  }
  wgmma_hold(d);
  OP_END
  OP_SINK(emp_fold(d) ^ h.fold() ^ lv.fold() ^ u.fold() ^ threadIdx.x)
}

#define PACK(S, SN, R, K) \
  OP_KERNEL(emp_pack_##SN##_r##R##_k##K, ATPB) { emp_pack<S, R, K>(a); }
PACK(S_I, i, 0, 0)
PACK(S_I, i, 0, 64)
PACK(S_I, i, 0, 128)
PACK(S_I, i, 128, 0)
PACK(S_I, i, 128, 64)
PACK(S_I, i, 128, 128)
PACK(S_I, i, 256, 0)
PACK(S_I, i, 256, 64)
PACK(S_I, i, 256, 128)
#undef PACK

#define WGI "IGMMA.64x256x32.S8.S8=4"
#define HON "IMMA.16832.S8.S8=8,IMAD=32,LEA.HI=32,LOP3.LUT=32"
// counted: the chain's UIMAD (K > 0) or the IGMMA (K = 0). FFMA = R is the filler; UIMAD = ULOP3 = K/2 is the chain.
#define P(SN, R, K, EX, CNT) \
  {.name = "emp_pack_" #SN "_r" #R "_k" #K, .kernel = (const void*)emp_pack_##SN##_r##R##_k##K, .tpb = ATPB, \
   .expect = EX, .counted = CNT, .iters = AITERS}
static const OpProbe PROBES[] = {
    P(i, 0, 0, WGI "," HON, "IGMMA.64x256x32.S8.S8"),
    P(i, 0, 64, WGI "," HON ",UIMAD=32,ULOP3.LUT=32", "UIMAD"),
    P(i, 0, 128, WGI "," HON ",UIMAD=64,ULOP3.LUT=64", "UIMAD"),
    P(i, 128, 0, WGI "," HON ",FFMA=128", "IGMMA.64x256x32.S8.S8"),
    P(i, 128, 64, WGI "," HON ",FFMA=128,UIMAD=32,ULOP3.LUT=32", "UIMAD"),
    P(i, 128, 128, WGI "," HON ",FFMA=128,UIMAD=64,ULOP3.LUT=64", "UIMAD"),
    P(i, 256, 0, WGI "," HON ",FFMA=256", "IGMMA.64x256x32.S8.S8"),
    P(i, 256, 64, WGI "," HON ",FFMA=256,UIMAD=32,ULOP3.LUT=32", "UIMAD"),
    P(i, 256, 128, WGI "," HON ",FFMA=256,UIMAD=64,ULOP3.LUT=64", "UIMAD"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
