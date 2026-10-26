// RT-B200-EMP (MM red, 24h push 19 Oct): A2u, the issue ceiling with the uniform datapath included, tested where it
// could fail on sm_100a (B200): every SMSP's issue saturated by lane instructions while warp 0 keeps the tensor pipe
// saturated with tcgen05.mma and issues K warp-uniform chain instructions an iteration.
//
// A2/A2u (Security/Definitions/Accounting/Assumptions/Sheet.lean) say an SMSP issues at most one warp instruction a cycle,
// uniform ones included. RT-ISA r2's utc_ (sm100_async.cu) found the chain flat in K beside the async op, but its one
// warp leaves the SMSP's issue ~3/4 idle (4 UTCIMMA hold it ~512 cycles), so the chain only filled idle slots: that
// can't tell a free uniform slot from an idle one. Here nothing is idle. If the uniform instructions cost no slot, the
// block issues more than 4 warp instructions an SM clock: (fixed loop's instructions x iters + the OP_UNTIL loop's
// instructions x trips) / (4 x the stop's cycles) > 1, and A2u is false. If they cost a slot each, it stays <= 1
// (a2u.py reads every loop's full SASS and the per-block trips and stop cycles).
//
//   a2_mix        control: warp 0 runs L_MIX for a fixed count, warps 1-31 run it until warp 0 stops; the lane
//                 stream's own issue (scalar.cu's s_mix reaches 4.000 a SM clock)
//   a2_tc_<S><K>  warp 0: 4 tcgen05.mma i8 128x256x32 a commit group (as sm100_async's utc_) + K chain instructions of
//                 uchain.cuh's form S (i: UIMAD/ULOP3, x: ULOP3) an iteration; warps 1-31: L_MIX until it stops
#include "../../opcode_probe/probe.cuh"
#include "../dc.cuh"
#include "lanes.cuh"
#include "uchain.cuh"

// The TMEM setup, descriptors and the i8 128x256 commit group mirror sm100_async.cu's (whose gate pins the same 4
// UTCIMMA a group); kept here rather than shared so that file's gate-pinned SASS is untouched.
constexpr int WTPB = 1024, WITERS = 2048, R = 128, KB = 32, AROWS = 128, BROWS = 256, TMEM_COLS = 512;

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

OP_KERNEL(a2_mix, WTPB) {
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN_STOP
  if (warp_u() == 0) {
    OP_LOOP(it) { lane_ops<L_MIX, R>(l); }
    OP_DONE
  } else {
    OP_UNTIL(n) { lane_ops<L_MIX, R>(l); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  OP_SINK(l.fold() ^ threadIdx.x)
}

template <int S, int K>
DC_DEV void a2_tc(const OpArgs& a) {
  __shared__ __align__(1024) Smem s;
  const uint32_t tmem = setup(s, a.mode);
  const uint64_t da = desc_a(s), db = desc_b(s);
  UChains<K, S> u(0x5eed0000u ^ (uint32_t)a.mode);
  Lane l(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN_STOP
  if (warp_u() == 0) {
    OP_LOOP(it) {
      if (elect_one()) mma4(tmem, da, db);
      u.step();
      __syncwarp();
    }
    OP_DONE
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  } else {
    OP_UNTIL(n) { lane_ops<L_MIX, R>(l); }
    OP_TRIPS(n)
  }
  OP_END_STOP
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  const uint32_t x = sink_ld(tmem) ^ u.fold() ^ l.fold();
  teardown(tmem);
  OP_SINK(x ^ threadIdx.x)
}

#define A2TC(S, N, K) \
  OP_KERNEL(a2_tc_##N##K, WTPB) { a2_tc<S, K>(a); }
A2TC(S_I, i, 0)
A2TC(S_I, i, 32)
A2TC(S_I, i, 64)
A2TC(S_I, i, 96)
A2TC(S_X, x, 96)

// Two loops a probe: the fixed role (warp 0, `iters` trips) then the OP_UNTIL role (warps 1-31). Each loop's expect is
// its opcode histogram as ptxas 13.0 emits it (the branch aside); a2u.py counts both loops in full from the SASS.
#define P_A(nm, EX, CNT) \
  {.name = #nm, .kernel = (const void*)nm, .tpb = WTPB, .expect = EX, .counted = CNT, .warps = "1;31", \
   .iters = WITERS, .until = 2}

// The OP_UNTIL loop's LDS is its stop poll, which a2u.py tells the two loops apart by.
#define UNTIL_EX "I*ADD*=64,FFMA=64,LDS=1"
static const OpProbe PROBES[] = {
    P_A(a2_mix, "I*ADD*=64,FFMA=64,UIADD3=1,UISETP.NE.AND=1;" UNTIL_EX, "FFMA"),
    P_A(a2_tc_i0, "UTCIMMA=4,UMOV=2,UIADD3=1,UISETP.NE.AND=1,NOP=1;" UNTIL_EX, "UTCIMMA"),
    P_A(a2_tc_i32, "UIMAD=16,ULOP3.LUT=16,UTCIMMA=4,UMOV=2,UIADD3=1,UISETP.NE.AND=1;" UNTIL_EX, "UTCIMMA"),
    P_A(a2_tc_i64, "UIMAD=32,ULOP3.LUT=32,UTCIMMA=4,UMOV=2,UIADD3=1,UISETP.NE.AND=1;" UNTIL_EX, "UTCIMMA"),
    P_A(a2_tc_i96, "UIMAD=48,ULOP3.LUT=48,UTCIMMA=4,UMOV=2,UIADD3=1,UISETP.NE.AND=1;" UNTIL_EX, "UTCIMMA"),
    P_A(a2_tc_x96, "ULOP3.LUT=134,UTCIMMA=4,UMOV=3,UIADD3=1,UISETP.NE.AND=1;" UNTIL_EX, "UTCIMMA"),
};

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
