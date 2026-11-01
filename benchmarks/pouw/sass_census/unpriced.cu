// The census's unpriced sm_100a classes, timed beside the classes that price them:
// FFMA2 and FMUL2 (fma.rn.f32x2, mul.rn.f32x2: two FP32 results a lane, which loads.class_of folds into FFMA) beside
// FFMA and FMUL, and DMMA (mma.sync m8n8k4 f64, which no DS_tensor row lists) beside IMMA (mma.sync m16n8k32 s8) and
// beside a saturated tcgen05.mma kind::i8 (dmmaU_tc, tcU_dmma).
// Built and gated by ../opcode_probe/probe.py build --arch sm_100a (f32x2 is sm_100+); rates.sh is the job.
//   s_<op>     1024 threads (8 warps a SMSP), 8 independent chains a thread; lane ops 64 an iteration, MMAs 8
//   p_<x>_<y>  the same with x and y alternating (32 each): FFMA2's issue slots beside LOP3 and IMAD
// A rate is warp instructions a SM a clock (probe.py); FP32 results a SM-clock are 32 x it (FFMA, FMUL) or 64 x it
// (FFMA2, FMUL2), MACs a SM-clock 256 x it (DMMA) or 4096 x it (IMMA).
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"
#include "unpriced.cuh"

// ---- DMMA beside a saturated tcgen05.mma: does it share the tensor pipe, as mma.sync IMMA does (row M)? -------------
// sm100_tcgen05.cu's tc_alone and sync_until_mma with DMMA for IMMA: thread 0 of warp 0 issues tcgen05.mma kind::i8
// 128x256x32 into TMEM, warps 8-15 (2 a SMSP) issue 8 DMMA an iteration; one role fixed, the other until it ends
// (OP_UNTIL), so both rates are exact. t + s > 1 (t the tcgen05 rate over tc_i8's, s the DMMA rate over s_dmma_w8's)
// says DMMA runs beside the tensor pipe and adds its words to a saturated tcgen05's.
constexpr int KB = 32, AROWS = 128, BROWS = 256, TMEM_COLS = 512;
struct Tc {
  unsigned char tile[(AROWS + BROWS) * KB];
  uint64_t bar;
  uint32_t tmem;
};
UP_DEV uint32_t tc_setup(Tc& s, int mode) {
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
UP_DEV void tc_finish(Tc& s, uint32_t tmem, uint32_t w, uint32_t& x) {
  if (w == 0) {
    if (elect_one()) umma_commit(&s.bar);
    __syncwarp();
  }
  mbar_wait(&s.bar, 0);
  tc_after_sync();
  uint32_t r[16];
  tmem_ld16(tmem + ((32u * ((threadIdx.x >> 5) & 3u)) << 16), r);
  tmem_wait_ld();
#pragma unroll
  for (int i = 0; i < 16; i++) x ^= r[i];
  tc_before_sync();
  __syncthreads();
  tc_after_sync();
  if (threadIdx.x < 32) tmem_dealloc(tmem, TMEM_COLS);
}
// Four MMAs an iteration into two accumulators (sm100_tcgen05.cu's mma4<0, 128, 256>).
UP_DEV void tc_mma4(uint32_t tmem, uint64_t da, uint64_t db) {
  constexpr uint32_t idesc = umma_idesc(2, 1, 1, 128, 256);
  if (elect_one()) {
    umma_i8(tmem, da, db, idesc, 0);
    umma_i8(tmem + 256, da, db, idesc, 0);
    umma_i8(tmem, da, db, idesc, 1);
    umma_i8(tmem + 256, da, db, idesc, 1);
  }
  __syncwarp();
}
// ROLE 0: tcgen05 alone (warp 0); 1: tcgen05 fixed, DMMA until; 2: DMMA fixed, tcgen05 until.
template <int ROLE>
UP_DEV void dmma_tc(const OpArgs& a) {
  __shared__ __align__(1024) Tc s;
  const uint32_t tmem = tc_setup(s, a.mode);
  const uint64_t da = umma_desc(smem_u32(s.tile), kmaj_lbo(), kmaj_sbo(KB));
  const uint64_t db = umma_desc(smem_u32(s.tile) + AROWS * KB, kmaj_lbo(), kmaj_sbo(KB));
  Chains c(threadIdx.x ^ (uint32_t)a.mode);
  const uint32_t w = warp_u();
  auto dmma8 = [&] {
#pragma unroll
    for (int q = 0; q < 8; q++) c.op<U_DMMA>(q);
  };
  OP_BEGIN_STOP
  if constexpr (ROLE == 0) {
    if (w == 0) {
      OP_LOOP(it) { tc_mma4(tmem, da, db); }
    }
  } else if constexpr (ROLE == 1) {
    if (w == 0) {
      OP_LOOP(it) { tc_mma4(tmem, da, db); }
      OP_DONE
    } else if (w >= 8) {
      OP_UNTIL(n) { dmma8(); }
      OP_TRIPS(n)
    }
  } else {
    if (w >= 8) {
      OP_LOOP(it) { dmma8(); }
      OP_DONE
    } else if (w == 0) {
      OP_UNTIL(n) { tc_mma4(tmem, da, db); }
      OP_TRIPS(n)
    }
  }
  OP_END_STOP
  uint32_t x = c.fold<U_DMMA>();
  tc_finish(s, tmem, w, x);
  OP_SINK(x ^ threadIdx.x)
}

OP_KERNEL(s_ffma, 1024) { s_alone<U_FFMA, 64>(a); }
OP_KERNEL(s_ffma2, 1024) { s_alone<U_FFMA2, 64>(a); }
OP_KERNEL(s_ffma2c, 1024) { s_alone<U_FFMA2C, 64>(a); }
OP_KERNEL(s_fmul, 1024) { s_alone<U_FMUL, 64>(a); }
OP_KERNEL(s_fmul2, 1024) { s_alone<U_FMUL2, 64>(a); }
OP_KERNEL(s_dmma, 1024) { s_alone<U_DMMA, 8>(a); }
OP_KERNEL(s_imma, 1024) { s_alone<U_IMMA, 8>(a); }
OP_KERNEL(s_dmma_w8, 256) { s_alone<U_DMMA, 8>(a); }
OP_KERNEL(s_lop3, 1024) { s_alone<U_LOP3, 64>(a); }
OP_KERNEL(s_imad, 1024) { s_alone<U_IMAD, 64>(a); }
OP_KERNEL(p_ffma_lop3, 1024) { s_pair<U_FFMA, U_LOP3, 64>(a); }
OP_KERNEL(p_ffma2_lop3, 1024) { s_pair<U_FFMA2C, U_LOP3, 64>(a); }
OP_KERNEL(p_ffma_imad, 1024) { s_pair<U_FFMA, U_IMAD, 64>(a); }
OP_KERNEL(p_ffma2_imad, 1024) { s_pair<U_FFMA2C, U_IMAD, 64>(a); }
OP_KERNEL(tc_i8, 512) { dmma_tc<0>(a); }
OP_KERNEL(dmmaU_tc, 512) { dmma_tc<1>(a); }
OP_KERNEL(tcU_dmma, 512) { dmma_tc<2>(a); }

#define UI "UTCIMMA*=4"
#define P(NAME, PAT, IT, NOTE) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .iters = IT, .note = NOTE}
static const OpProbe PROBES[] = {
    P(s_ffma, "FFMA=64", 4096, "FP32 results = 32 x rate"),
    P(s_ffma2, "FFMA2*=64", 4096, "FP32 results = 64 x rate; class_of prices it as one FFMA"),
    P(s_ffma2c, "FFMA2*=64", 4096, "s_ffma2 with one constant addend pair (FFMA's 1.0), in case three pairs conflict"),
    P(s_fmul, "FMUL=64", 4096, "FP32 results = 32 x rate"),
    P(s_fmul2, "FMUL2*=64", 4096, "FP32 results = 64 x rate; class_of prices it as one FFMA"),
    P(s_dmma, "DMMA*=8", 256, "f64 MACs = 256 x rate; exact for int8 x int8 sums"),
    P(s_imma, "IMMA*=8", 1024, "s8 MACs = 4096 x rate: the unit DMMA is read against"),
    P(s_lop3, "LOP3.LUT=64", 4096, "the pairs' integer lane alone"),
    P(s_imad, "IMAD=64", 4096, "the pairs' integer lane alone"),
    P(p_ffma_lop3, "FFMA=32,LOP3.LUT=32", 4096, "the control: FFMA takes one issue slot"),
    P(p_ffma2_lop3, "FFMA2*=32,LOP3.LUT=32", 4096, "FFMA2's issue slots: the two rates' sum"),
    P(p_ffma_imad, "FFMA=32,IMAD=32", 4096, "the control beside IMAD"),
    P(p_ffma2_imad, "FFMA2*=32,IMAD=32", 4096, "FFMA2 beside IMAD (does IMAD share its pipe?)"),
    {.name = "s_dmma_w8", .kernel = (const void*)s_dmma_w8, .tpb = 256, .expect = "DMMA*=8", .iters = 1024,
     .note = "DMMA alone at the joint's 8 warps: s's unit"},
    {.name = "tc_i8", .kernel = (const void*)tc_i8, .tpb = 512, .expect = UI, .warps = "1", .iters = 2048,
     .note = "tcgen05 i8 128x256x32 alone in the joint's block: t's unit"},
    {.name = "dmmaU_tc", .kernel = (const void*)dmmaU_tc, .tpb = 512, .expect = UI ";DMMA*=8", .warps = "1;8",
     .iters = 2048, .note = "tcgen05 fixed, DMMA until", .until = 2},
    {.name = "tcU_dmma", .kernel = (const void*)tcU_dmma, .tpb = 512, .expect = "DMMA*=8;" UI, .warps = "8;1",
     .iters = 2048, .note = "DMMA fixed, tcgen05 until", .until = 2},
};
#undef P
#undef UI

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
