// The census's unpriced sm_100a classes, timed beside the classes that price them (note:20261019T0520Z-draft-rt-isa G3):
// FFMA2 and FMUL2 (fma.rn.f32x2, mul.rn.f32x2: two FP32 results a lane, which loads.class_of folds into FFMA) beside
// FFMA and FMUL, and DMMA (mma.sync m8n8k4 f64, which no DS_tensor row lists) beside IMMA (mma.sync m16n8k32 s8) and
// beside a saturated tcgen05.mma kind::i8 (dmmaU_tc, tcU_dmma).
// Built and gated by ../opcode_probe/probe.py build --arch sm_100a (f32x2 is sm_100+); rates.sh is the job.
//   s_<op>   1024 threads (8 warps a SMSP), 8 independent chains a thread; lane ops 64 an iteration, MMAs 8
// A rate is warp instructions a SM a clock (probe.py); FP32 results a SM-clock are 32 x it (FFMA, FMUL) or 64 x it
// (FFMA2, FMUL2), MACs a SM-clock 256 x it (DMMA) or 4096 x it (IMMA).
#include "../opcode_probe/probe.cuh"
#include "../dc/dc.cuh"

#define UP_DEV __device__ __forceinline__

enum { U_FFMA, U_FFMA2, U_FFMA2C, U_FMUL, U_FMUL2, U_DMMA, U_IMMA };

struct Chains {
  float f[8];
  unsigned long long p[8];   // f32x2: two FP32 values a register pair
  double d[8][2];            // DMMA m8n8k4: a thread's two f64 accumulators a chain
  uint32_t r[8][4];          // IMMA m16n8k32: a thread's four s32 accumulators a chain
  unsigned long long one2;   // {1.0f, 1.0f}: FFMA2C's addend, FFMA's 1.0 as a pair
  double da, db;
  uint32_t ia[4], ib[2];
  UP_DEV explicit Chains(uint32_t tid) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const uint32_t h = (tid * 0x9e3779b9u + i * 0x85ebca6bu) | 1u;
      f[i] = 1.0f + (float)(h & 0xffu) * 0x1p-12f;
      const float g = 1.0f + (float)((h >> 8) & 0xffu) * 0x1p-12f;
      asm("mov.b64 %0, {%1, %2};" : "=l"(p[i]) : "f"(f[i]), "f"(g));
      d[i][0] = (double)(h & 0xfffu);
      d[i][1] = (double)(h >> 20);
#pragma unroll
      for (int j = 0; j < 4; j++) r[i][j] = h * (j + 1);
    }
    asm("mov.b64 %0, {%1, %1};" : "=l"(one2) : "f"(1.0f));
    da = (double)(int8_t)(tid * 37u);
    db = (double)(int8_t)(tid * 91u);
#pragma unroll
    for (int j = 0; j < 4; j++) ia[j] = tid * 0x01010101u * (2 * j + 3);
    ib[0] = tid * 0x02040801u;
    ib[1] = tid * 0x80402011u;
  }
  // Chain s reads chain t's previous value (as dc/probes/lanes.cuh), so ptxas can neither fold nor hoist it.
  template <int X>
  UP_DEV void op(int i) {
    const int s = i & 7, t = (i + 1) & 7;
    if constexpr (X == U_FFMA) asm volatile("fma.rn.f32 %0,%0,%1,0f3F800000;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == U_FFMA2) asm volatile("fma.rn.f32x2 %0,%0,%1,%2;" : "+l"(p[s]) : "l"(p[t]), "l"(p[(i + 2) & 7]));
    if constexpr (X == U_FFMA2C) asm volatile("fma.rn.f32x2 %0,%0,%1,%2;" : "+l"(p[s]) : "l"(p[t]), "l"(one2));
    if constexpr (X == U_FMUL) asm volatile("mul.rn.f32 %0,%0,%1;" : "+f"(f[s]) : "f"(f[t]));
    if constexpr (X == U_FMUL2) asm volatile("mul.rn.f32x2 %0,%0,%1;" : "+l"(p[s]) : "l"(p[t]));
    if constexpr (X == U_DMMA)
      asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                   : "+d"(d[s][0]), "+d"(d[s][1]) : "d"(da), "d"(db));
    if constexpr (X == U_IMMA)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(r[s][0]), "+r"(r[s][1]), "+r"(r[s][2]), "+r"(r[s][3])
                   : "r"(ia[0]), "r"(ia[1]), "r"(ia[2]), "r"(ia[3]), "r"(ib[0]), "r"(ib[1]));
  }
  // Only the timed op's chains, so the others are dead and the kernel fits 64 registers a thread.
  template <int X>
  UP_DEV uint32_t fold() const {
    uint32_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (X == U_FFMA || X == U_FMUL) x += __float_as_uint(f[i]);
      if constexpr (X == U_FFMA2 || X == U_FFMA2C || X == U_FMUL2) x += (uint32_t)p[i] ^ (uint32_t)(p[i] >> 32);
      if constexpr (X == U_DMMA) x += (uint32_t)__double_as_longlong(d[i][0]) ^ (uint32_t)__double_as_longlong(d[i][1]);
      if constexpr (X == U_IMMA) x += r[i][0] ^ r[i][1] ^ r[i][2] ^ r[i][3];
    }
    return x;
  }
};

template <int X, int N>
UP_DEV void s_alone(const OpArgs& a) {
  Chains c(threadIdx.x ^ (uint32_t)a.mode);
  OP_BEGIN
  OP_LOOP(it) {
#pragma unroll
    for (int q = 0; q < N; q++) c.op<X>(q);
  }
  OP_END
  OP_SINK(c.template fold<X>() ^ threadIdx.x)
}

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
UP_DEV uint32_t warp_u() { return __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0); }
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
    {.name = "s_dmma_w8", .kernel = (const void*)s_dmma_w8, .tpb = 256, .expect = "DMMA*=8", .iters = 256,
     .note = "DMMA alone at the joint's 8 warps: s's unit"},
    {.name = "tc_i8", .kernel = (const void*)tc_i8, .tpb = 512, .expect = UI, .warps = "1", .iters = 2048,
     .note = "tcgen05 i8 128x256x32 alone in the joint's block: t's unit"},
    {.name = "dmmaU_tc", .kernel = (const void*)dmmaU_tc, .tpb = 512, .expect = UI ";DMMA*=8", .warps = "1;8",
     .iters = 2048, .until = 2, .note = "tcgen05 fixed, DMMA until"},
    {.name = "tcU_dmma", .kernel = (const void*)tcU_dmma, .tpb = 512, .expect = "DMMA*=8;" UI, .warps = "8;1",
     .iters = 2048, .until = 2, .note = "DMMA fixed, tcgen05 until"},
};
#undef P
#undef UI

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
