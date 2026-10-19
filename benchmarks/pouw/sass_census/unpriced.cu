// The census's unpriced sm_100a classes, timed beside the classes that price them (note:20261019T0520Z-draft-rt-isa G3):
// FFMA2 and FMUL2 (fma.rn.f32x2, mul.rn.f32x2: two FP32 results a lane, which loads.class_of folds into FFMA) beside
// FFMA and FMUL, and DMMA (mma.sync m8n8k4 f64, which no DS_tensor row lists) beside IMMA (mma.sync m16n8k32 s8).
// Built and gated by ../opcode_probe/probe.py build --arch sm_100a (f32x2 is sm_100+); rates.sh is the job.
//   s_<op>   1024 threads (8 warps a SMSP), 8 independent chains a thread; lane ops 64 an iteration, MMAs 8
// A rate is warp instructions a SM a clock (probe.py); FP32 results a SM-clock are 32 x it (FFMA, FMUL) or 64 x it
// (FFMA2, FMUL2), MACs a SM-clock 256 x it (DMMA) or 4096 x it (IMMA).
#include "../opcode_probe/probe.cuh"

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

OP_KERNEL(s_ffma, 1024) { s_alone<U_FFMA, 64>(a); }
OP_KERNEL(s_ffma2, 1024) { s_alone<U_FFMA2, 64>(a); }
OP_KERNEL(s_ffma2c, 1024) { s_alone<U_FFMA2C, 64>(a); }
OP_KERNEL(s_fmul, 1024) { s_alone<U_FMUL, 64>(a); }
OP_KERNEL(s_fmul2, 1024) { s_alone<U_FMUL2, 64>(a); }
OP_KERNEL(s_dmma, 1024) { s_alone<U_DMMA, 8>(a); }
OP_KERNEL(s_imma, 1024) { s_alone<U_IMMA, 8>(a); }

#define P(NAME, PAT, IT, NOTE) {.name = #NAME, .kernel = (const void*)NAME, .tpb = 1024, .expect = PAT, .iters = IT, .note = NOTE}
static const OpProbe PROBES[] = {
    P(s_ffma, "FFMA=64", 4096, "FP32 results = 32 x rate"),
    P(s_ffma2, "FFMA2*=64", 4096, "FP32 results = 64 x rate; class_of prices it as one FFMA"),
    P(s_ffma2c, "FFMA2*=64", 4096, "s_ffma2 with one constant addend pair (FFMA's 1.0), in case three pairs conflict"),
    P(s_fmul, "FMUL=64", 4096, "FP32 results = 32 x rate"),
    P(s_fmul2, "FMUL2*=64", 4096, "FP32 results = 64 x rate; class_of prices it as one FFMA"),
    P(s_dmma, "DMMA*=8", 256, "f64 MACs = 256 x rate; exact for int8 x int8 sums"),
    P(s_imma, "IMMA*=8", 1024, "s8 MACs = 4096 x rate: the unit DMMA is read against"),
};
#undef P

int main(int argc, char** argv) { return op_main(argc, argv, PROBES); }
